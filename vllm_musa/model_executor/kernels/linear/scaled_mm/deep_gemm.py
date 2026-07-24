import os
from typing import ClassVar

import torch
import torch.nn.functional as F
import vllm.envs as envs
from vllm.model_executor.kernels.linear import (
    Fp8BlockScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
    fp8_gemm_nt,
    is_deep_gemm_supported,
    should_use_deepgemm_for_fp8_linear,
)
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_musa.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block,
    per_token_group_quant_fp8,
)


def _use_row_major_activation_scales(use_deep_gemm_e8m0: bool) -> bool:
    if use_deep_gemm_e8m0:
        return False
    value = os.getenv("VLLM_MUSA_DEEPGEMM_ROW_MAJOR_ACT_SCALES", "1").lower()
    return value not in ("0", "false", "no", "off")


def _read_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


_MUSA_FP8_SMALL_M_GEMV_ENABLED = os.getenv(
    "VLLM_MUSA_FP8_SMALL_M_GEMV", "0"
).lower() in ("1", "true", "yes", "on")
_MUSA_FP8_SMALL_M_GEMV_MAX_M = _read_int_env("VLLM_MUSA_FP8_SMALL_M_GEMV_MAX_M", 3)


def _should_use_musa_fp8_small_m_gemv(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    use_deep_gemm_e8m0: bool,
) -> bool:
    if (
        not _MUSA_FP8_SMALL_M_GEMV_ENABLED
        or not current_platform.is_musa()
        or use_deep_gemm_e8m0
        or group_size != 128
        or input.dim() != 2
        or weight.dim() != 2
        or weight_scale.dim() != 2
        or input.shape[0] < 1
        or input.shape[0] > _MUSA_FP8_SMALL_M_GEMV_MAX_M
        or input.shape[1] != weight.shape[1]
        or weight.shape[1] % group_size != 0
        or input.dtype != torch.bfloat16
        or weight.dtype != current_platform.fp8_dtype()
        or weight_scale.dtype != torch.float32
        or not input.is_contiguous()
        or not weight.is_contiguous()
        or not weight_scale.is_contiguous()
    ):
        return False

    expected_scale_shape = (
        (weight.shape[0] + group_size - 1) // group_size,
        weight.shape[1] // group_size,
    )
    return tuple(weight_scale.shape) == expected_scale_shape


class MUSADeepGemmFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    apply_input_quant: ClassVar[bool] = False

    def __init__(self, config: FP8ScaledMMLinearLayerConfig):
        super().__init__(config)
        self.use_deep_gemm_e8m0 = False
        act_scale_descriptor = config.activation_quant_key.scale
        self.is_deep_gemm_supported = is_deep_gemm_supported()
        self.quant_fp8 = QuantFP8(
            static=False,
            group_shape=act_scale_descriptor.group_shape,
            use_ue8m0=self.use_deep_gemm_e8m0,
            tma_aligned_scales=envs.VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES,
            column_major_scales=not _use_row_major_activation_scales(
                self.use_deep_gemm_e8m0
            ),
        )

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        params = self._get_layer_params(layer)
        assert layer.weight_block_size is not None

        if self.is_deep_gemm_supported:
            weight_scale_invs = params.weight_scale_inv
            scale_attr = (
                params.WEIGHT_SCALE_INV
                if weight_scale_invs is not None
                else params.WEIGHT_SCALE
            )

            dg_weight, dg_weight_scale = deepgemm_post_process_fp8_weight_block(
                wq=params.weight,
                ws=(
                    weight_scale_invs
                    if weight_scale_invs is not None
                    else params.weight_scale
                ),
                quant_block_shape=tuple(layer.weight_block_size),
                use_e8m0=self.use_deep_gemm_e8m0,
                is_bmm=getattr(layer, "is_bmm", False),
                bmm_batch_size=getattr(layer, "bmm_batch_size", 0),
            )
            replace_parameter(layer, params.WEIGHT, dg_weight)
            replace_parameter(layer, scale_attr, dg_weight_scale)

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_musa():
            return False, "DeepGEMM is only supported on musa platform"
        if not is_deep_gemm_supported():
            return False, "Currently, only musa is supported."
        return True, None

    @classmethod
    def can_implement(cls, config):
        can_implement_base, reason = super().can_implement(config)
        if not can_implement_base:
            return can_implement_base, reason
        if config.out_dtype != torch.bfloat16:
            return (False, "Supports only output dtype of bfloat16")

        if not should_use_deepgemm_for_fp8_linear(
            config.out_dtype, config.weight_shape
        ):
            return False, "The provided metadata is not supported."
        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        group_size = self.weight_group_shape.col

        return run_deepgemm(
            A, B, Bs, group_size, self.use_deep_gemm_e8m0, output=kwargs.get("output")
        )


def run_deepgemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    use_deep_gemm_e8m0: bool,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    if _should_use_musa_fp8_small_m_gemv(
        input, weight, weight_scale, group_size, use_deep_gemm_e8m0
    ):
        return torch.ops.vllm.musa_fp8_small_m_gemv_op(
            input,
            weight,
            weight_scale,
            output,
        )
    return torch.ops.vllm.musa_deepgemm_fp8_op(
        input,
        weight,
        weight_scale,
        group_size,
        use_deep_gemm_e8m0,
        output,
    )


def _musa_deepgemm_fp8_op(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    use_deep_gemm_e8m0: bool,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    if not input.is_contiguous():
        input = input.contiguous()
    q_input, input_scale = per_token_group_quant_fp8(
        input,
        group_size=group_size,
        column_major_scales=not _use_row_major_activation_scales(use_deep_gemm_e8m0),
        use_ue8m0=use_deep_gemm_e8m0,
    )
    if output is None:
        output = torch.empty(
            (q_input.shape[0], weight.shape[0]),
            dtype=torch.bfloat16,
            device=q_input.device,
        )

    fp8_gemm_nt(
        (q_input, input_scale),
        (weight, weight_scale),
        output,
        is_deep_gemm_e8m0_used=use_deep_gemm_e8m0,
    )
    return output


def _musa_silu_deepgemm_fp8_op(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    use_deep_gemm_e8m0: bool,
) -> torch.Tensor:
    """Fuse dense SwiGLU activation with DeepGEMM input quantization.

    The existing dense path materializes a BF16 SwiGLU tensor and then reads it
    again in ``per_token_group_quant_8bit_vec``.  For the supported Qwen FP8
    path, use the shipped SiLU+mul+group-quant kernel and feed its outputs to
    the unchanged DeepGEMM call.  Keep a semantic fallback so direct callers do
    not inherit the compiler pattern's narrower preconditions.
    """
    hidden_size = input.shape[-1] // 2 if input.dim() == 2 else 0
    if (
        group_size != 128
        or use_deep_gemm_e8m0
        or not _use_row_major_activation_scales(use_deep_gemm_e8m0)
        or input.dim() != 2
        or input.shape[-1] % (2 * group_size) != 0
        or hidden_size != weight.shape[-1]
        or input.dtype not in (torch.bfloat16, torch.float16)
        or not input.is_contiguous()
    ):
        d = input.shape[-1] // 2
        activated = F.silu(input[..., :d]) * input[..., d:]
        return _musa_deepgemm_fp8_op(
            activated,
            weight,
            weight_scale,
            group_size,
            use_deep_gemm_e8m0,
        )

    q_input = torch.empty(
        (input.shape[0], hidden_size),
        dtype=current_platform.fp8_dtype(),
        device=input.device,
    )
    input_scale = torch.empty(
        (input.shape[0], hidden_size // group_size),
        dtype=torch.float32,
        device=input.device,
    )
    fp8_min, fp8_max = get_fp8_min_max()
    torch.ops._C_musa_ops.silu_and_mul_per_token_group_fp8_quant(
        input,
        q_input,
        input_scale,
        group_size,
        1e-10,
        fp8_min,
        fp8_max,
    )

    output = torch.empty(
        (q_input.shape[0], weight.shape[0]),
        dtype=torch.bfloat16,
        device=q_input.device,
    )
    fp8_gemm_nt(
        (q_input, input_scale),
        (weight, weight_scale),
        output,
        is_deep_gemm_e8m0_used=use_deep_gemm_e8m0,
    )
    return output


def _musa_fused_add_rms_deepgemm_fp8_op(
    input: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    use_deep_gemm_e8m0: bool,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse Qwen residual RMSNorm, group-128 quantization, and DeepGEMM.

    The fused MUSA kernel is deliberately narrow: it mirrors the current
    compiled Qwen dense path (BF16, hidden 4096, E4M3 group-128, row-major
    activation scales). Unsupported direct callers retain the ordinary
    residual RMSNorm plus DeepGEMM sequence.
    """
    input_ranges = tuple(
        (tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size())
        for tensor in (input, residual, norm_weight)
    )
    inputs_nonoverlapping = all(
        left_end <= right_begin or right_end <= left_begin
        for index, (left_begin, left_end) in enumerate(input_ranges)
        for right_begin, right_end in input_ranges[index + 1 :]
    )
    supported = (
        group_size == 128
        and not use_deep_gemm_e8m0
        and _use_row_major_activation_scales(use_deep_gemm_e8m0)
        and input.dim() == 2
        and input.shape[-1] == 4096
        and residual.shape == input.shape
        and norm_weight.dim() == 1
        and norm_weight.shape[0] == input.shape[-1]
        and input.dtype == torch.bfloat16
        and residual.dtype == input.dtype
        and norm_weight.dtype == input.dtype
        and input.is_contiguous()
        and residual.is_contiguous()
        and norm_weight.is_contiguous()
        and all(
            tensor.data_ptr() % 16 == 0 for tensor in (input, residual, norm_weight)
        )
        and inputs_nonoverlapping
    )
    if not supported:
        summed = input.float() + residual.float()
        residual_out = summed.to(input.dtype)
        variance = summed.square().mean(dim=-1, keepdim=True)
        # Match the captured MUSA Inductor lowering used by the serving path:
        # the nominal intermediate cast is folded through the gain multiply.
        normalized = summed * torch.rsqrt(variance + epsilon)
        normalized = (normalized * norm_weight.float()).to(input.dtype)
        return (
            _musa_deepgemm_fp8_op(
                normalized,
                weight,
                weight_scale,
                group_size,
                use_deep_gemm_e8m0,
            ),
            residual_out,
        )

    q_input = torch.empty(
        input.shape,
        dtype=current_platform.fp8_dtype(),
        device=input.device,
    )
    input_scale = torch.empty(
        (input.shape[0], input.shape[1] // group_size),
        dtype=torch.float32,
        device=input.device,
    )
    residual_out = torch.empty_like(input)
    torch.ops._C_musa_ops.fused_add_rms_norm_per_token_group_fp8_quant(
        input,
        residual,
        norm_weight,
        residual_out,
        q_input,
        input_scale,
        epsilon,
    )

    output = torch.empty(
        (q_input.shape[0], weight.shape[0]),
        dtype=torch.bfloat16,
        device=q_input.device,
    )
    fp8_gemm_nt(
        (q_input, input_scale),
        (weight, weight_scale),
        output,
        is_deep_gemm_e8m0_used=use_deep_gemm_e8m0,
    )
    return output, residual_out


def _musa_fp8_small_m_gemv_op(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    if output is None:
        output = torch.empty(
            (input.shape[0], weight.shape[0]),
            dtype=torch.bfloat16,
            device=input.device,
        )
    torch.ops._C_musa_ops.musa_fused_gemv(
        input,
        weight,
        output,
        None,
        weight_scale,
        False,
        False,
        False,
        None,
        1e-6,
    )
    return output


def _musa_fp8_small_m_gemv_op_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    del weight_scale
    if output is not None:
        return output
    return torch.empty(
        (input.shape[0], weight.shape[0]),
        dtype=torch.bfloat16,
        device=input.device,
    )


def _musa_deepgemm_fp8_op_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    use_deep_gemm_e8m0: bool,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    if output is not None:
        return output
    return torch.empty(
        (input.shape[0], weight.shape[0]),
        dtype=torch.bfloat16,
        device=input.device,
    )


def _musa_silu_deepgemm_fp8_op_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    use_deep_gemm_e8m0: bool,
) -> torch.Tensor:
    del weight_scale, group_size, use_deep_gemm_e8m0
    return torch.empty(
        (input.shape[0], weight.shape[0]),
        dtype=torch.bfloat16,
        device=input.device,
    )


def _musa_fused_add_rms_deepgemm_fp8_op_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    use_deep_gemm_e8m0: bool,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del residual, norm_weight, weight_scale, group_size, use_deep_gemm_e8m0, epsilon
    return (
        torch.empty(
            (input.shape[0], weight.shape[0]),
            dtype=torch.bfloat16,
            device=input.device,
        ),
        torch.empty_like(input),
    )


direct_register_custom_op(
    "musa_fp8_small_m_gemv_op",
    _musa_fp8_small_m_gemv_op,
    mutates_args=["output"],
    fake_impl=_musa_fp8_small_m_gemv_op_fake,
)

direct_register_custom_op(
    "musa_deepgemm_fp8_op",
    _musa_deepgemm_fp8_op,
    mutates_args=["output"],
    fake_impl=_musa_deepgemm_fp8_op_fake,
)

direct_register_custom_op(
    "musa_silu_deepgemm_fp8_op",
    _musa_silu_deepgemm_fp8_op,
    fake_impl=_musa_silu_deepgemm_fp8_op_fake,
)

direct_register_custom_op(
    "musa_fused_add_rms_deepgemm_fp8_op",
    _musa_fused_add_rms_deepgemm_fp8_op,
    fake_impl=_musa_fused_add_rms_deepgemm_fp8_op_fake,
)
