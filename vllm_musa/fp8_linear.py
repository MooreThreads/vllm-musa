from typing import ClassVar

import torch
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
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from vllm_musa.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block,
    per_token_group_quant_fp8,
)


class MUSAFp8BlockScaledMMLinearKernel(Fp8BlockScaledMMLinearKernel):
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
            column_major_scales=True,
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
    ) -> torch.Tensor:
        group_size = self.weight_group_shape.col

        return run_deepgemm(A, B, Bs, group_size, self.use_deep_gemm_e8m0)


def run_deepgemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    use_deep_gemm_e8m0: bool,
) -> torch.Tensor:
    q_input, input_scale = per_token_group_quant_fp8(
        input,
        group_size=group_size,
        column_major_scales=True,
        use_ue8m0=use_deep_gemm_e8m0,
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
