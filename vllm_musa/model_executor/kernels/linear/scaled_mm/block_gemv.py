# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import torch
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import FP8ScaledMMLinearLayerConfig
from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    TritonFp8BlockScaledMMKernel,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_to_fp32,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from vllm_musa.kernels.fp8_block_bf16 import (
    fp8_block_bf16_linear,
    fp8_block_bf16_supported,
)
from vllm_musa.kernels.fp8_block_gemm import (
    fp8_block_scaled_mm,
    fp8_block_scaled_mm_supported,
)
from vllm_musa.kernels.fp8_block_gemv import (
    fp8_block_gemv,
    fp8_block_gemv_quant,
    fp8_block_gemv_quant_supported,
    fp8_block_gemv_supported,
)
from vllm_musa.kernels.fp8_block_requant import (
    fp8_block128_bf16_linear,
    fp8_block128_prequantized_linear,
    requantize_fp8_block32_to_block128,
)
from vllm_musa.kernels.fp8_block_splitk import fp8_block_splitk_config
from vllm_musa.optimization_contract import OptimizationFeature, prefers_feature

logger = init_logger(__name__)

# The GEMV re-reads the weight per row (or per few rows); past a few rows the
# block GEMM wins. Crossovers measured on MTT S5000 for the DeepSeek-V4.1 TP8
# projections, keyed by (N, K); other shapes use the default.
_GEMV_MAX_M_DEFAULT = 8
_GEMV_MAX_M_MEASURED = {
    (576, 5120): 32,
    (1792, 5120): 12,
    (4096, 1280): 6,
    (5120, 1024): 6,
    (5120, 288): 4,
    (25600, 6144): 2,
    (25600, 768): 1,
}


def _env_int(name: str) -> int | None:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return None


_GEMV_MAX_M_OVERRIDE = _env_int("VLLM_MUSA_FP8_BLOCK_GEMV_MAX_M")

# Small-M SwiGLU down projections past the GEMV rows quantize the clamped SwiGLU in
# one kernel instead of a separate activation and quantization.
_NATIVE_SWIGLU_QUANT = os.environ.get("VLLM_MUSA_FP8_NATIVE_GEMV", "1") != "0"
_NATIVE_SWIGLU_QUANT_MAX_ROWS = 64


def _gemv_max_m(weight: torch.Tensor) -> int:
    if _GEMV_MAX_M_OVERRIDE is not None:
        return _GEMV_MAX_M_OVERRIDE
    return _GEMV_MAX_M_MEASURED.get(tuple(weight.shape), _GEMV_MAX_M_DEFAULT)


def _small_m_path(weight: torch.Tensor, weight_scale: torch.Tensor, rows: int) -> bool:
    """Whether ``rows`` run the small-M kernels (GEMV or split-K GEMM) rather than
    the block GEMM."""
    if 0 < rows <= _gemv_max_m(weight):
        return True
    return fp8_block_splitk_config(weight, weight_scale, rows) is not None


class MUSAFp8BlockGemvScaledMMKernel(TritonFp8BlockScaledMMKernel):
    """Block-FP8 linear for scale blocks finer than DeepGEMM's 128.

    Small M runs the block GEMV, which also quantizes the activation; larger M
    keeps the separate activation quantization and runs a block GEMM whose K
    tile spans several scale groups; prefill row counts run the BF16 GEMM on a
    dequantized weight.
    """

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_musa():
            return False, "only supported on MUSA"
        return True, None

    @classmethod
    def can_implement(cls, config: FP8ScaledMMLinearLayerConfig):
        can_implement_base, reason = super().can_implement(config)
        if not can_implement_base:
            return can_implement_base, reason
        if config.out_dtype != torch.bfloat16:
            return False, "Supports only output dtype of bfloat16"
        act_group = config.activation_quant_key.scale.group_shape
        weight_group = config.weight_quant_key.scale.group_shape
        if weight_group.col not in (16, 32, 64) or act_group.row != 1:
            return False, "Supports only K blocks of 16, 32 or 64."
        if act_group.col != weight_group.col:
            return False, "Activation groups must match the weight K block."
        return True, None

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        params = self._get_layer_params(layer)
        scale = params.block_scale
        fp32_scale = (
            _upcast_e8m0_to_fp32(scale)
            if scale.dtype == torch.float8_e8m0fnu
            else scale.to(torch.float32)
        ).contiguous()
        if fp32_scale is not scale:
            replace_parameter(layer, params.block_scale_attr, fp32_scale)
        config = get_current_vllm_config()
        if (
            prefers_feature(config, OptimizationFeature.DEEPSEEK_V41_DENSE_FP8_BLOCK128)
            and tuple(layer.weight_block_size) == (32, 32)
            and not getattr(layer, "is_bmm", False)
            and ".shared_experts." not in getattr(layer, "prefix", "")
            and params.weight.ndim == 2
            and params.weight.shape[0] % 64 == 0
        ):
            weight128, scales128 = requantize_fp8_block32_to_block128(
                params.weight, fp32_scale
            )
            replace_parameter(layer, params.WEIGHT, weight128)
            replace_parameter(layer, params.block_scale_attr, scales128)
            layer.weight_block_size = [128, 128]
            self.weight_group_shape = GroupShape(128, 128)
            self._musa_block128 = True
            logger.info_once(
                "Converted DSV4.1 dense weights to one block128 FP8 "
                "representation; all M use MATE muDNN W8A8."
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        rows = x.numel() // x.shape[-1] if x.dim() > 0 else 0
        params = self._get_layer_params(layer)
        if getattr(self, "_musa_block128", False):
            output = fp8_block128_bf16_linear(
                x.reshape(rows, x.shape[-1]),
                params.weight,
                params.block_scale,
                self.config.out_dtype,
            ).view(*x.shape[:-1], params.weight.shape[0])
            return output + bias if bias is not None else output
        group_size = self.config.activation_quant_key.scale.group_shape.col
        if (
            _small_m_path(params.weight, params.block_scale, rows)
            and x.dtype in (torch.bfloat16, torch.float16)
            and x.stride(-1) == 1
            and params.input_scale is None
            and fp8_block_gemv_quant_supported(
                params.weight, params.block_scale, group_size
            )
        ):
            x_2d = x.reshape(rows, x.shape[-1])
            output = torch.empty(
                (rows, params.weight.shape[0]),
                dtype=self.config.out_dtype,
                device=x.device,
            )
            fp8_block_gemv_quant(
                x_2d, params.weight, params.block_scale, output, group_size
            )
            if bias is not None:
                output = output + bias
            return output.view(*x.shape[:-1], params.weight.shape[0])
        if (
            x.dtype == torch.bfloat16
            and params.input_scale is None
            and fp8_block_bf16_supported(params.weight, params.block_scale, rows)
        ):
            output = fp8_block_bf16_linear(
                x.reshape(rows, x.shape[-1]), params.weight, params.block_scale
            )
            if bias is not None:
                output = output + bias
            return output.view(*x.shape[:-1], params.weight.shape[0])
        return super().apply_weights(layer, x, bias, **kwargs)

    def apply_swiglu_clamp_weights(
        self,
        layer: torch.nn.Module,
        gate_up: torch.Tensor,
        swiglu_limit: float,
    ) -> torch.Tensor | None:
        """``apply_weights`` of the clamped SwiGLU of ``gate_up`` as one GEMV.

        Returns ``None`` outside the small-M GEMV path.
        """
        group_size = self.config.activation_quant_key.scale.group_shape.col
        rows = gate_up.numel() // gate_up.shape[-1] if gate_up.dim() > 0 else 0
        params = self._get_layer_params(layer)
        if not (
            0 < rows
            and gate_up.dtype == torch.bfloat16
            and gate_up.is_contiguous()
            and gate_up.shape[-1] == 2 * params.weight.shape[1]
            and params.input_scale is None
            and fp8_block_gemv_quant_supported(
                params.weight, params.block_scale, group_size
            )
        ):
            return None
        if rows > _gemv_max_m(params.weight):
            return self._swiglu_clamp_block_mm(
                gate_up, rows, params, group_size, swiglu_limit
            )
        output = torch.empty(
            (rows, params.weight.shape[0]),
            dtype=self.config.out_dtype,
            device=gate_up.device,
        )
        fp8_block_gemv_quant(
            gate_up.view(rows, gate_up.shape[-1]),
            params.weight,
            params.block_scale,
            output,
            group_size,
            swiglu_limit=swiglu_limit,
        )
        return output.view(*gate_up.shape[:-1], params.weight.shape[0])

    def _swiglu_clamp_block_mm(
        self,
        gate_up: torch.Tensor,
        rows: int,
        params,
        group_size: int,
        swiglu_limit: float,
    ) -> torch.Tensor | None:
        """Past the GEMV rows: the clamped SwiGLU quantized in one kernel, then the
        block GEMM. Returns ``None`` where the native quantization does not apply."""
        if not (_NATIVE_SWIGLU_QUANT and rows <= _NATIVE_SWIGLU_QUANT_MAX_ROWS):
            return None
        if group_size != 32 or params.weight.shape[1] % 32:
            return None
        from vllm_musa.jit_kernel.csrc import fp8_block_gemv as native

        aq, a_scale = native.act_quant(gate_up.view(rows, -1), swiglu_limit)
        if not fp8_block_scaled_mm_supported(
            aq, a_scale, params.weight, params.block_scale
        ):
            return None
        output = fp8_block_scaled_mm(
            aq, a_scale, params.weight, params.block_scale, self.config.out_dtype
        )
        return output.view(*gate_up.shape[:-1], params.weight.shape[0])

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        if A.dim() == 2 and getattr(self, "_musa_block128", False):
            logger.info_once("DSV4.1 prequantized linear uses FP8 group128 muDNN.")
            return fp8_block128_prequantized_linear(A, As, B, Bs, self.config.out_dtype)
        if (
            A.dim() == 2
            and _small_m_path(B, Bs, A.shape[0])
            and A.stride(-1) == 1
            and As.dim() == 2
            and As.stride(-1) == 1
            and fp8_block_gemv_supported(As, B, Bs)
        ):
            output = torch.empty(
                (A.shape[0], B.shape[0]), dtype=self.config.out_dtype, device=A.device
            )
            return fp8_block_gemv(A, As, B, Bs, output)
        if fp8_block_scaled_mm_supported(A, As, B, Bs):
            return fp8_block_scaled_mm(A, As, B, Bs, self.config.out_dtype)
        return super().apply_block_scaled_mm(A, B, As, Bs)
