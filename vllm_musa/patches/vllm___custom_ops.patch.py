# SPDX-License-Identifier: Apache-2.0
"""MUSA-safe fallbacks for direct ``vllm._custom_ops`` calls."""

from collections.abc import Callable

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_musa.patches._shared import musa_safe_rms_norm, musa_safe_rotary_embedding

logger = init_logger(__name__)

PATCHES: list = []


def _expand_static_fp8_scale(
    scale: torch.Tensor,
    input_shape: tuple[int, int],
    group_shape: tuple[int, int] | None,
) -> torch.Tensor:
    rows, cols = input_shape
    if scale.numel() == 1:
        return scale
    if group_shape is None:
        if scale.ndim == 1:
            raise ValueError("A 1D FP8 scale requires an explicit group_shape")
        return scale

    group_rows, group_cols = group_shape
    group_rows = rows if group_rows == -1 else group_rows
    group_cols = cols if group_cols == -1 else group_cols
    if group_rows <= 0 or group_cols <= 0:
        raise ValueError(f"Invalid FP8 group shape: {group_shape}")

    if scale.ndim == 1:
        if group_rows == rows and group_cols == 1:
            scale = scale.reshape(1, -1)
        elif group_rows == 1 and group_cols == cols:
            scale = scale.reshape(-1, 1)
        else:
            raise ValueError(
                "A 1D FP8 scale requires per-channel (-1, 1) or "
                "per-token (1, -1) group_shape"
            )
    if scale.ndim != 2:
        raise ValueError("MUSA static FP8 fallback expects a scalar, 1D, or 2D scale")

    expected_rows = (rows + group_rows - 1) // group_rows
    expected_cols = (cols + group_cols - 1) // group_cols
    if scale.shape != (expected_rows, expected_cols):
        raise ValueError(
            f"FP8 scale shape {tuple(scale.shape)} does not match "
            f"group_shape={group_shape} for input_shape={input_shape}"
        )
    return scale.repeat_interleave(group_rows, dim=0).repeat_interleave(
        group_cols, dim=1
    )[:rows, :cols]


def _make_musa_scaled_fp8_quant(
    original: Callable,
) -> Callable:
    """Wrap vLLM's CUDA-only static FP8 custom op for MUSA.

    Dynamic FP8 quantization continues to use the original vLLM implementation.
    """

    def musa_scaled_fp8_quant(
        input: torch.Tensor,
        scale: torch.Tensor | None = None,
        num_token_padding: int | None = None,
        scale_ub: torch.Tensor | None = None,
        use_per_token_if_dynamic: bool = False,
        output: torch.Tensor | None = None,
        group_shape: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if scale is None:
            return original(
                input,
                scale,
                num_token_padding,
                scale_ub,
                use_per_token_if_dynamic,
                output,
                group_shape,
            )

        if input.ndim != 2:
            raise ValueError("MUSA static FP8 fallback expects a 2D input")
        rows, cols = input.shape
        out_rows = max(rows, num_token_padding or rows)
        out_dtype = current_platform.fp8_dtype()
        if output is None:
            output = torch.empty((out_rows, cols), device=input.device, dtype=out_dtype)
        elif num_token_padding is not None:
            raise ValueError("MUSA static FP8 fallback does not support output padding")
        elif output.dtype != out_dtype:
            raise ValueError(f"MUSA static FP8 output must have dtype {out_dtype}")

        expanded_scale = _expand_static_fp8_scale(scale, (rows, cols), group_shape)
        fp8_info = torch.finfo(out_dtype)
        scaled = (input / expanded_scale).clamp(
            min=fp8_info.min,
            max=fp8_info.max,
        )
        output[:rows].copy_(scaled.to(out_dtype))
        if out_rows > rows:
            output[rows:].zero_()
        return output, scale

    setattr(musa_scaled_fp8_quant, "_musa_static_fp8_fallback", True)
    return musa_scaled_fp8_quant


def apply() -> None:
    try:
        from vllm import _custom_ops as vllm_custom_ops
    except Exception:
        return

    current = getattr(vllm_custom_ops, "rms_norm", None)
    if not getattr(current, "_musa_safe_rms_norm", False):
        setattr(musa_safe_rms_norm, "_musa_safe_rms_norm", True)
        vllm_custom_ops.rms_norm = musa_safe_rms_norm

    current = getattr(vllm_custom_ops, "rotary_embedding", None)
    if not getattr(current, "_musa_safe_rotary_embedding", False):
        setattr(musa_safe_rotary_embedding, "_musa_safe_rotary_embedding", True)
        vllm_custom_ops.rotary_embedding = musa_safe_rotary_embedding

    current = getattr(vllm_custom_ops, "scaled_fp8_quant", None)
    if current is not None and not getattr(current, "_musa_static_fp8_fallback", False):
        vllm_custom_ops.scaled_fp8_quant = _make_musa_scaled_fp8_quant(current)
