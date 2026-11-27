# SPDX-License-Identifier: Apache-2.0
"""MUSA-safe fallbacks and fused Triton ops for direct ``vllm._custom_ops`` calls."""

from collections.abc import Callable

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from vllm_musa.patches._shared import musa_safe_rms_norm, musa_safe_rotary_embedding

logger = init_logger(__name__)

PATCHES: list = []


@triton.jit
def _musa_static_scaled_fp8_quant_kernel(
    out_ptr,
    in_ptr,
    scale_ptr,
    rows,
    cols,
    out_stride_r,
    in_stride_r,
    in_stride_c,
    scale_stride_r,
    scale_stride_c,
    group_rows,
    group_cols,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    IS_SCALAR_SCALE: tl.constexpr,
    CONTIGUOUS_IN: tl.constexpr,
    HAS_ROW_PADDING: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_r = tl.program_id(0).to(tl.int64)
    pid_c = tl.program_id(1).to(tl.int64)
    offs_c = pid_c * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    col_mask = offs_c < cols
    out_offs = pid_r * out_stride_r + offs_c

    if HAS_ROW_PADDING and pid_r >= rows:
        tl.store(
            out_ptr + out_offs,
            tl.zeros((BLOCK_COLS,), dtype=out_ptr.dtype.element_ty),
            mask=col_mask,
        )
        return

    if CONTIGUOUS_IN:
        in_offs = pid_r * in_stride_r + offs_c
    else:
        in_offs = pid_r * in_stride_r + offs_c * in_stride_c
    x = tl.load(in_ptr + in_offs, mask=col_mask, other=0.0).to(tl.float32)

    if IS_SCALAR_SCALE:
        s = tl.load(scale_ptr).to(tl.float32)
    else:
        sr = pid_r // group_rows
        sc = offs_c // group_cols
        s = tl.load(
            scale_ptr + sr * scale_stride_r + sc * scale_stride_c,
            mask=col_mask,
            other=1.0,
        ).to(tl.float32)

    q = tl.clamp(x / s, FP8_MIN, FP8_MAX).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + out_offs, q, mask=col_mask)


def _normalize_static_fp8_scale(
    scale: torch.Tensor,
    input_shape: tuple[int, int],
    group_shape: tuple[int, int] | None,
) -> tuple[torch.Tensor, bool, int, int]:
    """Validate and normalize ``scale`` to ``(norm_scale, is_scalar, g_r, g_c)``."""
    rows, cols = input_shape
    if scale.numel() == 1:
        return scale, True, max(rows, 1), max(cols, 1)

    if group_shape is None:
        if scale.ndim == 1:
            raise ValueError("A 1D FP8 scale requires an explicit group_shape")
        if scale.ndim != 2:
            raise ValueError(
                "MUSA static FP8 fallback expects a scalar, 1D, or 2D scale"
            )
        if rows == 0 or cols == 0:
            return scale, False, 1, 1
        group_rows = (rows + scale.shape[0] - 1) // scale.shape[0]
        group_cols = (cols + scale.shape[1] - 1) // scale.shape[1]
        if (
            group_rows <= 0
            or group_cols <= 0
            or (rows + group_rows - 1) // group_rows != scale.shape[0]
            or (cols + group_cols - 1) // group_cols != scale.shape[1]
        ):
            raise ValueError(
                f"FP8 scale shape {tuple(scale.shape)} is incompatible with "
                f"input_shape={input_shape}"
            )
        return scale, False, group_rows, group_cols

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
    return scale, False, group_rows, group_cols


def _quantize_static_fp8_cpu_fallback(
    input: torch.Tensor,
    norm_scale: torch.Tensor,
    is_scalar: bool,
    group_rows: int,
    group_cols: int,
    output: torch.Tensor,
    out_rows: int,
    out_dtype: torch.dtype,
) -> None:
    rows, cols = input.shape
    fp8_info = torch.finfo(out_dtype)
    if rows > 0 and cols > 0:
        in_f32 = input.to(torch.float32)
        if is_scalar:
            scaled = in_f32 / norm_scale.to(torch.float32)
        else:
            r_idx = torch.arange(rows, device=input.device) // group_rows
            c_idx = torch.arange(cols, device=input.device) // group_cols
            s_view = norm_scale.to(torch.float32)[r_idx[:, None], c_idx[None, :]]
            scaled = in_f32 / s_view
        output[:rows].copy_(
            scaled.clamp(min=fp8_info.min, max=fp8_info.max).to(out_dtype)
        )
    if out_rows > rows:
        output[rows:out_rows].zero_()


def _make_musa_scaled_fp8_quant(
    original: Callable,
) -> Callable:
    """Wrap vLLM's CUDA-only static FP8 custom op with a single-pass MUSA Triton kernel.

    Dynamic FP8 quantization continues to use the original vLLM C++ implementation.
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
        else:
            if output.dtype != out_dtype:
                raise ValueError(f"MUSA static FP8 output must have dtype {out_dtype}")
            if (
                output.ndim != 2
                or output.shape[1] != cols
                or output.shape[0] < out_rows
            ):
                raise ValueError(
                    f"MUSA static FP8 output shape {tuple(output.shape)} cannot hold "
                    f"({out_rows}, {cols})"
                )
            if output.stride(1) != 1:
                raise ValueError(
                    "MUSA static FP8 output must be row-contiguous in dim 1"
                )

        norm_scale, is_scalar, group_rows, group_cols = _normalize_static_fp8_scale(
            scale, (rows, cols), group_shape
        )

        if out_rows == 0 or cols == 0:
            return output, scale

        if input.device.type != "musa":
            _quantize_static_fp8_cpu_fallback(
                input,
                norm_scale,
                is_scalar,
                group_rows,
                group_cols,
                output,
                out_rows,
                out_dtype,
            )
            return output, scale

        fp8_info = torch.finfo(out_dtype)
        block_cols = min(max(triton.next_power_of_2(cols), 64), 1024)
        num_warps = 4 if block_cols >= 512 else 2
        grid = (out_rows, triton.cdiv(cols, block_cols))
        _musa_static_scaled_fp8_quant_kernel[grid](
            output,
            input,
            norm_scale,
            rows,
            cols,
            output.stride(0),
            input.stride(0),
            input.stride(1),
            0 if is_scalar else norm_scale.stride(0),
            0 if is_scalar else norm_scale.stride(1),
            group_rows,
            group_cols,
            FP8_MIN=float(fp8_info.min),
            FP8_MAX=float(fp8_info.max),
            IS_SCALAR_SCALE=is_scalar,
            CONTIGUOUS_IN=(input.stride(1) == 1),
            HAS_ROW_PADDING=(out_rows > rows),
            BLOCK_COLS=block_cols,
            num_warps=num_warps,
        )
        return output, scale

    musa_scaled_fp8_quant._musa_static_fp8_fallback = True
    return musa_scaled_fp8_quant


def apply() -> None:
    try:
        from vllm import _custom_ops as vllm_custom_ops
    except Exception:
        return

    current = getattr(vllm_custom_ops, "rms_norm", None)
    if not getattr(current, "_musa_safe_rms_norm", False):
        musa_safe_rms_norm._musa_safe_rms_norm = True
        vllm_custom_ops.rms_norm = musa_safe_rms_norm

    current = getattr(vllm_custom_ops, "rotary_embedding", None)
    if not getattr(current, "_musa_safe_rotary_embedding", False):
        musa_safe_rotary_embedding._musa_safe_rotary_embedding = True
        vllm_custom_ops.rotary_embedding = musa_safe_rotary_embedding

    current = getattr(vllm_custom_ops, "scaled_fp8_quant", None)
    if current is not None and not getattr(current, "_musa_static_fp8_fallback", False):
        vllm_custom_ops.scaled_fp8_quant = _make_musa_scaled_fp8_quant(current)
