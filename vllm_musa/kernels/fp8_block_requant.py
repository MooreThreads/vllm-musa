from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.triton_utils import tl, triton


@triton.jit
def _coarsen_fp8_scales_kernel(
    scales,
    output,
    SCALE_ROWS: tl.constexpr,
    SCALE_COLS: tl.constexpr,
    stride_row: tl.constexpr,
    stride_col: tl.constexpr,
):
    block_row = tl.program_id(0)
    block_col = tl.program_id(1)
    source_rows = block_row * 4 + tl.arange(0, 4)
    source_cols = block_col * 4 + tl.arange(0, 4)
    values = tl.load(
        scales + source_rows[:, None] * stride_row + source_cols[None, :] * stride_col,
        mask=(source_rows[:, None] < SCALE_ROWS) & (source_cols[None, :] < SCALE_COLS),
        other=0.0,
    )
    scale = tl.max(tl.max(values, axis=1), axis=0)
    scale = tl.where(scale > 0, scale, 1.0)
    tl.store(output + block_row * tl.cdiv(SCALE_COLS, 4) + block_col, scale)


@triton.jit
def _requant_fp8_blocks_kernel(
    weight,
    source_scales,
    target_scales,
    output,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    PADDED_COLS: tl.constexpr,
    stride_weight_row: tl.constexpr,
    stride_weight_col: tl.constexpr,
    stride_scale_row: tl.constexpr,
    stride_scale_col: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows = offsets // PADDED_COLS
    cols = offsets % PADDED_COLS
    valid = (rows < ROWS) & (cols < COLS)
    values = tl.load(
        weight + rows * stride_weight_row + cols * stride_weight_col,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    source_scale = tl.load(
        source_scales
        + (rows // 32) * stride_scale_row
        + (cols // 32) * stride_scale_col,
        mask=valid,
        other=0.0,
    )
    target_scale = tl.load(
        target_scales + (rows // 128) * (PADDED_COLS // 128) + cols // 128,
        mask=rows < ROWS,
        other=1.0,
    )
    values = values * (source_scale / target_scale)
    tl.store(output + offsets, values.to(output.dtype.element_ty), mask=rows < ROWS)


def requantize_fp8_block32_to_block128(
    weight: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coarsen FP8 scales once without materializing a dense dequantized weight."""
    if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
        raise ValueError("Expected a two-dimensional E4M3 weight")
    rows, cols = weight.shape
    if rows == 0 or cols == 0:
        raise ValueError("Expected nonempty weights")
    if scales.shape != (triton.cdiv(rows, 32), triton.cdiv(cols, 32)):
        raise ValueError("Expected 32x32 weight scales")
    if scales.dtype != torch.float32 or scales.device != weight.device:
        raise ValueError("Expected FP32 scales on the weight device")
    padded_cols = triton.cdiv(cols, 128) * 128
    target_scales = torch.empty(
        (triton.cdiv(rows, 128), padded_cols // 128),
        dtype=torch.float32,
        device=weight.device,
    )
    output = torch.empty((rows, padded_cols), dtype=weight.dtype, device=weight.device)
    _coarsen_fp8_scales_kernel[target_scales.shape](
        scales,
        target_scales,
        scales.shape[0],
        scales.shape[1],
        scales.stride(0),
        scales.stride(1),
        num_warps=4,
    )
    _requant_fp8_blocks_kernel[(triton.cdiv(rows * padded_cols, 1024),)](
        weight,
        scales,
        target_scales,
        output,
        rows,
        cols,
        padded_cols,
        weight.stride(0),
        weight.stride(1),
        scales.stride(0),
        scales.stride(1),
        BLOCK=1024,
        num_warps=4,
    )
    return output, target_scales


@triton.jit
def _requant_fp8_activation_kernel(
    inputs,
    scales,
    output,
    output_scales,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    PADDED_COLS: tl.constexpr,
    stride_input_row: tl.constexpr,
    stride_input_col: tl.constexpr,
    stride_scale_row: tl.constexpr,
    stride_scale_col: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    group = tl.program_id(1)
    cols = group * 128 + tl.arange(0, 128)
    values = tl.load(
        inputs + rows[:, None] * stride_input_row + cols[None, :] * stride_input_col,
        mask=(rows[:, None] < ROWS) & (cols[None, :] < COLS),
        other=0.0,
    ).to(tl.float32)
    source_scales = tl.load(
        scales
        + rows[:, None] * stride_scale_row
        + (cols[None, :] // 32) * stride_scale_col,
        mask=(rows[:, None] < ROWS) & (cols[None, :] < COLS),
        other=0.0,
    )
    target_scale = tl.max(source_scales, axis=1)
    target_scale = tl.where(target_scale > 0, target_scale, 1.0)
    values = values * (source_scales / target_scale[:, None])
    tl.store(
        output + rows[:, None] * PADDED_COLS + cols[None, :],
        values.to(output.dtype.element_ty),
        mask=rows[:, None] < ROWS,
    )
    tl.store(
        output_scales + rows * (PADDED_COLS // 128) + group,
        target_scale,
        mask=rows < ROWS,
    )


def requantize_fp8_activation32_to128(
    inputs: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.ndim != 2 or inputs.dtype != torch.float8_e4m3fn:
        raise ValueError("Expected two-dimensional E4M3 activations")
    rows, cols = inputs.shape
    if cols == 0:
        raise ValueError("Expected nonempty activation columns")
    if scales.dtype != torch.float32 or scales.device != inputs.device:
        raise ValueError("Expected FP32 activation scales on the input device")
    if cols % 128 == 0 and scales.shape == (rows, cols // 128):
        return inputs, scales
    if scales.shape != (rows, triton.cdiv(cols, 32)):
        raise ValueError("Expected group32 activation scales")
    padded_cols = triton.cdiv(cols, 128) * 128
    output = torch.empty((rows, padded_cols), dtype=inputs.dtype, device=inputs.device)
    output_scales = torch.empty(
        (rows, padded_cols // 128), dtype=torch.float32, device=inputs.device
    )
    if rows:
        _requant_fp8_activation_kernel[(triton.cdiv(rows, 8), padded_cols // 128)](
            inputs,
            scales,
            output,
            output_scales,
            rows,
            cols,
            padded_cols,
            inputs.stride(0),
            inputs.stride(1),
            scales.stride(0),
            scales.stride(1),
            BLOCK_ROWS=8,
            num_warps=4,
        )
    return output, output_scales


def fp8_block128_prequantized_linear(
    inputs: torch.Tensor,
    input_scales: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    from mate.deep_gemm import fp8_gemm_nt

    quantized, scales = requantize_fp8_activation32_to128(inputs, input_scales)
    output = torch.empty(
        (inputs.shape[0], weight.shape[0]), dtype=out_dtype, device=inputs.device
    )
    if inputs.shape[0] == 0:
        return output
    fp8_gemm_nt((quantized, scales), (weight, weight_scales), output, backend="mudnn")
    return output


def fp8_block128_bf16_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    from mate.deep_gemm import fp8_gemm_nt

    from vllm_musa.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    if inputs.ndim != 2 or inputs.dtype != torch.bfloat16:
        raise ValueError("Expected two-dimensional BF16 activations")
    padding = weight.shape[1] - inputs.shape[1]
    if padding < 0:
        raise ValueError("Activation K exceeds weight K")
    output = torch.empty(
        (inputs.shape[0], weight.shape[0]), dtype=out_dtype, device=inputs.device
    )
    if inputs.shape[0] == 0:
        return output
    if padding:
        inputs = F.pad(inputs, (0, padding))
    quantized, scales = per_token_group_quant_fp8(
        inputs, 128, use_ue8m0=False, column_major_scales=False
    )
    fp8_gemm_nt((quantized, scales), (weight, weight_scales), output, backend="mudnn")
    return output
