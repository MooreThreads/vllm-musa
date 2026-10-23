# SPDX-License-Identifier: Apache-2.0
"""BF16 GEMM for 32x32 block-FP8 linears past decode row counts.

From a few dozen rows the vendor BF16 GEMM outruns the FP8 split-K and block GEMMs
on MUSA. The weight is dequantized per call into a scratch BF16 copy, which is
exact for power-of-two block scales, and the activation stays BF16 instead of
being quantized per 32-wide group. An FP8 activation with per-row group scales is
dequantized the same way.
"""

from __future__ import annotations

import os

import torch
from vllm.triton_utils import tl, triton

# Rows from which dequantization plus the BF16 GEMM beats the split-K and block
# GEMMs, measured in CUDA graphs on MTT S5000 for the DeepSeek-V4.1 TP8 projections
# and keyed by (N, K); other shapes switch at MIN_ROWS. They apply only while a CUDA
# graph is captured (decode); eager steps, prefill included, switch at MIN_ROWS. The
# split-K row ranges end below these.
MIN_ROWS = 512
_MIN_ROWS_BY_SHAPE = {
    (1792, 5120): 60,
    (576, 5120): 65,
    (1024, 4096): 192,
    (4096, 1280): 60,
    (5120, 1024): 60,
    (5120, 15360): 65,
    (25600, 768): 65,
}
_ENABLED = os.environ.get("VLLM_MUSA_FP8_BLOCK_BF16_GEMM", "1") != "0"
# Dequantization tile, measured on MTT S5000.
_BLOCK_R, _BLOCK_C = 64, 256


# R is the token count when dequantizing an activation; one kernel serves all of them.
@triton.jit(do_not_specialize=["R"])
def _fp8_dequant_kernel(
    src_ptr,
    scale_ptr,
    out_ptr,
    R,
    C,
    stride_src,
    stride_scale,
    stride_out,
    GROUP_R: tl.constexpr,
    GROUP_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # One scale per (row group, column group), broadcast over its columns.
    offs_r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_g = tl.program_id(1) * (BLOCK_C // GROUP_C) + tl.arange(0, BLOCK_C // GROUP_C)
    mask = (offs_r[:, None] < R) & (offs_c[None, :] < C)
    rows = offs_r[:, None].to(tl.int64)
    v = tl.load(src_ptr + rows * stride_src + offs_c[None, :], mask=mask, other=0.0)
    s = tl.load(
        scale_ptr + (offs_r[:, None] // GROUP_R) * stride_scale + offs_g[None, :],
        mask=(offs_r[:, None] < R) & (offs_g[None, :] < tl.cdiv(C, GROUP_C)),
        other=0.0,
    )
    v = tl.reshape(v.to(tl.float32), (BLOCK_R, BLOCK_C // GROUP_C, GROUP_C))
    v = v * s[:, :, None]
    out = tl.reshape(v, (BLOCK_R, BLOCK_C)).to(tl.bfloat16)
    tl.store(out_ptr + rows * stride_out + offs_c[None, :], out, mask=mask)


def _dequant(
    src: torch.Tensor, scale: torch.Tensor, group_r: int, group_c: int
) -> torch.Tensor:
    rows, cols = src.shape
    assert _BLOCK_C % group_c == 0, group_c
    out = torch.empty((rows, cols), dtype=torch.bfloat16, device=src.device)
    _fp8_dequant_kernel[(triton.cdiv(rows, _BLOCK_R), triton.cdiv(cols, _BLOCK_C))](
        src,
        scale,
        out,
        rows,
        cols,
        src.stride(0),
        scale.stride(0),
        out.stride(0),
        GROUP_R=group_r,
        GROUP_C=group_c,
        BLOCK_R=_BLOCK_R,
        BLOCK_C=_BLOCK_C,
        num_warps=4,
    )
    return out


def fp8_block_bf16_supported(
    weight: torch.Tensor, weight_scale: torch.Tensor, rows: int
) -> bool:
    """Whether ``rows`` activation rows run the BF16 GEMM on a 32x32 block-FP8 weight."""
    if not _ENABLED or weight.dim() != 2:
        return False
    n, k = weight.shape
    min_rows = MIN_ROWS
    if torch.cuda.is_current_stream_capturing():
        min_rows = _MIN_ROWS_BY_SHAPE.get((n, k), MIN_ROWS)
    if rows < min_rows:
        return False
    return (
        weight.dtype == torch.float8_e4m3fn
        and weight_scale.dtype == torch.float32
        and weight.stride(-1) == 1
        and weight_scale.stride(-1) == 1
        and tuple(weight_scale.shape) == (triton.cdiv(n, 32), triton.cdiv(k, 32))
    )


def fp8_block_bf16_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ dequant(weight).T`` in BF16.

    ``x`` is BF16, or FP8 with per-row ``x_scale`` groups along K.
    """
    if x_scale is not None:
        x = _dequant(x, x_scale, 1, x.shape[1] // x_scale.shape[1])
    return torch.nn.functional.linear(x, _dequant(weight, weight_scale, 32, 32))
