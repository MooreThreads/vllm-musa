# SPDX-License-Identifier: Apache-2.0
"""Split-K FP8 GEMM for small M and 32x32 block-scaled weights.

Each program multiplies a 16- or 32-row tile by a BLOCK_N x (K / SPLIT) weight
slice, one 32-wide scale group per FP8 dot, so there are enough programs to keep
the GPU busy at decode M; row tiles that share a weight slice run back to back.
A BF16 activation is quantized per 32-wide group inside the kernel with the
arithmetic of ``per_token_group_quant_fp8``; each group lies inside one K slice.
A second kernel sums the SPLIT partials in a fixed order and rounds once, so a
row's result depends on neither M nor the tile height.
"""

from __future__ import annotations

import os

import torch
from vllm.triton_utils import tl, triton

from vllm_musa.tactics.selectors.fp8_block_splitk import config_for

_FP8_MAX = 448.0
_QUANT_EPS = 1e-10
BLOCK_M = 16

# The tuned numbers live in ``tactics/fp8_block_splitk.json``, keyed by projection
# shape, capture mode and row band; ``tactics.selectors.fp8_block_splitk`` turns a
# matched row into the tuple this module launches with.  What stays here is what
# needs a tensor: the dtype, stride and scale checks below.
_ENABLED = os.environ.get("VLLM_MUSA_FP8_SPLITK_MM", "1") != "0"


@triton.jit
def _fp8_block_splitk_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    part_ptr,
    M,
    N,
    stride_am,
    stride_asm,
    stride_bn,
    stride_bsn,
    QUANT_A: tl.constexpr,
    A_GROUP: tl.constexpr,
    GROUP_N: tl.constexpr,
    K_PER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
    EVEN_N: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_EPS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 32)
    m_mask = offs_m < M
    n_mask = offs_n < N
    a_rows = a_ptr + offs_m[:, None].to(tl.int64) * stride_am + offs_k[None, :]
    b_cols = b_ptr + offs_n[None, :].to(tl.int64) * stride_bn + offs_k[:, None]
    bs_cols = b_scale_ptr + (offs_n // GROUP_N).to(tl.int64) * stride_bsn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_start = pid_k * K_PER
    for k0 in range(k_start, k_start + K_PER, 32 * UNROLL):
        for j in tl.static_range(UNROLL):
            kk = k0 + j * 32
            if EVEN_N:
                b = tl.load(b_cols + kk)
                b_s = tl.load(bs_cols + kk // 32)
            else:
                b = tl.load(b_cols + kk, mask=n_mask[None, :], other=0.0)
                b_s = tl.load(bs_cols + kk // 32, mask=n_mask, other=0.0)
            if QUANT_A:
                x = tl.load(a_rows + kk, mask=m_mask[:, None], other=0.0).to(tl.float32)
                a_s = tl.maximum(tl.max(tl.abs(x), axis=1), QUANT_EPS) / FP8_MAX
                a = tl.clamp(x / a_s[:, None], -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
            else:
                a = tl.load(a_rows + kk, mask=m_mask[:, None], other=0.0)
                a_s = tl.load(
                    a_scale_ptr + offs_m * stride_asm + kk // A_GROUP,
                    mask=m_mask,
                    other=0.0,
                )
            acc += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
    parts = part_ptr + (pid_k * M + offs_m[:, None]).to(tl.int64) * N + offs_n[None, :]
    tl.store(parts, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _fp8_block_splitk_reduce_kernel(
    part_ptr,
    c_ptr,
    M,
    N,
    stride_cm,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * N
    acc = tl.zeros((BLOCK,), tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(part_ptr + s * M * N + offs, mask=mask, other=0.0)
    rows = offs // N
    cols = offs - rows * N
    tl.store(c_ptr + rows * stride_cm + cols, acc.to(c_ptr.dtype.element_ty), mask=mask)


def fp8_block_splitk_config(
    weight: torch.Tensor, weight_scale: torch.Tensor, rows: int
) -> tuple[int, int, int, int, int, int, int] | None:
    """Measured config for ``rows`` rows, or None when the GEMM does not apply."""
    if not _ENABLED or weight.dim() != 2:
        return None
    n, k = weight.shape
    config = config_for(
        n, k, rows, graph_capture=torch.cuda.is_current_stream_capturing()
    )
    if config is None:
        return None
    if (
        weight.dtype != torch.float8_e4m3fn
        or weight_scale.dtype != torch.float32
        or weight.stride(-1) != 1
        or weight_scale.stride(-1) != 1
        or tuple(weight_scale.shape) != (triton.cdiv(n, 32), k // 32)
        or k % (config[0] * 32 * config[3])
    ):
        return None
    return config


def _tile_rows(rows: int, tile_m: int) -> int:
    """``tile_m`` unless it pads more rows than ``BLOCK_M``-row tiles."""
    if triton.cdiv(rows, tile_m) * tile_m == triton.cdiv(rows, BLOCK_M) * BLOCK_M:
        return tile_m
    return BLOCK_M


def fp8_block_splitk_mm(
    a: torch.Tensor,
    a_scale: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    config: tuple[int, int, int, int, int, int, int],
) -> torch.Tensor:
    """``output = dequant(a) @ dequant(weight).T`` for the rows ``config`` admits.

    Without ``a_scale``, ``a`` is BF16 and quantized like
    ``per_token_group_quant_fp8(a, 32)``. With it, ``a`` is FP8 with 32/64/128-wide
    group scales.
    """
    split, block_n, num_warps, unroll, _, _, tile_m = config
    m = a.shape[0]
    n, k = weight.shape
    block_m = _tile_rows(m, tile_m)
    parts = torch.empty((split, m, n), dtype=torch.float32, device=a.device)
    quant = a_scale is None
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n), split)
    _fp8_block_splitk_kernel[grid](
        a,
        a if quant else a_scale,
        weight,
        weight_scale,
        parts,
        m,
        n,
        a.stride(0),
        0 if quant else a_scale.stride(0),
        weight.stride(0),
        weight_scale.stride(0),
        QUANT_A=quant,
        A_GROUP=32 if quant else k // a_scale.shape[-1],
        GROUP_N=32,
        K_PER=k // split,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        UNROLL=unroll,
        EVEN_N=n % block_n == 0,
        FP8_MAX=_FP8_MAX,
        QUANT_EPS=_QUANT_EPS,
        num_warps=num_warps,
        num_stages=1,
    )
    # Narrow blocks keep more reduction programs in flight; measured on MTT S5000.
    _fp8_block_splitk_reduce_kernel[(triton.cdiv(m * n, 128),)](
        parts,
        output,
        m,
        n,
        output.stride(0),
        SPLIT=split,
        BLOCK=128,
        num_warps=2 if split >= 16 else 4,
    )
    return output
