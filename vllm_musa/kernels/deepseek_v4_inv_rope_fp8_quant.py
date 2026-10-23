# SPDX-License-Identifier: Apache-2.0
"""Inverse RoPE and per-128 FP8 quantization of the DeepSeek-V4 attention output.

Each program rotates BLOCK_T tokens of one 512-wide head (448 NoPE and 64 RoPE
dims) back by their position and quantizes every 128 dims with a power-of-two
scale. The outputs keep the layout of the native op: FP8 [group, token, heads *
512] and FP32 group-local [token, block] scales whose rows are padded to a
multiple of four tokens with zeros.
"""

from __future__ import annotations

import os

import torch
from vllm.triton_utils import tl, triton

_ENABLED = os.environ.get("VLLM_MUSA_INV_ROPE_FP8_QUANT_TRITON", "1") != "0"
_NOPE_DIM = 448
_ROPE_DIM = 64
_HEAD_DIM = _NOPE_DIM + _ROPE_DIM
_QUANT_GROUP = 128
_FP8_MAX = 448.0
_SCALE_EPS = 1e-10
# (max tokens, BLOCK_T, num_warps) measured in CUDA graphs on MTT S5000 for eight
# heads per rank; the last row covers the rest.
_CONFIGS = ((80, 1, 4), (512, 1, 2), (1 << 62, 4, 4))


# The row counts change with every batch; one kernel serves all of them.
@triton.jit(do_not_specialize=["num_tokens", "aligned_tokens"])
def _dsv4_inv_rope_quant_rows_kernel(
    o_ptr,
    pos_ptr,
    cache_ptr,
    out_ptr,
    scale_ptr,
    num_tokens,
    aligned_tokens,
    stride_ot,
    stride_oh,
    stride_cache,
    stride_out_g,
    stride_out_t,
    stride_scale_g,
    stride_scale_t,
    HEADS_PER_GROUP: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    GROUP: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    FP8_MAX: tl.constexpr,
    SCALE_EPS: tl.constexpr,
):
    HEAD_DIM: tl.constexpr = GROUP * CHUNKS
    head = tl.program_id(1)
    g = head // HEADS_PER_GROUP
    h = head - g * HEADS_PER_GROUP
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    d = tl.arange(0, HEAD_DIM)
    live = t < num_tokens
    rows = t.to(tl.int64)
    src = o_ptr + rows[:, None] * stride_ot + head.to(tl.int64) * stride_oh
    x = tl.load(src + d[None, :], mask=live[:, None], other=0.0).to(tl.float32)

    # fma(x[2i+1], sin, x[2i] * cos) and fma(x[2i+1], cos, -x[2i] * sin) on the
    # RoPE dims, the rounding of the native op.
    is_rope = d >= NOPE_DIM
    rope_mask = live[:, None] & is_rope[None, :]
    partner = tl.load(src + (d ^ 1)[None, :], mask=rope_mask, other=0.0).to(tl.float32)
    pos = tl.load(pos_ptr + t, mask=live, other=0).to(tl.int64)
    pair = tl.maximum(d - NOPE_DIM, 0) >> 1
    cs = cache_ptr + pos[:, None] * stride_cache + pair[None, :]
    cos = tl.load(cs, mask=rope_mask, other=1.0)
    sin = tl.load(cs + HALF_ROPE, mask=rope_mask, other=0.0)
    even = ((d & 1) == 0)[None, :]
    rotated = tl.where(
        even, tl.fma(partner, sin, x * cos), tl.fma(x, cos, -(partner * sin))
    )
    x = tl.where(is_rope[None, :], rotated, x)

    x = tl.reshape(x, (BLOCK_T, CHUNKS, GROUP))
    scale = tl.maximum(tl.max(tl.abs(x), axis=2) / FP8_MAX, SCALE_EPS)
    # exp2(ceil(log2(s))) like the native op: log2 of a scale just above a power
    # of two can round to that power's exponent.
    scale = tl.exp2(tl.ceil(tl.log2(scale)))
    q = tl.clamp(x / scale[:, :, None], -FP8_MAX, FP8_MAX)
    q = tl.reshape(q, (BLOCK_T, HEAD_DIM)).to(out_ptr.dtype.element_ty)
    g = g.to(tl.int64)
    out = out_ptr + g * stride_out_g + rows[:, None] * stride_out_t + h * HEAD_DIM
    tl.store(out + d[None, :], q, mask=live[:, None])

    c = tl.arange(0, CHUNKS)
    scales = scale_ptr + g * stride_scale_g + rows[:, None] * stride_scale_t
    scales += h * CHUNKS
    tl.store(
        scales + c[None, :],
        tl.where(live[:, None], scale, 0.0),
        mask=(t < aligned_tokens)[:, None],
    )


def inv_rope_fp8_quant_supported(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    nope_dim: int,
    rope_dim: int,
    quant_group_size: int,
    tma_aligned_scales: bool,
) -> bool:
    """Whether the Triton kernel covers the call; packed UE8M0 scales do not."""
    return (
        _ENABLED
        and not tma_aligned_scales
        and (nope_dim, rope_dim, quant_group_size)
        == (_NOPE_DIM, _ROPE_DIM, _QUANT_GROUP)
        and o.dim() == 3
        and o.dtype == torch.bfloat16
        and o.shape[2] == _HEAD_DIM
        and o.stride(2) == 1
        and positions.dim() == 1
        and positions.dtype in (torch.int32, torch.int64)
        and positions.numel() == o.shape[0]
        and positions.is_contiguous()
        and o.device == positions.device == cos_sin_cache.device
        and cos_sin_cache.dtype == torch.float32
        and cos_sin_cache.dim() == 2
        and cos_sin_cache.shape[1] == _ROPE_DIM
        and cos_sin_cache.stride(1) == 1
    )


def inv_rope_fp8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 ``[token, group, heads * 512]`` and its ``[token, group, block]`` scales."""
    num_tokens, num_heads, _ = o.shape
    assert num_heads == n_groups * heads_per_group and positions.numel() == num_tokens
    d = heads_per_group * _HEAD_DIM
    blocks = d // _QUANT_GROUP
    aligned = triton.cdiv(num_tokens, 4) * 4
    out = torch.empty(
        (n_groups, num_tokens, d), dtype=torch.float8_e4m3fn, device=o.device
    )
    scale = torch.empty(
        n_groups * blocks * aligned, dtype=torch.float32, device=o.device
    ).as_strided((n_groups, num_tokens, blocks), (blocks * aligned, blocks, 1))
    if num_tokens and num_heads:
        block_t, num_warps = next(
            (bt, w) for limit, bt, w in _CONFIGS if num_tokens <= limit
        )
        _dsv4_inv_rope_quant_rows_kernel[(triton.cdiv(aligned, block_t), num_heads)](
            o,
            positions,
            cos_sin_cache,
            out,
            scale,
            num_tokens,
            aligned,
            o.stride(0),
            o.stride(1),
            cos_sin_cache.stride(0),
            out.stride(0),
            out.stride(1),
            blocks * aligned,
            blocks,
            HEADS_PER_GROUP=heads_per_group,
            NOPE_DIM=_NOPE_DIM,
            HALF_ROPE=_ROPE_DIM // 2,
            GROUP=_QUANT_GROUP,
            CHUNKS=_HEAD_DIM // _QUANT_GROUP,
            BLOCK_T=block_t,
            FP8_MAX=_FP8_MAX,
            SCALE_EPS=_SCALE_EPS,
            num_warps=num_warps,
        )
    return out.transpose(0, 1), scale.transpose(0, 1)
