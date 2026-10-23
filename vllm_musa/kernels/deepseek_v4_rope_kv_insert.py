# SPDX-License-Identifier: Apache-2.0
"""Query RoPE and SWA cache insert of the DeepSeek-V4 attention input.

Each program writes HEADS query heads (448 NoPE and 64 RoPE dims) of one token:
the real heads, optionally RMS-normalized, with their RoPE pairs rotated by the
token position, and the padding heads the sparse MLA kernels read as zeros. One
more program per token stores its KV row in the FP8 cache layout of the native
op: seven 64-wide NoPE groups as FP8 with power-of-two scales kept as exponent
bytes, then the rotated RoPE dims as BF16. Every rounding step follows the
native op, so both write the same bits.
"""

from __future__ import annotations

import os

import torch
from vllm.triton_utils import tl, triton

_ENABLED = os.environ.get("VLLM_MUSA_ROPE_KV_INSERT_TRITON", "1") != "0"
_NOPE_DIM = 448
_ROPE_DIM = 64
_HEAD_DIM = _NOPE_DIM + _ROPE_DIM
_QUANT_GROUP = 64
_TOKEN_BYTES = _NOPE_DIM + _ROPE_DIM * 2
_SCALE_BYTES = 8
# (max tokens, HEADS, num_warps) measured on MTT S5000 for eight heads per rank
# padded to 64; the last row covers the rest, prefill included, so the CUDA graph
# captures compile every config.
_CONFIGS = ((16, 2, 4), (1 << 62, 1, 2))


@triton.jit
def _sum_halves(x, HEADS: tl.constexpr, N: tl.constexpr):
    """``x[:, j] + x[:, j + N // 2]`` of a ``[HEADS, N]`` tile."""
    lo, hi = tl.split(tl.permute(tl.reshape(x, (HEADS, 2, N // 2)), (0, 2, 1)))
    return lo + hi


# The slot count changes with every batch; one kernel serves all of them.
@triton.jit(do_not_specialize=["num_slots"])
def _dsv4_rope_kv_insert_kernel(
    q_ptr,
    out_ptr,
    kv_ptr,
    cache_ptr,
    slot_ptr,
    pos_ptr,
    cos_sin_ptr,
    num_slots,
    cache_slots,
    block_size,
    block_stride,
    stride_cos_sin,
    eps,
    APPLY_NORM: tl.constexpr,
    IN_HEADS: tl.constexpr,
    OUT_HEADS: tl.constexpr,
    HEADS: tl.constexpr,
    KV_BLOCK: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    GROUP: tl.constexpr,
    TOKEN_BYTES: tl.constexpr,
):
    HEAD_DIM: tl.constexpr = NOPE_DIM + 2 * HALF_ROPE
    PAIRS: tl.constexpr = HEAD_DIM // 2
    t = tl.program_id(0)
    head_block = tl.program_id(1)
    row = t.to(tl.int64)
    d = tl.arange(0, HEAD_DIM)
    pair = tl.arange(0, PAIRS)
    rope_pair = pair >= NOPE_DIM // 2

    if head_block == KV_BLOCK:
        slot = tl.load(slot_ptr + t, mask=t < num_slots, other=-1).to(tl.int64)
        if (slot >= 0) & (slot < cache_slots):
            block = slot // block_size
            in_block = slot - block * block_size
            block_base = cache_ptr + block * block_stride
            token = block_base + in_block * TOKEN_BYTES
            scale = block_base + block_size * TOKEN_BYTES + in_block * 8
            kv = tl.load(kv_ptr + row * HEAD_DIM + d).to(tl.float32)

            # ceil(log2(amax / 448)) with amax >= 1e-4: the exponent of amax minus
            # 8, plus one when its significand is above 1.75.
            groups = tl.reshape(kv, (HEAD_DIM // GROUP, GROUP))
            amax = tl.maximum(tl.max(tl.abs(groups), axis=1), 1.0e-4)
            bits = amax.to(tl.int32, bitcast=True)
            exponent = (bits >> 23) - 135 + ((bits & 0x7FFFFF) > 0x600000).to(tl.int32)
            inv_scale = ((127 - exponent) << 23).to(tl.float32, bitcast=True)
            codes = tl.clamp(groups * inv_scale[:, None], -448.0, 448.0)
            codes = tl.reshape(codes, (HEAD_DIM,)).to(tl.float8e4nv)
            tl.store(token + d, codes.to(tl.uint8, bitcast=True), mask=d < NOPE_DIM)
            g = tl.arange(0, HEAD_DIM // GROUP)
            byte = tl.minimum(tl.maximum(exponent + 127, 0), 255)
            byte = tl.where(g < NOPE_DIM // GROUP, byte, 0)
            tl.store(scale + g, byte.to(tl.uint8))

            # fma(even, c, -odd * s) and fma(even, s, odd * c), the native rounding.
            pos = tl.load(pos_ptr + t).to(tl.int64)
            cs = cos_sin_ptr + pos * stride_cos_sin + (pair - NOPE_DIM // 2)
            cos = tl.load(cs, mask=rope_pair, other=1.0)
            sin = tl.load(cs + HALF_ROPE, mask=rope_pair, other=0.0)
            even, odd = tl.split(tl.reshape(kv, (PAIRS, 2)))
            rotated = tl.join(
                tl.fma(even, cos, -(odd * sin)), tl.fma(even, sin, odd * cos)
            )
            rotated = tl.reshape(rotated, (HEAD_DIM,)).to(tl.bfloat16)
            rope_out = (token + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
            tl.store(rope_out + (d - NOPE_DIM), rotated, mask=d >= NOPE_DIM)
    else:
        h = head_block * HEADS + tl.arange(0, HEADS)
        offs = h[:, None] * HEAD_DIM + d[None, :]
        out = out_ptr + row * (OUT_HEADS * HEAD_DIM) + offs
        out_mask = (h < OUT_HEADS)[:, None]
        if head_block * HEADS < IN_HEADS:
            real = (h < IN_HEADS)[:, None]
            x = tl.load(
                q_ptr + row * (IN_HEADS * HEAD_DIM) + offs, mask=real, other=0.0
            )
            x = x.to(tl.float32)
            if APPLY_NORM:
                # The native sum of squares: v[j]^2 + v[j + 256]^2 in one FMA per
                # lane j, then a halving tree over the 256 lanes.
                lo, hi = tl.split(
                    tl.permute(tl.reshape(x, (HEADS, 2, PAIRS)), (0, 2, 1))
                )
                acc = _sum_halves(tl.fma(hi, hi, lo * lo), HEADS, PAIRS)
                acc = _sum_halves(acc, HEADS, PAIRS // 2)
                acc = _sum_halves(acc, HEADS, PAIRS // 4)
                acc = _sum_halves(acc, HEADS, PAIRS // 8)
                acc = _sum_halves(acc, HEADS, PAIRS // 16)
                acc = _sum_halves(acc, HEADS, PAIRS // 32)
                acc = _sum_halves(acc, HEADS, PAIRS // 64)
                acc = _sum_halves(acc, HEADS, PAIRS // 128)
                x = x * tl.math.rsqrt(acc * (1.0 / HEAD_DIM) + eps)
            pos = tl.load(pos_ptr + t).to(tl.int64)
            cs = cos_sin_ptr + pos * stride_cos_sin + (pair - NOPE_DIM // 2)
            cos = tl.load(cs, mask=rope_pair, other=1.0)[None, :]
            sin = tl.load(cs + HALF_ROPE, mask=rope_pair, other=0.0)[None, :]
            even, odd = tl.split(tl.reshape(x, (HEADS, PAIRS, 2)))
            rot_even = tl.where(
                rope_pair[None, :], tl.fma(even, cos, -(odd * sin)), even
            )
            rot_odd = tl.where(rope_pair[None, :], tl.fma(even, sin, odd * cos), odd)
            y = tl.reshape(tl.join(rot_even, rot_odd), (HEADS, HEAD_DIM))
            y = tl.where(real, y, 0.0)
            tl.store(out, y.to(out_ptr.dtype.element_ty), mask=out_mask)
        else:
            zeros = tl.zeros((HEADS, HEAD_DIM), dtype=out_ptr.dtype.element_ty)
            tl.store(out, zeros, mask=out_mask)


def rope_kv_insert_supported(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    cache_block_size: int,
    q_head_padded: int,
) -> bool:
    """Whether the Triton kernel covers the call: the native layouts."""
    index_dtypes = (torch.int32, torch.int64)
    devices = {
        q.device,
        kv.device,
        kv_cache.device,
        slot_mapping.device,
        positions.device,
        cos_sin_cache.device,
    }
    return (
        _ENABLED
        and len(devices) == 1
        and q.dim() == 3
        and q.dtype == torch.bfloat16
        and q.shape[2] == _HEAD_DIM
        and q.is_contiguous()
        and (q_head_padded == 0 or q_head_padded >= q.shape[1])
        and kv.dtype == torch.bfloat16
        and kv.shape == (q.shape[0], _HEAD_DIM)
        and kv.is_contiguous()
        and kv_cache.dtype == torch.uint8
        and kv_cache.dim() >= 3
        and kv_cache.shape[1] == cache_block_size
        and kv_cache.stride(-1) == 1
        and kv_cache.stride(0) >= cache_block_size * (_TOKEN_BYTES + _SCALE_BYTES)
        and slot_mapping.dim() == 1
        and slot_mapping.dtype in index_dtypes
        and slot_mapping.is_contiguous()
        and slot_mapping.numel() <= q.shape[0]
        and positions.dim() == 1
        and positions.dtype in index_dtypes
        and positions.is_contiguous()
        and positions.numel() == q.shape[0]
        and cos_sin_cache.dtype == torch.float32
        and cos_sin_cache.dim() == 2
        and cos_sin_cache.shape[1] == _ROPE_DIM
        and cos_sin_cache.is_contiguous()
    )


def rope_kv_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    cache_block_size: int,
    q_head_padded: int,
    apply_q_norm: bool,
) -> torch.Tensor:
    """The rotated query, padded to ``q_head_padded`` heads; ``kv`` goes to the cache."""
    num_tokens, in_heads, _ = q.shape
    out_heads = max(q_head_padded, in_heads)
    q_out = q
    if out_heads > in_heads:
        q_out = torch.empty(
            (num_tokens, out_heads, _HEAD_DIM), dtype=q.dtype, device=q.device
        )
    if num_tokens == 0:
        return q_out
    heads, num_warps = next((hb, w) for limit, hb, w in _CONFIGS if num_tokens <= limit)
    q_blocks = triton.cdiv(out_heads, heads)
    _dsv4_rope_kv_insert_kernel[(num_tokens, q_blocks + 1)](
        q,
        q_out,
        kv,
        kv_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        slot_mapping.numel(),
        kv_cache.shape[0] * cache_block_size,
        cache_block_size,
        kv_cache.stride(0),
        cos_sin_cache.stride(0),
        eps,
        APPLY_NORM=apply_q_norm,
        IN_HEADS=in_heads,
        OUT_HEADS=out_heads,
        HEADS=heads,
        KV_BLOCK=q_blocks,
        NOPE_DIM=_NOPE_DIM,
        HALF_ROPE=_ROPE_DIM // 2,
        GROUP=_QUANT_GROUP,
        TOKEN_BYTES=_TOKEN_BYTES,
        num_warps=num_warps,
    )
    return q_out
