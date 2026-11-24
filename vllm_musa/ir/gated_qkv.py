# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inductor-visible gated QKV post-processing for MUSA.

The SOL MUSA-C kernel is very fast in isolation, but invoking it through an
opaque custom operator costs more end-to-end than the kernel saves. This
implementation expresses the same one-warp-per-head algorithm as a Triton HOP.
``wrap_triton`` records the two fresh output mutations as an Inductor HOP, so
the generated MUSA wrapper launches the kernel directly without a Python or
custom-op call at runtime.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.library import wrap_triton
from vllm import ir
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_HEAD_DIM = 256
_ROTARY_DIM = 64
_ROTARY_HALF = _ROTARY_DIM // 2


@triton.jit
def _gated_qk_norm_rope_token_kernel(
    packed_qkv_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_out_ptr,
    k_out_ptr,
    position_stride,
    cache_stride,
    epsilon: tl.constexpr,
    weight_offset: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    ROTARY_HALF: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    """One token block with one warp for each Q/K head, like SOL #7283."""
    token = tl.program_id(0).to(tl.int64)
    heads = tl.arange(0, BLOCK_HEADS)[:, None]
    offsets = tl.arange(0, HEAD_DIM)[None, :]
    valid_head = heads < NUM_Q_HEADS + 1
    is_k = heads == NUM_Q_HEADS

    packed_width = (2 * NUM_Q_HEADS + 2) * HEAD_DIM
    source_head = tl.where(is_k, 2 * NUM_Q_HEADS, 2 * heads)
    input_offsets = token * packed_width + source_head * HEAD_DIM + offsets
    values = tl.load(
        packed_qkv_ptr + input_offsets,
        mask=valid_head,
        other=0.0,
    ).to(tl.float32)
    weight_ptr = tl.where(is_k, k_weight_ptr, q_weight_ptr)
    weights = tl.load(weight_ptr + offsets).to(tl.float32)
    variance = tl.sum(values * values, axis=1) / HEAD_DIM
    inv_rms = tl.rsqrt(variance + epsilon)[:, None]
    normalized = (values * inv_rms * (weights + weight_offset)).to(tl.bfloat16)

    q_output_offsets = (token * NUM_Q_HEADS + heads) * HEAD_DIM + offsets
    k_output_offsets = token * HEAD_DIM + heads * 0 + offsets
    tail_mask = offsets >= ROTARY_DIM
    tl.store(
        q_out_ptr + q_output_offsets,
        normalized,
        mask=(heads < NUM_Q_HEADS) & tail_mask,
    )
    tl.store(
        k_out_ptr + k_output_offsets,
        normalized,
        mask=is_k & tail_mask,
    )

    rotary_offsets = tl.arange(0, ROTARY_HALF)[None, :]
    rotary_input_base = token * packed_width + source_head * HEAD_DIM
    x1 = tl.load(
        packed_qkv_ptr + rotary_input_base + rotary_offsets,
        mask=valid_head,
        other=0.0,
    ).to(tl.float32)
    x2 = tl.load(
        packed_qkv_ptr + rotary_input_base + ROTARY_HALF + rotary_offsets,
        mask=valid_head,
        other=0.0,
    ).to(tl.float32)
    w1 = tl.load(weight_ptr + rotary_offsets).to(tl.float32)
    w2 = tl.load(weight_ptr + ROTARY_HALF + rotary_offsets).to(tl.float32)
    x1 = (x1 * inv_rms * (w1 + weight_offset)).to(tl.bfloat16).to(tl.float32)
    x2 = (x2 * inv_rms * (w2 + weight_offset)).to(tl.bfloat16).to(tl.float32)

    modality = rotary_offsets % 3
    position = tl.load(positions_ptr + modality * position_stride + token).to(tl.int64)
    cache_base = cos_sin_cache_ptr + position * cache_stride
    cosine = tl.load(cache_base + rotary_offsets).to(tl.float32)
    sine = tl.load(cache_base + ROTARY_HALF + rotary_offsets).to(tl.float32)

    x1_cos = (x1 * cosine).to(tl.bfloat16).to(tl.float32)
    x2_sin = (x2 * sine).to(tl.bfloat16).to(tl.float32)
    x2_cos = (x2 * cosine).to(tl.bfloat16).to(tl.float32)
    x1_sin = (x1 * sine).to(tl.bfloat16).to(tl.float32)
    out1 = (x1_cos - x2_sin).to(tl.bfloat16)
    out2 = (x2_cos + x1_sin).to(tl.bfloat16)
    q_rotary_offsets = (token * NUM_Q_HEADS + heads) * HEAD_DIM + rotary_offsets
    k_rotary_offsets = token * HEAD_DIM + heads * 0 + rotary_offsets
    q_mask = heads < NUM_Q_HEADS
    tl.store(q_out_ptr + q_rotary_offsets, out1, mask=q_mask)
    tl.store(
        q_out_ptr + q_rotary_offsets + ROTARY_HALF,
        out2,
        mask=q_mask,
    )
    tl.store(k_out_ptr + k_rotary_offsets, out1, mask=is_k)
    tl.store(
        k_out_ptr + k_rotary_offsets + ROTARY_HALF,
        out2,
        mask=is_k,
    )


def _supports_gated_qkv(
    packed_qkv: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    cos_sin_cache: Tensor,
    positions: Tensor,
    epsilon: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope_section: list[int],
    mrope_interleaved: bool,
    is_neox_style: bool,
    weight_offset: float,
) -> bool:
    del epsilon
    if len(mrope_section) != 3:
        return False

    # vLLM's interleaved selector uses axis 1 for f % 3 == 1 while
    # f < 3 * section[1], and similarly for axis 2. These bounds guarantee
    # the simple f % 3 rule for every rotary frequency in [0, 32).
    repeating_three_axis_mrope = (
        sum(mrope_section) == _ROTARY_HALF
        and mrope_section[1] >= 11
        and mrope_section[2] >= 10
    )
    expected_width = (2 * num_q_heads + 2 * num_kv_heads) * head_dim
    tensors = (packed_qkv, q_weight, k_weight, cos_sin_cache, positions)
    if not all(tensor.device.type == "musa" for tensor in tensors):
        return False
    if not all(tensor.device == packed_qkv.device for tensor in tensors):
        return False

    device_id = packed_qkv.device.index or 0
    if device_id < 0 or device_id >= torch.musa.device_count():
        return False

    return (
        num_q_heads in (2, 3)
        and num_kv_heads == 1
        and head_dim == _HEAD_DIM
        and rotary_dim == _ROTARY_DIM
        and repeating_three_axis_mrope
        and mrope_interleaved
        and is_neox_style
        and weight_offset == 1.0
        and current_platform.is_device_capability((3, 1), device_id=device_id)
        and packed_qkv.dtype == torch.bfloat16
        and q_weight.dtype == packed_qkv.dtype
        and k_weight.dtype == packed_qkv.dtype
        and cos_sin_cache.dtype == packed_qkv.dtype
        and positions.dtype == torch.int64
        and packed_qkv.dim() == 2
        and packed_qkv.shape[1] == expected_width
        and q_weight.shape == (_HEAD_DIM,)
        and k_weight.shape == (_HEAD_DIM,)
        and cos_sin_cache.dim() == 2
        and cos_sin_cache.shape[1] == _ROTARY_DIM
        and positions.dim() == 2
        and positions.shape[0] == 3
        and positions.shape[1] == packed_qkv.shape[0]
        and packed_qkv.is_contiguous()
        and q_weight.is_contiguous()
        and k_weight.is_contiguous()
        and cos_sin_cache.is_contiguous()
        and positions.stride(1) == 1
    )


@ir.ops.gated_qkv_rms_norm_rope.register_impl(
    "musa_inductor",
    supports_args=_supports_gated_qkv,
    supported=current_platform.is_musa(),
)
def gated_qkv_rms_norm_rope(
    packed_qkv: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    cos_sin_cache: Tensor,
    positions: Tensor,
    epsilon: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope_section: list[int],
    mrope_interleaved: bool,
    is_neox_style: bool,
    weight_offset: float,
) -> tuple[Tensor, Tensor]:
    """Direct token-block Inductor lowering for the SOL shape family."""
    del num_kv_heads, rotary_dim, mrope_section, mrope_interleaved, is_neox_style
    num_tokens = packed_qkv.shape[0]
    query = torch.empty(
        (num_tokens, num_q_heads * head_dim),
        dtype=packed_qkv.dtype,
        device=packed_qkv.device,
    )
    key = torch.empty(
        (num_tokens, head_dim),
        dtype=packed_qkv.dtype,
        device=packed_qkv.device,
    )
    if num_tokens == 0:
        return query, key
    wrap_triton(_gated_qk_norm_rope_token_kernel)[(num_tokens,)](
        packed_qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        query,
        key,
        positions.stride(0),
        cos_sin_cache.stride(0),
        epsilon,
        weight_offset,
        NUM_Q_HEADS=num_q_heads,
        HEAD_DIM=_HEAD_DIM,
        ROTARY_DIM=_ROTARY_DIM,
        ROTARY_HALF=_ROTARY_HALF,
        BLOCK_HEADS=4,
        num_warps=4,
        num_stages=2,
    )
    return query, key


__all__ = ["gated_qkv_rms_norm_rope"]
