# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct exact-gated FA3 metadata preparation for MUSA."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

_BLOCK_SIZE = 256
_SINGLE_REQUEST_SCHEDULER_SIZE = 16
_BS16 = 16
_BS16_SCHEDULER_SIZE = _BS16 * 4
_SUPPORTED_QWEN_GEOMETRIES = frozenset(
    {
        (12, 2, 128),
        (14, 2, 64),
        (16, 2, 128),
        (16, 8, 128),
        (28, 4, 128),
        (32, 4, 128),
        (32, 8, 128),
        (40, 8, 128),
        (64, 4, 128),
        (64, 8, 128),
    }
)


@triton.jit
def _build_qwen_single_request_fa3_metadata_kernel(
    seq_lens_ptr,
    cu_seqlens_k_ptr,
    scheduler_dst_ptr,
    scheduler_dst_size,
    NUM_HEADS_KV: tl.constexpr,
    MAX_NUM_SPLITS: tl.constexpr,
    MAX_KV_BLOCKS_IN_L2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    seq_len = tl.load(seq_lens_ptr)
    safe_seq_len = tl.where(seq_len > 0, seq_len, 1)

    # Batch-one specialization of the MATE 0.2.4 metadata scheduler. Runtime
    # gates fix BF16, causal decode, TileN/page size 64, pack-GQA and 60 active
    # MPs. In this envelope num_m_blocks is one and global split-budget
    # reduction is a no-op.
    num_n_blocks = (safe_seq_len + 63) // 64
    # MATE uses ceilf(n_blocks * 1.1f * nheads / 60). The host gate limits
    # n_blocks to 64 and KV heads to 2/4/8, where this rational form is exactly
    # equivalent and avoids backend-dependent float rounding in Triton.
    blocks_per_mp = (num_n_blocks * 11 * NUM_HEADS_KV + 599) // 600
    split_count = (num_n_blocks + blocks_per_mp - 1) // blocks_per_mp
    split_count = tl.where(
        split_count <= MAX_NUM_SPLITS,
        split_count,
        MAX_NUM_SPLITS,
    )
    split_n_blocks = (num_n_blocks + split_count - 1) // split_count
    num_nheads_in_l2 = tl.where(
        split_n_blocks * 16 <= MAX_KV_BLOCKS_IN_L2,
        16,
        tl.where(
            split_n_blocks * 8 <= MAX_KV_BLOCKS_IN_L2,
            8,
            tl.where(
                split_n_blocks * 4 <= MAX_KV_BLOCKS_IN_L2,
                4,
                tl.where(
                    split_n_blocks * 2 <= MAX_KV_BLOCKS_IN_L2,
                    2,
                    1,
                ),
            ),
        ),
    )
    num_nheads_in_l2 = tl.where(
        num_nheads_in_l2 <= NUM_HEADS_KV,
        num_nheads_in_l2,
        NUM_HEADS_KV,
    )
    scheduler_values = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    scheduler_values = tl.where(offsets == 0, split_count, scheduler_values)
    # The four batch-one fields start at 0, 4, 8 and 12 after b is rounded
    # up to four. batch_table[0] is zero and is already covered by the fill.
    scheduler_values = tl.where(offsets == 8, 1, scheduler_values)
    scheduler_values = tl.where(
        offsets == 12,
        num_nheads_in_l2,
        scheduler_values,
    )
    tl.store(
        scheduler_dst_ptr + offsets,
        scheduler_values,
        mask=offsets < scheduler_dst_size,
    )

    cu_values = tl.where(offsets == 0, 0, safe_seq_len)
    tl.store(
        cu_seqlens_k_ptr + offsets,
        cu_values,
        mask=(tl.program_id(0) == 0) & (offsets < 2),
    )


@triton.jit
def _build_qwen_bs16_fa3_metadata_kernel(
    seq_lens_ptr,
    cu_seqlens_k_ptr,
    scheduler_dst_ptr,
    NUM_HEADS_KV: tl.constexpr,
    MAX_KV_BLOCKS_IN_L2: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BATCH_SIZE)
    seq_lens = tl.load(seq_lens_ptr + offsets)
    safe_seq_lens = tl.where(seq_lens > 0, seq_lens, 1)
    num_n_blocks = (safe_seq_lens + 63) // 64

    # MATE's causal metadata path sorts by descending N-block count. Pack the
    # original index into a unique key so equal lengths retain stable order.
    stable_keys = num_n_blocks * 32 + (31 - offsets)
    sorted_keys = tl.sort(stable_keys, descending=True)
    sorted_batch_indices = 31 - sorted_keys % 32
    sorted_n_blocks = sorted_keys // 32

    num_nheads_in_l2 = tl.where(
        sorted_n_blocks * 16 <= MAX_KV_BLOCKS_IN_L2,
        16,
        tl.where(
            sorted_n_blocks * 8 <= MAX_KV_BLOCKS_IN_L2,
            8,
            tl.where(
                sorted_n_blocks * 4 <= MAX_KV_BLOCKS_IN_L2,
                4,
                tl.where(sorted_n_blocks * 2 <= MAX_KV_BLOCKS_IN_L2, 2, 1),
            ),
        ),
    )
    num_nheads_in_l2 = tl.where(
        num_nheads_in_l2 <= NUM_HEADS_KV,
        num_nheads_in_l2,
        NUM_HEADS_KV,
    )

    # Pinned MATE 0.2.4 layout for b_rounded == 16:
    # [dynamic splits, sorted batch table, M blocks, L2 head swizzle].
    tl.store(scheduler_dst_ptr + offsets, 1)
    tl.store(scheduler_dst_ptr + BATCH_SIZE + offsets, sorted_batch_indices)
    tl.store(scheduler_dst_ptr + BATCH_SIZE * 2 + offsets, 1)
    tl.store(
        scheduler_dst_ptr + BATCH_SIZE * 3 + offsets,
        num_nheads_in_l2,
    )

    tl.store(cu_seqlens_k_ptr, 0)
    tl.store(
        cu_seqlens_k_ptr + offsets + 1,
        tl.cumsum(safe_seq_lens, axis=0),
    )


def _supports_qwen_fa3_scheduler_geometry(
    max_seq_len: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    max_num_splits: int,
) -> bool:
    return (
        1 <= max_seq_len <= 4096
        and (num_heads_q, num_heads_kv, head_dim) in _SUPPORTED_QWEN_GEOMETRIES
        and max_num_splits == (60 + num_heads_kv - 1) // num_heads_kv
    )


def supports_qwen_single_request_fa3_scheduler_lookup(
    seq_lens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scheduler_dst: torch.Tensor,
    max_seq_len: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    max_num_splits: int,
) -> bool:
    tensors = (seq_lens, cu_seqlens_k, scheduler_dst)
    return (
        _supports_qwen_fa3_scheduler_geometry(
            max_seq_len,
            num_heads_q,
            num_heads_kv,
            head_dim,
            max_num_splits,
        )
        and seq_lens.device.type == "musa"
        and all(tensor.dtype == torch.int32 for tensor in tensors)
        and all(tensor.device == seq_lens.device for tensor in tensors)
        and all(tensor.is_contiguous() for tensor in tensors)
        and seq_lens.numel() == 1
        and cu_seqlens_k.numel() == 2
        and scheduler_dst.numel() >= _SINGLE_REQUEST_SCHEDULER_SIZE
    )


def try_build_qwen_single_request_fa3_metadata(
    seq_lens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scheduler_dst: torch.Tensor,
    max_seq_len: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    max_num_splits: int,
) -> bool:
    """Build the exact-gated batch-one FA3 schedule and cu_seqlens_k."""
    if not supports_qwen_single_request_fa3_scheduler_lookup(
        seq_lens,
        cu_seqlens_k,
        scheduler_dst,
        max_seq_len,
        num_heads_q,
        num_heads_kv,
        head_dim,
        max_num_splits,
    ):
        return False

    grid = (triton.cdiv(scheduler_dst.numel(), _BLOCK_SIZE),)
    _build_qwen_single_request_fa3_metadata_kernel[grid](
        seq_lens,
        cu_seqlens_k,
        scheduler_dst,
        scheduler_dst.numel(),
        NUM_HEADS_KV=num_heads_kv,
        MAX_NUM_SPLITS=max_num_splits,
        MAX_KV_BLOCKS_IN_L2=96 if head_dim == 64 else 48,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return True


def supports_qwen_bs16_fa3_metadata(
    seq_lens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scheduler_dst: torch.Tensor,
    max_seq_len: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    max_num_splits: int,
) -> bool:
    """Check the exact Qwen3-8B BS16 persistent-scheduler envelope."""
    tensors = (seq_lens, cu_seqlens_k, scheduler_dst)
    return (
        1 <= max_seq_len <= 8192
        and (num_heads_q, num_heads_kv, head_dim) == (32, 8, 128)
        and max_num_splits == 1
        and seq_lens.device.type == "musa"
        and all(tensor.dtype == torch.int32 for tensor in tensors)
        and all(tensor.device == seq_lens.device for tensor in tensors)
        and all(tensor.is_contiguous() for tensor in tensors)
        and seq_lens.numel() == _BS16
        and cu_seqlens_k.numel() == _BS16 + 1
        and scheduler_dst.numel() == _BS16_SCHEDULER_SIZE
    )


def try_build_qwen_bs16_fa3_metadata(
    seq_lens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scheduler_dst: torch.Tensor,
    max_seq_len: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    max_num_splits: int,
) -> bool:
    """Build BS16 MATE-compatible metadata without changing its scheduler."""
    if not supports_qwen_bs16_fa3_metadata(
        seq_lens,
        cu_seqlens_k,
        scheduler_dst,
        max_seq_len,
        num_heads_q,
        num_heads_kv,
        head_dim,
        max_num_splits,
    ):
        return False

    _build_qwen_bs16_fa3_metadata_kernel[(1,)](
        seq_lens,
        cu_seqlens_k,
        scheduler_dst,
        NUM_HEADS_KV=num_heads_kv,
        MAX_KV_BLOCKS_IN_L2=48,
        BATCH_SIZE=_BS16,
    )
    return True


__all__ = [
    "supports_qwen_bs16_fa3_metadata",
    "supports_qwen_single_request_fa3_scheduler_lookup",
    "try_build_qwen_bs16_fa3_metadata",
    "try_build_qwen_single_request_fa3_metadata",
]
