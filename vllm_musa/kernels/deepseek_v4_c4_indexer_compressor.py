# SPDX-License-Identifier: Apache-2.0
"""Native C4 indexer compressor for DeepSeek-V4 decode on MUSA.

The production source patch imports this module lazily. Supported small-token
decode shapes use the native kernel; unsupported shapes, including prefill,
fall back to vLLM's Triton implementation.
"""

from __future__ import annotations

import torch

_HEAD_DIM = 128
_ROPE_DIM = 64
_STATE_BLOCK_SIZE = 4
_STATE_WIDTH = 256
_STATE_ROW_WIDTH = 512
_MAX_DECODE_ROWS = 320
_SUPPORTED_KV_BLOCK_SIZES = (64, 256)
_INDEX_DTYPES = (torch.int32, torch.int64)


def _is_musa_tensor(tensor: torch.Tensor) -> bool:
    return getattr(tensor, "device", None) is not None and tensor.device.type == "musa"


def _guard_c4_indexer_compressor(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    state_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    state_block_size: int,
    state_width: int,
    kv_block_size: int,
) -> tuple[bool, str]:
    tensors = (
        state_cache,
        token_to_req_indices,
        positions,
        state_slot_mapping,
        block_table,
        rms_norm_weight,
        cos_sin_cache,
        kv_cache,
        kv_slot_mapping,
    )
    if not all(_is_musa_tensor(tensor) for tensor in tensors):
        return False, "all tensors must be on MUSA"
    if len({tensor.device for tensor in tensors}) != 1:
        return False, "all tensors must be on the same MUSA device"
    if state_cache.dtype != torch.float32:
        return False, f"state_cache must be float32, got {state_cache.dtype}"
    if state_cache.dim() != 3 or tuple(state_cache.shape[1:]) != (
        _STATE_BLOCK_SIZE,
        _STATE_ROW_WIDTH,
    ):
        return False, (
            "state_cache must have shape [num_blocks, 4, 512], got "
            f"{tuple(state_cache.shape)}"
        )
    if (
        state_cache.stride(-1) != 1
        or state_cache.stride(1) % 4 != 0
        or state_cache.stride(0) % 4 != 0
        or state_cache.storage_offset() % 4 != 0
    ):
        return False, "state_cache does not satisfy aligned float4 row loads"
    if int(state_block_size) != _STATE_BLOCK_SIZE or int(state_width) != _STATE_WIDTH:
        return False, (
            "native C4 contract requires state_block_size=4 and state_width=256, "
            f"got {state_block_size=} {state_width=}"
        )

    if state_slot_mapping.dim() != 1:
        return False, "state_slot_mapping must be 1D"
    num_rows = state_slot_mapping.numel()
    if not 0 < num_rows <= _MAX_DECODE_ROWS:
        return (
            False,
            f"native C4 path supports 1..{_MAX_DECODE_ROWS} decode rows, "
            f"got {num_rows}",
        )
    for name, tensor in (
        ("token_to_req_indices", token_to_req_indices),
        ("positions", positions),
        ("state_slot_mapping", state_slot_mapping),
        ("kv_slot_mapping", kv_slot_mapping),
        ("block_table", block_table),
    ):
        if tensor.dtype not in _INDEX_DTYPES:
            return False, f"{name} must be int32 or int64, got {tensor.dtype}"
    for name, tensor in (
        ("token_to_req_indices", token_to_req_indices),
        ("positions", positions),
        ("kv_slot_mapping", kv_slot_mapping),
    ):
        if tensor.dim() != 1 or tensor.numel() < num_rows:
            return False, f"{name} must be 1D and cover all {num_rows} rows"
    if block_table.dim() != 2 or block_table.shape[0] == 0 or block_table.shape[1] == 0:
        return False, "block_table must be a non-empty 2D tensor"
    if block_table.stride(1) != 1:
        return False, "block_table rows must be contiguous"
    if not all(
        tensor.is_contiguous()
        for tensor in (
            token_to_req_indices,
            positions,
            state_slot_mapping,
            kv_slot_mapping,
        )
    ):
        return False, "index vectors must be contiguous"

    if rms_norm_weight.dtype not in (torch.float32, torch.bfloat16):
        return False, (
            f"rms_norm_weight must be float32 or bfloat16, got {rms_norm_weight.dtype}"
        )
    if rms_norm_weight.dim() != 1 or rms_norm_weight.numel() != _HEAD_DIM:
        return False, (
            f"rms_norm_weight must have shape [128], got {tuple(rms_norm_weight.shape)}"
        )
    if not rms_norm_weight.is_contiguous():
        return False, "rms_norm_weight must be contiguous"
    if (
        cos_sin_cache.dtype != torch.float32
        or cos_sin_cache.dim() != 2
        or cos_sin_cache.shape[1] != _ROPE_DIM
        or cos_sin_cache.stride(1) != 1
    ):
        return False, (
            "cos_sin_cache must be row-contiguous float32 [max_position, 64], got "
            f"shape={tuple(cos_sin_cache.shape)} dtype={cos_sin_cache.dtype}"
        )

    if int(kv_block_size) not in _SUPPORTED_KV_BLOCK_SIZES:
        return False, (
            f"kv_block_size must be one of {_SUPPORTED_KV_BLOCK_SIZES}, "
            f"got {kv_block_size}"
        )
    if (
        kv_cache.dtype != torch.uint8
        or kv_cache.dim() < 2
        or kv_cache.shape[0] == 0
        or kv_cache.shape[1] != int(kv_block_size)
        or kv_cache.stride(-1) != 1
    ):
        return False, (
            "kv_cache must be a uint8 paged cache whose second dimension is the "
            f"KV block size, got shape={tuple(kv_cache.shape)} dtype={kv_cache.dtype}"
        )
    logical_page_bytes = int(kv_block_size) * (_HEAD_DIM + 4)
    if kv_cache.stride(0) < logical_page_bytes or kv_cache.stride(0) % 4 != 0:
        return False, (
            f"kv_cache page stride {kv_cache.stride(0)} cannot hold/aligned-store "
            f"the {logical_page_bytes}-byte page ABI"
        )
    if kv_cache.storage_offset() % 4 != 0:
        return False, "kv_cache storage offset must be 4-byte aligned"
    return True, ""


def try_musa_deepseek_v4_c4_indexer_compressor(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    state_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    rms_eps: float,
    state_block_size: int,
    state_width: int,
    kv_block_size: int,
) -> tuple[bool, str]:
    """Run the native decode path or request the Triton shape fallback."""
    supported, reason = _guard_c4_indexer_compressor(
        state_cache,
        token_to_req_indices,
        positions,
        state_slot_mapping,
        block_table,
        rms_norm_weight,
        cos_sin_cache,
        kv_cache,
        kv_slot_mapping,
        state_block_size,
        state_width,
        kv_block_size,
    )
    if not supported:
        return False, reason

    from vllm_musa import _custom_ops as musa_ops

    musa_ops.deepseek_v4_c4_indexer_compress_cache(
        state_cache,
        token_to_req_indices,
        positions,
        state_slot_mapping,
        block_table,
        rms_norm_weight,
        cos_sin_cache,
        kv_cache,
        kv_slot_mapping,
        float(rms_eps),
        int(state_block_size),
        int(state_width),
        int(kv_block_size),
    )
    return True, "musa_native_c4_indexer_compressor"
