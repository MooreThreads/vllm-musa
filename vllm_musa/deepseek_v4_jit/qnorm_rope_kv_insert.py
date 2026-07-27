# SPDX-License-Identifier: Apache-2.0
"""TileLang-backed DeepSeek-V4 QNorm/RoPE/KV-cache insert helper.

This module is intentionally imported lazily by the DeepSeek-V4 source patch.
TileLang is optional in the current remote images, so import or compile failures
must leave the existing torch correctness fallback available.
"""

from __future__ import annotations

import logging
import torch

logger = logging.getLogger(__name__)

_HIDDEN_SIZE = 512
_NOPE_DIM = 448
_ROPE_DIM = 64
_HALF_ROPE_DIM = _ROPE_DIM // 2
_SCALE_DIM = _NOPE_DIM // 64
_TOKEN_VALUE_BYTES = _NOPE_DIM + _ROPE_DIM * 2
_TOKEN_SCALE_BYTES = _SCALE_DIM + 1
_FP8_MAX = 448.0


def _is_musa_tensor(tensor: torch.Tensor) -> bool:
    return getattr(tensor, "device", None) is not None and tensor.device.type == "musa"


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.int32:
        return "int32"
    if dtype == torch.int64:
        return "int64"
    return str(dtype).split(".")[-1]


def _guard_tilelang_qnorm_rope_kv_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    block_size: int,
) -> tuple[bool, str]:
    tensors = (q, kv, k_cache_2d, slot_mapping, positions, cos_sin_cache)
    if not all(_is_musa_tensor(tensor) for tensor in tensors):
        return False, "all tensors must be on MUSA"
    if len({tensor.device for tensor in tensors}) != 1:
        return False, "all tensors must be on the same MUSA device"
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        return False, f"expected bf16 q/kv, got q={q.dtype} kv={kv.dtype}"
    if k_cache_2d.dtype != torch.uint8:
        return False, f"expected uint8 cache, got {k_cache_2d.dtype}"
    if cos_sin_cache.dtype != torch.float32:
        return False, f"expected float32 cos_sin_cache, got {cos_sin_cache.dtype}"
    if positions.dtype not in (torch.int32, torch.int64):
        return False, f"unsupported positions dtype {positions.dtype}"
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        return False, f"unsupported slot_mapping dtype {slot_mapping.dtype}"
    if positions.dtype != torch.int64 or slot_mapping.dtype != torch.int64:
        return False, "TileLang path currently requires int64 positions and slots"
    if q.dim() != 3 or q.shape[-1] != _HIDDEN_SIZE:
        return False, f"expected q shape [tokens, heads, 512], got {tuple(q.shape)}"
    if kv.dim() != 2 or kv.shape[-1] != _HIDDEN_SIZE:
        return False, f"expected kv shape [tokens, 512], got {tuple(kv.shape)}"
    if q.shape[0] != kv.shape[0]:
        return False, f"q/kv token mismatch: q={q.shape[0]} kv={kv.shape[0]}"
    if positions.dim() != 1 or positions.shape[0] != kv.shape[0]:
        return False, "positions must be 1D and match token count"
    if slot_mapping.dim() != 1 or slot_mapping.shape[0] < kv.shape[0]:
        return False, "slot_mapping must be 1D and cover every token"
    if cos_sin_cache.dim() != 2 or cos_sin_cache.shape[-1] != _ROPE_DIM:
        return False, (
            "cos_sin_cache must have shape [positions, 64], got "
            f"{tuple(cos_sin_cache.shape)}"
        )
    if not q.is_contiguous() or not kv.is_contiguous():
        return False, "q and kv must be contiguous"
    if cos_sin_cache.stride(-1) != 1:
        return False, "cos_sin_cache must have contiguous last dimension"
    if int(block_size) <= 0:
        return False, f"block_size must be positive, got {block_size}"
    if k_cache_2d.dim() != 2:
        return False, "k_cache_2d must be a 2D uint8 tensor"
    expected_cache_row = int(block_size) * (_TOKEN_VALUE_BYTES + _TOKEN_SCALE_BYTES)
    if k_cache_2d.shape[1] != expected_cache_row:
        return False, (
            f"cache row bytes {k_cache_2d.shape[1]} != expected {expected_cache_row} "
            f"for block_size={block_size}"
        )
    if k_cache_2d.stride(1) != 1:
        return False, "k_cache_2d rows must have contiguous byte layout"
    if k_cache_2d.stride(0) < expected_cache_row:
        return False, (
            f"k_cache_2d row stride {k_cache_2d.stride(0)} is smaller than "
            f"logical row bytes {expected_cache_row}"
        )
    if (
        k_cache_2d.storage_offset() % 4 != 0
        or k_cache_2d.shape[1] % 4 != 0
        or k_cache_2d.stride(0) % 4 != 0
    ):
        return False, "k_cache_2d must be viewable as uint32 for RoPE stores"
    if not positions.is_contiguous() or not slot_mapping.is_contiguous():
        return False, "positions and slot_mapping must be contiguous"
    try:
        cache_u8 = _tilelang_cache_u8_view(k_cache_2d)
    except RuntimeError as exc:
        return False, f"k_cache_2d cannot expose a physical row view: {exc}"
    if not cache_u8.is_contiguous():
        return False, "k_cache_2d physical row view must be contiguous for TileLang"
    return True, ""


def _tilelang_cache_u8_view(k_cache_2d: torch.Tensor) -> torch.Tensor:
    if k_cache_2d.is_contiguous():
        return k_cache_2d
    row_stride = int(k_cache_2d.stride(0))
    return k_cache_2d.as_strided(
        (k_cache_2d.shape[0], row_stride),
        (row_stride, 1),
    )


def try_tilelang_qnorm_rope_kv_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    block_size: int,
) -> tuple[bool, str]:
    """Try the TileLang path and report whether it handled the call."""
    supported, reason = _guard_tilelang_qnorm_rope_kv_insert(
        q, kv, k_cache_2d, slot_mapping, positions, cos_sin_cache, block_size
    )
    if not supported:
        return False, reason

    try:
        from vllm_musa.deepseek_v4_jit.tilelang_kernels import (
            kv_rope_pack_kernel,
            qnorm_rope_kernel,
        )

        q_out = torch.empty_like(q)
        qnorm_rope_kernel()(
            q,
            q_out,
            cos_sin_cache,
            positions,
            float(eps),
        )
        cache_u8 = _tilelang_cache_u8_view(k_cache_2d)
        kv_rope_pack_kernel()(
            kv,
            cache_u8,
            cache_u8.view(torch.uint32),
            slot_mapping[: kv.shape[0]],
            positions,
            cos_sin_cache,
            int(block_size),
        )
        q.copy_(q_out)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "tilelang"
