from __future__ import annotations

import os

import torch

from vllm_musa.jit_kernel.csrc.jit import load_musa_jit
from vllm_musa.jit_kernel.utils import cache_once


def _layout_strides(
    tensor: torch.Tensor,
    positions_ndim: int,
    head_size: int,
) -> tuple[int, int, int, int]:
    if tensor.dim() not in (positions_ndim + 1, positions_ndim + 2):
        raise ValueError(
            "tensor must have shape [..., hidden_size] "
            "or [..., num_heads, head_size]"
        )

    batch_stride = tensor.stride(0) if positions_ndim == 2 else 0
    token_stride = tensor.stride(positions_ndim - 1)
    if tensor.dim() == positions_ndim + 2:
        head_stride = tensor.stride(-2)
        dim_stride = tensor.stride(-1)
    else:
        dim_stride = tensor.stride(-1)
        head_stride = head_size * dim_stride
    return batch_stride, token_stride, head_stride, dim_stride


def _store_hint(num_tokens: int, rot_dim: int) -> str:
    hint = os.getenv("VLLM_MUSA_ROPE_STORE_HINT", "auto").lower()
    if hint in {"default", "stwb", "stcg", "stcs"}:
        return hint
    if hint != "auto":
        return "stwb"
    if rot_dim == 64 and 512 <= num_tokens <= 2048:
        return "stcg"
    return "stwb"


def _musa_arch_tag() -> str:
    try:
        major, minor = torch.musa.get_device_capability()
        return f"mp{int(major)}{int(minor)}"
    except Exception:
        return "mp31"


@cache_once
def _rope_module(arch_tag: str, store_hint: str):
    hint_value = {"default": 0, "stwb": 1, "stcg": 2, "stcs": 3}[store_hint]
    return load_musa_jit(
        f"vllm_musa_rope_{arch_tag}_{store_hint}",
        ("rope/rotary_embedding.mu",),
        extra_musa_cflags=(f"-DVLLM_MUSA_ROPE_STORE_HINT={hint_value}",),
    )


def _rotary_embedding_kernel(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_tokens: int,
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
    rot_dim: int,
    seq_len: int,
    query_batch_stride: int,
    key_batch_stride: int,
    query_stride: int,
    key_stride: int,
    query_head_stride: int,
    key_head_stride: int,
    query_dim_stride: int,
    key_dim_stride: int,
    is_neox: bool,
    has_key: bool,
) -> None:
    _rope_module(
        _musa_arch_tag(), _store_hint(int(num_tokens), int(rot_dim))
    ).sgl_rotary_embedding(
        positions,
        query,
        key,
        cos_sin_cache,
        int(num_tokens),
        int(head_size),
        int(num_heads),
        int(num_kv_heads),
        int(rot_dim),
        int(seq_len),
        int(query_batch_stride),
        int(key_batch_stride),
        int(query_stride),
        int(key_stride),
        int(query_head_stride),
        int(key_head_stride),
        int(query_dim_stride),
        int(key_dim_stride),
        bool(is_neox),
        bool(has_key),
    )


def _rotary_embedding_impl(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    num_tokens = positions.numel()
    positions_ndim = positions.dim()
    if positions_ndim not in (1, 2):
        raise ValueError(
            "positions must have shape [num_tokens] or [batch_size, seq_len]"
        )
    if positions_ndim == 1:
        if query.size(0) != positions.size(0) or (
            key is not None and key.size(0) != positions.size(0)
        ):
            raise ValueError(
                "query, key and positions must have the same number of tokens"
            )
    else:
        if (
            query.size(0) != positions.size(0)
            or query.size(1) != positions.size(1)
            or (key is not None and key.size(0) != positions.size(0))
            or (key is not None and key.size(1) != positions.size(1))
        ):
            raise ValueError(
                "query, key and positions must have the same batch_size and seq_len"
            )

    query_hidden_size = query.numel() // num_tokens
    key_hidden_size = key.numel() // num_tokens if key is not None else 0
    if query_hidden_size % head_size != 0:
        raise ValueError("query hidden size must be divisible by head_size")
    if key is not None and key_hidden_size % head_size != 0:
        raise ValueError("key hidden size must be divisible by head_size")

    num_heads = query_hidden_size // head_size
    num_kv_heads = key_hidden_size // head_size if key is not None else num_heads
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    rot_dim = cos_sin_cache.size(1)
    if rot_dim % 2 != 0 or rot_dim > head_size:
        raise ValueError("rot_dim must be even and no larger than head_size")
    if positions.dtype != torch.long:
        raise TypeError("positions must be torch.int64")
    if query.dtype != cos_sin_cache.dtype or (
        key is not None and key.dtype != query.dtype
    ):
        raise TypeError("query, key and cos_sin_cache must have the same dtype")
    if not positions.is_contiguous():
        positions = positions.contiguous()
    if not cos_sin_cache.is_contiguous():
        cos_sin_cache = cos_sin_cache.contiguous()

    (
        query_batch_stride,
        query_stride,
        query_head_stride,
        query_dim_stride,
    ) = _layout_strides(query, positions_ndim, head_size)
    if key is not None:
        (
            key_batch_stride,
            key_stride,
            key_head_stride,
            key_dim_stride,
        ) = _layout_strides(key, positions_ndim, head_size)
    else:
        key_batch_stride = query_batch_stride
        key_stride = query_stride
        key_head_stride = query_head_stride
        key_dim_stride = query_dim_stride

    seq_len = positions.size(1) if positions_ndim == 2 else 0
    _rotary_embedding_kernel(
        positions.reshape(-1),
        query,
        key if key is not None else query,
        cos_sin_cache,
        int(num_tokens),
        int(head_size),
        int(num_heads),
        int(num_kv_heads),
        int(rot_dim),
        int(seq_len),
        int(query_batch_stride),
        int(key_batch_stride),
        int(query_stride),
        int(key_stride),
        int(query_head_stride),
        int(key_head_stride),
        int(query_dim_stride),
        int(key_dim_stride),
        bool(is_neox),
        key is not None,
    )


# @register_custom_op(
#     op_name="musa_rotary_embedding",
#     mutates_args=["query", "key"],
# )
def _rotary_embedding_custom(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    _rotary_embedding_impl(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox,
    )


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    _rotary_embedding_custom(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox,
    )
