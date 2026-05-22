"""MUSA TileLang rotary embedding kernels."""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from vllm_musa.jit_kernel.tilelang.utils import (
    MUSA_COMMON_PASS_CONFIGS,
    MUSA_COMPILE_FLAGS,
    layout_strides,
    storage_window,
    tilelang_dtype,
)

_ROPE_PASS_CONFIGS = dict(MUSA_COMMON_PASS_CONFIGS)
for _key, _value in (
    ("TL_DISABLE_SAFE_COPY_PREDICATION", True),
    ("TL_DISABLE_SAFE_ROBUST_COPY_PREDICATION", True),
    ("TL_CONFIG_INDEX_BITWIDTH", 32),
):
    if hasattr(tilelang.PassConfigKey, _key):
        _ROPE_PASS_CONFIGS[getattr(tilelang.PassConfigKey, _key)] = _value


@functools.lru_cache(maxsize=64)
@tilelang.jit(
    out_idx=[],
    target="musa",
    pass_configs=_ROPE_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _rotary_embedding_decode_kernel(
    dtype: str,
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
    rot_dim: int,
    query_storage_size: int,
    key_storage_size: int,
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
    block_pairs: int,
):
    num_tokens = T.symbolic("num_tokens")
    max_position = T.symbolic("max_position")
    half_rot_dim = rot_dim // 2
    max_head_pairs = max(num_heads, num_kv_heads) * half_rot_dim

    @T.prim_func
    def sglang_musa_rotary_embedding_decode_kernel(
        positions: T.Tensor((num_tokens,), "int64"),
        query: T.Tensor((query_storage_size,), dtype),
        key: T.Tensor((key_storage_size,), dtype),
        cos_sin_cache: T.Tensor((max_position, rot_dim), dtype),
    ):
        with T.Kernel(
            num_tokens,
            T.ceildiv(max_head_pairs, block_pairs),
            threads=block_pairs,
        ) as (token_idx, pair_block):
            pos = positions[token_idx]
            for i in T.Parallel(block_pairs):
                pair_idx = pair_block * block_pairs + i

                if pair_idx < num_heads * half_rot_dim:
                    q_head_idx = pair_idx // half_rot_dim
                    q_rot_offset = pair_idx - q_head_idx * half_rot_dim
                    if seq_len > 0:
                        q_batch_idx = token_idx // seq_len
                        q_seq_idx = token_idx - q_batch_idx * seq_len
                        q_token_base = (
                            q_batch_idx * query_batch_stride + q_seq_idx * query_stride
                        )
                    else:
                        q_token_base = token_idx * query_stride
                    q_base = q_token_base + q_head_idx * query_head_stride
                    if is_neox:
                        q_x_idx = q_rot_offset
                        q_y_idx = half_rot_dim + q_rot_offset
                    else:
                        q_x_idx = 2 * q_rot_offset
                        q_y_idx = q_x_idx + 1

                    q_cos = cos_sin_cache[pos, q_rot_offset]
                    q_sin = cos_sin_cache[pos, half_rot_dim + q_rot_offset]
                    q_x_offset = q_base + q_x_idx * query_dim_stride
                    q_y_offset = q_base + q_y_idx * query_dim_stride
                    q_x = query[q_x_offset].astype("float32")
                    q_y = query[q_y_offset].astype("float32")
                    q_cos_f = q_cos.astype("float32")
                    q_sin_f = q_sin.astype("float32")
                    query[q_x_offset] = (q_x * q_cos_f - q_y * q_sin_f).astype(dtype)
                    query[q_y_offset] = (q_y * q_cos_f + q_x * q_sin_f).astype(dtype)

                if has_key and pair_idx < num_kv_heads * half_rot_dim:
                    k_head_idx = pair_idx // half_rot_dim
                    k_rot_offset = pair_idx - k_head_idx * half_rot_dim
                    if seq_len > 0:
                        k_batch_idx = token_idx // seq_len
                        k_seq_idx = token_idx - k_batch_idx * seq_len
                        k_token_base = (
                            k_batch_idx * key_batch_stride + k_seq_idx * key_stride
                        )
                    else:
                        k_token_base = token_idx * key_stride
                    k_base = k_token_base + k_head_idx * key_head_stride
                    if is_neox:
                        k_x_idx = k_rot_offset
                        k_y_idx = half_rot_dim + k_rot_offset
                    else:
                        k_x_idx = 2 * k_rot_offset
                        k_y_idx = k_x_idx + 1

                    k_cos = cos_sin_cache[pos, k_rot_offset]
                    k_sin = cos_sin_cache[pos, half_rot_dim + k_rot_offset]
                    k_x_offset = k_base + k_x_idx * key_dim_stride
                    k_y_offset = k_base + k_y_idx * key_dim_stride
                    k_x = key[k_x_offset].astype("float32")
                    k_y = key[k_y_offset].astype("float32")
                    k_cos_f = k_cos.astype("float32")
                    k_sin_f = k_sin.astype("float32")
                    key[k_x_offset] = (k_x * k_cos_f - k_y * k_sin_f).astype(dtype)
                    key[k_y_offset] = (k_y * k_cos_f + k_x * k_sin_f).astype(dtype)

    return sglang_musa_rotary_embedding_decode_kernel


_rotary_embedding_decode_kernel.mode = "lazy"


@functools.lru_cache(maxsize=64)
@tilelang.jit(out_idx=[], target="musa", pass_configs=_ROPE_PASS_CONFIGS)
def _rotary_embedding_prefill_kernel(
    dtype: str,
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
    rot_dim: int,
    query_storage_size: int,
    key_storage_size: int,
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
    block_pairs: int,
    vec_size: int,
):
    num_tokens = T.symbolic("num_tokens")
    max_position = T.symbolic("max_position")
    half_rot_dim = rot_dim // 2
    max_head_pairs = max(num_heads, num_kv_heads) * half_rot_dim
    pair_blocks = T.ceildiv(max_head_pairs, block_pairs * vec_size)
    cache_blocks = T.ceildiv(rot_dim, block_pairs)

    @T.prim_func
    def sglang_musa_rotary_embedding_prefill_kernel(
        positions: T.Tensor((num_tokens,), "int64"),
        query: T.Tensor((query_storage_size,), dtype),
        key: T.Tensor((key_storage_size,), dtype),
        cos_sin_cache: T.Tensor((max_position, rot_dim), dtype),
    ):
        with T.Kernel(num_tokens, threads=block_pairs) as token_idx:
            pos = positions[token_idx]
            cache_shared = T.alloc_shared((rot_dim,), dtype)
            for cache_block in T.serial(cache_blocks):
                for i in T.Parallel(block_pairs):
                    cache_idx = cache_block * block_pairs + i
                    if cache_idx < rot_dim:
                        cache_shared[cache_idx] = cos_sin_cache[pos, cache_idx]
            T.sync_threads()

            for pair_block in T.serial(pair_blocks):
                for i in T.Parallel(block_pairs):
                    pair_base = (pair_block * block_pairs + i) * vec_size
                    for v in T.vectorized(vec_size):
                        pair_idx = pair_base + v

                        if pair_idx < num_heads * half_rot_dim:
                            q_head_idx = pair_idx // half_rot_dim
                            q_rot_offset = pair_idx - q_head_idx * half_rot_dim
                            if seq_len > 0:
                                q_batch_idx = token_idx // seq_len
                                q_seq_idx = token_idx - q_batch_idx * seq_len
                                q_token_base = (
                                    q_batch_idx * query_batch_stride
                                    + q_seq_idx * query_stride
                                )
                            else:
                                q_token_base = token_idx * query_stride
                            q_base = q_token_base + q_head_idx * query_head_stride
                            if is_neox:
                                q_x_idx = q_rot_offset
                                q_y_idx = half_rot_dim + q_rot_offset
                            else:
                                q_x_idx = 2 * q_rot_offset
                                q_y_idx = q_x_idx + 1

                            q_cos = cache_shared[q_rot_offset]
                            q_sin = cache_shared[half_rot_dim + q_rot_offset]
                            q_x_offset = q_base + q_x_idx * query_dim_stride
                            q_y_offset = q_base + q_y_idx * query_dim_stride
                            q_x = query[q_x_offset].astype("float32")
                            q_y = query[q_y_offset].astype("float32")
                            q_cos_f = q_cos.astype("float32")
                            q_sin_f = q_sin.astype("float32")
                            query[q_x_offset] = (q_x * q_cos_f - q_y * q_sin_f).astype(
                                dtype
                            )
                            query[q_y_offset] = (q_y * q_cos_f + q_x * q_sin_f).astype(
                                dtype
                            )

                        if has_key and pair_idx < num_kv_heads * half_rot_dim:
                            k_head_idx = pair_idx // half_rot_dim
                            k_rot_offset = pair_idx - k_head_idx * half_rot_dim
                            if seq_len > 0:
                                k_batch_idx = token_idx // seq_len
                                k_seq_idx = token_idx - k_batch_idx * seq_len
                                k_token_base = (
                                    k_batch_idx * key_batch_stride
                                    + k_seq_idx * key_stride
                                )
                            else:
                                k_token_base = token_idx * key_stride
                            k_base = k_token_base + k_head_idx * key_head_stride
                            if is_neox:
                                k_x_idx = k_rot_offset
                                k_y_idx = half_rot_dim + k_rot_offset
                            else:
                                k_x_idx = 2 * k_rot_offset
                                k_y_idx = k_x_idx + 1

                            k_cos = cache_shared[k_rot_offset]
                            k_sin = cache_shared[half_rot_dim + k_rot_offset]
                            k_x_offset = k_base + k_x_idx * key_dim_stride
                            k_y_offset = k_base + k_y_idx * key_dim_stride
                            k_x = key[k_x_offset].astype("float32")
                            k_y = key[k_y_offset].astype("float32")
                            k_cos_f = k_cos.astype("float32")
                            k_sin_f = k_sin.astype("float32")
                            key[k_x_offset] = (k_x * k_cos_f - k_y * k_sin_f).astype(
                                dtype
                            )
                            key[k_y_offset] = (k_y * k_cos_f + k_x * k_sin_f).astype(
                                dtype
                            )

    return sglang_musa_rotary_embedding_prefill_kernel


_rotary_embedding_prefill_kernel.mode = "lazy"


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor],
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
    ) = layout_strides(query, positions_ndim, head_size)
    if key is not None:
        (
            key_batch_stride,
            key_stride,
            key_head_stride,
            key_dim_stride,
        ) = layout_strides(key, positions_ndim, head_size)
    else:
        key_batch_stride = query_batch_stride
        key_stride = query_stride
        key_head_stride = query_head_stride
        key_dim_stride = query_dim_stride

    seq_len = positions.size(1) if positions_ndim == 2 else 0
    query_arg = storage_window(query)
    key_arg = storage_window(key) if key is not None else query_arg
    split_block_pairs = min(512, max(32, num_heads * rot_dim // 2))
    loop_block_pairs = min(512, max(32, num_heads * rot_dim // 2))
    if (
        num_tokens > 16384
        and query.is_contiguous()
        and (key is None or key.is_contiguous())
    ):
        loop_block_pairs = min(256, max(32, num_heads * rot_dim // 2))
    prefill_block_pairs = loop_block_pairs
    prefill_vec_size = 1
    if (
        128 < num_tokens <= 16384
        and positions_ndim == 1
        and query_dim_stride == 1
        and key_dim_stride == 1
    ):
        prefill_vec_size = 2

    if num_tokens <= 128:
        kernel = _rotary_embedding_decode_kernel(
            tilelang_dtype(query.dtype),
            int(head_size),
            int(num_heads),
            int(num_kv_heads),
            int(rot_dim),
            int(query_arg.numel()),
            int(key_arg.numel()),
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
            int(split_block_pairs),
        )
    else:
        kernel = _rotary_embedding_prefill_kernel(
            tilelang_dtype(query.dtype),
            int(head_size),
            int(num_heads),
            int(num_kv_heads),
            int(rot_dim),
            int(query_arg.numel()),
            int(key_arg.numel()),
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
            int(prefill_block_pairs),
            int(prefill_vec_size),
        )
    kernel(
        positions.reshape(-1),
        query_arg,
        key_arg,
        cos_sin_cache,
    )
