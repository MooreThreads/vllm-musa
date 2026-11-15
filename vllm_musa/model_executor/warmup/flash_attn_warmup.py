# SPDX-License-Identifier: Apache-2.0
"""Load paged BF16 FA3 variants before mixed traffic reaches lazy JIT paths."""

from __future__ import annotations

import importlib
import math

from vllm.logger import init_logger

logger = init_logger(__name__)
_WARMED: set[tuple] = set()


def _query_points(max_tokens: int, ratio: int) -> list[int]:
    # MATE changes the packed query tile at these effective-M boundaries and
    # disables packed GQA above 256 query tokens. Include both sides.
    points = {1, max_tokens, 256, 257}
    for tile in (16, 32, 64, 128, 192, 256):
        points.update((tile // ratio, tile // ratio + 1))
    return sorted(point for point in points if 1 <= point <= max_tokens)


def _fa3_warmup_configs(
    dispatch, heads: int, kv_heads: int, head_dim: int, max_tokens: int, max_seqs: int
) -> tuple[list[dict], list[dict]]:
    ratio = heads // kv_heads
    forward = {}
    packs = set()
    for query in _query_points(max_tokens, ratio):
        (
            tile_m,
            tile_n,
            stages_k,
            stages_v,
            rounded_qk,
            rounded_vo,
            consumers_qk,
            consumers_pv,
            tile_head_dim,
            pack_gqa,
        ) = dispatch(query, ratio, head_dim, head_dim, 2)
        packs.add(pack_gqa)
        config = {
            "has_cu_seqlens_q": True,
            "has_cu_seqlens_k": False,
            "has_seqused_q": False,
            "has_seqused_k": True,
            "has_cu_seqlens_k_new": False,
            "has_kv_batch_idx": False,
            "has_leftpad_k": False,
            "has_q_descale": True,
            "has_k_descale": True,
            "has_v_descale": True,
            "paged_kv": True,
            "has_softcap": False,
            "is_append_kv": False,
            "has_learnable_sink": False,
            "is_local": False,
            "is_causal": True,
            "is_packgqa": pack_gqa,
            "head_ratio": ratio,
            "element": "mutlass::bfloat16_t",
            "use_lsu_load_k": False,
            "use_lsu_load_v": False,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "stages_k": stages_k,
            "stages_v": stages_v,
            "head_dim_qk": rounded_qk,
            "head_dim_vo": rounded_vo,
            "consumers_qk": consumers_qk,
            "consumers_pv": consumers_pv,
            "tile_head_dim": tile_head_dim,
            "is_even_headdim": True,
            "has_metadata": True,
            "enable_cp": False,
            "is_rotary": False,
            "is_rotary_interleaved": False,
            "has_seqlens_rotary": False,
            "has_attention_chunk": False,
            "has_qv": False,
            "only_qv": False,
        }
        forward[tuple(sorted(config.items()))] = config
        if query == 1:
            # Mixed decode omits scheduler metadata and Q descale. Pure decode
            # can also omit metadata, but retains Q descale.
            for q_descale in (False, True):
                decode = dict(config, has_metadata=False, has_q_descale=q_descale)
                forward[tuple(sorted(decode.items()))] = decode
    metadata = []
    for pack_gqa in sorted(packs):
        for warps in range(1, min(math.ceil(max_seqs / 31), 32) + 1):
            metadata.append(
                {
                    "has_cu_seqlens_q": True,
                    "has_cu_seqlens_k": False,
                    "has_seqused_q": False,
                    "has_seqused_k": True,
                    "has_cu_seqlens_k_new": False,
                    "has_leftpad_k": False,
                    "is_causal": True,
                    "is_local": False,
                    "is_packgqa": pack_gqa,
                    "head_ratio": ratio,
                    "element_size": 2,
                    "sort": True,
                    "num_warps": warps,
                    "element": "mutlass::bfloat16_t",
                }
            )
    return list(forward.values()), metadata


def _load_fa3_modules(
    heads: int,
    kv_heads: int,
    head_dim: int,
    max_tokens: int,
    max_seqs: int,
    forward_module=None,
    metadata_module=None,
) -> tuple[int, int]:
    if forward_module is None:
        forward_module = importlib.import_module("mate.jit.attention.fmha.fmha_fwd")
    if metadata_module is None:
        metadata_module = importlib.import_module(
            "mate.jit.attention.fmha.fmha_get_metadata"
        )
    forward, metadata = _fa3_warmup_configs(
        forward_module._get_fwd_kernel_config,
        heads,
        kv_heads,
        head_dim,
        max_tokens,
        max_seqs,
    )
    for config in metadata:
        metadata_module._fmha_metadata_module(config)
    for config in forward:
        forward_module._fmha_fwd_module(config)
    return len(forward), len(metadata)


def _model_fa3_geometries(model) -> set[tuple[int, int, int]]:
    geometries = set()
    for layer in model.modules():
        impl = getattr(layer, "impl", None)
        if impl is None:
            continue
        cls = type(impl)
        if (
            cls.__name__ != "FlashAttentionImpl"
            or cls.__module__ != "vllm_musa.v1.attention.backends.flash_attn"
            or impl.vllm_flash_attn_version != 3
            or impl.attn_type != "decoder"
            or impl.kv_cache_dtype not in ("auto", "bfloat16")
            or impl.sinks is not None
            or impl.alibi_slopes is not None
            or impl.logits_soft_cap
            or impl.sliding_window != (-1, -1)
        ):
            continue
        heads, kv_heads, head_dim = impl.num_heads, impl.num_kv_heads, impl.head_size
        # Other geometries retain their existing warmup until qualified.
        if (
            kv_heads > 0
            and heads == kv_heads * 8
            and head_dim == 128
            and getattr(layer, "head_size_v", head_dim) == head_dim
        ):
            geometries.add((heads, kv_heads, head_dim))
    return geometries


def _runtime_context():
    import torch
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    return config, torch.bfloat16, torch.cuda.current_device()


def warmup_flash_attention(model, max_tokens: int, *, config=None) -> None:
    geometries = _model_fa3_geometries(model)
    if not geometries:
        return
    active_config, bf16, device = _runtime_context()
    if config is None:
        config = active_config
    if (
        config is None
        or config.model_config.dtype != bf16
        or config.cache_config.cache_dtype not in ("auto", "bfloat16")
        or config.parallel_config.decode_context_parallel_size != 1
        or config.cache_config.block_size != 64
    ):
        return
    max_seqs = min(config.scheduler_config.max_num_seqs, max_tokens)
    if max_tokens < 1 or max_seqs < 1:
        return
    for geometry in sorted(geometries):
        key = (*geometry, max_tokens, max_seqs, device)
        if key in _WARMED:
            continue
        logger.info(
            "Loading MUSA paged BF16 FA3 variants before serving: heads=%s "
            "max_tokens=%d max_seqs=%d",
            geometry,
            max_tokens,
            max_seqs,
        )
        forward_count, metadata_count = _load_fa3_modules(
            *geometry, max_tokens, max_seqs
        )
        _WARMED.add(key)
        logger.info(
            "Loaded MUSA paged BF16 FA3 variants: forward=%d metadata=%d",
            forward_count,
            metadata_count,
        )
