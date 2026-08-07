# SPDX-License-Identifier: Apache-2.0
"""Qwen3 QK RMSNorm + full RoPE + KV-cache side-effect custom op."""

from __future__ import annotations

import torch
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)


def qwen3_qk_rope_and_unified_kv_cache_update_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    is_neox: bool,
    mrope_section_t: int,
    mrope_section_h: int,
    mrope_section_w: int,
    is_interleaved: bool,
    eps: float,
    gemma: bool,
    layer_name: LayerNameType,
) -> torch.Tensor:
    """Populate Q/K outputs and the current layer's paged KV cache."""
    layer_name = _resolve_layer_name(layer_name)
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    attn_layer.impl.do_qwen3_qk_rope_and_kv_cache_update(
        attn_layer,
        q,
        k,
        v,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        q_out,
        k_out,
        kv_cache,
        layer_slot_mapping,
        is_neox,
        mrope_section_t,
        mrope_section_h,
        mrope_section_w,
        is_interleaved,
        eps,
        gemma,
    )
    return torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)


def qwen3_qk_rope_and_unified_kv_cache_update_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    is_neox: bool,
    mrope_section_t: int,
    mrope_section_h: int,
    mrope_section_w: int,
    is_interleaved: bool,
    eps: float,
    gemma: bool,
    layer_name: LayerNameType,
) -> torch.Tensor:
    del (
        k,
        v,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        q_out,
        k_out,
        is_neox,
        mrope_section_t,
        mrope_section_h,
        mrope_section_w,
        is_interleaved,
        eps,
        gemma,
        layer_name,
    )
    return torch.empty(0, device=q.device, dtype=q.dtype)


direct_register_custom_op(
    op_name="musa_qwen3_qk_rope_and_unified_kv_cache_update",
    op_func=qwen3_qk_rope_and_unified_kv_cache_update_impl,
    mutates_args=["q_out", "k_out"],
    fake_impl=qwen3_qk_rope_and_unified_kv_cache_update_fake,
)


FUSED_QWEN3_QK_ROPE_KV_OP = (
    torch.ops.vllm.musa_qwen3_qk_rope_and_unified_kv_cache_update.default
)


__all__ = ["FUSED_QWEN3_QK_ROPE_KV_OP"]
