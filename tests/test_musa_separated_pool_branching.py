# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Branching tests for the MUSA separate Mamba/attention KV pool.

The MUSA dedicated-pool path may only build attention tensors from the resolved
layout strides when one model has *more than one* attention KV cache group.
That happens when a speculative-decoding draft owns its own attention cache
(e.g. DFlash2/DSpark2 on Qwen3.8); MTP and every non-speculative Mamba/hybrid
model keep a single attention group, and attention-only models never reach the
MUSA builder at all.

These tests pin the three dispatch decisions:

* ``_musa_separate_pool_eligible`` needs both a ``MambaSpec`` group and a
  non-Mamba group, so a model without Mamba state never enters the MUSA pool;
* one attention group keeps the legacy per-slot construction (one layer per
  tensor, tensor size == layer stride, offset 0) that every existing model
  uses today;
* two attention groups take the layout-aware construction, alias one attention
  buffer, and every view stays inside its backing tensor (the storage overrun
  this path fixes).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core import kv_cache_utils as kcu
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    MambaSpec,
)
from vllm.v1.worker.utils import allocate_kv_cache

BLOCK_SIZE = 16
HEAD_SIZE = 64


def _mamba_spec() -> MambaSpec:
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((4, 8),),
        dtypes=(torch.float32,),
    )


def _attention_spec(num_kv_heads: int) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=num_kv_heads,
        head_size=HEAD_SIZE,
        dtype=torch.float32,
    )


def _vllm_config(layout: KVCacheLayout = KVCacheLayout.BLHNC) -> SimpleNamespace:
    cache_config = SimpleNamespace(
        num_gpu_blocks_override=None,
        mamba_cache_mode="none",
        prefix_cache_retention_interval=None,
        get_resolved_kv_cache_layout=lambda: layout,
    )
    return SimpleNamespace(
        cache_config=cache_config,
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        attention_config=SimpleNamespace(hisparse_config=None),
        kv_transfer_config=None,
    )


def _split_tensors(tensors):
    attention = [t for t in tensors if not t.musa_mamba_pool]
    mamba = [t for t in tensors if t.musa_mamba_pool]
    return attention, mamba


def _last_addressed_byte(tensor, num_blocks: int, page_size: int) -> int:
    """Highest byte a ``num_blocks``-block view of ``tensor`` can touch."""
    return (
        tensor.offset
        + (len(tensor.layers) - 1) * tensor.layer_stride
        + (num_blocks - 1) * tensor.block_stride
        + page_size
    )


def test_separate_pool_needs_mamba_and_attention(monkeypatch):
    monkeypatch.setattr(kcu, "musa_mamba_separate_pool_enabled", lambda: True)
    config = _vllm_config()
    attention = KVCacheGroupSpec(["target.0"], _attention_spec(2))
    mamba = KVCacheGroupSpec(["mamba.0"], _mamba_spec())

    # Attention-only (no Mamba state): never eligible.
    assert not kcu._musa_separate_pool_eligible(config, [attention])
    # Mamba-only (attention-free model): no attention pool to split.
    assert not kcu._musa_separate_pool_eligible(config, [mamba])
    # Mamba state plus an attention cache: the MUSA pool split applies.
    assert kcu._musa_separate_pool_eligible(config, [mamba, attention])


def test_single_attention_group_keeps_per_slot_tensors(monkeypatch):
    """A non-speculative / MTP model keeps one layer per tensor and its own pool."""
    monkeypatch.setattr(kcu, "musa_mamba_separate_pool_enabled", lambda: True)
    spec = _attention_spec(2)
    attention_layers = [f"target.{i}" for i in range(3)]
    groups = [
        KVCacheGroupSpec(attention_layers, spec),
        KVCacheGroupSpec(["mamba.0"], _mamba_spec()),
    ]

    tensors, num_blocks, mamba_blocks = kcu._musa_separated_mamba_attn_tensors(
        _vllm_config(), groups, 1 << 20
    )
    attention, mamba = _split_tensors(tensors)

    assert num_blocks > 0 and mamba_blocks > 0
    assert len(attention) == len(attention_layers)
    for tensor in attention:
        assert len(tensor.layers) == 1
        assert tensor.offset == 0
        # Legacy per-slot layout: the tensor is exactly one layer's pool.
        assert tensor.size == tensor.layer_stride
        assert tensor.block_stride == spec.page_size_bytes
        last_byte = _last_addressed_byte(tensor, num_blocks, spec.page_size_bytes)
        assert last_byte <= tensor.size
    assert len(mamba) == 1
    assert mamba[0].layers == ["mamba.0"]


def test_single_group_allocation_keeps_one_buffer_per_layer(monkeypatch):
    monkeypatch.setattr(kcu, "musa_mamba_separate_pool_enabled", lambda: True)
    spec = _attention_spec(2)
    attention_layers = ["target.0", "target.1", "target.2"]
    groups = [
        KVCacheGroupSpec(attention_layers, spec),
        KVCacheGroupSpec(["mamba.0"], _mamba_spec()),
    ]
    tensors, num_blocks, mamba_blocks = kcu._musa_separated_mamba_attn_tensors(
        _vllm_config(), groups, 1 << 20
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        musa_mamba_num_blocks=mamba_blocks,
    )

    caches = allocate_kv_cache(config, torch.device("cpu"), KVCacheLayout.BLHNC)
    attention_ptrs = {
        caches[name].untyped_storage().data_ptr() for name in attention_layers
    }
    # Each per-slot tensor owns its buffer, exactly as before this change.
    assert len(attention_ptrs) == len(attention_layers)
    assert caches["mamba.0"].untyped_storage().data_ptr() not in attention_ptrs


@pytest.mark.parametrize(
    ("layout", "draft_heads"),
    [
        (KVCacheLayout.BLHNC, 2),
        (KVCacheLayout.BLHNC, 4),
        (KVCacheLayout.LBNHC, 2),
    ],
)
def test_two_attention_groups_take_layout_path(monkeypatch, layout, draft_heads):
    """The draft-owned attention cache is a second group -> layout strides."""
    monkeypatch.setattr(kcu, "musa_mamba_separate_pool_enabled", lambda: True)
    config = _vllm_config(layout)
    target = KVCacheGroupSpec(["target.0", "target.1"], _attention_spec(2))
    draft = KVCacheGroupSpec(
        ["draft.0", "draft.1"], _attention_spec(draft_heads)
    )
    mamba = KVCacheGroupSpec(["mamba.0"], _mamba_spec())
    groups = [target, draft, mamba]

    tensors, num_blocks, mamba_blocks = kcu._musa_separated_mamba_attn_tensors(
        config, groups, 1 << 20
    )
    attention, _ = _split_tensors(tensors)

    bytes_per_block = kcu._get_kv_cache_bytes_per_block([target, draft], layout)
    total_size = bytes_per_block * num_blocks
    assert len(attention) == 2
    assert {len(t.layers) for t in attention} == {2}
    for tensor in attention:
        # Both groups alias one attention pool from byte 0.
        assert tensor.offset == 0
        assert tensor.size == total_size
        page = next(
            group.kv_cache_spec.page_size_bytes
            for group in (target, draft)
            if tensor.layers[0] in group.layer_names
        )
        # The multi-layer view fits; the old per-slot sizing did not.
        assert _last_addressed_byte(tensor, num_blocks, page) <= tensor.size

    kv_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        musa_mamba_num_blocks=mamba_blocks,
    )
    caches = allocate_kv_cache(kv_config, torch.device("cpu"), layout)
    attention_ptrs = {
        caches[name].untyped_storage().data_ptr()
        for name in ("target.0", "target.1", "draft.0", "draft.1")
    }
    # Layout-aware tensors alias one attention buffer; Mamba stays separate.
    assert len(attention_ptrs) == 1
    assert caches["mamba.0"].untyped_storage().data_ptr() not in attention_ptrs


@pytest.mark.parametrize("layout", [KVCacheLayout.BLHNC, KVCacheLayout.LBNHC])
def test_single_layer_attention_groups_share_one_buffer(monkeypatch, layout):
    """Two one-layer attention groups still share one attention buffer.

    The layout-aware construction sizes every attention tensor from the whole
    pool, so gating the shared buffer on ``len(t.layers) > 1`` would allocate
    one full pool buffer per group here.
    """
    monkeypatch.setattr(kcu, "musa_mamba_separate_pool_enabled", lambda: True)
    config = _vllm_config(layout)
    target = KVCacheGroupSpec(["target.0"], _attention_spec(2))
    draft = KVCacheGroupSpec(["draft.0"], _attention_spec(2))
    mamba = KVCacheGroupSpec(["mamba.0"], _mamba_spec())
    groups = [target, draft, mamba]

    tensors, num_blocks, mamba_blocks = kcu._musa_separated_mamba_attn_tensors(
        config, groups, 1 << 20
    )
    attention, _ = _split_tensors(tensors)
    assert len(attention) == 2
    assert {len(t.layers) for t in attention} == {1}
    total_size = attention[0].size
    assert all(t.size == total_size for t in attention)

    kv_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        musa_mamba_num_blocks=mamba_blocks,
    )
    caches = allocate_kv_cache(kv_config, torch.device("cpu"), layout)
    target_ptr = caches["target.0"].untyped_storage().data_ptr()
    draft_ptr = caches["draft.0"].untyped_storage().data_ptr()
    assert target_ptr == draft_ptr, (
        "one-layer attention groups must alias one pool buffer, not one full "
        "buffer each"
    )
    assert caches["target.0"].untyped_storage().nbytes() == total_size
    assert caches["mamba.0"].untyped_storage().data_ptr() != target_ptr


def test_attention_only_model_never_reaches_musa_builder(monkeypatch):
    """No Mamba group -> the MUSA separate-pool builder is not called."""

    def _must_not_run(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError(
            "MUSA separate-pool builder must not run for an attention-only model"
        )

    monkeypatch.setattr(kcu, "_musa_separated_mamba_attn_tensors", _must_not_run)
    groups = [KVCacheGroupSpec(["target.0"], _attention_spec(2))]

    config = kcu.get_kv_cache_config_from_groups(
        _vllm_config(KVCacheLayout.LBNHC), groups, 1 << 20
    )

    assert config.musa_mamba_num_blocks is None
    assert config.num_blocks > 0
    assert config.kv_cache_tensors
