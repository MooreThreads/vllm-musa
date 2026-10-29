# SPDX-License-Identifier: Apache-2.0
"""CPU-only guards for per-attention-window FA3 scheduler metadata.

MATE's FA3 AOT schedule depends on the layer's attention window: the window
bounds the visible KV blocks, which bounds ``num_n_blocks`` and therefore the
split count. A hybrid model (gemma2/3/4, gpt-oss, qwen3-next/3.5, mimo, ...)
mixes full-attention and sliding-window layers, so one schedule cannot serve
both. Upstream vLLM disabled AOT scheduling for the whole model as soon as it
saw more than one window config, which silently cost every such model the
prebuilt schedule on every layer.

These tests pin the replacement contract without a GPU: the layout scan, the
one-schedule-per-window build, the per-layer selection, the persistent
cuda-graph buffers, and the cases that must stay single-slot (DCP, cascade,
spec-decode fast build, single-window models).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_musa.v1.attention.backends import flash_attn

FULL = (-1, -1)
SWA_128 = (127, 0)
SWA_1024 = (1023, 0)
SCHEDULER_SLOTS = 16


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
def _common_metadata(*, num_reqs=2, num_actual_tokens=2, max_seq_len=4096, causal=True):
    query_lens = [num_actual_tokens // num_reqs] * num_reqs
    for index in range(num_actual_tokens % num_reqs):
        query_lens[index] += 1
    query_start_loc = [0]
    for query_len in query_lens:
        query_start_loc.append(query_start_loc[-1] + query_len)
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_actual_tokens=num_actual_tokens,
        max_query_len=1,
        max_seq_len=max_seq_len,
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32),
        seq_lens=torch.full((num_reqs,), max_seq_len, dtype=torch.int32),
        block_table_tensor=torch.zeros((num_reqs, 64), dtype=torch.int32),
        slot_mapping=torch.zeros(num_actual_tokens, dtype=torch.int64),
        causal=causal,
    )


def _metadata(*, scheduler_metadata=None, kind_scheduler_metadata=None):
    return flash_attn.FlashAttentionMetadata(
        num_actual_tokens=2,
        max_query_len=1,
        query_start_loc=torch.zeros(3, dtype=torch.int32),
        max_seq_len=8,
        seq_lens=torch.zeros(2, dtype=torch.int32),
        block_table=torch.zeros((2, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(2, dtype=torch.int64),
        num_decodes=2,
        num_decode_tokens=2,
        decode_query_start_loc=None,
        decode_seq_lens=None,
        decode_block_table=None,
        num_prefills=0,
        num_prefill_tokens=0,
        prefill_query_start_loc=None,
        prefill_max_seq_len=0,
        cu_seqlens_k=None,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        scheduler_metadata=scheduler_metadata,
        kind_scheduler_metadata=kind_scheduler_metadata,
    )


def _make_builder(
    *,
    layouts=None,
    use_full_cuda_graph=True,
    aot_schedule=True,
    dcp_world_size=1,
    max_num_splits=8,
):
    """A builder carrying only the state ``build()`` reads (no ``__init__``)."""
    builder = object.__new__(flash_attn.FlashAttentionMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.vllm_config = SimpleNamespace()
    builder.num_heads_q = 16
    builder.num_heads_kv = 8
    builder.headdim = 128
    builder.block_size = 64
    builder.kv_cache_dtype = torch.bfloat16
    builder.model_config = SimpleNamespace(dtype=torch.bfloat16)
    builder.cache_config = SimpleNamespace(cache_dtype="auto")
    builder.dcp_world_size = dcp_world_size
    builder.dcp_rank = 0
    builder.cp_kv_cache_interleave_size = 1
    builder.num_speculative_tokens = 0
    builder.decode_threshold = 1
    builder.max_num_splits = max_num_splits
    builder.aot_schedule = aot_schedule
    builder._sm_count = 60
    builder._sm_count_query_succeeded = True
    builder.use_full_cuda_graph = use_full_cuda_graph
    builder.max_cudagraph_size = 8
    builder._use_qwen_single_request_scheduler_lookup = False
    builder._musa_optimization_contract = SimpleNamespace(
        prefers=lambda *_args, **_kwargs: False
    )
    builder.aot_sliding_window = None
    builder.aot_window_layouts = layouts
    builder._kind_scheduler_metadata_buffers = {}
    builder.scheduler_metadata = torch.zeros(33, dtype=torch.int32)
    builder._cu_seqlens_k_buffer = (
        torch.zeros(9, dtype=torch.int32) if use_full_cuda_graph else None
    )
    return builder


class _SchedulerStub:
    """Records ``get_scheduler_metadata`` calls and the tensors returned."""

    def __init__(self):
        self.calls: list[dict] = []
        self.returned: dict[tuple[int, int], torch.Tensor] = {}

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        tensor = torch.full(
            (SCHEDULER_SLOTS,), len(self.calls), dtype=torch.int32
        )
        self.returned[kwargs["window_size"]] = tensor
        return tensor

    @property
    def windows(self):
        return [call["window_size"] for call in self.calls]


@pytest.fixture
def stub(monkeypatch):
    """Neutralize everything in build() that is not the metadata-kind logic."""
    monkeypatch.setattr(flash_attn.envs, "VLLM_BATCH_INVARIANT", False, raising=False)
    monkeypatch.setattr(
        flash_attn,
        "split_decodes_and_prefills",
        lambda *_args, **_kwargs: (2, 0, 2, 0),
    )
    monkeypatch.setattr(
        flash_attn, "_use_musa_qwen_direct_decode_schedule", lambda **_kwargs: False
    )
    monkeypatch.setattr(
        flash_attn, "_get_attention_window_layouts", lambda _config: (FULL,)
    )
    # Most tests exercise the per-window path, which is opt-in.
    monkeypatch.setattr(flash_attn, "_AOT_PER_WINDOW_ENABLED", True)
    scheduler = _SchedulerStub()
    monkeypatch.setattr(
        flash_attn, "get_scheduler_metadata", scheduler, raising=False
    )
    return scheduler


def _layer(*, window, attn_type=None):
    impl = object.__new__(flash_attn.FlashAttentionImpl)
    impl.sliding_window = window
    impl.attn_type = attn_type or flash_attn.AttentionType.DECODER
    return SimpleNamespace(impl=impl)


# --------------------------------------------------------------------------- #
# layout scan
# --------------------------------------------------------------------------- #
def test_window_normalization_only_maps_the_none_spelling():
    assert flash_attn._normalize_attention_window(None) == FULL
    assert flash_attn._normalize_attention_window((-1, -1)) == FULL
    assert flash_attn._normalize_attention_window(SWA_128) == SWA_128
    assert flash_attn._normalize_attention_window((15, 15)) == (15, 15)
    # Right-bounded means right-bounded: same visible KV only under a causal
    # mask, so it must stay a distinct kind rather than share a slot.
    assert flash_attn._normalize_attention_window((-1, 0)) == (-1, 0)
    assert flash_attn._normalize_attention_window((0, -1)) == (0, -1)


def test_layouts_are_deduplicated_and_full_attention_first(monkeypatch):
    layers = {
        "a": _layer(window=SWA_128),
        "b": _layer(window=(-1, -1)),
        "c": _layer(window=SWA_1024),
        "d": _layer(window=SWA_128),
    }
    monkeypatch.setattr(flash_attn, "get_layers_from_vllm_config", lambda *_a: layers)
    assert flash_attn._get_attention_window_layouts(SimpleNamespace()) == (
        FULL,
        SWA_128,
        SWA_1024,
    )


def test_layouts_exclude_encoder_layers(monkeypatch):
    """An encoder layer never reads a KV-cache schedule, so it is not a kind."""
    layers = {
        "full": _layer(window=(-1, -1)),
        "enc": _layer(
            window=(15, 15), attn_type=flash_attn.AttentionType.ENCODER_ONLY
        ),
    }
    monkeypatch.setattr(flash_attn, "get_layers_from_vllm_config", lambda *_a: layers)
    assert flash_attn._get_attention_window_layouts(SimpleNamespace()) == (FULL,)


def test_layouts_default_to_full_attention_without_layers(monkeypatch):
    monkeypatch.setattr(flash_attn, "get_layers_from_vllm_config", lambda *_a: {})
    assert flash_attn._get_attention_window_layouts(SimpleNamespace()) == (FULL,)


# --------------------------------------------------------------------------- #
# builder: hybrid models keep AOT scheduling
# --------------------------------------------------------------------------- #
def test_hybrid_model_keeps_aot_scheduling(monkeypatch, stub):
    """The upstream regression: >1 sliding-window config must not kill AOT."""
    monkeypatch.setattr(
        flash_attn, "_get_attention_window_layouts", lambda _config: (FULL, SWA_128)
    )
    builder = _make_builder()
    builder.build(0, _common_metadata())
    assert builder.aot_schedule is True
    assert builder.aot_sliding_window == FULL
    assert builder.aot_window_layouts == (FULL, SWA_128)


def test_hybrid_builds_one_schedule_per_window(stub):
    builder = _make_builder(layouts=(FULL, SWA_128, SWA_1024))
    metadata = builder.build(0, _common_metadata())

    assert stub.windows == [FULL, SWA_128, SWA_1024]
    kinds = metadata.kind_scheduler_metadata
    assert kinds is not None
    assert set(kinds) == {FULL, SWA_128, SWA_1024}
    assert kinds[FULL] is metadata.scheduler_metadata
    # Distinct schedules, not one tensor shared by every kind.
    assert len({tensor.data_ptr() for tensor in kinds.values()}) == 3


def test_single_window_model_carries_no_kind_map(stub):
    """A plain full-attention model must keep the upstream single-slot path."""
    builder = _make_builder(layouts=(FULL,))
    metadata = builder.build(0, _common_metadata())
    assert stub.windows == [FULL]
    assert metadata.kind_scheduler_metadata is None
    assert metadata.scheduler_metadata is not None


def test_single_sliding_window_model_uses_that_window(stub):
    builder = _make_builder(layouts=(SWA_128,))
    metadata = builder.build(0, _common_metadata())
    assert stub.windows == [SWA_128]
    assert metadata.kind_scheduler_metadata is None


def test_layouts_resolve_on_first_aot_build_not_on_fast_build(
    monkeypatch, stub
):
    """A spec-decode fast build must not pin the model to full attention."""
    resolved = []

    def layouts(_config):
        resolved.append(True)
        return (FULL, SWA_128)

    monkeypatch.setattr(flash_attn, "_get_attention_window_layouts", layouts)
    builder = _make_builder(layouts=None)

    builder.build(0, _common_metadata(), fast_build=True)
    assert builder.aot_window_layouts is None
    assert resolved == []

    builder.build(0, _common_metadata())
    assert builder.aot_window_layouts == (FULL, SWA_128)
    assert resolved == [True]


# --------------------------------------------------------------------------- #
# builder: paths that must stay single-slot
# --------------------------------------------------------------------------- #
def test_multi_window_aot_is_opt_in(monkeypatch, stub):
    """Default behaviour must match upstream: one window schedule only."""
    monkeypatch.setattr(flash_attn, "_AOT_PER_WINDOW_ENABLED", False)
    monkeypatch.setattr(
        flash_attn, "_get_attention_window_layouts", lambda _config: (FULL, SWA_128)
    )
    builder = _make_builder()
    metadata = builder.build(0, _common_metadata())
    assert builder.aot_schedule is False
    assert builder.aot_window_layouts == (FULL,)
    assert builder.aot_sliding_window == FULL
    assert stub.calls == []
    assert metadata.scheduler_metadata is None
    assert metadata.kind_scheduler_metadata is None


def test_opt_in_multi_window_schedules_each_window(monkeypatch, stub):
    monkeypatch.setattr(flash_attn, "_AOT_PER_WINDOW_ENABLED", True)
    monkeypatch.setattr(
        flash_attn, "_get_attention_window_layouts", lambda _config: (FULL, SWA_128)
    )
    builder = _make_builder()
    metadata = builder.build(0, _common_metadata())
    assert builder.aot_schedule is True
    assert stub.windows == [FULL, SWA_128]
    assert metadata.scheduler_metadata_for(SWA_128) is not None


def test_dcp_window_mixed_model_keeps_aot_disabled(monkeypatch, stub):
    """DCP still reads one two-sided slot, so a mixed model must not enable AOT."""
    monkeypatch.setattr(
        flash_attn, "_get_attention_window_layouts", lambda _config: (FULL, SWA_128)
    )
    monkeypatch.setattr(
        flash_attn, "get_dcp_local_seq_lens", lambda seq_lens, *_a, **_k: seq_lens
    )
    builder = _make_builder(dcp_world_size=2)
    metadata = builder.build(0, _common_metadata())
    assert builder.aot_schedule is False
    assert builder.aot_window_layouts == (FULL,)
    assert stub.calls == []
    assert metadata.scheduler_metadata is None
    assert metadata.kind_scheduler_metadata is None


def test_dcp_single_window_model_keeps_aot(monkeypatch, stub):
    monkeypatch.setattr(
        flash_attn, "_get_attention_window_layouts", lambda _config: (FULL,)
    )
    monkeypatch.setattr(
        flash_attn, "get_dcp_local_seq_lens", lambda seq_lens, *_a, **_k: seq_lens
    )
    builder = _make_builder(dcp_world_size=2)
    metadata = builder.build(0, _common_metadata())
    assert stub.windows == [FULL]
    assert metadata.kind_scheduler_metadata is None


def test_cascade_batch_keeps_single_slot(monkeypatch, stub):
    """Cascade attention is only reachable for full-attention layers."""
    monkeypatch.setattr(
        flash_attn, "_get_attention_window_layouts", lambda _config: (FULL, SWA_128)
    )
    builder = _make_builder()
    metadata = builder.build(64, _common_metadata())
    assert metadata.use_cascade is True
    assert metadata.kind_scheduler_metadata is None


def test_fast_build_spec_decode_has_no_aot_schedule(stub):
    builder = _make_builder(layouts=(FULL, SWA_128))
    metadata = builder.build(0, _common_metadata(), fast_build=True)
    assert metadata.scheduler_metadata is None
    assert metadata.kind_scheduler_metadata is None
    assert stub.calls == []


# --------------------------------------------------------------------------- #
# builder: graph buffers
# --------------------------------------------------------------------------- #
def test_graph_capture_uses_one_persistent_buffer_per_window(stub):
    builder = _make_builder(layouts=(FULL, SWA_128), use_full_cuda_graph=True)
    metadata = builder.build(0, _common_metadata())

    swa = metadata.kind_scheduler_metadata[SWA_128]
    buffer = builder._kind_scheduler_metadata_buffers[SWA_128]
    assert swa.data_ptr() == buffer.data_ptr()
    assert swa.numel() == SCHEDULER_SLOTS
    # The full-attention slot is still the upstream buffer.
    assert (
        metadata.kind_scheduler_metadata[FULL].data_ptr()
        == builder.scheduler_metadata.data_ptr()
    )
    # The unused tail is zeroed so no thread block can read a stale schedule.
    assert torch.count_nonzero(buffer[SCHEDULER_SLOTS:]).item() == 0


def test_graph_buffers_are_rewritten_in_place_each_build(stub):
    builder = _make_builder(layouts=(FULL, SWA_128), use_full_cuda_graph=True)
    builder.build(0, _common_metadata())
    buffer = builder._kind_scheduler_metadata_buffers[SWA_128]
    first = buffer[0].item()

    builder.build(0, _common_metadata())
    assert builder._kind_scheduler_metadata_buffers[SWA_128] is buffer
    assert buffer[0].item() != first


def test_eager_build_passes_fresh_schedules_through(stub):
    builder = _make_builder(layouts=(FULL, SWA_128), use_full_cuda_graph=False)
    metadata = builder.build(0, _common_metadata())
    assert builder._kind_scheduler_metadata_buffers == {}
    assert (
        metadata.kind_scheduler_metadata[SWA_128]
        is stub.returned[SWA_128]
    )
    assert (
        metadata.kind_scheduler_metadata[FULL]
        is stub.returned[FULL]
    )


# --------------------------------------------------------------------------- #
# metadata: per-layer selection
# --------------------------------------------------------------------------- #
def test_layer_selects_its_own_window_schedule():
    full = torch.zeros(4, dtype=torch.int32)
    swa = torch.ones(4, dtype=torch.int32)
    metadata = _metadata(
        scheduler_metadata=full,
        kind_scheduler_metadata={FULL: full, SWA_128: swa},
    )
    assert metadata.scheduler_metadata_for(FULL) is full
    assert metadata.scheduler_metadata_for(SWA_128) is swa
    assert metadata.scheduler_metadata_for(None) is full
    # A window the map does not know gets no schedule at all: handing it another
    # window's schedule would describe a different visible-KV problem.
    assert metadata.scheduler_metadata_for(SWA_1024) is None


def test_single_slot_metadata_resolves_to_the_primary_schedule():
    full = torch.zeros(4, dtype=torch.int32)
    metadata = _metadata(scheduler_metadata=full)
    assert metadata.scheduler_metadata_for(SWA_128) is full


def test_layer_without_a_schedule_resolves_to_none():
    metadata = _metadata()
    assert metadata.scheduler_metadata_for(FULL) is None


# --------------------------------------------------------------------------- #
# metadata forwarding
# --------------------------------------------------------------------------- #
def test_update_block_table_preserves_the_kind_map(stub):
    builder = _make_builder(layouts=(FULL, SWA_128), use_full_cuda_graph=False)
    metadata = builder.build(0, _common_metadata())
    kind_map = metadata.kind_scheduler_metadata
    table = torch.ones((2, 64), dtype=torch.int32)

    updated = builder.update_block_table(metadata, table, metadata.slot_mapping)

    assert updated.kind_scheduler_metadata is kind_map
    assert updated.block_table is table
    assert updated.scheduler_metadata is metadata.scheduler_metadata


# --------------------------------------------------------------------------- #
# forwarding: the schedule that reaches the kernel call
# --------------------------------------------------------------------------- #
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_SIZE = 8
BLOCK_SIZE = 4
NUM_BLOCKS = 8


def _impl(*, window):
    """The layer half of the backend, without __init__ (needs MATE present)."""
    impl = object.__new__(flash_attn.FlashAttentionImpl)
    impl.num_heads = NUM_HEADS
    impl.head_size = HEAD_SIZE
    impl.scale = HEAD_SIZE**-0.5
    impl.num_kv_heads = NUM_KV_HEADS
    impl.num_queries_per_kv = NUM_HEADS // NUM_KV_HEADS
    impl.alibi_slopes = None
    impl.sliding_window = window
    impl._scheduler_metadata_window = flash_attn._normalize_attention_window(window)
    impl.kv_cache_dtype = "auto"
    impl.logits_soft_cap = 0
    impl.kv_sharing_target_layer_name = None
    impl.attn_type = flash_attn.AttentionType.DECODER
    impl.vllm_flash_attn_version = 3
    impl.batch_invariant_enabled = False
    impl.sinks = None
    impl._mubin_prefill_ok = window == FULL
    impl.supports_quant_query_input = True
    impl.dcp_world_size = 1
    return impl


def _layer_weights():
    return SimpleNamespace(
        _q_scale=torch.ones(1), _k_scale=torch.ones(1), _v_scale=torch.ones(1)
    )


def _kv_cache():
    # MUSA keeps K and V packed side by side in the last dimension.
    return torch.zeros(
        2, NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, 2 * HEAD_SIZE, dtype=torch.bfloat16
    )


def _decode_metadata(kind_map, primary):
    return flash_attn.FlashAttentionMetadata(
        num_actual_tokens=2,
        max_query_len=1,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        max_seq_len=2 * BLOCK_SIZE,
        seq_lens=torch.tensor([2 * BLOCK_SIZE, 2 * BLOCK_SIZE], dtype=torch.int32),
        block_table=torch.zeros((2, 2), dtype=torch.int32),
        slot_mapping=torch.zeros(2, dtype=torch.int64),
        num_decodes=2,
        num_decode_tokens=2,
        decode_query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        decode_seq_lens=torch.tensor(
            [2 * BLOCK_SIZE, 2 * BLOCK_SIZE], dtype=torch.int32
        ),
        decode_block_table=torch.zeros((2, 2), dtype=torch.int32),
        num_prefills=0,
        num_prefill_tokens=0,
        prefill_query_start_loc=None,
        prefill_max_seq_len=0,
        cu_seqlens_k=None,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        scheduler_metadata=primary,
        kind_scheduler_metadata=kind_map,
        max_num_splits=1,
    )


def _run_forward(monkeypatch, impl, metadata, *, num_tokens=2, kernel_name):
    calls = []

    def kernel(**kwargs):
        calls.append(kwargs)
        return torch.zeros(
            num_tokens, NUM_HEADS * HEAD_SIZE, dtype=torch.bfloat16
        )

    monkeypatch.setattr(flash_attn, kernel_name, kernel, raising=False)
    monkeypatch.setattr(
        flash_attn, "flash_attn_varlen_func", kernel, raising=False
    )
    output = torch.zeros(num_tokens, NUM_HEADS * HEAD_SIZE, dtype=torch.bfloat16)
    result = impl.forward(
        layer=_layer_weights(),
        query=torch.zeros(num_tokens, NUM_HEADS, HEAD_SIZE, dtype=torch.bfloat16),
        key=torch.zeros(num_tokens, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.bfloat16),
        value=torch.zeros(num_tokens, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.bfloat16),
        kv_cache=_kv_cache(),
        attn_metadata=metadata,
        output=output,
    )
    assert result is output
    return calls


def test_decode_forward_uses_the_layers_own_schedule(monkeypatch):
    """A SWA layer in a hybrid model must not run on the full-attention schedule."""
    full = torch.zeros(SCHEDULER_SLOTS, dtype=torch.int32)
    swa = torch.ones(SCHEDULER_SLOTS, dtype=torch.int32)
    kind_map = {FULL: full, SWA_128: swa}

    swa_calls = _run_forward(
        monkeypatch,
        _impl(window=SWA_128),
        _decode_metadata(kind_map, full),
        kernel_name="flash_attn_with_kvcache",
    )
    assert swa_calls[0]["scheduler_metadata"] is swa

    full_calls = _run_forward(
        monkeypatch,
        _impl(window=FULL),
        _decode_metadata(kind_map, full),
        kernel_name="flash_attn_with_kvcache",
    )
    assert full_calls[0]["scheduler_metadata"] is full


def test_paged_varlen_forward_uses_the_layers_own_schedule(monkeypatch):
    """The paged prefill/verify fallback selects per layer too."""
    full = torch.zeros(SCHEDULER_SLOTS, dtype=torch.int32)
    swa = torch.ones(SCHEDULER_SLOTS, dtype=torch.int32)
    kind_map = {FULL: full, SWA_128: swa}
    metadata = _decode_metadata(kind_map, full)
    # Two query tokens on one decode row: neither the pure-decode fast path nor
    # the MTP-verify path, so this lands on the paged varlen call.
    metadata.num_actual_tokens = 2
    metadata.max_query_len = 2
    metadata.decode_seq_lens = torch.tensor([2 * BLOCK_SIZE], dtype=torch.int32)
    metadata.decode_query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    metadata.decode_block_table = torch.zeros((1, 2), dtype=torch.int32)
    metadata.num_decodes = 1
    metadata.num_decode_tokens = 1
    metadata.query_start_loc = torch.tensor([0, 2], dtype=torch.int32)

    calls = _run_forward(
        monkeypatch,
        _impl(window=SWA_128),
        metadata,
        kernel_name="flash_attn_varlen_func",
    )
    assert calls[0]["scheduler_metadata"] is swa


def test_layer_window_absent_from_the_map_drops_aot(monkeypatch):
    """No schedule for this window is slower, never wrong."""
    full = torch.zeros(SCHEDULER_SLOTS, dtype=torch.int32)
    metadata = _decode_metadata({FULL: full}, full)
    calls = _run_forward(
        monkeypatch,
        _impl(window=SWA_1024),
        metadata,
        kernel_name="flash_attn_with_kvcache",
    )
    assert calls[0]["scheduler_metadata"] is None
