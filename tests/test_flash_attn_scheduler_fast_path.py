# SPDX-License-Identifier: Apache-2.0
"""Exact gates for the MUSA Qwen direct FA3 decode schedule."""

import inspect
from types import SimpleNamespace

import pytest
import torch

import vllm_musa.v1.attention.backends.flash_attn as flash_attn
from vllm_musa.v1.attention.backends.flash_attn import (
    FlashAttentionMetadataBuilder,
    _is_musa_qwen_text_generation_architecture,
    _use_musa_qwen_direct_decode_schedule,
)


@pytest.mark.parametrize(
    ("architectures", "expected"),
    [
        (["Qwen2ForCausalLM"], True),
        (["Qwen2MoeForCausalLM"], True),
        (["Qwen3ForCausalLM"], True),
        (["Qwen3MoeForCausalLM"], True),
        (["Qwen3_5ForConditionalGeneration"], True),
        (["Qwen3_5MoeForConditionalGeneration"], True),
        (["CosyVoice3Model"], True),
        (["Qwen2AudioForConditionalGeneration"], False),
        (["Qwen2VLForConditionalGeneration"], False),
        (["Qwen2_5OmniForConditionalGeneration"], False),
        (["Qwen3ASRForConditionalGeneration"], False),
        (["Qwen3OmniMoeForConditionalGeneration"], False),
        (["Qwen3VLForConditionalGeneration"], False),
        (["Qwen3ForSequenceClassification"], False),
        (["Qwen3NextMTP"], False),
        (["LlamaForCausalLM"], False),
        (None, False),
    ],
)
def test_musa_qwen_family_gate(architectures, expected: bool) -> None:
    model_config = SimpleNamespace(architectures=architectures)
    assert _is_musa_qwen_text_generation_architecture(model_config) is expected


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, True),
        ({"aot_schedule": False}, False),
        ({"causal": False}, False),
        ({"is_bfloat16": False}, False),
        ({"is_qwen_family": False}, False),
        ({"use_full_cuda_graph": False}, False),
        ({"common_prefix_len": 1}, False),
        ({"dcp_world_size": 2}, False),
        ({"graph_num_reqs": 63, "graph_num_decodes": 63}, False),
        ({"graph_num_decodes": 63}, False),
        ({"graph_num_tokens": 63, "graph_num_decode_tokens": 63}, False),
        ({"graph_num_decode_tokens": 63}, False),
        ({"max_query_len": 2}, False),
        ({"max_num_splits": 2}, False),
    ],
)
def test_qwen_direct_decode_schedule_gate(kwargs, expected: bool) -> None:
    values = {
        "aot_schedule": True,
        "causal": True,
        "is_bfloat16": True,
        "is_qwen_family": True,
        "use_full_cuda_graph": True,
        "common_prefix_len": 0,
        "dcp_world_size": 1,
        "graph_num_reqs": 64,
        "graph_num_decodes": 64,
        "graph_num_tokens": 64,
        "graph_num_decode_tokens": 64,
        "max_query_len": 1,
        "max_num_splits": 1,
    }
    values.update(kwargs)
    assert _use_musa_qwen_direct_decode_schedule(**values) is expected


def test_mtp_decode_threshold_includes_speculative_tokens() -> None:
    source = " ".join(
        inspect.getsource(flash_attn.FlashAttentionMetadataBuilder.__init__).split()
    )

    assert (
        "self.decode_threshold = ( self.reorder_batch_threshold "
        "+ self.num_speculative_tokens )" in source
    )


def test_mtp_verify_uses_graph_safe_kvcache_kernel() -> None:
    source = inspect.getsource(flash_attn.FlashAttentionImpl.forward)
    branch = source[source.index("if use_fused_kv_verify:") :]

    assert "num_decode_tokens > num_decodes" in source
    assert "max_seqlen_q > 1" in source
    assert branch.index("flash_attn_with_kvcache(") < branch.index("else:")


def _make_graph_size_64_builder(*, is_qwen_family: bool):
    builder = object.__new__(FlashAttentionMetadataBuilder)
    builder.aot_schedule = True
    builder.aot_sliding_window = (-1, -1)
    builder.use_full_cuda_graph = True
    builder.max_cudagraph_size = 64
    builder.max_num_splits = 32
    builder._sm_count = 60
    builder.num_heads_q = 32
    builder.num_heads_kv = 8
    builder.headdim = 128
    builder.block_size = 32
    builder.dcp_world_size = 1
    builder.dcp_rank = 0
    builder.cp_kv_cache_interleave_size = 1
    builder.reorder_batch_threshold = 1
    builder.num_speculative_tokens = 0
    builder.decode_threshold = 1
    builder.kv_cache_dtype = torch.bfloat16
    builder.model_config = SimpleNamespace(dtype=torch.bfloat16)
    builder.cache_config = SimpleNamespace(cache_dtype="auto")
    builder._musa_qwen_fa3_scheduler = is_qwen_family
    builder._use_qwen_single_request_scheduler_lookup = True
    builder._sm_count_query_succeeded = True
    builder._cu_seqlens_k_buffer = torch.empty(65, dtype=torch.int32)
    builder.scheduler_metadata = torch.zeros(257, dtype=torch.int32)
    return builder


def _make_graph_size_64_metadata():
    return SimpleNamespace(
        num_reqs=64,
        num_actual_tokens=64,
        max_query_len=1,
        max_seq_len=4159,
        query_start_loc=torch.arange(65, dtype=torch.int32),
        seq_lens=torch.arange(4096, 4160, dtype=torch.int32),
        block_table_tensor=torch.zeros((64, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(64, dtype=torch.int64),
        causal=True,
    )


def test_builder_skips_aot_metadata_for_qwen_graph_size_64(monkeypatch) -> None:
    from vllm_musa.jit_kernel import fa3_metadata

    monkeypatch.setattr(
        flash_attn,
        "split_decodes_and_prefills",
        lambda *_args, **_kwargs: (64, 0, 64, 0),
    )

    def unexpected_scheduler_call(**_kwargs):
        pytest.fail("direct graph-size-64 path called the AOT scheduler")

    monkeypatch.setattr(flash_attn, "get_scheduler_metadata", unexpected_scheduler_call)
    monkeypatch.setattr(
        fa3_metadata,
        "try_build_qwen_single_request_fa3_metadata",
        lambda *_args, **_kwargs: pytest.fail(
            "graph-size-64 path called the batch-one metadata lookup"
        ),
    )
    metadata = _make_graph_size_64_builder(is_qwen_family=True).build(
        common_prefix_len=0,
        common_attn_metadata=_make_graph_size_64_metadata(),
    )

    assert metadata.max_num_splits == 1
    assert metadata.scheduler_metadata is None


def test_builder_keeps_aot_metadata_outside_qwen_allowlist(monkeypatch) -> None:
    monkeypatch.setattr(
        flash_attn,
        "split_decodes_and_prefills",
        lambda *_args, **_kwargs: (64, 0, 64, 0),
    )
    calls = 0

    def scheduler_metadata(**_kwargs):
        nonlocal calls
        calls += 1
        return torch.ones(1, dtype=torch.int32)

    monkeypatch.setattr(flash_attn, "get_scheduler_metadata", scheduler_metadata)
    metadata = _make_graph_size_64_builder(is_qwen_family=False).build(
        common_prefix_len=0,
        common_attn_metadata=_make_graph_size_64_metadata(),
    )

    assert calls == 1
    assert metadata.scheduler_metadata is not None
