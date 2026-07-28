# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_musa.jit_kernel import fa3_metadata


def _make_builder(flash_attn):
    builder = object.__new__(flash_attn.FlashAttentionMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.dcp_world_size = 1
    builder.dcp_rank = 0
    builder.cp_kv_cache_interleave_size = 1
    builder.reorder_batch_threshold = 1
    builder.use_full_cuda_graph = True
    builder.max_cudagraph_size = 8
    builder.max_num_splits = 32
    builder.aot_schedule = True
    builder.aot_sliding_window = (-1, -1)
    builder._sm_count = 60
    builder._sm_count_query_succeeded = True
    builder._use_qwen_single_request_scheduler_lookup = True
    builder._use_qwen_bs16_batched_metadata = False
    builder._logged_qwen_bs16_metadata = False
    builder._cu_seqlens_k_buffer = torch.zeros(9, dtype=torch.int32)
    builder.scheduler_metadata = torch.full((33,), -1, dtype=torch.int32)
    builder.num_heads_q = 16
    builder.num_heads_kv = 8
    builder.headdim = 128
    builder.block_size = 64
    builder.kv_cache_dtype = torch.bfloat16
    builder.model_config = SimpleNamespace(dtype=torch.bfloat16)
    builder._musa_qwen_family = True
    builder.cache_config = SimpleNamespace(cache_dtype="auto")
    return builder


def _make_common_metadata(
    *,
    num_reqs=1,
    num_actual_tokens=1,
    max_query_len=1,
    max_seq_len=128,
    causal=True,
):
    query_lens = [num_actual_tokens // num_reqs] * num_reqs
    for index in range(num_actual_tokens % num_reqs):
        query_lens[index] += 1
    query_start_loc = [0]
    for query_len in query_lens:
        query_start_loc.append(query_start_loc[-1] + query_len)

    return SimpleNamespace(
        num_reqs=num_reqs,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32),
        seq_lens=torch.full((num_reqs,), max_seq_len, dtype=torch.int32),
        block_table_tensor=torch.zeros((num_reqs, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(num_actual_tokens, dtype=torch.int64),
        causal=causal,
    )


def _make_lookup_config(*, model_type="qwen3", tensor_parallel_size=1):
    architecture = "Qwen3ForCausalLM" if model_type == "qwen3" else "LlamaForCausalLM"
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type=model_type,
                architectures=[architecture],
            ),
            architectures=[architecture],
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tensor_parallel_size,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        speculative_config=None,
    )


def test_qwen_fa3_scheduler_lookup_builder_direct_and_fallback(monkeypatch):
    from vllm_musa.v1.attention.backends import flash_attn

    builder = _make_builder(flash_attn)
    common_metadata = _make_common_metadata()
    monkeypatch.setattr(
        flash_attn,
        "split_decodes_and_prefills",
        lambda *_args, **_kwargs: (1, 0, 1, 0),
    )
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")

    direct_enabled = True
    direct_calls = []
    mate_calls = []

    def direct_builder(seq_lens, cu_seqlens_k, scheduler_dst, *args):
        direct_calls.append((seq_lens.clone(), args))
        if not direct_enabled:
            return False
        scheduler_dst.zero_()
        scheduler_dst[0] = 2
        scheduler_dst[8] = 1
        scheduler_dst[12] = 8
        cu_seqlens_k.copy_(torch.tensor([0, 128], dtype=torch.int32))
        return True

    def mate_builder(**_kwargs):
        mate_calls.append(True)
        return torch.arange(16, dtype=torch.int32)

    monkeypatch.setattr(
        fa3_metadata,
        "try_build_qwen_single_request_fa3_metadata",
        direct_builder,
    )
    monkeypatch.setattr(
        flash_attn,
        "get_scheduler_metadata",
        mate_builder,
        raising=False,
    )

    direct_result = builder.build(0, common_metadata)
    assert len(direct_calls) == 1
    assert direct_calls[0][1] == (128, 16, 8, 128, 8)
    assert not mate_calls
    assert direct_result.scheduler_metadata.tolist() == [
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        8,
        0,
        0,
        0,
    ]
    assert direct_result.cu_seqlens_k.tolist() == [0, 128]

    direct_enabled = False
    direct_calls.clear()
    fallback_result = builder.build(0, common_metadata)
    assert len(direct_calls) == 1
    assert len(mate_calls) == 1
    assert fallback_result.scheduler_metadata.tolist() == list(range(16))
    assert fallback_result.cu_seqlens_k.tolist() == [0, 128]
    assert builder.scheduler_metadata[16].item() == 0

    builder._sm_count = 56
    direct_calls.clear()
    mate_calls.clear()
    builder.build(0, common_metadata)
    assert not direct_calls
    assert len(mate_calls) == 1


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            {"builder": {"_use_qwen_single_request_scheduler_lookup": False}},
            id="config-disabled",
        ),
        pytest.param({"lookup_config": {"model_type": "llama"}}, id="non-qwen-config"),
        pytest.param(
            {"lookup_config": {"tensor_parallel_size": 2}},
            id="tensor-parallel-config",
        ),
        pytest.param({"builder": {"use_full_cuda_graph": False}}, id="no-full-graph"),
        pytest.param(
            {"builder": {"aot_schedule": False}, "mate_calls": 0},
            id="aot-schedule-disabled",
        ),
        pytest.param({"fast_build": True, "mate_calls": 0}, id="fast-build"),
        pytest.param(
            {"builder": {"_cu_seqlens_k_buffer": None}},
            id="missing-cu-seqlens-buffer",
        ),
        pytest.param(
            {
                "metadata": {"num_reqs": 2, "num_actual_tokens": 2},
                "split": (2, 0, 2, 0),
            },
            id="multi-request",
        ),
        pytest.param({"split": (0, 1, 0, 1)}, id="prefill"),
        pytest.param(
            {
                "metadata": {"num_actual_tokens": 2},
                "split": (1, 0, 2, 0),
            },
            id="multi-token-decode",
        ),
        pytest.param(
            {"metadata": {"max_query_len": 2}},
            id="max-query-len",
        ),
        pytest.param(
            {"metadata": {"max_seq_len": 0}},
            id="zero-sequence-length",
        ),
        pytest.param(
            {"metadata": {"max_seq_len": 4097}},
            id="sequence-too-long",
        ),
        pytest.param({"metadata": {"causal": False}}, id="non-causal"),
        pytest.param(
            {"builder": {"aot_sliding_window": (127, 0)}},
            id="sliding-window",
        ),
        pytest.param(
            {"common_prefix_len": 64, "mate_calls": 2},
            id="cascade",
        ),
        pytest.param(
            {"builder": {"dcp_world_size": 2}},
            id="decode-context-parallel",
        ),
        pytest.param(
            {"builder": {"_sm_count_query_succeeded": False}},
            id="mp-query-failed",
        ),
        pytest.param({"builder": {"_sm_count": 56}}, id="wrong-mp-count"),
        pytest.param(
            {"builder": {"max_cudagraph_size": 0}},
            id="zero-split-budget",
        ),
        pytest.param(
            {"builder": {"max_num_splits": 7}, "direct_calls": 1},
            id="unsupported-exact-split-count",
        ),
        pytest.param({"builder": {"num_heads_kv": 33}}, id="too-many-kv-heads"),
        pytest.param(
            {"builder": {"num_heads_q": 15}},
            id="non-integral-gqa-ratio",
        ),
        pytest.param({"builder": {"num_heads_q": 0}}, id="zero-gqa-ratio"),
        pytest.param(
            {"builder": {"num_heads_q": 64, "num_heads_kv": 1}},
            id="gqa-ratio-too-large",
        ),
        pytest.param({"builder": {"headdim": 256}}, id="unsupported-head-dim"),
        pytest.param({"builder": {"block_size": 128}}, id="unsupported-page-size"),
        pytest.param(
            {"builder": {"kv_cache_dtype": torch.float16}},
            id="non-bf16-kv-cache",
        ),
        pytest.param(
            {
                "builder": {"num_heads_q": 24, "num_heads_kv": 4},
                "direct_calls": 1,
            },
            id="unsupported-exact-geometry",
        ),
    ],
)
def test_qwen_fa3_scheduler_lookup_runtime_gate_fallback(monkeypatch, case):
    from vllm_musa.v1.attention.backends import flash_attn

    builder = _make_builder(flash_attn)
    for name, value in case.get("builder", {}).items():
        setattr(builder, name, value)

    lookup_config = case.get("lookup_config")
    if lookup_config is not None:
        builder._use_qwen_single_request_scheduler_lookup = (
            flash_attn._is_qwen_family_scheduler_lookup_base_config(
                _make_lookup_config(**lookup_config)
            )
        )

    common_metadata = _make_common_metadata(**case.get("metadata", {}))
    split_result = case.get("split", (1, 0, 1, 0))
    monkeypatch.setattr(
        flash_attn,
        "split_decodes_and_prefills",
        lambda *_args, **_kwargs: split_result,
    )
    monkeypatch.setattr(
        flash_attn,
        "get_dcp_local_seq_lens",
        lambda seq_lens, *_args, **_kwargs: seq_lens,
    )
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")

    direct_calls = []
    mate_calls = []

    def reject_direct_builder(
        _seq_lens,
        _cu_seqlens_k,
        _scheduler_dst,
        max_seq_len,
        num_heads_q,
        num_heads_kv,
        head_dim,
        max_num_splits,
    ):
        direct_calls.append(
            (max_seq_len, num_heads_q, num_heads_kv, head_dim, max_num_splits)
        )
        assert not fa3_metadata._supports_qwen_fa3_scheduler_geometry(
            max_seq_len,
            num_heads_q,
            num_heads_kv,
            head_dim,
            max_num_splits,
        )
        return False

    def mate_builder(**kwargs):
        mate_calls.append(kwargs)
        return torch.arange(16, dtype=torch.int32)

    monkeypatch.setattr(
        fa3_metadata,
        "try_build_qwen_single_request_fa3_metadata",
        reject_direct_builder,
    )
    monkeypatch.setattr(
        flash_attn,
        "get_scheduler_metadata",
        mate_builder,
        raising=False,
    )

    result = builder.build(
        case.get("common_prefix_len", 0),
        common_metadata,
        fast_build=case.get("fast_build", False),
    )

    assert len(direct_calls) == case.get("direct_calls", 0)
    assert len(mate_calls) == case.get("mate_calls", 1)
    if mate_calls:
        assert result.scheduler_metadata.tolist() == list(range(16))
    else:
        assert result.scheduler_metadata is None


def test_qwen_single_request_fa3_scheduler_lookup_support_gate():
    seq_lens = torch.zeros(1, dtype=torch.int32)
    cu_seqlens_k = torch.zeros(2, dtype=torch.int32)
    scheduler_dst = torch.zeros(17, dtype=torch.int32)

    assert fa3_metadata._supports_qwen_fa3_scheduler_geometry(4096, 16, 8, 128, 8)
    assert not fa3_metadata._supports_qwen_fa3_scheduler_geometry(4097, 16, 8, 128, 8)
    assert not fa3_metadata._supports_qwen_fa3_scheduler_geometry(128, 16, 8, 256, 8)
    assert not fa3_metadata._supports_qwen_fa3_scheduler_geometry(128, 8, 2, 256, 30)
    assert not fa3_metadata._supports_qwen_fa3_scheduler_geometry(128, 24, 25, 128, 3)
    assert not fa3_metadata._supports_qwen_fa3_scheduler_geometry(128, 16, 8, 128, 7)

    # Tensor support must fail closed away from a real MUSA device.
    assert not fa3_metadata.supports_qwen_single_request_fa3_scheduler_lookup(
        seq_lens, cu_seqlens_k, scheduler_dst, 128, 16, 8, 128, 8
    )


def test_qwen_single_request_fa3_scheduler_lookup_launches_once(monkeypatch):
    launches = []

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, args, kwargs))

            return launch

    monkeypatch.setattr(
        fa3_metadata,
        "_build_qwen_single_request_fa3_metadata_kernel",
        FakeKernel(),
    )
    monkeypatch.setattr(
        fa3_metadata,
        "supports_qwen_single_request_fa3_scheduler_lookup",
        lambda *args: True,
    )
    seq_lens = torch.zeros(1, dtype=torch.int32)
    cu_seqlens_k = torch.zeros(2, dtype=torch.int32)
    scheduler_dst = torch.zeros(513, dtype=torch.int32)

    assert fa3_metadata.try_build_qwen_single_request_fa3_metadata(
        seq_lens, cu_seqlens_k, scheduler_dst, 128, 16, 8, 128, 8
    )
    assert len(launches) == 1
    assert launches[0][0] == (3,)


def test_qwen_bs16_fa3_metadata_support_and_launch_gate(monkeypatch):
    seq_lens = torch.zeros(16, dtype=torch.int32)
    cu_seqlens_k = torch.zeros(17, dtype=torch.int32)
    scheduler_dst = torch.zeros(64, dtype=torch.int32)

    assert not fa3_metadata.supports_qwen_bs16_fa3_metadata(
        seq_lens, cu_seqlens_k, scheduler_dst, 4096, 32, 8, 128, 1
    )

    launches = []

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, args, kwargs))

            return launch

    monkeypatch.setattr(
        fa3_metadata,
        "_build_qwen_bs16_fa3_metadata_kernel",
        FakeKernel(),
    )
    monkeypatch.setattr(
        fa3_metadata,
        "supports_qwen_bs16_fa3_metadata",
        lambda *args: True,
    )
    assert fa3_metadata.try_build_qwen_bs16_fa3_metadata(
        seq_lens, cu_seqlens_k, scheduler_dst, 4096, 32, 8, 128, 1
    )
    assert len(launches) == 1
    assert launches[0][0] == (1,)
    assert launches[0][2]["BATCH_SIZE"] == 16


def test_qwen_bs16_builder_preserves_aot_scheduler(monkeypatch):
    from vllm_musa.v1.attention.backends import flash_attn

    builder = _make_builder(flash_attn)
    builder.max_cudagraph_size = 16
    builder.max_num_splits = 1
    builder.num_heads_q = 32
    builder.num_heads_kv = 8
    builder._cu_seqlens_k_buffer = torch.zeros(17, dtype=torch.int32)
    builder.scheduler_metadata = torch.full((65,), -1, dtype=torch.int32)
    common_metadata = _make_common_metadata(
        num_reqs=16,
        num_actual_tokens=16,
        max_query_len=1,
        max_seq_len=128,
    )
    monkeypatch.setattr(
        flash_attn,
        "split_decodes_and_prefills",
        lambda *_args, **_kwargs: (16, 0, 16, 0),
    )
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    builder._use_qwen_bs16_batched_metadata = True

    direct_calls = []

    def direct_builder(seq_lens, cu_seqlens_k, scheduler_dst, *args):
        direct_calls.append((seq_lens.clone(), args))
        scheduler_dst.copy_(torch.arange(64, dtype=torch.int32))
        cu_seqlens_k.copy_(torch.arange(17, dtype=torch.int32) * 128)
        return True

    monkeypatch.setattr(
        fa3_metadata,
        "try_build_qwen_bs16_fa3_metadata",
        direct_builder,
    )
    monkeypatch.setattr(
        flash_attn,
        "get_scheduler_metadata",
        lambda **_kwargs: pytest.fail("MATE metadata fallback should not run"),
        raising=False,
    )

    result = builder.build(0, common_metadata)
    assert len(direct_calls) == 1
    assert direct_calls[0][1] == (128, 32, 8, 128, 1)
    assert result.scheduler_metadata is not None
    assert result.scheduler_metadata.tolist() == list(range(64))
    assert result.cu_seqlens_k is not None
    assert result.cu_seqlens_k.tolist() == [index * 128 for index in range(17)]
    assert builder._logged_qwen_bs16_metadata


@pytest.mark.skipif(
    not (hasattr(torch, "musa") and torch.musa.is_available()),
    reason="requires a MUSA device",
)
@pytest.mark.parametrize(
    "seq_lens_values",
    [
        [1] * 16,
        [65] * 16,
        [4108] * 16,
        [8192] * 16,
        [
            1,
            64,
            65,
            127,
            128,
            129,
            511,
            512,
            513,
            1023,
            2047,
            4095,
            4096,
            4108,
            5000,
            8192,
        ],
    ],
)
def test_qwen_bs16_fa3_metadata_matches_mate(seq_lens_values):
    from vllm_musa.v1.attention.backends.fa_utils import get_scheduler_metadata

    seq_lens = torch.tensor(seq_lens_values, dtype=torch.int32, device="musa")
    cu_seqlens_q = torch.arange(17, dtype=torch.int32, device="musa")
    cu_seqlens_k = torch.full((17,), -1, dtype=torch.int32, device="musa")
    scheduler_dst = torch.full((64,), -1, dtype=torch.int32, device="musa")
    reference = get_scheduler_metadata(
        batch_size=16,
        max_seqlen_q=1,
        max_seqlen_k=max(seq_lens_values),
        num_heads_q=32,
        num_heads_kv=8,
        headdim=128,
        cache_seqlens=seq_lens,
        qkv_dtype=torch.bfloat16,
        cu_seqlens_q=cu_seqlens_q,
        page_size=64,
        causal=True,
        window_size=(-1, -1),
        num_splits=1,
    )

    assert fa3_metadata.try_build_qwen_bs16_fa3_metadata(
        seq_lens,
        cu_seqlens_k,
        scheduler_dst,
        max(seq_lens_values),
        32,
        8,
        128,
        1,
    )
    torch.musa.synchronize()
    assert torch.equal(scheduler_dst, reference)
    expected_cu = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device="musa"),
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
        )
    )
    assert torch.equal(cu_seqlens_k, expected_cu)


@pytest.mark.skipif(
    not (hasattr(torch, "musa") and torch.musa.is_available()),
    reason="requires a MUSA device",
)
@pytest.mark.parametrize(
    ("num_heads_q", "num_heads_kv", "head_dim"),
    [
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
    ],
)
@pytest.mark.parametrize("seq_len", [1, 64, 65, 384, 385, 4096])
def test_qwen_single_request_fa3_scheduler_lookup_values(
    seq_len,
    num_heads_q,
    num_heads_kv,
    head_dim,
):
    from vllm_musa.v1.attention.backends.fa_utils import get_scheduler_metadata

    max_num_splits = (60 + num_heads_kv - 1) // num_heads_kv
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="musa")
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device="musa")
    cu_seqlens_k = torch.full((2,), -1, dtype=torch.int32, device="musa")
    scheduler_dst = torch.full((513,), -1, dtype=torch.int32, device="musa")
    reference = get_scheduler_metadata(
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_k=seq_len,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        headdim=head_dim,
        cache_seqlens=seq_lens,
        qkv_dtype=torch.bfloat16,
        cu_seqlens_q=cu_seqlens_q,
        page_size=64,
        causal=True,
        window_size=(-1, -1),
        num_splits=max_num_splits,
    )

    assert fa3_metadata.try_build_qwen_single_request_fa3_metadata(
        seq_lens,
        cu_seqlens_k,
        scheduler_dst,
        seq_len,
        num_heads_q,
        num_heads_kv,
        head_dim,
        max_num_splits,
    )
    torch.musa.synchronize()

    semantic_indices = [0, 4, 8, 12]
    candidate_values = scheduler_dst[:16].cpu().tolist()
    reference_values = reference.cpu().tolist()
    assert [candidate_values[index] for index in semantic_indices] == [
        reference_values[index] for index in semantic_indices
    ]
    assert all(
        value == 0
        for index, value in enumerate(candidate_values)
        if index not in semantic_indices
    )
    assert torch.count_nonzero(scheduler_dst[16:]).item() == 0
    assert cu_seqlens_k.cpu().tolist() == [0, seq_len]
