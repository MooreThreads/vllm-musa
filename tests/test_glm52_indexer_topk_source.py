# SPDX-License-Identifier: Apache-2.0
"""Source-level coverage for the GLM-5.2 native indexer top-k path."""

from pathlib import Path

import ast
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _read(path: str) -> str:
    return (ROOT / path).read_text()


def test_glm52_indexer_kernel_matches_model_shape_and_topk():
    source = _read("csrc/musa/attention/glm52_indexer_topk.mu")

    assert "constexpr int64_t kNumHeads = 32;" in source
    assert "constexpr int64_t kHeadDim = 128;" in source
    assert "constexpr int64_t kMaxTopK = 2048;" in source
    assert "constexpr int64_t kMaxSeqLen = 8192;" in source
    assert "for (int shift = 24; shift >= 0; shift -= 8)" in source
    assert "for (int size = 2; size <= kMaxTopK; size <<= 1)" in source
    assert "if (seq_len <= topk)" in source
    assert "if (row_len <= topk)" in source


def test_glm52_indexer_ops_are_built_registered_and_wrapped():
    setup = _read("setup.py")
    bindings = _read("csrc/musa/torch_bindings.cpp")
    headers = _read("csrc/musa/musa_ops.h")
    custom_ops = _read("vllm_musa/_custom_ops.py")

    assert '"csrc/musa/attention/glm52_indexer_topk.mu"' in setup
    for op in (
        "sparse_indexer_fill_all",
        "sparse_indexer_topk",
        "sparse_indexer_topk_decode",
        "glm52_indexer_topk_decode",
        "glm52_indexer_topk_prefill",
    ):
        assert f'"{op}(' in bindings
        assert f"&{op}" in bindings
        assert f"void {op}(" in headers
        assert f"def {op}(" in custom_ops
        assert f"torch.ops._C_musa_ops.{op}" in custom_ops


def test_glm52_indexer_dispatch_has_shortcut_and_safe_fallback_boundary():
    series = _read(
        "vllm_musa/patches/series/"
        "0013-MUSA-vllm.model_executor.layers.sparse_attn_indexer.patch"
    )

    assert "VLLM_MUSA_GLM52_INDEXER_TOPK_NATIVE" in series
    assert "_musa_try_fill_all_sparse_indexer_indices" in series
    assert "int(metadata.max_seq_len) > int(topk_tokens)" in series
    assert "_musa_custom_ops.sparse_indexer_fill_all(" in series
    assert "VLLM_MUSA_GLM52_INDEXER_TOPK_MATERIALIZED" in series
    assert 'VLLM_MUSA_GLM52_INDEXER_TOPK_FUSED", "0"' in series
    assert "q_quant.shape[1] == 32" in series
    assert "topk_tokens <= 2048" in series
    assert "<= 8192" in series
    assert "_musa_custom_ops.glm52_indexer_topk_decode(" in series
    assert "_musa_custom_ops.glm52_indexer_topk_prefill(" in series
    assert "_musa_custom_ops.sparse_indexer_topk_decode(" in series
    assert "decode_metadata.schedule_metadata" in series
    assert "is_glm52_graph_path" in series


def test_musa_decode_schedule_metadata_is_precomputed_for_graph_capture():
    series = _read(
        "vllm_musa/patches/series/"
        "0079-MUSA-MUSA-vllm.v1.attention.backends.mla.indexer.patch"
    )

    assert "current_platform.is_musa()" in series
    assert "get_paged_mqa_logits_metadata" in series
    assert "self.scheduler_metadata_buffer" in series
    assert "schedule_metadata = (" in series
    assert "schedule_metadata[:] = metadata" in series


def test_deepseek_v4_dispatch_stays_shape_isolated():
    series = _read(
        "vllm_musa/patches/series/"
        "0013-MUSA-vllm.model_executor.layers.sparse_attn_indexer.patch"
    )

    assert "q_quant.shape[1] != 64" in series
    assert "topk_tokens > 512" in series
    assert "_musa_custom_ops.deepseek_v4_indexer_topk_decode(" in series
    assert "_musa_custom_ops.deepseek_v4_indexer_topk_prefill(" in series


def test_sparse_mla_sanitizes_indices_before_kv_loads():
    source = _read("vllm_musa/v1/attention/ops/sparse_mla_tilelang.py")

    assert 'kv_indices = T.alloc_fragment([BI], "int32")' in source
    assert "valid = (index >= 0) & (index < seq_len_kv)" in source
    assert "kv_indices[bi_i] = T.if_then_else(valid, index, 0)" in source
    assert "KV[b_i, kv_indices[bi_i], g_i, d_i]" in source
    assert "b_i, kv_indices[bi_i], g_i, D + d_i" in source


@pytest.mark.parametrize("shard_decode", [False, True])
@pytest.mark.parametrize(
    "warmup,skip,pcp,slots,k_rows,fp4",
    [
        (True, False, False, 3, 8, False),
        (True, False, False, 3, 8, True),
        (False, True, False, 3, 8, False),
        (False, False, False, 3, 8, False),
        (False, False, False, 5, 8, False),
        (False, False, False, 7, 8, False),
        (False, False, True, 8, 4, False),
        (False, False, True, 4, 4, False),
        (False, True, False, 3, 8, True),
        (False, False, False, 3, 8, True),
    ],
)
def test_musa_native_inserts_cache_before_exact_shortcut(
    warmup, skip, pcp, slots, k_rows, fp4, shard_decode
):
    """Exercise the real entry without importing GPU-dependent vLLM modules."""
    tree = ast.parse(_read(
        "third_party/vllm/vllm/model_executor/layers/sparse_attn_indexer.py"
    ))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "SparseAttnIndexer")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == "forward_native")
    for arg in method.args.args:
        arg.annotation = None
    method.returns = None
    events = []
    sliced_k = object()

    class Key:
        shape = (k_rows, 128)

        def __getitem__(self, index):
            assert index.start is None and index.step is None
            expected = slots // 2 if pcp and slots > k_rows else slots
            assert index.stop == expected
            return sliced_k

    metadata = SimpleNamespace(
        slot_mapping=SimpleNamespace(shape=(slots,)), num_decode_tokens=2,
        decode=None,
    )
    cache = object()
    buffer = object()
    gathered_k, gathered_slots = object(), object()
    gather = Mock(return_value=(gathered_k, gathered_slots))

    def insert(*args):
        assert args == (gathered_k, cache, gathered_slots, 128, "ue8m0")
        events.append("insert")

    def exact(*args):
        events.append("exact")
        return buffer

    platform = SimpleNamespace(**{
        f"is_{name}": (lambda: False) for name in ("cuda", "xpu", "rocm", "cpu")
    }, is_musa=lambda: True)
    namespace = dict(
        current_platform=platform,
        get_forward_context=lambda: SimpleNamespace(
            attn_metadata=None if warmup else {"resolved": metadata}
        ),
        _resolve_layer_name=lambda name: "resolved",
        DeepseekV32IndexerMetadata=SimpleNamespace,
        get_pcp_group=lambda: SimpleNamespace(world_size=2),
        maybe_gather_indexer_k=gather,
        ops=SimpleNamespace(indexer_k_quant_and_cache=insert),
        _musa_fill_exact_sparse_indexer_indices=exact,
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<forward_native>",
                 "exec"), namespace)
    layer = SimpleNamespace(
        use_musa_native_indexer=True, use_pcp=pcp,
        pcp_shard_decode_requests=shard_decode, skip_k_cache_insert=skip,
        scale_fmt="ue8m0", use_fp4_cache=fp4, quant_block_size=128,
        k_cache=SimpleNamespace(prefix="encoded", kv_cache=cache),
        topk_indices_buffer=buffer, topk_tokens=2048, head_dim=128,
        max_model_len=8192, use_musa_materialized_prefill=False,
    )
    forward = namespace["forward_native"]
    if fp4 and not skip and not warmup:
        with pytest.raises(AssertionError, match="Unfused FP4 Insert"):
            forward(layer, object(), object(), Key(), object())
        assert events == []
    else:
        assert forward(layer, object(), object(), None if skip else Key(),
                       object()) is buffer
        assert events == ([] if warmup else ["exact"] if skip
                          else ["insert", "exact"])
    if not warmup and not skip:
        gather.assert_called_once_with(
            sliced_k, metadata.slot_mapping, 2, pcp,
            pcp_shard_decode_requests=shard_decode,
        )
    else:
        gather.assert_not_called()
