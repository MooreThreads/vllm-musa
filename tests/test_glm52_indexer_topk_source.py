# SPDX-License-Identifier: Apache-2.0
"""Source-level coverage for the GLM-5.2 native indexer top-k path."""

from pathlib import Path


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
        assert f'&{op}' in bindings
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
        "0085-MUSA-vllm.v1.attention.backends.mla.indexer.patch"
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
