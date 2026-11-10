"""Source contracts for the DSV4 learned-only sparse indexer."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "vllm_musa/patches/series"
INDEXER = ROOT / "third_party/vllm/vllm/model_executor/layers/sparse_attn_indexer.py"
ATTENTION = ROOT / "third_party/vllm/vllm/models/deepseek_v4/attention.py"
NATIVE = ROOT / "csrc/musa/attention/deepseek_v4_indexer_topk.mu"


def _additions(patch: Path) -> str:
    return "\n".join(
        line[1:] for line in patch.read_text().splitlines() if line.startswith("+")
    )


def test_recent_indexer_metadata_is_removed() -> None:
    indexer = INDEXER.read_text()
    builder = (
        ROOT / "third_party/vllm/vllm/v1/attention/backends/mla/indexer.py"
    ).read_text()

    assert "recent_indices" not in indexer
    assert "_build_recent_indices" not in builder
    assert "recent_indices_ready" not in builder


def test_attention_dispatch_keeps_learned_indexer_only() -> None:
    source = ATTENTION.read_text()
    indexer = INDEXER.read_text()

    assert "aux_fns[1] = indexer_weights_proj" in source
    assert "lambda: indexer(" in source
    assert "skip_indexer_weights" not in source
    assert "forward_musa_recent" not in indexer
    assert "uses_musa_recent_indices" not in indexer


def test_graph_decode_uses_learned_indexer_not_recent_window() -> None:
    patch = (
        SERIES / "0169-MUSA-carry-the-DeepSeek-V4-MUSA-paths-onto-the-atten.patch"
    ).read_text()
    source = INDEXER.read_text()

    assert "-    def uses_musa_recent_indices(self) -> bool:" in patch
    assert "-    def forward_musa_recent(self, hidden_states: torch.Tensor)" in patch
    assert "def _musa_fill_exact_sparse_indexer_indices_capture(" in source
    assert "_musa_try_fill_decode_topk_from_indexer_cache_native(" in source
    assert "MUSA learned-indexer capture supports the FP8 path only" in source
    assert "_musa_fill_recent_sparse_indexer_indices" not in source


def test_graph_decode_length_gate_never_host_syncs() -> None:
    additions = _additions(
        SERIES / "0169-MUSA-carry-the-DeepSeek-V4-MUSA-paths-onto-the-atten.patch"
    )

    assert "def _musa_decode_seq_len_fits_native_contract(" in additions
    source = INDEXER.read_text()
    assert (
        "if _musa_sparse_indexer_is_current_stream_capturing():\n        return True"
        in source
    )
    assert "_musa_decode_seq_len_fits_native_contract(" in source
    assert "block_table.size(1) * kv_cache.size(1) <= kMaxSeqLen" in NATIVE.read_text()


def test_dsv4_decode_can_use_learned_paged_mqa() -> None:
    source = INDEXER.read_text()
    builder = (
        ROOT / "third_party/vllm/vllm/v1/attention/backends/mla/indexer.py"
    ).read_text()

    assert "q_quant.shape[1] == 64" in source
    assert "or is_deepseek_v4" in source
    assert (
        "materialized_width = max(int(topk), min(int(max_model_len), page_capacity))"
        in source
    )
    assert "meta_lens = seq_lens" in builder
