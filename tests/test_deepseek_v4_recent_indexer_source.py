"""Source contracts for the graph recent-window indexer optimization."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa/patches/series/0092-MUSA-skip-unused-DeepSeek-V4-recent-indexer-Q-work.patch"
)


def test_recent_path_skips_q_and_weights_but_keeps_indexer_compressor():
    source = PATCH.read_text()

    assert "VLLM_MUSA_DEEPSEEK_V4_RECENT_INDEXER_SKIP_Q" not in source
    assert "if not use_recent_indexer:" in source
    assert "aux_fns[2] = indexer_compressor_kv_score" in source
    assert "self.compressor(compressed_kv_score, positions, rotary_emb)" in source
    assert "return self.indexer_op.forward_musa_recent(hidden_states)" in source


def test_recent_indices_are_hoisted_to_metadata_once_per_c4_group():
    source = PATCH.read_text()

    assert "recent_indices: torch.Tensor | None = None" in source
    assert "self.recent_indices_buffer" in source
    assert "self.compress_ratio == 4" in source
    assert "recent_indices = self._build_recent_indices(" in source
    assert "metadata.recent_indices_ready" in source
    assert "def _build_recent_indices(" in source


def test_eager_break_does_not_recheck_stream_capture_for_recent_callback():
    source = PATCH.read_text()
    method = source.split(
        "    def forward_musa_recent(self, hidden_states: torch.Tensor)"
    )[1].split("    def forward_cuda(", 1)[0]

    # The selection predicate runs before the eager break.  Rechecking capture
    # inside the callback would be false and would disable or raise at runtime.
    assert "_musa_sparse_indexer_is_current_stream_capturing" not in method
    assert "raise RuntimeError" not in method
