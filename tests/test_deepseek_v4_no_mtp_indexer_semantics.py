from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "vllm_musa" / "patches" / "series"
LEARNED_CAPTURE_PATCH = (
    SERIES / "0125-MUSA-keep-DSV4-learned-indexer-on-graph-capture.patch"
)
DISPATCH_PATCH = SERIES / "0130-MUSA-keep-DSV4-learned-indexer-dispatch.patch"
GRAPH_PATCH = SERIES / "0131-MUSA-keep-DSV4-learned-indexer-on-graph-decode.patch"


def test_dsv4_native_indexer_does_not_select_metadata_only_recent_path() -> None:
    """No-MTP DSV4 must retain learned-indexer semantics during graph capture."""
    patch = LEARNED_CAPTURE_PATCH.read_text()
    assert (
        "-        # v4.1 two-level selection: the candidate source indexer writes the"
        in patch
    )
    assert (
        "-        # top candidate blocks here; later indexers mask their scores with it."
        in patch
    )


def test_dsv4_attention_and_graph_decode_are_learned_only() -> None:
    dispatch = DISPATCH_PATCH.read_text()
    graph = GRAPH_PATCH.read_text()
    dispatch_additions = "\n".join(
        line[1:] for line in dispatch.splitlines() if line.startswith("+")
    )
    graph_additions = "\n".join(
        line[1:] for line in graph.splitlines() if line.startswith("+")
    )

    assert (
        "_musa_deepseek_v4_serialize_long_prefill(hidden_states.size(0))"
        in dispatch_additions
    )
    assert "use_stream_waits=current_platform.is_musa()" in dispatch_additions
    assert "q = self.wq_b(qr).view" in dispatch_additions
    assert "_wq_b_proj" not in dispatch_additions
    assert "def uses_musa_recent_indices(self)" in graph_additions
    assert "if self.use_musa_native_indexer:" in graph_additions
    assert "forward_musa_recent" in graph_additions
    assert "_musa_fill_recent_sparse_indexer_indices" in graph_additions
