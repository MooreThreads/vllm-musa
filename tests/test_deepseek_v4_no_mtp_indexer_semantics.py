from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "vllm_musa" / "patches" / "series"
LEARNED_CAPTURE_PATCH = SERIES / "0130-MUSA-keep-DSV4-learned-indexer-on-graph-capture.patch"
METADATA_PATCH = SERIES / "0158-MUSA-remove-DSV4-recent-indexer-metadata.patch"
DISPATCH_PATCH = SERIES / "0135-MUSA-keep-DSV4-learned-indexer-dispatch.patch"
GRAPH_PATCH = SERIES / "0136-MUSA-keep-DSV4-learned-indexer-on-graph-decode.patch"
GRAPH_SAFE_PATCH = SERIES / "0161-MUSA-keep-DSV4-learned-decode-graph-safe.patch"
PAGED_MQA_PATCH = SERIES / "0162-MUSA-enable-DSV4-learned-paged-MQA-decode.patch"


def test_dsv4_native_indexer_does_not_select_metadata_only_recent_path() -> None:
    """No-MTP DSV4 must retain learned-indexer semantics during graph capture."""
    patch = LEARNED_CAPTURE_PATCH.read_text()
    snippet = (
        "if self.use_musa_native_indexer:\n"
        "+            return False"
    )
    assert snippet in patch
    assert "The recent table is metadata-only" in patch


def test_dsv4_recent_indexer_is_not_constructed_or_filled() -> None:
    """The DSV4 MLA metadata path must not allocate or build a recent table."""
    patch = METADATA_PATCH.read_text()
    assert "self.recent_indices_buffer" in patch
    assert "_build_recent_indices" in patch
    assert "-    recent_indices: torch.Tensor | None = None" in patch
    additions = "\n".join(
        line[1:] for line in patch.splitlines() if line.startswith("+")
    )
    assert "recent_indices" not in additions
    assert "_build_recent_indices" not in additions


def test_dsv4_attention_and_graph_decode_are_learned_only() -> None:
    dispatch = DISPATCH_PATCH.read_text()
    graph = GRAPH_PATCH.read_text()
    dispatch_additions = "\n".join(
        line[1:] for line in dispatch.splitlines() if line.startswith("+")
    )
    graph_additions = "\n".join(
        line[1:] for line in graph.splitlines() if line.startswith("+")
    )

    assert "self._run_parallel_input_projections(hidden_states)" in dispatch_additions
    assert "aux_fns[1] = indexer_weights_proj" in dispatch_additions
    assert "forward_musa_recent" not in dispatch_additions
    assert "return _musa_fill_exact_sparse_indexer_indices_capture(" in graph_additions
    assert "_musa_decode_request_max_seq_len" in graph_additions
    assert "_musa_fill_recent_sparse_indexer_indices" not in graph_additions

    graph_safe = GRAPH_SAFE_PATCH.read_text()
    graph_safe_additions = "\n".join(
        line[1:] for line in graph_safe.splitlines() if line.startswith("+")
    )
    assert "_musa_decode_seq_len_fits_native_contract" in graph_safe_additions
    assert "if _musa_sparse_indexer_is_current_stream_capturing():" in graph_safe_additions
    assert "q_quant.to(torch.float32)" not in graph_safe_additions

    paged = PAGED_MQA_PATCH.read_text()
    paged_additions = "\n".join(
        line[1:] for line in paged.splitlines() if line.startswith("+")
    )
    assert "is_deepseek_v4" in paged_additions
    assert "or is_deepseek_v4" in paged_additions
    assert "meta_lens" in paged_additions
