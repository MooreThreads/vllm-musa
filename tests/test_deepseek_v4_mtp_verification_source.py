from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0110-MUSA-adapt-DSV4-MTP-v028.patch"
)


def _patch_text() -> str:
    return PATCH.read_text(encoding="utf-8")


def test_mtp_verification_uses_graph_descriptor_before_config() -> None:
    text = _patch_text()

    descriptor_pos = text.index(
        "batch_descriptor = get_forward_context().batch_descriptor"
    )
    config_pos = text.index("config = get_current_vllm_config()")

    assert descriptor_pos < config_pos
    assert "batch_descriptor.num_tokens > batch_descriptor.num_reqs" in text
    assert "speculative_config.method != \"mtp\"" in text


def test_mtp_verification_keeps_learned_indexer_semantics() -> None:
    text = _patch_text()

    assert "or _musa_sparse_indexer_mtp_requires_learned()" in text
    assert (
        "if _musa_sparse_indexer_mtp_requires_learned():\n"
        "+            return False"
    ) in text


def test_multi_request_mtp_graph_serializes_both_aux_groups() -> None:
    text = _patch_text()

    assert "and batch_descriptor.num_reqs > 1" in text
    assert text.count("_musa_dsv4_mtp_multi_request_graph()") == 3
    assert "_musa_dsv4_mtp_multi_request_graph()" in text
    assert "if _musa_dsv4_mtp_multi_request_graph():" in text


def test_patch_is_scoped_to_deepseek_v4_runtime() -> None:
    changed_files = {
        line.removeprefix("+++ b/")
        for line in _patch_text().splitlines()
        if line.startswith("+++ b/")
    }

    assert changed_files == {
        "vllm/model_executor/layers/sparse_attn_indexer.py",
        "vllm/models/deepseek_v4/attention.py",
    }
