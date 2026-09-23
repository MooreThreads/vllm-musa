from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0170-MUSA-bound-DSV4-long-prefill-indexer-logits.patch"
)
POLICY = ROOT / "vllm_musa" / "optimization_contract" / "policy.py"
INIT = ROOT / "vllm_musa" / "optimization_contract" / "__init__.py"


def _text() -> str:
    return PATCH.read_text(encoding="utf-8")


def _changed_files(text: str) -> set[str]:
    return {
        line[len("+++ b/") :].split("\t", 1)[0]
        for line in text.splitlines()
        if line.startswith("+++ b/")
    }


def _diff_lines(text: str, prefix: str) -> str:
    return "\n".join(
        line[1:]
        for line in text.splitlines()
        if line.startswith(prefix) and not line.startswith(prefix * 3)
    )


def test_long_prefill_uses_bounded_materialized_logits() -> None:
    text = _text()
    added = _diff_lines(text, "+")
    removed = _diff_lines(text, "-")

    assert "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB" in added
    assert "rows_for_budget" in added
    assert "use_bounded_long_prefill" in added
    assert "_musa_custom_ops.sparse_indexer_topk(" in added
    assert "pages_per_block = block_size // 64" in added
    assert "paged_kv_cache = kv_cache.view(-1, 64" in added
    assert "needed_src_pages" in added
    assert "paged_block_table[:, : max(1, needed_src_pages)]" in added
    assert "if use_bounded_long_prefill and block_size != 64" in added
    assert "deepseek_v4_long_prefill_tp_partition_min_seq_len()" in added
    assert "tp_partition_rows = (" in added
    assert "tp_group.cpu_group" in added
    assert "torch.distributed.all_gather(" in added
    assert "gathered_parts" in added
    assert "Using bounded MUSA DeepSeek-V4 materialized indexer prefill" in added
    assert "or (is_deepseek_v4 and int(chunk.total_seq_lens) > 4096)" in removed


def test_long_prefill_routes_before_the_4k_native_kernel_gate() -> None:
    added = _diff_lines(_text(), "+")
    route = added.index("allow_deepseek_v4=True")
    provider_return = added.index("return True", route)

    assert route < provider_return
    assert "and int(chunk.total_seq_lens) > 4096" in added[:route]


def test_page64_view_is_initialized_inside_the_prefill_helper() -> None:
    added = _diff_lines(_text(), "+")
    initialized = added.index("paged_block_size = block_size")
    used_meta = added.index("context_chunk, paged_block_size", initialized)
    used_view = added.index("paged_kv_cache = kv_cache.view(-1, 64", initialized)
    assert initialized < used_view < used_meta


def test_long_prefill_patch_only_changes_sparse_indexer() -> None:
    assert _changed_files(_text()) == {
        "vllm/model_executor/layers/sparse_attn_indexer.py"
    }


def test_long_prefill_logits_budget_is_exported() -> None:
    policy = POLICY.read_text(encoding="utf-8")
    init = INIT.read_text(encoding="utf-8")
    assert "_DEEPSEEK_V4_LONG_PREFILL_LOGITS_MB = 512" in policy
    assert "def deepseek_v4_long_prefill_logits_budget_mb() -> int:" in policy
    assert "_DEEPSEEK_V4_LONG_PREFILL_TP_PARTITION_MIN_SEQ_LEN = 65536" in policy
    assert "def deepseek_v4_long_prefill_tp_partition_min_seq_len() -> int:" in policy
    assert "deepseek_v4_long_prefill_logits_budget_mb" in init
    assert "deepseek_v4_long_prefill_tp_partition_min_seq_len" in init
