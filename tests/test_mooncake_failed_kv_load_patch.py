from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0182-Bugfix-Mooncake-Report-failed-remote-KV-loads-to-the.patch"
)
UPSTREAM_COMMIT = "2c7ee87223f0c94f614a0b8600f6f193e6e4388f"
SCHEDULER_PATCH = PATCH.with_name(
    "0181-Core-Fix-ValueError-on-KV-load-failure-with-a-hybrid.patch"
)
SCHEDULER_UPSTREAM_COMMIT = "c351fd3c649569d8a585eb22cfaf2ee728e5fc22"
TEST_SPEC_PATCH = PATCH.with_name(
    "0183-MUSA-run-the-DeepSeek-V4-KV-load-failure-test-on-the.patch"
)


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


def test_patch_only_touches_mooncake_connector() -> None:
    assert _changed_files(_text()) == {
        "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"
    }


def test_patch_is_the_upstream_commit() -> None:
    assert f"(cherry picked from commit {UPSTREAM_COMMIT})" in _text()


def test_failed_pulls_are_reported_to_the_scheduler() -> None:
    text = _text()
    added = _diff_lines(text, "+")
    removed = _diff_lines(text, "-")

    # A failed pull finishes receiving with its blocks marked as load errors,
    # so the scheduler fails or recomputes the request and frees its blocks.
    assert "def _handle_failed_recv(" in added
    assert "self._invalid_block_ids.put(invalid)" in added
    assert "self.finished_recving_reqs.add(pull_meta.d_req_id)" in added
    assert "def get_block_ids_with_load_errors(self) -> set[int]:" in added
    assert "return self.connector_worker.get_block_ids_with_load_errors()" in added

    # Every receiver failure path reports instead of only logging.
    assert 'self._handle_failed_recv(pull_metas, req_ids, f"transfer failed: {e}")' in added
    assert 'response.err_msg or "transfer error"' in added
    assert 'pull_metas, response.err_reqs, response.err_msg or "unknown error"' in added
    assert "remote engine_id {remote_engine_id} not found from bootstrap" in added
    assert 'logger.error("MooncakeXferMetadata transfer failed for %s: %s", req_ids, e)' in removed


def test_success_after_a_failure_is_not_counted() -> None:
    added = _diff_lines(_text(), "+")

    assert "failed: bool = False" in added
    assert "if pull_meta.failed:" in added


def test_scheduler_handles_load_failures_on_a_hybrid_kv_cache() -> None:
    # The Mooncake report reaches the scheduler's invalid-block path, which must
    # accept one block-id list per KV cache group (DeepSeek-V4 has several).
    text = SCHEDULER_PATCH.read_text(encoding="utf-8")
    added = _diff_lines(text, "+")
    removed = _diff_lines(text, "-")

    assert f"(cherry picked from commit {SCHEDULER_UPSTREAM_COMMIT})" in text
    assert "vllm/v1/core/sched/scheduler.py" in _changed_files(text)
    assert "(req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)" in removed
    assert "req_block_ids_per_group = self.kv_cache_manager.get_block_ids(req_id)" in added
    assert "if len(req_block_ids_per_group) > 1:" in added
    assert "null_block_id = self.kv_cache_manager.block_pool.null_block.block_id" in added


def test_backported_deepseek_v4_test_uses_the_pinned_spec_field() -> None:
    text = TEST_SPEC_PATCH.read_text(encoding="utf-8")

    assert _changed_files(text) == {
        "tests/v1/kv_connector/unit/test_kv_load_failure_recovery.py"
    }
    assert "tokens_per_state=4," in _diff_lines(text, "-")
    assert "compress_ratio=4," in _diff_lines(text, "+")
