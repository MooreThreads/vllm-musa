from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0173-MUSA-skip-Prometheus-observe-for-connectors-without-.patch"
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


def test_patch_only_touches_multi_connector() -> None:
    assert _changed_files(_text()) == {
        "vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py"
    }


def test_observe_skips_connectors_without_prom_metrics() -> None:
    text = _text()
    added = _diff_lines(text, "+")
    removed = _diff_lines(text, "-")

    assert "assert connector_id in self._prom_metrics" in removed
    assert "prom_metrics = self._prom_metrics.get(connector_id)" in added
    assert "if prom_metrics is not None:" in added
    assert 'prom_metrics.observe(stats_data["data"], engine_idx)' in added
