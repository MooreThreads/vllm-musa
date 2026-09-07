from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0110-MUSA-adapt-DSV4-MTP-v028.patch"
)


def _text() -> str:
    return PATCH.read_text(encoding="utf-8")


def _changed_files(text: str) -> set[str]:
    return {
        line[len("+++ b/") :]
        for line in text.splitlines()
        if line.startswith("+++ b/")
    }


def test_mtp_target_uses_independent_overlap_events() -> None:
    text = _text()

    assert "self.mtp_multi_ln_events = [torch.cuda.Event() for _ in range(4)]" in text
    assert "is_mtp_multi = _musa_dsv4_mtp_multi_request_graph()" in text
    assert "ln_events = self.mtp_multi_ln_events if is_mtp_multi else self.ln_events" in text
    assert "aux_streams = self.aux_stream_list" in text
    assert "ln_events[0]" in text
    assert "ln_events[1:4]" in text


def test_mtp_event_isolation_patch_only_changes_attention() -> None:
    assert "vllm/models/deepseek_v4/attention.py" in _changed_files(_text())
