# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

from tools.sync_vllm_requirements import (
    GENERATED_HEADER,
    render_snapshot,
    sync_snapshot,
)


def test_sync_snapshot_writes_source_verbatim_after_generated_header(
    tmp_path: Path,
) -> None:
    source = tmp_path / "upstream" / "requirements.txt"
    output = tmp_path / "generated" / "requirements.txt"
    source.parent.mkdir()
    source.write_text("# upstream comment\nfoo>=1\nbar; python_version < '3.12'\n")

    assert sync_snapshot(source, output)
    assert output.read_text() == GENERATED_HEADER + source.read_text()
    assert sync_snapshot(source, output, check=True)


def test_sync_snapshot_check_detects_drift(tmp_path: Path, capsys) -> None:
    source = tmp_path / "upstream.txt"
    output = tmp_path / "generated.txt"
    source.write_text("foo\n")
    output.write_text("stale\n")

    assert not sync_snapshot(source, output, check=True)
    assert "generated requirements snapshot is stale" in capsys.readouterr().err


def test_sync_snapshot_reports_missing_upstream(tmp_path: Path) -> None:
    with pytest.raises(
        FileNotFoundError, match="upstream requirements file is missing"
    ):
        render_snapshot(tmp_path / "missing.txt")


def test_makefile_exposes_sync_and_check_targets() -> None:
    makefile = (Path(__file__).parents[1] / "Makefile.sync").read_text()
    assert "sync-vllm-requirements" in makefile
    assert "check-vllm-requirements" in makefile
