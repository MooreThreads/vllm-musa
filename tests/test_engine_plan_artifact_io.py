# SPDX-License-Identifier: Apache-2.0

import json
import os
import stat
from pathlib import Path

import pytest

from vllm_musa.engine_plan.artifact_io import (
    ArtifactFileError,
    load_json_object_file,
    write_json_object_file,
)
from vllm_musa.engine_plan.core import EnginePlanError, load_plan


def test_load_json_object_file_rejects_symlink(tmp_path: Path):
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "plan.json"
    link.symlink_to(target)

    with pytest.raises(ArtifactFileError, match="symbolic link"):
        load_json_object_file(link)


def test_runtime_plan_loader_uses_bounded_regular_file_reader(tmp_path: Path):
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "plan.json"
    link.symlink_to(target)

    with pytest.raises(EnginePlanError, match="symbolic link"):
        load_plan(link)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFO support")
def test_load_json_object_file_rejects_fifo_without_blocking(tmp_path: Path):
    fifo = tmp_path / "plan.fifo"
    os.mkfifo(fifo)

    with pytest.raises(ArtifactFileError, match="regular file"):
        load_json_object_file(fifo)


def test_load_json_object_file_rejects_oversized_input(tmp_path: Path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"padding": "x" * 128}), encoding="utf-8")

    with pytest.raises(ArtifactFileError, match="exceeds.*bytes"):
        load_json_object_file(path, max_bytes=64)


def test_load_json_object_file_requires_object_root(tmp_path: Path):
    path = tmp_path / "plan.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ArtifactFileError, match="root must be a JSON object"):
        load_json_object_file(path)


@pytest.mark.parametrize(
    "payload",
    (
        '{"plan_id": "first", "plan_id": "second"}',
        '{"variant": {"fingerprint": "first", "fingerprint": "second"}}',
    ),
)
def test_load_json_object_file_rejects_duplicate_keys(
    tmp_path: Path,
    payload: str,
):
    path = tmp_path / "plan.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ArtifactFileError, match="duplicate JSON object key"):
        load_json_object_file(path)


def test_write_json_object_file_is_atomic_and_preserves_mode(tmp_path: Path):
    path = tmp_path / "plan.json"
    path.write_text('{"old": true}\n', encoding="utf-8")
    path.chmod(0o640)

    write_json_object_file(path, {"new": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"new": True}
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert not list(tmp_path.glob(f".{path.name}.tmp.*"))


def test_write_json_object_file_rejects_symlink_destination(tmp_path: Path):
    target = tmp_path / "target.json"
    target.write_text('{"safe": true}\n', encoding="utf-8")
    link = tmp_path / "plan.json"
    link.symlink_to(target)

    with pytest.raises(ArtifactFileError, match="symbolic link"):
        write_json_object_file(link, {"unsafe": True})

    assert json.loads(target.read_text(encoding="utf-8")) == {"safe": True}
