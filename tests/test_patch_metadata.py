# SPDX-License-Identifier: Apache-2.0
"""Contracts for advisory patch-series metadata."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "vllm_musa" / "patches" / "series"
METADATA = ROOT / "vllm_musa" / "patches" / "metadata"
spec = importlib.util.spec_from_file_location("patch_metadata", ROOT / "tools" / "patch_metadata.py")
patch_metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch_metadata)


def _fixture(tmp_path):
    series = tmp_path / "series"
    metadata = tmp_path / "metadata"
    series.mkdir()
    metadata.mkdir()
    for number in (1, 2):
        (series / f"{number:04d}-example.patch").write_text("mailbox\n")
    (series / "README.md").write_text(
        "| v0.28 | Disposition / v0.32 source evidence | Validation still needed |\n"
        "| --- | --- | --- |\n"
        "| `0065` | Superseded by `0001`. | Model smoke. |\n"
    )
    lineage = {"schema_version": 1, "omissions": [{
        "legacy_id": "0065", "disposition": "superseded", "owners": ["0001"],
        "rationale": "Direct compiler fix.", "validation_required": "Model smoke.",
    }]}
    groups = {"schema_version": 1, "groups": [{
        "id": "compiler", "purpose": "Compile MUSA kernels", "patches": ["0001", "0002"],
        "depends_on": [], "required_checks": ["series", "replay", "round_trip", "source_contract", "musa_runtime"],
    }]}
    return series, metadata, lineage, groups


def _validate(series, metadata, lineage, groups):
    (metadata / "legacy-lineage.json").write_text(json.dumps(lineage))
    (metadata / "patch-groups.json").write_text(json.dumps(groups))
    return patch_metadata.validate(series, metadata)


def test_real_series_has_exact_coverage():
    result = patch_metadata.validate(SERIES, METADATA)
    assert result.errors == []
    assert result.patch_count == len(list(SERIES.glob("*.patch")))
    assert result.omission_count == 24
    assert result.group_count > 1


@pytest.mark.parametrize("mutation,expected", [
    (lambda l, g: l["omissions"].append(copy.deepcopy(l["omissions"][0])), "duplicate legacy"),
    (lambda l, g: l["omissions"].clear(), "missing legacy"),
    (lambda l, g: l["omissions"][0]["owners"].append("9999"), "unknown owner"),
    (lambda l, g: g["groups"][0]["patches"].append("0001"), "duplicate patch"),
    (lambda l, g: g["groups"][0]["patches"].pop(), "missing patch"),
    (lambda l, g: g["groups"][0]["patches"].append("9999"), "unknown patch"),
    (lambda l, g: g["groups"][0]["required_checks"].append("not-a-check"), "unknown check"),
    (lambda l, g: g["groups"][0]["depends_on"].append("missing"), "unknown dependency"),
    (lambda l, g: l["omissions"][0].update(disposition="passed"), "unknown disposition"),
])
def test_invalid_metadata_fails(tmp_path, mutation, expected):
    series, metadata, lineage, groups = _fixture(tmp_path)
    mutation(lineage, groups)
    assert expected in "\n".join(_validate(series, metadata, lineage, groups).errors)


def test_readme_omission_must_be_documented(tmp_path):
    series, metadata, lineage, groups = _fixture(tmp_path)
    (series / "README.md").write_text(
        (series / "README.md").read_text() +
        "| `0069` | Unresolved offset. | MUSA sparse decode. |\n"
    )
    assert "missing legacy" in "\n".join(_validate(series, metadata, lineage, groups).errors)


def test_group_membership_is_ordered_and_unique(tmp_path):
    series, metadata, lineage, groups = _fixture(tmp_path)
    groups["groups"].append({**groups["groups"][0], "id": "second", "patches": ["0002"]})
    assert "duplicate patch" in "\n".join(_validate(series, metadata, lineage, groups).errors)
    groups["groups"].pop()
    groups["groups"][0]["patches"] = ["0002", "0001"]
    assert "order" in "\n".join(_validate(series, metadata, lineage, groups).errors)


def test_duplicate_and_cyclic_dependencies_fail(tmp_path):
    series, metadata, lineage, groups = _fixture(tmp_path)
    groups["groups"][0]["patches"] = ["0001"]
    groups["groups"].append({
        **groups["groups"][0], "id": "second", "patches": ["0002"],
        "depends_on": ["compiler", "compiler"],
    })
    assert "duplicate dependency" in "\n".join(_validate(series, metadata, lineage, groups).errors)
    groups["groups"][1]["depends_on"] = ["compiler"]
    groups["groups"][0]["depends_on"] = ["second"]
    assert "dependency cycle" in "\n".join(_validate(series, metadata, lineage, groups).errors)


def test_missing_and_malformed_files_fail(tmp_path):
    series, metadata, lineage, groups = _fixture(tmp_path)
    result = patch_metadata.validate(series, metadata)
    assert "missing metadata" in "\n".join(result.errors)
    (metadata / "legacy-lineage.json").write_text("{")
    (metadata / "patch-groups.json").write_text(json.dumps(groups))
    assert "invalid JSON" in "\n".join(patch_metadata.validate(series, metadata).errors)
