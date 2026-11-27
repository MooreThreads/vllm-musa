# SPDX-License-Identifier: Apache-2.0
"""manifest <-> filesystem completeness gate.

The MDM manifest (``vllm_musa/patches/manifest.py``) must stay in lock-step with
what is actually on disk: every build-applied diff in ``series/`` has exactly one
manifest entry and vice-versa; every object patch under ``patches/`` has a cat-6
entry; every cat-4a/5 module path and every cat-1/2/3/4b ``.patch`` exists. A CI
run of this test fails the moment the two drift apart (e.g. a patch added to
``series/`` without a manifest row, or a shadow module deleted but still listed).

Stdlib-only — loads manifest.py by file path (no vllm/torch import), so it runs
offline in normal CI.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PATCHES = ROOT / "vllm_musa" / "patches"


@pytest.fixture(scope="module")
def m():
    spec = importlib.util.spec_from_file_location(
        "musa_manifest_completeness", PATCHES / "manifest.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["musa_manifest_completeness"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_ids_unique(m):
    ids = [e.id for e in m.ENTRIES]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate manifest ids: {dupes}"


def test_every_entry_path_exists(m):
    missing = [e.path for e in m.ENTRIES if not (ROOT / e.path).exists()]
    assert not missing, f"manifest paths with no file on disk: {missing}"


def test_categories_and_phases_valid(m):
    for e in m.ENTRIES:
        assert e.category in m.VALID_CATEGORIES, e
        assert e.apply_phase in m.VALID_PHASES, e
    # build-applied categories must be pre-install/pre-compile; 4a never applied;
    # 5/6 runtime
    for e in m.ENTRIES:
        if e.category in m.BUILD_APPLIED_CATEGORIES:
            assert e.apply_phase in ("pre-install", "pre-compile"), e
        elif e.category == "4a":
            assert e.apply_phase == "none", e
        else:  # 5, 6
            assert e.apply_phase == "runtime", e


def test_series_bijection_with_manifest(m):
    """Every series/*.patch ⇔ exactly one cat-1/2/3/4b manifest entry."""
    series = PATCHES / "series"
    on_disk = {
        p.relative_to(series).as_posix()
        for p in [*series.glob("*.patch"), *(series / "upstream").glob("*.patch")]
    }
    in_manifest = {
        e.path.split("/series/", 1)[1]
        for e in m.ENTRIES
        if e.category in m.BUILD_APPLIED_CATEGORIES
    }
    assert on_disk == in_manifest, (
        f"series⇔manifest mismatch: only on disk={on_disk - in_manifest}; "
        f"only in manifest={in_manifest - on_disk}"
    )


def test_series_readme_count_matches_disk() -> None:
    readme = (PATCHES / "series" / "README.md").read_text(encoding="utf-8")
    match = re.search(r"Currently \*\*(\d+) patches\*\*", readme)
    assert match is not None, "series README must state the current patch count"
    on_disk = sum(1 for _ in (PATCHES / "series").glob("*.patch"))
    assert int(match.group(1)) == on_disk
    upstream = re.search(r"\*\*(\d+) upstream entries\*\*", readme)
    assert upstream is not None, "series README must state the upstream entry count"
    assert int(upstream.group(1)) == sum(
        1 for _ in (PATCHES / "series" / "upstream").glob("*.patch")
    )


def test_object_patches_have_cat6_entries(m):
    """Every patches/*.patch.py object patch has a cat-6 manifest entry."""
    on_disk = {p.name for p in PATCHES.glob("*.patch.py")}
    in_manifest = {Path(e.path).name for e in m.ENTRIES if e.category == "6"}
    assert on_disk == in_manifest, (
        f"object⇔cat-6 mismatch: only on disk={on_disk - in_manifest}; "
        f"only in manifest={in_manifest - on_disk}"
    )


def test_cat5_entries_have_seam_or_are_new(m):
    """Every cat-5 entry either declares an upstream seam (verify can probe it) or
    is explicitly a new MUSA module (upstream_path None)."""
    for e in m.ENTRIES:
        if e.category == "5":
            assert e.upstream_path is None or e.upstream_path.startswith("vllm/"), e


def test_after_refs_resolve(m):
    """Every DivSpec.after reference names a real entry id."""
    ids = {e.id for e in m.ENTRIES}
    for e in m.ENTRIES:
        for dep in e.after:
            assert dep in ids, f"{e.id}.after references unknown id {dep!r}"
