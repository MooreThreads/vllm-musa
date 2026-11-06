# SPDX-License-Identifier: Apache-2.0
"""the MDM manifest (manifest.py) + build_apply apply_phase / --check-only.

Stdlib-only — manifest.py and build_apply.py are loaded by file path (the same way
setup.py loads them at build), so this runs locally with no MUSA hardware and no
vllm/torch import.
"""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PATCHES = ROOT / "vllm_musa" / "patches"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register before exec (canonical safe by-path load)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def manifest():
    return _load("musa_manifest_under_test", PATCHES / "manifest.py")


@pytest.fixture(scope="module")
def build_apply():
    return _load("musa_build_apply_under_test_0312", PATCHES / "build_apply.py")


def _git(repo: Path, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _make_patch(tmp_path: Path, sub: str):
    """A throwaway git repo + a 1-line patch reset off HEAD; returns (repo, patch)."""
    repo = tmp_path / sub
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "f.txt").write_text("a\nb\n")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-qm", "base")
    base = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (repo / "f.txt").write_text("a\nB\n")
    _git(repo, "commit", "-aqm", "ch")
    patch = subprocess.run(
        ["git", "-C", str(repo), "format-patch", "-1", "--stdout"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _git(repo, "reset", "-q", "--hard", base)
    pf = tmp_path / f"{sub}.patch"
    pf.write_text(patch)
    return repo, pf


# ---------------- manifest.py ----------------


def test_entries_nonempty_and_valid(manifest):
    assert manifest.ENTRIES
    for e in manifest.ENTRIES:
        assert e.category in manifest.VALID_CATEGORIES, e
        assert e.apply_phase in manifest.VALID_PHASES, e
        assert e.id and e.path


def test_series_classified_by_target(manifest):
    root = PATCHES / "series"
    series = sorted(root.glob("*.patch")) + sorted((root / "upstream").glob("*.patch"))
    build_applied = [e for e in manifest.ENTRIES if e.category in ("1", "2", "3", "4b")]
    # every series patch maps to exactly one build-applied entry
    assert len(build_applied) == len(series), (len(build_applied), len(series))
    assert {e.path.split("/series/", 1)[1] for e in build_applied} == {
        p.relative_to(root).as_posix() for p in series
    }
    # csrc patches → cat-2/3 (pre-compile); python patches → cat-1 (pre-install)
    for e in build_applied:
        if e.upstream_path and e.upstream_path.startswith("csrc/"):
            assert e.category in ("2", "3"), e
            assert e.apply_phase == "pre-compile", e
        else:
            assert e.category == "1", e
            assert e.apply_phase == "pre-install", e


def test_cat6_object_patches(manifest):
    cat6 = manifest.object_entries()
    assert {e.id for e in cat6} == {
        "vllm__distributed__parallel_state",
        "vllm__v1__spec_decode__eagle",
        # torch/vLLM config compat shims migrated to cat-6
        "torch___functorch__config",
        "torch___inductor__config",
        "vllm__compilation__backends",
        "vllm__compilation__compiler_interface",
        "vllm___custom_ops",
        "torch___inductor__aot_cache_safelist",
    }
    for e in cat6:
        assert e.apply_phase == "runtime"
        assert (ROOT / e.path).is_file(), e.path
    eagle = next(e for e in cat6 if e.id == "vllm__v1__spec_decode__eagle")
    assert "vllm__distributed__parallel_state" in eagle.after


def test_entries_for_phase(manifest):
    runtime = manifest.entries_for_phase("runtime")
    # runtime = cat-5 tracked modules + cat-6 object patches
    assert runtime and all(e.category in ("5", "6") for e in runtime)
    assert len(runtime) == len(
        [e for e in manifest.ENTRIES if e.apply_phase == "runtime"]
    )
    assert all(e.category == "1" for e in manifest.entries_for_phase("pre-install"))
    # cat-4a drift tripwires are never build-applied (apply_phase = "none")
    assert all(e.category == "4a" for e in manifest.entries_for_phase("none"))


def test_series_apply_order_resolves(manifest):
    order = manifest.series_apply_order()
    assert order, "no build-applied entries"
    assert all(p.is_file() for p in order), "a series_apply_order path is missing"
    # only cat-1 is seeded so far → order == the series patches
    assert len(order) == len(
        [e for e in manifest.ENTRIES if e.category in ("1", "2", "3", "4b")]
    )
    # phase-filtered: pre-compile order == the cat-2/3 csrc patches
    assert len(manifest.series_apply_order(phase="pre-compile")) == len(
        [e for e in manifest.ENTRIES if e.category in ("2", "3")]
    )


def test_divspec_validates(manifest):
    with pytest.raises(AssertionError):
        manifest.DivSpec(id="x", category="9", path="p")
    with pytest.raises(AssertionError):
        manifest.DivSpec(id="x", category="1", path="p", apply_phase="bogus")


# ---------------- build_apply apply_phase / --check-only ----------------


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_build_apply_check_only_does_not_mutate(build_apply, tmp_path):
    repo, pf = _make_patch(tmp_path, "r")
    # check_only reports would-apply and does NOT touch the tree
    assert build_apply.apply_patch(repo, pf, check_only=True) == "would-apply"
    assert (repo / "f.txt").read_text() == "a\nb\n", "check_only mutated the tree"
    # real apply, then check_only sees already-applied
    assert build_apply.apply_patch(repo, pf) == "applied"
    assert build_apply.apply_patch(repo, pf, check_only=True) == "already-applied"


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_apply_patch_series_order_param_dry_run(build_apply, tmp_path):
    repo, pf = _make_patch(tmp_path, "r2")
    # order= overrides series_dir discovery; check_only dry-runs the explicit list
    res = build_apply.apply_patch_series(repo, order=[pf], check_only=True)
    assert res == [("r2.patch", "would-apply")], res
    assert (repo / "f.txt").read_text() == "a\nb\n", "dry-run mutated the tree"
