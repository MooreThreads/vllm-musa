# SPDX-License-Identifier: Apache-2.0
"""the musa_sync driver (report / verify plumbing).

Stdlib-only (musa_sync loads manifest.py + build_apply.py by file path) → runs
locally. Network-dependent paths are exercised on the MUSA container against a
real vLLM checkout; here we cover `report`, `regen`, the probe helper, and the
verify row-builder against synthetic checkouts.
"""

import importlib.util
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(repo: Path, content: str = "alpha\n") -> str:
    repo.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", str(repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(repo, "config", "user.email", "musa-sync@example.invalid")
    _git(repo, "config", "user.name", "musa-sync test")
    (repo / "value.txt").write_text(content)
    _git(repo, "add", "value.txt")
    _git(repo, "commit", "--quiet", "-m", "fixture")
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


@pytest.fixture(scope="module")
def ms():
    spec = importlib.util.spec_from_file_location(
        "musa_sync_under_test", ROOT / "tools" / "musa_sync.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["musa_sync_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_report(ms, capsys):
    rc = ms.main(["report"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "174 divergences" in out
    assert "'1': 112" in out and "'2': 23" in out and "'3': 1" in out
    assert "'4a': 2" in out and "'5': 28" in out and "'6': 8" in out


def test_report_doc(ms, capsys):
    rc = ms.main(["report", "--doc"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.lstrip().startswith("| id | cat |")
    assert "vllm__v1__spec_decode__eagle" in out


def test_probe_upstream(ms, tmp_path):
    (tmp_path / "vllm").mkdir()
    (tmp_path / "vllm" / "x.py").write_text("")
    assert ms._probe_upstream(tmp_path, "vllm/x.py")
    assert not ms._probe_upstream(tmp_path, "vllm/nope.py")
    assert ms._probe_upstream(tmp_path, None)  # no declared target → vacuously present


def test_read_pin(ms):
    assert ms.read_pin("VLLM_TAG") == "v0.28.0"
    assert ms.read_pin("DOES_NOT_EXIST", "fallback") == "fallback"


def test_default_target_prefers_exact_commit(ms, monkeypatch):
    pins = {"VLLM_COMMIT": "0123456789abcdef", "VLLM_TAG": "v0.24.0"}
    monkeypatch.setattr(
        ms, "read_pin", lambda key, default=None: pins.get(key, default)
    )
    assert ms._default_target() == "0123456789abcdef"

    pins.pop("VLLM_COMMIT")
    assert ms._default_target() == "v0.24.0"


def test_series_readme_count_matches_directory():
    series_dir = ROOT / "vllm_musa" / "patches" / "series"
    patch_count = len(list(series_dir.glob("*.patch")))
    readme = (series_dir / "README.md").read_text()
    assert f"Currently **{patch_count} patches**" in readme


def test_series_uses_documented_prefixes_and_canonical_metadata():
    series_dir = ROOT / "vllm_musa" / "patches" / "series"
    patches = sorted(series_dir.glob("*.patch"))
    prefixes = [p.name.split("-", 1)[0] for p in patches]
    assert all(len(prefix) == 4 and prefix.isdigit() for prefix in prefixes)
    assert [int(prefix) for prefix in prefixes] == sorted(
        int(prefix) for prefix in prefixes
    )
    duplicate_prefixes = {
        prefix for prefix, count in Counter(prefixes).items() if count > 1
    }
    assert duplicate_prefixes == set()
    headers = [p.read_bytes().splitlines()[:2] for p in patches]
    zero_commit_header = (
        b"From 0000000000000000000000000000000000000000 Mon Sep 17 00:00:00 2001"
    )
    assert all(lines[0] == zero_commit_header for lines in headers)
    assert {lines[1] for lines in headers} == {b"From: musa <musa@local>"}


def test_normalize_patch_author_preserves_non_utf8_bytes(ms, tmp_path):
    patch = tmp_path / "0001-non-utf8.patch"
    patch.write_bytes(
        b"From 0000000000000000000000000000000000000000 Mon Sep 17 00:00:00 2001\n"
        b"From: contributor <contributor@example.com>\n"
        b"Subject: [PATCH] preserve bytes\n\n"
        b"non-utf8 payload: \xff\xfe\n"
    )

    ms._normalize_patch_author(patch)

    assert patch.read_bytes() == (
        b"From 0000000000000000000000000000000000000000 Mon Sep 17 00:00:00 2001\n"
        b"From: musa <musa@local>\n"
        b"Subject: [PATCH] preserve bytes\n\n"
        b"non-utf8 payload: \xff\xfe\n"
    )


def test_regen_requires_pinned_target(ms, monkeypatch, capsys):
    monkeypatch.setattr(ms, "_default_target", lambda: None)

    assert ms.main(["regen"]) == 1
    assert "VLLM_COMMIT or VLLM_TAG is required" in capsys.readouterr().out


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_regen_rejects_noncanonical_author(ms, tmp_path, monkeypatch, capsys):
    workdir = tmp_path / "vllm"
    base = _init_repo(workdir)
    (workdir / "value.txt").write_text("beta\n")
    _git(workdir, "commit", "--all", "--quiet", "-m", "change")

    monkeypatch.setattr(ms, "WORKDIR", workdir)
    monkeypatch.setattr(ms, "SERIES_DIR", tmp_path / "series")
    monkeypatch.setattr(ms, "_default_target", lambda: base)
    monkeypatch.setattr(ms, "_normalize_patch_author", lambda _path: None)

    assert ms.main(["regen"]) == 1
    assert "non-canonical patch author headers" in capsys.readouterr().out


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_ensure_clone_accepts_exact_commit_sha(ms, tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    first = _init_repo(origin)
    (origin / "value.txt").write_text("second\n")
    _git(origin, "commit", "--all", "--quiet", "-m", "second")
    latest = _git(origin, "rev-parse", "HEAD").stdout.strip()
    assert first != latest

    monkeypatch.setattr(ms, "VLLM_URL", str(origin))
    clone, temporary = ms._ensure_clone(first, None)
    try:
        assert temporary
        assert _git(clone, "rev-parse", "HEAD").stdout.strip() == first
        assert (clone / "value.txt").read_text() == "alpha\n"
        assert _git(origin, "rev-parse", "HEAD").stdout.strip() == latest
    finally:
        shutil.rmtree(clone.parent, ignore_errors=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_checkout_fetches_exact_commit_sha(ms, tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    first = _init_repo(origin)
    (origin / "value.txt").write_text("second\n")
    _git(origin, "commit", "--all", "--quiet", "-m", "second")
    latest = _git(origin, "rev-parse", "HEAD").stdout.strip()

    workdir = tmp_path / "checkout"
    monkeypatch.setattr(ms, "VLLM_URL", str(origin))
    monkeypatch.setattr(ms, "WORKDIR", workdir)
    assert ms._checkout(first) == 0
    assert _git(workdir, "rev-parse", "HEAD").stdout.strip() == first
    assert ms._checkout(latest) == 0
    assert _git(workdir, "rev-parse", "HEAD").stdout.strip() == latest


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_regen_replaces_series_and_prunes_stale_files(
    ms, tmp_path, monkeypatch, capsys
):
    workdir = tmp_path / "vllm"
    base = _init_repo(workdir)
    _git(workdir, "config", "user.name", "Xiaodong Ye")
    _git(workdir, "config", "user.email", "xiaodong.ye@mthreads.com")
    (workdir / "value.txt").write_text("beta\n")
    _git(workdir, "commit", "--all", "--quiet", "-m", "first change")
    _git(workdir, "config", "user.name", "musa-sync test")
    _git(workdir, "config", "user.email", "musa-sync@example.invalid")
    (workdir / "extra.txt").write_text("extra\n")
    _git(workdir, "add", "extra.txt")
    _git(workdir, "commit", "--quiet", "-m", "second change")

    series_dir = tmp_path / "series"
    series_dir.mkdir()
    (series_dir / "README.md").write_text("keep me\n")
    (series_dir / "0099-stale.patch").write_text("stale\n")
    monkeypatch.setattr(ms, "WORKDIR", workdir)
    monkeypatch.setattr(ms, "SERIES_DIR", series_dir)
    monkeypatch.setattr(ms, "_default_target", lambda: base)

    assert ms.main(["regen"]) == 0
    out = capsys.readouterr().out
    patches = sorted(series_dir.glob("*.patch"))
    assert [p.name for p in patches] == [
        "0001-first-change.patch",
        "0002-second-change.patch",
    ]
    headers = [p.read_bytes().splitlines()[:2] for p in patches]
    assert all(
        lines[0].startswith(b"From 0000000000000000000000000000000000000000 ")
        for lines in headers
    )
    assert all(lines[1] == b"From: musa <musa@local>" for lines in headers)
    assert (series_dir / "README.md").read_text() == "keep me\n"
    assert "pruned 1 stale files" in out


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_verify_rows_applies_dependent_patches_cumulatively_without_mutating_repo(
    ms, tmp_path, monkeypatch
):
    repo = tmp_path / "upstream"
    _init_repo(repo)
    project = tmp_path / "project"
    patch_dir = project / "patches"
    patch_dir.mkdir(parents=True)
    first = patch_dir / "0001-add-beta.patch"
    first.write_text(
        """diff --git a/value.txt b/value.txt
--- a/value.txt
+++ b/value.txt
@@ -1 +1,2 @@
 alpha
+beta
"""
    )
    second = patch_dir / "0002-rewrite-beta.patch"
    second.write_text(
        """diff --git a/value.txt b/value.txt
--- a/value.txt
+++ b/value.txt
@@ -1,2 +1,2 @@
 alpha
-beta
+gamma
"""
    )
    conflict = patch_dir / "0003-conflict.patch"
    conflict.write_text(
        """diff --git a/value.txt b/value.txt
--- a/value.txt
+++ b/value.txt
@@ -1 +1 @@
-not-present
+still-not-present
"""
    )

    entries = [
        ms.manifest.DivSpec(
            id=patch.stem,
            category="1",
            path=str(patch.relative_to(project)),
            upstream_path="value.txt",
        )
        for patch in (first, second, conflict)
    ]
    monkeypatch.setattr(ms, "ROOT", project)
    monkeypatch.setattr(ms.manifest, "ENTRIES", entries)

    # The second patch cannot apply to pristine upstream; it only becomes valid
    # after the first patch advances the disposable verification checkout.
    assert ms.build_apply.apply_patch(repo, second, check_only=True) == "conflict"
    before_status = _git(repo, "status", "--porcelain=v1").stdout
    rows = ms._verify_rows(repo)

    assert [(row[0], row[2]) for row in rows] == [
        ("0001-add-beta", "clean"),
        ("0002-rewrite-beta", "clean"),
        ("0003-conflict", "conflict"),
    ]
    assert (repo / "value.txt").read_text() == "alpha\n"
    assert _git(repo, "status", "--porcelain=v1").stdout == before_status


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_verify_rows_synthetic(ms, tmp_path):
    # A synthetic checkout that has the cat-6 target files → those probe "present".
    clone = tmp_path / "vllm"
    clone.mkdir()
    for e in ms.manifest.object_entries():
        if e.upstream_path is None:  # torch.* cat-6 not probeable from the vLLM clone
            continue
        p = clone / e.upstream_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
    rows = ms._verify_rows(clone)
    assert len(rows) == len(ms.manifest.ENTRIES)
    cat6 = {did: status for did, cat, status, _ in rows if cat == "6"}
    assert cat6 and all(s == "present" for s in cat6.values()), cat6
    # a cat-6 entry whose target is absent must report missing-target
    missing = clone / "vllm" / "v1" / "spec_decode" / "eagle.py"
    missing.unlink()
    rows2 = ms._verify_rows(clone)
    statuses = {did: s for did, cat, s, _ in rows2 if cat == "6"}
    assert statuses["vllm__v1__spec_decode__eagle"] == "missing-target", statuses
