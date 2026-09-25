# SPDX-License-Identifier: Apache-2.0
"""the musa_sync driver (report / verify plumbing).

Stdlib-only (musa_sync loads manifest.py + build_apply.py by file path) → runs
locally. Network-dependent paths are exercised on the MUSA container against a
real vLLM checkout; here we cover `report`, `regen`, the probe helper, and the
verify row-builder against synthetic checkouts.
"""

import importlib.util
import os
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
    # The census follows the manifest, which grows with the base: assert the
    # shape against the manifest rather than today's numbers.
    assert f"{len(ms.manifest.ENTRIES)} divergences" in out
    by_cat = Counter(e.category for e in ms.manifest.ENTRIES)
    for category, n in sorted(by_cat.items()):
        assert f"'{category}': {n}" in out


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
    first.write_text("""diff --git a/value.txt b/value.txt
--- a/value.txt
+++ b/value.txt
@@ -1 +1,2 @@
 alpha
+beta
""")
    second = patch_dir / "0002-rewrite-beta.patch"
    second.write_text("""diff --git a/value.txt b/value.txt
--- a/value.txt
+++ b/value.txt
@@ -1,2 +1,2 @@
 alpha
-beta
+gamma
""")
    conflict = patch_dir / "0003-conflict.patch"
    conflict.write_text("""diff --git a/value.txt b/value.txt
--- a/value.txt
+++ b/value.txt
@@ -1 +1 @@
-not-present
+still-not-present
""")

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


def _series_fixture(tmp_path: Path, *names: str) -> Path:
    """A series dir holding copies of one real entry under the given names."""
    series = tmp_path / "series"
    series.mkdir()
    donor = sorted((ROOT / "vllm_musa" / "patches" / "series").glob("*.patch"))[0]
    for name in names:
        shutil.copyfile(donor, series / name)
    return series


def _donor_patch() -> Path:
    """A real series entry, used as the template for synthetic fixtures."""
    return sorted((ROOT / "vllm_musa" / "patches" / "series").glob("*.patch"))[0]


def _write_series(series: Path, name: str, text: bytes) -> Path:
    series.mkdir(parents=True, exist_ok=True)
    path = series / name
    path.write_bytes(text)
    return path


def _mailbox(subject: str, diff: str) -> bytes:
    """A canonical `regen` mailbox around ``diff`` (no commit-message body)."""
    return (
        b"From 0000000000000000000000000000000000000000 "
        b"Mon Sep 17 00:00:00 2001\n"
        b"From: musa <musa@local>\n"
        b"Date: Thu, 1 Jan 2026 00:00:00 +0000\n"
        b"Subject: [PATCH] " + subject.encode() + b"\n\n" + diff.encode()
    )


def _canonical_entry(slug: str, marker: str = "") -> bytes:
    """A canonical entry whose subject slug matches ``NNNN-<slug>.patch``.

    Every check in the gate passes on it, so a test can mutate exactly one
    property and attribute the resulting row to that mutation.
    """
    return _mailbox(
        slug,
        "diff --git a/value.txt b/value.txt\n"
        "index 1111111111..2222222222 100644\n"
        "--- a/value.txt\n+++ b/value.txt\n"
        "@@ -1 +1,2 @@\n alpha\n"
        f"+{slug}{marker}\n",
    )


def _canonical_series(tmp_path: Path, *slugs: str) -> Path:
    """A series of canonical entries ``0001-<slug>.patch`` with distinct bodies."""
    series = tmp_path / "series"
    for i, slug in enumerate(slugs, start=1):
        _write_series(series, f"{i:04d}-{slug}.patch", _canonical_entry(slug))
    return series


def _formatted_entry(
    tmp_path: Path, message: str = "a", content: bytes = b"alpha\nbeta\n"
):
    """``(repo, entry name, entry bytes)`` built by real ``git format-patch``.

    The entry is byte-for-byte what ``regen`` writes for that commit, so tests
    can assert both halves of the fixed point against something git itself
    produced. ``repo`` is left at the parent commit, entry unapplied.
    """
    repo = tmp_path / "upstream"
    base = _init_repo(repo)
    _git(repo, "config", "user.email", "musa@local")
    _git(repo, "config", "user.name", "musa")
    (repo / "value.txt").write_bytes(content)
    _git(repo, "commit", "--all", "--quiet", "-m", message)
    staged = tmp_path / "generated"
    subprocess.run(
        [
            "git", "-C", str(repo), "format-patch", "--no-signature",
            "--no-numbered", "--zero-commit", "-o", str(staged), base,
        ],
        check=True,
        capture_output=True,
    )
    entry = sorted(staged.glob("*.patch"))[0]
    _git(repo, "reset", "--hard", "--quiet", base)
    return repo, entry.name, entry.read_bytes()


def _cli_tree(tmp_path: Path, files: dict) -> Path:
    """A minimal repo tree whose ``tools/musa_sync.py`` is the module under test.

    ``SERIES_DIR`` is derived from the module's own location and ``manifest.py``
    reads every entry at import time, so the CLI-level findings (a chmod-000,
    symlink or directory entry killing the process before any verdict) can only
    be reproduced through a real import: an in-process load happens *before* the
    hostile file exists.
    """
    (tmp_path / "tools").mkdir(parents=True)
    (tmp_path / "vllm_musa" / "patches" / "series").mkdir(parents=True)
    shutil.copyfile(ROOT / "tools" / "musa_sync.py", tmp_path / "tools" / "musa_sync.py")
    for rel in ("vllm_musa/patches/manifest.py", "vllm_musa/patches/build_apply.py"):
        shutil.copyfile(ROOT / rel, tmp_path / rel)
    for name, text in files.items():
        (tmp_path / "vllm_musa" / "patches" / "series" / name).write_bytes(text)
    return tmp_path


def _run_cli(tree: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(tree / "tools" / "musa_sync.py"), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _blob_id(repo: Path, content: bytes) -> str:
    """The blob id git records for ``content`` (computed, never written)."""
    return (
        subprocess.run(
            ["git", "-C", str(repo), "hash-object", "--stdin"],
            input=content,
            capture_output=True,
            check=True,
        )
        .stdout.decode()
        .strip()
    )


def test_series_numbering_rows_reports_duplicate_and_gap(ms):
    rows = [
        ("0001-a.patch", "clean", ""),
        ("0002-b.patch", "clean", ""),
        ("0002-c.patch", "clean", ""),
        ("0004-d.patch", "clean", ""),
    ]
    assert ms._series_numbering_rows(rows) == [
        ("series/0002", "duplicate-number", "0002-b.patch, 0002-c.patch"),
        (
            "series numbering",
            "non-contiguous",
            "expected 0001..0004, missing 0003 "
            "(numbers present: 0001, 0002, 0004)",
        ),
    ]


def test_series_numbering_rows_flags_unnumbered_and_accepts_healthy(ms):
    rows = [("0001-a.patch", "clean", ""), ("fix-thing.patch", "clean", "")]
    assert [r[1] for r in ms._series_numbering_rows(rows)] == ["unnumbered"]
    healthy = [("0001-a.patch", "clean", ""), ("0002-b.patch", "clean", "")]
    assert ms._series_numbering_rows(healthy) == []
    assert ms._series_numbering_rows([]) == []  # no vacuous `0001..0000` claim


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_check_series_flags_duplicate_number(ms, tmp_path, monkeypatch, capsys):
    # Two PRs each picking "the next free number" apply cleanly, so nothing
    # else notices until a regeneration — the gate has to catch it earlier.
    monkeypatch.setattr(
        ms,
        "SERIES_DIR",
        _series_fixture(tmp_path, "0001-a.patch", "0002-b.patch", "0002-c.patch"),
    )
    rc = ms.main(["check-series"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "duplicate-number" in out and "0002-b.patch, 0002-c.patch" in out
    # 0001..0002 is gapless, so the duplicate is the whole finding: the old
    # count-derived expectation also invented a `missing 0003`.
    assert "non-contiguous" not in out


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_check_series_flags_gap(ms, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        ms, "SERIES_DIR", _series_fixture(tmp_path, "0001-a.patch", "0003-c.patch")
    )
    rc = ms.main(["check-series"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "non-contiguous" in out and "missing 0002" in out


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_check_series_accepts_unique_contiguous(ms, tmp_path, monkeypatch, capsys):
    # A canonical fixture: the numbering verdict is only reachable on entries
    # that pass every other per-entry check (see the canonical-form tests).
    monkeypatch.setattr(ms, "SERIES_DIR", _canonical_series(tmp_path, "a", "b", "c"))
    rc = ms.main(["check-series"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "numbering: 3 entries are unique and contiguous 0001..0003" in out


# ---------------------------------------------------------------- gate hardening
# One regression test per review finding on the series-format gate.


def test_subcommands_are_wired(ms):
    """patch_validate.py branches on SUBCOMMANDS, so it must match the parser."""
    assert "{" + ",".join(ms.SUBCOMMANDS) + "}" in ms._build_parser().format_help()


def test_check_series_fails_closed_without_a_series_dir(ms, tmp_path, monkeypatch, capsys):
    # Globbing zero entries used to print "0 clean / 0 total / 0 need
    # attention" and exit 0: a deleted series directory stayed green.
    missing = tmp_path / "not-there"
    monkeypatch.setattr(ms, "SERIES_DIR", missing)

    rc = ms.main(["check-series"])
    out = capsys.readouterr().out
    rows = ms._series_format_rows()

    assert rc == 1
    assert [r[1] for r in rows] == ["missing-series-dir"]
    assert str(missing) in rows[0][2]  # the detail names the path
    assert "missing-series-dir" in out and str(missing) in out
    assert "--- numbering:" not in out  # no vacuous `0001..0000`
    assert out.strip().endswith(
        "=== musa_sync check-series: FAIL (series-format=1, numbering=0) ==="
    )


def test_check_series_fails_closed_on_an_empty_series_dir(
    ms, tmp_path, monkeypatch, capsys
):
    empty = tmp_path / "series"
    empty.mkdir()
    monkeypatch.setattr(ms, "SERIES_DIR", empty)

    rc = ms.main(["check-series"])
    out = capsys.readouterr().out

    assert rc == 1
    assert ms._series_format_rows() == [
        (str(empty), "empty-series", f"{empty} holds no *.patch entries")
    ]
    assert "empty-series" in out and "--- numbering:" not in out


def test_check_series_requires_the_author_header(ms, tmp_path, monkeypatch, capsys):
    # A canonical separator without a `From:` ident used to pass, and `git am`
    # then died on 'fatal: empty ident name (for <>) not allowed'.
    lines = _donor_patch().read_bytes().split(b"\n")
    assert lines[1].startswith(b"From: ")
    series = tmp_path / "series"
    _write_series(series, "0001-entry.patch", b"\n".join([lines[0], *lines[2:]]))
    monkeypatch.setattr(ms, "SERIES_DIR", series)

    rc = ms.main(["check-series"])
    out = capsys.readouterr().out

    assert rc == 1
    assert "missing-author" in out
    assert "empty ident name" in out and "git am" in out


def test_check_series_requires_an_index_line(ms, tmp_path, monkeypatch, capsys):
    # Without one, `git am -3` dies on
    # 'error: sha1 information is lacking or useless'.
    text = _donor_patch().read_bytes()
    stripped = b"\n".join(
        line for line in text.split(b"\n") if not line.startswith(b"index ")
    )
    series = tmp_path / "series"
    _write_series(series, "0001-entry.patch", stripped)
    monkeypatch.setattr(ms, "SERIES_DIR", series)

    rc = ms.main(["check-series"])
    out = capsys.readouterr().out

    assert rc == 1
    assert "no-index-line" in out and "3-way" in out


def test_index_line_detection_searches_past_the_mailbox_header(
    ms, tmp_path, monkeypatch, capsys
):
    # 34 of the shipped entries carry a commit-message body, which pushes the
    # first `index` line past line 12: a header-only scan reads them all as
    # clean no matter what their diff says. The subject slug has to match the
    # filename, or the canonical-form check answers first.
    lines = _donor_patch().read_bytes().split(b"\n")
    body = [f"commit-message body line {i}".encode() for i in range(20)]
    text = b"\n".join([*lines[:5], *body, *lines[5:]])
    first_index = next(
        i for i, line in enumerate(text.split(b"\n")) if line.startswith(b"index ")
    )
    assert first_index > 12, first_index  # the fixture is the point
    subject = next(
        line for line in text.split(b"\n") if line.startswith(b"Subject: ")
    )
    text = text.replace(subject, b"Subject: [PATCH] entry")
    series = tmp_path / "series"
    _write_series(series, "0001-entry.patch", text)
    monkeypatch.setattr(ms, "SERIES_DIR", series)

    assert ms.main(["check-series"]) == 0
    assert "no-index-line" not in capsys.readouterr().out


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_repo_half_accepts_blobs_the_series_itself_produces(
    ms, tmp_path, monkeypatch, capsys
):
    """The blob half must not go red on a checkout that simply lacks the MUSA
    objects: a fresh clone at the pin resolves 0/230 postimages and 133/225
    preimages, yet `git am -3` replays all 171 entries."""
    repo = tmp_path / "upstream"
    _init_repo(repo, "alpha\n")
    base = _blob_id(repo, b"alpha\n")  # upstream content: must be resolvable
    beta = _blob_id(repo, b"alpha\nbeta\n")  # entry 0001 produces this
    gamma = _blob_id(repo, b"alpha\nbeta\ngamma\n")  # entry 0002 produces this

    series = tmp_path / "series"
    _write_series(
        series,
        "0001-add-beta.patch",
        _mailbox(
            "add beta",
            f"""diff --git a/value.txt b/value.txt
index {base}..{beta} 100644
--- a/value.txt
+++ b/value.txt
@@ -1 +1,2 @@
 alpha
+beta
""",
        ),
    )
    _write_series(
        series,
        "0002-add-gamma.patch",
        _mailbox(
            "add gamma",
            f"""diff --git a/value.txt b/value.txt
index {beta}..{gamma} 100644
--- a/value.txt
+++ b/value.txt
@@ -1,2 +1,3 @@
 alpha
 beta
+gamma
""",
        ),
    )
    monkeypatch.setattr(ms, "SERIES_DIR", series)

    # 0002 anchors on the blob 0001 creates, which this pin never had.
    assert ms.main(["check-series", "--repo", str(repo)]) == 0
    assert "missing-index-blob" not in capsys.readouterr().out

    # Anchoring on a blob no entry produced is the real rot: reported, with
    # the offending hash and the entry that declares it.
    stale = "eeeeeeeeee"
    assert len(stale) == 10
    _write_series(
        series,
        "0002-add-gamma.patch",
        _mailbox(
            "add gamma",
            f"""diff --git a/value.txt b/value.txt
index {stale}..{gamma} 100644
--- a/value.txt
+++ b/value.txt
@@ -1,2 +1,3 @@
 alpha
 beta
+gamma
""",
        ),
    )
    rc = ms.main(["check-series", "--repo", str(repo)])
    out = capsys.readouterr().out

    assert rc == 1
    assert "missing-index-blob" in out
    assert stale in out and "0002-add-gamma.patch" in out


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_repo_half_ignores_null_index_blobs(ms, tmp_path, monkeypatch, capsys):
    # `index 0000000000..` is a creation hunk (4 shipped entries have one): the
    # null id is never resolvable in any repo and never required.
    repo = tmp_path / "upstream"
    _init_repo(repo)
    blob = _blob_id(repo, b"hello\n")
    series = tmp_path / "series"
    _write_series(
        series,
        "0001-new-file.patch",
        _mailbox(
            "new file",
            f"""diff --git a/new.txt b/new.txt
new file mode 100644
index 0000000000..{blob}
--- /dev/null
+++ b/new.txt
@@ -0,0 +1 @@
+hello
""",
        ),
    )
    monkeypatch.setattr(ms, "SERIES_DIR", series)

    assert ms.main(["check-series", "--repo", str(repo)]) == 0
    assert "missing-index-blob" not in capsys.readouterr().out


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_repo_half_only_asks_for_blobs_the_series_cannot_produce(ms, tmp_path, monkeypatch):
    """A blob half that demanded every declared id exist would turn the shipped
    171-entry series red on any pristine clone: a fresh clone at the pin resolves
    0/230 postimages and 133/225 preimages, yet `git am -3` replays all 171.

    So the ids handed to the repo are exactly the preimages the series does not
    already declare as the postimage of an *earlier* entry (and never a null id,
    which no repo has, and never a postimage, which the entry itself creates).
    """
    queried: list[str] = []

    def fake_types(repo, blobs):
        queried.extend(blobs)
        return {blob: "blob" for blob in blobs}  # a clone that has them all

    monkeypatch.setattr(ms, "_repo_object_types", fake_types)

    rows = ms._series_format_rows(tmp_path)

    assert [r[1] for r in rows if r[1] != "clean"] == []
    produced = {
        post
        for patch in ms.SERIES_DIR.glob("*.patch")
        for _, post in ms._declared_blobs(patch.read_bytes())
    }
    assert queried, "the blob half must still consult the repo"
    assert not set(queried) & produced  # MUSA-side ids are exempt
    assert not [b for b in queried if ms._ZERO_BLOB.match(b.encode())]


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_repo_half_exempts_an_anchor_only_an_earlier_entry_produces(
    ms, tmp_path, monkeypatch
):
    """The ordering rule that `test_repo_half_only_asks_...` cannot state: an
    anchor a *later* entry produces is not in the odb yet when this one replays,
    and is therefore still asked for (see the F2 tests below)."""
    queried: list[str] = []

    def fake_types(repo, blobs):
        queried.extend(blobs)
        # A repo that has everything except the one anchor nothing produces.
        return {b: ("missing" if b == "eeeeeeeeee" else "blob") for b in blobs}

    monkeypatch.setattr(ms, "_repo_object_types", fake_types)
    _write_series(
        tmp_path / "series",
        "0001-first.patch",
        _mailbox(
            "first",
            "diff --git a/value.txt b/value.txt\n"
            "index eeeeeeeeee..1111111111 100644\n"
            "--- a/value.txt\n+++ b/value.txt\n"
            "@@ -1 +1,2 @@\n alpha\n+first\n",
        ),
    )
    _write_series(
        tmp_path / "series",
        "0002-second.patch",
        _mailbox(
            "second",
            "diff --git a/value.txt b/value.txt\n"
            "index 1111111111..2222222222 100644\n"
            "--- a/value.txt\n+++ b/value.txt\n"
            "@@ -1,2 +1,3 @@\n alpha\n first\n+second\n",
        ),
    )
    monkeypatch.setattr(ms, "SERIES_DIR", tmp_path / "series")

    rows = {
        name: (status, detail)
        for name, status, detail in ms._series_format_rows(tmp_path)
    }

    # 0002 anchors on the blob 0001 produces earlier in the series: exempt, and
    # never asked of the repo at all.
    assert "1111111111" not in queried
    assert rows["0002-second.patch"][0] == "clean"
    # 0001's own anchor (eeeeeeeeee) has no producer before it.
    assert rows["0001-first.patch"][0] == "missing-index-blob"
    assert "eeeeeeeeee" in queried


def test_mailbox_status_splits_real_sha_from_garbage(ms, tmp_path, monkeypatch, capsys):
    # A real-sha mailbox is perfectly replayable; only the *canonical* all-zero
    # separator is the `regen` fixed point the PR contracts on.
    donor = _donor_patch().read_bytes().split(b"\n")
    real_sha = b"From 0123456789abcdef0123456789abcdef01234567 Mon Sep 17 00:00:00 2001"
    series = tmp_path / "series"
    _write_series(series, "0001-real-sha.patch", b"\n".join([real_sha, *donor[1:]]))
    _write_series(
        series,
        "0002-bare-diff.patch",
        b"--- a/value.txt\n+++ b/value.txt\n@@ -1 +1 @@\n-alpha\n+beta\n",
    )
    monkeypatch.setattr(ms, "SERIES_DIR", series)

    rc = ms.main(["check-series"])
    out = capsys.readouterr().out
    rows = {name: (status, detail) for name, status, detail in ms._series_format_rows()}

    assert rc == 1
    assert rows["0001-real-sha.patch"][0] == "non-canonical-mailbox"
    assert "re-run `regen`" in rows["0001-real-sha.patch"][1]
    assert "cannot replay" not in rows["0001-real-sha.patch"][1]
    assert rows["0002-bare-diff.patch"][0] == "not-a-mailbox"
    assert "cannot replay it" in rows["0002-bare-diff.patch"][1]
    assert "non-canonical-mailbox" in out and "not-a-mailbox" in out


def test_numbering_uses_a_strict_ascii_prefix(ms):
    # str.isdigit() is Unicode-aware, and name[:4] mis-slices wider prefixes.
    assert ms._number_prefix("0001-b.patch") == "0001"
    assert ms._number_prefix("0001") == "0001"
    assert ms._number_prefix("00123-b.patch") is None
    assert ms._number_prefix("0001b.patch") is None
    assert ms._number_prefix("٠٠٠١-b.patch") is None  # Arabic-Indic digits

    for name in ("00123-b.patch", "٠٠٠١-b.patch", "fix-thing.patch"):
        assert [r[1] for r in ms._series_numbering_rows([(name, "clean", "")])] == [
            "unnumbered"
        ]


def test_numbering_expected_range_follows_the_numbers_present(ms):
    # `0001`+`0003` used to be reported as "expected 0001..0002, missing 0002".
    assert ms._series_numbering_rows(
        [("0001-a.patch", "clean", ""), ("0003-c.patch", "clean", "")]
    ) == [
        (
            "series numbering",
            "non-contiguous",
            "expected 0001..0003, missing 0002 (numbers present: 0001, 0003)",
        )
    ]


def test_series_gate_reports_a_missing_git_instead_of_raising(
    ms, tmp_path, monkeypatch, capsys
):
    # Canonical: the gate has to get as far as running git for the missing
    # binary to be the thing it reports.
    monkeypatch.setattr(ms, "SERIES_DIR", _canonical_series(tmp_path, "a"))
    monkeypatch.setattr(ms.shutil, "which", lambda name: None)

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(ms.subprocess, "run", no_git)

    rc = ms.main(["check-series"])
    out = capsys.readouterr().out

    assert rc == 1
    assert "git-unavailable" in out and "git executable not found on PATH" in out


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_series_gate_runs_git_with_an_explicit_cwd(ms, tmp_path, monkeypatch, capsys):
    # The gate used to inherit the caller's cwd for `git apply --stat`.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ms, "SERIES_DIR", _canonical_series(tmp_path, "a"))
    seen = []
    real = ms._run_git

    def record(cwd, *args, **kwargs):
        seen.append((cwd, args[:1]))
        return real(cwd, *args, **kwargs)

    monkeypatch.setattr(ms, "_run_git", record)
    assert ms.main(["check-series"]) == 0
    assert seen and all(cwd is not None for cwd, _ in seen)


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_verify_series_gate_precedes_the_divergence_summary_and_ends_in_a_verdict(
    ms, tmp_path, monkeypatch, capsys
):
    lines = _donor_patch().read_bytes().split(b"\n")
    no_author = b"\n".join([lines[0], *lines[2:]])
    series = tmp_path / "series"
    for i in range(1, 12):  # 11 bad rows: the section truncates at 10
        _write_series(series, f"{i:04d}-entry.patch", no_author)
    monkeypatch.setattr(ms, "SERIES_DIR", series)
    monkeypatch.setattr(ms, "_ensure_clone", lambda target, repo: (tmp_path, False))
    # The blob half is covered elsewhere; keep this test on the output contract.
    monkeypatch.setattr(
        ms, "_repo_object_types", lambda repo, blobs: {b: "blob" for b in blobs}
    )
    monkeypatch.setattr(
        ms, "_verify_rows", lambda clone: [("entry-id", "1", "clean", "")]
    )

    rc = ms.main(["verify", "--target", "deadbeef"])
    out = capsys.readouterr().out
    printed = [line for line in out.splitlines() if line.strip()]

    assert rc == 1
    assert out.index("--- series format:") < out.index(
        "--- 1 clean / 1 total / 0 need attention (none) ---"
    )
    assert "… 1 more" in out  # truncation is marked, not silent
    assert "empty ident name" in out  # the row detail survives verify
    assert printed[-1] == (
        "=== musa_sync verify: FAIL "
        "(divergence=0, series-format=11, numbering=0) ==="
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_verify_ends_in_a_pass_verdict(ms, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ms, "SERIES_DIR", _canonical_series(tmp_path, "a"))
    monkeypatch.setattr(ms, "_ensure_clone", lambda target, repo: (tmp_path, False))
    # The blob half is covered elsewhere; keep this test on the output contract.
    monkeypatch.setattr(
        ms, "_repo_object_types", lambda repo, blobs: {b: "blob" for b in blobs}
    )
    monkeypatch.setattr(
        ms, "_verify_rows", lambda clone: [("entry-id", "1", "clean", "")]
    )

    rc = ms.main(["verify", "--target", "deadbeef"])
    out = capsys.readouterr().out

    assert rc == 0
    assert out.splitlines()[-1] == "=== musa_sync verify: PASS ==="


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_shipped_series_passes_the_format_gate(ms, capsys):
    rc = ms.main(["check-series"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "--- 171 clean / 171 total / 0 need attention ---" in out
    assert out.splitlines()[-1] == "=== musa_sync check-series: PASS ==="


def test_patch_validate_passes_known_subcommands_through():
    validate = ROOT / "tools" / "patch_validate.py"
    passthrough = subprocess.run(
        [sys.executable, str(validate), "check-series", "--help"],
        capture_output=True,
        text=True,
    )
    default = subprocess.run(
        [sys.executable, str(validate), "--help"], capture_output=True, text=True
    )

    # `check-series` used to be mangled into `verify check-series` (argparse
    # error, exit 2); a bare flag still defaults to `verify`.
    assert passthrough.returncode == 0
    assert "musa_sync check-series" in passthrough.stdout
    assert default.returncode == 0
    assert "musa_sync verify" in default.stdout


# ---------------------------------------------------------------- audit round 2
# One regression test per adversarial-audit finding on the series gate. Each
# docstring names the mutation of tools/musa_sync.py that reddens it again.


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_check_series_replay_reports_the_entry_git_cannot_replay(
    ms, tmp_path, monkeypatch, capsys
):
    """F1: a well-formed mailbox whose hunk no longer matches anything is green
    on the format half, and only a real replay notices.

    Mutation: drop the ``_replay_rows`` call from ``_series_format_rows``.
    """
    repo, name, text = _formatted_entry(tmp_path)
    series = tmp_path / "series"
    _write_series(series, name, text.replace(b"\n alpha\n", b"\n gamma\n"))
    monkeypatch.setattr(ms, "SERIES_DIR", series)

    # The cheap half is green — it cannot see applicability, and says so.
    assert ms.main(["check-series", "--repo", str(repo)]) == 0
    capsys.readouterr()
    assert ms.main(["check-series", "--repo", str(repo), "--replay"]) == 1
    out = capsys.readouterr().out

    assert "replay-failed" in out and name in out
    assert "error:" in out  # git's own first error line survives into the row
    assert out.strip().endswith(
        "=== musa_sync check-series: FAIL (series-format=1, numbering=0) ==="
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_check_series_replay_and_round_trip_need_a_repo(ms, capsys):
    """F1/F4b: both modes are opt-in and neither guesses a repository.

    Mutation: return 0 instead of 2 for a missing ``--repo``.
    """
    assert ms.main(["check-series", "--replay"]) == 2
    assert "pass --repo" in capsys.readouterr().out
    assert ms.main(["check-series", "--round-trip"]) == 2
    assert "pass --repo" in capsys.readouterr().out


def _ghost_anchor_diff(ghost: str) -> str:
    """A hunk whose context matches nothing, so replay *needs* the index blob."""
    return (
        "diff --git a/value.txt b/value.txt\n"
        f"index {ghost}..{ghost} 100644\n"
        "--- a/value.txt\n+++ b/value.txt\n"
        "@@ -1 +1,2 @@\n-gamma\n+delta\n+epsilon\n"
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_repo_half_rejects_an_anchor_the_entry_itself_declares_as_a_postimage(
    ms, tmp_path, monkeypatch, capsys
):
    """F2: `produced` used to be built from *all* entries' postimages, so an
    entry could exempt its own anchor (``index X..X``) and pass while git am -3
    died on `sha1 information is lacking or useless`.

    Mutation: build ``produced`` from every entry's postimages up front instead
    of only from the entries that precede the one being checked.
    """
    repo = tmp_path / "upstream"
    _init_repo(repo)
    ghost = "ab" * 5
    series = tmp_path / "series"
    entry = _write_series(
        series, "0001-ghost.patch", _mailbox("ghost", _ghost_anchor_diff(ghost))
    )
    monkeypatch.setattr(ms, "SERIES_DIR", series)

    assert ms.main(["check-series", "--repo", str(repo)]) == 1
    out = capsys.readouterr().out
    assert "missing-index-blob" in out and ghost in out and "0001-ghost.patch" in out
    # The gate now agrees with git: this entry cannot replay at that pin.
    assert ms._replay_rows(repo, [entry], round_trip=False)[0][1] == "replay-failed"


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_repo_half_rejects_an_anchor_only_a_later_entry_produces(
    ms, tmp_path, monkeypatch, capsys
):
    """F2: a *later* entry used to exempt an earlier one's anchor, which cannot
    work — at the moment 0001 replays, 0002 has not run yet.

    Mutation: exempt any postimage the series declares, whatever its position.
    """
    repo = tmp_path / "upstream"
    _init_repo(repo)
    ghost = "cd" * 5
    series = tmp_path / "series"
    _write_series(
        series, "0001-ghost.patch", _mailbox("ghost", _ghost_anchor_diff(ghost))
    )
    base = _blob_id(repo, b"alpha\n")
    _write_series(
        series,
        "0002-producer.patch",
        _mailbox(
            "producer",
            "diff --git a/value.txt b/value.txt\n"
            f"index {base}..{ghost} 100644\n"
            "--- a/value.txt\n+++ b/value.txt\n"
            "@@ -1 +1,2 @@\n alpha\n+beta\n",
        ),
    )
    monkeypatch.setattr(ms, "SERIES_DIR", series)

    rc = ms.main(["check-series", "--repo", str(repo)])
    out = capsys.readouterr().out

    assert rc == 1
    assert "missing-index-blob" in out and ghost in out
    assert "0001-ghost.patch" in out
    # 0002 declares the ghost id as its own postimage: that is not a defect.
    assert "0002-producer.patch" not in out.split("--- 2 clean")[0]


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_repo_half_rejects_an_anchor_that_is_not_a_blob(
    ms, tmp_path, monkeypatch, capsys
):
    """F3: any `cat-file` line not ending in ` missing` used to satisfy the
    anchor check, so a commit or tree id passed and the entry then failed to
    replay. Only a blob is a usable 3-way ancestor.

    Mutation: accept any resolvable id in ``_missing_index_blobs`` again.
    """
    repo = tmp_path / "upstream"
    commit = _init_repo(repo)
    assert _git(repo, "cat-file", "-t", commit).stdout.strip() == "commit"
    series = tmp_path / "series"
    _write_series(
        series, "0001-commit-anchor.patch", _mailbox("commit anchor", _ghost_anchor_diff(commit))
    )
    monkeypatch.setattr(ms, "SERIES_DIR", series)

    assert ms.main(["check-series", "--repo", str(repo)]) == 1
    out = capsys.readouterr().out

    assert "missing-index-blob" in out
    assert commit in out and "commit" in out and "not a blob" in out


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_check_series_rejects_a_crlf_entry(ms, tmp_path, monkeypatch, capsys):
    """F4a/F5: CRLF passed every check while the build applier rejected the old
    patch and `regen` rewrote the file.

    Mutation: delete the ``b"\\r" in text`` branch of ``_mailbox_form_problem``.
    """
    _, name, text = _formatted_entry(tmp_path)
    series = tmp_path / "series"
    _write_series(series, name, text.replace(b"\n", b"\r\n"))
    monkeypatch.setattr(ms, "SERIES_DIR", series)

    assert ms.main(["check-series"]) == 1
    out = capsys.readouterr().out
    assert "non-canonical-crlf" in out and "build/replay divergence" in out


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_crlf_entry_replays_with_git_am_but_not_with_the_build_applier(
    ms, tmp_path, monkeypatch
):
    """F5: the two paths disagree, and the gate must side with the build.

    Mutation: same as above — drop the CRLF branch.
    """
    repo, name, text = _formatted_entry(tmp_path)
    series = tmp_path / "series"
    crlf = _write_series(series, name, text.replace(b"\n", b"\r\n"))
    monkeypatch.setattr(ms, "SERIES_DIR", series)

    assert ms._replay_rows(repo, [crlf], round_trip=False) == []  # git am -3: fine
    assert ms.build_apply.apply_patch(repo, crlf, check_only=True) == "conflict"
    assert [r[1] for r in ms._series_format_rows()] == ["non-canonical-crlf"]


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_check_series_rejects_a_bom_and_a_missing_final_newline(
    ms, tmp_path, monkeypatch, capsys
):
    """F4a: both are `regen` rewrites, and both used to be reported (at best) as
    a downstream symptom — `not-a-mailbox` and a `git apply --stat` corruption.

    Mutation: delete the BOM or the final-newline branch of
    ``_mailbox_form_problem``.
    """
    _, name, text = _formatted_entry(tmp_path)
    for label, payload, status in (
        ("bom", b"\xef\xbb\xbf" + text, "non-canonical-bom"),
        ("eof", text.rstrip(b"\n"), "non-canonical-eof"),
    ):
        series = tmp_path / label / "series"
        _write_series(series, name, payload)
        monkeypatch.setattr(ms, "SERIES_DIR", series)

        assert ms.main(["check-series"]) == 1, label
        out = capsys.readouterr().out
        assert status in out, label


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_check_series_rejects_a_missing_or_noncanonical_date(
    ms, tmp_path, monkeypatch, capsys
):
    """F4a: without the author date `regen` writes, the entry is not a fixed
    point (a missing Date silently becomes "now" on the next replay).

    Mutation: delete the `Date:` branch of ``_canonical_form_problem``.
    """
    _, name, text = _formatted_entry(tmp_path)
    stripped = b"\n".join(
        line for line in text.split(b"\n") if not line.startswith(b"Date:")
    )
    assert stripped != text
    for label, payload in (
        ("missing", stripped),
        ("edited", text.replace(b"Date: ", b"Date: whenever ", 1)),
    ):
        series = tmp_path / label / "series"
        _write_series(series, name, payload)
        monkeypatch.setattr(ms, "SERIES_DIR", series)

        assert ms.main(["check-series"]) == 1, label
        out = capsys.readouterr().out
        assert "non-canonical-date" in out, label


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_check_series_rejects_a_subject_slug_that_disagrees_with_the_filename(
    ms, tmp_path, monkeypatch, capsys
):
    """F4a: `regen` names an entry from its subject, so a filename that does not
    match the slug is a rename waiting to happen.

    Mutation: delete the slug comparison in ``_canonical_form_problem``.
    """
    _, name, text = _formatted_entry(tmp_path)
    series = tmp_path / "series"
    _write_series(series, "0001-something-else.patch", text)  # slug says "a"
    monkeypatch.setattr(ms, "SERIES_DIR", series)

    assert ms.main(["check-series"]) == 1
    out = capsys.readouterr().out
    assert "non-canonical-subject" in out
    assert "does not match the filename slug" in out
    assert "0001-a.patch" in out  # the name regen would write


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_check_series_rejects_a_numbered_subject(ms, tmp_path, monkeypatch, capsys):
    """F4a: `[PATCH 1/1]` is 1-of-1 numbering, which `--no-numbered` rewrites.

    Mutation: delete the numbered-subject branch of ``_canonical_form_problem``.
    """
    _, name, text = _formatted_entry(tmp_path)
    numbered = text.replace(b"Subject: [PATCH] a", b"Subject: [PATCH 1/1] a")
    assert numbered != text
    series = tmp_path / "series"
    _write_series(series, name, numbered)
    monkeypatch.setattr(ms, "SERIES_DIR", series)

    assert ms.main(["check-series"]) == 1
    out = capsys.readouterr().out
    assert "non-canonical-subject" in out and "numbered subject" in out


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_check_series_rejects_a_duplicated_diff_body(ms, tmp_path, monkeypatch, capsys):
    """F4a: two entries with the same body apply twice and only fall out at the
    next `regen`, which emits one entry per commit.

    Mutation: make ``_duplicate_bodies`` return an empty mapping.
    """
    _, name, text = _formatted_entry(tmp_path)
    series = tmp_path / "series"
    _write_series(series, name, text)  # 0001-a.patch
    # Same diff body, its own subject, and a filename that matches that subject.
    copy = text.replace(b"Subject: [PATCH] a", b"Subject: [PATCH] b")
    _write_series(series, "0002-b.patch", copy)
    monkeypatch.setattr(ms, "SERIES_DIR", series)

    assert ms.main(["check-series"]) == 1
    out = capsys.readouterr().out
    assert "non-canonical-duplicate" in out
    assert "0002-b.patch" in out and "byte-identical to 0001-a.patch" in out


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_check_series_round_trip_reports_what_regen_would_rewrite(
    ms, tmp_path, monkeypatch, capsys
):
    """F4b: an entry can pass every cheap check and still not be a `regen` fixed
    point — a changed author ident is the clean example (git am keeps it, the
    author rewrite drops it).

    Mutation: make ``_round_trip_rows`` return [] after a successful replay.
    """
    repo, name, text = _formatted_entry(tmp_path)
    series = tmp_path / "series"
    _write_series(
        series, name, text.replace(b"From: musa <musa@local>", b"From: someone <s@x>")
    )
    monkeypatch.setattr(ms, "SERIES_DIR", series)

    # The cheap half and a plain replay are both happy with it.
    assert ms.main(["check-series", "--repo", str(repo)]) == 0
    capsys.readouterr()
    assert ms.main(["check-series", "--repo", str(repo), "--replay"]) == 0
    capsys.readouterr()
    assert ms.main(["check-series", "--repo", str(repo), "--round-trip"]) == 1
    out = capsys.readouterr().out

    assert "round-trip-dirty" in out and "regen writes" in out


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_check_series_round_trip_accepts_a_regen_fixed_point(
    ms, tmp_path, monkeypatch, capsys
):
    """F4b: the mode the shipped series is held to must be green when the entry
    really is what `regen` writes.

    Mutation: make ``_round_trip_rows`` report every entry.
    """
    repo, name, text = _formatted_entry(tmp_path)
    _write_series(tmp_path / "series", name, text)
    monkeypatch.setattr(ms, "SERIES_DIR", tmp_path / "series")

    assert ms.main(["check-series", "--repo", str(repo), "--round-trip"]) == 0
    assert capsys.readouterr().out.strip().endswith(
        "=== musa_sync check-series: PASS ==="
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_check_series_flags_a_0000_prefix_and_the_observed_range(
    ms, tmp_path, monkeypatch, capsys
):
    """F6: `0000-x.patch` passed and the numbering line then claimed
    `contiguous 0001..0001`.

    Mutation: delete the ``numbered[0] == "0000"`` row, or print
    ``0001..{len(rows)}`` again in ``_numbering_line``.
    """
    _write_series(tmp_path / "series", "0000-x.patch", _canonical_entry("x"))
    monkeypatch.setattr(ms, "SERIES_DIR", tmp_path / "series")

    assert ms.main(["check-series"]) == 1
    out = capsys.readouterr().out

    assert "bad-range" in out and "series/0000" in out
    assert "contiguous 0001..0001" not in out
    assert "--- numbering: 1 problem(s) ---" in out


def test_numbering_line_reports_the_range_it_observed(ms):
    """F6: the range is the observed min..max, never `0001..<count>`.

    Mutation: print ``f"0001..{len(rows):04d}"`` again.
    """
    rows = [("0007-g.patch", "clean", ""), ("0008-h.patch", "clean", "")]
    assert ms._numbering_line(rows, []) == (
        "--- numbering: 2 entries are unique and contiguous 0007..0008 ---"
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_check_series_rejects_an_empty_slug(ms, tmp_path, monkeypatch, capsys):
    """F7: `0001-.patch` passed, and `regen` renamed it.

    Mutation: delete the ``_EMPTY_SLUG`` row from ``_entry_name_problem``.
    """
    _write_series(tmp_path / "series", "0001-.patch", _canonical_entry("a"))
    monkeypatch.setattr(ms, "SERIES_DIR", tmp_path / "series")

    assert ms.main(["check-series"]) == 1
    out = capsys.readouterr().out
    assert "bad-name" in out and "empty slug" in out


def test_check_series_rejects_a_non_regular_entry(ms, tmp_path, monkeypatch, capsys):
    """F8: a symlink entry passed, and `regen` would write *through* it, outside
    series/. A directory or fifo entry used to kill or hang the CLI.

    Mutation: delete the ``stat.S_ISREG`` branch of ``_read_entry``.
    """
    target = tmp_path / "outside.patch"
    target.write_bytes(_canonical_entry("a"))
    series = tmp_path / "series"
    series.mkdir()
    os.symlink(target, series / "0001-symlink.patch")
    (series / "0002-directory.patch").mkdir()
    os.mkfifo(series / "0003-fifo.patch")
    monkeypatch.setattr(ms, "SERIES_DIR", series)

    rows = {name: status for name, status, _ in ms._series_format_rows()}

    assert rows == {
        "0001-symlink.patch": "not-regular-file",
        "0002-directory.patch": "not-regular-file",
        "0003-fifo.patch": "not-regular-file",
    }
    assert target.read_bytes() == _canonical_entry("a")  # never written through
    assert ms.main(["check-series"]) == 1
    assert "not a regular file" in capsys.readouterr().out


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
@pytest.mark.parametrize(
    "kind,status",
    [
        ("chmod-000", "unreadable"),
        ("dangling-symlink", "not-regular-file"),
        ("directory", "not-regular-file"),
    ],
)
def test_cli_never_tracebacks_on_a_broken_entry_file(
    tmp_path, kind, status
):
    """F9: chmod-000 / dangling symlink / directory entries used to kill the CLI
    at import time (`manifest.py: _patch_target`), rc 1 with a traceback and no
    verdict line.

    Mutation: let ``manifest._patch_target`` raise OSError again (it is read at
    import time, before any subcommand can print anything).
    """
    tree = _cli_tree(tmp_path, {"0001-a.patch": _canonical_entry("a")})
    entry = tree / "vllm_musa" / "patches" / "series" / "0001-a.patch"
    if kind == "chmod-000":
        entry.chmod(0o000)
    else:
        entry.unlink()
        if kind == "dangling-symlink":
            os.symlink("/nonexistent", entry)
        else:
            entry.mkdir()

    try:
        result = _run_cli(tree, "check-series")
    finally:
        if kind == "chmod-000":
            entry.chmod(0o644)  # let pytest's tmp_path cleanup succeed

    assert "Traceback" not in result.stdout + result.stderr
    assert result.returncode == 1
    assert status in result.stdout
    assert result.stdout.strip().endswith(
        "=== musa_sync check-series: FAIL (series-format=1, numbering=0) ==="
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_check_series_fails_closed_on_a_repo_it_cannot_read(ms, tmp_path, capsys):
    """F10: an unusable ``--repo`` is a reported failure, never a green run.

    Mutation: let ``_repo_object_types`` return {} instead of raising
    ``RepoUnusable`` when git fails.
    """
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    _write_series(tmp_path / "series", "0001-a.patch", _canonical_entry("a"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ms, "SERIES_DIR", tmp_path / "series")
        assert ms.main(["check-series", "--repo", str(plain)]) == 1
    out = capsys.readouterr().out

    assert "repo-unusable" in out and str(plain) in out
    assert out.strip().endswith("=== musa_sync check-series: FAIL (series-format=1, numbering=0) ===")


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_check_series_replay_fails_closed_on_a_repo_it_cannot_clone(
    ms, tmp_path, monkeypatch, capsys
):
    """F10: the replay path must fail closed too — a repo it cannot clone is a
    ``repo-unusable`` row, not a traceback or a silent pass.

    Mutation: return [] instead of raising ``RepoUnusable`` when the clone
    fails.
    """
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    entry = _write_series(tmp_path / "series", "0001-a.patch", _canonical_entry("a"))
    monkeypatch.setattr(ms, "SERIES_DIR", tmp_path / "series")

    assert ms.main(["check-series", "--repo", str(plain), "--replay"]) == 1
    assert "repo-unusable" in capsys.readouterr().out
    # ... and the clone failure itself is a RepoUnusable, not a traceback.
    with pytest.raises(ms.RepoUnusable, match="cannot clone"):
        ms._replay_rows(plain, [entry], round_trip=False)


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_series_gate_pins_the_git_apply_stat_corruption_check(
    ms, tmp_path, monkeypatch, capsys
):
    """F11: `git apply --stat` is what catches a diff whose hunk counts no
    longer match its body — a corruption the mailbox parser cannot see.

    Mutation: skip the ``git apply --stat`` call entirely (``problem`` stays
    None and the entry is reported clean).
    """
    _, name, text = _formatted_entry(tmp_path)
    broken = text.replace(b"@@ -1 +1,2 @@", b"@@ -1 +1,9 @@")
    assert broken != text
    _write_series(tmp_path / "series", name, broken)
    monkeypatch.setattr(ms, "SERIES_DIR", tmp_path / "series")

    assert ms.main(["check-series"]) == 1
    out = capsys.readouterr().out
    assert "corrupt" in out


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_check_series_reports_a_missing_final_newline_as_its_own_row(
    ms, tmp_path, monkeypatch, capsys
):
    """F11: the missing-final-newline corruption used to be caught only by
    ``git apply --stat`` and reported as a raw `corrupt`; it is a canonical-form
    defect and now says so.

    Mutation: delete the final-newline branch of ``_mailbox_form_problem``.
    """
    _, name, text = _formatted_entry(tmp_path)
    _write_series(tmp_path / "series", name, text.rstrip(b"\n"))
    monkeypatch.setattr(ms, "SERIES_DIR", tmp_path / "series")

    assert ms.main(["check-series"]) == 1
    assert "non-canonical-eof" in capsys.readouterr().out


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_entry_name_with_a_newline_cannot_forge_report_lines(tmp_path):
    """F12: a newline in an entry filename used to splice forged lines — a fake
    PASS verdict — into the report.

    Mutation: print the raw name instead of ``_safe_name(name)``.
    """
    forged = "=== musa_sync check-series: PASS (forged) ==="
    name = f"0001-a\n{forged}\n.patch"
    tree = _cli_tree(tmp_path, {name: _canonical_entry("a", marker="!")[:-1]})

    result = _run_cli(tree, "check-series")
    lines = result.stdout.splitlines()

    assert forged not in lines
    assert "entry-name-hostile" in result.stdout
    assert "0001-a\\x0a" in result.stdout  # the escape, not the raw byte
    assert lines[-1].startswith("=== musa_sync check-series: FAIL")


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_git_slug_reproduces_git_format_patch_filenames(ms, tmp_path):
    """The slug check is only useful if it is *exactly* git's rule: it is what
    keeps the gate from going red on entries `regen` would leave alone.

    Mutation: drop the 52-character truncation, or stop collapsing runs.
    """
    subjects = [
        "MUSA: vllm.compilation.backends",
        "MUSA: a_b.c-d e",
        "  leading   spaces  ",
        "trailing dash -",
        "MUSA(csrc-file): csrc/mamba/mamba_ssm/selective_scan_fwd.cu",
        "aaaaaaaaaa bbbbbbbbbb cccccccccc dddddddddd eeeeeeeeee ffffff",
        "a" * 51 + " b",
        "a" * 51 + "_b",
        "ab!@#cd",
        "unicode: h\u00e9llo w\u00f6rld",
    ]
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    for subject in subjects:
        _git(repo, "commit", "--quiet", "--allow-empty", "-m", subject)
    staged = tmp_path / "generated"
    subprocess.run(
        [
            "git", "-C", str(repo), "format-patch", "--no-signature",
            "--no-numbered", "--zero-commit", "-o", str(staged), base,
        ],
        check=True,
        capture_output=True,
    )

    names = sorted(p.name for p in staged.glob("*.patch"))
    assert len(names) == len(subjects)
    for subject, name in zip(subjects, names):
        assert ms._git_slug(subject) == name.split("-", 1)[1][: -len(".patch")], subject


def test_developer_guide_documents_the_gate_truthfully():
    """F13: the guide claimed a per-PR fixed-point gate that did not exist.

    Mutation: remove the documented modes from docs/mdm-developer-guide.md.
    """
    guide = (ROOT / "docs" / "mdm-developer-guide.md").read_text()

    for flag in ("--repo", "--replay", "--round-trip"):
        assert flag in guide, flag
    assert "no in-repo CI" in guide
    assert "runs on every PR" not in guide
