#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""the MDM driver — one tool for the whole divergence lifecycle.

The MUSA Divergence Manifest (MDM) keeps vllm-musa's divergence from upstream
vLLM in one declarative census (``vllm_musa/patches/manifest.py``) applied to a
pinned, cloned vLLM. This is the single CLI over that system.

Subcommands::

    apply  <repo> [--phase P] [--check-only] [--no-strict]
        Build-time: apply the build-applied diff series (categories 1/2/3/4b) to
        the cloned vLLM at <repo>, honoring apply_phase order. (setup.py calls this.)

    verify [--target REF] [--repo PATH]
        OFFLINE pre-bump gate (no MUSA hardware). Fresh checkout of vllm@REF (or
        --repo for an existing checkout), then for EVERY divergence emit one
        status row: build-applied diffs are cumulatively applied to a disposable
        copy (the supplied repo is never modified) and classified as
        (clean/obsolete/conflict); cat-5/cat-6 existence-probe their upstream
        target. Exits non-zero on any conflict / missing / orphaned divergence —
        the bounded review surface a version bump needs.

    rebase <ref> [--drop-upstreamed --history PATH]
        Checkout the clone at <ref> and ``git am -3`` the series in order,
        ``upstream/`` last (true 3-way; trivial upstream drift auto-merges,
        real conflicts halt). With ``--drop-upstreamed``, the entries
        ``upgrade-plan`` drops are skipped.

    upgrade-plan [<ref>] [--history PATH] [--fetch] [--no-content] [--all]
        Per series entry, ``keep``, ``drop`` or ``review`` on a pin bump to <ref>
        (default: the current pin), from the upstream history in PATH. An
        ``upstream/`` entry names its commit in its ``(cherry picked from
        commit …)`` line and its PR in the ``(#N)`` its subject ends in; it is
        in <ref> when the commit is an ancestor, a commit subject ends in its
        ``(#N)``, a commit says it was cherry-picked from it, or a commit has
        its patch-id. Such an entry is dropped when it changes what its
        upstream commit changes, and reviewed when it differs; a MUSA fix in
        ``upstream/`` is reviewed unless every upstream entry before it is
        kept. The series is also applied in order
        to the pristine <ref>, which flags a MUSA entry whose change is already
        there. Fails while an entry needs review.

    backport <sha> [--from PATH] [--no-regen]
        Cherry-pick an upstream vLLM commit (``-x``) on top of the series held
        in the clone and regenerate the series, which makes it the last
        ``upstream/`` entry. After resolving a cherry-pick conflict, finish
        with ``regen``.

    regen
        Regenerate the series from the clone's commits: the MUSA commits into
        ``series/``; the cherry-picks, with the MUSA fixes made on top of them,
        into ``series/upstream/``; each numbered from 0001 (``git format-patch
        --no-signature --no-numbered --zero-commit`` with ``core.abbrev=10``,
        keeping the ``index`` blob lines for the next bump's 3-way).

    report [--doc]
        Census of the manifest: a status line per divergence, or (--doc) the
        Markdown census table.

    check-series [--repo PATH] [--replay | --round-trip]
        Fail-closed form gate on series/. By default it is the cheap, offline
        half: every entry must *be* a canonical ``git format-patch`` mailbox —
        all-zero separator, ``From:`` author, RFC-2822 ``Date:``,
        ``Subject: [PATCH] `` whose slug is the one git would put in the
        filename (RFC-2047 words decoded, ``.`` runs collapsed, a trailing
        ``.``/``-`` stripped, 52 characters), an ``index`` line whenever the
        entry has a hunk, LF-terminated *structural* lines with a final newline
        (a CR that is content — an added line of a CRLF file — is legitimate),
        a diff ``git apply --stat`` can parse — the numbering must be unique and contiguous from
        ``0001`` in each directory, no two entries may share a diff body,
        ``upstream/`` must start with a ``git cherry-pick -x`` commit, and a
        MUSA entry must not be one. ``--repo``
        additionally resolves the blobs those ``index`` lines declare to
        BLOBS (a preimage may be absent only when an EARLIER entry declares it
        as its postimage; an id that resolves to a commit or tree is reported).

        The default mode proves *shape*, never replayability: it cannot know
        whether a hunk still applies, nor whether ``regen`` would rewrite the
        entry. The two deeper modes each need the checkout in a specific state
        and are mutually exclusive (asking for both is a usage error):

          * ``--replay`` (checkout AT THE PIN) runs ``git am -3`` over the
            series in a disposable clone, names the entry when a step applies
            nothing (``replay-noop``: the checkout already holds the series),
            then checks two things only a replay can settle — that every
            postimage the series declares is reachable from the commits the
            replay created (an exemption earned by an invented id, or by the
            pin's own copy of the file, is not one) and that the series also
            applies the way the build does, sequentially with
            ``git apply --recount -p1``;
          * ``--round-trip`` (checkout HOLDING the series; run ``rebase``
            first) runs exactly what ``cmd_regen`` runs against *that* checkout
            and rows any entry whose bytes or filename ``regen`` would rewrite,
            plus any count mismatch. A checkout that does not hold the series
            is reported, never passed on a guess.

        Neither deep mode is the default: both clone a repository and spawn git
        per entry. The cheap, replay, and round-trip gates also require valid advisory
        lineage/group metadata. CI runs the deep gates against separate pin
        and commit-bearing checkouts.

Every **gate** subcommand (``check-series``, ``verify``, ``regen``, ``rebase``,
``upgrade-plan``, ``backport``) that has something to report ends with one explicit
``=== musa_sync <cmd>: PASS|FAIL ===`` verdict line whose counters agree with the
exit code: 0 = PASS, 1 = FAIL, 2 = usage/config error (e.g. ``verify`` with no
resolvable target). Three exit paths print only an ``ERROR:`` line, and a caller
that parses output must read the exit code for them: a usage/config error (exit
2), a missing ``git`` on ``PATH`` (exit 1), and ``regen``'s internal checks on the
patches it generated (exit 1). The two reporting commands do not print a verdict
at all: ``report`` renders the manifest census and ``apply`` prints
``--- N applied, … ---``.
(``regen --area module`` is a mode of ``regen``, not a subcommand.)

Stdlib-only; loads manifest.py + build_apply.py BY FILE PATH so it never imports
the ``vllm_musa`` package (works before install, in plain CI).
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import re
from email.header import decode_header, make_header
import shutil
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent  # tools/ -> repo root
PATCHES = ROOT / "vllm_musa" / "patches"
SERIES_DIR = PATCHES / "series"
REMOVED_PATCHES_PATH = PATCHES / "metadata" / "removed-patches.json"
MODULE_DRIFT_DIR = (
    PATCHES / "module-drift"
)  # cat-4a drift tripwires (never build-applied)
WORKDIR = ROOT / "third_party" / "vllm"
PINS = ROOT / "third_party" / "PINS"
VLLM_URL = "https://github.com/vllm-project/vllm.git"
_ZERO_COMMIT_HEADER = (
    b"From 0000000000000000000000000000000000000000 Mon Sep 17 00:00:00 2001"
)
_CANONICAL_PATCH_AUTHOR = b"From: musa <musa@local>"
# Subcommand names in parser order. patch_validate.py uses this to tell an
# explicit subcommand from a bare flag; test_subcommands_are_wired keeps it
# honest against the real parser.
SUBCOMMANDS = (
    "apply",
    "verify",
    "check-series",
    "rebase",
    "upgrade-plan",
    "backport",
    "regen",
    "report",
)
# Series generation form (see _series_format_rows).
_MBOX_SEPARATOR = re.compile(rb"^From \S+ Mon Sep 17 00:00:00 2001$")
_INDEX_LINE = re.compile(rb"^index ([0-9a-f]{7,40})\.\.([0-9a-f]{7,40})", re.M)
_ZERO_BLOB = re.compile(rb"^0+$")
_NUMBER_PREFIX = re.compile(r"^([0-9]{4})(?:-|$)")
# Canonical-form constants (see _canonical_form_problem). _TITLE_CHARS and
# _MAX_SUBJECT_SLUG mirror `git format-patch`'s filename slug; the date pattern
# is the RFC-2822 form it writes for the commit's author date.
_UTF8_BOM = b"\xef\xbb\xbf"

#: Lines whose CRLF is a *file* defect rather than file content. Everything
#: inside a hunk may legitimately carry CR (that is what a CRLF source file's
#: diff looks like); these tokens cannot be hunk content, so a CR on one of them
#: means the patch *file* was written with CRLF — which `git apply` refuses while
#: `git am -3` (and therefore `regen`) accepts, i.e. exactly the divergence this
#: row exists to catch.
_STRUCTURAL_LINE = re.compile(
    rb"^(?:diff --git |index |old mode |new mode |deleted file mode |"
    rb"new file mode |rename from |rename to |similarity index |"
    rb"dissimilarity index |copy from |copy to |Binary files |"
    rb"GIT binary patch|\\ No newline)"
)
_TITLE_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._"
)
_MAX_SUBJECT_SLUG = 52
_TITLE_PREFIX = b"[PATCH] "
_DATE_LINE = re.compile(
    rb"^Date: (?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), [0-9]{1,2} "
    rb"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) [0-9]{4} "
    rb"[0-9]{2}:[0-9]{2}:[0-9]{2} [+-][0-9]{4}$"
)
_DIFF_START = b"diff --git "
_EMPTY_SLUG = re.compile(r"^[0-9]{4}-+\.patch$")
_HUNK_RE = re.compile(rb"^@@ ", re.M)
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
# Rows that mean "series/ cannot be gated at all": one row, no numbering claim.
_SERIES_FATAL = (
    "missing-series-dir",
    "empty-series",
    "git-unavailable",
    "repo-unusable",
)


def _declared_removed_patch_numbers() -> set[str]:
    try:
        data = json.loads(REMOVED_PATCHES_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return set()
    removed = data.get("removed", [])
    if not isinstance(removed, list) or not all(
        isinstance(number, str) and re.fullmatch(r"[0-9]{4}", number)
        for number in removed
    ):
        return set()
    return set(removed)


_REMOVED_PATCHES = _declared_removed_patch_numbers()


def _load(name: str, path: Path):
    """Load a stdlib-only helper module by file path (no vllm_musa import)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register before exec (safe by-path load)
    spec.loader.exec_module(mod)
    return mod


manifest = _load("musa_mdm_manifest", PATCHES / "manifest.py")
build_apply = _load("musa_mdm_build_apply", PATCHES / "build_apply.py")
patch_metadata = _load("musa_patch_metadata", ROOT / "tools" / "patch_metadata.py")


def _metadata_result():
    return patch_metadata.validate(PATCHES / "series", PATCHES / "metadata")


def _metadata_errors(result) -> bool:
    for error in result.errors:
        print(f"ERROR: patch metadata: {error}")
    return bool(result.errors)


def read_pin(key: str, default: str | None = None) -> str | None:
    if not PINS.is_file():
        return default
    for line in PINS.read_text().splitlines():
        line = line.strip()
        if line.startswith(key + "="):
            return line[len(key) + 1 :].split("#", 1)[0].strip()
    return default


def _default_target() -> str | None:
    """Return the exact upstream ref when available, otherwise the release tag."""
    return read_pin("VLLM_COMMIT") or read_pin("VLLM_TAG")


def _normalize_patch_author(path: Path) -> None:
    """Give generated patches the repository's canonical synthetic author."""
    lines = path.read_bytes().splitlines(keepends=True)
    if len(lines) < 2 or not lines[1].startswith(b"From: "):
        return
    author = lines[1].rstrip(b"\r\n")
    if author == _CANONICAL_PATCH_AUTHOR:
        return
    lines[1] = _CANONICAL_PATCH_AUTHOR + lines[1][len(author) :]
    path.write_bytes(b"".join(lines))


class GitMissing(RuntimeError):
    """The ``git`` executable is not on PATH — a reported failure, not a
    traceback."""


class RepoUnusable(RuntimeError):
    """The ``--repo`` handed to the series gate is not a checkout git can read
    objects from."""


def _run_git(
    cwd: Path | None, *args: str, stdin_text: str | None = None
) -> subprocess.CompletedProcess:
    """Run ``git`` with an explicit cwd.

    ``cwd=None`` keeps the caller's directory, which ``_git`` pairs with
    ``-C <repo>``; the series format gate always passes an explicit ``cwd`` so
    its verdict cannot depend on where the user invoked it from. A missing
    ``git`` raises GitMissing instead of an uncaught FileNotFoundError.
    """
    try:
        return subprocess.run(
            ["git", *args],
            cwd=None if cwd is None else str(cwd),
            input=stdin_text,
            capture_output=True,
            text=True,
            # A series entry may carry a non-UTF-8 byte in its headers; without
            # this, decoding git's own output raises UnicodeDecodeError out of
            # subprocess and the gate dies with a traceback instead of a verdict.
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise GitMissing(
            "git executable not found on PATH"
            if shutil.which("git") is None
            else str(exc)
        ) from exc


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return _run_git(None, "-C", str(repo), *args)


def _probe_upstream(clone: Path, upstream_path: str | None) -> bool:
    """cat-5/cat-6 existence probe: does the declared upstream target exist in the
    clone? (File-level for now; symbol-level is a later enhancement.)"""
    if not upstream_path:
        return True
    return (Path(clone) / upstream_path).is_file()


def _module_tripwire(clone: Path, entry) -> str | None:
    """cat-4a drift tripwire: the stable unified diff of the upstream file (in the
    clone) vs the MUSA shadow copy. Returns None if the upstream file is gone.
    difflib (no timestamps) so the stored tripwire is byte-stable across runs."""
    up = Path(clone) / (entry.upstream_path or "")
    shadow = ROOT / entry.path
    if not entry.upstream_path or not up.is_file() or not shadow.is_file():
        return None
    a = up.read_text(errors="replace").splitlines(keepends=True)
    b = shadow.read_text(errors="replace").splitlines(keepends=True)
    diff = difflib.unified_diff(a, b, fromfile="upstream", tofile="musa", n=3)
    # Keep stored .diff files friendly to git diff --check even when either
    # source contains whitespace-only or trailing-whitespace lines.
    return "".join(
        line.rstrip(" \t\r\n") + ("\n" if line.endswith(("\n", "\r")) else "")
        for line in diff
    )


def _tripwire_path(entry) -> Path:
    return MODULE_DRIFT_DIR / (entry.id + ".diff")


# --------------------------------------------------------------------------- apply
def cmd_apply(args) -> int:
    repo = Path(args.repo)
    order = manifest.series_apply_order(phase=args.phase)
    results = build_apply.apply_patch_series(
        repo, order=order, strict=not args.no_strict, check_only=args.check_only
    )
    for name, status in results:
        print(f"{status:16} {name}")
    n_conflict = sum(1 for _, s in results if s == "conflict")
    verb = "would-apply" if args.check_only else "applied"
    n_ok = sum(1 for _, s in results if s in ("applied", "would-apply"))
    n_skip = sum(1 for _, s in results if s == "already-applied")
    print(f"--- {n_ok} {verb}, {n_skip} already-applied, {n_conflict} conflict ---")
    return 1 if n_conflict else 0


# -------------------------------------------------------------------------- verify
# Map a build_apply status from the cumulative disposable copy to a verdict.
_VERIFY_STATUS = {
    "would-apply": "clean",  # applies cleanly to pristine upstream
    "applied": "clean",  # applied to the disposable cumulative checkout
    "already-applied": "obsolete",  # already in upstream -> candidate for removal
    "conflict": "conflict",  # drifted -> needs re-anchor/retire
}
_BAD = {"conflict", "missing-symbol", "missing-target", "orphaned", "drifted-copy"}


def _ensure_clone(target: str, repo_arg: str | None):
    """Return (checkout_path, is_temporary). Uses --repo if given, else a fresh
    shallow checkout of vllm@target. Fetching the ref explicitly supports both
    advertised tags and exact commit SHAs."""
    if repo_arg:
        return Path(repo_arg), False
    tmp = Path(tempfile.mkdtemp(prefix="musa-verify-"))
    clone = tmp / "vllm"
    try:
        subprocess.run(
            ["git", "init", "--quiet", str(clone)],
            check=True,
            capture_output=True,
            text=True,
        )
        _git(clone, "remote", "add", "origin", VLLM_URL).check_returncode()
        checkout_ref = _fetch_target(clone, target)
        _git(clone, "checkout", "--force", "--detach", checkout_ref).check_returncode()
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return clone, True


def _fetch_target(repo: Path, target: str) -> str:
    """Fetch ``target`` and return the local ref that resolves to it.

    Some servers reject direct fetches of an exact, unadvertised commit. In
    that case fetch the advertised history and verify that the requested commit
    became reachable before checkout.
    """
    direct = _git(repo, "fetch", "--depth", "1", "origin", target)
    if direct.returncode == 0:
        return "FETCH_HEAD"

    shallow = _git(repo, "rev-parse", "--is-shallow-repository")
    fetch_args = (
        ("fetch", "--unshallow", "origin")
        if shallow.returncode == 0 and shallow.stdout.strip() == "true"
        else ("fetch", "origin")
    )
    fallback = _git(repo, *fetch_args)
    if fallback.returncode != 0:
        raise subprocess.CalledProcessError(
            fallback.returncode, fallback.args, fallback.stdout, fallback.stderr
        )
    resolved = _git(repo, "cat-file", "-e", f"{target}^{{commit}}")
    if resolved.returncode != 0:
        raise subprocess.CalledProcessError(
            direct.returncode, direct.args, direct.stdout, direct.stderr
        )
    return target


@contextmanager
def _disposable_checkout(source: Path):
    """Yield an independent copy of ``source`` suitable for cumulative patching.

    Verification may receive a caller-owned checkout via ``--repo``. A real copy
    (including symlinks as symlinks, but excluding repository metadata) ensures
    neither its worktree nor its Git metadata can be modified while later patches
    are evaluated against the results of earlier patches.
    """
    tmp = Path(tempfile.mkdtemp(prefix="musa-verify-series-"))
    checkout = tmp / "vllm"
    try:
        shutil.copytree(
            source,
            checkout,
            symlinks=True,
            ignore=shutil.ignore_patterns(".git"),
        )
        subprocess.run(
            ["git", "init", "--quiet", str(checkout)],
            check=True,
            capture_output=True,
            text=True,
        )
        yield checkout
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _verify_rows(clone: Path) -> list[tuple]:
    rows: list[tuple] = []
    with _disposable_checkout(clone) as cumulative:
        for e in manifest.ENTRIES:
            if e.category in manifest.BUILD_APPLIED_CATEGORIES:
                patch = ROOT / e.path
                if not patch.is_file():
                    rows.append((e.id, e.category, "orphaned", "patch file missing"))
                    continue
                status = build_apply.apply_patch(cumulative, patch, check_only=False)
                rows.append((e.id, e.category, _VERIFY_STATUS.get(status, status), ""))
            elif e.category == "4a":
                # Probes intentionally use the caller's pristine checkout, not
                # the cumulative series copy whose upstream files are changing.
                cur = _module_tripwire(clone, e)
                stored = _tripwire_path(e)
                if cur is None:
                    rows.append(
                        (e.id, e.category, "missing-target", e.upstream_path or "")
                    )
                elif not stored.is_file():
                    rows.append(
                        (e.id, e.category, "no-tripwire", "run `regen --area module`")
                    )
                elif stored.read_text() == cur:
                    rows.append(
                        (e.id, e.category, "clean", "tripwire matches upstream")
                    )
                else:
                    rows.append(
                        (
                            e.id,
                            e.category,
                            "drifted-copy",
                            "upstream changed under the copy",
                        )
                    )
            elif e.category == "5":
                ok = _probe_upstream(clone, e.upstream_path)
                rows.append(
                    (
                        e.id,
                        e.category,
                        "present" if ok else "missing-symbol",
                        e.upstream_path or "",
                    )
                )
            elif e.category == "6":
                ok = _probe_upstream(clone, e.upstream_path)
                rows.append(
                    (
                        e.id,
                        e.category,
                        "present" if ok else "missing-target",
                        e.upstream_path or "",
                    )
                )
    return rows


def _upstream_dir() -> Path:
    """``series/upstream/``: the upstream vLLM commits, applied last."""
    return SERIES_DIR / manifest.UPSTREAM_SUBDIR


def _series_dir_paths() -> tuple[list[Path], tuple[str, str] | None]:
    """The series entries in apply order (``upstream/`` last), or a
    ``(status, detail)`` refusal when there is nothing to gate.

    A missing or empty ``series/`` is a *failure*, never a vacuous pass: a gate
    that globs zero entries reports "0 clean / 0 total / 0 need attention" and
    exits 0, which is how a deleted series directory stays green.
    """
    if not SERIES_DIR.is_dir():
        return [], ("missing-series-dir", f"no series directory at {SERIES_DIR}")
    paths = sorted(SERIES_DIR.glob("*.patch")) + sorted(_upstream_dir().glob("*.patch"))
    if not paths:
        return [], ("empty-series", f"{SERIES_DIR} holds no *.patch entries")
    return paths, None


def _safe_name(name: str) -> str:
    """``name`` with control characters escaped, safe to print on one line.

    ``series/*.patch`` is globbed, not curated, so an entry filename can carry a
    newline — which would otherwise inject forged lines (a fake verdict, say)
    into the report of any row that prints the name.
    """
    return _CONTROL_CHARS.sub(lambda m: f"\\x{ord(m.group()):02x}", name)


def _file_kind(mode: int) -> str:
    """The non-regular kind a ``lstat`` mode describes."""
    for test, name in (
        (stat.S_ISLNK, "a symlink"),
        (stat.S_ISDIR, "a directory"),
        (stat.S_ISSOCK, "a socket"),
        (stat.S_ISFIFO, "a fifo"),
    ):
        if test(mode):
            return name
    return "a special file"


def _entry_name_problem(name: str) -> tuple[str, str] | None:
    """The ``series/`` filename's own problem, or None when it is well formed."""
    if _CONTROL_CHARS.search(name):
        return (
            "entry-name-hostile",
            "control character in the filename: the report would splice a "
            "forged line into its own verdict",
        )
    if _EMPTY_SLUG.match(name):
        return (
            "bad-name",
            "empty slug: `regen` names an entry <NNNN>-<subject-slug>.patch, so "
            "it would rewrite this filename",
        )
    return None


def _read_entry(patch: Path, texts: dict[str, bytes]) -> tuple[str, str] | None:
    """Read one entry into ``texts``; a ``(status, detail)`` row when it cannot
    be read.

    Everything here is a row, never a traceback: an entry file may be chmod-000
    (``PermissionError``), a dangling symlink (``FileNotFoundError``), a
    directory (``IsADirectoryError``) or a socket — and ``manifest.py`` reads
    the same files at import time, before any subcommand gets to print a
    verdict. A symlink is refused outright: ``regen`` writes the entries, and a
    symlink would make it write *through* the link, outside ``series/``.
    """
    try:
        mode = patch.lstat().st_mode
    except OSError as exc:
        return ("unreadable", f"{exc.strerror or exc}: cannot stat the entry")
    if not stat.S_ISREG(mode):
        return (
            "not-regular-file",
            f"{_file_kind(mode)} is not a regular file: entries are real patch "
            "files, and a symlink would make `regen` write outside series/",
        )
    try:
        texts[build_apply.entry_name(patch)] = patch.read_bytes()
    except OSError as exc:
        return ("unreadable", str(exc))
    return None


def _header_lines(text: bytes) -> list[bytes]:
    """The mailbox header lines (after the ``From `` separator, before the
    first empty line)."""
    out: list[bytes] = []
    for line in text.split(b"\n")[1:]:
        if not line.strip(b"\r"):
            break
        out.append(line)
    return out


def _header_value(header: list[bytes], field: bytes) -> bytes | None:
    """The RFC-2822 unfolded value of ``field`` in a mailbox header, or None.

    Long subjects are folded across lines at a space; unfolding concatenates
    the continuation lines, whose leading space restores the original space.
    """
    value: bytes | None = None
    for line in header:
        if value is not None:
            if line.startswith((b" ", b"\t")):
                value += line
                continue
            break
        if line.startswith(field + b": "):
            value = line[len(field) + 2 :]
    return value


def _decode_header_word(value: bytes) -> str:
    """An RFC-2822 header value with any RFC-2047 encoded words decoded.

    ``git format-patch`` writes a non-ASCII subject as an encoded word
    (``=?UTF-8?q?...?=``) in the header while slugging the *decoded* text into
    the filename, so comparing the raw header against the filename rejects
    every entry with a non-ASCII subject — including ones already in this
    repository's history.
    """
    text = value.decode("utf-8", "replace")
    if "=?" not in text:
        return text
    try:
        return str(make_header(decode_header(text)))
    except (ValueError, LookupError, UnicodeError):
        return text


def _git_slug(subject: str) -> str:
    """The filename slug ``git format-patch`` derives from ``subject``.

    Empirically pinned against real ``git format-patch`` output (see
    ``test_git_slug_matches_format_patch``): characters outside
    ``[A-Za-z0-9._]`` become ``-``, runs of ``-`` collapse, runs of ``.``
    collapse to one, trailing ``-`` and ``.`` are stripped, then the slug is
    truncated to ``_MAX_SUBJECT_SLUG`` — the shipped series' longest slug is
    exactly 52 characters. ``regen`` renames an entry to this slug, so a
    filename that does not match is not a fixed point.
    """
    out: list[str] = []
    for ch in subject:
        if ch in _TITLE_CHARS:
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    slug = "".join(out)
    slug = re.sub(r"\.{2,}", ".", slug)
    return slug.rstrip("-.")[:_MAX_SUBJECT_SLUG]


def _whole_entry_crlf(text: bytes) -> str | None:
    """The whole-patch CRLF form, or None.

    Measured on this checkout, per line: a patch whose *every* line ends in CR
    is refused by ``git apply --recount -p1`` (the build path) while ``git am
    -3`` accepts it — a real build/replay divergence. A CR on a single
    structural line, and a CR that is *content* (an added line of a CRLF file,
    which is what ``format-patch`` writes for such a commit), are both accepted
    by *both* paths, so neither is reported here; ``regen`` writes LF, so
    ``--round-trip`` still rewrites the former, which is the mode documented to
    prove the fixed point.
    """
    lines = [line for line in text.split(b"\n") if line]
    if not lines or not all(line.endswith(b"\r") for line in lines):
        return None
    return lines[0][:40].decode("utf-8", "replace")


def _mailbox_form_problem(text: bytes) -> tuple[str, str] | None:
    """The cheap file-level canonical-form problem, or None.

    Checked before anything parses the entry: a BOM hides the separator
    entirely, and CRLF is the nastiest case in the series — the build applier
    (``git apply --recount``) rejects a CRLF diff while ``git am -3`` replays it
    happily, so only the gate can see the divergence before a build breaks.
    """
    if text.startswith(_UTF8_BOM):
        return (
            "non-canonical-bom",
            "UTF-8 BOM before the mailbox separator: `git format-patch` never "
            "writes one",
        )
    crlf = _whole_entry_crlf(text)
    if crlf is not None:
        return (
            "non-canonical-crlf",
            f"every line of this entry ends in CRLF ({crlf}): `regen` writes "
            "LF, and such an entry is refused by `git apply --recount -p1` "
            "(the build path) while `git am -3` accepts it — a build/replay "
            "divergence. A CR on a single line, or a CR that is *content* (an "
            "added line of a CRLF file), is accepted by both paths and is not "
            "reported.",
        )
    if not text:
        return (
            "empty-entry",
            "0 bytes: `git format-patch` writes nothing for a commit with no "
            "changes (an `--allow-empty` commit in the range), so `regen` "
            "cannot produce a series that replays this history",
        )
    if not text.endswith(b"\n"):
        return (
            "non-canonical-eof",
            "no final newline: `regen` ends every entry with one",
        )
    return None


def _duplicate_bodies(texts: dict[str, bytes]) -> dict[str, str]:
    """Entry name -> the earlier entry whose diff body it duplicates.

    Two entries with the same body (a copy-paste, or a whole file copied under
    the next number) apply twice and only fall out at the next ``regen``, which
    emits one entry per commit. The body starts at its first ``diff --git`` so
    mailing-header edits cannot hide the duplication.
    """
    seen: dict[bytes, str] = {}
    out: dict[str, str] = {}
    for name, text in texts.items():
        start = text.find(_DIFF_START)
        if start < 0:  # no diff at all: `_series_entry_problem` owns that
            continue
        body = text[start:]
        if body in seen:
            out[name] = seen[body]
        else:
            seen[body] = name
    return out


def _canonical_form_problem(
    text: bytes, name: str, duplicates: dict[str, str]
) -> tuple[str, str] | None:
    """The first canonical ``regen`` fixed-point problem in one entry, or None.

    These are the *cheap* half of the fixed point: they prove the entry is
    shaped like ``regen`` output, never that ``regen`` would leave it byte-for
    -byte alone (only ``check-series --round-trip`` can). Deliberately not
    checked offline — they need the clone's commits: an extra ``Signed-off-by``,
    a deleted ``---`` diffstat, an edited Date that is still a valid date, a
    changed author ident.
    """
    header = _header_lines(text)
    date = _header_value(header, b"Date")
    if date is None:
        return (
            "non-canonical-date",
            "no `Date:` header: `regen` writes the commit's author date",
        )
    if not _DATE_LINE.match(b"Date: " + date):
        return (
            "non-canonical-date",
            f"`Date:` is not the RFC-2822 form `git format-patch` writes "
            f"({date.decode('utf-8', 'replace')[:40]!r})",
        )
    subject = _header_value(header, b"Subject")
    if subject is None:
        return ("non-canonical-subject", "no `Subject:` header")
    if not subject.startswith(_TITLE_PREFIX):
        if subject.startswith(b"[PATCH "):
            return (
                "non-canonical-subject",
                "numbered subject: `regen` runs `git format-patch "
                "--no-numbered`, which rewrites it",
            )
        return (
            "non-canonical-subject",
            "no `[PATCH] ` subject prefix: `regen` writes one",
        )
    slug = _git_slug(_decode_header_word(subject[len(_TITLE_PREFIX) :]))
    stem = name[: -len(".patch")]
    if "-" in stem:
        number, _, filename_slug = stem.partition("-")
        if slug != filename_slug:
            return (
                "non-canonical-subject",
                f"subject slug {slug!r} does not match the filename slug "
                f"{filename_slug!r}: `regen` renames the entry to "
                f"{number}-{slug}.patch",
            )
    if name in duplicates:
        return (
            "non-canonical-duplicate",
            f"diff body is byte-identical to {duplicates[name]}: `regen` emits "
            "one entry per commit",
        )
    return None


# A file-mode-only submodule entry: `format-patch` writes the mode on a
# `new file mode`/`deleted file mode`/`old mode`/`new mode` line for an addition,
# a removal or a type change, and in the mode field of the `index` line for a bump.
_GITLINK = re.compile(
    rb"(?m)^(?:new file mode|deleted file mode|old mode|new mode) 160000\n"
    rb"|^index [0-9a-f]+\.\.[0-9a-f]+ 160000\n"
)


_DIFF_HEADER_LINE = re.compile(
    rb"^(index |new file mode |deleted file mode |old mode |new mode "
    rb"|similarity index |dissimilarity index |rename from |rename to "
    rb"|copy from |copy to )"
)


def _header_run_start(text: bytes, pos: int) -> int:
    """Start of the diff-header run directly above ``pos`` (at most one `index`).

    A bare diff still carries its header lines — ``index``, modes, rename/copy —
    and the ``index`` line sits above the ``--- a/…`` pair the caller found. At
    most ONE ``index`` line is swallowed: `git diff` writes one per file section,
    and a commit message that quotes an ``index deadbeef..cafebabe`` line directly
    above the diff (no blank line between them) is prose, which this must not read
    as an anchor — that promise is why the slice exists. The residual ambiguity is
    irreducible from bytes alone: a bare diff whose only anchor line is such a
    quoted line reads as prose and is rowed ``no-index-line`` (a shape complaint;
    ``--replay`` still says whether `git am` can replay it).
    """
    start = pos
    seen_index = False
    while start > 0:
        line_start = text.rfind(b"\n", 0, start - 1) + 1
        if line_start >= start or not _DIFF_HEADER_LINE.match(text[line_start:start]):
            break
        if text.startswith(b"index ", line_start):
            if seen_index:
                break
            seen_index = True
        start = line_start
    return start


def _diff_body(text: bytes) -> bytes:
    """The entry's diff, from its first ``diff --git`` or ``--- a/… +++ b/…`` pair.

    The whole *diff* is searched, not just the mailbox header: a commit message
    with a body pushes the first ``index`` line far below the separator (line 22
    in the longest shipped entry), so a first-12-lines scan reads clean on
    entries that have no ``index`` line at all. What is excluded is the prose
    *above* the diff: a commit message that quotes an ``index deadbeef..cafebabe``
    line, or the words ``@@``, is text, and reading it as an anchor asked authors
    to avoid quoting their own upstream.

    A **bare diff** has no ``diff --git`` line, and returning ``b""`` for it hid
    its hunks from every caller: an entry that is a bare diff with a real hunk and
    no ``index`` line passed all four modes, which is the shape this gate exists to
    catch. The fallback therefore accepts the ``--- a/…`` + ``+++ b/…`` header pair
    a real diff carries; a commit message that merely mentions a path has no such
    adjacent pair.
    """
    at = text.find(b"diff --git ")
    if at >= 0:
        return text[at:]
    m = re.search(rb"(?m)^--- [^\n]*\n\+\+\+ [^\n]*\n", text)
    if m is None:
        return b""
    return text[_header_run_start(text, m.start()) :]


def _target_resolves(repo: Path | None, target: str) -> bool:
    """Whether ``target`` resolves in ``repo`` (or can be fetched into a clone).

    ``verify`` used to report a green run for a target that the checkout it was
    handed does not contain: `--repo` was accepted and the divergence rows were
    computed against whatever that checkout had, so a mistyped or stale pin read
    as PASS. Without a repo the fresh clone is fetched first, so nothing can be
    decided here and the answer is "assume yes".
    """
    if repo is None:
        return True
    r = _git(Path(repo), "rev-parse", "--verify", "--quiet", f"{target}^{{commit}}")
    return r.returncode == 0


def _unquote_git_path(raw: bytes) -> str:
    """Undo git's C-quoting of a path (``core.quotePath`` defaults to true).

    ``git diff``/``format-patch`` write a path containing a non-ASCII byte, a
    quote or a backslash as a C string — ``"a/caf\\303\\251.txt"`` — and a parser
    that reads the quoted form as the literal path misses the section entirely.
    That is how a hand-edited ``index`` line in such a file escaped the postimage
    check (measured: the entry replayed, the exemption stayed unverified).
    """
    if not (raw.startswith(b'"') and raw.endswith(b'"')):
        return raw.decode("utf-8", "replace")
    body, out, i = raw[1:-1], bytearray(), 0
    while i < len(body):
        ch = body[i]
        if ch == 0x5C and i + 1 < len(body):
            nxt = body[i + 1]
            if 0x30 <= nxt <= 0x37:  # up to three octal digits
                digits = b""
                j = i + 1
                while j < len(body) and len(digits) < 3 and 0x30 <= body[j] <= 0x37:
                    digits += bytes([body[j]])
                    j += 1
                out.append(int(digits, 8) & 0xFF)
                i = j
                continue
            out.append(
                {0x61: 0x07, 0x62: 0x08, 0x74: 0x09, 0x6E: 0x0A, 0x76: 0x0B,
                 0x66: 0x0C, 0x72: 0x0D, 0x22: 0x22, 0x5C: 0x5C}.get(nxt, nxt)
            )
            i += 2
            continue
        out.append(ch)
        i += 1
    return out.decode("utf-8", "replace")


def _diff_sections(body: bytes) -> list[bytes]:
    """Split a diff body into per-file sections.

    A `diff --git` body is one section per file. A **bare** diff has no such line,
    so several files can follow one another — each with its own `--- a/…` +
    `+++ b/…` pair — and each section starts at the header run *above* its pair
    (`index`, modes, rename/copy), not at the pair itself: splitting at the pair
    would leave the `index` line in the previous section and attribute it, and its
    postimage, to the previous file's path.
    """
    sections: list[bytes] = []
    for chunk in re.split(rb"(?m)^(?=diff --git )", body):
        if not chunk.strip():
            continue
        if re.search(rb"(?m)^diff --git ", chunk) is None:
            pairs = list(re.finditer(rb"(?m)^--- [^\n]*\n\+\+\+ [^\n]*\n", chunk))
            starts = sorted({_header_run_start(chunk, m.start()) for m in pairs})
            for i, start in enumerate(starts):
                end = starts[i + 1] if i + 1 < len(starts) else len(chunk)
                if chunk[start:end].strip():
                    sections.append(chunk[start:end])
        else:
            sections.append(chunk)
    return sections


def _diff_section_path(section: bytes) -> str | None:
    """The ``b`` side path of one diff section, or None for a deletion."""
    token = rb'"(?:[^"\\]|\\.)*"|\S+'
    head = re.search(rb"(?m)^diff --git (" + token + rb") (" + token + rb")", section)
    if head:
        raw = head.group(2)
    else:
        plus = re.search(
            rb"(?m)^\+\+\+ (" + token + rb")(?:\t[^\n]*)?\n", section
        )
        minus = re.search(
            rb"(?m)^--- (" + token + rb")(?:\t[^\n]*)?\n", section
        )
        if plus is None or minus is None:
            return None
        raw = plus.group(1)
        if _unquote_git_path(raw).rstrip("\r") == "/dev/null":
            raw = minus.group(1)  # a deletion: the declared path is the a side
    path = _unquote_git_path(raw).rstrip("\r")
    for prefix in ("b/", "a/"):
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def _declared_postimages(text: bytes) -> list[tuple[str, str]]:
    """``(path, postimage)`` for every ``index`` line, in diff order.

    The path comes from the section's ``diff --git a/… b/…`` header, or from its
    ``+++ b/…`` line for a bare diff; git's C-quoting is undone, and a trailing
    TAB timestamp is dropped. This is what lets the replay check an entry's
    declaration against *the commit that entry created*.

    Two shapes that used to be skipped are now read, because skipping them was a
    way to fake an exemption:

    * a **deletion** (``+++ /dev/null``) whose ``index`` line names a *non-null*
      postimage — no commit can hold a blob at a path it deleted, so the check
      must see it and say so (a real deletion declares the all-zero id and is
      skipped by the caller);
    * a **gitlink** (``new file mode 160000``) whose ``index`` line names a
      submodule *commit*, which is not a blob at all and cannot be earned — such
      a section is skipped here, because rowing it would reject a legitimate
      submodule bump (it is the one shape where the declaration is not a blob).
    """
    body = _diff_body(text)
    out: list[tuple[str, str]] = []
    for section in _diff_sections(body):
        if _GITLINK.search(section):
            continue  # a gitlink: the "postimage" is a commit, not a blob
        path = _diff_section_path(section)
        if path is None:
            continue
        for m in _INDEX_LINE.finditer(section):
            out.append((path, m.group(2).decode()))
    return out


def _declared_blobs(text: bytes) -> list[tuple[str, str]]:
    """``(preimage, postimage)`` blob ids from every ``index`` line in the diff.

    Gitlink sections are skipped: their ids are *commits* in the submodule's object
    database, so neither "the anchor must resolve to a blob here" nor "the entry
    must be able to produce this postimage here" has a meaning for them. Both
    directions are false FAILs on a legitimate submodule add, bump or removal —
    measured on real `format-patch` output for each of the three shapes.
    """
    body = _diff_body(text)
    out: list[tuple[str, str]] = []
    for section in _diff_sections(body):
        if _GITLINK.search(section):
            continue
        for m in _INDEX_LINE.finditer(section):
            out.append((m.group(1).decode(), m.group(2).decode()))
    return out


def _require_usable_repo(repo: Path) -> None:
    """Raise unless ``repo`` is a git repository with a resolvable HEAD.

    ``/dev/null`` and a file that is not a directory raise ``OSError`` from the
    subprocess itself; that is the same answer as "not a repository", so it gets
    the same row rather than a different one.
    """
    try:
        r = _run_git(repo, "rev-parse", "--git-dir")
        if r.returncode:
            raise RepoUnusable(f"{repo} is not a git repository")
        r = _run_git(repo, "rev-parse", "--verify", "HEAD^{commit}")
    except (OSError, GitMissing) as exc:
        # `/dev/null`, a file, an empty directory and a typo'd path all land
        # here: a repository that cannot be opened is the same verdict as one
        # that is not a repository, and it must never pass silently.
        raise RepoUnusable(f"{repo} cannot be read: {exc}") from exc
    if r.returncode:
        raise RepoUnusable(f"{repo} has no resolvable HEAD")


def _repo_object_types(repo: Path, blobs: list[str]) -> dict[str, str]:
    """``blob id -> object type`` for every id, ``"missing"`` when absent.

    One ``git cat-file --batch-check`` for the whole series rather than one
    process per blob: the shipped series declares 355 distinct ids, which would
    otherwise cost ~1.4 s of process spawns. The *type* matters: an ``index``
    line anchors to a **blob**, so a commit or tree id that happens to resolve
    is still an unusable anchor, and "the object exists" alone would let it
    through.
    """
    if not blobs:
        return {}
    r = _run_git(
        None,
        "-C",
        str(repo),
        "cat-file",
        "--batch-check",
        stdin_text="\n".join(blobs) + "\n",
    )
    if r.returncode:
        raise RepoUnusable(
            f"git cannot read objects from {repo}: "
            f"{(r.stderr or r.stdout).strip()[:100]}"
        )
    out: dict[str, str] = {}
    for blob, line in zip(blobs, r.stdout.splitlines()):
        fields = line.split()
        # `<id> <type> <size>`, or `<id> missing` for an absent object.
        out[blob] = fields[1] if len(fields) >= 3 and fields[1] != "missing" else "missing"
    return out


def _missing_index_blobs(
    repo: Path, texts: dict[str, bytes], order: list[str]
) -> dict[str, list[str]]:
    """Entry name -> declared anchor ids this repo cannot use, in series order.

    An entry's ``index`` line records the blobs its hunks are anchored to, and
    ``git am -3`` uses them to build a real 3-way ancestor — ids left behind by
    a series regenerated against some *other* base make the entry replay as a
    plain apply or not at all.

    A blob the series itself creates is *expected* to be absent from a checkout
    that never had the series applied: only ``rebase``/``regen`` put those
    objects in the odb. Measured on the shipped 172-entry series, a fresh clone
    at the pin resolves 133/232 preimages and 0/232 postimages, yet ``git am -3``
    replays all 172 of them — so "missing from this repo" alone is not a defect,
    while "missing and not produced by an entry that already ran" is.

    Two rules keep that exemption honest, both enforced here:

      * only **preimages** are exempt — a postimage is what the entry itself
        creates, so requiring it to exist would be a demand on the future;
      * only postimages declared by entries that **precede** this one in
        ``order`` (the series' filename order) exempt it. A postimage the entry
        declares itself, or one a *later* entry produces, is not yet in the odb
        when this entry replays, so it cannot justify an absent anchor;
      * an exempt id that resolves here to something that is **not a blob**
        (a commit or tree id) is not exempt: no replay can produce it as a
        postimage, so it stays a reported anchor.

    The exemption is still only as strong as the earlier entry's *declaration*:
    an id that resolves to nothing is deferred, not proven, and only
    ``check-series --replay`` settles whether the series produces it. Run the
    replay on any series whose anchors were edited by hand.

    Null ids (``0000000000``, a creation hunk) are never resolvable and never
    required.
    """
    zero = lambda blob: bool(_ZERO_BLOB.match(blob.encode()))  # noqa: E731
    needed: dict[str, set[str]] = {}
    produced: set[str] = set()
    exempt_declared: set[str] = set()
    for name in order:
        text = texts.get(name)
        if text is None:  # already rowed as not-regular-file / unreadable
            continue
        pairs = _declared_blobs(text)
        exempt_declared.update(
            pre for pre, _post in pairs if not zero(pre) and pre in produced
        )
        wanted = {
            pre for pre, _post in pairs if not zero(pre) and pre not in produced
        }
        if wanted:
            needed[name] = wanted
        produced.update(post for _, post in pairs if not zero(post))
    if repo is not None:
        # A `--repo` that cannot answer must fail closed *here*, before the
        # early return below: a series that only creates or renames files
        # declares no non-zero preimage, so the old code returned a clean sheet
        # for a typo'd path, a not-yet-cloned checkout, /dev/null, or an empty
        # repository — indistinguishable from a good one, which is exactly what
        # the docs promise not to happen.
        _require_usable_repo(repo)
    if not needed and not exempt_declared:
        return {}
    queried = sorted(
        {blob for wanted in needed.values() for blob in wanted} | exempt_declared
    )
    types = _repo_object_types(repo, queried)
    # An exempt id is exempt because an *earlier entry declares* it as its
    # postimage — a declaration, not proof: the series may never produce it (a
    # hand-edited ``index`` line), and the id may not even be a blob id. Where
    # the id resolves here, demand a blob now; the ids that resolve to nothing
    # are exactly the ones only a real replay can settle, which is what
    # ``--replay`` does.
    def unusable_exempt(blob: str) -> bool:
        return types.get(blob) not in (None, "blob", "missing")

    for name in order:
        text = texts.get(name)
        if text is None:
            continue
        bad = {
            pre
            for pre, _post in _declared_blobs(text)
            if not zero(pre) and unusable_exempt(pre)
        }
        if bad:
            needed.setdefault(name, set()).update(bad)
    out: dict[str, list[str]] = {}
    for name, wanted in needed.items():
        bad = []
        for blob in sorted(wanted):
            kind = types.get(blob, "missing")
            if kind == "blob":
                continue
            bad.append(
                f"{blob} (missing from the repo)"
                if kind == "missing"
                else f"{blob} (a {kind} object, not a blob)"
            )
        if bad:
            out[name] = bad
    return out


def _series_entry_problem(
    text: bytes, unresolved: list[str]
) -> tuple[str, str] | None:
    """The first generation-form problem in one entry, or None when it is a
    canonical ``regen`` fixed point."""
    lines = [line.rstrip(b"\r") for line in text.split(b"\n")[:4]]
    first = lines[0] if lines else b""
    if first != _ZERO_COMMIT_HEADER and _MBOX_SEPARATOR.match(first):
        return (
            "non-canonical-mailbox",
            "valid mbox but not the all-zero `regen` separator: re-run `regen` "
            "to canonicalise it",
        )
    if first != _ZERO_COMMIT_HEADER:
        return (
            "not-a-mailbox",
            "no 'From <token> Mon Sep 17 00:00:00 2001' separator: git am "
            "cannot replay it",
        )
    # git am takes the author ident from the mailbox body: without a non-empty
    # `From: ` in the first 3 lines it dies on 'empty ident name (for <>)'.
    if not any(
        line.startswith(b"From: ") and line[len(b"From: ") :].strip()
        for line in lines[1:4]
    ):
        return (
            "missing-author",
            "no 'From: <name> <email>' in the first 3 lines: git am rejects it "
            "(empty ident name)",
        )
    # Only a hunk needs the index: `git am -3` builds its 3-way ancestor from
    # it and dies with "sha1 information is lacking or useless" without one.
    # A rename, a mode-only change and a binary blob all carry no hunk and no
    # index line — real `git format-patch` output that both paths apply
    # (`git am -3` and `git apply --recount -p1`), so requiring one there would
    # reject a legitimate entry.
    if _HUNK_RE.search(_diff_body(text)) and not _INDEX_LINE.search(_diff_body(text)):
        return (
            "no-index-line",
            "no 'index <blob>..<blob>' line: `git format-patch` writes one for "
            "every hunk, and without it `git am -3` has no 3-way ancestor to fall "
            "back on (the entry still applies while its context is unchanged; "
            "`--replay` says whether it does)",
        )
    if unresolved:
        return (
            "missing-index-blob",
            f"index anchor(s) {', '.join(unresolved)} cannot be used in the "
            "given repo and are not produced by an entry that precedes this one "
            "there: the entry cannot replay against that pin",
        )
    return None


def _series_format_rows(
    repo: Path | None = None, *, replay: bool = False, round_trip: bool = False
) -> list[tuple[str, str, str]]:
    """``(patch name, status, detail)`` for every entry in ``series/``.

    The build applies the series with ``git apply --recount -p1``
    (``build_apply.py``), which silently repairs wrong hunk counts and accepts
    bare diffs — but the *generation* path (``rebase``/``regen``) needs real
    mailboxes that ``git am`` can consume. An entry that is hand-edited or a
    bare diff therefore passes every build-side gate and only breaks the next
    version bump, which is exactly how the series rotted before MUSA-100050.

    Cheap offline checks, first problem wins:
      * the filename is a plain regular file (never a symlink, directory or
        socket) with no control characters in it and a non-empty slug;
      * it reads at all (chmod-000 / dangling symlink are rows, not crashes);
      * it is LF-only, BOM-free and ends in a newline;
      * it is a ``git format-patch`` mailbox with the all-zero separator
        ``regen`` writes (a real-sha mbox is replayable but not a fixed point);
      * it carries a ``From: `` author ident, without which ``git am`` dies on
        ``empty ident name``;
      * it carries at least one ``index <blob>..<blob>`` line (searched over the
        whole entry, not just the header), without which ``git am -3`` dies on
        ``sha1 information is lacking or useless``;
      * with ``repo``, every non-null anchor those ``index`` lines declare
        resolves to a **blob** there, or is a postimage an EARLIER entry
        produces (``_missing_index_blobs``);
      * it has the canonical ``Date:`` and ``Subject: [PATCH] `` whose slug
        matches the filename, and no other entry repeats its diff body
        (``_canonical_form_problem``);
      * ``git apply --stat`` parses it (catches hunk counts that no longer
        match the body);
      * the series' numbering is unique and contiguous
        (``_series_numbering_rows``).

    What this *cannot* prove, however green it is: that the hunks still apply
    to the pin, and that ``regen`` would leave the entry byte-for-byte alone.
    ``--replay``/``round_trip`` ask git itself, in a disposable clone of
    ``repo`` (the caller's checkout is never opened for writing):

      * ``replay``: ``git am -3`` every entry in order **in a disposable clone
        of** ``repo`` and report the first one git refuses, with git's own first
        error line;
      * ``round_trip``: run ``git format-patch`` (what ``regen`` runs) in the
        checkout you passed — it needs one that *holds* the series — and report
        every entry whose bytes or filename ``regen`` would change, or the single
        count/state row when the checkout holds a different history instead.

    ``repo`` is optional so the cheap half stays usable where cloning is not
    (a PR runner with no checkout): without it the blob half is skipped.
    """
    paths, refusal = _series_dir_paths()
    if refusal:
        return [(str(SERIES_DIR), refusal[0], refusal[1])]
    status: dict[str, tuple[str, str]] = {}
    texts: dict[str, bytes] = {}
    names = {patch: build_apply.entry_name(patch) for patch in paths}
    first_upstream = next((patch for patch in paths if _is_upstream_entry(patch)), None)
    for patch in paths:
        problem = _entry_name_problem(patch.name) or _read_entry(patch, texts)
        if problem is not None:
            status[names[patch]] = problem
    order = [names[patch] for patch in paths]
    try:
        unresolved = _missing_index_blobs(Path(repo), texts, order) if repo else {}
        duplicates = _duplicate_bodies(texts)
        for patch in paths:
            name = names[patch]
            if name in status:
                continue
            text = texts[name]
            problem = (
                _mailbox_form_problem(text)
                or _series_entry_problem(text, unresolved.get(name, []))
                or _canonical_form_problem(text, name, duplicates)
                or _group_problem(patch, text, patch == first_upstream)
            )
            if problem is None:
                parsed = _run_git(ROOT, "apply", "--stat", str(patch))
                if parsed.returncode:
                    problem = ("corrupt", (parsed.stderr or parsed.stdout).strip()[:100])
            status[name] = problem or ("clean", "")
        rows = [(_safe_name(names[patch]), *status[names[patch]]) for patch in paths]
        if repo and replay:
            rows = _merge_replay_rows(
                rows, _replay_rows(Path(repo), paths, texts)
            )
        elif repo and round_trip:
            # Not a replay: `--round-trip` asks whether `regen` reproduces the
            # series from a checkout that already HOLDS it (see
            # `_round_trip_rows`). Replaying here would compare the series with
            # itself — and a checkout that holds the series cannot be replayed
            # into anyway: the entries are already applied.
            rows = _merge_replay_rows(
                rows, _round_trip_rows(Path(repo), paths)
            )
    except GitMissing as exc:
        return [(str(SERIES_DIR), "git-unavailable", str(exc))]
    except RepoUnusable as exc:
        return [(str(SERIES_DIR), "repo-unusable", str(exc))]
    return rows


#: git's generic failure text. `git am` prints it *before* the line that says
#: what actually went wrong, so taking the first `error:` line blind reports the
#: symptom and hides the cause.
_GENERIC_GIT_ERRORS = (
    "error: Failed to merge in the changes.",
    "error: could not build fake ancestor",
)


def _git_first_error(r: subprocess.CompletedProcess) -> str:
    """git's own most informative error line from a failed invocation."""
    lines = [
        line.strip()
        for line in f"{r.stdout or ''}\n{r.stderr or ''}".splitlines()
        if line.strip()
    ]
    errors = [
        line
        for line in lines
        if line.startswith(("error:", "fatal:")) and line not in _GENERIC_GIT_ERRORS
    ]
    if errors:
        return errors[0][:200]
    for line in lines:
        if line.startswith(("error:", "fatal:")):
            return line[:200]
    return lines[0][:200] if lines else "git failed without a message"


def _first_diff_line(want: bytes, got: bytes) -> str:
    """The first difference between two byte blobs, and how many there are.

    The count matters: a non-canonical ``From:`` line is the first difference
    *and* exactly one line long, while a rewritten hunk behind it would be
    invisible if only the first difference were reported.
    """
    a, b = want.splitlines(), got.splitlines()
    differing = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return (
                f"line {i + 1} of {differing} differing: the entry has "
                f"{x[:60]!r}, regen writes {y[:60]!r}"
            )
    return f"{len(a)} lines in the entry vs {len(b)} from regen"


def _series_base(repo: Path, count: int) -> str | None:
    """The ref ``regen`` would diff from, when ``repo`` holds the series.

    ``cmd_regen`` diffs ``VLLM_COMMIT``/``VLLM_TAG`` from ``third_party/PINS``
    against the worktree's HEAD, so the pin is the first candidate. A checkout
    that holds some other pin is accepted when it is exactly ``count`` commits
    ahead of its own ``HEAD~count`` — the shape the series implies — so the
    check can also run against a scratch clone; anything else yields None and
    the caller reports that this repository cannot answer the question.
    """
    target = _default_target()
    if target:
        r = _git(repo, "rev-parse", "--verify", "--quiet", f"{target}^{{commit}}")
        if r.returncode == 0:
            return target
    if count:
        r = _git(repo, "rev-parse", "--verify", "--quiet", f"HEAD~{count}^{{commit}}")
        if r.returncode == 0:
            return f"HEAD~{count}"
    return None


def _round_trip_rows(
    repo: Path, paths: list[Path]
) -> list[tuple[str, str, str]]:
    """Entries ``regen`` would rewrite, diffed against the *caller's* repo.

    Exactly what ``cmd_regen`` runs (``_format_series``: ``git format-patch
    --no-signature --no-numbered --zero-commit`` per directory plus the
    canonical author rewrite), so the
    comparison catches everything the cheap checks must not guess at: an extra
    ``Signed-off-by``, a deleted ``---`` diffstat, an edited-but-valid ``Date:``,
    a renamed slug, a renumbered subject. Entries pair up by position, which is
    how ``regen`` numbers them (one entry per commit, ``0001..NNNN``).

    The diff runs in ``repo`` — the checkout the caller passed — and not in the
    disposable replay clone. Format-patching the clone's own ``git am`` commits
    would compare the series with itself and call any self-consistent series a
    fixed point, including one whose hunks or ``index`` postimages were edited
    by hand: the clone holds the series, the checkout holds the history the
    series is supposed to describe.
    """
    tmp = Path(tempfile.mkdtemp(prefix="musa-gate-roundtrip-"))
    try:
        return _round_trip_rows_in(repo, tmp, paths)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _entry_subject_slug(text: bytes) -> str:
    """The slug `git format-patch` would name this entry by (``""`` if none)."""
    subject = _header_value(_header_lines(text), b"Subject")
    if subject is None or not subject.startswith(_TITLE_PREFIX):
        return ""
    return _git_slug(_decode_header_word(subject[len(_TITLE_PREFIX) :]))


def _round_trip_rows_in(
    repo: Path, tmp: Path, paths: list[Path]
) -> list[tuple[str, str, str]]:
    """``_round_trip_rows`` with an explicit scratch directory."""
    base = _series_base(repo, len(paths))
    if base is None:
        return [
            (
                str(SERIES_DIR),
                "round-trip-unverifiable",
                f"{repo} does not hold this series (no such pin, and HEAD is not "
                f"{len(paths)} commits ahead), so `regen` has nothing to compare "
                "against: run `musa_sync rebase <pin>` first, or drop "
                "--round-trip",
            )
        ]
    staged = tmp / "staged"
    staged.mkdir()
    generated, error = _format_series(repo, base, staged)
    if error:
        return [("series replay", "round-trip-dirty", error)]
    if not generated:
        # `regen` produces nothing when the checkout has no commits above the
        # base — i.e. `--repo` is at the pin, a state this mode cannot check.
        # Report that, rather than the 300-row cascade of "regen produces no
        # entry with this name" that a per-entry comparison emits for a series
        # the checkout does not hold.
        return [
            (
                str(SERIES_DIR),
                "round-trip-unverifiable",
                f"`regen` writes no entries from {repo}: this checkout does not "
                "hold the series, so the fixed point cannot be checked here "
                "(run `rebase` first, or `--replay` against a checkout at the pin)",
            )
        ]
    if len(generated) != len(paths):
        # A checkout that does not hold *this* series — the pin plus a commit of
        # its own, another branch, the upstream/vllm-musa repo itself — makes
        # `regen` write a different number of entries, and pairing them one by one
        # described that state as 170-odd broken entries. One fact, one row.
        return [
            (
                str(SERIES_DIR),
                "round-trip-count",
                f"`regen` writes {len(generated)} entries from {repo} but the "
                f"series has {len(paths)}: this checkout does not hold the series, "
                "so the fixed point cannot be checked here (run `rebase` first, or "
                "`--replay` against a checkout at the pin)",
            )
        ]
    # Same count is not the same series: `_series_base` accepts `HEAD~count`, so a
    # checkout with `count` commits of *its own* (measured: two real workspace
    # checkouts, 2535 and 2399 commits) pairs the series with unrelated history and
    # reports a `round-trip-dirty` row for every entry of a state. The comparison is
    # **positional**: `regen` numbers the entries and derives each name from its
    # commit's subject, so entry *i* of a checkout that holds the series agrees with
    # entry *i* of the series — a set intersection would accept a foreign history
    # that merely reuses a quarter of the slugs, and (on a series whose subjects
    # repeat) refuse the very checkout that holds it byte-for-byte. The bar is half
    # rounded **down** (``len(paths) // 2``, at least 1), so 86 of the shipped 172.
    want = [_entry_subject_slug(entry.read_bytes()) for entry in paths]
    got = [_entry_subject_slug(patch.read_bytes()) for patch in generated]
    agree = sum(1 for a, b in zip(want, got) if a and a == b)
    if agree < max(1, len(paths) // 2):
        return [
            (
                str(SERIES_DIR),
                "round-trip-unverifiable",
                f"{repo} has {len(generated)} commits above {base}, but only "
                f"{agree} of their subject slugs match the series' {len(paths)} "
                f"position for position (at least {max(1, len(paths) // 2)}, half "
                "rounded down, must): this checkout does not "
                "hold the series, so the fixed point cannot be checked here (run "
                "`rebase` first, or `--replay` against a checkout at the pin)",
            )
        ]
    rows: list[tuple[str, str, str]] = []
    for entry, regen in zip(paths, generated):
        name, regen_name = build_apply.entry_name(entry), build_apply.entry_name(regen)
        want, got = entry.read_bytes(), regen.read_bytes()
        if name == regen_name and want == got:
            continue
        if name != regen_name:
            rows.append(
                (
                    _safe_name(name),
                    "round-trip-dirty",
                    f"`regen` names this entry {regen_name}: the filename is "
                    "not a fixed point",
                )
            )
        else:
            rows.append(
                (
                    _safe_name(name),
                    "round-trip-dirty",
                    "`regen` rewrites this entry: "
                    f"{_first_diff_line(want, got)}",
                )
            )
    for regen in generated[len(paths) :]:
        rows.append(
            (
                _safe_name(build_apply.entry_name(regen)),
                "round-trip-dirty",
                "`regen` produces this entry; series/ has no file with that name",
            )
        )
    for entry in paths[len(generated) :]:
        rows.append(
            (
                _safe_name(build_apply.entry_name(entry)),
                "round-trip-dirty",
                "`regen` produces no entry with this name",
            )
        )
    return rows


def _replay_rows(
    repo: Path, paths: list[Path], texts: dict[str, bytes]
) -> list[tuple[str, str, str]]:
    """Replay the series in a disposable clone of ``repo``; rows for what fails.

    The clone is ``--shared``: git reads objects through the alternate but only
    ever writes into the throwaway checkout, so the caller's repository (their
    only copy of a pin, on a machine with no network) is never touched. The
    replay starts from that clone's HEAD, and the clone gets its own committer
    identity so a machine without a global ``user.email`` does not turn every
    replay into a spurious failure.

    A replay that applies also settles two questions the cheap half can only
    defer, both answered here and nowhere else:

      * every postimage the series *declares* must exist as a blob once the
        replay has run. An ``index`` line naming an id no entry produces is a
        hand-edited anchor, and the preimages it exempts were never verified
        (``exemption-unearned``);
      * the series must also apply the way the *build* applies it — sequentially
        with ``git apply --recount -p1``. ``git am -3`` rescues a stale hunk with
        a 3-way merge, the build path does not, so a series can replay green and
        still break ``build_apply.py`` (``build-path-conflict``).
    """
    tmp = Path(tempfile.mkdtemp(prefix="musa-gate-replay-"))
    try:
        clone = tmp / "repo"
        r = _run_git(tmp, "clone", "--quiet", "--shared", str(repo), str(clone))
        if r.returncode:
            raise RepoUnusable(
                f"cannot clone {repo} for a replay: {_git_first_error(r)}"
            )
        _run_git(clone, "config", "user.email", "musa@local")
        _run_git(clone, "config", "user.name", "musa")
        applied = 0
        made: dict[str, str | None] = {}
        for patch in paths:
            name = build_apply.entry_name(patch)
            head_before = _git(clone, "rev-parse", "HEAD").stdout.strip()
            r = _git(clone, "am", "-3", "--keep-non-patch", str(patch))
            head_after = _git(clone, "rev-parse", "HEAD").stdout.strip()
            if r.returncode == 0 and head_after == head_before:
                # `git am` exits 0 and creates NO commit when the entry is
                # already present ("No changes -- Patch already applied."). Read
                # as a pass, that silently swallows every entry of a series the
                # checkout already holds and then blames whichever entry first
                # conflicts for a state reason — an entry that is green at the
                # pin. One row, on the entry that applied nothing.
                return [
                    (
                        _safe_name(name),
                        "replay-noop",
                        "`git am` applied nothing (the change is already present "
                        "in this checkout): if it already holds the series, "
                        "replay needs a checkout at the pin, and `--round-trip` "
                        "is the mode for the maintenance state",
                    )
                ]
            if r.returncode != 0:
                detail = _git_first_error(r)
                _git(clone, "am", "--abort")
                if applied == 0 and texts.get(name, b"").strip():
                    detail += (
                        f" (the first entry that applies fails here, after "
                        f"{applied} applied: if this checkout already holds the "
                        "series, replay needs one at the pin, and `--round-trip` "
                        "is the mode for the maintenance state)"
                    )
                return [(_safe_name(name), "replay-failed", detail)]
            applied += 1
            made[name] = head_after
        return _unearned_exemption_rows(clone, paths, texts, made) + (
            _build_path_rows(tmp, repo, paths)
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _unearned_exemption_rows(
    clone: Path,
    paths: list[Path],
    texts: dict[str, bytes],
    applied: dict[str, str | None],
) -> list[tuple[str, str, str]]:
    """Postimages an entry declares that *that entry's own commit* does not hold.

    "Produced" is a per-entry fact, not a reachability one: after the replay the
    declared id must be the blob at the declared path **in the commit the entry
    itself created**. Three weaker readings were measured, and all three are
    wrong:

    * existence in the object database — every clone already holds the base's own
      blobs, so an `index` line naming the pin's version of the file would pass;
    * reachability from *any* commit the replay created — order-blind, so a
      fabricated id that a **later** entry happens to produce passes as well;
    * reachability from ``pre_head..HEAD`` via ``git rev-list --objects`` — that
      range omits blobs also reachable from the base, so an entry creating a file
      whose content already exists there (a new empty ``__init__.py``; the pin
      tracks 252 empty files) was a false FAIL.

    Measured on the shipped series: 0 of the 232 declared postimages resolve in a
    fresh clone at the pin, and 232/232 are the declared path's blob in the very
    commit that declares them, so the strict reading costs nothing here.
    """
    declared: dict[tuple[str, str], str] = {}
    for patch in paths:
        name = build_apply.entry_name(patch)
        text = texts.get(name)
        if text is None:
            continue
        for path, post in _declared_postimages(text):
            if not _ZERO_BLOB.match(post.encode()):
                declared.setdefault((name, post), path)
    if not declared:
        return []
    ids = sorted({blob for _name, blob in declared})
    types = _repo_object_types(clone, ids)
    rows = []
    for (name, blob), path in sorted(declared.items()):
        full = _git(clone, "rev-parse", "--verify", f"{blob}^{{blob}}")
        if full.returncode:
            kind = types.get(blob, "missing")
            rows.append(
                (
                    _safe_name(name),
                    "exemption-unearned",
                    f"the entry declares {blob} as its postimage, but the replay "
                    f"produced no such blob ({kind}): any preimage exempted by "
                    "that declaration is unverified",
                )
            )
            continue
        commit = applied.get(name)
        at_path = (
            _git(clone, "rev-parse", "--verify", f"{commit}:{path}") if commit else None
        )
        if at_path is not None and at_path.returncode == 0 and (
            at_path.stdout.strip() == full.stdout.strip()
        ):
            continue
        rows.append(
            (
                _safe_name(name),
                "exemption-unearned",
                f"the entry declares {blob} as the postimage of {path}, but the "
                "commit it created does not hold that blob there"
                + (
                    f" (it holds {(at_path.stdout or '').strip()[:10] or 'nothing'})"
                    if at_path is not None and at_path.returncode == 0
                    else " (the entry created no commit)"
                    if not commit
                    else ""
                )
                + ": any preimage exempted by that declaration is unverified",
            )
        )
    return rows


def _build_path_rows(
    tmp: Path, repo: Path, paths: list[Path]
) -> list[tuple[str, str, str]]:
    """Apply the series the way ``build_apply.py`` does; rows for what breaks.

    ``build_apply.py`` walks the directory in filename order and applies each
    entry with ``git apply --recount -p1``, with no 3-way fallback. The replay
    above is ``git am -3``, which *does* fall back, so this is the only place
    the gate can see a series that replays and still fails a build.
    """
    clone = tmp / "build-path"
    r = _run_git(tmp, "clone", "--quiet", "--shared", str(repo), str(clone))
    if r.returncode:
        return [
            (
                str(SERIES_DIR),
                "repo-unusable",
                f"cannot clone {repo} for the build-path probe: "
                f"{_git_first_error(r)}",
            )
        ]
    for patch in paths:
        r = _git(clone, "apply", "--recount", "-p1", str(patch))
        if r.returncode:
            return [
                (
                    _safe_name(build_apply.entry_name(patch)),
                    "build-path-conflict",
                    "`git apply --recount -p1` (what build_apply.py runs) fails "
                    f"here: {(r.stderr or r.stdout).strip().splitlines()[-1][:120]}",
                )
            ]
    return []


def _merge_replay_rows(
    rows: list[tuple[str, str, str]], extra: list[tuple[str, str, str]]
) -> list[tuple[str, str, str]]:
    """Fold replay/round-trip rows into the per-entry rows.

    A *clean* per-entry row is replaced, so a green format half keeps exactly
    one row per entry and the totals stay comparable. A row that is already
    dirty keeps its own, more specific diagnosis *for the same failure kind*:
    one entry can be wrong in two independent ways at once — an entry whose
    declared postimage never materializes *and* whose hunks the build path
    refuses — and collapsing those by name hid the second. Kinds are therefore
    deduplicated, not names. A name the format half never saw — ``regen``
    producing a file the series does not have — is appended.
    """

    def find(name: str, status: str) -> int | None:
        for i, (row_name, row_status, _) in enumerate(rows):
            if row_name == name and row_status == status:
                return i
        return None

    for name, status, detail in extra:
        same = find(name, status)
        if same is not None:
            rows[same] = (name, status, detail)
            continue
        clean = find(name, "clean")
        if clean is not None:
            rows[clean] = (name, status, detail)
            continue
        rows.append((name, status, detail))
    return rows


def _number_prefix(name: str) -> str | None:
    """The entry's ``NNNN`` number, or None when the name is not numbered.

    Strictly ASCII ``[0-9]{4}`` followed by ``-`` or the end of the name.
    ``str.isdigit()`` is Unicode-aware and ``name[:4]`` silently mis-slices a
    wider prefix (``00123-b.patch`` reads as ``0012``).
    """
    m = _NUMBER_PREFIX.match(name)
    return m.group(1) if m else None


def _series_numbering_rows(rows) -> list[tuple[str, str, str]]:
    """Series-level numbering: numbered entries are unique and gapless except
    for patch IDs explicitly listed in ``metadata/removed-patches.json``.
    """
    if any(status in _SERIES_FATAL for _, status, _ in rows):
        return []  # nothing was gated: never claim `0001..0000`
    numbers = {name: _number_prefix(name) for name, _, _ in rows}
    out = [
        (name, "unnumbered", "expected NNNN-<slug>.patch")
        for name, number in numbers.items()
        if number is None
    ]
    numbered = sorted({n for n in numbers.values() if n is not None})
    if numbered and numbered[0] == "0000":
        out.append(
            (
                "series/0000",
                "bad-range",
                "numbering starts at 0000, but NNNN is 1-based: the first entry "
                "must be 0001",
            )
        )
    for num in numbered:
        same = [name for name, n in numbers.items() if n == num]
        if len(same) > 1:
            out.append((f"series/{num}", "duplicate-number", ", ".join(same)))
    if numbered:
        expected = {
            f"{i:04d}"
            for i in range(1, int(numbered[-1]) + 1)
            if f"{i:04d}" not in _REMOVED_PATCHES
        }
        missing = sorted(expected - set(numbered))
        present_removed = sorted(set(numbered) & _REMOVED_PATCHES)
        if present_removed:
            out.append(
                (
                    "series numbering",
                    "removed-patch-present",
                    f"declared removed patches still present: {', '.join(present_removed)}",
                )
            )
        if missing:
            out.append(
                (
                    "series numbering",
                    "non-contiguous",
                    f"expected 0001..{numbered[-1]}, missing {', '.join(missing)} "
                    f"(numbers present: {', '.join(numbered)})",
                )
            )
    return out


def _numbering_rows(rows) -> list[tuple[str, str, str]]:
    """``_series_numbering_rows`` per directory: ``upstream/`` and the MUSA
    entries are numbered apart."""
    prefix = manifest.UPSTREAM_SUBDIR + "/"
    out = _series_numbering_rows([r for r in rows if not r[0].startswith(prefix)])
    upstream = [(r[0][len(prefix) :], *r[1:]) for r in rows if r[0].startswith(prefix)]
    for name, status, detail in _series_numbering_rows(upstream):
        if name == "series numbering":
            name = f"series/{manifest.UPSTREAM_SUBDIR} numbering"
        elif name.startswith("series/"):
            name = f"series/{prefix}{name[len('series/') :]}"
        else:
            name = prefix + name
        out.append((name, status, detail))
    return out


def _numbering_line(rows, number_rows) -> str | None:
    """The series-level numbering verdict, or None when there is nothing to
    report (a missing or empty series has no numbering).

    The range printed is the range *observed* (min..max of the prefixes found),
    never ``0001..<count>``: an entry numbered ``0000`` used to pass and be
    reported as ``contiguous 0001..0001``.
    """
    if number_rows:
        return f"--- numbering: {len(number_rows)} problem(s) ---"
    if not rows or any(status in _SERIES_FATAL for _, status, _ in rows):
        return None
    prefix = manifest.UPSTREAM_SUBDIR + "/"
    groups = (
        ("", [name for name, _, _ in rows if not name.startswith(prefix)]),
        (
            prefix + " ",
            [name[len(prefix) :] for name, _, _ in rows if name.startswith(prefix)],
        ),
    )
    parts = []
    for label, names in groups:
        present = sorted(n for n in map(_number_prefix, names) if n is not None)
        if present:
            parts.append(
                f"{label}{len(names)} entries are unique and contiguous "
                f"{present[0]}..{present[-1]}"
            )
    if not parts:  # unnumbered entries: already rowed, so unreachable
        return None
    return f"--- numbering: {'; '.join(parts)} ---"


def _series_file_names() -> set[str]:
    """Entry-file names in ``series/``, both raw and as rows display them.

    `_entry_file_rows` keys on the *display* name (control characters escaped), so
    the real listing has to be offered in both forms or a hostile filename is rowed
    and then dropped from the totals.
    """
    try:
        names = {p.name for p in SERIES_DIR.iterdir() if p.is_file()}
        if _upstream_dir().is_dir():
            names |= {
                build_apply.entry_name(p)
                for p in _upstream_dir().iterdir()
                if p.is_file()
            }
    except OSError:
        return set()
    return names | {_safe_name(name) for name in names}


def _entry_file_rows(
    rows: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """One row per entry *file* in ``series/``, for counting and numbering.

    Two kinds of row must not be counted as entries. Series-level rows describe
    the *run* — ``round-trip-unverifiable`` is keyed by the series directory
    itself, and counting it made the pin-state summary read "171 clean / 172
    total" and handed the directory to the numbering gate, which reported it as an
    entry with a missing ``NNNN-`` prefix. And a *file* can carry two rows (a
    rename plus a rewrite, say), which made a one-entry series report "2 entries
    in series/" and number itself twice. Both counts read this list instead.
    """
    seen: dict[str, tuple[str, str, str]] = {}
    for row in rows:
        # ``row[0]`` is the *display* name (control characters escaped), so a real
        # entry with a hostile name must be matched against the directory listing
        # too — otherwise its row vanishes from the counts while still failing the
        # run, and the summary reads `0 total`.
        if (SERIES_DIR / row[0]).is_file() or row[0] in _series_file_names():
            seen.setdefault(row[0], row)
    return [seen[name] for name in sorted(seen)]


def _print_series_section(
    rows,
    number_rows,
    *,
    label: str = "",
    limit: int | None = None,
    entries=None,
) -> None:
    """Print the series-format evidence: bad rows with their detail, the clean
    summary, then numbering.

    Shared by ``check-series`` and ``verify`` so a red series looks the same in
    both, including the ``… N more`` marker when ``limit`` truncates the rows.

    ``entries`` is the subset of ``rows`` that names an entry file; the summary
    counts and the numbering line read it, so a series-level row (which is
    printed like any other) cannot inflate "171 entries" to 172.
    """
    bad = [r for r in rows if r[1] != "clean"]
    shown = bad if limit is None else bad[:limit]
    for name, status, detail in shown:
        print(f"  {status:<14} {name}")
        if detail:
            print(f"                 {detail}")
    if len(shown) < len(bad):
        print(f"  … {len(bad) - len(shown)} more (run `check-series` for the full list)")
    counted = rows if entries is None else entries
    # "clean" counts *entries* that are clean, "need attention" counts every bad
    # row: a series-level row (e.g. a state problem the entries cannot express)
    # is a problem without being an entry, and folding it into the clean count
    # claimed one of 171 clean entries was dirty.
    dirty_entries = sum(1 for _, status, _ in counted if status != "clean")
    print(
        f"--- {label}{len(counted) - dirty_entries} clean / {len(counted)} total / "
        f"{len(bad)} need attention ---"
    )
    for name, status, detail in number_rows:
        print(f"  {status:<14} {name}")
        if detail:
            print(f"                 {detail}")
    line = _numbering_line(counted, number_rows)
    if line:
        print(line)


def cmd_check_series(args) -> int:
    if args.replay and args.round_trip:
        print(
            "ERROR: --replay and --round-trip need different repositories: "
            "--replay applies the series to a checkout AT THE PIN (run it "
            "before `rebase`), --round-trip asks whether `regen` reproduces the "
            "series from a checkout that already HOLDS it (run it after "
            "`rebase`). Run them as two commands."
        )
        print("=== musa_sync check-series: FAIL (usage) ===")
        return 2
    if (args.replay or args.round_trip) and not args.repo:
        print(
            "ERROR: --replay needs --repo <vllm checkout at the pin>; "
            "--round-trip needs --repo <checkout holding the series>"
        )
        print("=== musa_sync check-series: FAIL (usage) ===")
        return 2
    # Resolve --repo once, here: the replay clones it from a temporary
    # directory, so a relative path (the documented invocation is
    # `check-series --repo third_party/vllm`) would otherwise be resolved
    # against that tempdir and reported as "repository ... does not exist".
    metadata = _metadata_result()
    metadata_bad = _metadata_errors(metadata)
    repo = Path(args.repo).resolve() if args.repo else None
    rows = _series_format_rows(repo, replay=args.replay, round_trip=args.round_trip)
    # Numbering and the per-entry counts read the *files* in series/, so they
    # skip the series-level rows the deep modes can add (see _entry_file_rows);
    # the verdict below still fails on every row.
    entries = _entry_file_rows(rows)
    number_rows = _numbering_rows(entries)
    bad = [r for r in rows if r[1] != "clean"]
    if any(status in _SERIES_FATAL for _, status, _ in rows):
        print(f"=== musa_sync check-series: {SERIES_DIR} cannot be gated ===")
    else:
        print(f"=== musa_sync check-series: {len(entries)} entries in series/ ===")
    _print_series_section(rows, number_rows, entries=entries)
    detail = f"series-format={len(bad)}, numbering={len(number_rows)}"
    if metadata_bad:
        detail += f", metadata={len(metadata.errors)}"
    verdict = "PASS" if not (bad or number_rows or metadata_bad) else f"FAIL ({detail})"
    print(f"=== musa_sync check-series: {verdict} ===")
    return 1 if (bad or number_rows or metadata_bad) else 0


def cmd_verify(args) -> int:
    """Probe the manifest's divergences against a checkout of the pin.

    ``--repo`` is a *probe* checkout: it must be at the target and unpatched, since
    the cat-4a/4b rows compare its own files (measured: a checkout that holds the
    series reports 34 false divergences). The target must resolve in it — a stale or
    mistyped pin used to read as a green run.
    """
    target = args.target or _default_target()
    if not target:
        print(
            "ERROR: no target ref (pass --target or set VLLM_COMMIT/VLLM_TAG "
            "in third_party/PINS)"
        )
        return 2
    if not _target_resolves(args.repo, target):
        print(
            f"ERROR: target {target} does not resolve in "
            f"{args.repo or 'the fresh clone'}: `verify` would otherwise report a "
            "green run for a commit that is not the pin"
        )
        print("=== musa_sync verify: FAIL (bad-target) ===")
        return 2
    clone, temp = _ensure_clone(target, args.repo)
    try:
        rows = _verify_rows(clone)
        # The divergence rows are build-side only; also gate the series'
        # *generation* form, which they cannot see (see _series_format_rows).
        # Computed while the clone exists: --repo (or the fresh clone) resolves
        # the blobs the series' `index` lines declare.
        fmt_rows = _series_format_rows(clone)
    finally:
        if temp:
            shutil.rmtree(clone.parent, ignore_errors=True)
    number_rows = _numbering_rows(fmt_rows)
    fmt_bad = [r for r in fmt_rows if r[1] != "clean"]
    bad = [r for r in rows if r[2] in _BAD]
    n_clean = sum(1 for r in rows if r[2] == "clean")
    print(f"=== musa_sync verify: {len(rows)} divergences vs vllm@{target} ===")
    # Series format comes first, and the divergence summary stays last of the
    # summaries: the documented success line is the one a reader checks, so a
    # red series run must not end on `0 need attention`.
    _print_series_section(
        fmt_rows,
        number_rows,
        label="series format: ",
        limit=10,
        entries=_entry_file_rows(fmt_rows),
    )
    for did, cat, status, detail in rows:
        line = f"  [{cat:>2}] {status:<14} {did}"
        if detail:
            line += f"   ({detail})"
        print(line)
    print(
        f"--- {n_clean} clean / {len(rows)} total / {len(bad)} need attention "
        f"({', '.join(sorted({r[2] for r in bad})) or 'none'}) ---"
    )
    verdict = (
        "PASS"
        if not (bad or fmt_bad or number_rows)
        else f"FAIL (divergence={len(bad)}, series-format={len(fmt_bad)}, "
        f"numbering={len(number_rows)})"
    )
    print(f"=== musa_sync verify: {verdict} ===")
    return 1 if (bad or fmt_bad or number_rows) else 0


# -------------------------------------------------------------------- rebase / regen
def _checkout(target: str) -> int:
    if not WORKDIR.exists():
        WORKDIR.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            ["git", "init", "--quiet", str(WORKDIR)], capture_output=True, text=True
        )
        if r.returncode:
            print(r.stderr)
            return 1
        r = _git(WORKDIR, "remote", "add", "origin", VLLM_URL)
        if r.returncode:
            print(r.stderr)
            return 1
    try:
        checkout_ref = _fetch_target(WORKDIR, target)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr)
        return 1
    r = _git(WORKDIR, "checkout", "--force", "--detach", checkout_ref)
    if r.returncode:
        print(r.stderr)
        return 1
    _git(WORKDIR, "update-ref", _BASE_REF, "HEAD")
    return 0


def cmd_rebase(args) -> int:
    target = args.tag or _default_target()
    if not target:
        print(
            "ERROR: no target ref: set VLLM_COMMIT or VLLM_TAG in third_party/PINS "
            "(or pass a tag/ref)"
        )
        print("=== musa_sync rebase: FAIL (usage) ===")
        return 2
    drop: set[str] = set()
    if args.drop_upstreamed:
        history = Path(args.history).resolve() if args.history else WORKDIR
        try:
            rows, planned = _upgrade_plan_rows(
                history, target, _default_target(), fetch=args.fetch
            )
            target = planned.target  # rebase onto the commit the plan judged
        except RepoUnusable as exc:
            print(f"ERROR: {exc}")
            print("=== musa_sync rebase: FAIL (history) ===")
            return 2
        for name, group, verdict, reason in rows:
            if verdict == _PLAN_DROP:
                drop.add(name)
                print(f"  drop    {group:<9} {name}\n                    {reason}")
            elif verdict == _PLAN_REVIEW:
                print(f"  review  {group:<9} {name}\n                    {reason}")
    if _checkout(target):
        print("=== musa_sync rebase: FAIL (checkout) ===")
        return 1
    order = manifest.series_apply_order()
    order = [p for p in order if build_apply.entry_name(p) not in drop]
    for patch in order:
        # --keep-non-patch keeps an upstream subject's "[Core]"-style tags.
        r = _git(WORKDIR, "am", "-3", "--keep-non-patch", str(patch))
        if r.returncode != 0:
            print(f"CONFLICT: {build_apply.entry_name(patch)}\n{r.stdout}\n{r.stderr}")
            print(
                "Resolve in third_party/vllm, then "
                "`git -C third_party/vllm am --continue`, then `git am -3 "
                "--keep-non-patch` the entries after it (running rebase again "
                "starts over; see docs/mdm-developer-guide.md §5); or "
                "`git -C third_party/vllm am --abort` to back out."
            )
            print("=== musa_sync rebase: FAIL (conflict) ===")
            return 1
    dropped = f", dropped {len(drop)} already upstream" if drop else ""
    print(f"rebased {len(order)} patches onto vllm@{target} (git am -3){dropped}")
    if _base_problem(_default_target() or ""):
        print(
            f"Set VLLM_COMMIT and VLLM_TAG in third_party/PINS to {target}, then "
            "run `musa_sync regen`."
        )
    elif drop:
        print("Run `musa_sync regen` to renumber the series without them.")
    print("=== musa_sync rebase: PASS ===")
    return 0


def _regen_module_tripwires() -> int:
    """(re)generate the cat-4a drift tripwires from a PRISTINE upstream
    checkout (the tripwire is the MUSA shadow's delta vs the pinned upstream)."""
    target = _default_target()
    if _checkout(target):  # reset WORKDIR to pristine vllm@<pin>
        return 1
    MODULE_DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for e in manifest.ENTRIES:
        if e.category != "4a":
            continue
        tw = _module_tripwire(WORKDIR, e)
        if tw is None:
            print(f"  WARN no upstream for {e.id} ({e.upstream_path})")
            continue
        _tripwire_path(e).write_text(tw)
        print(f"  tripwire: {e.id} ({len(tw.splitlines())} diff lines)")
        n += 1
    print(f"regenerated {n} cat-4a module tripwires in {MODULE_DRIFT_DIR}")
    return 0


def cmd_regen(args) -> int:
    target = _default_target()
    if args.area == "module":
        if not target:
            print(
                "ERROR: no target ref: set VLLM_COMMIT or VLLM_TAG in "
                "third_party/PINS (or pass --target)"
            )
            print("=== musa_sync regen: FAIL (usage) ===")
            return 2
        rc = _regen_module_tripwires()
        print(f"=== musa_sync regen: {'PASS' if rc == 0 else 'FAIL (module-4a)'} ===")
        return rc
    if not target:
        print(
            "ERROR: no target ref: set VLLM_COMMIT or VLLM_TAG in third_party/PINS "
            "(or pass --target)"
        )
        print("=== musa_sync regen: FAIL (usage) ===")
        return 2
    problem = _base_problem(target)
    if problem:
        print(f"ERROR: {problem}")
        print("=== musa_sync regen: FAIL (base) ===")
        return 1
    with tempfile.TemporaryDirectory(prefix="musa-regen-series-") as tmp:
        staged = Path(tmp)
        generated, error = _format_series(WORKDIR, target, staged)
        if error:
            print(f"ERROR: {error}")
            print("=== musa_sync regen: FAIL (format-patch) ===")
            return 1
        if not generated:
            print(f"ERROR: no patches generated from vllm@{target}..HEAD")
            print("=== musa_sync regen: FAIL (no-patches) ===")
            return 1

        groups = (
            (staged / manifest.UPSTREAM_SUBDIR, _upstream_dir()),
            (staged, SERIES_DIR),
        )
        for src, _ in groups:
            numbers = [p.name.split("-", 1)[0] for p in sorted(src.glob("*.patch"))]
            expected = [f"{i:04d}" for i in range(1, len(numbers) + 1)]
            if numbers != expected:
                print(
                    "ERROR: git format-patch produced a non-contiguous series: "
                    f"expected {expected}, got {numbers}"
                )
                return 1

        headers = {p: p.read_bytes().splitlines()[:2] for p in generated}
        noncanonical_commits = [
            p.name
            for p, lines in headers.items()
            if not lines or lines[0] != _ZERO_COMMIT_HEADER
        ]
        if noncanonical_commits:
            print(
                "ERROR: non-canonical patch commit headers: "
                f"{', '.join(noncanonical_commits)}"
            )
            return 1
        noncanonical_authors = [
            p.name
            for p, lines in headers.items()
            if len(lines) < 2 or lines[1] != _CANONICAL_PATCH_AUTHOR
        ]
        if noncanonical_authors:
            print(
                "ERROR: non-canonical patch author headers: "
                f"{', '.join(noncanonical_authors)}"
            )
            return 1

        known = {p.name for p in _upstream_dir().glob("*.patch")}
        fixes = [
            p.name
            for p in sorted((staged / manifest.UPSTREAM_SUBDIR).glob("*.patch"))
            if p.name not in known
            and not manifest.parse_upstream_pick(p.read_bytes()).commits
        ]
        SERIES_DIR.mkdir(parents=True, exist_ok=True)
        stale: list[Path] = []
        upstream_count = 0
        for src, dest in groups:
            fresh = sorted(src.glob("*.patch"))
            if dest != SERIES_DIR:
                upstream_count = len(fresh)
            if fresh:
                dest.mkdir(parents=True, exist_ok=True)
            existing = sorted(dest.glob("*.patch")) if dest.is_dir() else []
            for patch in fresh:
                shutil.copyfile(patch, dest / patch.name)
            names = {patch.name for patch in fresh}
            stale += [patch for patch in existing if patch.name not in names]
        for patch in stale:
            patch.unlink()
        if _upstream_dir().is_dir() and not any(_upstream_dir().iterdir()):
            _upstream_dir().rmdir()

    upstream_note = f" ({upstream_count} in upstream/)" if upstream_count else ""
    print(
        f"regenerated {len(generated)} contiguous patches{upstream_note} in "
        f"{SERIES_DIR} from vllm@{target}..HEAD; pruned {len(stale)} stale files"
    )
    for name in fixes:
        print(
            f"note: {manifest.UPSTREAM_SUBDIR}/{name} changes a file that an "
            "upstream commit changes, so it stays after them in upstream/; "
            "upgrade-plan reviews it unless every upstream entry before it is kept"
        )
    print("=== musa_sync regen: PASS ===")
    return 0


# ------------------------------------------------------------- upstream commits
# series/upstream/ holds the upstream vLLM commits the pin predates; they apply
# after the MUSA entries. ``backport`` adds one, ``upgrade-plan`` decides what a
# pin bump keeps, drops and reviews, and ``rebase --drop-upstreamed`` acts on it.

_PLAN_KEEP, _PLAN_DROP, _PLAN_REVIEW = "keep", "drop", "review"
_PR_SUFFIX = re.compile(r"\(#([0-9]+)\)\s*$")
_REVERT_SUBJECT = re.compile(r'^Revert "(.*)"(?:\s*\(#[0-9]+\))?\s*$')
# Records the commit `_checkout` put the clone on, which the series sits on.
_BASE_REF = "refs/musa-sync/base"
# Protocol v2 serves any reachable commit; v0, the default before git 2.29,
# only branch and tag tips.
_FETCH_ANY_COMMIT = ("-c", "protocol.version=2", "fetch", "--quiet", "--no-tags")


class _Ref(NamedTuple):
    """An upstream vLLM change: a commit sha or a pull-request number."""

    kind: str  # "commit" or "pr"
    value: str

    def __str__(self) -> str:
        return self.value[:12] if self.kind == "commit" else f"#{self.value}"


def _is_upstream_entry(patch: Path) -> bool:
    return patch.parent.name == manifest.UPSTREAM_SUBDIR


def _group_problem(patch: Path, text: bytes, first: bool) -> tuple[str, str] | None:
    """A MUSA entry is no cherry-pick, and ``upstream/`` starts with one; a later
    entry there without one is a MUSA fix for the upstream commit before it."""
    picked = manifest.parse_upstream_pick(text).commits
    if _is_upstream_entry(patch):
        if first and not picked:
            return (
                "not-a-cherry-pick",
                f"{manifest.UPSTREAM_SUBDIR}/ starts with a `git cherry-pick -x` of an "
                "upstream vLLM commit, but this message has no `(cherry picked from "
                "commit …)` line: add upstream commits with `musa_sync backport <sha>`",
            )
        return None
    if picked:
        return (
            "misplaced-cherry-pick",
            f"cherry-picked from {picked[0][:12]}: an upstream vLLM commit belongs "
            f"in {manifest.UPSTREAM_SUBDIR}/ (`musa_sync backport <sha>`); drop the "
            "line if this is a MUSA change",
        )
    return None


def _format_series(
    repo: Path, base: str, staged: Path
) -> tuple[list[Path], str | None]:
    """Write what ``regen`` writes for ``base..HEAD`` of ``repo`` into ``staged``.

    The MUSA entries go to ``staged/``. After them, ``upstream/`` takes the
    upstream commits (``git cherry-pick -x``) and the MUSA fixes made on top of
    them, the commits above the first cherry-pick that change a file those
    commits change. Any other commit above the first cherry-pick is a MUSA
    entry: it shares no file with ``upstream/``, so its patch is the same
    when it applies before them. Each directory is numbered from 0001, with the canonical
    author and a fixed ``core.abbrev``, so the ``index`` lines do not depend
    on how many objects the repository holds.
    Returns ``(entries in apply order, error)``.
    """
    split = _split_stack(repo, base)
    if isinstance(split, str):
        return [], split
    commits, first, upstream = split
    groups = (
        (staged, first, [sha for sha in commits[first:] if sha not in upstream]),
        (
            staged / manifest.UPSTREAM_SUBDIR,
            0,
            [sha for sha in commits if sha in upstream],
        ),
    )
    out: list[Path] = []
    for dest, lead, rest in groups:
        if not lead and not rest:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        jobs = [(1, f"{base}..{commits[lead - 1]}")] if lead else []
        jobs += [(lead + n + 1, "-1", sha) for n, sha in enumerate(rest)]
        for start, *revs in jobs:
            r = _git(
                repo,
                "-c",
                "core.abbrev=10",
                "format-patch",
                "--no-signature",
                "--no-numbered",
                "--zero-commit",
                f"--start-number={start}",
                "-o",
                str(dest),
                *revs,
            )
            if r.returncode:
                return [], f"`regen` (git format-patch) failed: {_git_first_error(r)}"
        group = sorted(dest.glob("*.patch"))
        for patch in group:
            _normalize_patch_author(patch)
        out.extend(group)
    return out, None


def _split_stack(repo: Path, base: str) -> tuple[list[str], int, set[str]] | str:
    """``(commits of base..HEAD, index of the first cherry-pick, the commits
    that go to upstream/)``, or an error message."""
    log = _git(repo, "log", "--reverse", "--format=%H%x1f%B%x1e", f"{base}..HEAD")
    if log.returncode:
        return f"`git log {base}..HEAD` failed: {_git_first_error(log)}"
    commits, picks = [], []
    for record in log.stdout.split("\x1e"):
        sha, sep, body = record.strip("\n").partition("\x1f")
        if sep:
            commits.append(sha.strip())
            picks.append(bool(manifest.picked_commits(body.splitlines())))
    first = picks.index(True) if True in picks else len(commits)
    files = {sha: _changed_paths(repo, sha) for sha in commits[first:]}
    upstream = {sha for sha, picked in zip(commits, picks) if picked}
    touched = set().union(*(files[sha] for sha in upstream))
    grew = True
    while grew:  # a MUSA fix on top of an upstream commit stays with it
        grew = False
        for sha in commits[first:]:
            if sha not in upstream and files[sha] & touched:
                upstream.add(sha)
                touched |= files[sha]
                grew = True
    return commits, first, upstream


def _changed_paths(repo: Path, sha: str) -> set[str]:
    """The paths a commit changes."""
    r = _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", "-z", sha)
    return (
        {path for path in r.stdout.split("\0") if path} if r.returncode == 0 else set()
    )


def _change_lines(diff: str) -> tuple:
    """The changed lines of a diff, per file, without context or line numbers:
    two diffs with equal results make the same change, whatever their context."""
    files: dict[str, list[str]] = {}
    path, in_hunk = None, False
    for line in diff.split("\n"):
        if line.startswith("diff --git "):
            path, in_hunk = line.split(" b/", 1)[-1], False
            files.setdefault(path, [])
        elif path is not None and line.startswith("@@"):
            in_hunk = True
        elif path is not None and in_hunk and line[:1] in ("+", "-"):
            files[path].append(line.rstrip())
    return tuple(sorted((p, tuple(lines)) for p, lines in files.items()))


class _UpstreamHistory:
    """Read-only questions about upstream vLLM history for a pin bump.

    ``repo`` should hold the full history of ``target``. A shallow clone
    cannot see most commits, so a "no" from it is reported as unknown.
    """

    def __init__(
        self, repo: Path, target: str, pin: str | None, *, fetch: bool = False
    ):
        _require_usable_repo(repo)
        self.repo = repo
        self.fetch = fetch
        self.target = self._resolve(target)
        if self.target is None:
            raise RepoUnusable(
                f"{target!r} is not a commit in {repo} (fetch it, or pass --fetch)"
            )
        self.pin = self._resolve(pin) if pin else None
        shallow = _run_git(repo, "rev-parse", "--is-shallow-repository")
        self.shallow = shallow.returncode == 0 and shallow.stdout.strip() == "true"
        self._range_ids: dict[str, str] | None = None

    def _resolve(self, ref: str) -> str | None:
        """The full sha of ``ref``; with ``fetch``, fetched from ``origin``
        when missing (a full fetch: ancestry needs the history)."""
        candidates = [ref]
        if self.fetch:
            candidates.append(None)
        for candidate in candidates:
            if candidate is None:
                fetched = _run_git(self.repo, *_FETCH_ANY_COMMIT, "origin", ref)
                if fetched.returncode:
                    return None
                candidate = "FETCH_HEAD"
            r = _run_git(
                self.repo, "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}"
            )
            if r.returncode == 0:
                return r.stdout.strip()
        return None

    def _log(self, *args: str) -> list[str]:
        r = _run_git(self.repo, "log", *args)
        return r.stdout.splitlines() if r.returncode == 0 else []

    def commit(self, sha: str) -> str | None:
        """``sha`` if this repository has it (fetching it if it can)."""
        return self._resolve(sha)

    def is_ancestor(self, sha: str) -> bool | None:
        r = _run_git(self.repo, "merge-base", "--is-ancestor", sha, self.target)
        if r.returncode == 0:
            return True
        return False if r.returncode == 1 and not self.shallow else None

    def merged_pr(self, number: str) -> str | None:
        """The target-history commit that merged ``#number``, or None.

        vLLM squash-merges a pull request into one commit whose subject ends
        in ``(#N)``; release-branch cherry-picks keep that subject.
        """
        for line in self._log(
            self.target, "-F", f"--grep=(#{number})", "--format=%H%x1f%s"
        ):
            sha, _, subject = line.partition("\x1f")
            m = _PR_SUFFIX.search(subject)
            if m and m.group(1) == number:
                return sha
        return None

    def cherry_pick_of(self, sha: str) -> str | None:
        hits = self._log(
            self.target, "-F", f"--grep=cherry picked from commit {sha}", "--format=%H"
        )
        return hits[0] if hits else None

    def revert_of(self, sha: str) -> str | None:
        """The latest commit after ``sha`` in the target that reverts it, unless
        a later commit lands the change again.

        git writes "This reverts commit <sha>." into a revert; vLLM squash-merges
        a revert PR as ``Revert "<title>" (#N)`` without that line.
        """
        subject = (self._log("-1", "--format=%s", sha) or [""])[0]
        title = _PR_SUFFIX.sub("", subject).rstrip()
        span = f"{sha}..{self.target}"
        hits = self._log(span, "-F", f"--grep=This reverts commit {sha}", "--format=%H")
        if title:
            for line in self._log(
                span, "-F", f'--grep=Revert "{title}', "--format=%H%x1f%s"
            ):
                commit, _, later = line.partition("\x1f")
                m = _REVERT_SUBJECT.match(later)
                if m and m.group(1).strip() in (subject.strip(), title):
                    hits.append(commit)
        if not hits:
            return None
        revert = self._log("--no-walk", "--format=%H", *hits)[0]
        for line in self._log(
            f"{revert}..{self.target}", "-F", f"--grep={title}", "--format=%s"
        ):
            later = _PR_SUFFIX.sub("", line).strip()
            if later == title or later.startswith(
                (f'Reapply "{title}', f'Revert "Revert "{title}')
            ):
                return None
        return revert

    def pr_of(self, sha: str) -> str | None:
        subject = self._log("-1", "--format=%s", sha)
        m = _PR_SUFFIX.search(subject[0]) if subject else None
        return m.group(1) if m else None

    def _patch_id(self, text: str) -> str | None:
        r = _run_git(self.repo, "patch-id", "--stable", stdin_text=text)
        words = r.stdout.split()
        return words[0] if r.returncode == 0 and words else None

    def same_change_as(self, sha: str) -> str | None:
        """A commit in pin..target whose ``git patch-id --stable`` equals
        ``sha``'s: the same change committed again (a cherry-pick without
        ``-x`` or with another subject)."""
        if self.pin is None:
            return None
        if self._range_ids is None:
            log = _run_git(
                self.repo, "log", "--no-merges", "-p", f"{self.pin}..{self.target}"
            )
            ids = _run_git(self.repo, "patch-id", "--stable", stdin_text=log.stdout)
            self._range_ids = {}
            for line in ids.stdout.splitlines():
                words = line.split()
                if len(words) == 2:
                    self._range_ids.setdefault(words[0], words[1])
        shown = _run_git(self.repo, "show", "--format=", sha)
        pid = self._patch_id(shown.stdout) if shown.returncode == 0 else None
        hit = self._range_ids.get(pid) if pid else None
        return hit if hit != sha else None

    def ref_state(self, ref) -> tuple[str, str]:
        """``(state, evidence)`` of an upstream reference in the target:
        ``in``, ``out``, ``reverted`` or ``unknown``."""
        found, how = None, ""
        if ref.kind == "commit":
            sha = self.commit(ref.value)
            if sha is None:
                return "unknown", f"{ref.value[:12]} is not in {self.repo}"
            if self.is_ancestor(sha):
                found, how = sha, f"{sha[:12]} is in the target history"
            else:
                pick = self.cherry_pick_of(sha)
                pr = None if pick else self.pr_of(sha)
                merged = self.merged_pr(pr) if pr else None
                same = None if (pick or merged) else self.same_change_as(sha)
                if pick:
                    found, how = pick, f"cherry-picked as {pick[:12]}"
                elif merged:
                    found, how = merged, f"#{pr} merged as {merged[:12]}"
                elif same:
                    found, how = same, f"same change as {same[:12]} (patch-id)"
        else:
            merged = self.merged_pr(ref.value)
            if merged:
                found, how = merged, f"#{ref.value} merged as {merged[:12]}"
        if found:
            revert = self.revert_of(found)
            if revert:
                return "reverted", f"{how}, reverted by {revert[:12]}"
            return "in", how
        if self.shallow:
            return "unknown", f"{ref} not found in shallow history"
        return "out", f"{ref} is not in the target"


def _refs_state(history: _UpstreamHistory, refs) -> tuple[str, str]:
    """``in`` when any ref is in the target; ``reverted``/``unknown`` win over
    ``out``."""
    states = [history.ref_state(ref) for ref in refs]
    if not states:
        return "out", ""
    ins = list(dict.fromkeys(ev for st, ev in states if st == "in"))
    if ins:
        return "in", "; ".join(ins)
    bad = [(st, ev) for st, ev in states if st in ("reverted", "unknown")]
    if bad:
        return bad[0]
    return "out", next(ev for st, ev in states if st == "out")


def _is_verbatim(history: _UpstreamHistory, patch: Path, commits) -> bool | None:
    """Whether the entry changes what its upstream commit changes (context and
    line numbers aside); None when no such commit is in the history."""
    try:
        ours = _change_lines(patch.read_text(errors="replace"))
    except OSError:
        return None
    for sha in commits:
        full = history.commit(sha)
        if full is None:
            continue
        shown = _run_git(history.repo, "show", "--format=", full)
        if shown.returncode == 0:
            return ours == _change_lines(shown.stdout)
    return None


def _plan_verdict(
    upstream: bool, state: tuple[str, str], content: str | None, verbatim: bool | None
) -> tuple[str, str]:
    """What a pin bump does with one entry: ``(keep|drop|review, reason)``.

    ``content`` is the entry's ``build_apply.apply_patch`` status when the
    series is applied in order to the pristine target (``already-applied``,
    ``applied`` or ``conflict``), or None when not checked. It settles what the
    history cannot: whether the change is in the target at all. An upstream
    entry that differs from its upstream commit is reviewed, not dropped: its
    local changes may still be needed.
    """
    if not upstream:
        if content == "already-applied":
            return _PLAN_REVIEW, "the change is already in the target: upstreamed?"
        return _PLAN_KEEP, ""
    st, evidence = state
    if st == "in" and content == "applied":
        return _PLAN_REVIEW, f"{evidence}, but the change still applies: reverted?"
    if content == "already-applied":
        # The target holds the whole entry, local changes included.
        return _PLAN_DROP, (
            evidence if st == "in" else "the change is already in the target"
        )
    if st == "in":
        if verbatim:
            return _PLAN_DROP, evidence
        if verbatim is None:
            return (
                _PLAN_REVIEW,
                f"{evidence}; its upstream commit is not in the history",
            )
        return _PLAN_REVIEW, (
            f"{evidence}; the entry differs from its upstream commit: drop it unless "
            "its local changes are still needed"
        )
    if st == "reverted":
        return _PLAN_REVIEW, evidence
    if st == "unknown" and content != "applied":
        return _PLAN_REVIEW, evidence + " (pass --history <full vLLM clone>)"
    return _PLAN_KEEP, ""


@contextmanager
def _exported_tree(repo: Path, sha: str):
    """A pristine copy of ``sha``'s tree (``git archive``; ``repo`` is only read)."""
    tmp = Path(tempfile.mkdtemp(prefix="musa-plan-"))
    try:
        tree = tmp / "vllm"
        tree.mkdir()
        archive = subprocess.run(
            ["git", "-C", str(repo), "archive", "--format=tar", sha],
            capture_output=True,
        )
        if archive.returncode:
            raise RepoUnusable(f"git archive {sha[:12]} failed in {repo}")
        subprocess.run(["tar", "-x", "-C", str(tree)], input=archive.stdout, check=True)
        subprocess.run(
            ["git", "init", "--quiet", str(tree)], check=True, capture_output=True
        )
        yield tree
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _upgrade_plan_rows(
    history_repo: Path,
    target: str,
    pin: str | None,
    *,
    content: bool = True,
    fetch: bool = False,
) -> tuple[list[tuple[str, str, str, str]], _UpstreamHistory]:
    """``(entry, group, verdict, reason)`` for every series entry."""
    paths, refusal = _series_dir_paths()
    if refusal:
        raise RepoUnusable(f"{SERIES_DIR}: {refusal[1]}")
    for p in paths:
        try:
            regular = stat.S_ISREG(p.lstat().st_mode)
        except OSError:
            regular = False
        if not regular:
            raise RepoUnusable(
                f"{build_apply.entry_name(p)} is not a regular file "
                "(run `musa_sync check-series`)"
            )
    history = _UpstreamHistory(history_repo, target, pin, fetch=fetch)
    statuses: dict[Path, str] = {}
    if content:
        # In order and cumulatively, like the build: an entry's preimage can
        # be an earlier entry's postimage.
        with _exported_tree(history_repo, history.target) as tree:
            for p in paths:
                statuses[p] = build_apply.apply_patch(tree, p)
    rows = []
    moved: list[tuple[str, str]] = []  # upstream commits so far that are not kept
    for p in paths:
        name = build_apply.entry_name(p)
        if not _is_upstream_entry(p):
            verdict = _plan_verdict(False, ("out", ""), statuses.get(p), None)
            rows.append((name, "musa", *verdict))
            continue
        pick = manifest.upstream_pick(p)
        if not pick.commits:
            # A MUSA fix serves the upstream commits before it.
            if not moved:
                rows.append((name, "fix", _PLAN_KEEP, ""))
            else:
                rows.append(
                    (
                        name,
                        "fix",
                        _PLAN_REVIEW,
                        f"a MUSA change on top of {moved[0][0]}, which is "
                        f"{moved[0][1]}: drop it with it, or keep it as a MUSA "
                        "patch if still needed",
                    )
                )
            continue
        refs = [_Ref("commit", sha) for sha in pick.commits]
        refs += [_Ref("pr", pick.pr)] if pick.pr else []
        state = _refs_state(history, refs)
        verbatim = _is_verbatim(history, p, pick.commits)
        verdict = _plan_verdict(True, state, statuses.get(p), verbatim)
        if verdict[0] != _PLAN_KEEP:
            moved.append((name, verdict[0]))
        rows.append((name, "upstream", *verdict))
    return rows, history


def cmd_upgrade_plan(args) -> int:
    metadata = _metadata_result()
    if _metadata_errors(metadata):
        print("=== musa_sync upgrade-plan: FAIL (metadata) ===")
        return 1
    pin = _default_target()
    target = args.target or pin
    if not target:
        print("ERROR: no target ref: pass one or set VLLM_COMMIT/VLLM_TAG in PINS")
        print("=== musa_sync upgrade-plan: FAIL (usage) ===")
        return 2
    repo = Path(args.history).resolve() if args.history else WORKDIR
    try:
        rows, history = _upgrade_plan_rows(
            repo, target, pin, content=not args.no_content, fetch=args.fetch
        )
    except RepoUnusable as exc:
        print(f"ERROR: {exc}")
        print("=== musa_sync upgrade-plan: FAIL (history) ===")
        return 2
    shallow = " (shallow: negatives are unknown)" if history.shallow else ""
    print(
        f"=== musa_sync upgrade-plan: {(history.pin or '?')[:12]} -> "
        f"{history.target[:12]} (history: {repo}{shallow}) ==="
    )
    for name, group, verdict, reason in rows:
        if verdict != _PLAN_KEEP or args.all:
            print(f"  {verdict:<7} {group:<9} {name}")
            if reason:
                print(f"                    {reason}")
    counts = {
        v: sum(1 for r in rows if r[2] == v)
        for v in (_PLAN_KEEP, _PLAN_DROP, _PLAN_REVIEW)
    }
    print(
        f"--- {counts[_PLAN_KEEP]} keep / {counts[_PLAN_DROP]} drop / "
        f"{counts[_PLAN_REVIEW]} review ---"
    )
    review = counts[_PLAN_REVIEW]
    verdict = f"FAIL (review={review})" if review else "PASS"
    print(f"=== musa_sync upgrade-plan: {verdict} ===")
    return 1 if review else 0


def _base_problem(pin: str) -> str | None:
    """Why the clone's commits do not sit on ``pin``, the PINS commit."""
    want = _run_git(WORKDIR, "rev-parse", "--verify", "--quiet", f"{pin}^{{commit}}")
    if not pin or want.returncode:
        return (
            f"{WORKDIR} has no commit {pin[:12] or '(no pin)'}: run `musa_sync rebase`"
        )
    held = _run_git(
        WORKDIR, "rev-parse", "--verify", "--quiet", f"{_BASE_REF}^{{commit}}"
    )
    if held.returncode == 0:
        on_it = _run_git(WORKDIR, "merge-base", "--is-ancestor", _BASE_REF, "HEAD")
        if on_it.returncode == 0 and held.stdout.strip() != want.stdout.strip():
            return (
                f"{WORKDIR} holds the series on {held.stdout.strip()[:12]}, but "
                f"third_party/PINS names {pin[:12]}: set VLLM_COMMIT and VLLM_TAG "
                "to the ref the series sits on"
            )
    if _run_git(WORKDIR, "merge-base", "--is-ancestor", pin, "HEAD").returncode:
        return (
            f"{WORKDIR} HEAD does not descend from {pin[:12]}: run `musa_sync rebase`"
        )
    return None


def _git_in_progress() -> str | None:
    """The git operation left unfinished in the clone, if any."""
    for marker, what in (
        ("rebase-apply", "git am"),
        ("CHERRY_PICK_HEAD", "git cherry-pick"),
        ("rebase-merge", "git rebase"),
    ):
        r = _run_git(WORKDIR, "rev-parse", "--git-path", marker)
        path = Path(r.stdout.strip())
        if (
            r.returncode == 0
            and (path if path.is_absolute() else WORKDIR / path).exists()
        ):
            return what
    return None


def _stack_problem(paths: list[Path], pin: str) -> str | None:
    """Why the clone does not hold exactly the series on ``pin``."""
    with tempfile.TemporaryDirectory(prefix="musa-stack-") as tmp:
        generated, error = _format_series(WORKDIR, pin, Path(tmp))
        if error:
            return error
        same = len(generated) == len(paths) and all(
            build_apply.entry_name(g) == build_apply.entry_name(p)
            and g.read_bytes() == p.read_bytes()
            for g, p in zip(generated, paths)
        )
    if same:
        return None
    return (
        f"{WORKDIR} does not hold the series on vllm@{pin[:12]}: run `musa_sync "
        "rebase` first (or `musa_sync regen` to keep commits made there)"
    )


def _restack(paths: list[Path], pin: str) -> str | None:
    """Rebuild the clone from the series on ``pin`` when it holds MUSA commits
    above upstream ones, so that a new commit lands after every entry."""
    split = _split_stack(WORKDIR, pin)
    if isinstance(split, str):
        return split
    commits, first, upstream = split
    if all(sha in upstream for sha in commits[first:]):
        return None
    head = _run_git(WORKDIR, "rev-parse", "HEAD").stdout.strip()
    _run_git(WORKDIR, "checkout", "--quiet", "--detach", pin)
    if not paths:
        return None
    r = _run_git(WORKDIR, "am", "-3", "--keep-non-patch", *[str(p) for p in paths])
    if r.returncode == 0:
        return None
    _run_git(WORKDIR, "am", "--abort")
    _run_git(WORKDIR, "checkout", "--quiet", "--detach", head)
    return f"cannot rebuild the series in {WORKDIR}: {_git_first_error(r)}"


def cmd_backport(args) -> int:
    pin = _default_target()
    if not pin:
        print("ERROR: no pin: set VLLM_COMMIT or VLLM_TAG in third_party/PINS")
        print("=== musa_sync backport: FAIL (usage) ===")
        return 2
    paths, refusal = _series_dir_paths()
    if refusal and refusal[0] not in ("missing-series-dir", "empty-series"):
        print(f"ERROR: {SERIES_DIR}: {refusal[1]}")
        print("=== musa_sync backport: FAIL (series) ===")
        return 2
    upstream = [p for p in paths if _is_upstream_entry(p)]
    busy = _git_in_progress()
    dirty = _run_git(WORKDIR, "status", "--porcelain", "--untracked-files=no")
    problem = (
        f"{WORKDIR} has an unfinished {busy}"
        if busy
        else (
            f"{WORKDIR} has uncommitted changes"
            if dirty.stdout.strip()
            else _base_problem(pin) or _stack_problem(paths, pin)
        )
    )
    if problem:
        print(f"ERROR: {problem}")
        print("=== musa_sync backport: FAIL (state) ===")
        return 2
    sha = _resolve_upstream_commit(args.commit, args.source)
    if sha is None:
        print(
            f"ERROR: cannot find upstream commit {args.commit} "
            "(pass its full sha, or --from <vLLM clone>)"
        )
        print("=== musa_sync backport: FAIL (commit) ===")
        return 1
    carried = next(
        (p for p in upstream if sha in manifest.upstream_pick(p).commits), None
    )
    if carried is not None:
        print(f"ERROR: {sha[:12]} is already {build_apply.entry_name(carried)}")
        print("=== musa_sync backport: FAIL (commit) ===")
        return 1
    problem = _restack(paths, pin)
    if problem:
        print(f"ERROR: {problem}")
        print("=== musa_sync backport: FAIL (state) ===")
        return 2
    head = _run_git(WORKDIR, "rev-parse", "HEAD").stdout.strip()
    picked = _run_git(WORKDIR, "cherry-pick", "-x", sha)
    if picked.returncode:
        unmerged = _run_git(WORKDIR, "diff", "--name-only", "--diff-filter=U")
        if unmerged.stdout.strip():
            print(f"CONFLICT: cherry-pick {sha[:12]}\n{picked.stdout}{picked.stderr}")
            print(
                "Resolve it in third_party/vllm, `git -C third_party/vllm cherry-pick "
                "--continue`, then `musa_sync regen`; or `git -C third_party/vllm "
                "cherry-pick --abort`."
            )
            print("=== musa_sync backport: FAIL (conflict) ===")
            return 1
        _run_git(WORKDIR, "cherry-pick", "--abort")
        _run_git(WORKDIR, "checkout", "--quiet", "--detach", head)
        print(f"ERROR: cherry-picking {sha[:12]} failed: {_git_first_error(picked)}")
        print("=== musa_sync backport: FAIL (commit) ===")
        return 1
    ours = _run_git(WORKDIR, "show", "--format=", "HEAD").stdout
    theirs = _run_git(WORKDIR, "show", "--format=", sha).stdout
    note = (
        ""
        if _change_lines(ours) == _change_lines(theirs)
        else " (its changed lines differ from upstream)"
    )
    print(
        f"backported {sha[:12]} as {manifest.UPSTREAM_SUBDIR}/"
        f"{len(upstream) + 1:04d}{note}"
    )
    if args.no_regen:
        print("=== musa_sync backport: PASS ===")
        return 0
    return cmd_regen(argparse.Namespace(area=None))


def _resolve_upstream_commit(commit: str, source: str | None) -> str | None:
    """The full sha of ``commit`` in the clone, fetched from ``source`` (or
    the clone's origin) when missing."""

    def resolve() -> str | None:
        r = _run_git(
            WORKDIR, "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}"
        )
        return r.stdout.strip() if r.returncode == 0 else None

    sha = resolve()
    if sha is None:
        remote = str(Path(source).resolve()) if source else "origin"
        _run_git(WORKDIR, *_FETCH_ANY_COMMIT, remote, commit)
        sha = resolve()
    return sha


# -------------------------------------------------------------------------- report
def cmd_report(args) -> int:
    if args.metadata:
        result = _metadata_result()
        if _metadata_errors(result):
            return 1
        print(f"Patch metadata: {result.patch_count} patches, "
              f"{result.group_count} functional groups, "
              f"{result.omission_count} legacy omissions")
        for group in result.groups:
            checks = ", ".join(group["required_checks"])
            print(f"  {group['id']}: {len(group['patches'])} patches; "
                  f"required: {checks}")
        unresolved = [row["legacy_id"] for row in result.lineage
                      if row["disposition"] in ("unresolved", "performance_opportunity")]
        print(f"  unresolved/opportunity: {', '.join(unresolved) or 'none'}")
        print("  Required checks are not pass claims; MUSA/model validation needs separate evidence.")
        return 0
    by_cat: dict[str, int] = {}
    for e in manifest.ENTRIES:
        by_cat[e.category] = by_cat.get(e.category, 0) + 1
    if args.doc:
        print("| id | cat | phase | required | upstream_path | intent |")
        print("|---|---|---|---|---|---|")
        for e in manifest.ENTRIES:
            print(
                f"| {e.id} | {e.category} | {e.apply_phase} | {e.required} | "
                f"{e.upstream_path or ''} | {e.intent} |"
            )
        return 0
    print(
        f"MDM manifest: {len(manifest.ENTRIES)} divergences by category {dict(sorted(by_cat.items()))}"
    )
    prefix = manifest.UPSTREAM_SUBDIR + "/"
    upstream = [e for e in manifest.ENTRIES if e.id.startswith(prefix)]
    if upstream:
        print(f"upstream commits ({prefix}, applied last): {len(upstream)}")
        for e in upstream:
            print(f"  {e.id}  [{', '.join(c[:12] for c in e.upstream_commits)}]")
    for e in manifest.ENTRIES:
        print(f"  [{e.category:>2}] {e.apply_phase:<11} {e.id}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="musa_sync", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "apply", help="build-time: apply the diff series to a cloned vLLM"
    )
    p.add_argument("repo")
    p.add_argument("--phase", default=None, help="restrict to one apply_phase")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--no-strict", action="store_true")
    p.set_defaults(func=cmd_apply)

    p = sub.add_parser(
        "verify", help="offline pre-bump gate: status of every divergence"
    )
    p.add_argument(
        "--target",
        default=None,
        help="vLLM ref (default: VLLM_COMMIT, falling back to VLLM_TAG from PINS)",
    )
    p.add_argument(
        "--repo", default=None, help="use an existing checkout instead of cloning"
    )
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser(
        "check-series",
        help="gate the series' generation form: every entry must be a git am mailbox",
    )
    p.add_argument(
        "--repo",
        default=None,
        help="also resolve every `index` blob in that checkout (no repo: format "
        "half only)",
    )
    p.add_argument(
        "--replay",
        action="store_true",
        help="replay the series with `git am -3` in a disposable clone of "
        "--repo (give a checkout AT THE PIN) and additionally require every "
        "declared postimage to exist afterwards and the series to apply the way "
        "the build applies it (`git apply --recount -p1`)",
    )
    p.add_argument(
        "--round-trip",
        action="store_true",
        help="compare the series with what `regen` writes from --repo (give a "
        "checkout HOLDING the series, i.e. after `rebase`) and fail on any "
        "entry it would rewrite; mutually exclusive with --replay, which needs "
        "a checkout in the opposite state",
    )
    p.set_defaults(func=cmd_check_series)

    p = sub.add_parser(
        "rebase", help="git am -3 the series onto vllm@<ref> in third_party/vllm"
    )
    p.add_argument("tag", metavar="ref", nargs="?", default=None)
    p.add_argument(
        "--drop-upstreamed",
        action="store_true",
        help="skip the entries `upgrade-plan` drops for <ref> (the upstream "
        "commits <ref> already has)",
    )
    p.add_argument(
        "--history",
        default=None,
        help="vLLM clone with full history for --drop-upstreamed "
        "(default: third_party/vllm)",
    )
    p.add_argument(
        "--fetch", action="store_true", help="fetch missing commits from origin"
    )
    p.set_defaults(func=cmd_rebase)

    p = sub.add_parser(
        "upgrade-plan",
        help="per entry: keep, drop or review on a pin bump to <ref>",
    )
    p.add_argument(
        "target",
        nargs="?",
        default=None,
        help="vLLM ref of the new pin (default: the current pin, which lists "
        "entries the pin already contains)",
    )
    p.add_argument(
        "--history",
        default=None,
        help="vLLM clone with the full history of <ref> (default: third_party/vllm)",
    )
    p.add_argument(
        "--fetch", action="store_true", help="fetch missing commits from origin"
    )
    p.add_argument(
        "--no-content",
        action="store_true",
        help="skip the per-entry apply check against the pristine target",
    )
    p.add_argument("--all", action="store_true", help="also list the kept entries")
    p.set_defaults(func=cmd_upgrade_plan)

    p = sub.add_parser(
        "backport",
        help="cherry-pick an upstream vLLM commit into series/upstream/",
    )
    p.add_argument("commit", help="upstream vLLM commit (full sha to fetch it)")
    p.add_argument(
        "--from",
        dest="source",
        default=None,
        help="fetch the commit from this vLLM clone instead of origin",
    )
    p.add_argument(
        "--no-regen", action="store_true", help="do not regenerate the series"
    )
    p.set_defaults(func=cmd_backport)

    p = sub.add_parser("regen", help="regenerate the series from the clone's commits")
    p.add_argument("--area", default=None, help="(reserved) py|csrc|module")
    p.set_defaults(func=cmd_regen)

    p = sub.add_parser("report", help="manifest census")
    p.add_argument(
        "--doc", action="store_true", help="render the Markdown census table"
    )
    p.add_argument(
        "--metadata", action="store_true", help="show validated lineage and group requirements"
    )
    p.set_defaults(func=cmd_report)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return args.func(args)
    except GitMissing as exc:
        print(f"ERROR: {exc}")
        return 1
    except OSError as exc:
        # An unusable file on the way to a gate (an entry series/ globbed that
        # is not a readable regular file, say) ends in a verdict, never in a
        # traceback: a caller parsing the report must be able to trust the
        # last line.
        print(f"ERROR: {exc}")
        print(f"=== musa_sync {args.cmd}: FAIL (os-error) ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
