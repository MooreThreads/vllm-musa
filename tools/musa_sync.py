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

    rebase <ref>
        Checkout the clone at <ref> and ``git am -3`` the series in order (true
        3-way; trivial upstream drift auto-merges, real conflicts halt).

    regen
        Regenerate the series from the clone's commits
        (``git format-patch --no-signature --no-numbered --zero-commit``,
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
        ``0001``, and no two entries may share a diff body. ``--repo``
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

        Neither is the default: both clone a repository and spawn git per entry.
        No in-repo CI or hook invokes this gate today.

Every **gate** subcommand (``check-series``, ``verify``, ``regen``, ``rebase``,
``module``) ends with one explicit ``=== musa_sync <cmd>: PASS|FAIL ===`` verdict
line whose counts agree with the exit code: 0 = PASS, 1 = FAIL, 2 = usage/config
error (e.g. ``verify`` with no resolvable target). The two reporting commands do
not print one: ``report`` renders the manifest census and ``apply`` prints
``--- N applied, … ---``, so a caller parsing either must read the exit code.

Stdlib-only; loads manifest.py + build_apply.py BY FILE PATH so it never imports
the ``vllm_musa`` package (works before install, in plain CI).
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import re
from email.header import decode_header, make_header
import shutil
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # tools/ -> repo root
PATCHES = ROOT / "vllm_musa" / "patches"
SERIES_DIR = PATCHES / "series"
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
SUBCOMMANDS = ("apply", "verify", "check-series", "rebase", "regen", "report")
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


def _load(name: str, path: Path):
    """Load a stdlib-only helper module by file path (no vllm_musa import)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register before exec (safe by-path load)
    spec.loader.exec_module(mod)
    return mod


manifest = _load("musa_mdm_manifest", PATCHES / "manifest.py")
build_apply = _load("musa_mdm_build_apply", PATCHES / "build_apply.py")


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


def _series_dir_paths() -> tuple[list[Path], tuple[str, str] | None]:
    """The series entries, or a ``(status, detail)`` refusal when there is
    nothing to gate.

    A missing or empty ``series/`` is a *failure*, never a vacuous pass: a gate
    that globs zero entries reports "0 clean / 0 total / 0 need attention" and
    exits 0, which is how a deleted series directory stays green.
    """
    if not SERIES_DIR.is_dir():
        return [], ("missing-series-dir", f"no series directory at {SERIES_DIR}")
    paths = sorted(SERIES_DIR.glob("*.patch"))
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
        texts[patch.name] = patch.read_bytes()
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
    start = m.start()
    # A bare diff still carries its header lines (`index`, modes, rename/copy),
    # and the `index` line in particular is below the separator but *above* the
    # `--- a/…` pair this fallback found — so walk back over exactly those.
    header = re.compile(
        rb"(?m)^(index |new file mode |deleted file mode |old mode |new mode "
        rb"|similarity index |dissimilarity index |rename from |rename to "
        rb"|copy from |copy to )"
    )
    while start > 0:
        line_start = text.rfind(b"\n", 0, start - 1) + 1
        if line_start >= start or not header.match(text[line_start:start]):
            break
        start = line_start
    return text[start:]


def _declared_postimages(text: bytes) -> list[tuple[str, str]]:
    """``(path, postimage)`` for every ``index`` line, in diff order.

    The path comes from the section's ``diff --git a/… b/…`` header, or from its
    ``+++ b/…`` line for a bare diff; a deletion has no postimage and is skipped.
    This is what lets the replay check an entry's declaration against *the commit
    that entry created* rather than against the object database.
    """
    body = _diff_body(text)
    out: list[tuple[str, str]] = []
    # Split into file sections; a bare diff is one section with no header.
    sections = re.split(rb"(?m)^(?=diff --git )", body)
    for section in sections:
        if not section.strip():
            continue
        m = re.search(rb"(?m)^diff --git a/([^\n]+) b/([^\n]+)\n", section)
        if m:
            path = m.group(2).decode("utf-8", "replace")
        else:
            plus = re.search(rb"(?m)^\+\+\+ b/([^\n]+)\n", section)
            if not plus:
                continue
            path = plus.group(1).decode("utf-8", "replace")
        if re.search(rb"(?m)^\+\+\+ /dev/null\n", section):
            continue
        for m2 in _INDEX_LINE.finditer(section):
            out.append((path, m2.group(2).decode()))
    return out


def _declared_blobs(text: bytes) -> list[tuple[str, str]]:
    """``(preimage, postimage)`` blob ids from every ``index`` line in the diff."""
    body = _diff_body(text)
    return [
        (m.group(1).decode(), m.group(2).decode())
        for m in _INDEX_LINE.finditer(body)
    ]


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
    objects in the odb. Measured on the shipped 171-entry series, a fresh clone
    at the pin resolves 133/225 preimages and 0/230 postimages, yet ``git am -3``
    replays all 171 of them — so "missing from this repo" alone is not a defect,
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
            "no 'index <blob>..<blob>' line: git am -3 cannot build its 3-way "
            "ancestor (sha1 information is lacking or useless)",
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
    for patch in paths:
        problem = _entry_name_problem(patch.name) or _read_entry(patch, texts)
        if problem is not None:
            status[patch.name] = problem
    order = [patch.name for patch in paths]
    try:
        unresolved = _missing_index_blobs(Path(repo), texts, order) if repo else {}
        duplicates = _duplicate_bodies(texts)
        for patch in paths:
            if patch.name in status:
                continue
            text = texts[patch.name]
            problem = (
                _mailbox_form_problem(text)
                or _series_entry_problem(text, unresolved.get(patch.name, []))
                or _canonical_form_problem(text, patch.name, duplicates)
            )
            if problem is None:
                parsed = _run_git(ROOT, "apply", "--stat", str(patch))
                if parsed.returncode:
                    problem = ("corrupt", (parsed.stderr or parsed.stdout).strip()[:100])
            status[patch.name] = problem or ("clean", "")
        rows = [
            (_safe_name(patch.name), *status[patch.name]) for patch in paths
        ]
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

    Exactly what ``cmd_regen`` runs (``git format-patch --no-signature
    --no-numbered --zero-commit`` plus the canonical author rewrite), so the
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
    r = _git(
        repo,
        "format-patch",
        "--no-signature",
        "--no-numbered",
        "--zero-commit",
        "-o",
        str(staged),
        base,
    )
    if r.returncode:
        return [
            (
                "series replay",
                "round-trip-dirty",
                f"`regen` (git format-patch) failed: {_git_first_error(r)}",
            )
        ]
    generated = sorted(staged.glob("*.patch"))
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
    for patch in generated:
        _normalize_patch_author(patch)  # what cmd_regen does before comparing
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
    rows: list[tuple[str, str, str]] = []
    for entry, regen in zip(paths, generated):
        want, got = entry.read_bytes(), regen.read_bytes()
        if entry.name == regen.name and want == got:
            continue
        if entry.name != regen.name:
            rows.append(
                (
                    _safe_name(entry.name),
                    "round-trip-dirty",
                    f"`regen` names this entry {regen.name}: the filename is "
                    "not a fixed point",
                )
            )
        else:
            rows.append(
                (
                    _safe_name(entry.name),
                    "round-trip-dirty",
                    "`regen` rewrites this entry: "
                    f"{_first_diff_line(want, got)}",
                )
            )
    for regen in generated[len(paths) :]:
        rows.append(
            (
                _safe_name(regen.name),
                "round-trip-dirty",
                "`regen` produces this entry; series/ has no file with that name",
            )
        )
    for entry in paths[len(generated) :]:
        rows.append(
            (
                _safe_name(entry.name),
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
            head_before = _git(clone, "rev-parse", "HEAD").stdout.strip()
            r = _git(clone, "am", "-3", str(patch))
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
                        _safe_name(patch.name),
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
                if applied == 0 and texts.get(patch.name, b"").strip():
                    detail += (
                        f" (the first entry that applies fails here, after "
                        f"{applied} applied: if this checkout already holds the "
                        "series, replay needs one at the pin, and `--round-trip` "
                        "is the mode for the maintenance state)"
                    )
                return [(_safe_name(patch.name), "replay-failed", detail)]
            applied += 1
            made[patch.name] = head_after
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

    Measured on the shipped series: 0 of the 230 declared postimages resolve in a
    fresh clone at the pin, and 230/230 are the declared path's blob in the very
    commit that declares them, so the strict reading costs nothing here.
    """
    declared: dict[tuple[str, str], str] = {}
    for patch in paths:
        text = texts.get(patch.name)
        if text is None:
            continue
        for path, post in _declared_postimages(text):
            if not _ZERO_BLOB.match(post.encode()):
                declared.setdefault((patch.name, post), path)
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
                    _safe_name(patch.name),
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
    """Series-level numbering: every entry numbered, unique, and gapless.

    ``regen`` numbers the entries itself (``git format-patch --no-numbered``
    plus a contiguous renumber), so a duplicate or a gap can only come from
    hand numbering. Both are silent everywhere else: two PRs each pick "the
    next free number" in isolation, each one applies cleanly, and the collision
    surfaces only when somebody regenerates — or not at all, if a duplicate
    entry happens to be applied by prefix order.

    The expected range comes from the numbers *present* (``max``), not from the
    entry count, so ``0001``+``0003`` reports the real gap instead of a phantom
    trailing one; the numbers present are listed alongside it whenever the set
    is not exactly ``0001..max``.

    ``NNNN`` is 1-based, so a ``0000`` prefix is a ``bad-range`` row of its own
    rather than a green series whose verdict line then claims
    ``contiguous 0001..0001``.

    Out of scope by design: this reads names only, so a rename, or a content
    swap that preserves the set of prefixes, is invisible here — the content
    gates in ``_series_format_rows`` are what a swapped entry has to satisfy.
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
        expected = {f"{i:04d}" for i in range(1, int(numbered[-1]) + 1)}
        missing = sorted(expected - set(numbered))
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
    present = sorted(
        n for n in (_number_prefix(name) for name, _, _ in rows) if n is not None
    )
    if not present:  # unnumbered entries: already rowed, so unreachable
        return None
    return (
        f"--- numbering: {len(rows)} entries are unique and contiguous "
        f"{present[0]}..{present[-1]} ---"
    )


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
        if (SERIES_DIR / row[0]).is_file():
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
    repo = Path(args.repo).resolve() if args.repo else None
    rows = _series_format_rows(repo, replay=args.replay, round_trip=args.round_trip)
    # Numbering and the per-entry counts read the *files* in series/, so they
    # skip the series-level rows the deep modes can add (see _entry_file_rows);
    # the verdict below still fails on every row.
    entries = _entry_file_rows(rows)
    number_rows = _series_numbering_rows(entries)
    bad = [r for r in rows if r[1] != "clean"]
    if any(status in _SERIES_FATAL for _, status, _ in rows):
        print(f"=== musa_sync check-series: {SERIES_DIR} cannot be gated ===")
    else:
        print(f"=== musa_sync check-series: {len(entries)} entries in series/ ===")
    _print_series_section(rows, number_rows, entries=entries)
    verdict = (
        "PASS"
        if not (bad or number_rows)
        else f"FAIL (series-format={len(bad)}, numbering={len(number_rows)})"
    )
    print(f"=== musa_sync check-series: {verdict} ===")
    return 1 if (bad or number_rows) else 0


def cmd_verify(args) -> int:
    target = args.target or _default_target()
    if not target:
        print(
            "ERROR: no target ref (pass --target or set VLLM_COMMIT/VLLM_TAG "
            "in third_party/PINS)"
        )
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
    number_rows = _series_numbering_rows(fmt_rows)
    fmt_bad = [r for r in fmt_rows if r[1] != "clean"]
    bad = [r for r in rows if r[2] in _BAD]
    n_clean = sum(1 for r in rows if r[2] == "clean")
    print(f"=== musa_sync verify: {len(rows)} divergences vs vllm@{target} ===")
    # Series format comes first, and the divergence summary stays last of the
    # summaries: the documented success line is the one a reader checks, so a
    # red series run must not end on `0 need attention`.
    _print_series_section(fmt_rows, number_rows, label="series format: ", limit=10)
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
    if _checkout(target):
        return 1
    order = manifest.series_apply_order()
    for patch in order:
        r = _git(WORKDIR, "am", "-3", str(patch))
        if r.returncode != 0:
            print(f"CONFLICT: {patch.name}\n{r.stdout}\n{r.stderr}")
            print(
                "Resolve in third_party/vllm, then "
                "`git -C third_party/vllm am --continue`, then re-run; or "
                "`git -C third_party/vllm am --abort` to back out."
            )
            return 1
    print(f"rebased {len(order)} patches onto vllm@{target} (git am -3)")
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
    if args.area == "module":
        return _regen_module_tripwires()
    target = _default_target()
    if not target:
        print(
            "ERROR: no target ref: set VLLM_COMMIT or VLLM_TAG in third_party/PINS "
            "(or pass --target)"
        )
        print("=== musa_sync regen: FAIL (usage) ===")
        return 2
    with tempfile.TemporaryDirectory(prefix="musa-regen-series-") as tmp:
        staged = Path(tmp)
        r = _git(
            WORKDIR,
            "format-patch",
            "--no-signature",
            "--no-numbered",
            "--zero-commit",
            "-o",
            str(staged),
            target,
        )
        if r.returncode:
            print(r.stderr)
            return 1

        generated = sorted(staged.glob("*.patch"))
        if not generated:
            print(f"ERROR: no patches generated from vllm@{target}..HEAD")
            return 1

        for patch in generated:
            _normalize_patch_author(patch)

        numbers = [p.name.split("-", 1)[0] for p in generated]
        expected = [f"{i:04d}" for i in range(1, len(generated) + 1)]
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

        SERIES_DIR.mkdir(parents=True, exist_ok=True)
        existing = list(SERIES_DIR.glob("*.patch"))
        generated_names = {p.name for p in generated}
        for patch in generated:
            shutil.copyfile(patch, SERIES_DIR / patch.name)
        stale = [p for p in existing if p.name not in generated_names]
        for patch in stale:
            patch.unlink()

    print(
        f"regenerated {len(generated)} contiguous patches in {SERIES_DIR} "
        f"from vllm@{target}..HEAD; pruned {len(stale)} stale files"
    )
    return 0


# -------------------------------------------------------------------------- report
def cmd_report(args) -> int:
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
    p.set_defaults(func=cmd_rebase)

    p = sub.add_parser("regen", help="regenerate the series from the clone's commits")
    p.add_argument("--area", default=None, help="(reserved) py|csrc|module")
    p.set_defaults(func=cmd_regen)

    p = sub.add_parser("report", help="manifest census")
    p.add_argument(
        "--doc", action="store_true", help="render the Markdown census table"
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
