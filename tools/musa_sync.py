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

    check-series [--repo PATH]
        Fail-closed form gate on series/ — the ``regen`` fixed point. Every
        entry must be a canonical ``git format-patch`` mailbox (all-zero
        separator, ``From:`` author, an ``index`` line, a diff git can parse);
        ``--repo`` additionally resolves the blobs those ``index`` lines
        declare. The format half needs no repository, so it runs on every PR.

Every subcommand ends with one explicit ``=== musa_sync <cmd>: PASS|FAIL ===``
verdict line whose counts agree with the exit code: 0 = PASS, 1 = FAIL, 2 =
usage/config error (e.g. ``verify`` with no resolvable target).

Stdlib-only; loads manifest.py + build_apply.py BY FILE PATH so it never imports
the ``vllm_musa`` package (works before install, in plain CI).
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import re
import shutil
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


def _declared_blobs(text: bytes) -> list[tuple[str, str]]:
    """``(preimage, postimage)`` blob ids from every ``index`` line in an entry.

    The whole entry is searched, not just its mailbox header: a commit message
    with a body pushes the first ``index`` line far below the separator (line 22
    in the longest shipped entry), so a first-12-lines scan reads clean on
    entries that have no ``index`` line at all.
    """
    return [
        (m.group(1).decode(), m.group(2).decode()) for m in _INDEX_LINE.finditer(text)
    ]


def _repo_has_blobs(repo: Path, blobs: list[str]) -> set[str]:
    """The subset of ``blobs`` that ``repo`` can resolve.

    One ``git cat-file --batch-check`` for the whole series rather than one
    process per blob: the shipped series declares 355 distinct ids, which would
    otherwise cost ~1.4 s of process spawns.
    """
    if not blobs:
        return set()
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
            f"git cat-file in {repo} failed: {(r.stderr or r.stdout).strip()[:100]}"
        )
    return {
        blob
        for blob, line in zip(blobs, r.stdout.splitlines())
        if not line.endswith(" missing")
    }


def _missing_index_blobs(
    repo: Path, texts: dict[str, bytes]
) -> dict[str, list[str]]:
    """Entry name -> declared blob ids this repo cannot resolve.

    An entry's ``index`` line records the blobs its hunks are anchored to, and
    ``git am -3`` uses them to build a real 3-way ancestor — ids left behind by
    a series regenerated against some *other* base make the entry replay as a
    plain apply or not at all.

    A blob the series itself creates (any entry's postimage) is *expected* to be
    absent from a checkout that never had the series applied: only
    ``rebase``/``regen`` put those objects in the odb. Measured on the shipped
    171-entry series, a fresh shallow clone at the pin resolves 133/225
    preimages and 0/230 postimages, yet ``git am -3`` replays all 171 of them —
    so "missing from this repo" alone is not a defect, while "missing and not
    produced by the series" is. Null ids (``0000000000``, a creation hunk) are
    never resolvable and never required.
    """
    produced = {
        post
        for text in texts.values()
        for _, post in _declared_blobs(text)
        if not _ZERO_BLOB.match(post.encode())
    }
    wanted: dict[str, set[str]] = {}
    for name, text in texts.items():
        for pre, post in _declared_blobs(text):
            # A postimage is always in `produced` (the entry creates it), so in
            # practice only the preimages reach the repo: they are the ids the
            # entry must anchor to.
            for blob in (pre, post):
                if _ZERO_BLOB.match(blob.encode()) or blob in produced:
                    continue
                wanted.setdefault(blob, set()).add(name)
    found = _repo_has_blobs(repo, sorted(wanted))
    out: dict[str, list[str]] = {}
    for blob in sorted(set(wanted) - found):
        for name in sorted(wanted[blob]):
            out.setdefault(name, []).append(blob)
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
    if not _INDEX_LINE.search(text):
        return (
            "no-index-line",
            "no 'index <blob>..<blob>' line: git am -3 cannot build its 3-way "
            "ancestor (sha1 information is lacking or useless)",
        )
    if unresolved:
        return (
            "missing-index-blob",
            f"index blob(s) {', '.join(unresolved)} cannot be resolved in the "
            "given repo and are not produced by this series: the entry cannot "
            "replay against that pin",
        )
    return None


def _series_format_rows(repo: Path | None = None) -> list[tuple[str, str, str]]:
    """``(patch name, status, detail)`` for every entry in ``series/``.

    The build applies the series with ``git apply --recount -p1``
    (``build_apply.py``), which silently repairs wrong hunk counts and accepts
    bare diffs — but the *generation* path (``rebase``/``regen``) needs real
    mailboxes that ``git am`` can consume. An entry that is hand-edited or a
    bare diff therefore passes every build-side gate and only breaks the next
    version bump, which is exactly how the series rotted before MUSA-100050.

    Cheap offline checks, first problem wins:
      * the entry is a ``git format-patch`` mailbox with the all-zero separator
        ``regen`` writes (a real-sha mbox is replayable but not a fixed point);
      * it carries a ``From: `` author ident, without which ``git am`` dies on
        ``empty ident name``;
      * it carries at least one ``index <blob>..<blob>`` line (searched over the
        whole entry, not just the header), without which ``git am -3`` dies on
        ``sha1 information is lacking or useless``;
      * with ``repo``, every non-null blob those ``index`` lines declare is
        resolvable there or produced by the series itself
        (``_missing_index_blobs``);
      * ``git apply --stat`` parses it (catches hunk counts that no longer
        match the body);
      * the series' numbering is unique and contiguous
        (``_series_numbering_rows``).

    ``repo`` is optional so the gate stays usable where cloning is not (a PR
    runner with no checkout): without it the blob half is skipped and only the
    format half runs.
    """
    paths, refusal = _series_dir_paths()
    if refusal:
        return [(str(SERIES_DIR), refusal[0], refusal[1])]
    status: dict[str, tuple[str, str]] = {}
    texts: dict[str, bytes] = {}
    for patch in paths:
        try:
            texts[patch.name] = patch.read_bytes()
        except OSError as exc:
            status[patch.name] = ("unreadable", str(exc))
    try:
        unresolved = _missing_index_blobs(Path(repo), texts) if repo else {}
        for patch in paths:
            text = texts.get(patch.name)
            if text is None:  # already rowed as unreadable
                continue
            problem = _series_entry_problem(text, unresolved.get(patch.name, []))
            if problem is None:
                parsed = _run_git(ROOT, "apply", "--stat", str(patch))
                if parsed.returncode:
                    problem = ("corrupt", (parsed.stderr or parsed.stdout).strip()[:100])
            status[patch.name] = problem or ("clean", "")
    except GitMissing as exc:
        return [(str(SERIES_DIR), "git-unavailable", str(exc))]
    except RepoUnusable as exc:
        return [(str(SERIES_DIR), "repo-unusable", str(exc))]
    return [(patch.name, *status[patch.name]) for patch in paths]


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
    report (a missing or empty series has no numbering)."""
    if number_rows:
        return f"--- numbering: {len(number_rows)} problem(s) ---"
    if not rows or any(status in _SERIES_FATAL for _, status, _ in rows):
        return None
    return (
        f"--- numbering: {len(rows)} entries are unique and contiguous "
        f"0001..{len(rows):04d} ---"
    )


def _print_series_section(
    rows, number_rows, *, label: str = "", limit: int | None = None
) -> None:
    """Print the series-format evidence: bad rows with their detail, the clean
    summary, then numbering.

    Shared by ``check-series`` and ``verify`` so a red series looks the same in
    both, including the ``… N more`` marker when ``limit`` truncates the rows.
    """
    bad = [r for r in rows if r[1] != "clean"]
    shown = bad if limit is None else bad[:limit]
    for name, status, detail in shown:
        print(f"  {status:<14} {name}")
        if detail:
            print(f"                 {detail}")
    if len(shown) < len(bad):
        print(f"  … {len(bad) - len(shown)} more (run `check-series` for the full list)")
    print(
        f"--- {label}{len(rows) - len(bad)} clean / {len(rows)} total / "
        f"{len(bad)} need attention ---"
    )
    for name, status, detail in number_rows:
        print(f"  {status:<14} {name}")
        if detail:
            print(f"                 {detail}")
    line = _numbering_line(rows, number_rows)
    if line:
        print(line)


def cmd_check_series(args) -> int:
    rows = _series_format_rows(args.repo)
    number_rows = _series_numbering_rows(rows)
    bad = [r for r in rows if r[1] != "clean"]
    if any(status in _SERIES_FATAL for _, status, _ in rows):
        print(f"=== musa_sync check-series: {SERIES_DIR} cannot be gated ===")
    else:
        print(f"=== musa_sync check-series: {len(rows)} entries in series/ ===")
    _print_series_section(rows, number_rows)
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
        print("ERROR: VLLM_COMMIT or VLLM_TAG is required in third_party/PINS")
        return 1
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


if __name__ == "__main__":
    sys.exit(main())
