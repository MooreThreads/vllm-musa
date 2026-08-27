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

Stdlib-only; loads manifest.py + build_apply.py BY FILE PATH so it never imports
the ``vllm_musa`` package (works before install, in plain CI).
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
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


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )


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
    finally:
        if temp:
            shutil.rmtree(clone.parent, ignore_errors=True)
    print(f"=== musa_sync verify: {len(rows)} divergences vs vllm@{target} ===")
    for did, cat, status, detail in rows:
        line = f"  [{cat:>2}] {status:<14} {did}"
        if detail:
            line += f"   ({detail})"
        print(line)
    bad = [r for r in rows if r[2] in _BAD]
    n_clean = sum(1 for r in rows if r[2] == "clean")
    print(
        f"--- {n_clean} clean / {len(rows)} total / {len(bad)} need attention "
        f"({', '.join(sorted({r[2] for r in bad})) or 'none'}) ---"
    )
    return 1 if bad else 0


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


def main(argv: list[str] | None = None) -> int:
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

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
