#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Make this checkout's Omni compat shim available to the base image's vllm-musa.

Once the shim ships in a vLLM-MUSA release the base image already carries it and
this is a no-op. Until then -- and whenever an Omni ref needs a shim entry newer
than the base was built with -- the shim has to reach an installed vllm-musa
that predates it.

Both cases are the same operation: copy ``vllm_musa/omni_compat/`` and the
top-level ``.pth`` bootstrap into the installed package, and refuse if the files
differ in a way that suggests the base is actually newer than this checkout.
Only pure-Python files move, so nothing is rebuilt and the base's compiled
extensions are untouched.
"""

from __future__ import annotations

import argparse
import filecmp
import importlib.util
import shutil
import sysconfig
from pathlib import Path


def installed_package_dir() -> Path:
    spec = importlib.util.find_spec("vllm_musa")  # does not execute __init__
    locations = (
        list(getattr(spec, "submodule_search_locations", None) or []) if spec else []
    )
    if not locations:
        raise SystemExit("vllm_musa is not installed in this interpreter")
    return Path(locations[0])


def purelib() -> Path:
    """Where a top-level module has to go to be importable.

    Not next to the package. A PEP 660 editable install of the kind that writes
    an import-hook finder (``__editable___vllm_musa_*_finder.py``, whose
    ``MAPPING`` names each distributed package rather than adding the source
    tree to ``sys.path``) maps exactly what the project declared, and the base
    image's vllm-musa was built before this shim existed,
    so its mapping names the ``vllm_musa`` package and nothing else. Its source
    directory is never on ``sys.path``, so a top-level module dropped beside the
    package is unreachable. A checkout
    that declares the bootstrap in ``py-modules`` does get it mapped, but that is
    the case where this staging step is already a no-op.
    """
    return Path(sysconfig.get_paths()["purelib"])


def _compare_trees(src: Path, target: Path) -> list[str]:
    """Every difference between two shim trees, including nested ones.

    ``filecmp.dircmp``'s ``left_only``/``right_only``/``diff_files`` describe
    the top level only, so a changed file under ``adapters/`` or ``backports/``
    -- where most of the shim lives -- compares as identical. This walks.
    """
    # filecmp memoizes on the (path, stat-signature) pair, and copying with
    # copy2 reproduces the source's size and mtime, so a comparison made after
    # a copy can be answered from a result computed before it.
    filecmp.clear_cache()
    differences: list[str] = []
    cmp = filecmp.dircmp(str(src), str(target), ignore=["__pycache__"])
    for name in cmp.left_only:
        differences.append(f"only in this checkout: {name}")
    for name in cmp.right_only:
        differences.append(f"only in the installed tree: {name}")
    # shallow=False: compare contents, not the (size, mtime) signature
    # ``dircmp.diff_files`` uses. Two revisions of a shim file are often the
    # same size, and a tree that arrived through a Docker COPY can carry
    # identical timestamps, so the signature says "same" for files that differ.
    _, mismatch, errors = filecmp.cmpfiles(
        str(src), str(target), cmp.common_files, shallow=False
    )
    for name in mismatch:
        differences.append(f"differs: {name}")
    for name in errors:
        differences.append(f"uncomparable: {name}")
    for name in cmp.common_dirs:
        for nested in _compare_trees(src / name, target / name):
            differences.append(f"{name}/{nested}")
    return differences


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="vllm-musa checkout in the build context",
    )
    args = parser.parse_args(argv)

    src_pkg = args.source / "vllm_musa" / "omni_compat"
    src_pth = args.source / "vllm_musa_omni_compat_pth.py"
    for path in (src_pkg, src_pth):
        if not path.exists():
            raise SystemExit(f"missing in the build context: {path}")

    target_pkg = installed_package_dir() / "omni_compat"
    target_pth = purelib() / "vllm_musa_omni_compat_pth.py"

    if target_pkg.exists():
        differences = _compare_trees(src_pkg, target_pkg)
        if not differences:
            print(f"shim already installed and identical: {target_pkg}")
            return 0
        # This checkout is the authority: an older shim in the base is exactly
        # the case this step exists for, and the Omni ref being built may need
        # a table row the base has never heard of.
        print(f"replacing the base's shim at {target_pkg}: {', '.join(differences)}")
        shutil.rmtree(target_pkg)
    shutil.copytree(src_pkg, target_pkg, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(src_pth, target_pth)
    print(f"staged {target_pkg}")
    print(f"staged {target_pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
