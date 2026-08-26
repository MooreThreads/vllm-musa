# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the patched upstream vLLM payload for the vllm-musa wheel."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

BUNDLED_DISTRIBUTION = "vllm-musa"
EXPECTED_METADATA_REWRITES = 5

_VLLM_METADATA_LOOKUP = re.compile(
    r"(?P<prefix>\b(?:importlib\.metadata\.)?"
    r"(?:version|metadata|distribution)\(\s*)"
    r"(?P<quote>['\"])vllm(?P=quote)"
)


@dataclass(frozen=True)
class BundleResult:
    files: int
    metadata_rewrites: int


def _copy_ignore(source: Path):
    def ignore(directory: str, names: list[str]) -> set[str]:
        directory_path = Path(directory)
        ignored = {
            name
            for name in names
            if name == "__pycache__" or name.endswith((".pyc", ".pyo"))
        }
        if directory_path.resolve() == source.resolve():
            ignored.update(
                name
                for name in names
                if name.endswith(".so")
                and name.startswith(("_C.", "_C_stable_libtorch.", "_moe_C."))
            )
        return ignored

    return ignore


def _rewrite_distribution_metadata(package_dir: Path) -> int:
    replacements = 0
    for path in package_dir.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        bundled, count = _VLLM_METADATA_LOOKUP.subn(
            rf"\g<prefix>\g<quote>{BUNDLED_DISTRIBUTION}\g<quote>", source
        )
        if count:
            path.write_text(bundled, encoding="utf-8")
            replacements += count

    if replacements != EXPECTED_METADATA_REWRITES:
        raise RuntimeError(
            "Unexpected number of vLLM distribution metadata lookups: "
            f"expected {EXPECTED_METADATA_REWRITES}, found {replacements}"
        )
    return replacements


def bundle_vllm_package(source: Path, destination: Path) -> BundleResult:
    """Copy patched vLLM into a wheel staging tree.

    The vllm-musa build owns the three C++/MUSA extensions, so stale copies of
    those files are excluded. Prebuilt Rust artifacts remain part of the
    payload when the source staging tree provides them.
    """
    if not (source / "__init__.py").is_file():
        raise RuntimeError(f"Prepared vLLM package is missing: {source}")
    if not (source / "_version.py").is_file():
        raise RuntimeError(f"Generated vLLM version module is missing: {source}")

    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(source, destination, ignore=_copy_ignore(source))
    metadata_rewrites = _rewrite_distribution_metadata(destination)
    files = sum(path.is_file() for path in destination.rglob("*"))
    return BundleResult(files=files, metadata_rewrites=metadata_rewrites)
