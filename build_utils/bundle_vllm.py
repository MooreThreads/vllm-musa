# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the patched upstream vLLM payload for the vllm-musa wheel."""

from __future__ import annotations

import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from packaging.version import Version

BUNDLED_DISTRIBUTION = "vllm-musa"
EXPECTED_METADATA_REWRITES = 5
_VLLM_NATIVE_EXTENSIONS = {"_C", "_C_stable_libtorch", "_moe_C"}

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
        ignored = {
            name
            for name in names
            if name == "__pycache__" or name.endswith((".pyc", ".pyo"))
        }
        if Path(directory) == source:
            ignored.update(
                name
                for name in names
                if name.endswith(".so")
                and name.partition(".")[0] in _VLLM_NATIVE_EXTENSIONS
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


def find_prebuilt_extensions(
    extension_paths: Sequence[Path], source_roots: Sequence[Path]
) -> dict[Path, Path]:
    """Find each native output by its exact build-relative path."""
    sources = {}
    missing = []
    for extension_path in extension_paths:
        for source_root in source_roots:
            source = source_root / extension_path
            if source.is_file():
                sources[extension_path] = source
                break
        else:
            missing.append(str(extension_path))
    if missing:
        raise RuntimeError(
            "VLLM_MUSA_REUSE_PREBUILT=1 requires prepared extension outputs, "
            f"missing: {', '.join(missing)}"
        )
    return sources


def bundle_vllm_package(
    source: Path, destination: Path, *, tag: str, revision: str
) -> BundleResult:
    """Copy patched vLLM into a wheel staging tree.

    The native build hook supplies the C++/MUSA extensions. Rust artifacts
    travel with the source package when present.
    """
    if not (source / "__init__.py").is_file():
        raise RuntimeError(f"Prepared vLLM package is missing: {source}")

    version = Version(tag)
    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(source, destination, ignore=_copy_ignore(source))
    (destination / "_version.py").write_text(
        f"__version__ = version = {str(version)!r}\n"
        f"__version_tuple__ = version_tuple = {version.release!r}\n"
        f"__commit_id__ = commit_id = {revision!r}\n",
        encoding="utf-8",
    )
    metadata_rewrites = _rewrite_distribution_metadata(destination)
    files = sum(path.is_file() for path in destination.rglob("*"))
    return BundleResult(files=files, metadata_rewrites=metadata_rewrites)
