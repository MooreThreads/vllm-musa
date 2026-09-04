#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Discover physical kernel definitions owned by the vLLM-MUSA repository."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA = "vllm-musa-owned-physical-kernel-inventory.v1"
NATIVE_ROOTS = (
    ("aot-native-physical", Path("csrc/musa"), ("*.mu", "*.cu")),
    ("jit-native-physical", Path("vllm_musa/jit_kernel/csrc"), ("*.mu",)),
)
TILELANG_ROOTS = (
    Path("vllm_musa/deepseek_v4_jit"),
    Path("vllm_musa/jit_kernel/tilelang"),
)
TILELANG_FILES = (Path("vllm_musa/v1/attention/ops/sparse_mla_tilelang.py"),)


@dataclass(frozen=True, order=True)
class PhysicalKernelEntry:
    id: str
    backend: str
    source: str
    symbol: str
    line: int
    owner: str | None
    discovery: str


def _source_files(root: Path, patterns: tuple[str, ...]) -> Iterable[Path]:
    if not root.exists():
        return ()
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in root.rglob(pattern) if path.is_file())
    return sorted(paths)


def _entry(
    backend: str,
    source: Path,
    symbol: str,
    line: int,
    owner: str | None,
    discovery: str,
) -> PhysicalKernelEntry:
    normalized_source = source.as_posix()
    return PhysicalKernelEntry(
        id=f"{backend}:{normalized_source}:{symbol}@{line}",
        backend=backend,
        source=normalized_source,
        symbol=symbol,
        line=line,
        owner=owner,
        discovery=discovery,
    )


def _native_global_symbols(text: str) -> list[tuple[str, int]]:
    symbols: list[tuple[str, int]] = []
    for match in re.finditer(r"\b__global__\b(?P<tail>.*?)(?:\{|;)", text, re.S):
        signature = re.sub(r"__launch_bounds__\s*\([^)]*\)", " ", match.group("tail"))
        names = re.findall(r"\b([A-Za-z_]\w*)\s*\(", signature)
        if names:
            line = text.count("\n", 0, match.start()) + 1
            symbols.append((names[0], line))
    return sorted(symbols)


def discover_native(root: Path) -> list[PhysicalKernelEntry]:
    entries: list[PhysicalKernelEntry] = []
    for backend, relative_root, patterns in NATIVE_ROOTS:
        for source in _source_files(root / relative_root, patterns):
            relative = source.relative_to(root)
            entries.extend(
                _entry(backend, relative, symbol, line, None, "native-__global__")
                for symbol, line in _native_global_symbols(
                    source.read_text(encoding="utf-8")
                )
            )
    return entries


def _decorator_name(decorator: ast.expr) -> str:
    if isinstance(decorator, ast.Call):
        return _decorator_name(decorator.func)
    if isinstance(decorator, ast.Attribute):
        prefix = _decorator_name(decorator.value)
        return f"{prefix}.{decorator.attr}" if prefix else decorator.attr
    if isinstance(decorator, ast.Name):
        return decorator.id
    return ""


def _tilelang_files(root: Path) -> list[Path]:
    sources: set[Path] = set()
    for relative_root in TILELANG_ROOTS:
        sources.update(_source_files(root / relative_root, ("*.py",)))
    for relative in TILELANG_FILES:
        source = root / relative
        if source.is_file():
            sources.add(source)
    return sorted(sources)


def discover_tilelang(root: Path) -> list[PhysicalKernelEntry]:
    entries: list[PhysicalKernelEntry] = []
    for source in _tilelang_files(root):
        relative = source.relative_to(root)
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for owner in tree.body:
            if not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(owner):
                if node is owner or not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                decorators = {_decorator_name(item) for item in node.decorator_list}
                if not any(name.endswith("T.prim_func") for name in decorators):
                    continue
                entries.append(
                    _entry(
                        "jit-tilelang-physical",
                        relative,
                        node.name,
                        node.lineno,
                        owner.name,
                        "tilelang-T.prim_func",
                    )
                )
    return entries


def discover_triton(root: Path) -> list[PhysicalKernelEntry]:
    entries: list[PhysicalKernelEntry] = []
    for source in _source_files(root / "vllm_musa", ("*.py",)):
        relative = source.relative_to(root)
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = {_decorator_name(item) for item in node.decorator_list}
            if any(name.endswith("triton.jit") for name in decorators):
                entries.append(
                    _entry(
                        "jit-triton-physical",
                        relative,
                        node.name,
                        node.lineno,
                        node.name,
                        "triton-jit",
                    )
                )
    return entries


def discover(root: Path) -> list[PhysicalKernelEntry]:
    entries = [
        *discover_native(root),
        *discover_tilelang(root),
        *discover_triton(root),
    ]
    by_id = {entry.id: entry for entry in entries}
    if len(by_id) != len(entries):
        raise RuntimeError("duplicate physical kernel inventory ids")
    return sorted(entries)


def payload(root: Path) -> dict[str, object]:
    entries = discover(root)
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.backend] = counts.get(entry.backend, 0) + 1
    return {
        "schema": SCHEMA,
        "root": str(root.resolve()),
        "counts": counts,
        "entries": [asdict(entry) for entry in entries],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(payload(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
