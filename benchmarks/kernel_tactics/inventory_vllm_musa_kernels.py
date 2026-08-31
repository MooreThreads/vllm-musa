#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Discover the vLLM-MUSA-owned AOT/JIT kernel coverage denominator.

The discovery layer is intentionally source-only and stdlib-only. It does not
decide whether a kernel is production reachable or tunable; the source-only
classifications live in ``full_kernel_sweep.json``. Hardware qualification
results are deliberately kept in local generated evidence. Keeping discovery
separate lets a completeness test fail whenever a new kernel is added without
an explicit campaign disposition.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA = "vllm-musa-owned-kernel-inventory.v1"
AOT_BINDINGS = Path("csrc/musa/torch_bindings.cpp")
JIT_CSRC_ROOT = Path("vllm_musa/jit_kernel/csrc")
TILELANG_ROOTS = (
    Path("vllm_musa/deepseek_v4_jit"),
    Path("vllm_musa/jit_kernel/tilelang"),
)
TILELANG_FILES = (Path("vllm_musa/v1/attention/ops/sparse_mla_tilelang.py"),)
REGISTER_CALLS = frozenset(
    {
        "_register",
        "direct_register_custom_op",
        "register_custom_op",
    }
)


@dataclass(frozen=True, order=True)
class KernelEntry:
    id: str
    backend: str
    source: str
    symbol: str
    discovery: str


def _source_files(root: Path, pattern: str = "*.py") -> Iterable[Path]:
    if not root.exists():
        return ()
    return sorted(path for path in root.rglob(pattern) if path.is_file())


def _call_name(call: ast.Call) -> str | None:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _decorator_name(decorator: ast.expr) -> str:
    if isinstance(decorator, ast.Call):
        return _decorator_name(decorator.func)
    if isinstance(decorator, ast.Attribute):
        prefix = _decorator_name(decorator.value)
        return f"{prefix}.{decorator.attr}" if prefix else decorator.attr
    if isinstance(decorator, ast.Name):
        return decorator.id
    return ""


def _entry(
    backend: str,
    source: Path,
    symbol: str,
    discovery: str,
) -> KernelEntry:
    normalized_source = source.as_posix()
    return KernelEntry(
        id=f"{backend}:{normalized_source}:{symbol}",
        backend=backend,
        source=normalized_source,
        symbol=symbol,
        discovery=discovery,
    )


def discover_aot(root: Path) -> list[KernelEntry]:
    source = root / AOT_BINDINGS
    text = source.read_text(encoding="utf-8")
    symbols = sorted(set(re.findall(r'\b\w+\.def\(\s*"([A-Za-z0-9_]+)\(', text)))
    return [
        _entry("aot-native", AOT_BINDINGS, symbol, "torch-library-binding")
        for symbol in symbols
    ]


def discover_jit_native(root: Path) -> list[KernelEntry]:
    entries: list[KernelEntry] = []
    for source in _source_files(root / JIT_CSRC_ROOT):
        relative = source.relative_to(root)
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        symbols: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) not in REGISTER_CALLS:
                continue
            candidates = list(node.args[:1])
            candidates.extend(
                keyword.value for keyword in node.keywords if keyword.arg == "op_name"
            )
            for candidate in candidates:
                if isinstance(candidate, ast.Constant) and isinstance(
                    candidate.value, str
                ):
                    symbols.add(candidate.value)
        entries.extend(
            _entry("jit-native", relative, symbol, "custom-op-registration")
            for symbol in sorted(symbols)
        )
    return entries


def discover_jit_native_ffi(root: Path) -> list[KernelEntry]:
    entries: list[KernelEntry] = []
    pattern = re.compile(r"TVM_FFI_DLL_EXPORT_TYPED_FUNC\(\s*([A-Za-z0-9_]+)")
    for source in _source_files(root / JIT_CSRC_ROOT, "*.mu"):
        relative = source.relative_to(root)
        symbols = sorted(set(pattern.findall(source.read_text(encoding="utf-8"))))
        entries.extend(
            _entry("jit-native-ffi", relative, symbol, "tvm-ffi-export")
            for symbol in symbols
        )
    return entries


def discover_triton(root: Path) -> list[KernelEntry]:
    entries: list[KernelEntry] = []
    for source in _source_files(root / "vllm_musa"):
        relative = source.relative_to(root)
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = {_decorator_name(item) for item in node.decorator_list}
            if not any(name.endswith("triton.jit") for name in decorators):
                continue
            entries.append(_entry("jit-triton", relative, node.name, "triton-jit"))
    return entries


def _tilelang_files(root: Path) -> list[Path]:
    sources: set[Path] = set()
    for relative_root in TILELANG_ROOTS:
        sources.update(_source_files(root / relative_root))
    for relative in TILELANG_FILES:
        source = root / relative
        if source.is_file():
            sources.add(source)
    return sorted(sources)


def discover_tilelang(root: Path) -> list[KernelEntry]:
    entries: list[KernelEntry] = []
    for source in _tilelang_files(root):
        relative = source.relative_to(root)
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = {_decorator_name(item) for item in node.decorator_list}
            if node.name == "_prefix_counts_no_pad_kernel":
                continue
            if not node.name.endswith("kernel") and not any(
                name.endswith("tilelang.jit") for name in decorators
            ):
                continue
            entries.append(
                _entry("jit-tilelang", relative, node.name, "tilelang-callable-factory")
            )
    return entries


def discover(root: Path) -> list[KernelEntry]:
    entries = [
        *discover_aot(root),
        *discover_jit_native(root),
        *discover_jit_native_ffi(root),
        *discover_triton(root),
        *discover_tilelang(root),
    ]
    by_id = {entry.id: entry for entry in entries}
    if len(by_id) != len(entries):
        duplicates = sorted(
            entry.id
            for entry in entries
            if sum(item.id == entry.id for item in entries) > 1
        )
        raise RuntimeError(f"duplicate discovered kernel ids: {duplicates}")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document = json.dumps(payload(args.root), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(document, end="")
    else:
        args.output.write_text(document, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
