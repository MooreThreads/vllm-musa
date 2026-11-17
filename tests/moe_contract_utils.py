# SPDX-License-Identifier: Apache-2.0
"""Execute production contract bodies without importing vLLM's GPU runtime."""

from __future__ import annotations

import ast
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
APPLIED_ROOT = Path(os.environ.get("MUSA_APPLIED_VLLM", ROOT / "third_party/vllm"))


def _execute(tree: ast.Module, path: Path, namespace: dict[str, Any]) -> None:
    code = compile(
        ast.fix_missing_locations(tree),
        str(path),
        "exec",
        flags=__import__("__future__").annotations.compiler_flag,
    )
    exec(code, namespace)


def load_mega_contract(torch: Any) -> ModuleType:
    """Use actual implementation bodies with imported runtime services stubbed."""
    path = ROOT / "vllm_musa/model_executor/layers/fused_moe/mate_mega_moe.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    class Activation(Enum):
        SILU = "silu"
        GELU = "gelu"

    module = ModuleType("mega_contract")
    namespace = module.__dict__
    namespace.update(
        {
            "torch": torch,
            "dataclass": dataclass,
            "Any": Any,
            "MoEActivation": Activation,
            "mk": SimpleNamespace(FusedMoEExpertsModular=object),
            "kFp8Dynamic128Sym": "dynamic128",
            "kFp8Static128BlockSym": "static128",
            "init_logger": lambda name: SimpleNamespace(info_once=lambda *args: None),
            "is_forward_context_available": lambda: False,
            "get_forward_context": lambda: None,
        }
    )
    previous = sys.modules.get(module.__name__)
    sys.modules[module.__name__] = module
    try:
        _execute(tree, path, namespace)
    finally:
        if previous is None:
            sys.modules.pop(module.__name__, None)
        else:
            sys.modules[module.__name__] = previous
    return module


def load_applied_contract(
    relative_path: str, names: list[str], namespace: dict[str, Any]
) -> SimpleNamespace:
    """Compile named helpers from the actual applied upstream checkout."""
    path = APPLIED_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Apply the patch series before contract tests: {path}")
    tree = ast.parse(path.read_text())
    selected = []
    for name in names:
        node = next(
            (
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name == name
            ),
            None,
        )
        if node is None:
            raise AssertionError(f"Applied source has no {name}: {path}")
        selected.append(node)
    _execute(ast.Module(body=selected, type_ignores=[]), path, namespace)
    return SimpleNamespace(**namespace)


def load_applied_methods(
    relative_path: str,
    class_name: str,
    methods: list[str],
    namespace: dict[str, Any],
) -> SimpleNamespace:
    path = APPLIED_ROOT / relative_path
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    selected = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    if {node.name for node in selected} != set(methods):
        raise AssertionError(f"Missing contract methods in {class_name}")
    stub = ast.ClassDef(
        name=class_name, bases=[], keywords=[], body=selected, decorator_list=[]
    )
    _execute(ast.Module(body=[stub], type_ignores=[]), path, namespace)
    return SimpleNamespace(**namespace)
