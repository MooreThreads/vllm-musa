# SPDX-License-Identifier: Apache-2.0
"""The custom-op preload manifest must stay complete.

Every module that calls ``direct_register_custom_op`` registers its op as an
import side effect. vLLM reads its AOT autograd cache during engine init, before
any of those imports necessarily happen; a missing op makes the cache load fail
and the engine recompiles the whole model on every boot. ``_register_ops()``
therefore preloads ``CUSTOM_OP_MODULES`` — and this test is what keeps that list
honest as modules are added.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "vllm_musa"
REGISTRAR = "direct_register_custom_op"


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _registered_names(tree: ast.Module) -> set[str]:
    """Local names that refer to the op registrar, including aliased imports."""
    names = {REGISTRAR}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "vllm.utils.torch_utils":
            for alias in node.names:
                if alias.name == REGISTRAR:
                    names.add(alias.asname or alias.name)
    return names


def _registers_custom_op(tree: ast.Module) -> bool:
    names = _registered_names(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            found = func.id
        elif isinstance(func, ast.Attribute):
            found = func.attr
        else:
            continue
        if found in names:
            return True
    return False


def test_manifest_covers_every_module_that_registers_a_custom_op() -> None:
    discovered = {
        _module_name(path)
        for path in PACKAGE_ROOT.rglob("*.py")
        if _registers_custom_op(ast.parse(path.read_text()))
    }
    from vllm_musa.custom_op_modules import CUSTOM_OP_MODULES

    missing = discovered - set(CUSTOM_OP_MODULES)
    stale = set(CUSTOM_OP_MODULES) - discovered
    assert not missing, f"modules that register ops but are not preloaded: {sorted(missing)}"
    assert not stale, f"preloaded modules that no longer register ops: {sorted(stale)}"


def test_manifest_has_no_duplicates() -> None:
    from vllm_musa.custom_op_modules import CUSTOM_OP_MODULES

    assert len(CUSTOM_OP_MODULES) == len(set(CUSTOM_OP_MODULES))


def test_every_manifest_entry_is_a_real_file() -> None:
    """Catches a rename or typo that the AST scan cannot (it sees only modules)."""
    from vllm_musa.custom_op_modules import CUSTOM_OP_MODULES

    for name in CUSTOM_OP_MODULES:
        parts = name.split(".")[1:]  # drop the leading "vllm_musa"
        candidates = [
            PACKAGE_ROOT.joinpath(*parts).with_suffix(".py"),
            PACKAGE_ROOT.joinpath(*parts) / "__init__.py",
        ]
        assert any(p.exists() for p in candidates), f"{name} does not exist under vllm_musa/"
