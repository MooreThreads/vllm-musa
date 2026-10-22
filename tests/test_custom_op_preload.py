import ast
from pathlib import Path

from vllm_musa.custom_op_modules import CUSTOM_OP_MODULES


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "vllm_musa"


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _registers_custom_op(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    registration_names = {"direct_register_custom_op"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "vllm.utils.torch_utils":
            for alias in node.names:
                if alias.name == "direct_register_custom_op":
                    registration_names.add(alias.asname or alias.name)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else None
        )
        if name in registration_names:
            return True
    return False


def test_custom_op_manifest_covers_all_custom_op_modules() -> None:
    discovered = {
        _module_name(path)
        for path in PACKAGE_ROOT.rglob("*.py")
        if _registers_custom_op(path)
    }
    assert set(CUSTOM_OP_MODULES) == discovered


def test_custom_op_manifest_has_no_duplicates() -> None:
    assert len(CUSTOM_OP_MODULES) == len(set(CUSTOM_OP_MODULES))
