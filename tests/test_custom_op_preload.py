import ast
from pathlib import Path

from vllm_musa.custom_op_modules import CUSTOM_OP_MODULES
from vllm_musa.operator_manifest import (
    OPERATOR_RECORDS,
    operator_report,
    validate_records,
)

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


def test_operator_manifest_is_complete() -> None:
    errors = validate_records(ROOT, OPERATOR_RECORDS)
    assert errors == []


def test_operator_manifest_matches_preload() -> None:
    assert {record.registration_module for record in OPERATOR_RECORDS if record.preload} == set(CUSTOM_OP_MODULES)


def test_operator_manifest_rejects_invalid_category() -> None:
    record = OPERATOR_RECORDS[0]
    invalid = record.__class__(
        name=record.name,
        category="unknown",
        contract=record.contract,
        provider=record.provider,
        registration_module=record.registration_module,
        source=record.source,
        tests=record.tests,
        fallback=record.fallback,
    )
    errors = validate_records(ROOT, (invalid,))
    assert any("invalid category" in error for error in errors)


def test_operator_manifest_rejects_missing_paths() -> None:
    record = OPERATOR_RECORDS[0]
    invalid = record.__class__(
        name=record.name,
        category=record.category,
        contract=record.contract,
        provider=record.provider,
        registration_module=record.registration_module,
        source=("missing/source.py",),
        tests=("missing/test.py",),
        fallback=record.fallback,
    )
    errors = validate_records(ROOT, (invalid,))
    assert "custom_all_reduce: missing path missing/source.py" in errors
    assert "custom_all_reduce: missing path missing/test.py" in errors


def test_operator_manifest_rejects_duplicate_names_and_modules() -> None:
    record = OPERATOR_RECORDS[0]
    errors = validate_records(ROOT, (record, record))
    assert any("duplicate operator name" in error for error in errors)
    assert any("duplicate registration module" in error for error in errors)


def test_operator_report_is_dependency_free_and_complete() -> None:
    report = operator_report(ROOT)
    assert len(report) == len(OPERATOR_RECORDS)
    assert all(
        row["source_present"]
        and row["tests_present"]
        and row["build_sources"]
        for row in report
    )
    assert {row["category"] for row in report} >= {"mate_adapter", "model_private"}
