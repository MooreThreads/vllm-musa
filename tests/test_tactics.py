# SPDX-License-Identifier: Apache-2.0
"""Tactic table lookup rules (pure data code, no device required)."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from vllm_musa import tactics


@dataclass(frozen=True)
class _DemoTactic(tactics.Tactic):
    """Stand-in for a kernel that has not shipped a real tactic type yet."""

    block_m: int
    split_k: int = 1


@pytest.fixture
def demo_kernel(monkeypatch):
    """Register the stand-in type, exactly as a new kernel would register its own."""
    for kernel in ("demo", "bare", "dup", "old", "typo"):
        monkeypatch.setitem(tactics.TACTIC_TYPES, kernel, _DemoTactic)


def _write(directory, kernel, entries) -> None:
    (directory / f"{kernel}.json").write_text(
        json.dumps(
            {"_schema": tactics.SCHEMA_VERSION, "kernel": kernel, "tactics": entries}
        )
    )


@pytest.fixture
def table(tmp_path):
    _write(
        tmp_path,
        "demo",
        [
            {
                "arch": "mp_31",
                "mp_count": 56,
                "values": {"block_m": 128, "split_k": 2},
                "note": "tuned",
            },
            {"arch": "mp_31", "mp_count": None, "values": {"block_m": 64, "split_k": 1}},
            {"arch": None, "mp_count": None, "values": {"block_m": 32, "split_k": 1}},
        ],
    )
    return tmp_path


def test_exact_bin_wins(table, demo_kernel) -> None:
    assert tactics.resolve_tactic(
        "demo", arch_name="mp_31", mp_count=56, directory=table
    ) == _DemoTactic(block_m=128, split_k=2)


def test_same_arch_other_bin_falls_back_to_arch_entry(table, demo_kernel) -> None:
    """An untuned bin of a tuned arch must not inherit another bin's numbers."""
    assert tactics.resolve_tactic(
        "demo", arch_name="mp_31", mp_count=60, directory=table
    ) == _DemoTactic(block_m=64, split_k=1)


def test_unknown_arch_falls_back_to_the_tables_null_entry(table, demo_kernel) -> None:
    assert tactics.resolve_tactic(
        "demo", arch_name="mp_22", mp_count=48, directory=table
    ) == _DemoTactic(block_m=32, split_k=1)


def test_the_legacy_default_arch_is_read_as_the_fallback(tmp_path, demo_kernel) -> None:
    """Tables written before ``arch: null`` keep working; the sentinel is internal."""
    _write(
        tmp_path,
        "demo",
        [{"arch": "default", "mp_count": None, "values": {"block_m": 32, "split_k": 1}}],
    )

    entry = tactics.load("demo", tmp_path)[(None, None)]
    assert entry.arch is None
    assert entry.note == ""


def test_no_default_and_no_caller_default_raises(table, demo_kernel) -> None:
    _write(
        table,
        "bare",
        [
            {
                "arch": "mp_31",
                "mp_count": 56,
                "values": {"block_m": 128, "split_k": 1},
            }
        ],
    )

    # the ladder describes the *requested* device, not the table's arch
    with pytest.raises(tactics.TacticNotFound, match=r"tried mp_22/mp48, mp_22, default"):
        tactics.resolve_tactic("bare", arch_name="mp_22", mp_count=48, directory=table)


def test_caller_default_is_an_explicit_opt_in(table, demo_kernel) -> None:
    _write(
        table,
        "bare",
        [
            {
                "arch": "mp_31",
                "mp_count": 56,
                "values": {"block_m": 128, "split_k": 1},
            }
        ],
    )

    got = tactics.resolve_tactic(
        "bare",
        arch_name="mp_22",
        mp_count=48,
        default={"block_m": 16, "split_k": 1},
        directory=table,
    )

    assert got == _DemoTactic(block_m=16, split_k=1)


def test_a_caller_default_is_typed_too(table, demo_kernel) -> None:
    """A hand-written default goes through the same check as a table entry."""
    _write(
        table,
        "bare",
        [{"arch": "mp_31", "mp_count": 56, "values": {"block_m": 128, "split_k": 1}}],
    )

    with pytest.raises(ValueError, match="has no knob 'block_M'"):
        tactics.resolve_tactic(
            "bare",
            arch_name="mp_22",
            mp_count=48,
            default={"block_M": 16},
            directory=table,
        )


def test_caller_default_does_not_override_a_real_entry(table, demo_kernel) -> None:
    got = tactics.resolve_tactic(
        "demo",
        arch_name="mp_31",
        mp_count=56,
        default={"block_m": 16, "split_k": 1},
        directory=table,
    )

    assert got == _DemoTactic(block_m=128, split_k=2)


def test_a_misspelled_knob_is_refused_not_ignored(tmp_path, demo_kernel) -> None:
    _write(
        tmp_path,
        "typo",
        [{"arch": "mp_31", "mp_count": 56, "values": {"block_m": 128, "split_K": 2}}],
    )

    with pytest.raises(ValueError, match="has no knob 'split_K'"):
        tactics.load("typo", tmp_path)


def test_a_missing_knob_is_refused(tmp_path, demo_kernel) -> None:
    _write(
        tmp_path,
        "typo",
        [{"arch": "mp_31", "mp_count": 56, "values": {"block_m": 128}}],
    )

    with pytest.raises(ValueError, match=r"needs \['split_k'\]"):
        tactics.load("typo", tmp_path)


def test_a_non_integer_knob_is_refused(tmp_path, demo_kernel) -> None:
    _write(
        tmp_path,
        "typo",
        [{"arch": "mp_31", "mp_count": 56, "values": {"block_m": "128", "split_k": 1}}],
    )

    with pytest.raises(ValueError, match="every _DemoTactic knob is an integer"):
        tactics.load("typo", tmp_path)


def test_a_kernel_without_a_registered_type_cannot_load(tmp_path, demo_kernel) -> None:
    """The registered type *is* the schema; an unknown kernel has none."""
    _write(tmp_path, "demo", [{"arch": None, "mp_count": None, "values": {"block_m": 1}}])

    with pytest.raises(ValueError, match="no tactic type registered for kernel 'nope'"):
        tactics.load("nope", tmp_path)


def test_missing_table_raises(tmp_path, demo_kernel) -> None:
    with pytest.raises(tactics.TacticNotFound, match="no tactic table for kernel 'demo'"):
        tactics.resolve_tactic("demo", arch_name="mp_31", directory=tmp_path)


def test_schema_version_is_checked(table, demo_kernel) -> None:
    (table / "old.json").write_text(
        json.dumps({"_schema": 0, "kernel": "old", "tactics": [{"arch": None, "values": {}}]})
    )

    with pytest.raises(ValueError, match="unsupported tactic schema"):
        tactics.load("old", table)


def test_duplicate_entries_are_rejected(table, demo_kernel) -> None:
    _write(
        table,
        "dup",
        [
            {"arch": "mp_31", "mp_count": None, "values": {"block_m": 1, "split_k": 1}},
            {"arch": "mp_31", "mp_count": None, "values": {"block_m": 2, "split_k": 1}},
        ],
    )

    with pytest.raises(ValueError, match="duplicate tactic"):
        tactics.load("dup", table)


def test_an_entry_missing_its_values_is_reported_with_its_index(table, demo_kernel) -> None:
    _write(table, "dup", [{"arch": "mp_31", "mp_count": 56}])

    with pytest.raises(ValueError, match=r"tactic #0 is malformed"):
        tactics.load("dup", table)


def test_shipped_tables_parse_and_resolve() -> None:
    """Every table in the package must load and answer for any arch."""
    tables = sorted(tactics.TACTICS_DIR.glob("*.json"))
    assert tables, f"no tactic tables found in {tactics.TACTICS_DIR}"
    for path in tables:
        kernel = path.stem
        tactics.load(kernel)
        # The shipped tables only carry a generic entry today, so an untuned
        # arch resolves instead of raising; that is the documented state, not a
        # silent fallback.
        assert tactics.resolve_tactic(kernel, arch_name="mp_22", mp_count=48)


def test_the_shipped_tables_fit_their_registered_types() -> None:
    """``TACTIC_TYPES`` and the directory must not drift apart in either direction."""
    on_disk = {path.stem for path in tactics.TACTICS_DIR.glob("*.json")}
    assert on_disk == set(tactics.TACTIC_TYPES), (
        f"tables without a type: {sorted(on_disk - set(tactics.TACTIC_TYPES))}; "
        f"types without a table: {sorted(set(tactics.TACTIC_TYPES) - on_disk)}"
    )


def test_package_data_ships_the_tables() -> None:
    """The tables are data files; without this they never reach a wheel."""
    import re

    pyproject = (tactics.TACTICS_DIR.parents[1] / "pyproject.toml").read_text()
    section = re.search(
        r"\[tool\.setuptools\.package-data\](.*?)(?:\n\[|\Z)", pyproject, re.DOTALL
    )
    assert section is not None, "no [tool.setuptools.package-data] section"
    assert "tactics/*.json" in section.group(1), section.group(1)


def test_resolve_for_device_reads_the_spec_not_the_build(monkeypatch, tmp_path, demo_kernel) -> None:
    """The device, not ``torch_musa.get_arch_list()``, decides the lookup key.

    ``get_arch_list()`` reports the arch the wheel was compiled for, so a fat
    wheel or a reused environment would answer for the wrong bin; the spec comes
    from the device that is actually present.
    """
    from vllm_musa import device_spec

    _write(
        tmp_path,
        "demo",
        [
            {
                "arch": "mp_31",
                "mp_count": 60,
                "values": {"block_m": 128, "split_k": 1},
            },
            {
                "arch": "mp_31",
                "mp_count": 56,
                "values": {"block_m": 64, "split_k": 1},
            },
        ],
    )

    class Spec:
        arch_name = "mp_31"
        mp_count = 56

    monkeypatch.setattr(device_spec, "current_device_spec", lambda: Spec())

    assert tactics.resolve_tactic_for_device(
        "demo", directory=tmp_path
    ) == _DemoTactic(block_m=64, split_k=1)
