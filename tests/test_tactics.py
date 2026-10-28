# SPDX-License-Identifier: Apache-2.0
"""Tactic table lookup rules (pure data code, no device required)."""

from __future__ import annotations

import json

import pytest

from vllm_musa import tactics


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
            {"arch": "mp_31", "mp_count": 56, "values": {"block_m": 128}, "note": "tuned"},
            {"arch": "mp_31", "mp_count": None, "values": {"block_m": 64}},
            {"arch": "default", "mp_count": None, "values": {"block_m": 32}},
        ],
    )
    return tmp_path


def test_exact_bin_wins(table) -> None:
    assert tactics.select("demo", arch_name="mp_31", mp_count=56, directory=table) == {
        "block_m": 128
    }


def test_same_arch_other_bin_falls_back_to_arch_entry(table) -> None:
    """An untuned bin of a tuned arch must not inherit another bin's numbers."""
    assert tactics.select("demo", arch_name="mp_31", mp_count=60, directory=table) == {
        "block_m": 64
    }


def test_unknown_arch_falls_back_to_the_tables_default(table) -> None:
    assert tactics.select("demo", arch_name="mp_22", mp_count=48, directory=table) == {
        "block_m": 32
    }


def test_no_default_and_no_caller_default_raises(table) -> None:
    _write(table, "bare", [{"arch": "mp_31", "mp_count": 56, "values": {"block_m": 128}}])

    # the ladder describes the *requested* device, not the table's arch
    with pytest.raises(tactics.TacticNotFound, match=r"tried mp_22/mp48, mp_22, default"):
        tactics.select("bare", arch_name="mp_22", mp_count=48, directory=table)


def test_caller_default_is_an_explicit_opt_in(table) -> None:
    _write(table, "bare", [{"arch": "mp_31", "mp_count": 56, "values": {"block_m": 128}}])

    got = tactics.select(
        "bare", arch_name="mp_22", mp_count=48, default={"block_m": 16}, directory=table
    )

    assert got == {"block_m": 16}


def test_caller_default_does_not_override_a_real_entry(table) -> None:
    got = tactics.select(
        "demo", arch_name="mp_31", mp_count=56, default={"block_m": 16}, directory=table
    )

    assert got == {"block_m": 128}


def test_missing_table_raises(tmp_path) -> None:
    with pytest.raises(tactics.TacticNotFound, match="no tactic table for kernel 'nope'"):
        tactics.select("nope", arch_name="mp_31", directory=tmp_path)


def test_schema_version_is_checked(table) -> None:
    (table / "old.json").write_text(
        json.dumps({"_schema": 0, "kernel": "old", "tactics": [{"arch": "default", "values": {}}]})
    )

    with pytest.raises(ValueError, match="unsupported tactic schema"):
        tactics.load("old", table)


def test_duplicate_entries_are_rejected(table) -> None:
    _write(
        table,
        "dup",
        [
            {"arch": "mp_31", "mp_count": None, "values": {"a": 1}},
            {"arch": "mp_31", "mp_count": None, "values": {"a": 2}},
        ],
    )

    with pytest.raises(ValueError, match="duplicate tactic"):
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
        assert tactics.select(kernel, arch_name="mp_22", mp_count=48)


def test_package_data_ships_the_tables() -> None:
    """The tables are data files; without this they never reach a wheel."""
    import re

    pyproject = (tactics.TACTICS_DIR.parents[1] / "pyproject.toml").read_text()
    section = re.search(
        r"\[tool\.setuptools\.package-data\](.*?)(?:\n\[|\Z)", pyproject, re.DOTALL
    )
    assert section is not None, "no [tool.setuptools.package-data] section"
    assert "tactics/*.json" in section.group(1), section.group(1)


def test_select_for_device_reads_the_spec_not_the_build(
    monkeypatch, tmp_path
) -> None:
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
            {"arch": "mp_31", "mp_count": 60, "values": {"block_m": 128}},
            {"arch": "mp_31", "mp_count": 56, "values": {"block_m": 64}},
        ],
    )

    class Spec:
        arch_name = "mp_31"
        mp_count = 56

    monkeypatch.setattr(device_spec, "current_device_spec", lambda: Spec())

    assert tactics.select_for_device("demo", directory=tmp_path) == {"block_m": 64}
