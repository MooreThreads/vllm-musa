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
    for path in sorted(tactics.TACTICS_DIR.glob("*.json")):
        kernel = path.stem
        tactics.load(kernel)
        # The shipped tables only carry a generic entry today, so an untuned
        # arch resolves instead of raising; that is the documented state, not a
        # silent fallback.
        assert tactics.select(kernel, arch_name="mp_22", mp_count=48)


def test_describe_lists_every_entry(tmp_path) -> None:
    _write(
        tmp_path,
        "demo",
        [
            {"arch": "mp_31", "mp_count": 56, "values": {"block_m": 128}, "note": "tuned"},
            {"arch": "default", "mp_count": None, "values": {"block_m": 32}},
        ],
    )

    text = tactics.describe("demo", tmp_path)

    assert "mp_31/mp56" in text and "tuned" in text and "default" in text
