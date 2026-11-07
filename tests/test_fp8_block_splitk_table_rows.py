# SPDX-License-Identifier: Apache-2.0
"""The split-K rows a tuner measurement derives, and their merge into the table; needs
no torch."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "fp8_block_splitk"))

import table_rows  # noqa: E402

NARROW = (16, 64, 4, 2, 32)


def _entry(n, k, launch, min_rows, capture, eager, mp=None) -> dict:
    entry = {} if mp is None else {"multiprocessor_count": mp}
    entry.update({"n": n, "k": k, **dict(zip(table_rows.LAUNCH_KEYS, launch))})
    entry.update({"min_rows": min_rows, "max_rows": {"capture": capture, "eager": eager}})
    return entry


TABLE_TEXT = table_rows.dumps(
    {
        "schema": 1,
        "kernel": "fp8_block_splitk",
        "source": "test table",
        "entries": [
            _entry(1792, 5120, NARROW, 2, 59, 192),
            _entry(576, 5120, (16, 64, 4, 2, 16), 4, 64, 320),
            _entry(1792, 5120, NARROW, 2, 59, 256, mp=56),
            _entry(576, 5120, (32, 64, 4, 1, 32), 2, 64, 511, mp=60),
            _entry(1792, 5120, NARROW, 2, 59, 272, mp=60),
        ],
    }
)
# The default row of 1792x5120 as the tuner records it, and a candidate config.
DEFAULT = [16, 64, 4, 2, 2, 59, 32]
DEFAULT_BANDS = {"min_rows": 2, "capture": 59, "eager": 192}
WIDE = [16, 128, 8, 1, 0, 0, 32]
WINS = {rows: 0.9 for rows in range(2, 17)}


def _log(config, ratios) -> list:
    """Capture band records, rows -> split-K time over the time without it."""
    return [
        {
            "stage": "band",
            "mode": "capture",
            "config": list(config),
            "rows": rows,
            "splitk_ms": ratio,
            "ref_ms": 1.0,
        }
        for rows, ratio in ratios.items()
    ]


def _default_shape(measured=None, best=None, best_score=1.0) -> dict:
    measured = measured or dict(DEFAULT_BANDS)
    shape = {
        "n": 1792,
        "k": 5120,
        "default": {"config": DEFAULT, "table": DEFAULT_BANDS, "measured": measured},
        "ranked": [{"config": [*DEFAULT[:4], 0, 0, DEFAULT[6]], "score": 1.0}],
        "log": _log(DEFAULT, WINS),
    }
    if best is not None:
        shape["ranked"].insert(0, {"config": best["config"], "score": best_score})
        shape["best"] = best
        shape["log"] += _log(best["config"], WINS)
    return shape


def _new_shape(bands=None, ratios=WINS, error=0.0, n=512, k=5120) -> dict:
    bands = bands or {"min_rows": 2, "capture": 64, "eager": 300}
    return {
        "n": n,
        "k": k,
        "ranked": [{"config": WIDE, "score": 0.8}],
        "best": {"config": WIDE, "measured": bands, "relative_error": error},
        "log": _log(WIDE, ratios),
    }


def _report(mp: int, *shapes) -> dict:
    return {"provenance": {"multiprocessor_count": mp}, "shapes": list(shapes)}


def _entries() -> list:
    return json.loads(TABLE_TEXT)["entries"]


def test_the_shipped_table_keeps_its_layout() -> None:
    text = table_rows.TABLE.read_text()
    assert table_rows.dumps(json.loads(text)) == text


def test_a_card_that_matches_the_default_row_gets_no_row() -> None:
    assert table_rows.table_row(_default_shape(), 60) == (None, [])


def test_a_faster_config_gets_a_row() -> None:
    best = {"config": WIDE, "measured": dict(DEFAULT_BANDS), "relative_error": 0.0}
    row, reasons = table_rows.table_row(_default_shape(best=best, best_score=0.9), 48)
    assert (row["block_n"], row["num_warps"], row["multiprocessor_count"]) == (128, 8, 48)
    assert reasons == ["config (16, 128, 8, 1, 32) 10.0% faster"]


def test_a_moved_band_edge_gives_a_row() -> None:
    measured = {"min_rows": 2, "capture": 59, "eager": 256}
    row, reasons = table_rows.table_row(_default_shape(measured), 56)
    assert row == _entry(1792, 5120, NARROW, 2, 59, 256, mp=56)
    assert reasons == ["eager 192->256"]


def test_a_new_shape_gets_a_row_for_the_measured_count() -> None:
    row, reasons = table_rows.table_row(_new_shape(), 60)
    assert row == {
        "multiprocessor_count": 60,
        "n": 512,
        "k": 5120,
        "split": 16,
        "block_n": 128,
        "num_warps": 8,
        "unroll": 1,
        "tile_m": 32,
        "min_rows": 2,
        "max_rows": {"capture": 64, "eager": 300},
    }
    assert reasons == ["no default row: config (16, 128, 8, 1, 32)"]


@pytest.mark.parametrize(
    "shape, reason",
    [
        ({**_new_shape(), "best": None}, "no search"),
        (_new_shape(error=0.05), "misses the output check"),
        (_new_shape(ratios={r: 1.1 for r in range(2, 17)}), "never clearly wins"),
    ],
    ids=["no-search", "output-check", "no-win"],
)
def test_a_new_shape_without_a_clear_row_gets_none(shape, reason) -> None:
    row, reasons = table_rows.table_row(shape, 60)
    assert row is None and reason in reasons[-1]


def test_min_rows_above_a_band_edge_gives_no_row() -> None:
    ratios = {r: 0.9 if r < 4 or r >= 8 else 1.1 for r in range(2, 17)}
    shape = _new_shape({"min_rows": 2, "capture": 3, "eager": 3}, ratios)
    row, reasons = table_rows.table_row(shape, 60)
    assert row is None and "MIN_ROWS 8 lies above a band edge" in reasons[-1]


def test_a_merge_replaces_only_the_measured_rows() -> None:
    entries = _entries()
    replaced = _entry(1792, 5120, NARROW, 3, 59, 272, mp=60)
    added = table_rows.table_row(_new_shape(), 60)[0]
    decisions = {
        (1792, 5120, 60): (replaced, []),
        (576, 5120, 60): (None, []),
        (512, 5120, 60): (added, []),
    }
    merged, changes = table_rows.merge(entries, decisions)
    assert changes == [
        ["added", 512, 5120, 60],
        ["removed", 576, 5120, 60],
        ["replaced", 1792, 5120, 60],
    ]
    defaults = [e for e in entries if "multiprocessor_count" not in e]
    assert merged[: len(defaults)] == defaults
    binned = [e for e in merged if "multiprocessor_count" in e]
    keys = [(e["multiprocessor_count"], e["n"], e["k"]) for e in binned]
    assert keys == sorted(keys)
    expected = {
        (e["multiprocessor_count"], e["n"], e["k"]): e
        for e in entries
        if "multiprocessor_count" in e
    }
    del expected[(60, 576, 5120)]
    expected[(60, 1792, 5120)] = replaced
    expected[(60, 512, 5120)] = added
    assert dict(zip(keys, binned)) == expected


def test_a_shape_measured_twice_for_one_count_is_refused() -> None:
    reports = [_report(60, _default_shape()), _report(60, _default_shape())]
    with pytest.raises(ValueError, match="measured twice for multiprocessor count 60"):
        table_rows.decide(reports, _entries())


@pytest.mark.parametrize(
    "shape, message",
    [
        (
            {**_default_shape(), "default": {**_default_shape()["default"], "table": {
                **DEFAULT_BANDS, "eager": 200}}},
            "another default row",
        ),
        ({**_default_shape(), "default": None}, "has one now"),
    ],
    ids=["other-default", "default-added-since"],
)
def test_a_measurement_against_another_default_row_is_refused(shape, message) -> None:
    with pytest.raises(ValueError, match=message):
        table_rows.decide([_report(60, shape)], _entries())


def test_a_new_shape_measured_on_two_counts_proposes_their_shared_band() -> None:
    reports = [
        _report(
            48,
            _new_shape(
                {"min_rows": 3, "capture": 48, "eager": 300},
                {r: 1.1 if r == 2 else 0.9 for r in range(2, 17)},
            ),
        ),
        _report(60, _new_shape({"min_rows": 2, "capture": 64, "eager": 256})),
        _report(56, _new_shape(ratios={r: 1.1 for r in range(2, 17)})),
    ]
    decisions = table_rows.decide(reports, _entries())
    (proposal,) = table_rows.default_proposals(decisions, _entries())
    assert proposal["measured_on"] == [48, 60]
    assert proposal["without_row"] == [56]
    row = proposal["default_row"]
    assert "multiprocessor_count" not in row
    assert (row["block_n"], row["min_rows"], row["max_rows"]) == (
        128,
        3,
        {"capture": 48, "eager": 256},
    )


def test_counts_that_chose_different_configs_propose_no_default_row() -> None:
    other = _new_shape()
    other["best"] = {**other["best"], "config": [8, 64, 4, 1, 0, 0, 16]}
    other["ranked"] = [{"config": other["best"]["config"], "score": 0.8}]
    other["log"] = _log(other["best"]["config"], WINS)
    decisions = table_rows.decide(
        [_report(48, _new_shape()), _report(60, other)], _entries()
    )
    (proposal,) = table_rows.default_proposals(decisions, _entries())
    assert proposal["default_row"] is None
    assert proposal["reason"] == "the counts chose different configs"


def test_a_shape_with_a_default_row_gets_no_proposal() -> None:
    decisions = table_rows.decide([_report(60, _default_shape())], _entries())
    assert table_rows.default_proposals(decisions, _entries()) == []


def test_a_write_that_changes_nothing_leaves_the_file_alone(tmp_path) -> None:
    table = tmp_path / "fp8_block_splitk.json"
    table.write_text(TABLE_TEXT)
    measured = {"min_rows": 2, "capture": 59, "eager": 256}
    result = table_rows.write(table, [_report(56, _default_shape(measured))])
    assert result["changes"] == []
    assert table.read_text() == TABLE_TEXT


def test_a_write_adds_a_new_count_and_stays_loadable(tmp_path) -> None:
    from vllm_musa import tuning

    table = tmp_path / "fp8_block_splitk.json"
    table.write_text(TABLE_TEXT)
    result = table_rows.write(table, [_report(64, _new_shape())])
    assert result["changes"] == [["added", 512, 5120, 64]]
    entries = tuning._parse(table.name, "fp8_block_splitk", table.read_text())
    assert entries[:-1] == _entries()
    assert entries[-1]["multiprocessor_count"] == 64


def test_an_invalid_merge_is_not_written(tmp_path, monkeypatch) -> None:
    table = tmp_path / "fp8_block_splitk.json"
    table.write_text(TABLE_TEXT)
    bad = {**table_rows.table_row(_new_shape(), 64)[0], "min_rows": 0}
    monkeypatch.setattr(table_rows, "table_row", lambda shape, mp: (bad, []))
    with pytest.raises(ValueError, match="positive integer"):
        table_rows.write(table, [_report(64, _new_shape())])
    assert table.read_text() == TABLE_TEXT


def test_the_command_line_writes_a_table(tmp_path, capsys) -> None:
    table = tmp_path / "fp8_block_splitk.json"
    table.write_text(TABLE_TEXT)
    result = tmp_path / "result.json"
    result.write_text(json.dumps(_report(64, _new_shape())))
    assert table_rows.main([str(result), "--write", str(table)]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0]["row"]["multiprocessor_count"] == 64
    assert lines[1]["default_row"]["n"] == 512
    assert lines[-1] == {"write": str(table), "changes": [["added", 512, 5120, 64]]}
    assert table.read_text() != TABLE_TEXT
