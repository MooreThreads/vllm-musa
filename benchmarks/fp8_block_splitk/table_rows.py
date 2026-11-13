#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The split-K rows a tune_fp8_block_splitk.py measurement gives a card, and their merge
into vllm_musa/tuning/fp8_block_splitk.json. Needs no torch.

    python benchmarks/fp8_block_splitk/table_rows.py RESULT.json [...] [--write [TABLE]]

For a shape with a default row, the card gets its own row where its best config is
CONFIG_GAIN faster than the default config, or a band edge moves by
max(EDGE_ROWS, EDGE_FRACTION). For a shape without one, the card's row is its best
config and band where split-K clearly wins, and a default row is proposed: the config
every measured count chose, with the band they all share.

``--write`` replaces, for each measured shape, the row of the card's multiprocessor
count: a new row is added, a row no longer needed is removed, everything else is kept.
Default rows are never written; a proposal is copied into the table by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TABLE = REPO / "vllm_musa" / "tuning" / "fp8_block_splitk.json"

CONFIG_GAIN = 0.03
EDGE_ROWS = 4
EDGE_FRACTION = 0.10
# MIN_ROWS keeps the default row's value unless the strict crossover moved this far.
MIN_ROWS_DELTA = 2
MAX_RELATIVE_ERROR = 0.01

LAUNCH_KEYS = ("split", "block_n", "num_warps", "unroll", "tile_m")


def _launch(config) -> tuple:
    return (*config[:4], config[6])


def _moved(a: int, b: int) -> bool:
    return abs(a - b) >= max(EDGE_ROWS, EDGE_FRACTION * max(a, b))


def _strict_min_rows(log: list, config) -> int | None:
    """Smallest measured row from 2 from which split-K is faster at every measured
    capture row up to 16."""
    ratio = {
        m["rows"]: m["splitk_ms"] / m["ref_ms"]
        for m in log
        if m.get("stage") == "band"
        and m.get("mode") == "capture"
        and m["config"] == list(config)
    }
    rows = sorted(r for r in ratio if 2 <= r <= 16)
    for i, r in enumerate(rows):
        if all(ratio[x] < 1 for x in rows[i:]):
            return r
    return None


def _row(shape: dict, mp: int, config, min_rows: int, bands: dict) -> dict | None:
    if min_rows > min(bands["capture"], bands["eager"]):
        return None
    split, block_n, warps, unroll, _, _, tile_m = config
    return {
        "multiprocessor_count": mp,
        "n": shape["n"],
        "k": shape["k"],
        "split": split,
        "block_n": block_n,
        "num_warps": warps,
        "unroll": unroll,
        "tile_m": tile_m,
        "min_rows": min_rows,
        "max_rows": {"capture": bands["capture"], "eager": bands["eager"]},
    }


def _new_shape_row(shape: dict, mp: int) -> tuple[dict | None, list[str]]:
    best = shape.get("best")
    if best is None:
        return None, ["no default row, and no search to derive one from"]
    if best.get("relative_error", math.inf) > MAX_RELATIVE_ERROR:
        return None, [f"best config {_launch(best['config'])} misses the output check"]
    bands = best["measured"]
    min_rows = _strict_min_rows(shape["log"], best["config"])
    if bands["min_rows"] is None or min_rows is None:
        return None, ["split-K never clearly wins"]
    row = _row(shape, mp, best["config"], min_rows, bands)
    if row is None:
        return None, [f"MIN_ROWS {min_rows} lies above a band edge {bands}"]
    return row, [f"no default row: config {_launch(best['config'])}"]


def table_row(shape: dict, mp: int) -> tuple[dict | None, list[str]]:
    """The table row this card gets for ``shape``, or None, and the reasons."""
    default, best = shape.get("default"), shape.get("best")
    if default is None:
        return _new_shape_row(shape, mp)
    scores = {_launch(r["config"]): r["score"] for r in shape.get("ranked", [])}
    default_score = scores.get(_launch(default["config"]))
    use_best = (
        best is not None
        and default_score is not None
        and _launch(best["config"]) != _launch(default["config"])
        and best.get("relative_error", math.inf) <= MAX_RELATIVE_ERROR
        and scores[_launch(best["config"])] <= default_score * (1 - CONFIG_GAIN)
    )
    chosen = best if use_best else default
    bands, table = chosen["measured"], default["table"]
    reasons = []
    if use_best:
        gain = 1 - scores[_launch(best["config"])] / default_score
        reasons.append(f"config {_launch(best['config'])} {gain:.1%} faster")
    min_rows = _strict_min_rows(shape["log"], chosen["config"])
    if bands["min_rows"] is None or min_rows is None:
        return None, reasons + ["split-K never clearly wins"]
    if abs(min_rows - table["min_rows"]) < MIN_ROWS_DELTA:
        min_rows = table["min_rows"]
    for mode in ("capture", "eager"):
        if _moved(bands[mode], table[mode]):
            reasons.append(f"{mode} {table[mode]}->{bands[mode]}")
    if not reasons:
        return None, reasons
    row = _row(shape, mp, chosen["config"], min_rows, bands)
    if row is None:
        return None, reasons + [f"MIN_ROWS {min_rows} lies above a band edge {bands}"]
    return row, reasons


def _config(entry: dict) -> list[int]:
    """A table entry as the 7-field config the tuner records."""
    return [
        entry["split"],
        entry["block_n"],
        entry["num_warps"],
        entry["unroll"],
        entry["min_rows"],
        entry["max_rows"]["capture"],
        entry["tile_m"],
    ]


def _check_default(shape: dict, entry: dict | None) -> None:
    """Refuse a measurement taken against another default row than ``entry``."""
    measured = shape.get("default")
    where = f"{shape['n']}x{shape['k']}"
    if measured is None:
        if entry is not None:
            raise ValueError(
                f"{where} was measured without a default row, and the table has one "
                "now: measure it again"
            )
        return
    if (
        entry is None
        or measured["config"] != _config(entry)
        or measured["table"]["eager"] != entry["max_rows"]["eager"]
    ):
        raise ValueError(
            f"{where} was measured against another default row than the table has: "
            "measure it again"
        )


def decide(reports: list[dict], entries: list[dict]) -> dict:
    """``(N, K, multiprocessor count) -> (row or None, reasons)`` for every measured
    shape of ``reports``."""
    defaults = {
        (e["n"], e["k"]): e for e in entries if e.get("multiprocessor_count") is None
    }
    decisions = {}
    for report in reports:
        mp = report["provenance"]["multiprocessor_count"]
        for shape in report["shapes"]:
            key = (shape["n"], shape["k"], mp)
            if key in decisions:
                raise ValueError(
                    f"{key[0]}x{key[1]} is measured twice for multiprocessor count {mp}"
                )
            _check_default(shape, defaults.get(key[:2]))
            decisions[key] = table_row(shape, mp)
    return decisions


def default_proposals(decisions: dict, entries: list[dict]) -> list[dict]:
    """A default row for each measured shape the table has none for: the config every
    count with a row chose, with the band all of them share."""
    defaults = {(e["n"], e["k"]) for e in entries if "multiprocessor_count" not in e}
    shapes = sorted({key[:2] for key in decisions} - defaults)
    proposals = []
    for n, k in shapes:
        rows = {
            mp: row
            for (n2, k2, mp), (row, _) in sorted(decisions.items())
            if (n2, k2) == (n, k) and row is not None
        }
        without = sorted(
            mp
            for (n2, k2, mp), (row, _) in decisions.items()
            if (n2, k2) == (n, k) and row is None
        )
        proposal = {"n": n, "k": k, "measured_on": sorted(rows), "without_row": without}
        launches = {tuple(row[key] for key in LAUNCH_KEYS) for row in rows.values()}
        if len(launches) != 1:
            proposal["default_row"] = None
            proposal["reason"] = (
                "no count derives a row"
                if not rows
                else "the counts chose different configs"
            )
        else:
            (launch,) = launches
            proposal["default_row"] = {
                "n": n,
                "k": k,
                **dict(zip(LAUNCH_KEYS, launch)),
                "min_rows": max(row["min_rows"] for row in rows.values()),
                "max_rows": {
                    mode: min(row["max_rows"][mode] for row in rows.values())
                    for mode in ("capture", "eager")
                },
            }
        proposals.append(proposal)
    return proposals


def _by_count(item) -> tuple:
    (n, k, mp), _ = item
    return mp, n, k


def merge(entries: list[dict], decisions: dict) -> tuple[list[dict], list[list]]:
    """``entries`` with each decided row replaced, and the changes made."""
    defaults = [e for e in entries if e.get("multiprocessor_count") is None]
    binned = {
        (e["n"], e["k"], e["multiprocessor_count"]): e
        for e in entries
        if e.get("multiprocessor_count") is not None
    }
    changes = []
    for key, (row, _) in sorted(decisions.items(), key=_by_count):
        old = binned.pop(key, None)
        if row is not None:
            binned[key] = row
        if old is None and row is not None:
            changes.append(["added", *key])
        elif old is not None and row is None:
            changes.append(["removed", *key])
        elif old != row:
            changes.append(["replaced", *key])
    order = sorted(binned, key=lambda key: (key[2], key[0], key[1]))
    return defaults + [binned[key] for key in order], changes


def dumps(table: dict) -> str:
    """The table in its file layout: one entry per line."""
    lines = []
    for key, value in table.items():
        if key == "entries":
            body = ",\n".join(f"    {json.dumps(e)}" for e in value)
            lines.append(f'  "entries": [\n{body}\n  ]')
        else:
            lines.append(f"  {json.dumps(key)}: {json.dumps(value)}")
    return "{\n" + ",\n".join(lines) + "\n}\n"


def write(path: Path, reports: list[dict]) -> dict:
    """Merge the rows of ``reports`` into the table at ``path``; validate before writing."""
    from vllm_musa import tuning

    text = path.read_text()
    table = json.loads(text)
    decisions = decide(reports, table["entries"])
    entries, changes = merge(table["entries"], decisions)
    new_text = dumps({**table, "entries": entries})
    tuning._parse(path.name, table["kernel"], new_text)
    for entry in entries:
        if entry["min_rows"] > min(entry["max_rows"].values()):
            raise ValueError(f"{path.name}: min_rows exceeds max_rows in {entry}")
    if new_text != text:
        path.write_text(new_text)
    return {
        "decisions": decisions,
        "changes": changes,
        "proposals": default_proposals(decisions, table["entries"]),
    }


def report_lines(decisions: dict, proposals: list[dict]) -> list[str]:
    lines = [
        json.dumps(
            {"multiprocessor_count": mp, "n": n, "k": k, "row": row, "reasons": reasons}
        )
        for (n, k, mp), (row, reasons) in sorted(decisions.items())
    ]
    return lines + [json.dumps(p) for p in proposals]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", nargs="+", type=Path, help="tuner --output files")
    parser.add_argument(
        "--write",
        nargs="?",
        const=TABLE,
        type=Path,
        help=f"merge the rows into this table (default {TABLE.relative_to(REPO)})",
    )
    args = parser.parse_args(argv)
    reports = [json.loads(p.read_text()) for p in args.results]
    if args.write is None:
        entries = json.loads(TABLE.read_text())["entries"]
        decisions = decide(reports, entries)
        print("\n".join(report_lines(decisions, default_proposals(decisions, entries))))
        return 0
    # An installed vllm_musa wins; a bare checkout validates with its own loader.
    sys.path.append(str(REPO))
    result = write(args.write, reports)
    print("\n".join(report_lines(result["decisions"], result["proposals"])))
    print(json.dumps({"write": str(args.write), "changes": result["changes"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
