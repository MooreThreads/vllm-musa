# SPDX-License-Identifier: Apache-2.0
"""The measured-table loader in vllm_musa.tuning; needs no torch."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from vllm_musa import tuning

ROOT = Path(__file__).resolve().parents[1]
TABLES = sorted((ROOT / "vllm_musa" / "tuning").glob("*.json"))


def _text(**overrides) -> str:
    table = {
        "schema": 1,
        "kernel": "demo",
        "source": "MTT S5000, cold L2",
        "entries": [{"n": 64, "k": 128, "limits": {"capture": 8, "eager": 16}}],
    }
    table.update(overrides)
    return json.dumps(table)


def test_a_valid_table_parses() -> None:
    entries = tuning._parse("demo.json", "demo", _text())
    assert entries == [{"n": 64, "k": 128, "limits": {"capture": 8, "eager": 16}}]


@pytest.mark.parametrize("path", TABLES, ids=lambda p: p.name)
def test_every_shipped_table_loads(path) -> None:
    assert tuning.load(path.stem)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"schema": 2}, "schema must be 1"),
        ({"kernel": "other"}, "kernel must be 'demo'"),
        ({"source": " "}, "source must name"),
        ({"entries": []}, "non-empty list"),
        ({"entries": [3]}, "entry 0: must be an object"),
        ({"entries": [{}]}, "entry 0: entry is empty"),
        ({"entries": [{"n": 0}]}, "entry 0: n must be a positive integer"),
        ({"entries": [{"n": -1}]}, "n must be a positive integer"),
        ({"entries": [{"n": True}]}, "n must be a positive integer"),
        ({"entries": [{"n": 1.5}]}, "n must be a positive integer"),
        ({"entries": [{"n": "64"}]}, "n must be a positive integer"),
        ({"entries": [{"n": 1, "m": {"capture": 0}}]}, "m.capture must be"),
        ({"entries": [{"n": 1, "m": {}}]}, "entry 0: m is empty"),
    ],
)
def test_a_bad_table_is_refused(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        tuning._parse("demo.json", "demo", _text(**overrides))


def test_unknown_or_missing_top_level_keys_are_refused() -> None:
    table = json.loads(_text())
    table["note"] = "x"
    with pytest.raises(ValueError, match="top-level keys"):
        tuning._parse("demo.json", "demo", json.dumps(table))
    del table["note"], table["source"]
    with pytest.raises(ValueError, match="top-level keys"):
        tuning._parse("demo.json", "demo", json.dumps(table))


def test_tables_are_package_data() -> None:
    assert '"tuning/*.json"' in (ROOT / "pyproject.toml").read_text()


def test_tuning_imports_without_torch() -> None:
    code = (
        "import sys; sys.modules['torch'] = None\n"
        "import vllm_musa.tuning as t\n"
        "print(t.FUSED_ADD_RMSNORM_MIN_ROWS)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "64"
