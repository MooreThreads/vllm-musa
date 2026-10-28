# SPDX-License-Identifier: Apache-2.0
"""The resolve ladder: table, then a computed bound, then the caller's default."""

from __future__ import annotations

from pathlib import Path

import pytest

from vllm_musa import tactics
from vllm_musa.tactics import MoeTactic, TuningProblem

MOE_DEFAULT = {
    "block_m": 64,
    "block_n": 64,
    "block_k": 32,
    "split_k": 1,
    "num_stages": 3,
}
MOE_MP56 = {"block_m": 32, "block_n": 128, "block_k": 64, "split_k": 2, "num_stages": 2}


class Spec:
    def __init__(self, mp_count: int = 56) -> None:
        self.arch_name = "mp_31"
        self.mp_count = mp_count
        self.warp_size = 32
        self.shared_memory_per_block = 96 * 1024
        self.shared_memory_per_multiprocessor = 192 * 1024
        self.regs_per_multiprocessor = 64 * 1024
        self.max_threads_per_block = 1024
        self.max_threads_per_multiprocessor = 2048


def write_table(directory: Path, entries: list[dict]) -> Path:
    import json

    path = directory / "moe.json"
    path.write_text(
        json.dumps(
            {"_schema": tactics.SCHEMA_VERSION, "kernel": "moe", "tactics": entries}
        )
    )
    return path


def entry(
    values: dict,
    *,
    arch=None,
    mp_count=None,
    m_range=None,
    shape=None,
    graph_capture=None,
    note="",
) -> dict:
    match = {
        "arch": arch,
        "mp_count": mp_count,
        "m_range": m_range,
        "shape": shape,
        "graph_capture": graph_capture,
    }
    return {
        "match": {k: v for k, v in match.items() if v is not None},
        "values": values,
        "note": note,
    }


def problem(mp_count: int = 56, m: int = 8) -> TuningProblem:
    return TuningProblem(
        layer=tactics.LayerShape(
            shape_n=4096, shape_k=4096, dtype="bf16", weight_dtype="fp8"
        ),
        m=m,
        device=tactics.DeviceProfile.from_spec(Spec(mp_count)),
    )


@pytest.fixture()
def table(tmp_path: Path) -> Path:
    write_table(
        tmp_path,
        [
            entry(MOE_DEFAULT),
            entry(MOE_MP56, arch="mp_31", mp_count=56, note="measured on 10.20.36.7"),
        ],
    )
    return tmp_path


def test_a_missing_row_with_no_default_raises(tmp_path: Path) -> None:
    write_table(tmp_path, [entry(MOE_MP56, arch="mp_31", mp_count=56, note="measured")])
    with pytest.raises(tactics.TacticNotFound, match="default="):
        tactics.resolve_tactic("moe", problem(mp_count=60), directory=tmp_path)


def test_the_caller_default_is_an_explicit_opt_in(tmp_path: Path) -> None:
    write_table(tmp_path, [entry(MOE_MP56, arch="mp_31", mp_count=56, note="measured")])
    fallback = MoeTactic.from_values(MOE_DEFAULT)
    got = tactics.resolve_tactic(
        "moe", problem(mp_count=60), default=fallback, directory=tmp_path
    )
    assert got == fallback


def test_the_caller_default_never_overrides_a_real_entry(table: Path) -> None:
    got = tactics.resolve_tactic(
        "moe", problem(), default=MoeTactic.from_values(MOE_DEFAULT), directory=table
    )
    assert got == MoeTactic.from_values(MOE_MP56)


def test_the_resolver_is_a_pure_function(table: Path) -> None:
    first = tactics.resolve_tactic("moe", problem(), directory=table)
    second = tactics.resolve_tactic("moe", problem(), directory=table)
    assert first == second


def test_a_computed_answer_wins_over_the_caller_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_table(tmp_path, [entry(MOE_MP56, arch="mp_31", mp_count=56, note="measured")])
    computed = MoeTactic.from_values(MOE_DEFAULT)
    monkeypatch.setitem(tactics.COMPUTED_FALLBACKS, "moe", lambda problem: computed)
    got = tactics.resolve_tactic(
        "moe",
        problem(mp_count=60),
        default=MoeTactic.from_values(MOE_MP56),
        directory=tmp_path,
    )
    assert got == computed


def test_a_computed_fallback_returning_none_still_reaches_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_table(tmp_path, [entry(MOE_MP56, arch="mp_31", mp_count=56, note="measured")])
    monkeypatch.setitem(tactics.COMPUTED_FALLBACKS, "moe", lambda problem: None)
    fallback = MoeTactic.from_values(MOE_DEFAULT)
    got = tactics.resolve_tactic(
        "moe", problem(mp_count=60), default=fallback, directory=tmp_path
    )
    assert got == fallback
