# SPDX-License-Identifier: Apache-2.0
"""The split-K FP8 GEMM selects from the tactic table, byte for byte as before.

The golden file was captured from ``kernels/fp8_block_splitk.py`` at
``0b155a2618``, before the migration: for every ``(N, K)`` the old code carried,
and for every row count in ``1..400``, the tuple it returned and the band it
admitted.  This test replays that grid, so any drift in the numbers or in the
band boundaries is a failure rather than a quiet perf change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_musa.tactics.selectors.fp8_block_splitk import config_for

GOLDEN = Path(__file__).parent / "data" / "fp8_block_splitk_golden.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text())


def test_every_shape_and_row_matches_the_pre_migration_behaviour(golden: dict) -> None:
    low, high = golden["grid"]["rows"]
    for shape_key, modes in golden["shapes"].items():
        shape_n, shape_k = (int(value) for value in shape_key.split("x"))
        for mode in modes:
            capturing = mode["capturing"]
            expected = tuple(mode["config"])
            admitted = mode["admitted"]
            for rows in range(low, high + 1):
                got = config_for(shape_n, shape_k, rows, graph_capture=capturing)
                if admitted is None or not admitted[0] <= rows <= admitted[1]:
                    assert got is None, (
                        f"{shape_key} capture={capturing} rows={rows} was outside "
                        "the band before the migration but now answers"
                    )
                else:
                    assert got is not None, (
                        f"{shape_key} capture={capturing} rows={rows} was inside "
                        "the band before the migration but now returns None"
                    )
                    # The launch-affecting fields must be identical: split,
                    # block_n, num_warps, unroll, tile_m.
                    assert got[0:4] + got[6:] == expected[0:4] + expected[6:], (
                        f"{shape_key} capture={capturing} rows={rows}: "
                        f"{got} != {expected}"
                    )
                    # Fields 4 and 5 are the band, which the old code got wrong
                    # for eager rows: it returned the capture-mode constant (59)
                    # for a launch that admitted 192 rows. Nothing reads them;
                    # here they are the band the matched row was measured for.
                    assert (got[4], got[5]) == tuple(admitted), (
                        f"{shape_key} capture={capturing} rows={rows}: band "
                        f"{(got[4], got[5])} != {tuple(admitted)}"
                    )


def test_a_shape_outside_the_table_fails_closed() -> None:
    assert config_for(2048, 2048, 8, graph_capture=False) is None


def test_an_empty_batch_is_not_a_tuning_problem() -> None:
    assert config_for(1792, 5120, 0, graph_capture=False) is None


def test_the_capture_mode_can_change_the_band_it_admits() -> None:
    # 1792x5120 admits 192 rows eagerly and only 59 inside a capture.
    eager = config_for(1792, 5120, 192, graph_capture=False)
    captured = config_for(1792, 5120, 192, graph_capture=True)
    assert eager is not None and captured is None
    assert config_for(1792, 5120, 59, graph_capture=True) is not None
    assert config_for(1792, 5120, 60, graph_capture=True) is None
