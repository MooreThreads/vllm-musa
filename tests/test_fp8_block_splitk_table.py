# SPDX-License-Identifier: Apache-2.0
"""The split-K FP8 GEMM selection against a golden capture of its measured rows."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm_musa.kernels import fp8_block_splitk as sk  # noqa: E402

GOLDEN = json.loads(
    (Path(__file__).parent / "data" / "fp8_block_splitk_golden.json").read_text()
)


def _tensors(n: int, k: int):
    weight = torch.empty((n, k), dtype=torch.float8_e4m3fn)
    scale = torch.empty((-(-n // 32), k // 32), dtype=torch.float32)
    return weight, scale


@pytest.mark.parametrize("capturing", [True, False], ids=["capture", "eager"])
def test_selection_matches_golden(monkeypatch, capturing) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    # A multiprocessor count without rows of its own takes the default rows.
    monkeypatch.setattr(sk, "_multiprocessor_count", lambda index: 0)
    mode = "capture" if capturing else "eager"
    for shape in GOLDEN["shapes"]:
        weight, scale = _tensors(shape["n"], shape["k"])
        first, last = shape[mode]["rows"]
        config = tuple(shape[mode]["config"])
        for rows in range(GOLDEN["rows"] + 1):
            expected = config if first <= rows <= last else None
            got = sk.fp8_block_splitk_config(weight, scale, rows)
            assert got == expected, (shape["n"], shape["k"], mode, rows)
    for n, k in GOLDEN["unseen"]:
        weight, scale = _tensors(n, k)
        for rows in range(GOLDEN["rows"] + 1):
            assert sk.fp8_block_splitk_config(weight, scale, rows) is None, (n, k)


def _entry(**overrides) -> dict:
    entry = {
        "n": 1792,
        "k": 5120,
        "split": 16,
        "block_n": 64,
        "num_warps": 4,
        "unroll": 2,
        "tile_m": 32,
        "min_rows": 2,
        "max_rows": {"capture": 59, "eager": 192},
    }
    entry.update(overrides)
    return entry


def test_a_bin_row_replaces_the_default_row(monkeypatch) -> None:
    rows = {key: row for key, row in sk._ROWS.items() if key[2] is None}
    rows.update(sk._from_table([_entry(multiprocessor_count=48, min_rows=3)]))
    monkeypatch.setattr(sk, "_ROWS", rows)
    monkeypatch.setattr(sk, "_BINNED_SHAPES", frozenset({(1792, 5120)}))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    weight, scale = _tensors(1792, 5120)
    for mp, min_rows in ((48, 3), (60, 2)):
        monkeypatch.setattr(sk, "_multiprocessor_count", lambda index, mp=mp: mp)
        assert sk.fp8_block_splitk_config(weight, scale, min_rows - 1) is None
        assert sk.fp8_block_splitk_config(weight, scale, min_rows)[4] == min_rows


@pytest.mark.parametrize(
    "entry, message",
    [
        (_entry(extra=1), "keys must be"),
        ({k: v for k, v in _entry().items() if k != "tile_m"}, "keys must be"),
        (_entry(max_rows={"capture": 59}), "max_rows must have exactly"),
        (_entry(max_rows={"capture": 59, "eager": 8, "prefill": 9}), "exactly"),
        (_entry(min_rows=60), "min_rows exceeds max_rows"),
        (_entry(block_n={"capture": 64, "eager": 128}), "must be integers"),
        (_entry(max_rows={"capture": 59, "eager": {"x": 1}}), "must be integers"),
    ],
)
def test_a_bad_entry_is_refused(entry, message) -> None:
    with pytest.raises(ValueError, match=message):
        sk._from_table([entry])


def test_a_repeated_shape_and_bin_is_refused() -> None:
    with pytest.raises(ValueError, match="repeats"):
        sk._from_table([_entry(), _entry()])
    with pytest.raises(ValueError, match="repeats"):
        sk._from_table([_entry(multiprocessor_count=48)] * 2)
    assert len(sk._from_table([_entry(), _entry(multiprocessor_count=48)])) == 2


def test_selection_does_not_read_the_table(monkeypatch) -> None:
    from vllm_musa import tuning

    def unexpected(kernel):
        raise AssertionError("the table is read once, at import")

    monkeypatch.setattr(tuning, "load", unexpected)
    monkeypatch.setattr(sk, "_multiprocessor_count", lambda index: 0)
    weight, scale = _tensors(1792, 5120)
    for rows in range(1000):
        sk.fp8_block_splitk_config(weight, scale, rows % 64)


def test_the_multiprocessor_count_is_read_once_per_device(monkeypatch) -> None:
    calls = []

    def num_compute_units(device_index):
        calls.append(device_index)
        return 48

    monkeypatch.setattr(sk.current_platform, "num_compute_units", num_compute_units)
    sk._multiprocessor_count.cache_clear()
    try:
        assert [sk._multiprocessor_count(0) for _ in range(3)] == [48] * 3
        assert sk._multiprocessor_count(1) == 48
        assert calls == [0, 1]
    finally:
        sk._multiprocessor_count.cache_clear()
