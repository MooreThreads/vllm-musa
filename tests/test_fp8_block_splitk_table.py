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
