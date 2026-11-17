# SPDX-License-Identifier: Apache-2.0
"""The numerical validation oracle must not depend on strided GPU indices."""

import ast
from pathlib import Path

import pytest
import torch


def _quantize(x, group_size, **kwargs):
    blocks = x.float().reshape(x.shape[0], -1, group_size)
    scales = blocks.abs().amax(dim=-1).clamp_min(1e-10) / 448.0
    quant = (blocks / scales[..., None]).to(torch.float8_e4m3fn)
    return quant.reshape_as(x), scales


def _helpers():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/fused_moe/validate_mate_mega_moe.py"
    )
    tree = ast.parse(path.read_text())
    wanted = {"block_quantize", "dequantize", "reference", "check_output"}
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {"torch": torch, "per_token_group_quant_fp8": _quantize}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def test_reference_normalizes_nonzero_column_indices(monkeypatch):
    helpers = _helpers()
    torch.manual_seed(310)
    hidden = intermediate = 128
    logical = [[], [], [], []]
    for _ in range(4):
        for offset, shape in (
            (0, (2 * intermediate, hidden)),
            (2, (hidden, intermediate)),
        ):
            q, scales = helpers["block_quantize"](torch.randn(shape) * 0.03)
            logical[offset].append(q)
            logical[offset + 1].append(scales)
    logical = tuple(torch.stack(table) for table in logical)
    x = (torch.randn(5, hidden) * 0.25).to(torch.bfloat16)
    ids = torch.tensor([[0, 1], [1, 2], [2, 0], [3, 1], [0, 3]])
    routing = torch.tensor([[0.7, 0.3]]).expand(5, 2).contiguous()
    reference = helpers["reference"](x, ids, routing, logical, intermediate)
    original_nonzero = torch.Tensor.nonzero
    original_add = torch.Tensor.index_add_
    calls = []

    def row_major_nonzero(tensor, *args, **kwargs):
        return original_nonzero(tensor, *args, **kwargs).contiguous()

    def checked_add(tensor, dim, index, source, *args, **kwargs):
        assert index.is_contiguous(), "GPU reference requires contiguous indices"
        calls.append(index.numel())
        return original_add(tensor, dim, index, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "nonzero", row_major_nonzero)
    monkeypatch.setattr(torch.Tensor, "index_add_", checked_add)
    observed = helpers["reference"](x, ids, routing, logical, intermediate)
    torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    assert len(calls) == 4
    assert helpers["check_output"](observed, reference)["passed"]


@pytest.mark.parametrize("bad", ["zero", "constant", "nan"])
def test_output_gate_rejects_poison(bad):
    helpers = _helpers()
    expected = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    actual = torch.zeros_like(expected)
    if bad == "constant":
        actual.fill_(1)
    elif bad == "nan":
        actual.fill_(float("nan"))
    assert not helpers["check_output"](actual, expected)["passed"]
