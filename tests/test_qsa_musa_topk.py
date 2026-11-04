# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checks for the MUSA QSA top-k dispatch."""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer


@pytest.mark.parametrize("columns,block_topk", [(8, 4), (3, 5), (0, 4)])
def test_musa_topk_ignores_unwritten_logits(monkeypatch, columns, block_topk):
    monkeypatch.setattr(qsa_indexer.current_platform, "is_musa", lambda: True)
    lengths = [0, min(2, columns), columns]
    logits = torch.full((len(lengths), columns), float("nan"))
    expected = []
    for row, length in enumerate(lengths):
        logits[row, :length] = torch.arange(length, dtype=torch.float32)
        selected = list(range(length - 1, max(-1, length - block_topk - 1), -1))
        expected.append(selected + [-1] * (block_topk - len(selected)))
    visible = torch.tensor(lengths, dtype=torch.int32)
    out = torch.full((len(lengths), block_topk), 123, dtype=torch.int32)
    qsa_indexer._topk(logits, visible, block_topk * 4, 4, out, torch.empty(0))
    torch.testing.assert_close(out, torch.tensor(expected, dtype=torch.int32))


def test_musa_topk_handles_empty_batch(monkeypatch):
    monkeypatch.setattr(qsa_indexer.current_platform, "is_musa", lambda: True)
    logits = torch.empty((0, 3), dtype=torch.float32)
    visible = torch.empty(0, dtype=torch.int32)
    out = torch.empty((0, 5), dtype=torch.int32)
    qsa_indexer._topk(logits, visible, 20, 4, out, torch.empty(0))
    assert out.shape == (0, 5)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA required")
@pytest.mark.parametrize(
    "block_topk,lengths",
    [
        (512, [0, 1, 17, 511, 512, 513, 1024, 4096]),
        (2048, [0, 1, 1024, 2047, 2048, 2049, 4096, 4095]),
    ],
)
def test_musa_fast_topk_matches_torch(block_topk, lengths):
    device = torch.device("musa")
    columns = 4096
    torch.manual_seed(0)
    logits = torch.randn((len(lengths), columns), dtype=torch.float32, device=device)
    visible = torch.tensor(lengths, dtype=torch.int32, device=device)
    out = torch.full((len(lengths), block_topk), -2, dtype=torch.int32, device=device)
    workspace = torch.empty(len(lengths) * 4, dtype=torch.uint8, device=device)

    qsa_indexer._topk(logits, visible, block_topk * 4, 4, out, workspace)
    torch.musa.synchronize()

    for row, length in enumerate(lengths):
        got = set(int(x) for x in out[row].tolist() if int(x) >= 0)
        assert len(got) == min(length, block_topk)
        if length <= block_topk:
            expected = set(range(length))
        else:
            expected = set(
                int(x)
                for x in torch.topk(logits[row, :length], block_topk).indices.tolist()
            )
        assert got == expected, (block_topk, row, length, got ^ expected)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA required")
def test_musa_fast_topk_reuses_preallocated_workspace():
    device = torch.device("musa")
    rows, columns, block_topk = 8, 4096, 512
    logits = torch.randn((rows, columns), dtype=torch.float32, device=device)
    visible = torch.full((rows,), columns, dtype=torch.int32, device=device)
    out = torch.empty((rows, block_topk), dtype=torch.int32, device=device)
    workspace = torch.empty(rows * 4, dtype=torch.uint8, device=device)

    qsa_indexer._topk(logits, visible, block_topk * 4, 4, out, workspace)
    torch.musa.synchronize()
    before = torch.musa.memory_allocated()
    for _ in range(8):
        qsa_indexer._topk(logits, visible, block_topk * 4, 4, out, workspace)
    torch.musa.synchronize()
    after = torch.musa.memory_allocated()
    assert after == before, (before, after)
