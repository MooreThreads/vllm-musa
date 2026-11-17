# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed batched all-gather: the dispatch tensors survive one byte gather."""

import pytest
import torch

from tests.moe_contract_utils import load_applied_contract

cc = load_applied_contract(
    "vllm/distributed/device_communicators/cuda_communicator.py",
    ["_row_bytes", "_pack_rows", "_unpack_rows", "_all_gather_packed"],
    {"torch": torch},
)


def _dispatch_tensors(rank: int, rows: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(rank)
    return [
        torch.randn(rows, 64, generator=g).to(torch.float8_e4m3fn),
        torch.rand(rows, 8, generator=g),
        torch.randint(0, 128, (rows, 8), generator=g, dtype=torch.int32),
        torch.rand(rows, 16, generator=g),
    ]


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


@pytest.mark.parametrize("rows", [[2, 2, 2, 2], [1, 3, 0, 2], [5]])
def test_one_gather_returns_every_ranks_rows(rows):
    ranks = [_dispatch_tensors(rank, n) for rank, n in enumerate(rows)]
    calls = []

    def gather(packed, sizes):
        calls.append((packed.dtype, tuple(packed.shape), sizes))
        return torch.cat([cc._pack_rows(tensors) for tensors in ranks])

    out = cc._all_gather_packed(ranks[0], gather, rows)

    assert calls == [(torch.uint8, (rows[0], 64 + 32 + 32 + 64), rows)]
    for i, got in enumerate(out):
        want = torch.cat([tensors[i] for tensors in ranks])
        assert got.dtype == want.dtype
        assert got.shape == want.shape
        assert got.is_contiguous()
        assert torch.equal(_bytes(got), _bytes(want))


def test_one_dimensional_tensors_round_trip():
    tensors = [
        torch.arange(3, dtype=torch.int64),
        torch.ones(3, 2, dtype=torch.bfloat16),
    ]
    out = cc._all_gather_packed(tensors, lambda packed, sizes: packed, None)
    assert [t.shape for t in out] == [t.shape for t in tensors]
    assert all(torch.equal(_bytes(a), _bytes(b)) for a, b in zip(out, tensors))


def test_mismatched_rows_fail_before_any_collective():
    a, b = torch.zeros(2, 4), torch.zeros(3, 4)
    seen = []

    def gather(t, sizes):
        seen.append(t)
        return t

    with pytest.raises(ValueError, match="row count"):
        cc._all_gather_packed([a, b], gather, None)
    assert seen == []
