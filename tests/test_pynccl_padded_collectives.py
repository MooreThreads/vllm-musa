# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Uneven pynccl all_gatherv/reduce_scatterv through one padded collective."""

import contextlib

import pytest
import torch

from tests.moe_contract_utils import load_applied_methods

pynccl = load_applied_methods(
    "vllm/distributed/device_communicators/pynccl.py",
    "PyNcclCommunicator",
    ["_all_gatherv_padded", "_reduce_scatterv_padded"],
    {"torch": torch},
)


@pytest.fixture
def no_stream(monkeypatch):
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())


def _comm(world, rank):
    comm = object.__new__(pynccl.PyNcclCommunicator)
    comm.world_size = world
    comm.rank = rank
    return comm


@pytest.mark.parametrize("sizes", [[1, 1, 20, 1], [3, 0, 2, 5], [4, 4, 4, 4]])
def test_all_gatherv_padded_compacts_each_ranks_rows(no_stream, sizes):
    world, max_size = len(sizes), max(sizes)
    per_rank = [
        torch.arange(size * 3, dtype=torch.float32).reshape(size, 3) + 100 * rank
        for rank, size in enumerate(sizes)
    ]
    comm = _comm(world, rank=0)

    def all_gather(out, inp, stream=None):
        assert inp.shape[0] == max_size
        assert torch.count_nonzero(inp[sizes[comm.rank] :]) == 0
        padded = []
        for rows in per_rank:
            block = torch.full((max_size, 3), -1.0)
            block[: rows.shape[0]] = rows
            padded.append(block)
        out.copy_(torch.cat(padded))

    comm.all_gather = all_gather
    out = torch.empty(sum(sizes), 3)
    comm._all_gatherv_padded(out, per_rank[0], sizes, stream=None)
    assert torch.equal(out, torch.cat(per_rank))


@pytest.mark.parametrize("rank", [0, 2, 3])
def test_reduce_scatterv_padded_returns_this_ranks_rows(no_stream, rank):
    sizes = [1, 1, 20, 1]
    world, max_size, total = len(sizes), max(sizes), sum(sizes)
    rows = torch.arange(total * 2, dtype=torch.float32).reshape(total, 2)
    comm = _comm(world, rank)

    def reduce_scatter(out, padded, op=None, stream=None):
        assert padded.shape[0] == world * max_size
        for peer, size in enumerate(sizes):
            assert (
                torch.count_nonzero(padded.view(world, max_size, 2)[peer, size:]) == 0
            )
        # Every rank contributes the same rows here, so the sum is world times them.
        out.copy_(padded.view(world, max_size, 2)[rank] * world)

    comm.reduce_scatter = reduce_scatter
    out = torch.empty(sizes[rank], 2)
    comm._reduce_scatterv_padded(out, rows, sizes, op=None, stream=None)
    start = sum(sizes[:rank])
    assert torch.equal(out, rows[start : start + sizes[rank]] * world)


@pytest.mark.parametrize("sizes", [[1, 1], [-1, 2, 0, 0]])
def test_gather_rejects_invalid_rank_sizes(no_stream, sizes):
    comm = _comm(4, rank=0)
    with pytest.raises(ValueError, match="nonnegative size per rank"):
        comm._all_gatherv_padded(torch.empty(2, 3), torch.empty(1, 3), sizes, None)


def test_gather_rejects_inconsistent_local_rows(no_stream):
    comm = _comm(4, rank=0)
    with pytest.raises(ValueError, match="local row count"):
        comm._all_gatherv_padded(
            torch.empty(4, 3), torch.empty(2, 3), [1, 1, 1, 1], None
        )


def test_reduce_rejects_inconsistent_local_rows(no_stream):
    comm = _comm(4, rank=0)
    with pytest.raises(ValueError, match="local row count"):
        comm._reduce_scatterv_padded(
            torch.empty(2, 3), torch.empty(4, 3), [1, 1, 1, 1], None, None
        )


def test_all_empty_sizes_are_a_no_op(no_stream):
    comm = _comm(4, rank=0)
    comm.all_gather = comm.reduce_scatter = None
    out = torch.empty(0, 3)
    comm._all_gatherv_padded(out, torch.empty(0, 3), [0, 0, 0, 0], stream=None)
    comm._reduce_scatterv_padded(out, torch.empty(0, 3), [0, 0, 0, 0], None, None)
