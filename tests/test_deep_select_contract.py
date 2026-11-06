# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the native MUSA top-k contract used by vLLM's indexer."""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")
if not getattr(torch.version, "musa", None):
    pytest.skip("requires a MUSA torch build", allow_module_level=True)
pytest.importorskip("torch_musa")
if not torch.musa.is_available():
    pytest.skip("requires MUSA hardware", allow_module_level=True)

deepselect_extension = importlib.import_module("vllm._deepselect_C")

FILL = -1
NAN_MARKER = 0x3F3F3F3F
TAIL_SENTINEL = 7


def _call(
    logits: torch.Tensor,
    topk_tokens: int,
    *,
    end: torch.Tensor | None = None,
    capacity: int | None = None,
) -> torch.Tensor:
    output = torch.full(
        (logits.shape[0], capacity if capacity is not None else topk_tokens),
        TAIL_SENTINEL,
        dtype=torch.int32,
        device=logits.device,
    )
    torch.ops.deep_select.topk(
        logits,
        topk_tokens,
        None,
        end,
        False,
        False,
        None,
        output,
        None,
        FILL,
        float("-inf"),
        False,
        False,
    )
    return output


def _assert_topk(
    logits: torch.Tensor,
    output: torch.Tensor,
    topk_tokens: int,
    end: torch.Tensor | None = None,
) -> None:
    reference = logits.cpu().float()
    indices = output[:, :topk_tokens].cpu().long()
    row_ends = (
        end.cpu().tolist() if end is not None else [logits.shape[1]] * logits.shape[0]
    )
    for row_index, row_end in enumerate(row_ends):
        count = min(topk_tokens, row_end)
        selected = indices[row_index, :count]
        assert torch.all((selected >= 0) & (selected < row_end))
        assert selected.unique().numel() == count
        actual = reference[row_index, selected]
        expected = torch.topk(reference[row_index, :row_end], count).values
        assert torch.equal(actual.sort().values, expected.sort().values)
        assert torch.all(indices[row_index, count:] == FILL)


def test_alignment_requirement() -> None:
    assert torch.ops.deep_select.get_alignment_requirement() == (32, 32)
    assert torch._C._dispatch_has_kernel_for_dispatch_key("deep_select::topk", "MUSA")
    assert torch._C._dispatch_has_kernel_for_dispatch_key("deep_select::topk", "Meta")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("topk_tokens", [512, 1024])
def test_full_rows_match_reference(dtype: torch.dtype, topk_tokens: int) -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, 4096, device="musa", dtype=dtype)
    output = _call(logits, topk_tokens)
    _assert_topk(logits, output, topk_tokens)
    assert output.dtype == torch.int32


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_end_bounds_pad_the_tail(dtype: torch.dtype) -> None:
    torch.manual_seed(1)
    logits = torch.randn(5, 2048, device="musa", dtype=dtype)
    end = torch.tensor([0, 23, 512, 1000, 2048], dtype=torch.int32, device="musa")
    output = _call(logits, 512, end=end)
    _assert_topk(logits, output, 512, end)


def test_k_smaller_than_capacity_keeps_the_tail() -> None:
    torch.manual_seed(2)
    logits = torch.randn(2, 512, device="musa", dtype=torch.float32)
    end = torch.tensor([23, 512], dtype=torch.int32, device="musa")
    output = _call(logits, 23, end=end, capacity=2048)
    _assert_topk(logits, output, 23, end)
    assert torch.all(output[:, 23:] == TAIL_SENTINEL)


def test_nan_marker_is_written_without_abort() -> None:
    torch.manual_seed(3)
    logits = torch.randn(4, 1024, device="musa", dtype=torch.float32)
    logits[1, 3] = float("nan")
    output = _call(logits, 64)
    assert int(output[1, 0]) == NAN_MARKER
    _assert_topk(logits[[0, 2, 3]], output[[0, 2, 3]], 64)


def test_topk_larger_than_vocab() -> None:
    torch.manual_seed(4)
    logits = torch.randn(2, 8, device="musa", dtype=torch.float32)
    end = torch.tensor([5, 8], dtype=torch.int32, device="musa")
    output = _call(logits, 16, end=end)
    _assert_topk(logits, output, 16, end)


def test_tied_bf16_values_give_unique_indices() -> None:
    torch.manual_seed(5)
    logits = torch.randint(0, 3, (4, 4096), device="musa").to(torch.bfloat16)
    output = _call(logits, 256)
    _assert_topk(logits, output, 256)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_graph_replay_uses_current_logits(dtype: torch.dtype) -> None:
    torch.manual_seed(6)
    logits = torch.randn(2, 131072, device="musa", dtype=dtype)
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        _call(logits, 512)
    stream.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=stream):
        output = _call(logits, 512)
    logits.normal_()
    torch.musa.synchronize()
    graph.replay()
    torch.musa.synchronize()
    _assert_topk(logits, output, 512)
