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
    indices_dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    output = torch.full(
        (logits.shape[0], capacity if capacity is not None else topk_tokens),
        TAIL_SENTINEL,
        dtype=indices_dtype,
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("route", ["deep_select", "auto", "nonzero", "unaligned"])
def test_vllm_indexer_routes_to_native_operators(
    dtype: torch.dtype, route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm.model_executor.layers import sparse_attn_indexer as indexer
    from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk

    width = 4100 if route == "unaligned" else 4096
    logits = torch.randn(3, width, device="musa", dtype=dtype)
    output = torch.full((3, 512), TAIL_SENTINEL, device="musa", dtype=torch.int32)
    starts = torch.tensor(
        [1, 16, 128] if route == "nonzero" else [0, 0, 0],
        device="musa",
        dtype=torch.int32,
    )
    ends = torch.tensor([23, width, 1024], device="musa", dtype=torch.int32)
    backend = "auto" if route == "auto" else "deep_select"
    calls = []
    native = indexer.deep_select_topk

    def record_native(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(indexer, "deep_select_topk", record_native)
    if route == "deep_select":
        assert SparseIndexerTopk(backend).resolve_backend(logits, 512, 3) == backend
    indexer._musa_select_indexer_topk_windows(
        logits, starts, ends, output, 512, route != "nonzero", backend
    )
    torch.musa.synchronize()
    assert len(calls) == int(route == "deep_select")
    for row, (start, end) in enumerate(zip(starts.cpu().tolist(), ends.cpu().tolist())):
        _assert_topk(logits[row : row + 1, start:end], output[row : row + 1], 512)


def test_alignment_requirement() -> None:
    assert torch.ops.deep_select.get_alignment_requirement() == (32, 32)
    assert torch._C._dispatch_has_kernel_for_dispatch_key("deep_select::topk", "MUSA")
    assert torch._C._dispatch_has_kernel_for_dispatch_key("deep_select::topk", "Meta")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fake_tensor_dispatch(dtype: torch.dtype) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        logits = torch.empty((2, 4096), device="musa", dtype=dtype)
        output = _call(logits, 512)
        assert output.shape == (2, 512)
        assert output.dtype == torch.int32
        assert output.device == logits.device


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


@pytest.mark.parametrize(
    "width, topk_tokens", [(4096, 512), (16384, 512), (16384, 1024)]
)
def test_sorted_values_match_reference(width: int, topk_tokens: int) -> None:
    torch.manual_seed(7)
    logits = torch.randn(2, width, device="musa", dtype=torch.float32)
    end = torch.tensor([23, width], dtype=torch.int32, device="musa")
    output = torch.full(
        (2, topk_tokens + 16), TAIL_SENTINEL, device="musa", dtype=torch.int64
    )
    values = torch.full((2, topk_tokens + 16), 7.0, device="musa", dtype=logits.dtype)
    torch.ops.deep_select.topk(
        logits,
        topk_tokens,
        None,
        end,
        True,
        False,
        values,
        output,
        None,
        FILL,
        float("-inf"),
        True,
        False,
    )
    _assert_topk(logits, output, topk_tokens, end)
    selected_values = values[:, :topk_tokens].cpu()
    assert torch.all(selected_values[:, :-1] >= selected_values[:, 1:])
    reference = logits.cpu()
    indices = output[:, :topk_tokens].cpu().long()
    for row_index, row_end in enumerate(end.cpu().tolist()):
        count = min(topk_tokens, row_end)
        assert torch.equal(
            selected_values[row_index, :count],
            reference[row_index, indices[row_index, :count]],
        )
        assert torch.all(selected_values[row_index, count:] == float("-inf"))
    assert torch.all(output[:, topk_tokens:] == TAIL_SENTINEL)
    assert torch.all(values[:, topk_tokens:] == 7.0)


def test_value_sort_requires_values_before_launch() -> None:
    logits = torch.randn(2, 4096, device="musa", dtype=torch.float32)
    output = torch.full((2, 512), TAIL_SENTINEL, device="musa", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="sorted_value requires return_value"):
        torch.ops.deep_select.topk(
            logits,
            512,
            None,
            None,
            True,
            False,
            None,
            output,
            None,
            FILL,
            float("-inf"),
            False,
            False,
        )
    assert torch.all(output == TAIL_SENTINEL)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("aligned", [False, True])
def test_input_storage_offset_alignment(dtype: torch.dtype, aligned: bool) -> None:
    from vllm.model_executor.layers.indexer_topk import is_deep_select_supported

    storage = torch.randn(2, 4112, device="musa", dtype=dtype)
    offset = 32 // storage.element_size() if aligned else 1
    logits = storage[:, offset : offset + 4096]
    assert logits.stride(0) * logits.element_size() % 32 == 0
    assert is_deep_select_supported(logits, 512) is aligned
    if aligned:
        _assert_topk(logits, _call(logits, 512), 512)
    else:
        with pytest.raises(RuntimeError, match="input data must be 32-byte aligned"):
            _call(logits, 512)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "rows, width, topk_tokens",
    [
        (0, 64, 7),
        (1, 64, 7),
        (4, 64, 23),
        (1, 4096, 512),
        (4, 32768, 4096),
        (1, 131072, 512),
    ],
)
def test_shape_and_index_dtype_matrix(
    dtype: torch.dtype,
    indices_dtype: torch.dtype,
    rows: int,
    width: int,
    topk_tokens: int,
) -> None:
    torch.manual_seed(rows + width + topk_tokens)
    logits = torch.randn(rows, width, device="musa", dtype=dtype)
    capacity = ((topk_tokens + 7) // 8) * 8
    output = _call(logits, topk_tokens, capacity=capacity, indices_dtype=indices_dtype)
    _assert_topk(logits, output, topk_tokens)
    assert output.dtype == indices_dtype
    assert torch.all(output[:, topk_tokens:] == TAIL_SENTINEL)


@pytest.mark.parametrize(
    "invalid, message",
    [
        ("topk", "topk must be"),
        ("input_rank", "2D tensor"),
        ("input_dtype", "input must be float32 or bfloat16"),
        ("output_dtype", "int32 or int64"),
        ("output_rows", "output_index must have at least"),
        ("output_stride", "int32 or int64 MUSA tensor"),
        ("output_device", "int32 or int64 MUSA tensor"),
        ("end_dtype", "end must be contiguous int32"),
        ("end_rows", "end must be contiguous int32"),
        ("end_stride", "end must be contiguous int32"),
        ("begin", "begin is not supported"),
        ("output_value", "output_value must be provided"),
    ],
)
def test_invalid_arguments_fail_before_launch(invalid: str, message: str) -> None:
    logits = torch.randn(2, 4096, device="musa", dtype=torch.float32)
    output = torch.full((2, 512), TAIL_SENTINEL, device="musa", dtype=torch.int32)
    options = dict(
        input=logits,
        topk=512,
        begin=None,
        end=None,
        sorted_value=False,
        sorted_index=False,
        output_value=None,
        output_index=output,
        output_idx_offset=None,
        idx_oob_fill_value=FILL,
        value_oob_fill_value=float("-inf"),
        return_value=False,
        abort_when_nan_found=False,
    )
    if invalid == "topk":
        options["topk"] = 0
    elif invalid == "input_rank":
        options["input"] = logits[0]
    elif invalid == "input_dtype":
        options["input"] = logits.to(torch.float16)
    elif invalid == "output_dtype":
        options["output_index"] = output.float()
    elif invalid == "output_rows":
        options["output_index"] = output[:1]
    elif invalid == "output_stride":
        options["output_index"] = torch.full(
            (2, 1024), TAIL_SENTINEL, device="musa", dtype=torch.int32
        )[:, ::2]
    elif invalid == "output_device":
        options["output_index"] = output.cpu()
    elif invalid == "end_dtype":
        options["end"] = torch.full((2,), 4096, device="musa", dtype=torch.int64)
    elif invalid == "end_rows":
        options["end"] = torch.full((1,), 4096, device="musa", dtype=torch.int32)
    elif invalid == "end_stride":
        options["end"] = torch.full((4,), 4096, device="musa", dtype=torch.int32)[::2]
    elif invalid == "begin":
        options["begin"] = torch.zeros(2, device="musa", dtype=torch.int32)
    elif invalid == "output_value":
        options["return_value"] = True
    with pytest.raises(RuntimeError, match=message):
        torch.ops.deep_select.topk(**options)
    assert torch.all(options["output_index"] == TAIL_SENTINEL)


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_inductor_preserves_output_mutation(dtype: torch.dtype) -> None:
    assert not torch._dynamo.config.disable
    graphs_before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    logits = torch.randn(2, 4096, device="musa", dtype=dtype)
    output = torch.full((2, 512), TAIL_SENTINEL, device="musa", dtype=torch.int32)

    def select(input_logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        torch.ops.deep_select.topk(
            input_logits,
            512,
            None,
            None,
            False,
            False,
            None,
            indices,
            None,
            FILL,
            float("-inf"),
            False,
            False,
        )
        return indices

    compiled = torch.compile(select, backend="inductor", fullgraph=True)
    result = compiled(logits, output)
    torch.musa.synchronize()
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > graphs_before
    assert result.data_ptr() == output.data_ptr()
    _assert_topk(logits, output, 512)
    logits.normal_()
    compiled(logits, output)
    torch.musa.synchronize()
    _assert_topk(logits, output, 512)
