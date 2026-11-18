from __future__ import annotations

import ast
import os
import sys
from bisect import bisect_left, bisect_right
from math import prod
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "vllm_musa/kernels/sparse_mqa.py"


class _FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: str,
        *,
        strides: tuple[int, ...] | None = None,
        values: tuple[int, ...] | None = None,
        device: str = "musa",
    ) -> None:
        if any(dimension < 0 for dimension in shape):
            raise ValueError("tensor dimensions must be nonnegative")
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device
        contiguous = []
        stride = 1
        for dimension in reversed(shape):
            contiguous.append(stride)
            stride *= max(dimension, 1)
        self._strides = tuple(reversed(contiguous)) if strides is None else strides
        assert len(self._strides) == self.ndim
        assert values is None or len(values) == self.numel()
        self._values = values
        self.base = self
        self._offset = 0

    def stride(self, axis: int | None = None) -> int | tuple[int, ...]:
        return self._strides if axis is None else self._strides[axis]

    def numel(self) -> int:
        return prod(self.shape)

    def storage_offset(self) -> int:
        return self._offset

    def is_contiguous(self) -> bool:
        if self.numel() == 0:
            return True
        expected = 1
        for dimension, stride in zip(reversed(self.shape), reversed(self._strides)):
            if dimension != 1 and stride != expected:
                return False
            expected *= max(dimension, 1)
        return True

    def __getitem__(self, index: slice) -> _FakeTensor:
        if not isinstance(index, slice):
            raise TypeError("host tensor fixture supports leading-axis slices only")
        start, stop, step = index.indices(self.shape[0])
        assert step > 0
        selected = range(start, stop, step)
        row_width = prod(self.shape[1:])
        values = (
            None
            if self._values is None
            else tuple(
                value
                for row in selected
                for value in self._values[row * row_width : (row + 1) * row_width]
            )
        )
        view = _FakeTensor(
            (len(selected), *self.shape[1:]),
            self.dtype,
            strides=(self._strides[0] * step, *self._strides[1:]),
            values=values,
            device=self.device,
        )
        view.base = self.base
        view._offset = self._offset + start * self._strides[0]
        return view

    def contiguous(self) -> _FakeTensor:
        if self.is_contiguous():
            return self
        return _FakeTensor(
            self.shape, self.dtype, values=self._values, device=self.device
        )

    def to(self, dtype: str) -> _FakeTensor:
        if dtype == self.dtype:
            return self
        return _FakeTensor(self.shape, dtype, values=self._values, device=self.device)

    def reshape(self, *shape: int) -> _FakeTensor:
        assert self.is_contiguous()
        if -1 in shape:
            assert shape.count(-1) == 1
            known = prod(dimension for dimension in shape if dimension != -1)
            assert known > 0 and self.numel() % known == 0
            shape = tuple(
                self.numel() // known if dimension == -1 else dimension
                for dimension in shape
            )
        assert prod(shape) == self.numel()
        view = _FakeTensor(shape, self.dtype, values=self._values, device=self.device)
        view.base = self.base
        view._offset = self._offset
        return view

    def tolist(self) -> list:
        assert self.ndim == 1 and self._values is not None
        return list(self._values)

    def __sub__(self, other: _FakeTensor) -> _FakeTensor:
        assert self.shape == other.shape and self.device == other.device
        return _FakeTensor(
            self.shape,
            self.dtype,
            values=tuple(
                left - right for left, right in zip(self.tolist(), other.tolist())
            ),
            device=self.device,
        )

    def __floordiv__(self, divisor: int) -> _FakeTensor:
        return _FakeTensor(
            self.shape,
            self.dtype,
            values=tuple(value // divisor for value in self.tolist()),
            device=self.device,
        )


def _arange(stop: int, *, device: str, dtype: str) -> _FakeTensor:
    return _FakeTensor((stop,), dtype, values=tuple(range(stop)), device=device)


def _searchsorted(
    boundaries: _FakeTensor, values: _FakeTensor, *, right: bool = False
) -> _FakeTensor:
    assert boundaries.device == values.device
    search = bisect_right if right else bisect_left
    return _FakeTensor(
        values.shape,
        "int64",
        values=tuple(search(boundaries.tolist(), value) for value in values.tolist()),
        device=values.device,
    )


def _host_functions() -> dict:
    names = {
        "_sparse_width",
        "_score_and_select",
        "_score_and_select_rows",
        "select_candidate_blocks",
        "try_candidate_indexer",
    }
    nodes = [
        node
        for node in ast.parse(MODULE.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "os": os,
        "torch": SimpleNamespace(
            float8_e4m3fn="fp8",
            uint8="uint8",
            float32="fp32",
            int32="int32",
            int64="int64",
            arange=_arange,
            empty=lambda shape, *, device, dtype: _tensor(shape, dtype, device=device),
            empty_like=lambda tensor: _tensor(
                tensor.shape, tensor.dtype, device=tensor.device
            ),
            searchsorted=_searchsorted,
        ),
        "triton": SimpleNamespace(
            cdiv=lambda numerator, denominator: (
                (numerator + denominator - 1) // denominator
            ),
            next_power_of_2=lambda value: 1 << (value - 1).bit_length(),
        ),
    }
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(MODULE), "exec"), namespace)
    return namespace


def _tensor(shape: tuple[int, ...], dtype: str, **kwargs) -> _FakeTensor:
    return _FakeTensor(shape, dtype, **kwargs)


@pytest.fixture
def scoring_calls(monkeypatch):
    namespace = _host_functions()
    kernels = {}
    for name in (
        "_candidate_block_scores_kernel",
        "_store_candidate_blocks_kernel",
        "_compact_candidate_blocks_kernel",
        "_paged_candidate_mqa_kernel",
        "_remap_candidate_topk_kernel",
    ):
        kernel = MagicMock()
        namespace[name] = kernel
        kernels[name] = kernel.__getitem__.return_value
    namespace["_topk"] = Mock()
    mask = Mock()
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.kernels.attention.dsa.candidate_blocks",
        SimpleNamespace(apply_candidate_mask=mask),
    )
    inputs = dict(
        query=_tensor((2, 32, 128), "fp8"),
        cache=_tensor((8, 64, 132), "uint8"),
        weights=_tensor((2, 32), "fp32"),
        block_table=_tensor((2, 4), "int32"),
        requests=None,
        lengths=_tensor((2,), "int32", values=(79, 1)),
        candidates=_tensor((2, 13), "int32"),
        candidate_block_size=8,
        output=_tensor((2, 16), "int32"),
        dense_width=256,
        candidate_write=False,
    )
    return SimpleNamespace(
        namespace=namespace, kernels=kernels, mask=mask, inputs=inputs
    )


@pytest.mark.parametrize("dense_width", [64, 104])
def test_dense_masked_consumer_filters_native_topk_output(scoring_calls, dense_width):
    scoring_calls.inputs["dense_width"] = dense_width
    scoring_calls.namespace["_score_and_select_rows"](**scoring_calls.inputs)
    scoring_calls.mask.assert_called_once()
    columns = scoring_calls.namespace["_topk"].call_args.args[2]
    assert columns is not scoring_calls.inputs["output"]
    remap = scoring_calls.kernels["_remap_candidate_topk_kernel"]
    remap.assert_called_once()
    assert remap.call_args.args[2] is columns
    assert remap.call_args.args[3] is scoring_calls.inputs["output"]
    assert remap.call_args.args[-1] is False


@pytest.mark.parametrize("sorted_unique", [False, True])
def test_only_sorted_unique_candidates_skip_compaction(scoring_calls, sorted_unique):
    scoring_calls.namespace["_score_and_select_rows"](
        **scoring_calls.inputs, candidates_sorted_unique=sorted_unique
    )
    compact = scoring_calls.kernels["_compact_candidate_blocks_kernel"]
    assert compact.call_count == int(not sorted_unique)
    score = scoring_calls.kernels["_paged_candidate_mqa_kernel"]
    assert score.call_args.args[-3] is sorted_unique


def test_candidate_source_keeps_direct_topk_fastpath(scoring_calls):
    scoring_calls.inputs["candidate_write"] = True
    scoring_calls.namespace["_score_and_select_rows"](**scoring_calls.inputs)
    assert (
        scoring_calls.namespace["_topk"].call_args.args[2]
        is scoring_calls.inputs["output"]
    )
    scoring_calls.kernels["_remap_candidate_topk_kernel"].assert_not_called()


@pytest.mark.parametrize("topk", [1, 13, 512, 2048])
def test_candidate_producer_store_gets_visible_counts_and_one_row_grid(
    scoring_calls, topk
):
    namespace = scoring_calls.namespace
    output = _tensor((2, topk), "int32", strides=(topk * 2, 2))
    namespace["select_candidate_blocks"](
        _tensor((2, 32768), "fp32"), None, scoring_calls.inputs["lengths"], 8, output
    )
    namespace["_store_candidate_blocks_kernel"].__getitem__.assert_called_once_with(
        (2,)
    )
    counts = scoring_calls.kernels["_candidate_block_scores_kernel"].call_args.args[4]
    store = scoring_calls.kernels["_store_candidate_blocks_kernel"].call_args.args
    assert store[2] is counts
    assert store[-1] == 1 << (topk - 1).bit_length()


def _prefill_chunk(
    start: int,
    stop: int,
    cumulative: tuple[int, ...],
    starts: tuple[int, ...],
    ends: tuple[int, ...],
    *,
    table_columns: int = 3,
    local_cumulative: tuple[int, ...] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        token_start=start,
        token_end=stop,
        cu_seq_lens=_tensor((len(cumulative),), "int32", values=cumulative),
        cu_seqlen_ks=_tensor((stop - start,), "int32", values=starts),
        cu_seqlen_ke=_tensor((stop - start,), "int32", values=ends),
        total_seq_lens=cumulative[-1],
        num_reqs=len(cumulative) - 1,
        block_table=_tensor((len(cumulative) - 1, table_columns), "int32"),
        local_cu_seq_lens=(
            None
            if local_cumulative is None
            else _tensor((len(local_cumulative),), "int32", values=local_cumulative)
        ),
        local_total_seq_lens=(
            cumulative[-1] if local_cumulative is None else local_cumulative[-1]
        ),
    )


@pytest.fixture
def candidate_call() -> SimpleNamespace:
    namespace = _host_functions()
    score = Mock()
    namespace["_score_and_select"] = score
    inputs = dict(
        query=_tensor((4, 32, 128), "fp8"),
        cache=_tensor((8, 64, 132), "uint8"),
        weights=_tensor((4, 32), "fp32"),
        metadata=SimpleNamespace(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=2,
            num_decode_tokens=4,
            prefill=None,
            decode=SimpleNamespace(
                requires_padding=False,
                seq_lens=_tensor((2, 2), "int32", values=(0, 1, 65, 66)),
                block_table=_tensor((2, 3), "int32"),
            ),
        ),
        candidates=_tensor((4, 16), "int32"),
        candidate_block_size=8,
        output=_tensor((4, 512), "int32"),
        max_model_len=4096,
        use_fp4_cache=False,
        candidate_write=False,
    )
    return SimpleNamespace(
        run=namespace["try_candidate_indexer"], inputs=inputs, score=score
    )


def _assert_score_call(
    call,
    inputs: dict,
    start: int,
    stop: int,
    table: _FakeTensor,
    requests: list[int],
    lengths: list[int],
    dense_width: int | None = None,
) -> None:
    names = (
        "query",
        "cache",
        "weights",
        "block_table",
        "requests",
        "lengths",
        "candidates",
        "candidate_block_size",
        "output",
        "dense_width",
        "candidate_write",
        "candidates_sorted_unique",
    )
    arguments = dict(zip(names, call.args))
    arguments.update(call.kwargs)
    request_repeat = arguments.pop("request_repeat", 1)
    identity_when_full = arguments.pop("candidates_identity_when_full", False)
    assert set(arguments) == set(names)
    scored = SimpleNamespace(**arguments)
    for name in ("query", "weights", "candidates", "output"):
        view, source = getattr(scored, name), inputs[name]
        assert view.base is source.base
        expected_offset = source.storage_offset() + start * source.stride(0)
        assert view.storage_offset() == expected_offset
        assert view.shape == (stop - start, *source.shape[1:])
        assert view.stride() == source.stride()
        assert view.dtype == source.dtype
        assert view.device == source.device
    assert scored.cache is inputs["cache"]
    assert scored.block_table is table
    assert identity_when_full == (scored.requests is None)
    if scored.requests is None:
        assert [row // request_repeat for row in range(stop - start)] == requests
    else:
        assert scored.requests.tolist() == requests
        assert scored.requests.dtype == "int32"
        assert scored.requests.device == inputs["query"].device
    assert scored.lengths.tolist() == lengths
    assert scored.lengths.dtype == "int32"
    assert scored.lengths.device == inputs["query"].device
    assert scored.candidate_block_size == inputs["candidate_block_size"]
    assert scored.candidate_write == inputs["candidate_write"]
    assert scored.candidates_sorted_unique is True
    if dense_width is not None:
        assert scored.dense_width == dense_width


def test_fake_tensor_preserves_view_layout_and_metadata_values() -> None:
    tensor = _tensor((6, 32), "fp32", strides=(80, 2))
    view = tensor[2:5]
    assert (view.shape, view.stride(), view.storage_offset()) == ((3, 32), (80, 2), 160)
    assert view.base is tensor and view.device == "musa" and view.dtype == "fp32"
    assert not view.is_contiguous()
    assert view.contiguous().stride() == (32, 1)
    values = _tensor((2, 2), "int32", values=(1, 2, 3, 4))
    assert values.reshape(-1)[1:3].tolist() == [2, 3]


@pytest.mark.parametrize("candidate_write", [False, True])
@pytest.mark.parametrize("implicit_requests", [False, True])
@pytest.mark.parametrize("identity_when_full", [False, True])
def test_scoring_tiles_preserve_decode_request_row_offset(
    candidate_write, implicit_requests, identity_when_full
) -> None:
    namespace = _host_functions()
    launch = Mock()
    namespace["_score_and_select_rows"] = launch
    dense_width = 262144
    width = dense_width if candidate_write else 2048 * 8
    tile_rows = (256 << 20) // (width * 4)
    rows = tile_rows + 3
    requests = None if implicit_requests else _tensor((rows,), "int32")
    namespace["_score_and_select"](
        _tensor((rows, 32, 128), "fp8"),
        _tensor((8, 64, 132), "uint8"),
        _tensor((rows, 32), "fp32"),
        _tensor((rows, 4), "int32"),
        requests,
        _tensor((rows,), "int32"),
        _tensor((rows, 2048), "int32"),
        8,
        _tensor((rows, 512), "int32"),
        dense_width,
        candidate_write,
        True,
        2,
        identity_when_full,
    )
    assert launch.call_count == 2
    for call, start in zip(launch.call_args_list, (0, tile_rows)):
        assert call.args[-3:] == (2, start, identity_when_full)
        if implicit_requests:
            assert call.args[4] is None
        else:
            assert call.args[4].storage_offset() == start


@pytest.mark.parametrize(
    ("candidates", "block_size", "dense_width", "expected"),
    [(5, 8, 1024, 40), (5, 16, 80, 0), (16, 16, 128, 0)],
)
def test_sparse_scoring_only_reduces_a_larger_dense_window(
    candidates: int,
    block_size: int,
    dense_width: int,
    expected: int,
) -> None:
    assert (
        _host_functions()["_sparse_width"](candidates, block_size, dense_width)
        == expected
    )


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        pytest.param("use_fp4_cache", True, id="fp4"),
        pytest.param("query", _tensor((4, 32, 128), "bf16"), id="query_dtype"),
        pytest.param("query", _tensor((4, 128), "fp8"), id="query_rank"),
        pytest.param("query", _tensor((4, 32, 64), "fp8"), id="head_dim"),
        pytest.param("query", _tensor((4, 16, 128), "fp8"), id="query_heads"),
        pytest.param(
            "query",
            _tensor((4, 32, 128), "fp8", strides=(8192, 256, 2)),
            id="query_inner_stride",
        ),
        pytest.param("cache", _tensor((8, 64, 132), "fp32"), id="cache_dtype"),
        pytest.param("cache", _tensor((8, 132), "uint8"), id="cache_rank"),
        pytest.param("cache", _tensor((8, 64, 136), "uint8"), id="cache_layout"),
        pytest.param(
            "cache",
            _tensor((8, 64, 132), "uint8", strides=(16896, 264, 2)),
            id="cache_element_stride",
        ),
        pytest.param(
            "cache",
            _tensor((8, 64, 132), "uint8", strides=(8512, 133, 1)),
            id="cache_token_stride",
        ),
        pytest.param("weights", _tensor((4, 32), "bf16"), id="weights_dtype"),
        pytest.param("weights", _tensor((128,), "fp32"), id="weights_rank_one"),
        pytest.param("weights", _tensor((4, 32, 1), "fp32"), id="weights_rank_three"),
        pytest.param("weights", _tensor((4, 31), "fp32"), id="weights_fewer_heads"),
        pytest.param("weights", _tensor((4, 33), "fp32"), id="weights_extra_heads"),
        pytest.param("weights", _tensor((3, 32), "fp32"), id="weights_row_capacity"),
        pytest.param("output", _tensor((4, 512), "fp32"), id="output_float_dtype"),
        pytest.param("output", _tensor((4, 512), "uint8"), id="output_uint8_dtype"),
        pytest.param("output", _tensor((), "int32"), id="output_rank_zero"),
        pytest.param("output", _tensor((512,), "int32"), id="output_rank_one"),
        pytest.param("output", _tensor((4, 512, 1), "int32"), id="output_rank_three"),
        pytest.param("output", _tensor((3, 512), "int32"), id="output_row_capacity"),
        pytest.param("output", _tensor((4, 0), "int32"), id="topk_zero"),
        pytest.param("output", _tensor((4, 513), "int32"), id="topk_oversized"),
        pytest.param(
            "output",
            _tensor((4, 512), "int32", strides=(1024, 2)),
            id="output_inner_stride",
        ),
        pytest.param("candidates", _tensor((4, 16), "fp32"), id="candidate_dtype"),
        pytest.param("candidates", _tensor((64,), "int32"), id="candidate_rank_one"),
        pytest.param(
            "candidates", _tensor((4, 16, 1), "int32"), id="candidate_rank_three"
        ),
        pytest.param(
            "candidates",
            _tensor((4, 16), "int32", strides=(32, 2)),
            id="candidate_inner_stride",
        ),
        pytest.param(
            "candidates",
            _tensor((4, 16), "int32", strides=(32, 1)),
            id="candidate_row_stride",
        ),
        pytest.param(
            "candidates", _tensor((3, 16), "int32"), id="candidate_fewer_rows"
        ),
        pytest.param(
            "candidates", _tensor((4, 0), "int32"), id="candidate_empty_width"
        ),
        pytest.param(
            "candidates", _tensor((4, 2049), "int32"), id="candidate_oversized"
        ),
        pytest.param("candidate_block_size", 32, id="block_size"),
    ],
)
def test_unsupported_layout_declines_before_launching(
    candidate_call: SimpleNamespace, argument: str, value
) -> None:
    candidate_call.inputs[argument] = value
    assert candidate_call.run(**candidate_call.inputs) is False
    candidate_call.score.assert_not_called()


def test_decode_padding_declines_before_any_prefill_or_decode_call(
    candidate_call: SimpleNamespace,
) -> None:
    metadata = candidate_call.inputs["metadata"]
    metadata.num_decodes = 1
    metadata.num_decode_tokens = 2
    metadata.num_prefills = 1
    metadata.num_prefill_tokens = 2
    metadata.decode.requires_padding = True
    metadata.decode.seq_lens = _tensor((1, 2), "int32", values=(65, 66))
    metadata.decode.block_table = _tensor((1, 3), "int32")
    metadata.prefill = SimpleNamespace(
        chunks=[_prefill_chunk(2, 4, (0, 8), (0, 0), (7, 8))]
    )
    assert candidate_call.run(**candidate_call.inputs) is False
    candidate_call.score.assert_not_called()


@pytest.mark.parametrize("candidate_write", [False, True])
@pytest.mark.parametrize(("topk", "output_dtype"), [(1, "int32"), (512, "int64")])
@pytest.mark.parametrize("decode_shape", [(4,), (2, 2)])
def test_supported_decode_slices_active_rows_and_maps_requests(
    candidate_call: SimpleNamespace,
    candidate_write: bool,
    topk: int,
    output_dtype: str,
    decode_shape: tuple[int, ...],
) -> None:
    inputs = candidate_call.inputs
    inputs.update(
        query=_tensor((6, 32, 128), "fp8"),
        weights=_tensor((8, 32), "fp32"),
        candidates=_tensor((8, 16), "int64"),
        output=_tensor((8, topk), output_dtype, strides=(1024, 1)),
        max_model_len=73,
        candidate_write=candidate_write,
    )
    metadata = inputs["metadata"]
    metadata.num_decodes = decode_shape[0]
    metadata.decode.seq_lens = _tensor(decode_shape, "int32", values=(0, 1, 65, 66))
    metadata.decode.block_table = _tensor((decode_shape[0], 3), "int32")
    assert candidate_call.run(**inputs) is True
    assert candidate_call.score.call_count == 1
    _assert_score_call(
        candidate_call.score.call_args,
        inputs,
        0,
        4,
        metadata.decode.block_table,
        [0, 1, 2, 3] if len(decode_shape) == 1 else [0, 0, 1, 1],
        [0, 1, 65, 66],
        73,
    )


@pytest.mark.parametrize("candidate_write", [False, True])
@pytest.mark.parametrize("block_size", [8, 16])
def test_supported_packed_prefill_and_decode_keep_chunk_local_slices(
    candidate_call: SimpleNamespace, candidate_write: bool, block_size: int
) -> None:
    inputs = candidate_call.inputs
    inputs.update(
        query=_tensor((9, 32, 128), "fp8"),
        weights=_tensor((9, 32), "fp32"),
        candidates=_tensor((9, 16), "int32"),
        output=_tensor((12, 512), "int32", strides=(1024, 1)),
        cache=_tensor((8, 64, 132), "uint8", strides=(8580, 132, 1)),
        candidate_block_size=block_size,
        candidate_write=candidate_write,
    )
    first = _prefill_chunk(2, 6, (0, 3, 8), (0, 0, 3, 3), (0, 3, 4, 8))
    second = _prefill_chunk(
        6, 9, (0, 128, 256), (0, 128, 128), (128, 129, 256), table_columns=2
    )
    metadata = inputs["metadata"]
    metadata.num_prefills = 4
    metadata.num_prefill_tokens = 7
    metadata.prefill = SimpleNamespace(chunks=[first, second])
    metadata.num_decode_tokens = 2
    metadata.decode.seq_lens = _tensor((2,), "int32", values=(41, 77))
    assert candidate_call.run(**inputs) is True
    assert candidate_call.score.call_count == 3
    calls = {
        call.args[0].storage_offset(): call
        for call in candidate_call.score.call_args_list
    }
    _assert_score_call(
        calls[2 * inputs["query"].stride(0)],
        inputs,
        2,
        6,
        first.block_table,
        [0, 0, 1, 1],
        [0, 3, 1, 5],
        8,
    )
    _assert_score_call(
        calls[6 * inputs["query"].stride(0)],
        inputs,
        6,
        9,
        second.block_table,
        [0, 1, 1],
        [128, 1, 128],
        128,
    )
    _assert_score_call(
        calls[0], inputs, 0, 2, metadata.decode.block_table, [0, 1], [41, 77], 192
    )


@pytest.mark.parametrize(
    ("global_cumulative", "local_cumulative", "requests"),
    [
        ((0, 120, 240), (0, 3, 8), [0, 0, 1, 1]),
        ((0, 120, 240, 480), (0, 0, 3, 8), [1, 1, 2, 2]),
    ],
)
def test_prefill_request_mapping_uses_local_cumulative_bounds(
    candidate_call: SimpleNamespace,
    global_cumulative: tuple[int, ...],
    local_cumulative: tuple[int, ...],
    requests: list[int],
) -> None:
    chunk = _prefill_chunk(
        0,
        4,
        global_cumulative,
        (0, 0, 3, 3),
        (0, 3, 4, 8),
        local_cumulative=local_cumulative,
    )
    metadata = candidate_call.inputs["metadata"]
    metadata.num_prefills = len(global_cumulative) - 1
    metadata.num_prefill_tokens = 4
    metadata.prefill = SimpleNamespace(chunks=[chunk])
    metadata.num_decodes = 0
    metadata.num_decode_tokens = 0
    metadata.decode = None
    assert candidate_call.run(**candidate_call.inputs) is True
    assert candidate_call.score.call_count == 1
    _assert_score_call(
        candidate_call.score.call_args,
        candidate_call.inputs,
        0,
        4,
        chunk.block_table,
        requests,
        [0, 3, 1, 5],
    )


@pytest.fixture
def musa():
    pytest.importorskip("torchada")
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    from vllm_musa.kernels import sparse_mqa

    return torch, sparse_mqa


def _paged_inputs(torch, block_size: int):
    generator = torch.Generator().manual_seed(20261113)
    keys = torch.randn((8, block_size, 128), generator=generator).to(
        torch.float8_e4m3fn
    )
    scales = torch.rand((8, block_size), generator=generator) + 0.25
    cache = torch.zeros((8, block_size, 132), dtype=torch.uint8)
    for page in range(8):
        raw = cache[page].view(-1)
        raw[: block_size * 128] = keys[page].view(torch.uint8).reshape(-1)
        raw[block_size * 128 :] = scales[page].view(torch.uint8).reshape(-1)
    query = torch.randn((4, 32, 128), generator=generator).to(torch.float8_e4m3fn)
    weights = torch.rand((4, 32), generator=generator)
    table = torch.tensor([[3, 1, 6, 0], [4, 2, 7, 5]], dtype=torch.int32)
    return keys, scales, query, weights, table, cache.to("musa")


def _pad_candidates_to_dense_window(torch, candidates, block_size, width):
    columns = max(candidates.shape[1], (width + block_size - 1) // block_size)
    padded = torch.full(
        (candidates.shape[0], columns),
        -1,
        dtype=candidates.dtype,
        device=candidates.device,
    )
    padded[:, : candidates.shape[1]].copy_(candidates)
    return padded


@pytest.mark.parametrize("cache_block", [64, 128])
@pytest.mark.parametrize("candidate_block", [8, 16])
@pytest.mark.parametrize("candidates_sorted_unique", [False, True])
def test_sparse_mqa_matches_fp32_scores_and_dense_mask(
    musa, cache_block, candidate_block, candidates_sorted_unique
):
    torch, module = musa
    keys, scales, query_cpu, weights_cpu, table_cpu, cache = _paged_inputs(
        torch, cache_block
    )
    query, weights, table = (
        query_cpu.to("musa"),
        weights_cpu.to("musa"),
        table_cpu.to("musa"),
    )
    requests_cpu = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    lengths_cpu = torch.tensor([0, 1, 79, 129], dtype=torch.int32)
    candidates_cpu = torch.tensor(
        [[0, 3, 7, 100, -1] if candidates_sorted_unique else [3, -1, 0, 3, 100]] * 4,
        dtype=torch.int32,
    )
    requests, lengths, candidates = (
        requests_cpu.to("musa"),
        lengths_cpu.to("musa"),
        candidates_cpu.to("musa"),
    )
    sparse = torch.full((4, 16), -99, dtype=torch.int32, device="musa")
    dense = torch.empty_like(sparse)
    args = (
        query,
        cache,
        weights,
        table,
        requests,
        lengths,
        candidates,
        candidate_block,
    )
    module._score_and_select(
        *args,
        sparse,
        cache_block * 4,
        False,
        candidates_sorted_unique=candidates_sorted_unique,
    )
    module._score_and_select(
        query,
        cache,
        weights,
        table,
        requests,
        lengths,
        _pad_candidates_to_dense_window(
            torch, candidates, candidate_block, cache_block * 4
        ),
        candidate_block,
        dense,
        cache_block * 4,
        False,
    )
    actual = sparse.cpu().tolist()
    dense_rows = dense.cpu().tolist()
    for row, length in enumerate(lengths_cpu.tolist()):
        tokens = sorted(
            {
                block * candidate_block + offset
                for block in candidates_cpu[row].tolist()
                if block >= 0
                for offset in range(candidate_block)
                if block * candidate_block + offset < length
            }
        )
        request = int(requests_cpu[row])
        cached = keys[table_cpu[request].long()].float().reshape(-1, 128)
        cached *= scales[table_cpu[request].long()].reshape(-1, 1)
        scores = (query_cpu[row].float() @ cached[tokens].T).clamp_min(0)
        scores = (scores * weights_cpu[row, :, None]).sum(0)
        order = torch.argsort(scores, descending=True, stable=True)[:16].tolist()
        expected = {tokens[column] for column in order}
        selected = [token for token in actual[row] if token >= 0]
        assert len(selected) == len(set(selected))
        assert set(selected) == expected
        assert set(actual[row]) == set(dense_rows[row])
        assert actual[row][len(selected) :] == [-1] * (16 - len(selected))


@pytest.mark.parametrize("num_candidates", [1, 13, 64, 512, 2048])
@pytest.mark.parametrize("dtype_name", ["int32", "int64"])
def test_candidate_compaction_cross_warp_and_strided_input(
    musa, num_candidates, dtype_name
):
    torch, module = musa
    rows = [
        [
            -1 if slot % 11 == 0 else 10000 if slot % 17 == 0 else slot % 67
            for slot in range(num_candidates)
        ],
        [slot % 7 for slot in range(num_candidates)],
    ]
    storage = torch.full(
        (2, num_candidates * 2),
        -1,
        device="musa",
        dtype=getattr(torch, dtype_name),
    )
    candidates = storage[:, ::2]
    candidates.copy_(torch.tensor(rows, device="musa", dtype=candidates.dtype))
    lengths = torch.tensor([1024, 17], device="musa", dtype=torch.int32)
    output = torch.empty((2, num_candidates), device="musa", dtype=torch.int32)
    ends = torch.empty((2,), device="musa", dtype=torch.int32)
    module._compact_candidate_blocks_kernel[(2,)](
        candidates,
        lengths,
        output,
        ends,
        *candidates.stride(),
        lengths.stride(0),
        num_candidates,
        8,
        module.triton.next_power_of_2(num_candidates),
    )
    expected = []
    expected_ends = []
    for values, length in zip(rows, [1024, 17]):
        compact = sorted({block for block in values if 0 <= block < (length + 7) // 8})
        expected.append(compact + [-1] * (num_candidates - len(compact)))
        expected_ends.append(len(compact) * 8)
    assert output.cpu().tolist() == expected
    assert ends.cpu().tolist() == expected_ends


@pytest.mark.parametrize("row_repeat", [1, 2])
def test_candidate_selection_respects_packed_bounds_and_pins_newest(musa, row_repeat):
    torch, module = musa
    logits = torch.arange(513, dtype=torch.float32).neg().repeat(2 * row_repeat, 1)
    starts = torch.tensor([17, 64], dtype=torch.int32)
    ends = torch.tensor([121, 64], dtype=torch.int32)
    for row in range(2 * row_repeat):
        logits[row, int(ends[row // row_repeat]) :] = float("nan")
    output = torch.full((2 * row_repeat, 3), -99, dtype=torch.int32, device="musa")
    module.select_candidate_blocks(
        logits.to("musa"), starts.to("musa"), ends.to("musa"), 16, output, row_repeat
    )
    for row, selected in enumerate(output.cpu().tolist()):
        assert set(selected) == ({0, 1, 6} if row // row_repeat == 0 else {-1})


@pytest.mark.parametrize("candidates_sorted_unique", [False, True])
@pytest.mark.parametrize("implicit_requests", [False, True])
def test_candidate_decode_graph_replays_updated_device_lengths(
    musa, candidates_sorted_unique, implicit_requests
):
    torch, module = musa
    _, _, query, weights, table, cache = _paged_inputs(torch, 64)
    query, weights, table = query.to("musa"), weights.to("musa"), table.to("musa")
    requests = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device="musa")
    lengths = torch.tensor([65, 67, 79, 129], dtype=torch.int32, device="musa")
    candidates = torch.tensor(
        [[0, 3, 7, 100, -1] if candidates_sorted_unique else [100, -1, 7, 0, 3]] * 4,
        dtype=torch.int32,
        device="musa",
    )
    output = torch.empty((4, 16), dtype=torch.int32, device="musa")
    args = (
        query,
        cache,
        weights,
        table,
        None if implicit_requests else requests,
        lengths,
        candidates,
        8,
        output,
        256,
        False,
        candidates_sorted_unique,
        2,
    )
    module._score_and_select(*args)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        module._score_and_select(*args)
    candidates.copy_(
        torch.tensor([[0, 1, -1, -1, -1]] * 4, dtype=torch.int32, device="musa")
    )
    for updated in ([0, 1, 31, 39], [129, 79, 0, 65]):
        lengths.copy_(torch.tensor(updated, dtype=torch.int32, device="musa"))
        graph.replay()
        captured = output.cpu()
        module._score_and_select(*args)
        assert torch.equal(captured, output.cpu())
        for length, selected in zip(updated, captured.tolist()):
            if length == 0:
                assert selected == [-1] * 16


@pytest.mark.parametrize("request_repeat", [1, 2])
@pytest.mark.parametrize("row_offset", [0, 1, 3])
@pytest.mark.parametrize("candidate_write", [False, True])
def test_decode_request_mapping_matches_explicit_rows(
    musa, request_repeat, row_offset, candidate_write
):
    torch, module = musa
    _, _, query, weights, table, cache = _paged_inputs(torch, 64)
    query, weights, table = (
        query.to("musa"),
        weights.to("musa"),
        table.repeat(2, 1).to("musa"),
    )
    lengths = torch.tensor([0, 1, 79, 129], dtype=torch.int32, device="musa")
    candidates = torch.tensor(
        [[0, 3, 7, 100, -1]] * 4, dtype=torch.int32, device="musa"
    )
    requests = torch.arange(4, dtype=torch.int32, device="musa") // request_repeat
    expected = torch.empty((4 - row_offset, 16), dtype=torch.int32, device="musa")
    actual = torch.empty_like(expected)
    args = query[row_offset:], cache, weights[row_offset:], table
    tail = lengths[row_offset:], candidates[row_offset:], 8
    module._score_and_select_rows(
        *args, requests[row_offset:], *tail, expected, 256, candidate_write, True
    )
    module._score_and_select_rows(
        *args,
        None,
        *tail,
        actual,
        256,
        candidate_write,
        True,
        request_repeat=request_repeat,
        request_row_offset=row_offset,
    )
    assert torch.equal(expected.cpu(), actual.cpu())


@pytest.mark.parametrize("strided", [False, True])
def test_candidate_producer_preserves_unique_selection_and_missing_blocks(
    musa, strided, monkeypatch
):
    torch, module = musa
    monkeypatch.setattr(module, "_compact_candidate_blocks_kernel", None)
    logits = torch.zeros((3, 65), device="musa", dtype=torch.float32)
    logits[1, 8:24] = -float("inf")
    lengths = torch.tensor([17, 32, 65], device="musa", dtype=torch.int32)
    storage = torch.full(
        (3, 8 if strided else 4), -99, device="musa", dtype=torch.int32
    )
    output = storage[:, ::2] if strided else storage
    module.select_candidate_blocks(logits, None, lengths, 8, output)
    actual = output.cpu().tolist()
    assert set(actual[0]) == {0, 1, 2, -1}
    assert set(actual[1]) == {0, 3, -1}
    for row in actual:
        selected = [block for block in row if block >= 0]
        assert len(selected) == len(set(selected))
        assert selected == sorted(selected)
        assert all(block >= -1 for block in row)
    assert 8 in actual[2]
    if strided:
        assert (storage[:, 1::2].cpu() == -99).all()


@pytest.mark.parametrize("dtype_name", ["int32", "int64"])
def test_remap_full_and_partial_rows_keep_prefix_and_padding(musa, dtype_name):
    torch, module = musa
    topk, block_size, num_candidates = 512, 8, 64
    dtype = getattr(torch, dtype_name)
    candidates = torch.arange(num_candidates - 1, -1, -1, dtype=dtype).repeat(4, 1)
    candidates[3, 7] = -1
    columns = torch.arange(topk, dtype=dtype).repeat(4, 1)
    columns[1, ::7] = -1
    logits = torch.zeros((4, topk), dtype=torch.float32)
    logits[1, 3::11] = -float("inf")
    logits[2] = -float("inf")
    storage = torch.full((4, topk + 16), -99, dtype=dtype, device="musa")
    output = storage[:, :topk]
    device_candidates, device_columns, device_logits = (
        candidates.to("musa"),
        columns.to("musa"),
        logits.to("musa"),
    )
    module._remap_candidate_topk_kernel[(4,)](
        device_candidates,
        device_logits,
        device_columns,
        output,
        device_columns.stride(0),
        output.stride(0),
        topk,
        num_candidates,
        block_size,
        topk,
        topk,
    )
    for row, actual in enumerate(output.cpu().tolist()):
        expected = []
        for column in columns[row].tolist():
            if column < 0 or logits[row, column] == -float("inf"):
                continue
            block = int(candidates[row, column // block_size])
            if block >= 0:
                expected.append(block * block_size + column % block_size)
        assert actual == expected + [-1] * (topk - len(expected))
    assert (storage[:, topk:].cpu() == -99).all()


@pytest.mark.parametrize("cache_block", [64, 128])
def test_identity_when_full_replays_short_long_and_holey_rows(musa, cache_block):
    torch, module = musa
    _, _, query, weights, table, cache = _paged_inputs(torch, cache_block)
    query, weights, table = query.to("musa"), weights.to("musa"), table.to("musa")
    lengths = torch.tensor([0, 63, 16384, 16385], dtype=torch.int32, device="musa")
    producer = torch.arange(32768, dtype=torch.float32).neg().repeat(4, 1)
    producer[:, 8:16] = -float("inf")
    producer = producer.to("musa")
    candidates = torch.empty((4, 2048), dtype=torch.int32, device="musa")
    expected = torch.empty((4, 16), dtype=torch.int32, device="musa")
    actual = torch.empty_like(expected)

    def launch():
        module.select_candidate_blocks(producer, None, lengths, 8, candidates)
        for output, identity in ((expected, False), (actual, True)):
            module._score_and_select_rows(
                query,
                cache,
                weights,
                table,
                None,
                lengths,
                candidates,
                8,
                output,
                32768,
                False,
                candidates_sorted_unique=True,
                request_repeat=2,
                candidates_identity_when_full=identity,
            )

    def check():
        for reference, selected in zip(expected.cpu().tolist(), actual.cpu().tolist()):
            assert set(reference) == set(selected)
            count = sum(index >= 0 for index in selected)
            assert all(index >= 0 for index in selected[:count])
            assert selected[count:] == [-1] * (len(selected) - count)

    launch()
    check()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        launch()
    for updated in ([129, 16385, 0, 16383], [16385, 0, 16384, 63]):
        lengths.copy_(torch.tensor(updated, dtype=torch.int32))
        graph.replay()
        check()


def test_candidate_producer_2048_replays_across_budget(musa):
    torch, module = musa
    width, topk, block_size = 32768, 2048, 8
    logits = torch.arange(width, dtype=torch.float32).neg().repeat(4, 1)
    logits[:, 8:16] = -float("inf")
    logits = logits.to("musa")
    lengths = torch.tensor([16383, 16384, 16385, 17408], dtype=torch.int32)
    device_lengths = lengths.to("musa")
    output = torch.full((4, topk), -99, dtype=torch.int32, device="musa")

    def launch():
        module.select_candidate_blocks(logits, None, device_lengths, block_size, output)

    def check(expected_lengths):
        for length, selected in zip(expected_lengths, output.cpu().tolist()):
            count = (length + block_size - 1) // block_size
            available = [block for block in range(count) if block != 1]
            expected = available[:topk]
            if count and count - 1 not in expected:
                expected[-1] = count - 1
            actual = [block for block in selected if block >= 0]
            assert len(actual) == len(set(actual))
            assert actual == sorted(actual)
            assert set(actual) == set(expected)
            assert selected.count(-1) == topk - len(expected)

    launch()
    check(lengths.tolist())
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        launch()
    for updated in ([17408, 16385, 0, 1], [16384, 16383, 16385, 17408]):
        device_lengths.copy_(torch.tensor(updated, dtype=torch.int32))
        graph.replay()
        check(updated)


def test_sorted_unique_candidates_do_not_sort(musa, monkeypatch):
    torch, module = musa
    _, _, query, weights, table, cache = _paged_inputs(torch, 64)
    query, weights, table = query.to("musa"), weights.to("musa"), table.to("musa")
    requests = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device="musa")
    lengths = torch.tensor([0, 1, 79, 129], dtype=torch.int32, device="musa")
    candidates = torch.tensor(
        [[0, 3, 7, 100, -1]] * 4, dtype=torch.int32, device="musa"
    )
    expected = torch.empty((4, 16), dtype=torch.int32, device="musa")
    actual = torch.empty_like(expected)
    args = query, cache, weights, table, requests, lengths, candidates, 8
    module._score_and_select(*args, expected, 256, False)
    monkeypatch.setattr(module, "_compact_candidate_blocks_kernel", None)
    module._score_and_select(*args, actual, 256, False, candidates_sorted_unique=True)
    for reference, selected in zip(expected.cpu().tolist(), actual.cpu().tolist()):
        assert set(reference) == set(selected)
        count = sum(index >= 0 for index in selected)
        assert all(index >= 0 for index in selected[:count])
        assert selected[count:] == [-1] * (len(selected) - count)


def test_producer_and_consumers_replay_across_full_coverage_boundary(musa):
    torch, module = musa
    _, _, query, weights, table, cache = _paged_inputs(torch, 64)
    query, weights, table = query.to("musa"), weights.to("musa"), table.to("musa")
    requests = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device="musa")
    lengths = torch.tensor([17, 32, 65, 129], dtype=torch.int32, device="musa")
    source_logits = torch.arange(256, device="musa", dtype=torch.float32).repeat(4, 1)
    candidates = torch.empty((4, 4), dtype=torch.int32, device="musa")
    output = torch.empty((4, 16), dtype=torch.int32, device="musa")

    def run():
        module.select_candidate_blocks(source_logits, None, lengths, 8, candidates)
        module._score_and_select(
            query,
            cache,
            weights,
            table,
            requests,
            lengths,
            candidates,
            8,
            output,
            256,
            False,
            candidates_sorted_unique=True,
        )

    run()
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        run()
    for updated in ([129, 65, 32, 0], [0, 17, 129, 65]):
        lengths.copy_(torch.tensor(updated, dtype=torch.int32, device="musa"))
        graph.replay()
        actual = output.cpu()
        actual_candidates = candidates.cpu()
        run()
        for captured, eager in zip(
            actual_candidates.tolist(), candidates.cpu().tolist()
        ):
            assert set(captured) == set(eager)
        for captured, eager in zip(actual.tolist(), output.cpu().tolist()):
            assert set(captured) == set(eager)
            valid = [index for index in captured if index >= 0]
            assert len(valid) == len(set(valid))
            assert captured[len(valid) :] == [-1] * (16 - len(valid))


def _assert_tied_candidate_selection(
    output, lengths, candidates, block_size: int, dense_width: int
) -> None:
    topk = output.shape[1]
    for actual, length, blocks in zip(output.cpu().tolist(), lengths, candidates):
        expected = sorted(
            {
                block * block_size + offset
                for block in blocks
                if block >= 0
                for offset in range(block_size)
                if block * block_size + offset < min(length, dense_width)
            }
        )[:topk]
        assert len(set(actual[: len(expected)])) == len(expected)
        assert set(actual[: len(expected)]) == set(expected)
        assert actual[len(expected) :] == [-1] * (topk - len(expected))


@pytest.mark.parametrize("dtype_name", ["int32", "int64"])
@pytest.mark.parametrize("topk", [4, 16, 512])
@pytest.mark.parametrize("candidate_capacity", [4, 32])
@pytest.mark.parametrize("dense_width", [32, 256])
def test_masked_topk_pads_empty_partial_and_insufficient_rows(
    musa, dtype_name, topk, candidate_capacity, dense_width
):
    torch, module = musa
    _, _, query, weights, table, cache = _paged_inputs(torch, 64)
    query, table = query.to("musa"), table.to("musa")
    weights = torch.zeros_like(weights, device="musa")
    host_lengths = [79, 1, 0, 29]
    host_candidates = [[0, -1, -1, -1], [-1] * 4, [7, -1, -1, -1], [3, -1, -1, -1]]
    host_candidates = [
        row + [-1] * (candidate_capacity - len(row)) for row in host_candidates
    ]
    lengths = torch.tensor(host_lengths, dtype=torch.int32, device="musa")
    candidates = torch.tensor(host_candidates, dtype=torch.int32, device="musa")
    dtype = getattr(torch, dtype_name)
    storage = torch.full((4, topk + 8), -99, dtype=dtype, device="musa")
    output = storage[:, :topk]
    module._score_and_select(
        query,
        cache,
        weights,
        table,
        None,
        lengths,
        candidates,
        8,
        output,
        dense_width,
        False,
        request_repeat=2,
    )
    _assert_tied_candidate_selection(
        output, host_lengths, host_candidates, 8, dense_width
    )
    assert (storage[:, topk:].cpu() == -99).all()


@pytest.mark.parametrize("dtype_name", ["int32", "int64"])
def test_dense_masked_topk_graph_replay_updates_padding(musa, dtype_name):
    torch, module = musa
    _, _, query, weights, table, cache = _paged_inputs(torch, 64)
    query, table = query.to("musa"), table.to("musa")
    weights = torch.zeros_like(weights, device="musa")
    lengths = torch.tensor([79, 1, 0, 29], dtype=torch.int32, device="musa")
    candidates = torch.tensor([[0] + [-1] * 31] * 4, dtype=torch.int32, device="musa")
    output = torch.empty((4, 16), dtype=getattr(torch, dtype_name), device="musa")

    def run():
        module._score_and_select(
            query,
            cache,
            weights,
            table,
            None,
            lengths,
            candidates,
            8,
            output,
            256,
            False,
            request_repeat=2,
        )

    run()
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        run()
    for updated_lengths, updated_candidates in (
        ([0, 1, 17, 129], [[0], [0], [2], [15]]),
        ([79, 29, 1, 0], [[0, 3], [3], [-1], [7]]),
    ):
        padded = [row + [-1] * (32 - len(row)) for row in updated_candidates]
        lengths.copy_(torch.tensor(updated_lengths, dtype=torch.int32))
        candidates.copy_(torch.tensor(padded, dtype=torch.int32))
        graph.replay()
        _assert_tied_candidate_selection(output, updated_lengths, padded, 8, 256)
        captured = output.cpu()
        run()
        assert torch.equal(captured, output.cpu())


@pytest.mark.parametrize("cache_block", [64, 128])
@pytest.mark.parametrize("dtype_name", ["int32", "int64"])
def test_producer_canonical_order_preserves_tied_consumer_topk(
    musa, monkeypatch, cache_block, dtype_name
):
    torch, module = musa
    _, _, query, weights, table, cache = _paged_inputs(torch, cache_block)
    width, block_size = 2056, 8
    query = query.to("musa")
    weights = torch.zeros_like(weights, device="musa")
    table = table[:, :1].repeat(1, (width + cache_block - 1) // cache_block).to("musa")
    lengths = torch.full((4,), width, dtype=torch.int32, device="musa")
    source = torch.zeros((4, width), dtype=torch.float32, device="musa")
    source[:, block_size : 2 * block_size] = 1.0
    storage = torch.full((4, 4), -99, dtype=getattr(torch, dtype_name), device="musa")
    candidates = storage[:, ::2]
    sparse = torch.empty((4, 4), dtype=getattr(torch, dtype_name), device="musa")
    dense = torch.empty_like(sparse)

    def consume(output, candidate_rows):
        module._score_and_select(
            query,
            cache,
            weights,
            table,
            None,
            lengths,
            candidate_rows,
            block_size,
            output,
            width,
            False,
            candidates_sorted_unique=True,
            request_repeat=2,
            candidates_identity_when_full=True,
        )

    unsorted = torch.tensor([[256, 1]] * 4, dtype=candidates.dtype, device="musa")
    module._score_and_select(
        query,
        cache,
        weights,
        table,
        None,
        lengths,
        unsorted,
        block_size,
        sparse,
        width,
        False,
        request_repeat=2,
    )
    assert sparse.cpu().tolist() == [[8, 9, 10, 11]] * 4
    monkeypatch.setattr(module, "_compact_candidate_blocks_kernel", None)

    def run():
        module.select_candidate_blocks(source, None, lengths, block_size, candidates)
        consume(sparse, candidates)

    def check(low_block):
        assert candidates.cpu().tolist() == [[low_block, 256]] * 4
        assert (storage[:, 1::2].cpu() == -99).all()
        first = low_block * block_size
        assert sparse.cpu().tolist() == [list(range(first, first + 4))] * 4
        consume(
            dense, _pad_candidates_to_dense_window(torch, candidates, block_size, width)
        )
        assert torch.equal(sparse.cpu(), dense.cpu())

    run()
    check(1)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        run()
    for low_block in (5, 1):
        source.zero_()
        source[:, low_block * block_size : (low_block + 1) * block_size] = 1.0
        graph.replay()
        check(low_block)
