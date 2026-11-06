import ast
import itertools
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "vllm_musa/patches/series/upstream"


@pytest.fixture
def patched_repo():
    repo = Path(os.environ.get("VLLM_PCP_TEST_REPO", ROOT / "third_party/vllm"))
    if not (repo / "vllm/models/deepseek_v41/attention.py").is_file():
        pytest.skip("requires the pinned vLLM with the complete patch series")
    return repo


def _function(source, name):
    return next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _compile(nodes, namespace):
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    code = compile(ast.fix_missing_locations(module), "<source-contract>", "exec")
    exec(code, namespace)


@pytest.mark.parametrize(
    "entry,commit",
    [
        (1, "bfdbfa1632ef638db7d201f366c6f46cc26ca2b6"),
        (2, "d12d5460a5"),
        (3, "fa75d3456d60a5ffe923818c4b495295675b38e1"),
        (4, "fc47cc8fff"),
        (5, "85752a43c3"),
        (6, "c7476c7a6f"),
    ],
)
def test_native_pcp_backports_keep_upstream_provenance(entry, commit):
    patch = next(SERIES.glob(f"{entry:04d}-*.patch")).read_text()
    assert f"(cherry picked from commit {commit}" in patch
    assert "Subject: [PATCH] [" in patch


def test_worker_does_not_intercept_native_pcp():
    worker = (ROOT / "vllm_musa/worker.py").read_text()
    assert "DeepseekPCPModelRunner" not in worker
    assert "preallocate_deepseek_request_token_staging" in worker
    assert "model_runner = super()._make_model_runner()" in worker
    for filename in (
        "distributed/deepseek_pcp.py",
        "v1/worker/deepseek_pcp_model_runner.py",
        "v1/worker/deepseek_pcp_model_state.py",
    ):
        assert not (ROOT / "vllm_musa" / filename).exists()


@pytest.mark.parametrize("use_v2", [False, True])
def test_worker_keeps_native_runner_and_only_preallocates_for_v2(monkeypatch, use_v2):
    make_runner = Mock(return_value=sentinel.runner)
    preallocate = Mock()
    monkeypatch.setitem(
        sys.modules,
        "vllm_musa.v1.worker.request_token_staging",
        SimpleNamespace(preallocate_deepseek_request_token_staging=preallocate),
    )
    worker_source = (ROOT / "vllm_musa/worker.py").read_text()
    worker = ast.ClassDef(
        name="Worker",
        bases=[ast.Name(id="BaseWorker", ctx=ast.Load())],
        keywords=[],
        body=[_function(worker_source, "_make_model_runner")],
        decorator_list=[],
    )
    base = type("BaseWorker", (), {"_make_model_runner": make_runner})
    namespace = {"BaseWorker": base}
    _compile([worker], namespace)
    instance = namespace["Worker"]()
    instance.use_v2_model_runner = use_v2
    instance.vllm_config = sentinel.config

    assert instance._make_model_runner() is sentinel.runner
    make_runner.assert_called_once_with()
    if use_v2:
        preallocate.assert_called_once_with(sentinel.config, sentinel.runner)
    else:
        preallocate.assert_not_called()


def test_kv_only_native_binding_reuses_the_existing_packing_kernel():
    bindings = (ROOT / "csrc/musa/torch_bindings.cpp").read_text()
    header = (ROOT / "csrc/musa/musa_ops.h").read_text()
    source = (ROOT / "csrc/musa/attention/deepseek_v4_cache_store.mu").read_text()
    assert '"deepseek_v4_kv_rope_insert(Tensor kv, Tensor! kv_cache, "' in bindings
    assert "&deepseek_v4_kv_rope_insert" in bindings
    assert "void deepseek_v4_kv_rope_insert(" in header
    implementation = source.split("void deepseek_v4_kv_rope_insert(")[1].split(
        "torch::Tensor deepseek_v4_qnorm_rope_kv_insert(", 1
    )[0]
    assert "deepseek_v4_kv_rope_pack_kernel<<<" in implementation
    assert "torch::empty(" not in implementation
    assert "slot_mapping.numel() == 0" in implementation


@pytest.mark.parametrize(
    "path",
    [
        "tests/models/test_deepseek_v41_replay_start.py",
        "tests/models/test_deepseek_v41_replay_batch.py",
        "tests/v1/worker/test_gpu_pcp_manager.py",
    ],
)
def test_upstream_replay_and_padding_tests_do_not_skip_musa(patched_repo, path):
    source = (patched_repo / path).read_text()
    assert "current_platform.is_musa()" in source
    assert "torch.cuda.is_available()" not in source
    assert "torch.device(current_platform.device_type" in source


def test_native_cache_publication_and_local_query_slots(patched_repo):
    source = (patched_repo / "vllm/models/deepseek_v41/attention.py").read_text()
    helper = ast.unparse(_function(source, "_musa_deepseek_v4_kv_rope_insert"))
    assert "deepseek_v4_kv_rope_insert(" in helper
    assert "zeros(" not in helper
    assert "qnorm_rope_kv_insert(" not in helper
    method = ast.unparse(_function(source, "_fused_qnorm_rope_kv_insert"))
    assert "slot_mapping = slot_mapping[:0]" in method
    call = next(
        node
        for node in ast.walk(ast.parse(method))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "deepseek_v4_qnorm_rope_kv_insert"
    )
    assert ast.unparse(call.args[3]) == "slot_mapping"
    assert "prepare_deepseek_pcp_attention" not in source
    kv_forward = ast.unparse(_function(source, "forward_kv"))
    assert (
        "_musa_deepseek_v4_kv_rope_insert("
        "self, cache_kv, slot_mapping, cache_positions)"
    ) in kv_forward


def test_native_compressor_exchanges_boundary_rows_not_full_scores(patched_repo):
    source = (patched_repo / "vllm/models/deepseek_v41/compressor.py").read_text()
    assert "prev_row_indices" in source
    assert "ring_write_slots" in source
    assert "save_ring_rows(prev_rows, ring_write_slots, state_cache)" in source
    attention = (patched_repo / "vllm/models/deepseek_v41/attention.py").read_text()
    assert "all_gather(latent, dim=0)" in attention
    assert "context.gather(kv_score)" not in attention


def test_native_lookback_masks_empty_chunks_and_promotes_addresses(patched_repo):
    source = (
        patched_repo / "vllm/models/deepseek_v41/nvidia/model_state.py"
    ).read_text()
    method = ast.unparse(_function(source, "_gather_lookback_kernel"))
    assert "in_batch = in_batch & (end > begin)" in method
    assert "req_state_idx.to(tl.int64) * all_token_ids_stride" in method
    assert "MUSA DeepSeek PCP does not support decoder replay" not in source
    assert "_pcp_replay_start" in source


@pytest.mark.parametrize("num_slots", [0, 1, 3, 4096])
def test_kv_only_launch_has_no_query_programs(num_slots):
    source = (
        ROOT / "vllm_musa/kernels/deepseek_v4_rope_kv_insert.py"
    ).read_text()
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))

            return launch

    namespace = {
        "_dsv4_rope_kv_insert_kernel": Kernel(),
        "_NOPE_DIM": 448,
        "_ROPE_DIM": 64,
        "_QUANT_GROUP": 64,
        "_TOKEN_BYTES": 576,
        "_SCALE_BYTES": 8,
    }
    _compile([_function(source, "kv_rope_insert")], namespace)
    tensor = SimpleNamespace(shape=(80, 64, 584), stride=lambda axis: 64 * 584)
    slots = SimpleNamespace(numel=lambda: num_slots)
    namespace["kv_rope_insert"](tensor, tensor, slots, tensor, tensor, 64)
    if not num_slots:
        assert not calls
        return
    grid, args, keywords = calls[0]
    assert grid == (num_slots, 1)
    assert args[0] is args[1] is args[2] is tensor
    assert keywords["KV_ONLY"] is True
    assert keywords["IN_HEADS"] == keywords["OUT_HEADS"] == 0
    assert "if KV_ONLY or head_block == KV_BLOCK:" in source


def _tensor_metadata(shape, dtype, strides=None, storage_bytes=None, offset=0):
    if strides is None:
        strides = tuple(math.prod(shape[axis + 1 :]) for axis in range(len(shape)))
    return SimpleNamespace(
        shape=shape,
        dtype=dtype,
        device="musa",
        dim=lambda: len(shape),
        stride=lambda axis: strides[axis],
        numel=lambda: math.prod(shape),
        is_contiguous=lambda: True,
        storage_offset=lambda: offset,
        untyped_storage=lambda: SimpleNamespace(nbytes=lambda: storage_bytes),
    )


@pytest.mark.parametrize(
    "shape,strides,storage_bytes,offset,block_size,expected",
    [
        ((1, 64, 1), (37376, 1, 1), 64, 0, 64, False),
        ((1, 64, 584), (37376, 584, 1), 37376, 0, 64, True),
        ((2, 64, 584), (74752, 584, 1), 112128, 0, 64, True),
        ((2, 64, 584), (74752, 584, 1), 112160, 32, 64, True),
        ((2, 64, 1), (74752, 1, 1), 112160, 33, 64, False),
        ((1, 64, 0), (37376, 1, 1), 37376, 0, 64, False),
        ((0, 64, 584), (37376, 584, 1), 0, 0, 64, True),
        ((1, 0, 584), (37376, 584, 1), 0, 0, 0, False),
    ],
)
def test_cache_layout_checks_storage_span_without_device_access(
    shape, strides, storage_bytes, offset, block_size, expected
):
    source = (ROOT / "vllm_musa/kernels/deepseek_v4_rope_kv_insert.py").read_text()
    dtypes = SimpleNamespace(
        int32=sentinel.int32,
        int64=sentinel.int64,
        bfloat16=sentinel.bfloat16,
        uint8=sentinel.uint8,
        float32=sentinel.float32,
    )
    namespace = {
        "torch": dtypes,
        "_ENABLED": True,
        "_HEAD_DIM": 512,
        "_ROPE_DIM": 64,
        "_TOKEN_BYTES": 576,
        "_SCALE_BYTES": 8,
    }
    _compile([_function(source, "rope_kv_insert_supported")], namespace)
    q = _tensor_metadata((1, 1, 512), dtypes.bfloat16)
    kv = _tensor_metadata((1, 512), dtypes.bfloat16)
    cache = _tensor_metadata(shape, dtypes.uint8, strides, storage_bytes, offset)
    indices = _tensor_metadata((1,), dtypes.int64)
    rope = _tensor_metadata((16, 64), dtypes.float32)

    assert (
        namespace["rope_kv_insert_supported"](
            q, kv, cache, indices, indices, rope, block_size, 0
        )
        is expected
    )


class _DenseTensor:
    def __init__(self, shape, values=None):
        self.shape = tuple(shape)
        self.values = tuple(values) if values is not None else tuple(self.indices())

    def indices(self):
        return itertools.product(*(range(size) for size in self.shape))

    def movedim(self, source, destination):
        axes = list(range(len(self.shape)))
        axis = axes.pop(source)
        axes.insert(destination % len(self.shape), axis)
        data = dict(zip(self.indices(), self.values))
        shape = tuple(self.shape[axis] for axis in axes)
        inverse = [axes.index(axis) for axis in range(len(axes))]
        values = [
            data[tuple(index[axis] for axis in inverse)]
            for index in itertools.product(*(range(size) for size in shape))
        ]
        return _DenseTensor(shape, values)

    def contiguous(self):
        return self


@pytest.mark.parametrize("shape", [(4, 6), (2, 4, 6), (2, 4, 6, 8)])
@pytest.mark.parametrize("missing_pynccl", [True, False])
def test_mccl_fallback_preserves_arbitrary_scatter_axes(
    patched_repo, shape, missing_pynccl
):
    source = (
        patched_repo / "vllm/distributed/device_communicators/cuda_communicator.py"
    ).read_text()

    class BaseCommunicator:
        def reduce_scatter(self, tensor, dim):
            assert dim == 0
            width = tensor.shape[0] // self.world_size
            values = [
                value
                for index, value in zip(tensor.indices(), tensor.values)
                if self.rank * width <= index[0] < (self.rank + 1) * width
            ]
            return _DenseTensor((width, *tensor.shape[1:]), values)

    class_node = ast.ClassDef(
        name="Communicator",
        bases=[ast.Name(id="BaseCommunicator", ctx=ast.Load())],
        keywords=[],
        body=[_function(source, "reduce_scatter")],
        decorator_list=[],
    )
    namespace = {
        "BaseCommunicator": BaseCommunicator,
        "current_platform": SimpleNamespace(is_musa=lambda: True),
    }
    _compile([class_node], namespace)
    communicator = namespace["Communicator"]()
    communicator.world_size = 2
    communicator.pynccl_comm = (
        None if missing_pynccl else SimpleNamespace(disabled=True)
    )
    tensor = _DenseTensor(shape)
    for rank in range(2):
        communicator.rank = rank
        for axis in range(-len(shape), len(shape)):
            output = communicator.reduce_scatter(tensor, axis)
            positive_axis = axis % len(shape)
            expected_shape = list(shape)
            expected_shape[positive_axis] //= 2
            assert output.shape == tuple(expected_shape)
            width = expected_shape[positive_axis]
            expected = tuple(
                value
                for value in tensor.values
                if rank * width <= value[positive_axis] < (rank + 1) * width
            )
            assert output.values == expected
