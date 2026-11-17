from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import sys
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, replace
from enum import Enum
from itertools import groupby, product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
VLLM = ROOT / "third_party/vllm/vllm"
SPEC = importlib.util.spec_from_file_location(
    "sparse_mqa_dispatch", ROOT / "vllm_musa/kernels/sparse_mqa_dispatch.py"
)
dispatch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dispatch)


def _load_contract_threshold():
    path = ROOT / "vllm_musa/optimization_contract/policy.py"
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "deepseek_v41_indexer_context_threshold"
    )
    namespace = {"Any": object}
    tree = ast.Module(body=[function], type_ignores=[])
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace[function.name]


deepseek_v41_indexer_context_threshold = _load_contract_threshold()


class _GraphMode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = 3

    def __bool__(self):
        return self != self.NONE

    def mixed_mode(self):
        return self.NONE if self == self.FULL_DECODE_ONLY else self

    def decode_mode(self):
        return self.FULL if self == self.FULL_DECODE_ONLY else self

    def separate_routine(self):
        return self == self.FULL_DECODE_ONLY

    def has_mode(self, mode):
        return mode in (self.mixed_mode(), self.decode_mode())

    def has_full_cudagraphs(self):
        return self.has_mode(self.FULL)

    @classmethod
    def valid_runtime_modes(cls):
        return {cls.NONE, cls.PIECEWISE, cls.FULL}


def _load_nodes(filename, names, namespace):
    tree = ast.parse(filename.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(filename), "exec"), namespace)


@pytest.fixture
def dispatcher():
    namespace = dict(
        dataclass=dataclass,
        replace=replace,
        product=product,
        groupby=groupby,
        defaultdict=defaultdict,
        CUDAGraphMode=_GraphMode,
        logger=Mock(),
        torch=SimpleNamespace(inference_mode=lambda: lambda function: function),
        round_up=lambda value, multiple: (value + multiple - 1) // multiple * multiple,
    )
    _load_nodes(VLLM / "forward_context.py", {"BatchDescriptor"}, namespace)
    _load_nodes(
        VLLM / "v1/worker/gpu/cudagraph_utils.py",
        {"BatchExecutionDescriptor", "_is_compatible", "CudaGraphManager"},
        namespace,
    )
    manager = namespace["CudaGraphManager"].__new__(namespace["CudaGraphManager"])
    manager.vllm_config = SimpleNamespace(
        speculative_config=None,
        cache_config=SimpleNamespace(use_kda_recoverssm=False),
    )
    manager.compilation_config = SimpleNamespace(
        max_cudagraph_capture_size=4, cudagraph_capture_sizes=[1, 2, 4]
    )
    manager.max_num_reqs = 4
    manager.decode_query_len = 1
    manager.cudagraph_mode = _GraphMode.FULL_DECODE_ONLY
    manager.varlen_decode = False
    manager.ubatch_runner = None
    manager.lora_capture_cases = [0]
    manager._lora_dispatch_map = {}
    manager.sparse_indexer_context_threshold = 32768
    manager._candidates = {}
    manager._capture_descs = {}
    manager._init_candidates()
    manager._graphs_captured = True
    return manager


@pytest.mark.parametrize(
    ("computed", "expected"),
    [
        ([], False),
        ([0], False),
        ([4095], False),
        ([8191], False),
        ([16383], False),
        ([32766], False),
        ([32767], True),
        ([32768], True),
        ([131071], True),
        ([32767, 65535], True),
        ([4095, 131071], True),
        ([131071, 4095], True),
    ],
)
def test_current_context_cutoff(computed, expected):
    assert dispatch.use_candidate_indexer(computed, 32768) is expected


@pytest.mark.parametrize("prompt_tokens", [4096, 8192, 16384])
def test_short_requests_use_the_original_indexer(prompt_tokens):
    for computed in range(0, prompt_tokens, 8192):
        scheduled = min(8192, prompt_tokens - computed)
        assert not dispatch.use_candidate_indexer([computed], 32768, [scheduled])
    for computed in range(prompt_tokens, prompt_tokens + 1024):
        assert not dispatch.use_candidate_indexer([computed], 32768, [1])


@pytest.mark.parametrize(
    ("computed", "scheduled", "expected"),
    [
        ([0], [32767], False),
        ([0], [32768], True),
        ([0], [65536], True),
        ([24575], [8192], False),
        ([24576], [8192], True),
        ([32767], [0], False),
        ([4095, 24576], [1, 8192], True),
        ([4095, 8191], [1, 1], False),
    ],
)
def test_prefill_last_scheduled_row_controls_mixed_chunk_cutoff(
    computed, scheduled, expected
):
    assert dispatch.use_candidate_indexer(computed, 32768, scheduled) is expected


def test_mismatched_host_lengths_fail_before_graph_selection():
    with pytest.raises(ValueError, match="equal length"):
        dispatch.use_candidate_indexer([0], 32768, [1, 1])


def test_contract_context_cutoff():
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type="deepseek_v41", candidate_topk_blocks=2048
            )
        ),
        speculative_config=None,
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(use_ubatching=False, data_parallel_size=1),
    )
    assert deepseek_v41_indexer_context_threshold(config) == 32768


def test_other_models_retain_unspecialized_routing():
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type="deepseek_v41", candidate_topk_blocks=2048
            )
        ),
        speculative_config=None,
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(use_ubatching=False, data_parallel_size=1),
    )
    config.model_config.hf_text_config.model_type = "deepseek_v4"
    assert deepseek_v41_indexer_context_threshold(config) == 0


@pytest.mark.parametrize(
    "unsupported", ["speculation", "v1", "no_candidates", "ubatching", "dp"]
)
def test_threshold_does_not_specialize_other_execution_paths(unsupported):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type="deepseek_v41", candidate_topk_blocks=2048
            )
        ),
        speculative_config=object() if unsupported == "speculation" else None,
        use_v2_model_runner=unsupported != "v1",
        parallel_config=SimpleNamespace(
            use_ubatching=unsupported == "ubatching",
            data_parallel_size=2 if unsupported == "dp" else 1,
        ),
    )
    if unsupported == "no_candidates":
        config.model_config.hf_text_config.candidate_topk_blocks = 0
    threshold = deepseek_v41_indexer_context_threshold(config)
    assert threshold == 0
    assert dispatch.use_candidate_indexer([4095], threshold)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("candidate_write", [False, True])
def test_short_context_uses_original_indexer_without_candidate_work(
    monkeypatch, enabled, candidate_write
):
    monkeypatch.setitem(
        sys.modules,
        "vllm_musa.kernels.sparse_mqa_dispatch",
        SimpleNamespace(candidate_path_enabled=lambda: enabled),
    )
    filename = VLLM / "model_executor/layers/sparse_attn_indexer.py"
    cls = next(
        node
        for node in ast.parse(filename.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "SparseAttnIndexer"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward_native"
    )
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    original = Mock(return_value="original")
    namespace = dict(
        current_platform=SimpleNamespace(
            is_musa=lambda: True,
            is_out_of_tree=lambda: True,
            is_cuda=lambda: False,
            is_xpu=lambda: False,
            is_rocm=lambda: False,
            is_cpu=lambda: False,
        ),
        _musa_fill_exact_sparse_indexer_indices=original,
    )
    exec(compile(ast.fix_missing_locations(tree), str(filename), "exec"), namespace)
    indexer = SimpleNamespace(
        candidate_blocks=object(),
        candidate_write=candidate_write,
        forward_cuda=Mock(return_value="candidate"),
        use_musa_native_indexer=True,
        k_cache=SimpleNamespace(prefix="layer", kv_cache=object()),
        topk_tokens=512,
        head_dim=128,
        max_model_len=262144,
        topk_indices_buffer=object(),
        use_fp4_cache=False,
        use_musa_materialized_prefill=True,
    )
    result = namespace["forward_native"](indexer, object(), object(), None, object())
    assert result == ("candidate" if enabled else "original")
    assert original.call_count == int(not enabled)
    assert indexer.forward_cuda.call_count == int(enabled)


@pytest.mark.parametrize("mode", [_GraphMode.FULL_DECODE_ONLY, _GraphMode.PIECEWISE])
def test_v2_capture_and_runtime_keys_remain_distinct(dispatcher, mode):
    dispatcher.cudagraph_mode = mode
    dispatcher._candidates.clear()
    dispatcher._capture_descs.clear()
    dispatcher._init_candidates()
    captures = [desc for descs in dispatcher._capture_descs.values() for desc in descs]
    assert len(captures) == 6
    for computed in ([4095], [32766], [32767], [131071], [4095, 131071], [8191]):
        base = dispatcher.dispatch(len(computed), len(computed), 1, 0)
        key = dispatch.select_candidate_graph(base, computed, 32768)
        assert key.cg_mode == mode.decode_mode()
        assert key in captures
        assert key.use_sparse_indexer_candidates is dispatch.use_candidate_indexer(
            computed, 32768
        )
        assert (
            replace(
                key, use_sparse_indexer_candidates=not key.use_sparse_indexer_candidates
            )
            != key
        )


def test_eager_and_disabled_cutoff(dispatcher):
    base = dispatcher.dispatch(5, 5, 1, 0)
    key = dispatch.select_candidate_graph(base, [4095] * 5, 32768)
    assert key.cg_mode == _GraphMode.NONE
    assert not key.use_sparse_indexer_candidates
    dispatcher.sparse_indexer_context_threshold = 0
    dispatcher._candidates.clear()
    dispatcher._capture_descs.clear()
    dispatcher._init_candidates()
    assert sum(map(len, dispatcher._capture_descs.values())) == 3


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "mode", [_GraphMode.NONE, _GraphMode.FULL, _GraphMode.PIECEWISE]
)
def test_v2_capture_passes_the_same_flag_to_forward_context(enabled, mode):
    filename = VLLM / "v1/worker/gpu/cudagraph_utils.py"
    forward_fn = next(
        node
        for node in ast.walk(ast.parse(filename.read_text()))
        if isinstance(node, ast.FunctionDef)
        and node.name == "forward_fn"
        and "use_sparse_indexer_candidates" in ast.unparse(node)
    )
    namespace = dict(
        dataclass=dataclass,
        CUDAGraphMode=_GraphMode,
        self=SimpleNamespace(
            sparse_indexer_context_threshold=32768,
            vllm_config=object(),
            run_pw_graph=Mock(),
        ),
        desc=SimpleNamespace(num_active_loras=0, use_sparse_indexer_candidates=enabled),
        num_tokens=1,
        has_lora=False,
        attn_metadata={},
        num_tokens_across_dp=None,
        slot_mappings={},
        input_buffers=SimpleNamespace(is_padding=[False]),
        model=Mock(),
        model_inputs={},
        store_capture_output=Mock(),
        set_forward_context=Mock(side_effect=lambda *args, **kwargs: nullcontext()),
    )
    _load_nodes(VLLM / "forward_context.py", {"BatchDescriptor"}, namespace)
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            forward_fn,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(filename), "exec"), namespace)
    namespace["forward_fn"](mode)
    context = namespace["set_forward_context"].call_args.kwargs
    assert context["batch_descriptor"].use_sparse_indexer_candidates is enabled
    if mode != _GraphMode.PIECEWISE:
        namespace["self"].sparse_indexer_context_threshold = 0
        namespace["forward_fn"](mode)
        assert (
            namespace["set_forward_context"].call_args.kwargs["batch_descriptor"]
            is None
        )


def test_producer_and_consumer_share_current_descriptor(monkeypatch, dispatcher):
    context = SimpleNamespace(batch_descriptor=None)
    monkeypatch.setitem(
        sys.modules,
        "vllm.forward_context",
        SimpleNamespace(get_forward_context=lambda: context),
    )
    assert dispatch.candidate_path_enabled()
    for computed in ([4095], [32767], [8191], [65535]):
        base = dispatcher.dispatch(1, 1, 1, 0)
        context.batch_descriptor = dispatch.select_candidate_graph(
            base, computed, 32768
        )
        assert dispatch.candidate_path_enabled() is dispatch.use_candidate_indexer(
            computed, 32768
        )


@pytest.mark.skipif(os.getenv("RUN_MUSA_TESTS") != "1", reason="MUSA-only test")
def test_musa_graph_replay_selects_current_threshold_bucket(dispatcher):
    importlib.import_module("torchada")
    torch = importlib.import_module("torch")
    source = torch.ones(1, device="musa")
    output = torch.empty_like(source)
    graphs = {}
    for computed in ([4095], [32767]):
        base = dispatcher.dispatch(1, 1, 1, 0)
        key = dispatch.select_candidate_graph(base, computed, 32768)
        output.copy_(source + int(key.use_sparse_indexer_candidates))
        torch.musa.synchronize()
        graph = torch.musa.MUSAGraph()
        with torch.musa.graph(graph):
            output.copy_(source + int(key.use_sparse_indexer_candidates))
        graphs[key] = graph
    for computed in ([4095], [32766], [32767], [32768], [131071], [8191]):
        base = dispatcher.dispatch(1, 1, 1, 0)
        key = dispatch.select_candidate_graph(base, computed, 32768)
        source.add_(1)
        graphs[key].replay()
        assert output.item() == source.item() + int(key.use_sparse_indexer_candidates)
