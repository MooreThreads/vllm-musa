from __future__ import annotations

import ast
import contextlib
import importlib.util
import sys
from enum import IntEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Callable
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "third_party/vllm/vllm/model_executor/layers/fused_moe/runner"


@pytest.fixture
def eligibility(monkeypatch):
    name = "vllm_musa.model_executor.layers.quantization.w4afp8"
    quantization = ModuleType(name)
    quantization.MusaW4AFp8MoEMethod = type("MusaW4AFp8MoEMethod", (), {})
    monkeypatch.setitem(sys.modules, name, quantization)
    optimization = ModuleType("vllm_musa.optimization_contract")
    optimization.OptimizationFeature = SimpleNamespace(
        DEEPSEEK_V41_W4AFP8_SHARED_EXPERT_OVERLAP=object()
    )
    optimization.prefers_feature = lambda config, feature: (
        config.contract_prefers_overlap
    )
    monkeypatch.setitem(sys.modules, "vllm_musa.optimization_contract", optimization)
    spec = importlib.util.spec_from_file_location(
        "shared_overlap_eligibility",
        ROOT / "vllm_musa/model_executor/layers/shared_experts.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = SimpleNamespace(
        model_config=SimpleNamespace(quantization="w4afp8"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            enable_expert_parallel=True,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            enable_eplb=False,
            enable_dbo=False,
        ),
        contract_prefers_overlap=True,
    )
    return module, config, quantization.MusaW4AFp8MoEMethod()


def test_only_the_validated_quant_method_is_enabled(eligibility):
    module, config, method = eligibility
    assert module.can_overlap_shared_experts(config, method)
    assert not module.can_overlap_shared_experts(SimpleNamespace(), object())
    config.contract_prefers_overlap = False
    assert not module.can_overlap_shared_experts(config, method)


def test_contract_is_the_only_policy_gate(eligibility):
    module, config, method = eligibility
    config.contract_prefers_overlap = False
    assert not module.can_overlap_shared_experts(config, method)
    config.contract_prefers_overlap = True
    assert module.can_overlap_shared_experts(config, method)


def test_overlap_helper_has_no_environment_gate():
    source = (ROOT / "vllm_musa/model_executor/layers/shared_experts.py").read_text()
    assert "VLLM_MUSA_W4AFP8_SHARED_OVERLAP" not in source
    assert "prefers_feature" in source


@pytest.fixture
def shared():
    torch = pytest.importorskip("torch")
    source = RUNNER / "shared_experts.py"
    if not source.is_file():
        pytest.skip("Apply the patch series to the optional upstream checkout first")
    nodes = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef)
    ]
    namespace = {
        "IntEnum": IntEnum,
        "Callable": Callable,
        "dbo_enabled": lambda: False,
        "dbo_current_ubatch_id": lambda: 0,
    }
    events = []

    class Event:
        def __init__(self):
            self.calls = []
            events.append(self)

        def record(self, stream):
            self.calls.append(("record", stream))

        def wait(self, stream):
            self.calls.append(("wait", stream))

    platform = SimpleNamespace(musa=True, cuda_alike=True)
    current = ["musa-launch-main"]
    auxiliary = object()
    auxiliary_stream = Mock(return_value=auxiliary)
    namespace.update(
        {
            "torch": SimpleNamespace(
                nn=torch.nn,
                cuda=SimpleNamespace(
                    Event=Event, stream=lambda _: contextlib.nullcontext()
                ),
                musa=SimpleNamespace(current_stream=lambda: current[0]),
            ),
            "current_stream": lambda: "cached-cuda-main",
            "current_platform": SimpleNamespace(
                is_musa=lambda: platform.musa, is_cuda_alike=lambda: platform.cuda_alike
            ),
            "envs": SimpleNamespace(
                VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=32,
                VLLM_DISABLE_SHARED_EXPERTS_STREAM=False,
            ),
            "logger": SimpleNamespace(debug_once=lambda message: None),
            "aux_stream": auxiliary_stream,
        }
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    exec(compile(module, str(source), "exec"), namespace)
    config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            enable_eplb=False,
            all2all_backend=None,
            use_fi_nvl_two_sided_kernels=False,
        )
    )

    def make(*, enabled=True, dbo=False, internal=False):
        return namespace["SharedExperts"](
            torch.nn.Identity(),
            config,
            dbo,
            lambda: internal,
            external_overlap=lambda: enabled,
        )

    return SimpleNamespace(
        make=make,
        platform=platform,
        current=current,
        events=events,
        order=namespace["SharedExpertsOrder"],
        envs=namespace["envs"],
        tensor=lambda count: torch.empty(count, 8),
        auxiliary=auxiliary,
        aux_stream=auxiliary_stream,
    )


@pytest.mark.parametrize(
    "tokens,enabled,expected",
    [(1, True, True), (2, True, True), (32, True, True), (1, False, False)],
)
def test_musa_external_overlap_honors_token_threshold_and_opt_in(
    shared, tokens, enabled, expected
):
    experts = shared.make(enabled=enabled)
    actual = experts._determine_shared_experts_order(shared.tensor(tokens))
    assert (actual == shared.order.MULTI_STREAM_OVERLAPPED) is expected


@pytest.mark.parametrize(
    "guard", ["sequence_parallel", "threshold", "auxiliary", "dbo"]
)
def test_existing_execution_guards_remain_effective(shared, guard):
    experts = shared.make(dbo=guard == "dbo")
    if guard == "sequence_parallel":
        experts._layer.shard_sequence_parallel = True
    elif guard == "threshold":
        shared.envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD = 0
    elif guard == "auxiliary":
        experts._stream = None
    assert (
        experts._determine_shared_experts_order(shared.tensor(1))
        == shared.order.NO_OVERLAP
    )


@pytest.mark.parametrize("musa", [True, False], ids=["musa", "cuda"])
@pytest.mark.parametrize("disabled", [True, False], ids=["disabled", "enabled"])
def test_constructor_stream_guard_overrides_overlap_eligibility(shared, musa, disabled):
    shared.platform.musa = musa
    shared.envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM = disabled
    experts = shared.make(enabled=musa)
    inputs = shared.tensor(1)

    assert experts._external_overlap_allowed is musa
    expected_stream = None if disabled else shared.auxiliary
    assert experts._stream is expected_stream
    assert len(shared.events) == (0 if disabled else 4)
    assert experts.maybe_forward_async(inputs) is (not disabled)
    if disabled:
        shared.aux_stream.assert_not_called()
        experts(inputs, shared.order.NO_OVERLAP)
    else:
        shared.aux_stream.assert_called_once_with()
        experts.wait()
    assert experts.output is inputs


def test_internal_overlap_is_not_replaced_with_external_overlap(shared):
    experts = shared.make(internal=True)
    assert (
        experts._determine_shared_experts_order(shared.tensor(1))
        == shared.order.MK_INTERNAL_OVERLAPPED
    )


def test_cuda_batch_policy_and_cached_stream_are_unchanged(shared):
    shared.platform.musa = False
    experts = shared.make(enabled=False)
    assert (
        experts._determine_shared_experts_order(shared.tensor(32))
        == shared.order.MULTI_STREAM_OVERLAPPED
    )
    assert experts._current_stream() == "cached-cuda-main"


def test_mock_event_calls_query_the_dynamic_musa_stream(shared):
    """Verify event stream arguments without real graph capture or replay."""
    experts = shared.make()
    assert experts.maybe_forward_async(shared.tensor(1))
    shared.current[0] = "musa-wait-main"
    experts.wait()
    assert shared.events[0].calls == [
        ("record", "musa-launch-main"),
        ("wait", shared.auxiliary),
    ]
    assert shared.events[2].calls == [
        ("record", shared.auxiliary),
        ("wait", "musa-wait-main"),
    ]


def test_eligibility_is_resolved_only_during_runner_initialization():
    source = RUNNER / "moe_runner.py"
    if not source.is_file():
        pytest.skip("Apply the patch series to the optional upstream checkout first")
    owners = [
        node.name
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "can_overlap_shared_experts"
            for child in ast.walk(node)
        )
    ]
    assert owners == ["__init__"]
