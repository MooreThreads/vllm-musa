"""Exercise v2 host routing and graph-slot ownership without a GPU runtime."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
COMM = ROOT / "vllm_musa/distributed/device_communicators/musa_jit_custom_all_reduce.py"


class Tensor:
    def __init__(
        self, shape=(16, 2048), dtype="bf16", device="musa:0", contiguous=True
    ):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.contiguous = contiguous

    def dim(self):
        return len(self.shape)

    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result

    def element_size(self):
        return 2

    def is_contiguous(self):
        return self.contiguous

    def data_ptr(self):
        return id(self)


def _host():
    spec = importlib.util.spec_from_file_location(
        "host_test_fused_policy",
        ROOT / "vllm_musa/optimization_contract/Fuse_CAR_RMSnorm.py",
    )
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    policy.dispatch_enabled = lambda: True
    # A frozen production KernelConfig is hashable; retain the same property.
    decision = policy.KernelConfig(
        kernel=policy.CustomARKernel.ROW_ONE,
        graph_rank_data=True,
        token_2stage=False,
        tuned_module=True,
    )
    policy.resolve_fused_car_rmsnorm = lambda **kwargs: decision
    events = []
    state = {"capturing": False, "fail_launch": False}

    def factory(*args, **kwargs):
        events.append(("factory", args, kwargs))

        def launch(*args):
            events.append(("launch", args))
            if state["fail_launch"]:
                raise RuntimeError("launch failed")

        return launch

    torch = SimpleNamespace(
        Tensor=Tensor,
        float16="fp16",
        bfloat16="bf16",
        float32="fp32",
        get_device_module=lambda: SimpleNamespace(
            get_device_properties=lambda device: SimpleNamespace(
                multi_processor_count=60
            )
        ),
        empty_like=lambda x: Tensor(x.shape, x.dtype, x.device),
    )
    ns = {
        "torch": torch,
        "fused_car_policy": policy,
        "_INT32_MAX": (1 << 31) - 1,
        "_is_current_stream_capturing": lambda: state["capturing"],
        "jit_ar": SimpleNamespace(v2_fused_allreduce_rmsnorm_func=factory),
        "car_trace": SimpleNamespace(
            enabled=lambda: True, record=lambda *args: events.append(("record", args))
        ),
    }
    tree = ast.parse(COMM.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MusaCustomAllReduceV2RMSNorm"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
    exec(compile(module, str(COMM), "exec"), ns)
    owner = SimpleNamespace(
        disabled=False,
        _device="musa:0",
        max_size=8 << 20,
        world_size=4,
        rank=1,
        _rank_data_device=object(),
        _fused_signal_ptrs_cpu=object(),
        _self_fused_signal_ptr=1000,
        _self_buffer_ptr=2000,
        _graph_capture_inputs=[],
        _graph_input_count=4,
        _graph_rank_data=[object() for _ in range(4)],
    )
    return ns["MusaCustomAllReduceV2RMSNorm"](owner), owner, policy, state, events


def test_eager_uses_fused_signals_and_preserves_three_output_contracts():
    host, owner, _, _, events = _host()
    input, residual, weight = Tensor(), Tensor(), Tensor((2048,))
    for residual_arg, need_raw in [(None, True), (residual, True), (residual, False)]:
        out = host.run(input, residual_arg, weight, 1e-6, need_raw=need_raw)
        assert len(out) == 3
        assert all(value is not input and value is not residual for value in out)
        assert (out[2] is not None) == need_raw
        args = next(event[1] for event in reversed(events) if event[0] == "launch")
        assert args[0] is owner._rank_data_device
        assert args[1] is owner._fused_signal_ptrs_cpu
        assert args[2] is input and args[3] is residual_arg
        assert args[8:10] == (1000, 2000)
        assert events[-1][0] == "record"
        assert host._last_dispatch["execution"] == "fused"
        assert host._last_dispatch["need_raw"] == need_raw
        assert host._last_dispatch["has_residual"] == (residual_arg is not None)
        assert host._last_dispatch["phase"] == "eager"
        import json

        json.dumps(host._last_dispatch)
    assert not owner._graph_capture_inputs


def test_prepare_does_not_reserve_graph_slots_and_capture_uses_actual_input():
    host, owner, _, state, events = _host()
    input, residual, weight = Tensor(), Tensor(), Tensor((2048,))
    host.prepare(input, residual, need_raw=False, capturing=True)
    assert owner._graph_capture_inputs == []
    state["capturing"] = True
    host.run(input, residual, weight, 1e-6, need_raw=False)
    assert owner._graph_capture_inputs == [input.data_ptr()]
    args = next(event[1] for event in events if event[0] == "launch")
    assert args[0] is owner._graph_rank_data[0]
    assert host._last_dispatch["registered"] is True
    assert host._last_dispatch["phase"] == "capture"
    assert events[0][2]["graph_rank_data"] is True


def test_fallback_does_not_allocate_graph_slot_or_record_fused_launch():
    host, owner, policy, state, events = _host()
    state["capturing"] = True
    policy.resolve_fused_car_rmsnorm = lambda **kwargs: policy.KernelConfig(
        kernel=policy.CustomARKernel.FALLBACK
    )
    assert host.run(Tensor(), Tensor(), Tensor((2048,)), 1e-6, need_raw=False) is None
    assert owner._graph_capture_inputs == []
    assert events == []
    assert host._last_dispatch["execution"] == "fallback"


def test_disabled_and_invalid_inputs_are_rejected_before_any_launch():
    host, owner, policy, _, events = _host()
    weight = Tensor((2048,))
    assert host.rejection_reason(Tensor(dtype="float32"), None, weight)
    assert host.rejection_reason(Tensor(contiguous=False), None, weight)
    assert host.rejection_reason(Tensor(), Tensor((1, 2048)), weight)
    assert host.rejection_reason(Tensor(), None, Tensor((1024,)))
    policy.dispatch_enabled = lambda: False
    assert host.rejection_reason(Tensor(), None, weight)
    assert events == [] and not owner._graph_capture_inputs


def test_launch_failure_does_not_emit_success_counter():
    host, _, _, state, events = _host()
    state["fail_launch"] = True
    try:
        host.run(Tensor(), Tensor(), Tensor((2048,)), 1e-6, need_raw=False)
    except RuntimeError as exc:
        assert str(exc) == "launch failed"
    else:
        raise AssertionError("launch failure was swallowed")
    assert not any(event[0] == "record" for event in events)


def test_fp32_weight_uses_composed_fallback_without_reserving_graph_slot():
    host, owner, _, state, events = _host()
    state["capturing"] = True
    weight = Tensor((2048,), dtype="fp32")
    assert host.rejection_reason(Tensor(), Tensor(), weight) is None
    assert host.run(Tensor(), Tensor(), weight, 1e-6, need_raw=True) is None
    assert not events and not owner._graph_capture_inputs


def test_gemma_is_a_distinct_launcher_and_dispatch_semantic():
    host, _, _, _, events = _host()
    x, residual, weight = Tensor(), Tensor(), Tensor((2048,))
    host.run(x, residual, weight, 1e-6, need_raw=False)
    host.run(x, residual, weight, 1e-6, need_raw=False, gemma=True)
    factories = [event[2] for event in events if event[0] == "factory"]
    assert [item["gemma"] for item in factories] == [False, True]
    assert len(host._launchers) == 2
    assert host._last_dispatch["gemma"] is True
    host.run(x, residual, weight, 1e-6, need_raw=False, gemma=True)
    assert len([event for event in events if event[0] == "factory"]) == 2


def test_gemma_capture_prepares_matching_semantics_without_allocating_slot():
    host, owner, _, state, events = _host()
    x, residual, weight = Tensor(), Tensor(), Tensor((2048,))
    host.prepare(x, residual, need_raw=False, capturing=True, gemma=True)
    assert not owner._graph_capture_inputs
    state["capturing"] = True
    host.run(x, residual, weight, 1e-6, need_raw=False, gemma=True)
    assert owner._graph_capture_inputs == [x.data_ptr()]
    assert len([event for event in events if event[0] == "factory"]) == 1
    assert host._last_dispatch["gemma"] is True


def test_registered_ops_keep_legacy_calls_and_gate_gemma_capability():
    tree = ast.parse(COMM.read_text())
    for suffix, count in [
        ("rmsnorm", 2),
        ("residual_rmsnorm", 3),
        ("residual_rmsnorm_no_raw", 3),
    ]:
        name = "_musa_jit_fused_allreduce_" + suffix
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        calls = []
        comm = SimpleNamespace()
        setattr(
            comm,
            "_fused_allreduce_" + suffix + "_impl",
            lambda *args, **kwargs: calls.append((args, kwargs)),
        )
        namespace = {"get_musa_jit_custom_allreduce_comm": lambda _: comm}
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        module = ast.fix_missing_locations(
            ast.Module(body=[future, function], type_ignores=[])
        )
        exec(compile(module, str(COMM), "exec"), namespace)
        args = [Tensor() for _ in range(count)] + [1e-6, 0]
        namespace[name](*args)
        assert calls[-1][1] == {}
        try:
            namespace[name](*args, gemma=True)
        except RuntimeError as error:
            assert "capable v2" in str(error)
        else:
            raise AssertionError("Gemma call accepted by legacy v1 communicator")
        comm.supports_fused_gemma_rmsnorm = True
        namespace[name](*args, gemma=True)
        assert calls[-1][1] == {"gemma": True}
