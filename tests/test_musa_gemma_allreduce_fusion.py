"""CPU FX checks for the functionalized Gemma RMSNorm fusion boundary."""

from __future__ import annotations

import ast
import operator
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

PASS = Path(__file__).parents[1] / "vllm_musa/_inductor/musa_allreduce_rms_fusion.py"
BINDINGS = Path(__file__).parents[1] / "vllm_musa/fused_allreduce_rmsnorm_ops.py"


def _production_bindings() -> dict[str, Any]:
    tree = ast.parse(BINDINGS.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(BINDINGS), "exec"),
        namespace,
    )
    return namespace


def _norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gemma: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    value = x.float() + residual.float()
    normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    norm = (
        (normalized * (weight.float() + 1)).to(x.dtype)
        if gemma
        else (normalized.to(weight.dtype) * weight).to(x.dtype)
    )
    return norm, value.to(x.dtype)


@pytest.fixture
def ops() -> Any:
    namespace = "test_car_gemma_" + uuid.uuid4().hex
    library = torch.library.Library(namespace, "DEF")
    library.define("musa_jit_custom_all_reduce(Tensor x, int comm_id) -> Tensor")
    library.define(
        "musa_csrc_fused_add_rmsnorm(Tensor(a!) input, Tensor(b!) residual, Tensor weight, float eps, bool gemma=False) -> ()"
    )
    bindings = _production_bindings()
    names = (
        "musa_fused_allreduce_residual_rms_norm",
        "musa_fused_allreduce_residual_rms_norm_no_raw",
    )
    for name in names:
        schema = torch.library.infer_schema(bindings[f"_{name}_impl"], mutates_args=())
        library.define(name + schema)

    def inplace(
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        gemma: bool = False,
    ) -> None:
        norm, summed = _norm(input, residual, weight, eps, gemma)
        input.copy_(norm)
        residual.copy_(summed)

    def fused(
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        comm_id: int,
        gemma: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        reduced = input * 2
        return (*_norm(reduced, residual, weight, eps, gemma), reduced)

    library.impl("musa_jit_custom_all_reduce", lambda x, comm_id: x * 2, "CPU")
    library.impl("musa_csrc_fused_add_rmsnorm", inplace, "CPU")
    comm = SimpleNamespace(
        fused_allreduce_residual_rmsnorm=lambda *a, **k: fused(*a, comm_id=0, **k),
        fused_allreduce_residual_rmsnorm_no_raw=lambda *a, **k: fused(
            *a, comm_id=0, **k
        )[:2],
    )
    bindings["get_musa_jit_custom_allreduce_comm"] = lambda _: comm
    for name in names:
        library.impl(name, bindings[f"_{name}_impl"], "CPU")
        torch.library.register_fake(
            f"{namespace}::{name}", bindings[f"_{name}_fake"], lib=library
        )
    yield getattr(torch.ops, namespace)


def _helpers(ops: Any) -> Any:
    tree = ast.parse(PASS.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "MusaAllReduceRMSNormFusionPass"
    )
    cls.bases = []
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("_")
        and node.name not in ("__init__", "__call__")
    ]
    namespace = {
        "torch": SimpleNamespace(
            Tensor=torch.Tensor,
            float16=torch.float16,
            bfloat16=torch.bfloat16,
            float32=torch.float32,
            ops=SimpleNamespace(vllm=ops, aten=torch.ops.aten),
        ),
        "fx": torch.fx,
        "FakeTensor": FakeTensor,
        "operator": operator,
        "auto_functionalized": auto_functionalized,
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(PASS), "exec"), namespace)
    instance = namespace[cls.name]()
    instance.comm_id = instance.jit_comm_id = 23
    instance.hidden_dim = 16
    instance.supports_fused_gemma_rmsnorm = True
    return instance


def _graph(
    ops: Any,
    *,
    raw: bool = False,
    early_raw: bool = False,
    gemma: bool = True,
    comm_id: int = 23,
    extra_slot: bool = False,
) -> torch.fx.Graph:
    graph = torch.fx.Graph()
    x, residual, weight = [
        graph.placeholder(name) for name in ("x", "residual", "weight")
    ]
    mode = FakeTensorMode()
    for node, shape in ((x, (4, 16)), (residual, (4, 16)), (weight, (16,))):
        node.meta["val"] = FakeTensor(
            mode,
            torch.empty(shape, dtype=torch.bfloat16, device="meta"),
            torch.device("musa", 0),
        )
    car = graph.call_function(ops.musa_jit_custom_all_reduce.default, args=(x, comm_id))
    car.meta["val"] = x.meta["val"]
    raw_user = graph.call_function(torch.neg, args=(car,)) if early_raw else None
    norm = graph.call_function(
        auto_functionalized,
        args=(ops.musa_csrc_fused_add_rmsnorm.default,),
        kwargs=dict(input=car, residual=residual, weight=weight, eps=1e-6, gemma=gemma),
    )
    norm_out = graph.call_function(operator.getitem, args=(norm, 1))
    residual_out = graph.call_function(operator.getitem, args=(norm, 2))
    graph.call_function(torch.ops.aten.copy_.default, args=(residual, residual_out))
    outputs = [norm_out, residual_out, residual]
    if raw:
        outputs.append(
            raw_user if early_raw else graph.call_function(torch.neg, args=(car,))
        )
    if extra_slot:
        outputs.append(graph.call_function(operator.getitem, args=(norm, 0)))
    graph.output(tuple(outputs))
    return graph


@pytest.mark.parametrize("raw,early_raw", [(False, False), (True, False), (True, True)])
def test_functionalized_gemma_rewrite_preserves_values_alias_copy_and_meta(
    ops: Any, raw: bool, early_raw: bool
) -> None:
    graph = _graph(ops, raw=raw, early_raw=early_raw)
    original = torch.fx.GraphModule({}, graph)
    x = torch.randn(4, 16, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.linspace(-0.25, 0.5, 16).to(x.dtype)
    expected = original(x.clone(), residual.clone(), weight.clone())
    helpers = _helpers(ops)
    assert helpers._manual_rewrite_residual_musa_jit_car_rmsnorm(graph) == (
        (0, 1) if raw else (1, 0)
    )
    graph.lint()
    fused_nodes = [
        node for node in graph.nodes if "musa_fused_allreduce" in str(node.target)
    ]
    assert len(fused_nodes) == 1 and fused_nodes[0].kwargs == {"gemma": True}
    assert not any(node.target is auto_functionalized for node in graph.nodes)
    assert not any(
        "musa_jit_custom_all_reduce" in str(node.target) for node in graph.nodes
    )
    assert all(isinstance(value, FakeTensor) for value in fused_nodes[0].meta["val"])
    assert all("val" in node.meta for node in fused_nodes[0].users)
    actual_x, actual_residual = x.clone(), residual.clone()
    actual = torch.fx.GraphModule({}, graph)(actual_x, actual_residual, weight)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual_x, x, rtol=0, atol=0)
    assert actual[2].data_ptr() == actual_residual.data_ptr()
    assert actual[0].data_ptr() != actual_x.data_ptr()


@pytest.mark.parametrize("kind", ["v1", "ordinary", "wrong_comm", "non_tensor_slot"])
def test_gemma_rewrite_fails_closed_without_matching_contract(
    ops: Any, kind: str
) -> None:
    graph = _graph(
        ops,
        gemma=kind != "ordinary",
        comm_id=24 if kind == "wrong_comm" else 23,
        extra_slot=kind == "non_tensor_slot",
    )
    helpers = _helpers(ops)
    helpers.supports_fused_gemma_rmsnorm = kind != "v1"
    before = str(graph)
    assert helpers._manual_rewrite_residual_musa_jit_car_rmsnorm(graph) == (0, 0)
    assert str(graph) == before


@pytest.mark.parametrize("dependent", ["residual", "weight"])
def test_gemma_rewrite_rejects_collective_dependency_or_late_weight(
    ops: Any, dependent: str
) -> None:
    graph = _graph(ops, raw=True, early_raw=True)
    car = next(
        node for node in graph.nodes if "musa_jit_custom_all_reduce" in str(node.target)
    )
    norm = next(node for node in graph.nodes if node.target is auto_functionalized)
    replacement = car
    if dependent == "weight":
        with graph.inserting_before(norm):
            replacement = graph.call_function(torch.neg, args=(norm.kwargs["weight"],))
        replacement.meta["val"] = norm.kwargs["weight"].meta["val"]
    norm.kwargs = {**norm.kwargs, dependent: replacement}
    before = str(graph)
    assert _helpers(ops)._manual_rewrite_residual_musa_jit_car_rmsnorm(graph) == (0, 0)
    assert str(graph) == before
