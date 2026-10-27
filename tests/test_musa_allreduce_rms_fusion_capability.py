"""CPU behavior checks for fused communicator eligibility before graph rewriting."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

ROOT = Path(__file__).parents[1]
FUSION_PASS = ROOT / "vllm_musa/_inductor/musa_allreduce_rms_fusion.py"
_MISSING = object()


def _load_pass_init_and_call(comm: Any) -> type:
    """Execute production initialization and call guards with CPU test doubles."""
    source = ast.parse(FUSION_PASS.read_text())
    fusion_class = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef)
        and node.name == "MusaAllReduceRMSNormFusionPass"
    )
    methods = {
        "__init__",
        "__call__",
        "_fused_capability_enabled",
        "_max_fusable_tokens",
        "is_applicable_for_range",
        "uuid",
    }
    fusion_class.body = [
        node
        for node in fusion_class.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    assert {node.name for node in fusion_class.body} == methods
    for node in fusion_class.body:
        if node.name == "__call__":
            node.decorator_list = []

    class BasePass:
        def __init__(self, config: Any) -> None:
            self.model_dtype = config.model_config.dtype
            self.device = "cpu"
            self.registered = False
            self._manual_rewrite_residual_musa_jit_car_rmsnorm = Mock(
                return_value=(0, 0)
            )

        def register_patterns(self) -> None:
            self.registered = True
            self.disabled = False

        def dump_patterns(self, config: Any, patterns: Any) -> None:
            pass

        @staticmethod
        def hash_source(*args: Any) -> str:
            return "source"

        @staticmethod
        def hash_dict(state: dict[str, Any]) -> str:
            return repr(state)

    tp_group = SimpleNamespace(
        unique_name="tp0",
        device_communicator=SimpleNamespace(
            ca_comm=SimpleNamespace(disabled=False, _jit_comm=comm)
        ),
    )

    class Pattern:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def register(self, patterns: Any) -> None:
            pass

    namespace = {
        "VllmPatternMatcherPass": BasePass,
        "current_platform": SimpleNamespace(is_musa=lambda: True),
        "get_tensor_model_parallel_world_size": lambda: 2,
        "get_tp_group": lambda: tp_group,
        "logger": Mock(),
        "torch": SimpleNamespace(
            float16="float16",
            bfloat16="bfloat16",
            empty=lambda *args, **kwargs: SimpleNamespace(element_size=lambda: 2),
        ),
        "PatternMatcherPass": lambda **kwargs: SimpleNamespace(
            apply=Mock(return_value=2)
        ),
        "MusaAllReduceRMSNormPattern": Pattern,
        "MusaAllReduceResidualRMSNormPattern": Pattern,
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            fusion_class,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(FUSION_PASS), "exec"), namespace
    )
    return namespace["MusaAllReduceRMSNormFusionPass"]


def _make_pass(capability: Any = _MISSING, gemma_capability: Any = _MISSING) -> Any:
    comm = SimpleNamespace(_comm_id=23, max_size=1024 * 1024)
    if capability is not _MISSING:
        comm.supports_fused_rmsnorm = capability
    if gemma_capability is not _MISSING:
        comm.supports_fused_gemma_rmsnorm = gemma_capability
    config = SimpleNamespace(
        model_config=SimpleNamespace(get_hidden_size=lambda: 4096, dtype="bfloat16"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=128),
    )
    return _load_pass_init_and_call(comm)(config)


def test_unavailable_fusion_does_not_register_or_rewrite() -> None:
    for capability in (False, None, 1):
        fusion = _make_pass(capability)
        assert fusion.disabled
        assert not fusion.registered
        assert not fusion.is_applicable_for_range(SimpleNamespace(end=1))
        fusion(object())
        fusion._manual_rewrite_residual_musa_jit_car_rmsnorm.assert_not_called()
        assert not hasattr(fusion, "patterns")


def test_supported_communicator_registers_and_runs_existing_rewrite() -> None:
    for capability in (_MISSING, True):
        fusion = _make_pass(capability)
        assert not fusion.disabled
        assert fusion.registered
        assert fusion.comm_id == fusion.jit_comm_id == 23
        assert fusion.is_applicable_for_range(SimpleNamespace(end=128))
        assert not fusion.is_applicable_for_range(SimpleNamespace(end=129))
        graph = object()
        fusion(graph)
        fusion._manual_rewrite_residual_musa_jit_car_rmsnorm.assert_called_once_with(
            graph
        )
        fusion.patterns.apply.assert_called_once_with(graph)
        assert fusion.matched_count == 2


def test_v2_capability_changes_compilation_cache_key() -> None:
    assert _make_pass(False).uuid() != _make_pass(True).uuid()


def test_gemma_rewrite_requires_explicit_capability_and_changes_cache_key() -> None:
    assert not _make_pass().supports_fused_gemma_rmsnorm
    assert not _make_pass(True).supports_fused_gemma_rmsnorm
    assert not _make_pass(True, 1).supports_fused_gemma_rmsnorm
    assert _make_pass(True, True).supports_fused_gemma_rmsnorm
    assert _make_pass(True, False).uuid() != _make_pass(True, True).uuid()
