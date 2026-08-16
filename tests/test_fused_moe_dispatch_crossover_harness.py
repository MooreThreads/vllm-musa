# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
import types
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_musa.engine_plan.fused_moe_evidence import (
    FUSED_MOE_CROSSOVER_SCHEMA,
    FUSED_MOE_MIN_QUALIFIED_L2_FLUSH_MB,
)
from vllm_musa.engine_plan.fused_moe_evidence import (
    FUSED_MOE_MAX_ERROR_THRESHOLDS as IMPORTER_MAX_ERROR_THRESHOLDS,
)
from vllm_musa.engine_plan.fused_moe_evidence import (
    FUSED_MOE_MIN_COSINE_THRESHOLDS as IMPORTER_MIN_COSINE_THRESHOLDS,
)

BENCHMARK = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "fused_moe"
    / "benchmark_dispatch_crossover.py"
)
CONTRACT = (
    Path(__file__).parents[1] / "vllm_musa" / "engine_plan" / "fused_moe_contract.py"
)


def _stub_module(monkeypatch, name: str, *, package: bool = False):
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def harness(monkeypatch):
    _stub_module(monkeypatch, "torchada")
    torch = _stub_module(monkeypatch, "torch", package=True)
    torch_nn = _stub_module(monkeypatch, "torch.nn", package=True)
    torch_functional = _stub_module(monkeypatch, "torch.nn.functional")
    torch.nn = torch_nn
    torch_nn.functional = torch_functional

    mate = _stub_module(monkeypatch, "mate", package=True)
    mate_testing = _stub_module(monkeypatch, "mate.testing", package=True)
    mate_utils = _stub_module(monkeypatch, "mate.testing.utils")
    mate.testing = mate_testing
    mate_testing.utils = mate_utils
    mate_utils.bench_gpu_time_with_musa_event = lambda *args, **kwargs: []

    vllm_musa = _stub_module(monkeypatch, "vllm_musa", package=True)
    engine_plan = _stub_module(monkeypatch, "vllm_musa.engine_plan", package=True)
    contract_spec = importlib.util.spec_from_file_location(
        "vllm_musa.engine_plan.fused_moe_contract", CONTRACT
    )
    assert contract_spec is not None and contract_spec.loader is not None
    contract = importlib.util.module_from_spec(contract_spec)
    monkeypatch.setitem(
        sys.modules, "vllm_musa.engine_plan.fused_moe_contract", contract
    )
    contract_spec.loader.exec_module(contract)
    engine_plan.fused_moe_contract = contract
    vllm_musa.engine_plan = engine_plan
    model_executor = _stub_module(monkeypatch, "vllm_musa.model_executor", package=True)
    layers = _stub_module(monkeypatch, "vllm_musa.model_executor.layers", package=True)
    fused_moe_package = _stub_module(
        monkeypatch,
        "vllm_musa.model_executor.layers.fused_moe",
        package=True,
    )
    dispatch_policy = _stub_module(
        monkeypatch,
        "vllm_musa.model_executor.layers.fused_moe.dispatch_policy",
    )
    vllm_musa.model_executor = model_executor
    model_executor.layers = layers
    layers.fused_moe = fused_moe_package
    fused_moe_package.fused_moe = SimpleNamespace()
    fused_moe_package.dispatch_policy = dispatch_policy
    dispatch_policy.MusaFusedMoeBackend = object
    dispatch_policy.MusaFusedMoeShape = object

    module_name = f"test_dispatch_crossover_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, BENCHMARK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _valid_measurement_args():
    return SimpleNamespace(
        warmup=2,
        repeats_per_round=10,
        rounds=6,
        l2_flush_mb=FUSED_MOE_MIN_QUALIFIED_L2_FLUSH_MB,
        seed=11,
        gemv_max_relative_l2=0.06,
        gemv_min_cosine=0.998,
        gemv_max_row_relative_l2=0.08,
        gemv_min_row_cosine=0.995,
        gemv_max_normalized_abs_diff=0.10,
        gemv_oracle_max_relative_l2=0.01,
        gemv_oracle_min_cosine=0.9999,
        gemv_oracle_max_row_relative_l2=0.02,
        gemv_oracle_min_row_cosine=0.9998,
        gemv_oracle_max_normalized_abs_diff=0.05,
        grouped_max_relative_l2=0.01,
        grouped_min_cosine=0.9999,
        grouped_max_row_relative_l2=0.02,
        grouped_min_row_cosine=0.999,
        grouped_max_normalized_abs_diff=0.05,
        regression_margin=0.03,
        max_iqr_ratio=0.10,
    )


def test_device_dump_evidence_preserves_preexisting_and_rejects_new(
    harness, tmp_path
) -> None:
    preexisting = tmp_path / "preexisting.mudmp"
    preexisting.write_text("old")
    baseline, errors = harness._scan_musa_device_dumps([tmp_path])
    created = tmp_path / "created-by-run.mudmp"
    created.write_text("new")

    evidence = harness._device_dump_evidence(
        roots=[tmp_path], baseline=baseline, baseline_errors=errors
    )

    assert evidence["passed"] is False
    assert [item["path"] for item in evidence["preexisting_dumps"]] == [
        str(preexisting.resolve())
    ]
    assert [item["path"] for item in evidence["new_dumps"]] == [str(created.resolve())]
    assert preexisting.read_text() == "old"
    assert created.read_text() == "new"


def test_device_dump_evidence_accepts_unchanged_preexisting(harness, tmp_path) -> None:
    preexisting = tmp_path / "preexisting.mudmp"
    preexisting.write_text("old")
    baseline, errors = harness._scan_musa_device_dumps([tmp_path])

    evidence = harness._device_dump_evidence(
        roots=[tmp_path], baseline=baseline, baseline_errors=errors
    )

    assert evidence["passed"] is True
    assert evidence["new_dumps"] == []
    assert evidence["changed_preexisting_dumps"] == []


def test_device_dump_evidence_rejects_missing_preexisting(harness, tmp_path) -> None:
    preexisting = tmp_path / "preexisting.mudmp"
    preexisting.write_text("old")
    baseline, errors = harness._scan_musa_device_dumps([tmp_path])
    preexisting.unlink()

    evidence = harness._device_dump_evidence(
        roots=[tmp_path], baseline=baseline, baseline_errors=errors
    )

    assert evidence["passed"] is False
    assert evidence["missing_preexisting_dumps"] == [str(preexisting.resolve())]


def test_exclusive_benchmark_lock_rejects_same_visible_device(
    harness, monkeypatch, tmp_path
) -> None:
    source_root = tmp_path / "source"
    (source_root / ".git").mkdir(parents=True)
    monkeypatch.setenv("VLLM_MUSA_SOURCE_ROOT", str(source_root))
    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "3")

    first, evidence = harness.acquire_exclusive_benchmark_lock()
    try:
        assert evidence["acquired"] is True
        assert evidence["visible_devices"] == "3"
        with pytest.raises(RuntimeError, match="already owns"):
            harness.acquire_exclusive_benchmark_lock()
    finally:
        first.close()

    replacement, _ = harness.acquire_exclusive_benchmark_lock()
    replacement.close()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("warmup", 0, "warmup must be positive"),
        ("repeats_per_round", 2, "at least three"),
        ("l2_flush_mb", 0, "must be positive"),
        ("seed", -1, "seed must be non-negative"),
        ("max_iqr_ratio", 0.20, "must be 0.10"),
        ("gemv_max_relative_l2", 0.061, "canonical maximum"),
        ("gemv_max_relative_l2", float("inf"), "finite and non-negative"),
        ("gemv_min_cosine", 0.997, r"canonical minimum"),
        ("gemv_min_cosine", float("nan"), r"must be in \[0.998, 1\]"),
    ),
)
def test_measurement_parameter_validation_rejects_unqualified_settings(
    harness, field, value, message
) -> None:
    args = _valid_measurement_args()
    setattr(args, field, value)

    with pytest.raises(ValueError, match=message):
        harness.validate_measurement_parameters(args, backend_count=2)


def test_measurement_parameter_validation_requires_balanced_rounds(harness) -> None:
    args = _valid_measurement_args()
    args.rounds = 5

    with pytest.raises(ValueError, match="multiple of backend count"):
        harness.validate_measurement_parameters(args, backend_count=2)


def test_shape_validation_requires_positive_aligned_dimensions(harness) -> None:
    shape = harness.Shape(
        name="unaligned",
        experts=128,
        hidden_size=4097,
        intermediate_size=256,
        top_k=8,
        block_size=128,
    )

    with pytest.raises(ValueError, match="hidden-size must be aligned"):
        harness.validate_shape(shape)


def test_candidate_recommendation_clears_unqualified_results(harness) -> None:
    result = harness.Result(
        shape={},
        tokens=1,
        backend="gemv",
        samples_ms=[1.0, 1.0, 1.0],
        round_medians_ms=[1.0, 1.0, 1.0],
        median_ms=1.0,
        p20_ms=1.0,
        p95_ms=1.0,
        iqr_ms=0.0,
        min_ms=1.0,
        max_ms=1.0,
        max_abs_diff=0.0,
        mean_abs_diff=0.0,
        relative_l2_error=0.0,
        cosine_similarity=1.0,
        max_row_relative_l2=0.0,
        min_row_cosine=1.0,
        max_abs_diff_over_reference_absmax=0.0,
        output_absmax=1.0,
        output_std=1.0,
        finite=True,
        non_poison=True,
        correctness_pass=True,
        correctness_basis="test",
        reference_backend="upstream",
        error=None,
    )

    assert harness.candidate_recommendation(
        [result],
        [1],
        qualified=False,
        regression_margin=0.03,
        max_iqr_ratio=0.10,
    ) == {"gemv_max_tokens": None, "grouped_gemm_min_tokens": None}


def test_producer_and_importer_v8_contracts_match(harness) -> None:
    assert harness.FUSED_MOE_CROSSOVER_SCHEMA == FUSED_MOE_CROSSOVER_SCHEMA
    assert (
        harness.FUSED_MOE_MIN_QUALIFIED_L2_FLUSH_MB
        == FUSED_MOE_MIN_QUALIFIED_L2_FLUSH_MB
    )
    assert harness.FUSED_MOE_MAX_ERROR_THRESHOLDS == IMPORTER_MAX_ERROR_THRESHOLDS
    assert harness.FUSED_MOE_MIN_COSINE_THRESHOLDS == IMPORTER_MIN_COSINE_THRESHOLDS
