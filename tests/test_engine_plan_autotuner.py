# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
import types
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_musa.engine_plan.artifacts import (
    BenchmarkCase,
    ObservationStatus,
    PlanningArtifactError,
    PlanTarget,
    TacticDefinition,
    TacticKind,
    TimingCache,
)
from vllm_musa.engine_plan.autotune_runtime import (
    AutoTuneRuntimeError,
    FusedAddRmsNormCollectConfig,
    FusedAddRmsNormCollector,
    FusedAddRmsNormRunner,
    collect_fused_add_rms_norm,
)
from vllm_musa.engine_plan.autotuner import (
    CandidateMeasurement,
    collect_operation,
)
from vllm_musa.engine_plan.core import EnginePlanError
from vllm_musa.engine_plan.planner import BuildPolicy, build_plan_document
from vllm_musa.engine_plan.tactic_fingerprints import (
    runtime_decision_implementation,
)
from vllm_musa.engine_plan.tuning_domains import (
    FUSED_ADD_RMS_NORM_DOMAIN_HIDDEN_SIZES,
    FUSED_ADD_RMS_NORM_DOMAIN_ID,
    FUSED_ADD_RMS_NORM_H4096_DOMAIN_ID,
    FUSED_MOE_FP8_BLOCK_DOMAIN_ID,
    TUNING_DOMAIN_PROVENANCE_KEY,
    get_tuning_domain,
    validate_tuning_domain_target,
)
from vllm_musa.engine_plugins.api import EngineIrProviderMetadata
from vllm_musa.runtime_plan import RuntimeDecision
from vllm_musa.runtime_plan.catalog import validate_runtime_decision_value
from vllm_musa.tuning import (
    DEFAULT_FUSED_ADD_RMSNORM_MIN_ROWS,
    FUSED_ADD_RMSNORM_TUNED_HIDDEN_SIZES,
    configure_fused_add_rmsnorm_min_rows,
    get_fused_add_rmsnorm_min_rows,
    is_fused_add_rmsnorm_tuned_hidden_size,
    override_fused_add_rmsnorm_min_rows,
)

OPERATION = RuntimeDecision.MUSA_FUSED_ADD_RMSNORM_MIN_ROWS.value


def _target(*, image_digest: str = "sha256:daily") -> PlanTarget:
    return PlanTarget.from_document(
        {
            "model": {
                "architecture": "Qwen3ForCausalLM",
                "profile": "qwen3-h5120-bf16",
                "model_id": "Qwen/Qwen3-32B",
                "hidden_size": 5120,
                "dtype": "bfloat16",
                "quantization": "none",
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
            },
            "hardware": {
                "platform": "musa",
                "device_name": "MTT S5000",
                "device_capability": "3.1",
                "multiprocessor_count": 80,
                "device_count": 1,
                "device_uuid": "GPU-test",
            },
            "software": {
                "versions": {
                    "driver": "5.2.0",
                    "musa": "5.2.0",
                    "torch": "2.11.0",
                    "torch-musa": "2.11.0.post1",
                    "vllm": "0.24.0",
                    "vllm-musa": "0.24.0.dev0",
                    "torchada": "0.1.0",
                    "mate": "0.1.0",
                    "flash-attn-3": "0.1.0",
                    "flash-mla": "0.1.0",
                    "mthreads-ml-py": "0.1.0",
                },
                "source_revisions": {
                    "vllm": "sha-vllm",
                    "vllm-musa": "sha-vllm-musa",
                },
                "image_digest": image_digest,
            },
            "workload": {
                "max_num_seqs": 4,
                "max_num_batched_tokens": 64,
                "max_model_len": 64,
                "compile_mode": "VLLM_COMPILE",
                "graph_mode": "FULL_DECODE_ONLY",
                "phase": "serving",
                "batch_size": {"min": 1, "max": 4},
                "tokens": {"min": 1, "max": 64},
            },
        }
    )


def _target_with_hidden_size(hidden_size: int) -> PlanTarget:
    document = _target().to_document()
    document["model"]["hidden_size"] = hidden_size
    return PlanTarget.from_document(document)


def _definition(
    threshold: int,
    *,
    fingerprint: str = "impl-v1",
    description: str | None = None,
) -> TacticDefinition:
    prefix = "runtime.musa.fused_add_rms_norm.min_rows"
    return TacticDefinition(
        tactic_id=f"{prefix}:{threshold}",
        kind=TacticKind.RUNTIME_DECISION,
        operation=OPERATION,
        choice=threshold,
        fallback_id=f"{prefix}:64",
        implementation_fingerprint=fingerprint,
        description=description or f"threshold {threshold}",
    )


def _case(rows: int) -> BenchmarkCase:
    return BenchmarkCase.create(
        operation=OPERATION,
        phase="operator",
        batch_size=1,
        tokens=rows,
        rows=rows,
        hidden_size=5120,
        dtype="bfloat16",
    )


def _live_environment(
    target: PlanTarget,
    *,
    evidence_drift: bool = False,
) -> dict[str, object]:
    document = target.to_document()
    hardware = dict(document["hardware"])
    if evidence_drift:
        hardware["device_uuid"] = "GPU-live-evidence-only"
        hardware["device_count"] = 8
    return {
        "hardware": hardware,
        "software_versions": dict(document["software"]["versions"]),
        "source_revisions": dict(document["software"]["source_revisions"]),
    }


def _collect_args(
    tmp_path,
    *,
    target_value: PlanTarget | None = None,
    output_name: str = "timing.json",
    resume: str | None = None,
) -> SimpleNamespace:
    target = tmp_path / "target.json"
    target.write_text(
        json.dumps((target_value or _target()).to_document()),
        encoding="utf-8",
    )
    return SimpleNamespace(
        target=str(target),
        output=str(tmp_path / output_name),
        summary_output=None,
        rows=[32],
        eager=True,
        warmups=1,
        iterations=1,
        cold_cache_bytes=0,
        resume=resume,
    )


def _install_fake_live_collect(
    monkeypatch,
    target: PlanTarget,
    measurements: list[tuple[int, tuple[str, ...]]],
    *,
    evidence_drift: bool = False,
) -> None:
    from vllm_musa.engine_plan import cli, runtime

    monkeypatch.setattr(
        runtime,
        "collect_environment_identity",
        lambda device_count, software_names=None: _live_environment(
            target,
            evidence_drift=evidence_drift,
        ),
    )
    monkeypatch.setattr(
        cli,
        "_runtime_catalog",
        lambda include_tuning: [
            _definition(32).to_document(),
            _definition(64).to_document(),
        ],
    )

    def measure(self, context, missing):
        measurements.append(
            (context.case.rows, tuple(item.tactic_id for item in missing))
        )
        self.physical_measurements += 1
        return {
            item.tactic_id: CandidateMeasurement(
                values_ms=(1.0,),
                status=ObservationStatus.PASSED,
                correctness="passed",
                provenance=(("runner", "fake"),),
            )
            for item in missing
        }

    monkeypatch.setattr(FusedAddRmsNormCollector, "measure_missing", measure)


def _collect(
    *,
    definitions: tuple[TacticDefinition, ...],
    target: PlanTarget | None = None,
    cache: TimingCache | None = None,
    calls: list[tuple[int, tuple[str, ...]]] | None = None,
    seal: bool = True,
    measurement_identity: dict[str, object] | None = None,
) -> tuple[dict[str, object], object]:
    calls = calls if calls is not None else []

    def measure(context, missing):
        calls.append((context.case.rows, tuple(item.tactic_id for item in missing)))
        return {
            item.tactic_id: CandidateMeasurement(
                values_ms=(
                    0.8 if int(item.choice) == 32 and context.case.rows == 32 else 1.0,
                )
                * 5,
                status=ObservationStatus.PASSED,
                correctness="passed",
                provenance=(("runner", "fake"),),
            )
            for item in missing
        }

    return collect_operation(
        target=target or _target(),
        definitions=definitions,
        cases=tuple(_case(rows) for rows in (1, 2, 4, 8, 16, 32, 64)),
        measure_missing=measure,
        software_dependencies=(
            "driver",
            "musa",
            "torch",
            "torch-musa",
            "vllm",
            "vllm-musa",
        ),
        source_dependencies=("vllm", "vllm-musa"),
        existing_cache=cache,
        provenance={"test": "autotuner"},
        measurement_identity=measurement_identity,
        seal=seal,
    )


def _runtime_decision_variant(provider_fingerprint: str) -> dict[str, object]:
    implementation = runtime_decision_implementation(OPERATION)
    assert implementation is not None
    fingerprint = implementation.fingerprint(provider_fingerprint)
    timing_document, _ = _collect(
        definitions=(
            _definition(32, fingerprint=fingerprint),
            _definition(64, fingerprint=fingerprint),
        )
    )
    return build_plan_document(
        [timing_document],
        plan_id="runtime-decision-fingerprint-test",
    )["variants"][0]


def test_collect_reuses_exact_cache_without_calling_runner() -> None:
    definitions = (_definition(32), _definition(64))
    first_calls: list[tuple[int, tuple[str, ...]]] = []
    first_document, first_stats = _collect(
        definitions=definitions,
        calls=first_calls,
    )
    assert len(first_calls) == 7
    assert first_stats.cache_hits == 0
    assert first_stats.measured == 14

    def fail_if_measured(*args, **kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("cache hit attempted to measure")

    cache = TimingCache.from_document(first_document, require_fingerprint=True)
    second_document, second_stats = collect_operation(
        target=_target(),
        definitions=definitions,
        cases=tuple(_case(rows) for rows in (1, 2, 4, 8, 16, 32, 64)),
        measure_missing=fail_if_measured,
        software_dependencies=(
            "driver",
            "musa",
            "torch",
            "torch-musa",
            "vllm",
            "vllm-musa",
        ),
        source_dependencies=("vllm", "vllm-musa"),
        existing_cache=cache,
        provenance={"test": "autotuner"},
    )
    assert second_stats.cache_hits == 14
    assert second_stats.cache_misses == 0
    assert second_stats.measured == 0
    assert second_document == first_document


def test_mutable_collect_can_resume_before_sealing() -> None:
    definitions = (_definition(32), _definition(64))
    mutable, _ = _collect(definitions=definitions, seal=False)
    assert "fingerprint" not in mutable
    calls: list[tuple[int, tuple[str, ...]]] = []
    _, stats = _collect(
        definitions=definitions,
        cache=TimingCache.from_document(mutable, require_fingerprint=False),
        calls=calls,
        seal=False,
    )
    assert calls == []
    assert stats.cache_hits == 14


@pytest.mark.parametrize(
    "definitions,target",
    [
        (
            (_definition(32), _definition(64), _definition(128)),
            _target(),
        ),
        (
            (_definition(32, fingerprint="impl-v2"), _definition(64)),
            _target(),
        ),
        (
            (_definition(32), _definition(64)),
            _target(image_digest="sha256:new-compiler-image"),
        ),
    ],
)
def test_candidate_or_toolchain_changes_invalidate_affected_operation(
    definitions: tuple[TacticDefinition, ...],
    target: PlanTarget,
) -> None:
    original, _ = _collect(definitions=(_definition(32), _definition(64)))
    calls: list[tuple[int, tuple[str, ...]]] = []
    _, stats = _collect(
        definitions=definitions,
        target=target,
        cache=TimingCache.from_document(original, require_fingerprint=True),
        calls=calls,
    )
    assert len(calls) == 7
    assert stats.cache_hits == 0
    assert stats.invalidated == 14
    assert stats.cache_misses == 7 * len(definitions)


def test_description_only_change_reuses_measurements() -> None:
    original, _ = _collect(definitions=(_definition(32), _definition(64)))
    calls: list[tuple[int, tuple[str, ...]]] = []
    _, stats = _collect(
        definitions=(
            _definition(32, description="updated documentation"),
            _definition(64),
        ),
        cache=TimingCache.from_document(original, require_fingerprint=True),
        calls=calls,
    )
    assert calls == []
    assert stats.cache_hits == 14
    assert stats.invalidated == 0


def test_measurement_policy_change_invalidates_observations() -> None:
    original, _ = _collect(
        definitions=(_definition(32), _definition(64)),
        measurement_identity={"iterations": 5},
    )
    calls: list[tuple[int, tuple[str, ...]]] = []
    _, stats = _collect(
        definitions=(_definition(32), _definition(64)),
        cache=TimingCache.from_document(original, require_fingerprint=True),
        calls=calls,
        measurement_identity={"iterations": 10},
    )
    assert len(calls) == 7
    assert stats.cache_hits == 0
    assert stats.invalidated == 14


def test_threshold_selection_uses_bucket_mean_after_no_regression_gate() -> None:
    document, _ = _collect(definitions=(_definition(32), _definition(64)))
    plan = build_plan_document(
        [document],
        plan_id="autotune-test",
        policy=BuildPolicy(
            min_samples=3,
            min_speedup_pct=1.0,
            tie_tolerance_pct=0.1,
        ),
    )
    assert plan["variants"][0]["runtime_decisions"]["values"][OPERATION] == 32
    selection = plan["variants"][0]["selections"][0]
    assert selection["metric"] == "bucket_mean_of_medians_ms"
    assert selection["winner"].endswith(":32")


def test_threshold_runtime_decision_requires_positive_integer() -> None:
    assert (
        validate_runtime_decision_value(
            RuntimeDecision.MUSA_FUSED_ADD_RMSNORM_MIN_ROWS,
            32,
        )
        == 32
    )
    with pytest.raises(ValueError, match="positive integer"):
        validate_runtime_decision_value(
            RuntimeDecision.MUSA_FUSED_ADD_RMSNORM_MIN_ROWS,
            0,
        )


def test_tuning_state_override_is_scoped() -> None:
    configure_fused_add_rmsnorm_min_rows(DEFAULT_FUSED_ADD_RMSNORM_MIN_ROWS)
    with override_fused_add_rmsnorm_min_rows(32):
        assert get_fused_add_rmsnorm_min_rows() == 32
    assert get_fused_add_rmsnorm_min_rows() == DEFAULT_FUSED_ADD_RMSNORM_MIN_ROWS


def test_tuning_state_propagates_to_spawned_worker_import() -> None:
    configure_fused_add_rmsnorm_min_rows(32)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from vllm_musa.tuning import "
                "get_fused_add_rmsnorm_min_rows; "
                "print(get_fused_add_rmsnorm_min_rows())",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.stdout.strip().endswith("32")
    finally:
        configure_fused_add_rmsnorm_min_rows(DEFAULT_FUSED_ADD_RMSNORM_MIN_ROWS)


def test_live_catalog_can_include_threshold_candidates(monkeypatch) -> None:
    import vllm_musa.engine_plugins
    from vllm_musa.engine_plan.cli import _runtime_catalog
    from vllm_musa.tuning import FUSED_ADD_RMSNORM_THRESHOLD_CHOICES

    monkeypatch.setattr(
        vllm_musa.engine_plugins,
        "list_engine_ir_providers",
        lambda: (
            EngineIrProviderMetadata(
                operation="fused_add_rms_norm",
                provider="native",
                implementation_fingerprint="native-impl-v1",
            ),
            EngineIrProviderMetadata(
                operation="fused_add_rms_norm",
                provider="musa",
                implementation_fingerprint="musa-impl-v1",
            ),
        ),
    )
    catalog = _runtime_catalog(include_tuning=True)
    thresholds = [item for item in catalog if item["operation"] == OPERATION]
    assert [item["choice"] for item in thresholds] == list(
        FUSED_ADD_RMSNORM_THRESHOLD_CHOICES
    )
    assert {item["fallback_id"] for item in thresholds} == {
        "runtime.musa.fused_add_rms_norm.min_rows:64"
    }
    implementation = runtime_decision_implementation(OPERATION)
    assert implementation is not None
    assert {item["implementation_fingerprint"] for item in thresholds} == {
        implementation.fingerprint("musa-impl-v1")
    }
    moe = [
        item
        for item in catalog
        if item["operation"] == "musa.fused_moe.dispatch_policy"
    ]
    assert [item["choice"] for item in moe] == [
        "upstream",
        "gemv",
        "grouped_gemm",
    ]
    assert {item["fallback_id"] for item in moe} == {"runtime.musa.fused_moe:upstream"}


@pytest.mark.parametrize(
    ("section", "key", "value", "difference_path"),
    [
        (
            "hardware",
            "multiprocessor_count",
            79,
            "hardware.multiprocessor_count",
        ),
        ("software_versions", "driver", "5.3.0", "software.versions"),
        ("software_versions", "musa", "5.3.0", "software.versions"),
        ("software_versions", "mate", "0.2.0", "software.versions"),
        (
            "source_revisions",
            "vllm-musa",
            "sha-vllm-musa-drifted",
            "software.source_revisions",
        ),
    ],
    ids=(
        "hardware-bin",
        "driver",
        "runtime",
        "critical-package",
        "source-revision",
    ),
)
def test_live_identity_drift_fails_before_candidate_enumeration(
    monkeypatch,
    tmp_path,
    section: str,
    key: str,
    value: object,
    difference_path: str,
) -> None:
    from vllm_musa.engine_plan import cli, runtime

    environment = _live_environment(_target())
    identity_section = environment[section]
    assert isinstance(identity_section, dict)
    identity_section[key] = value
    monkeypatch.setattr(
        runtime,
        "collect_environment_identity",
        lambda device_count, software_names=None: environment,
    )

    def fail_if_catalogued(*args, **kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("identity drift reached candidate enumeration")

    monkeypatch.setattr(cli, "_runtime_catalog", fail_if_catalogued)

    with pytest.raises(AutoTuneRuntimeError, match=difference_path):
        collect_fused_add_rms_norm(_collect_args(tmp_path))

    assert not (tmp_path / "timing.json").exists()


@pytest.mark.parametrize(
    ("section", "key", "difference_path"),
    [
        ("software_versions", "torch-musa", "software.versions"),
        ("source_revisions", "vllm", "software.source_revisions"),
    ],
    ids=("missing-critical-package", "missing-source-revision"),
)
def test_missing_live_identity_fails_before_candidate_enumeration(
    monkeypatch,
    tmp_path,
    section: str,
    key: str,
    difference_path: str,
) -> None:
    from vllm_musa.engine_plan import cli, runtime

    environment = _live_environment(_target())
    identity_section = environment[section]
    assert isinstance(identity_section, dict)
    identity_section.pop(key)
    monkeypatch.setattr(
        runtime,
        "collect_environment_identity",
        lambda device_count, software_names=None: environment,
    )

    def fail_if_catalogued(*args, **kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("missing identity reached candidate enumeration")

    monkeypatch.setattr(cli, "_runtime_catalog", fail_if_catalogued)

    expected_error = (
        EnginePlanError if section == "source_revisions" else AutoTuneRuntimeError
    )
    expected_match = (
        "cannot verify live source revisions"
        if section == "source_revisions"
        else difference_path
    )
    with pytest.raises(expected_error, match=expected_match):
        collect_fused_add_rms_norm(_collect_args(tmp_path))

    assert not (tmp_path / "timing.json").exists()


def test_matching_live_identity_collects_and_ignores_evidence_only_drift(
    monkeypatch,
    tmp_path,
) -> None:
    measurements: list[tuple[int, tuple[str, ...]]] = []
    _install_fake_live_collect(
        monkeypatch,
        _target(),
        measurements,
        evidence_drift=True,
    )

    summary = collect_fused_add_rms_norm(_collect_args(tmp_path))

    assert measurements == [
        (
            32,
            (
                "runtime.musa.fused_add_rms_norm.min_rows:32",
                "runtime.musa.fused_add_rms_norm.min_rows:64",
            ),
        )
    ]
    assert summary["physical_measurements"] == 1
    assert summary["tuning_domain"] == FUSED_ADD_RMS_NORM_DOMAIN_ID
    timing = json.loads((tmp_path / "timing.json").read_text())
    assert timing["provenance"][TUNING_DOMAIN_PROVENANCE_KEY] == (
        FUSED_ADD_RMS_NORM_DOMAIN_ID
    )


def test_h4096_domain_collects_against_matching_target(
    monkeypatch,
    tmp_path,
) -> None:
    target = _target_with_hidden_size(4096)
    measurements: list[tuple[int, tuple[str, ...]]] = []
    _install_fake_live_collect(monkeypatch, target, measurements)
    args = _collect_args(tmp_path)
    args.domain = FUSED_ADD_RMS_NORM_H4096_DOMAIN_ID
    Path(args.target).write_text(json.dumps(target.to_document()), encoding="utf-8")

    summary = collect_fused_add_rms_norm(args)

    assert measurements[0][0] == 32
    assert summary["tuning_domain"] == FUSED_ADD_RMS_NORM_H4096_DOMAIN_ID
    timing = json.loads((tmp_path / "timing.json").read_text())
    assert timing["target"]["model"]["hidden_size"] == 4096
    assert timing["provenance"][TUNING_DOMAIN_PROVENANCE_KEY] == (
        FUSED_ADD_RMS_NORM_H4096_DOMAIN_ID
    )


def test_h4096_domain_rejects_h5120_target() -> None:
    with pytest.raises(PlanningArtifactError, match="requires 4096"):
        validate_tuning_domain_target(
            FUSED_ADD_RMS_NORM_H4096_DOMAIN_ID,
            _target(),
        )


def test_collect_rejects_out_of_envelope_rows_before_measurement(
    monkeypatch,
    tmp_path,
) -> None:
    target = _target()
    measurements: list[tuple[int, tuple[str, ...]]] = []
    _install_fake_live_collect(monkeypatch, target, measurements)
    args = _collect_args(tmp_path)
    args.rows = [8193]

    with pytest.raises(AutoTuneRuntimeError, match="target token envelope"):
        collect_fused_add_rms_norm(args)
    assert measurements == []


def test_collect_rejects_resume_from_another_domain(
    monkeypatch,
    tmp_path,
) -> None:
    target = _target()
    measurements: list[tuple[int, tuple[str, ...]]] = []
    _install_fake_live_collect(monkeypatch, target, measurements)
    args = _collect_args(tmp_path)
    collect_fused_add_rms_norm(args)
    resume_document = json.loads((tmp_path / "timing.json").read_text())
    resume_document["provenance"][TUNING_DOMAIN_PROVENANCE_KEY] = (
        FUSED_ADD_RMS_NORM_H4096_DOMAIN_ID
    )
    resume_path = tmp_path / "wrong-domain-resume.json"
    resume_path.write_text(json.dumps(resume_document), encoding="utf-8")
    args.resume = str(resume_path)
    args.output = str(tmp_path / "resumed.json")
    args.summary = str(tmp_path / "resumed-summary.json")
    measurements.clear()

    with pytest.raises(AutoTuneRuntimeError, match="does not match"):
        collect_fused_add_rms_norm(args)
    assert measurements == []


def test_collect_accepts_legacy_resume_without_domain_provenance(
    monkeypatch,
    tmp_path,
) -> None:
    target = _target()
    measurements: list[tuple[int, tuple[str, ...]]] = []
    _install_fake_live_collect(monkeypatch, target, measurements)
    args = _collect_args(tmp_path)
    collect_fused_add_rms_norm(args)
    resume_document = json.loads((tmp_path / "timing.json").read_text())
    resume_document["provenance"].pop(TUNING_DOMAIN_PROVENANCE_KEY)
    resume_path = tmp_path / "legacy-resume.json"
    resume_path.write_text(json.dumps(resume_document), encoding="utf-8")
    args.resume = str(resume_path)
    args.output = str(tmp_path / "resumed.json")
    args.summary = str(tmp_path / "resumed-summary.json")
    measurements.clear()

    summary = collect_fused_add_rms_norm(args)

    assert measurements == []
    assert summary["cache_misses"] == 0
    assert summary["cache_hits"] > 0


def test_fused_add_domains_cover_the_runtime_lowering_hidden_sizes() -> None:
    assert set(FUSED_ADD_RMS_NORM_DOMAIN_HIDDEN_SIZES.values()) == set(
        FUSED_ADD_RMSNORM_TUNED_HIDDEN_SIZES
    )


def test_hidden_size_capability_does_not_hash_or_coerce_symbolic_values() -> None:
    class SymbolicPredicate:
        def __init__(self, values: tuple[int, ...]) -> None:
            self.values = values

        def __or__(self, other):
            return SymbolicPredicate(self.values + other.values)

        def __bool__(self):
            raise AssertionError("symbolic predicate must not be coerced to bool")

    class UnhashableSymbolicDimension:
        __hash__ = None

        def __eq__(self, other):
            return SymbolicPredicate((other,))

    result = is_fused_add_rmsnorm_tuned_hidden_size(UnhashableSymbolicDimension())

    assert result.values == tuple(sorted(FUSED_ADD_RMSNORM_TUNED_HIDDEN_SIZES))


def test_autotune_domains_cli_exposes_operator_contracts(capsys) -> None:
    from vllm_musa.engine_plan.cli import main

    assert main(["autotune", "domains"]) == 0
    document = json.loads(capsys.readouterr().out)

    assert [item["id"] for item in document["domains"]] == [
        FUSED_ADD_RMS_NORM_H4096_DOMAIN_ID,
        FUSED_ADD_RMS_NORM_DOMAIN_ID,
        FUSED_MOE_FP8_BLOCK_DOMAIN_ID,
    ]
    assert all("model_identity_policy" in item for item in document["domains"])


def test_runtime_profile_is_not_a_tuning_domain_alias() -> None:
    with pytest.raises(PlanningArtifactError, match="unknown tuning domain"):
        get_tuning_domain("qwen3.moe")


def test_live_identity_probes_optional_software_keys(monkeypatch) -> None:
    from vllm_musa.engine_plan import runtime

    document = _target().to_document()
    document["software"]["versions"]["mcc"] = "4.3.0"
    target = PlanTarget.from_document(document)
    requested: list[set[str]] = []

    def collect_environment_identity(*, device_count, software_names=None):
        requested.append(set(software_names or ()))
        return _live_environment(target)

    monkeypatch.setattr(
        runtime,
        "collect_environment_identity",
        collect_environment_identity,
    )

    assert runtime.diff_environment_identity(target) == ()
    assert requested == [set(dict(target.software.versions))]
    assert "mcc" in requested[0]


def test_missing_build_manifest_has_actionable_fail_closed_error(
    monkeypatch,
) -> None:
    from vllm_musa.engine_plan import runtime

    environment = _live_environment(_target())
    environment["source_revisions"] = {}
    monkeypatch.setenv(
        runtime.ENGINE_BUILD_MANIFEST_ENV,
        "/task/evidence/build-manifest.json",
    )
    monkeypatch.setattr(
        runtime,
        "collect_environment_identity",
        lambda device_count, software_names=None: environment,
    )

    with pytest.raises(
        EnginePlanError,
        match=(
            "MUSA_ENGINE_BUILD_MANIFEST.*build-manifest.json.*"
            "musa.engine_build.v1.*vllm-musa"
        ),
    ):
        runtime.diff_environment_identity(_target())


def test_extra_source_revision_key_fails_as_unverifiable(monkeypatch) -> None:
    from vllm_musa.engine_plan import runtime

    document = _target().to_document()
    document["software"]["source_revisions"]["mate"] = "sha-mate"
    target = PlanTarget.from_document(document)

    def fail_if_probed(*args, **kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("unverifiable source keys reached the live probe")

    monkeypatch.setattr(runtime, "collect_environment_identity", fail_if_probed)

    with pytest.raises(EnginePlanError, match="unverifiable.*mate"):
        runtime.diff_environment_identity(target)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("model", "model_id"), "Qwen/Qwen3-different"),
        (("workload", "graph_mode"), "NONE"),
        (("software", "versions", "driver"), "5.3.0"),
        (("software", "source_revisions", "vllm"), "sha-vllm-drifted"),
    ],
    ids=("model", "workload", "software", "source"),
)
def test_resume_target_drift_fails_before_catalog_or_measurement(
    monkeypatch,
    tmp_path,
    path: tuple[str, ...],
    value: object,
) -> None:
    from vllm_musa.engine_plan import cli, runtime

    current = _target()
    resume_document, _ = _collect(
        definitions=(_definition(32), _definition(64)),
        target=current,
        seal=False,
    )
    nested = resume_document["target"]
    for component in path[:-1]:
        nested = nested[component]
    nested[path[-1]] = value
    resume_path = tmp_path / "resume.json"
    resume_path.write_text(json.dumps(resume_document), encoding="utf-8")
    monkeypatch.setattr(
        runtime,
        "collect_environment_identity",
        lambda device_count, software_names=None: _live_environment(current),
    )

    def fail_if_catalogued(*args, **kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("resume drift reached candidate enumeration")

    monkeypatch.setattr(cli, "_runtime_catalog", fail_if_catalogued)

    with pytest.raises(
        PlanningArtifactError,
        match="resume target does not match.*refusing to reuse or measure",
    ):
        collect_fused_add_rms_norm(_collect_args(tmp_path, resume=str(resume_path)))

    assert not (tmp_path / "timing.json").exists()


def test_resume_allows_evidence_only_drift_and_reuses_without_measurement(
    monkeypatch,
    tmp_path,
) -> None:
    current = _target()
    first_measurements: list[tuple[int, tuple[str, ...]]] = []
    _install_fake_live_collect(monkeypatch, current, first_measurements)
    first_args = _collect_args(tmp_path, output_name="first.json")
    first_summary = collect_fused_add_rms_norm(first_args)
    assert first_summary["physical_measurements"] == 1

    resume_document = json.loads((tmp_path / "first.json").read_text())
    resume_document["target"]["hardware"]["device_uuid"] = "GPU-resume-evidence-only"
    resume_document["target"]["hardware"]["device_count"] = 8
    resume_path = tmp_path / "resume-evidence-only.json"
    resume_path.write_text(json.dumps(resume_document), encoding="utf-8")
    second_measurements: list[tuple[int, tuple[str, ...]]] = []
    _install_fake_live_collect(monkeypatch, current, second_measurements)

    summary = collect_fused_add_rms_norm(
        _collect_args(
            tmp_path,
            output_name="second.json",
            resume=str(resume_path),
        )
    )

    assert second_measurements == []
    assert summary["cache_hits"] == 2
    assert summary["measured"] == 0
    assert summary["physical_measurements"] == 0


def test_resume_image_digest_change_invalidates_instead_of_false_cache_hit(
    monkeypatch,
    tmp_path,
) -> None:
    original = _target(image_digest="sha256:old")
    first_measurements: list[tuple[int, tuple[str, ...]]] = []
    _install_fake_live_collect(monkeypatch, original, first_measurements)
    collect_fused_add_rms_norm(
        _collect_args(
            tmp_path,
            target_value=original,
            output_name="first.json",
        )
    )

    current = _target(image_digest="sha256:new")
    second_measurements: list[tuple[int, tuple[str, ...]]] = []
    _install_fake_live_collect(monkeypatch, current, second_measurements)
    summary = collect_fused_add_rms_norm(
        _collect_args(
            tmp_path,
            target_value=current,
            output_name="second.json",
            resume=str(tmp_path / "first.json"),
        )
    )

    assert len(second_measurements) == 1
    assert summary["cache_hits"] == 0
    assert summary["invalidated"] == 2
    assert summary["measured"] == 2


def test_runtime_decision_tactics_match_live_implementation(monkeypatch) -> None:
    import vllm_musa.engine_plugins
    from vllm_musa.engine_plan import runtime

    variant = _runtime_decision_variant("musa-impl-v1")

    calls: list[tuple[str, str]] = []

    def find_provider(operation: str, provider: str) -> EngineIrProviderMetadata:
        calls.append((operation, provider))
        return EngineIrProviderMetadata(
            operation=operation,
            provider=provider,
            implementation_fingerprint="musa-impl-v1",
        )

    monkeypatch.setattr(
        vllm_musa.engine_plugins,
        "find_engine_ir_provider",
        find_provider,
    )

    assert runtime._tactic_registry_differences(variant) == ()
    assert calls == [("fused_add_rms_norm", "musa")]


def test_runtime_decision_tactics_fail_closed_across_live_source_drift(
    monkeypatch,
) -> None:
    import vllm_musa.engine_plugins
    from vllm_musa.engine_plan import runtime

    variant = _runtime_decision_variant("musa-impl-v1")
    monkeypatch.setattr(
        vllm_musa.engine_plugins,
        "find_engine_ir_provider",
        lambda operation, provider: EngineIrProviderMetadata(
            operation=operation,
            provider=provider,
            implementation_fingerprint="musa-impl-v2",
        ),
    )

    differences = runtime._tactic_registry_differences(variant)

    assert len(differences) == 2
    assert all("expected implementation=" in item for item in differences)
    assert any(
        item.startswith("tactic.runtime.musa.fused_add_rms_norm.min_rows:32:")
        for item in differences
    )
    assert any(
        item.startswith("tactic.runtime.musa.fused_add_rms_norm.min_rows:64:")
        for item in differences
    )


def test_runtime_decision_tactics_fail_closed_without_live_provider(
    monkeypatch,
) -> None:
    import vllm_musa.engine_plugins
    from vllm_musa.engine_plan import runtime

    variant = _runtime_decision_variant("musa-impl-v1")
    monkeypatch.setattr(
        vllm_musa.engine_plugins,
        "find_engine_ir_provider",
        lambda operation, provider: None,
    )

    differences = runtime._tactic_registry_differences(variant)

    assert len(differences) == 2
    assert all(
        item.endswith("provider_not_available:fused_add_rms_norm:musa")
        for item in differences
    )


def test_runtime_decision_tactics_fail_closed_without_identity_recipe(
    monkeypatch,
) -> None:
    from vllm_musa.engine_plan import runtime

    variant = _runtime_decision_variant("musa-impl-v1")
    monkeypatch.setattr(
        runtime,
        "runtime_decision_implementation",
        lambda operation: None,
    )

    differences = runtime._tactic_registry_differences(variant)

    assert len(differences) == 2
    assert all(
        item.endswith("runtime_decision_implementation_not_registered")
        for item in differences
    )


def test_fused_runner_implements_tunable_runner_contract() -> None:
    def function(*args):
        return args

    runner = FusedAddRmsNormRunner(
        runner_id="fake",
        implementation_fingerprint="fake-v1",
        function=function,
    )
    assert runner.supports(_case(32))
    assert runner.build_callable() is function


def test_collector_uses_exact_inclusive_vllm_compile_range(monkeypatch) -> None:
    observed: list[tuple[str, int, int]] = []

    class FakeRange:
        def __init__(self, start: int, end: int) -> None:
            self.start = start
            self.end = end

    @contextmanager
    def fake_pass_context(compile_range):
        observed.append(("enter", compile_range.start, compile_range.end))
        yield
        observed.append(("exit", compile_range.start, compile_range.end))

    fake_inductor_pass = types.ModuleType("vllm.compilation.passes.inductor_pass")
    fake_inductor_pass.pass_context = fake_pass_context
    fake_config_utils = types.ModuleType("vllm.config.utils")
    fake_config_utils.Range = FakeRange
    monkeypatch.setitem(
        sys.modules,
        "vllm.compilation.passes.inductor_pass",
        fake_inductor_pass,
    )
    monkeypatch.setitem(sys.modules, "vllm.config.utils", fake_config_utils)

    def function(*args):
        return args

    collector = FusedAddRmsNormCollector(
        FusedAddRmsNormCollectConfig(
            warmups=1,
            iterations=1,
            cold_cache_bytes=0,
            graph_capture=False,
        )
    )
    collector._runner_cache = {
        "jit": FusedAddRmsNormRunner(
            runner_id="jit",
            implementation_fingerprint="fake-v1",
            function=function,
        )
    }
    fake_torch = types.SimpleNamespace(compile=lambda fn, **kwargs: fn)
    monkeypatch.setattr(collector, "_torch", lambda: fake_torch)

    compiled, compile_context = collector._compile("jit", 32)
    assert compiled is function
    with compile_context:
        assert observed == [("enter", 32, 32)]
    assert observed == [("enter", 32, 32), ("exit", 32, 32)]


def test_graph_runner_uses_musa_graph_context_for_capture_teardown(monkeypatch) -> None:
    events: list[str] = []

    class FakeTensor:
        def clone(self):
            return self

        def copy_(self, other):
            return self

        def detach(self):
            return self

    class FakeGraph:
        def capture_begin(self):  # pragma: no cover - regression tripwire
            raise AssertionError("manual graph capture bypasses MUSA teardown")

        def capture_end(self):  # pragma: no cover - regression tripwire
            raise AssertionError("manual graph capture bypasses MUSA teardown")

        def replay(self):
            events.append("replay")

    class FakeEvent:
        def record(self):
            return None

        def elapsed_time(self, other):
            return 1.0

    fake_graph = FakeGraph()

    @contextmanager
    def graph_context(graph):
        assert graph is fake_graph
        events.append("capture-enter")
        yield
        events.append("capture-exit")

    fake_musa = types.SimpleNamespace(
        MUSAGraph=lambda: fake_graph,
        graph=graph_context,
        synchronize=lambda: None,
        Event=lambda **kwargs: FakeEvent(),
    )
    fake_torch = types.SimpleNamespace(musa=fake_musa)
    tensor = FakeTensor()
    collector = FusedAddRmsNormCollector(
        FusedAddRmsNormCollectConfig(
            warmups=1,
            iterations=1,
            cold_cache_bytes=0,
            graph_capture=True,
        )
    )
    monkeypatch.setattr(collector, "_torch", lambda: fake_torch)
    monkeypatch.setattr(
        collector,
        "_compile",
        lambda runner_id, rows: (
            lambda x, residual, weight: (x, residual),
            nullcontext(),
        ),
    )

    samples, _ = collector._run_runner("jit", _case(32), (tensor, tensor, tensor))

    assert samples == (1.0,)
    assert events == ["capture-enter", "capture-exit", "replay"]
