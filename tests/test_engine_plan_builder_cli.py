# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).parents[1]
PACKAGE_ROOT = REPO_ROOT
sys.path.insert(0, str(PACKAGE_ROOT))

from vllm_musa.engine_plan.artifacts import (  # noqa: E402
    BenchmarkCase,
    PlanningArtifactError,
    TimingCache,
    TimingCacheBuilder,
    merge_timing_cache_documents,
    seal_timing_cache_document,
)
from vllm_musa.engine_plan.core import (  # noqa: E402
    EnginePlanError,
    parse_plan_document,
    seal_plan_document,
)
from vllm_musa.engine_plan.importers import (  # noqa: E402
    import_operator_integration_campaign,
)
from vllm_musa.engine_plan.planner import (  # noqa: E402
    BuildPolicy,
    build_plan_document,
    explain_plan,
    inspect_plan,
    load_json_document,
)
from vllm_musa.runtime_plan.declarative import (  # noqa: E402
    declarative_profile_identity,
)


def timing_document(
    *,
    candidate_median: float = 0.8,
    fallback_median: float = 1.0,
    candidate_samples: int = 9,
    candidate_status: str = "passed",
    candidate_correctness: str = "passed",
) -> dict:
    def benchmark_case(rows: int) -> dict:
        return BenchmarkCase.create(
            operation="fused_add_rms_norm",
            phase="operator",
            batch_size=1,
            tokens=rows,
            rows=rows,
            hidden_size=4096,
            dtype="bfloat16",
        ).to_document()

    observations = []
    for rows in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
        observations.extend(
            [
                {
                    "tactic_id": "vllm.ir.fused_add_rms_norm:native",
                    "status": "passed",
                    "metric": "median_ms",
                    "values": [fallback_median] * 9,
                    "correctness": "passed",
                    "case": benchmark_case(rows),
                    "provenance": {
                        "case": f"native-display-name-{rows}",
                        "stage": "native_chain",
                    },
                },
                {
                    "tactic_id": "vllm.ir.fused_add_rms_norm:musa",
                    "status": candidate_status if rows == 1 else "passed",
                    "metric": "median_ms",
                    "values": [candidate_median] * candidate_samples,
                    "correctness": (candidate_correctness if rows == 1 else "passed"),
                    "case": benchmark_case(rows),
                    "provenance": {
                        "case": f"candidate-display-name-{rows}",
                        "stage": "musa_provider_chain",
                    },
                },
            ]
        )
    return {
        "schema_version": "musa.engine_timing.v2",
        "target": {
            "model": {
                "profile": "qwen3.text_generation",
                "architecture": "Qwen3ForCausalLM",
                "model_id": "Qwen/Qwen3-8B",
                "hidden_size": 4096,
                "dtype": "bfloat16",
                "quantization": "none",
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
            },
            "hardware": {
                "platform": "musa",
                "device_name": "MTT S5000",
                "device_uuid": "GPU-test-s5000-64mp",
                "device_capability": "3.1",
                "multiprocessor_count": 64,
                "device_count": 1,
            },
            "software": {
                "versions": {
                    "driver": "5.2.0-server",
                    "flash-attn-3": "0.2.4+musa",
                    "flash-mla": "0.2.4+musa",
                    "mate": "0.2.4",
                    "mthreads-ml-py": "2.4.1",
                    "musa": "5.2.0",
                    "torch": "2.11.0.post1",
                    "torch-musa": "2.11.0.post1",
                    "torchada": "0.1.81",
                    "vllm": "0.24.0",
                    "vllm-musa": "0.1.25.dev0",
                },
                "source_revisions": {
                    "vllm": "ee0da84ab",
                    "vllm-musa": "96a30cac6",
                },
                "image_digest": "sha256:test-image",
            },
            "workload": {
                "phase": "serving",
                "batch_size": {"min": 1, "max": 256},
                "tokens": {"min": 1, "max": 8192},
                "max_model_len": 32768,
                "max_num_batched_tokens": 8192,
                "max_num_seqs": 256,
                "compile_mode": "VLLM_COMPILE",
                "graph_mode": "FULL_AND_PIECEWISE",
            },
        },
        "catalog": [
            {
                "id": "vllm.ir.fused_add_rms_norm:native",
                "kind": "vllm_ir_provider",
                "operation": "fused_add_rms_norm",
                "choice": "native",
                "fallback_id": "vllm.ir.fused_add_rms_norm:native",
                "implementation_fingerprint": "sha256:native-impl",
                "description": "vLLM native safe fallback",
            },
            {
                "id": "vllm.ir.fused_add_rms_norm:musa",
                "kind": "vllm_ir_provider",
                "operation": "fused_add_rms_norm",
                "choice": "musa",
                "fallback_id": "vllm.ir.fused_add_rms_norm:native",
                "implementation_fingerprint": "sha256:musa-impl",
                "description": "registered vLLM-MUSA provider",
            },
        ],
        "observations": observations,
        "provenance": {
            "kind": "unit-test-fixture",
            "method": "compiled synthetic chain",
        },
    }


def build_plan(document: dict | None = None, **policy_kwargs):
    return parse_plan_document(
        build_plan_document(
            [document or timing_document()],
            plan_id="qwen3-fused-rmsnorm-test",
            policy=BuildPolicy(**policy_kwargs),
        )
    )


def operator_integration_campaign() -> dict:
    return {
        "schema_version": "vllm_musa.operator_integration.campaign.v1",
        "metadata": {
            "operator": "fused_add_rms_norm",
            "device": "MTT S5000",
            "dtype": "bfloat16",
            "rows": 2500,
            "hidden": 4096,
        },
        "results": [
            {
                "name": "fused_add_rms_norm[rows=2500,hidden=4096]",
                "correctness": {
                    "musa_provider_chain": {"close": [True, True]},
                },
                "stages": {
                    "native_chain": {
                        "graph_replay_1": {
                            "event_us": {"samples": [1000.0, 1010.0, 990.0]}
                        }
                    },
                    "musa_provider_chain": {
                        "graph_replay_1": {
                            "event_us": {"samples": [800.0, 810.0, 790.0]}
                        }
                    },
                },
            }
        ],
    }


def test_timing_cache_seal_and_strict_round_trip():
    sealed = seal_timing_cache_document(timing_document())

    parsed = TimingCache.from_document(sealed, require_fingerprint=True)

    assert parsed.fingerprint.startswith("sha256:")
    assert parsed.target.hardware.multiprocessor_count == 64
    assert {item.choice for item in parsed.catalog} == {"musa", "native"}
    assert seal_timing_cache_document(sealed) == sealed


def test_documentation_timing_template_tracks_current_schema():
    docs = (REPO_ROOT / "docs" / "vllm_musa" / "engine_plan.md").read_text(
        encoding="utf-8"
    )

    assert "python -m vllm_musa.engine_plan" in docs
    assert "timing evidence" in docs
    assert not (
        REPO_ROOT
        / "example"
        / "engine_plan"
        / "examples"
        / "qwen3-fused-rmsnorm-timing-template.json"
    ).exists()


def test_benchmark_side_builder_avoids_hand_written_timing_json():
    template = timing_document()
    collector = TimingCacheBuilder.from_documents(
        target=template["target"],
        catalog=template["catalog"],
        provenance={"run": "collector-test"},
    )
    for observation in template["observations"]:
        collector.add_observation(
            observation["tactic_id"],
            observation["values"],
            case=observation["case"],
            status=observation["status"],
            metric=observation["metric"],
            correctness=observation["correctness"],
            provenance=observation["provenance"],
        )

    sealed = collector.build()

    assert TimingCache.from_document(sealed, require_fingerprint=True)
    assert build_plan(sealed).variants[0]["selections"][0]["winner"].endswith(":musa")


def test_builder_rejects_candidate_with_one_regressing_evidence_bucket():
    document = timing_document()
    for observation in document["observations"]:
        observation["values"] = [
            1.0 if observation["tactic_id"].endswith(":native") else 0.1
        ] * 9
    document["observations"][2]["values"] = [10.0] * 9
    document["observations"][3]["values"] = [11.0] * 9

    selection = build_plan(document).variants[0]["selections"][0]

    assert selection["winner"] == "vllm.ir.fused_add_rms_norm:native"
    assert selection["rejected"] == [
        {
            "tactic_id": "vllm.ir.fused_add_rms_norm:musa",
            "reason": "bucket_regression_above_tie_tolerance",
            "median": 0.1,
            "samples": 126,
        }
    ]


def test_builder_keeps_legacy_unbound_case_evidence_on_native_fallback():
    document = timing_document()
    document["schema_version"] = "musa.engine_timing.v1"
    document["target"]["model"].pop("hidden_size")
    for observation in document["observations"]:
        observation.pop("case")

    selection = build_plan(document).variants[0]["selections"][0]

    assert selection["winner"] == "vllm.ir.fused_add_rms_norm:native"
    assert selection["rejected"][0]["reason"] == "bucket_provenance_missing"


def test_human_case_names_do_not_define_coverage_buckets():
    document = timing_document()
    for observation in document["observations"]:
        observation["provenance"]["case"] = "same-display-name"

    selection = build_plan(document).variants[0]["selections"][0]

    assert selection["winner"] == "vllm.ir.fused_add_rms_norm:musa"
    assert selection["coverage"]["status"] == "complete"


def test_typed_candidate_and_fallback_case_sets_must_match():
    document = timing_document()
    candidate = document["observations"][3]
    candidate["case"] = BenchmarkCase.create(
        operation="fused_add_rms_norm",
        phase="operator",
        batch_size=1,
        tokens=3,
        rows=3,
        hidden_size=4096,
        dtype="bfloat16",
    ).to_document()
    candidate["provenance"]["case"] = document["observations"][2]["provenance"]["case"]

    selection = build_plan(document).variants[0]["selections"][0]

    assert selection["winner"] == "vllm.ir.fused_add_rms_norm:native"
    assert selection["rejected"][0]["reason"] == "bucket_coverage_mismatch"


def test_typed_benchmark_case_fingerprint_tampering_fails_closed():
    document = timing_document()
    document["observations"][1]["case"]["operator_shape"]["rows"] = 3

    with pytest.raises(PlanningArtifactError, match="case.fingerprint mismatch"):
        seal_timing_cache_document(document)


def test_benchmark_case_can_bind_directly_to_actual_musa_tensor_shape():
    tensor = SimpleNamespace(
        shape=(128, 4096),
        device="musa:0",
        dtype="torch.bfloat16",
    )

    case = BenchmarkCase.from_2d_tensor(
        operation="fused_add_rms_norm",
        tensor=tensor,
    )

    assert case.rows == 128
    assert case.tokens == 128
    assert case.hidden_size == 4096
    assert case.dtype == "bfloat16"


def test_benchmark_case_rejects_non_musa_tensor_binding():
    tensor = SimpleNamespace(
        shape=(128, 4096),
        device="cuda:0",
        dtype="torch.bfloat16",
    )

    with pytest.raises(PlanningArtifactError, match="MUSA device"):
        BenchmarkCase.from_2d_tensor(
            operation="fused_add_rms_norm",
            tensor=tensor,
        )


def test_operator_integration_importer_reuses_existing_harness_evidence():
    template = timing_document()
    catalog = deepcopy(template["catalog"])
    catalog.append(
        {
            "id": "vllm.ir.fused_add_rms_norm:vllm_c",
            "kind": "vllm_ir_provider",
            "operation": "fused_add_rms_norm",
            "choice": "vllm_c",
            "fallback_id": "vllm.ir.fused_add_rms_norm:native",
            "implementation_fingerprint": "sha256:vllm-c-impl",
            "description": "unmeasured third provider",
        }
    )

    imported = import_operator_integration_campaign(
        operator_integration_campaign(),
        target=template["target"],
        catalog=catalog,
        operation="fused_add_rms_norm",
    )
    parsed = TimingCache.from_document(imported, require_fingerprint=True)
    plan = build_plan(imported)
    selection = plan.variants[0]["selections"][0]

    assert len(parsed.observations) == 2
    assert dict(parsed.provenance)["importer"].endswith("campaign.v1")
    assert "vllm_c" in dict(parsed.provenance)["excluded_unmeasured_tactics"]
    assert selection["winner"] == "vllm.ir.fused_add_rms_norm:native"
    assert imported["schema_version"] == "musa.engine_timing.v1"
    assert dict(parsed.provenance)["coverage_binding"] == "unbound:campaign-v1"
    assert selection["rejected"][0]["reason"] == "bucket_provenance_missing"


def test_operator_importer_binds_hidden_size_to_target_model():
    template = timing_document()
    campaign = operator_integration_campaign()
    campaign["metadata"]["hidden"] = 1

    with pytest.raises(PlanningArtifactError, match="target.model.hidden_size"):
        import_operator_integration_campaign(
            campaign,
            target=template["target"],
            catalog=template["catalog"],
            operation="fused_add_rms_norm",
        )


def test_selection_rejects_cases_measured_at_another_hidden_size():
    document = timing_document()
    for observation in document["observations"]:
        case = observation["case"]
        observation["case"] = BenchmarkCase.create(
            operation=case["operation"],
            phase=case["workload_point"]["phase"],
            batch_size=case["workload_point"]["batch_size"],
            tokens=case["workload_point"]["tokens"],
            rows=case["operator_shape"]["rows"],
            hidden_size=1,
            dtype=case["dtype"],
        ).to_document()

    selection = build_plan(document).variants[0]["selections"][0]

    assert selection["winner"] == "vllm.ir.fused_add_rms_norm:native"
    assert selection["rejected"][0]["reason"] == "bucket_provenance_invalid"


def test_timing_cache_fingerprint_drift_fails_closed():
    sealed = seal_timing_cache_document(timing_document())
    sealed["target"]["hardware"]["multiprocessor_count"] = 56

    with pytest.raises(PlanningArtifactError, match="fingerprint mismatch"):
        TimingCache.from_document(sealed, require_fingerprint=True)


@pytest.mark.parametrize("operation", ("build", "merge"))
def test_timing_cache_consumers_reject_stale_existing_fingerprint(operation):
    sealed = seal_timing_cache_document(timing_document())
    sealed["observations"][1]["values"][0] = 0.01

    with pytest.raises(PlanningArtifactError, match="fingerprint mismatch"):
        if operation == "build":
            build_plan_document([sealed], plan_id="stale-cache")
        else:
            merge_timing_cache_documents([sealed])


def test_timing_cache_requires_driver_and_runtime_invalidation_versions():
    document = timing_document()
    document["target"]["software"]["versions"].pop("driver")

    with pytest.raises(PlanningArtifactError, match="missing cache-invalidation"):
        seal_timing_cache_document(document)


def test_timing_cache_rejects_unresolved_runtime_invalidation_versions():
    document = timing_document()
    document["target"]["software"]["versions"]["mate"] = "unknown"

    with pytest.raises(PlanningArtifactError, match="unresolved.*mate"):
        seal_timing_cache_document(document)


def test_timing_cache_rejects_unresolved_optional_software_versions():
    document = timing_document()
    document["target"]["software"]["versions"]["mcc"] = "unknown"

    with pytest.raises(PlanningArtifactError, match="unresolved.*mcc"):
        seal_timing_cache_document(document)


def test_timing_cache_requires_exact_source_revisions():
    document = timing_document()
    document["target"]["software"]["source_revisions"].pop("vllm-musa")

    with pytest.raises(PlanningArtifactError, match="missing runtime keys"):
        seal_timing_cache_document(document)


def test_timing_cache_merge_preserves_repeated_samples_for_builder():
    first = timing_document(candidate_samples=2, candidate_median=0.8)
    first["provenance"]["run"] = "a"
    for observation in first["observations"]:
        observation["provenance"].update({"case": "same", "run": "a"})
    second = timing_document(candidate_samples=2, candidate_median=0.7)
    second["provenance"]["run"] = "b"
    for observation in second["observations"]:
        observation["provenance"].update({"case": "same", "run": "b"})

    merged = merge_timing_cache_documents([first, second])
    parsed = TimingCache.from_document(merged, require_fingerprint=True)
    plan = build_plan(merged)
    selection = plan.variants[0]["selections"][0]

    candidate_observations = [
        item
        for item in parsed.observations
        if item.tactic_id == "vllm.ir.fused_add_rms_norm:musa"
    ]
    assert len(candidate_observations) == 28
    assert sum(item.samples for item in candidate_observations) == 56
    assert selection["winner"] == "vllm.ir.fused_add_rms_norm:musa"
    assert selection["samples"] == 56


def test_timing_cache_merge_preserves_one_failed_observation_as_ineligible():
    failed = timing_document(candidate_status="failed")
    passed = timing_document()
    failed["provenance"]["run"] = "failed"
    passed["provenance"]["run"] = "passed"

    merged = merge_timing_cache_documents([failed, passed])
    selection = build_plan(merged).variants[0]["selections"][0]

    assert selection["winner"] == "vllm.ir.fused_add_rms_norm:native"
    assert selection["rejected"][0]["reason"] == "measurement_failed"


def test_builder_selects_fastest_eligible_tactic_and_native_fallback():
    plan = build_plan()
    variant = plan.variants[0]
    selection = variant["selections"][0]

    assert selection["winner"] == "vllm.ir.fused_add_rms_norm:musa"
    assert selection["fallback"] == "vllm.ir.fused_add_rms_norm:native"
    assert selection["reason"] == "lowest_eligible_median"
    assert selection["speedup_pct"] == pytest.approx(20.0)
    profile_config = declarative_profile_identity("qwen3.text_generation")
    assert profile_config is not None
    assert variant["runtime_decisions"] == {
        "profile": "qwen3.text_generation",
        "profile_config_id": profile_config[0],
        "profile_config_fingerprint": profile_config[1],
        "values": {"vllm.ir_op_priority": {"fused_add_rms_norm": ["musa", "native"]}},
    }


@pytest.mark.parametrize(
    ("document", "reason"),
    [
        (
            timing_document(candidate_samples=2),
            "samples_2_below_3",
        ),
        (
            timing_document(candidate_status="failed"),
            "measurement_failed",
        ),
        (
            timing_document(candidate_correctness="failed"),
            "correctness_failed",
        ),
    ],
)
def test_builder_rejects_incomplete_or_failed_candidate(document, reason):
    plan = build_plan(document)
    selection = plan.variants[0]["selections"][0]

    assert selection["winner"] == "vllm.ir.fused_add_rms_norm:native"
    assert any(item["reason"] == reason for item in selection["rejected"])


def test_builder_uses_fallback_below_minimum_speedup():
    plan = build_plan(
        timing_document(candidate_median=0.995),
        min_speedup_pct=1.0,
    )

    selection = plan.variants[0]["selections"][0]
    assert selection["winner"] == "vllm.ir.fused_add_rms_norm:native"
    assert selection["reason"] == "fallback_required:below_min_speedup"


def test_builder_rejects_non_serving_request_bucket_variants():
    decode = timing_document()
    prefill = deepcopy(decode)
    prefill["target"]["workload"]["phase"] = "prefill"
    prefill["target"]["workload"]["tokens"] = {"min": 128, "max": 8192}

    with pytest.raises(PlanningArtifactError, match="complete serving workload"):
        build_plan_document(
            [decode, prefill],
            plan_id="ambiguous",
        )


def test_builder_selects_typed_runtime_decision():
    document = timing_document()
    document["catalog"] = [
        {
            "id": "decision:qwen3.qk_rope_kv_presplit:false",
            "kind": "runtime_decision",
            "operation": "qwen3.qk_rope_kv_presplit",
            "choice": False,
            "fallback_id": "decision:qwen3.qk_rope_kv_presplit:false",
            "implementation_fingerprint": "sha256:runtime-plan",
            "description": "safe negative fallback",
        },
        {
            "id": "decision:qwen3.qk_rope_kv_presplit:true",
            "kind": "runtime_decision",
            "operation": "qwen3.qk_rope_kv_presplit",
            "choice": True,
            "fallback_id": "decision:qwen3.qk_rope_kv_presplit:false",
            "implementation_fingerprint": "sha256:runtime-plan",
            "description": "in-tree runtime-plan candidate",
        },
    ]
    for observation in document["observations"]:
        is_candidate = observation["tactic_id"].endswith(":musa")
        case = observation["case"]
        observation["tactic_id"] = (
            "decision:qwen3.qk_rope_kv_presplit:true"
            if is_candidate
            else "decision:qwen3.qk_rope_kv_presplit:false"
        )
        observation["case"] = BenchmarkCase.create(
            operation="qwen3.qk_rope_kv_presplit",
            phase=case["workload_point"]["phase"],
            batch_size=case["workload_point"]["batch_size"],
            tokens=case["workload_point"]["tokens"],
            rows=case["operator_shape"]["rows"],
            hidden_size=case["operator_shape"]["hidden_size"],
            dtype=case["dtype"],
        ).to_document()

    plan = build_plan_document([document], plan_id="typed-runtime-decision")

    assert plan["variants"][0]["runtime_decisions"]["values"] == {
        "qwen3.qk_rope_kv_presplit": True
    }


def test_runtime_decision_overlay_accepts_host_catalog_value():
    document = build_plan_document(
        [timing_document()],
        plan_id="typed-overlay",
        runtime_decisions={
            "qwen3.text_generation": {
                "musa.fused_add_rms_norm.min_rows": 256,
            }
        },
    )

    values = document["variants"][0]["runtime_decisions"]["values"]

    assert values["musa.fused_add_rms_norm.min_rows"] == 256


def test_runtime_decision_overlay_rejects_fixed_decision_at_build_time():
    with pytest.raises(EnginePlanError, match="cannot project a fixed"):
        build_plan_document(
            [timing_document()],
            plan_id="fixed-decision",
            runtime_decisions={
                "qwen3.text_generation": {
                    "hybrid.kv_cache_pool_layout": "shared",
                }
            },
        )


def test_runtime_decision_overlay_rejects_unknown_host_decision():
    with pytest.raises(EnginePlanError, match="unknown runtime decision"):
        build_plan_document(
            [timing_document()],
            plan_id="unknown-decision",
            runtime_decisions={
                "qwen3.text_generation": {"decision.unregistered": True}
            },
        )


def test_runtime_decision_overlay_rejects_invalid_host_value():
    with pytest.raises(EnginePlanError, match="must be a positive integer"):
        build_plan_document(
            [timing_document()],
            plan_id="invalid-decision-value",
            runtime_decisions={
                "qwen3.text_generation": {
                    "musa.fused_add_rms_norm.min_rows": 0,
                }
            },
        )


def test_runtime_decision_overlay_rejects_wrong_profile_family():
    with pytest.raises(EnginePlanError, match="not supported by runtime profile"):
        build_plan_document(
            [timing_document()],
            plan_id="wrong-profile-family",
            runtime_decisions={
                "qwen3.text_generation": {
                    "deepseek_v4.native_sparse_indexer": True,
                }
            },
        )


def test_qwen3_decision_does_not_match_qwen35_profile_prefix():
    document = timing_document()
    document["target"]["model"]["profile"] = "qwen3.5_3.6.text_generation"

    with pytest.raises(EnginePlanError, match="not supported by runtime profile"):
        build_plan_document(
            [document],
            plan_id="qwen3-is-not-qwen35",
            runtime_decisions={
                "qwen3.5_3.6.text_generation": {
                    "qwen3.qk_rope_kv_presplit": True,
                }
            },
        )


def test_runtime_decision_overlay_rejects_unknown_profile():
    with pytest.raises(PlanningArtifactError, match="without a timing variant"):
        build_plan_document(
            [timing_document()],
            plan_id="unknown-profile",
            runtime_decisions={"qwen2.text_generation": {"decision.bool": False}},
        )


def test_runtime_decision_overlay_cannot_override_timing_selection():
    with pytest.raises(PlanningArtifactError, match="must not override"):
        build_plan_document(
            [timing_document()],
            plan_id="provider-override",
            runtime_decisions={
                "qwen3.text_generation": {
                    "vllm.ir_op_priority": {"fused_add_rms_norm": ["native"]}
                }
            },
        )


def test_runtime_decision_overlay_rejects_non_json_plan_value():
    with pytest.raises(EnginePlanError, match="JSON-like runtime-decision value"):
        build_plan_document(
            [timing_document()],
            plan_id="invalid-decision-value",
            runtime_decisions={"qwen3.text_generation": {"decision.invalid": None}},
        )


def test_runtime_decision_overlay_rejects_non_finite_value():
    with pytest.raises(PlanningArtifactError, match="finite JSON values"):
        build_plan_document(
            [timing_document()],
            plan_id="non-finite-decision-value",
            runtime_decisions={
                "qwen3.text_generation": {"decision.invalid": float("nan")}
            },
        )


def test_inspect_and_explain_show_selection_and_context_mismatch():
    plan = build_plan()
    summary = inspect_plan(plan)
    matching_context = timing_document()["target"]

    selected = explain_plan(plan, runtime_target=matching_context)
    mismatched_context = deepcopy(matching_context)
    mismatched_context["hardware"]["multiprocessor_count"] = 56
    fallback = explain_plan(plan, runtime_target=mismatched_context)

    assert summary["variants"][0]["selections"][0]["winner"].endswith(":musa")
    assert selected["runtime_decision"]["status"] == "selected"
    assert fallback["runtime_decision"]["status"] == "fallback"
    differences = fallback["runtime_decision"]["variants"][0]["differences"]
    assert any("multiprocessor_count" in item for item in differences)


def test_cli_build_inspect_validate_and_explain(tmp_path):
    timings = tmp_path / "timings.json"
    plan = tmp_path / "plan.json"
    context = tmp_path / "context.json"
    decisions = tmp_path / "decisions.json"
    timings.write_text(json.dumps(timing_document()), encoding="utf-8")
    context.write_text(json.dumps(timing_document()["target"]), encoding="utf-8")
    decisions.write_text(
        json.dumps(
            {
                "qwen3.text_generation": {
                    "qwen3.qk_rope_kv_presplit": False,
                }
            }
        ),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(PACKAGE_ROOT),
    }

    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_musa.engine_plan",
            "build",
            "--timings",
            str(timings),
            "--output",
            str(plan),
            "--plan-id",
            "cli-test",
            "--runtime-decisions",
            str(decisions),
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    inspected = subprocess.run(
        [sys.executable, "-m", "vllm_musa.engine_plan", "inspect", str(plan)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    validated = subprocess.run(
        [sys.executable, "-m", "vllm_musa.engine_plan", "validate", str(plan)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    explained = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_musa.engine_plan",
            "explain",
            str(plan),
            "--context",
            str(context),
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert built.returncode == 0, built.stderr
    built_plan = parse_plan_document(json.loads(plan.read_text(encoding="utf-8")))
    assert (
        built_plan.variants[0]["runtime_decisions"]["values"][
            "qwen3.qk_rope_kv_presplit"
        ]
        is False
    )
    assert inspected.returncode == 0, inspected.stderr
    assert validated.returncode == 0, validated.stderr
    assert explained.returncode == 0, explained.stderr
    assert "fused_add_rms_norm" in inspected.stdout
    assert "Decide vllm.ir_op_priority" in inspected.stdout
    assert "Decide qwen3.qk_rope_kv_presplit: false" in inspected.stdout
    assert "Decision: selected" in explained.stdout


def test_cli_cache_seal_and_merge(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    sealed = tmp_path / "sealed.json"
    merged = tmp_path / "merged.json"
    first_document = timing_document()
    first_document["provenance"]["run"] = "a"
    second_document = timing_document()
    second_document["provenance"]["run"] = "b"
    first.write_text(json.dumps(first_document), encoding="utf-8")
    second.write_text(json.dumps(second_document), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(PACKAGE_ROOT)}

    seal_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_musa.engine_plan",
            "cache",
            "seal",
            str(first),
            "--output",
            str(sealed),
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    merge_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_musa.engine_plan",
            "cache",
            "merge",
            str(first),
            str(second),
            "--output",
            str(merged),
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert seal_result.returncode == 0, seal_result.stderr
    assert merge_result.returncode == 0, merge_result.stderr
    assert TimingCache.from_document(
        json.loads(sealed.read_text()), require_fingerprint=True
    )
    merged_cache = TimingCache.from_document(
        json.loads(merged.read_text()), require_fingerprint=True
    )
    assert len(merged_cache.observations) == 56


def test_target_command_auto_probes_environment_and_keeps_scope_explicit(
    monkeypatch,
    tmp_path,
):
    from vllm_musa.engine_plan import runtime
    from vllm_musa.engine_plan.cli import main

    output = tmp_path / "target.json"
    monkeypatch.setattr(
        runtime,
        "collect_environment_identity",
        lambda device_count: {
            "hardware": {
                "platform": "musa",
                "device_name": "MTT S5000",
                "device_uuid": "GPU-test-s5000-64mp",
                "device_capability": "3.1",
                "multiprocessor_count": 64,
                "device_count": device_count,
            },
            "software_versions": timing_document()["target"]["software"]["versions"],
        },
    )

    result = main(
        [
            "target",
            "--profile",
            "qwen3.text_generation",
            "--architecture",
            "Qwen3ForCausalLM",
            "--model-id",
            "Qwen/Qwen3-8B",
            "--hidden-size",
            "4096",
            "--dtype",
            "bfloat16",
            "--source-revision",
            "vllm=ee0da84ab",
            "--source-revision",
            "vllm-musa=3f8cefeb2",
            "--image-digest",
            "sha256:test-image",
            "--phase",
            "serving",
            "--batch-min",
            "1",
            "--batch-max",
            "1",
            "--tokens-min",
            "2500",
            "--tokens-max",
            "2500",
            "--max-model-len",
            "32768",
            "--max-num-batched-tokens",
            "8192",
            "--max-num-seqs",
            "256",
            "--compile-mode",
            "VLLM_COMPILE",
            "--graph-mode",
            "FULL_AND_PIECEWISE",
            "--cudagraph-capture-sizes",
            "1",
            "2",
            "4",
            "--output",
            str(output),
        ]
    )

    target = json.loads(output.read_text())
    assert result == 0
    assert target["hardware"]["multiprocessor_count"] == 64
    assert target["model"]["hidden_size"] == 4096
    assert target["software"]["source_revisions"]["vllm-musa"] == "3f8cefeb2"
    assert target["workload"]["cudagraph_capture_sizes"] == [1, 2, 4]


def test_repo_local_engine_plan_import_uses_host_package():
    env = {**os.environ, "PYTHONPATH": str(PACKAGE_ROOT)}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import vllm_musa.engine_plan; "
                "assert 'vllm_musa.engine_plan' in sys.modules"
            ),
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_schema_v5_plan_fingerprint_drift_fails_closed():
    document = build_plan_document(
        [timing_document()],
        plan_id="fingerprint-test",
    )
    document["variants"][0]["selections"][0]["reason"] = "tampered"

    with pytest.raises(EnginePlanError, match="fingerprint mismatch"):
        parse_plan_document(document)


def test_schema_v5_provider_decision_must_be_derived_from_sealed_selection():
    document = build_plan_document(
        [timing_document()],
        plan_id="decision-consistency",
    )
    document.pop("fingerprint")
    document["variants"][0]["runtime_decisions"]["values"]["vllm.ir_op_priority"][
        "fused_add_rms_norm"
    ] = ["native"]

    with pytest.raises(EnginePlanError, match="does not match.*selections"):
        seal_plan_document(document)


def test_schema_v5_selection_must_be_recomputed_from_correctness_evidence():
    document = build_plan_document(
        [timing_document(candidate_correctness="failed")],
        plan_id="correctness-selection",
    )
    document.pop("fingerprint")
    selection = document["variants"][0]["selections"][0]
    selection["winner"] = "vllm.ir.fused_add_rms_norm:musa"

    with pytest.raises(EnginePlanError, match="selection.*does not match sealed"):
        seal_plan_document(document)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: BuildPolicy(min_speedup_pct=float("nan")),
        lambda: BuildPolicy(tie_tolerance_pct=float("inf")),
    ),
)
def test_build_policy_rejects_non_finite_numbers(factory):
    with pytest.raises(PlanningArtifactError, match="finite"):
        factory()


def test_schema_v5_selection_rejects_nan_field():
    document = build_plan_document([timing_document()], plan_id="nan-selection")
    document.pop("fingerprint")
    document["variants"][0]["selections"][0]["winner_value"] = float("nan")

    with pytest.raises(EnginePlanError, match="finite"):
        seal_plan_document(document)


def test_json_loader_rejects_non_standard_nan_and_infinity():
    for literal in ("NaN", "Infinity", "-Infinity", "1e999"):
        with pytest.raises(EnginePlanError, match="finite JSON number"):
            load_json_document(f'{{"value": {literal}}}', source="fixture")


def test_cli_rejects_non_finite_json_input(tmp_path):
    source = tmp_path / "nan.json"
    output = tmp_path / "sealed.json"
    source.write_text('{"value": NaN}', encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_musa.engine_plan",
            "cache",
            "seal",
            str(source),
            "--output",
            str(output),
        ],
        env={**os.environ, "PYTHONPATH": str(PACKAGE_ROOT)},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "finite JSON number" in result.stderr
    assert not output.exists()


def _fake_runtime_config():
    class Config:
        pass

    priorities = SimpleNamespace(
        rms_norm=[],
        fused_add_rms_norm=[],
        gated_qkv_rms_norm_rope=[],
    )
    config = Config()
    config.kernel_config = SimpleNamespace(ir_op_priority=priorities)
    return config


def _matching_runtime_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Qwen3ForCausalLM"],
                hidden_size=4096,
            ),
            model="Qwen/Qwen3-8B",
            dtype="bfloat16",
            quantization=None,
            max_model_len=32768,
        ),
        quant_config=None,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=8192,
            max_num_seqs=256,
        ),
        compilation_config=SimpleNamespace(
            mode=SimpleNamespace(name="VLLM_COMPILE"),
            cudagraph_mode=SimpleNamespace(name="FULL_AND_PIECEWISE"),
        ),
    )


def test_active_musa_multiprocessor_count_uses_context_free_driver_query(
    monkeypatch,
):
    from vllm_musa.engine_plan import runtime

    calls = []

    def mu_init(flags):
        calls.append(("init", flags))
        return 0

    def mu_device_get(device_pointer, ordinal):
        calls.append(("device", ordinal))
        ctypes.cast(device_pointer, ctypes.POINTER(ctypes.c_int)).contents.value = 0
        return 0

    def mu_device_get_attribute(value_pointer, attribute, device):
        calls.append(("attribute", attribute, device.value))
        ctypes.cast(value_pointer, ctypes.POINTER(ctypes.c_int)).contents.value = 56
        return 0

    driver = SimpleNamespace(
        muInit=mu_init,
        muDeviceGet=mu_device_get,
        muDeviceGetAttribute=mu_device_get_attribute,
    )
    monkeypatch.setattr(
        runtime.ctypes,
        "CDLL",
        lambda library: driver if library == "libmusa.so" else None,
    )

    assert runtime._active_musa_multiprocessor_count() == 56
    assert calls == [
        ("init", 0),
        ("device", 0),
        ("attribute", 16, 0),
    ]


def test_hardware_identity_preserves_mtml_provenance_when_mp_query_fails(
    monkeypatch,
):
    from vllm_musa.engine_plan import runtime

    def success(*args):
        return 0

    def attribute_failure(*args):
        return 1

    driver = SimpleNamespace(
        muInit=success,
        muDeviceGet=success,
        muDeviceGetAttribute=attribute_failure,
    )
    mt_mgmt = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda index: f"device-{index}",
        nvmlDeviceGetName=lambda handle: b"MTT S5000",
        nvmlDeviceGetUUID=lambda handle: b"GPU-live-s5000",
        nvmlDeviceGetCudaComputeCapability=lambda handle: (3, 1),
        nvmlSystemGetDriverVersion=lambda: b"5.2.0-server",
    )
    monkeypatch.setattr(runtime.ctypes, "CDLL", lambda library: driver)
    monkeypatch.setitem(sys.modules, "pymtml", mt_mgmt)
    runtime._hardware_identity.cache_clear()
    try:
        identity = runtime._hardware_identity()
    finally:
        runtime._hardware_identity.cache_clear()

    assert identity == (
        "MTT S5000",
        "3.1",
        -1,
        "5.2.0-server",
        "GPU-live-s5000",
    )


def test_runtime_selector_matches_full_static_context_and_provider_uuid(monkeypatch):
    from vllm_musa.engine_plan import runtime

    plan = build_plan()
    expected_versions = timing_document()["target"]["software"]["versions"]
    monkeypatch.setattr(
        runtime,
        "_hardware_identity",
        lambda: (
            "MTT S5000",
            "3.1",
            64,
            "5.2.0-server",
            "GPU-test-s5000-64mp",
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_software_versions",
        lambda expected, driver_version: dict(expected_versions),
    )
    monkeypatch.setattr(
        runtime,
        "_runtime_source_revisions",
        lambda: dict(timing_document()["target"]["software"]["source_revisions"]),
    )
    monkeypatch.setattr(runtime, "_tactic_registry_differences", lambda variant: ())

    decision = runtime.select_runtime_variant(plan, _matching_runtime_config())

    assert decision.is_match
    assert decision.reason == "runtime_compatibility_key_early"


def test_runtime_selector_falls_back_across_runtime_or_source_drift(monkeypatch):
    from vllm_musa.engine_plan import runtime

    plan = build_plan()
    monkeypatch.setattr(
        runtime,
        "_hardware_identity",
        lambda: (
            "MTT S5000",
            "3.1",
            64,
            "5.3.0-server",
            "GPU-another-s5000-64mp",
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_software_versions",
        lambda expected, driver_version: {
            **dict(expected.target.software.versions),
            "torch_musa": "2.11.0.post1+musa5.3.0",
        },
    )
    monkeypatch.setattr(
        runtime,
        "_runtime_source_revisions",
        lambda: {"vllm-musa": "a-new-source-revision"},
    )
    monkeypatch.setattr(runtime, "_tactic_registry_differences", lambda variant: ())

    decision = runtime.select_runtime_variant(plan, _matching_runtime_config())

    assert not decision.is_match
    assert decision.reason == "no_matching_variant"
    assert any("software.versions" in item for item in decision.differences)
    assert any("software.source_revisions" in item for item in decision.differences)


def test_runtime_selector_falls_back_without_live_build_manifest(monkeypatch):
    from vllm_musa.engine_plan import runtime

    plan = build_plan()
    expected_versions = timing_document()["target"]["software"]["versions"]
    monkeypatch.setattr(
        runtime,
        "_hardware_identity",
        lambda: (
            "MTT S5000",
            "3.1",
            64,
            "5.2.0-server",
            "GPU-test-s5000-64mp",
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_software_versions",
        lambda expected, driver_version: dict(expected_versions),
    )
    monkeypatch.setattr(runtime, "_runtime_source_revisions", lambda: {})
    monkeypatch.setattr(runtime, "_tactic_registry_differences", lambda variant: ())

    decision = runtime.select_runtime_variant(plan, _matching_runtime_config())

    assert not decision.is_match
    assert decision.reason == "no_matching_variant"
    assert any("software.source_revisions" in item for item in decision.differences)


def test_runtime_selector_fails_closed_when_active_mp_probe_is_unavailable(
    monkeypatch,
):
    from vllm_musa.engine_plan import runtime

    plan = build_plan()
    expected_versions = timing_document()["target"]["software"]["versions"]
    monkeypatch.setattr(
        runtime,
        "_hardware_identity",
        lambda: (
            "MTT S5000",
            "3.1",
            -1,
            "5.2.0-server",
            "GPU-test-s5000-64mp",
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_software_versions",
        lambda expected, driver_version: dict(expected_versions),
    )
    monkeypatch.setattr(
        runtime,
        "_runtime_source_revisions",
        lambda: dict(timing_document()["target"]["software"]["source_revisions"]),
    )
    monkeypatch.setattr(runtime, "_tactic_registry_differences", lambda variant: ())

    decision = runtime.select_runtime_variant(plan, _matching_runtime_config())

    assert not decision.is_match
    assert decision.reason == "no_matching_variant"
    assert any("hardware.multiprocessor_count" in item for item in decision.differences)


def test_builder_rejects_narrow_static_workload_scope():
    document = timing_document()
    document["target"]["workload"].update(
        {
            "phase": "operator",
            "batch_size": {"min": 1, "max": 1},
            "tokens": {"min": 128, "max": 128},
        }
    )
    with pytest.raises(PlanningArtifactError, match="static_plan_requires_serving"):
        build_plan(document)


def test_builder_rejects_topology_runtime_cannot_activate():
    document = timing_document()
    document["target"]["model"]["tensor_parallel_size"] = 2

    with pytest.raises(PlanningArtifactError, match="supports TP1/PP1 only"):
        build_plan(document)


def test_schema_validator_rejects_narrow_static_workload_scope():
    document = build_plan_document([timing_document()], plan_id="narrow-schema")
    document.pop("fingerprint")
    timing = deepcopy(document["variants"][0]["timing_cache"])
    timing.pop("fingerprint")
    timing["target"]["workload"].update(
        {
            "phase": "operator",
            "batch_size": {"min": 1, "max": 1},
            "tokens": {"min": 128, "max": 128},
        }
    )
    document["variants"][0]["timing_cache"] = seal_timing_cache_document(timing)

    with pytest.raises(EnginePlanError, match="static serving workload"):
        seal_plan_document(document)


def test_plan_schema_version_requires_an_integer():
    document = build_plan_document([timing_document()], plan_id="float-schema")
    document.pop("fingerprint")
    document["schema_version"] = 4.0

    with pytest.raises(EnginePlanError, match="schema_version must be an integer"):
        seal_plan_document(document)


def test_final_runtime_validator_reuses_static_workload_scope_gate(monkeypatch):
    from vllm_musa.engine_plan import runtime

    plan = build_plan()
    variant = deepcopy(plan.variants[0])
    timing = deepcopy(variant["timing_cache"])
    timing.pop("fingerprint")
    timing["target"]["workload"].update(
        {
            "phase": "operator",
            "batch_size": {"min": 1, "max": 1},
            "tokens": {"min": 128, "max": 128},
        }
    )
    variant["timing_cache"] = seal_timing_cache_document(timing)
    expected_versions = timing_document()["target"]["software"]["versions"]
    monkeypatch.setattr(
        runtime,
        "_hardware_identity",
        lambda: (
            "MTT S5000",
            "0.0",
            64,
            "5.2.0-server",
            "GPU-test-s5000-64mp",
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_software_versions",
        lambda expected, driver_version: dict(expected_versions),
    )
    monkeypatch.setattr(
        runtime,
        "_runtime_source_revisions",
        lambda: dict(timing_document()["target"]["software"]["source_revisions"]),
    )
    monkeypatch.setattr(runtime, "_tactic_registry_differences", lambda candidate: ())

    _, differences = runtime.validate_runtime_variant(
        variant,
        _matching_runtime_config(),
    )

    assert any("static_plan_requires_serving" in item for item in differences)
    assert any("does_not_cover_max_num_batched_tokens" in item for item in differences)


def test_runtime_selector_requires_exact_model_id(monkeypatch):
    from vllm_musa.engine_plan import runtime

    plan = build_plan()
    expected_versions = timing_document()["target"]["software"]["versions"]
    monkeypatch.setattr(
        runtime,
        "_hardware_identity",
        lambda: (
            "MTT S5000",
            "0.0",
            64,
            "5.2.0-server",
            "GPU-test-s5000-64mp",
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_software_versions",
        lambda expected, driver_version: dict(expected_versions),
    )
    monkeypatch.setattr(
        runtime,
        "_runtime_source_revisions",
        lambda: dict(timing_document()["target"]["software"]["source_revisions"]),
    )
    monkeypatch.setattr(runtime, "_tactic_registry_differences", lambda variant: ())
    config = _matching_runtime_config()
    config.model_config.model = "Qwen/Another-Qwen3-8B"

    decision = runtime.select_runtime_variant(plan, config)

    assert not decision.is_match
    assert any("model.model_id" in item for item in decision.differences)


def test_runtime_selector_requires_exact_hidden_size(monkeypatch):
    from vllm_musa.engine_plan import runtime

    plan = build_plan()
    expected_versions = timing_document()["target"]["software"]["versions"]
    monkeypatch.setattr(
        runtime,
        "_hardware_identity",
        lambda: (
            "MTT S5000",
            "0.0",
            64,
            "5.2.0-server",
            "GPU-test-s5000-64mp",
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_software_versions",
        lambda expected, driver_version: dict(expected_versions),
    )
    monkeypatch.setattr(
        runtime,
        "_runtime_source_revisions",
        lambda: dict(timing_document()["target"]["software"]["source_revisions"]),
    )
    monkeypatch.setattr(runtime, "_tactic_registry_differences", lambda variant: ())
    config = _matching_runtime_config()
    config.model_config.hf_config.hidden_size = 1

    decision = runtime.select_runtime_variant(plan, config)

    assert not decision.is_match
    assert any("model.hidden_size" in item for item in decision.differences)


@pytest.mark.parametrize(
    ("tensor_parallel_size", "pipeline_parallel_size"),
    ((2, 1), (1, 2)),
)
def test_builder_rejects_multi_device_schema_v5_plan(
    tensor_parallel_size,
    pipeline_parallel_size,
):
    document = timing_document()
    document["target"]["model"]["tensor_parallel_size"] = tensor_parallel_size
    document["target"]["model"]["pipeline_parallel_size"] = pipeline_parallel_size
    document["target"]["hardware"]["device_count"] = (
        tensor_parallel_size * pipeline_parallel_size
    )
    with pytest.raises(PlanningArtifactError, match="supports TP1/PP1 only"):
        build_plan(document)


def test_runtime_target_does_not_copy_sealed_source_or_image_provenance(monkeypatch):
    from vllm_musa.engine_plan import runtime

    plan = build_plan()
    expected_versions = timing_document()["target"]["software"]["versions"]
    timing_cache = TimingCache.from_document(
        plan.variants[0]["timing_cache"],
        require_fingerprint=True,
    )
    monkeypatch.setattr(
        runtime,
        "_hardware_identity",
        lambda: (
            "MTT S5000",
            "0.0",
            64,
            "5.2.0-server",
            "GPU-test-s5000-64mp",
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_software_versions",
        lambda expected, driver_version: dict(expected_versions),
    )
    # The production image may carry a baked source manifest.  This test is
    # about not copying sealed evidence into a live target, so make the live
    # probe deterministic and explicitly empty rather than inheriting the
    # image's unrelated manifest.
    monkeypatch.setattr(runtime, "_runtime_source_revisions", lambda: {})

    target = runtime.collect_runtime_target(
        _matching_runtime_config(),
        timing_cache,
        final=False,
    )

    assert target["software"]["source_revisions"] == {}
    assert target["software"]["image_digest"] == "unknown"
    assert target["software"]["source_revisions"] != dict(
        timing_cache.target.software.source_revisions
    )


def test_runtime_selector_falls_back_on_s5000_mp_bin_change(monkeypatch):
    from vllm_musa.engine_plan import runtime

    plan = build_plan()
    expected_versions = timing_document()["target"]["software"]["versions"]
    monkeypatch.setattr(
        runtime,
        "_hardware_identity",
        lambda: (
            "MTT S5000",
            "0.0",
            56,
            "5.2.0-server",
            "GPU-different-s5000-56mp",
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_software_versions",
        lambda expected, driver_version: dict(expected_versions),
    )
    monkeypatch.setattr(
        runtime,
        "_runtime_source_revisions",
        lambda: dict(timing_document()["target"]["software"]["source_revisions"]),
    )
    monkeypatch.setattr(runtime, "_tactic_registry_differences", lambda variant: ())

    decision = runtime.select_runtime_variant(plan, _matching_runtime_config())

    assert not decision.is_match
    assert decision.reason == "no_matching_variant"
    assert any("multiprocessor_count" in item for item in decision.differences)


def test_runtime_selector_falls_back_on_live_tactic_fingerprint_change(monkeypatch):
    from vllm_musa.engine_plan import runtime

    plan = build_plan()
    expected_versions = timing_document()["target"]["software"]["versions"]
    monkeypatch.setattr(
        runtime,
        "_hardware_identity",
        lambda: (
            "MTT S5000",
            "0.0",
            64,
            "5.2.0-server",
            "GPU-test-s5000-64mp",
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_software_versions",
        lambda expected, driver_version: dict(expected_versions),
    )
    monkeypatch.setattr(
        runtime,
        "_runtime_source_revisions",
        lambda: dict(timing_document()["target"]["software"]["source_revisions"]),
    )
    monkeypatch.setattr(
        runtime,
        "_tactic_registry_differences",
        lambda variant: ("tactic.musa: implementation fingerprint changed",),
    )

    decision = runtime.select_runtime_variant(plan, _matching_runtime_config())

    assert not decision.is_match
    assert any("fingerprint changed" in item for item in decision.differences)


def test_schema_v5_runtime_applies_selected_variant_and_receipt(
    monkeypatch,
    tmp_path,
):
    from vllm_musa.engine_plan import plugin as runtime_plugin
    from vllm_musa.engine_plan.artifacts import runtime_key_fingerprint
    from vllm_musa.engine_plan.runtime import RuntimeVariantDecision
    from vllm_musa.engine_plugins import (
        ENGINE_PLAN_ENV,
        apply_engine_plugin_defaults,
        resolve_engine_plugin_runtime_decisions,
        validate_engine_plugin_runtime,
    )
    from vllm_musa.engine_plugins.registry import (
        _reset_engine_plugin_registry_for_test,
    )

    document = build_plan_document(
        [timing_document()],
        plan_id="runtime-v5",
        runtime_decisions={
            "qwen3.text_generation": {
                "qwen3.qk_rope_kv_presplit": False,
            }
        },
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    parsed = parse_plan_document(document)
    variant = dict(parsed.variants[0])
    runtime_target = deepcopy(timing_document()["target"])
    runtime_target["workload"]["cudagraph_capture_sizes"] = [1, 2, 4]
    monkeypatch.setenv(ENGINE_PLAN_ENV, str(path))
    monkeypatch.setattr(
        runtime_plugin,
        "select_runtime_variant",
        lambda plan, config: RuntimeVariantDecision(
            variant=variant,
            runtime_target=runtime_target,
            reason="runtime_compatibility_key_early",
            differences=(),
        ),
    )
    monkeypatch.setattr(
        runtime_plugin,
        "validate_runtime_variant",
        lambda selected, config: (runtime_target, ()),
    )
    _reset_engine_plugin_registry_for_test()
    runtime_plugin._PLUGIN._plans.clear()
    runtime_plugin.register()
    config = _fake_runtime_config()
    runtime_context = _matching_runtime_config()
    for name in (
        "model_config",
        "quant_config",
        "parallel_config",
        "scheduler_config",
        "compilation_config",
    ):
        setattr(config, name, getattr(runtime_context, name))
    config.compilation_config.cudagraph_capture_sizes = [1, 2, 4]

    application = apply_engine_plugin_defaults(config)
    state = runtime_plugin._PLUGIN._plans[id(config)]
    state.runtime_target["workload"]["cudagraph_capture_sizes"] = []
    state.variant["timing_cache"]["target"]["workload"]["cudagraph_capture_sizes"] = [
        1,
        2,
        4,
    ]
    runtime_plugin._PLUGIN.validate_resolved_cudagraph_config(config, application)
    original_graph_mode = config.compilation_config.cudagraph_mode
    config.compilation_config.cudagraph_mode = SimpleNamespace(name="FULL_DECODE_ONLY")
    with pytest.raises(EnginePlanError, match="mode differs"):
        runtime_plugin._PLUGIN.validate_resolved_cudagraph_config(
            config,
            application,
        )
    config.compilation_config.cudagraph_mode = original_graph_mode
    config.compilation_config.cudagraph_capture_sizes = [1, 4]
    with pytest.raises(EnginePlanError, match="capture ladder differs"):
        runtime_plugin._PLUGIN.validate_resolved_cudagraph_config(
            config,
            application,
        )
    config.compilation_config.cudagraph_capture_sizes = [1, 2, 4]
    decision_receipt = resolve_engine_plugin_runtime_decisions(config)
    from vllm_musa.runtime_plan import RuntimeDecision, resolve_runtime_plan

    materialized_plan = resolve_runtime_plan(config)
    # vLLM platform defaults may append providers after the plan's mandatory
    # native fallback. They are unreachable and must not invalidate the sealed
    # winner/fallback prefix.
    config.kernel_config.ir_op_priority.fused_add_rms_norm.append("vllm_c")
    runtime_plugin._PLUGIN._plans.clear()
    replay_application = runtime_plugin._PLUGIN.apply_config_defaults(config)
    receipt = validate_engine_plugin_runtime(config)

    assert replay_application.selected_variant == application.selected_variant
    assert application.selected_variant == variant["variant_id"]
    assert application.selected_tactics == ("vllm.ir.fused_add_rms_norm:musa",)
    assert application.fallback_reason == ""
    assert decision_receipt is not None
    assert dict(decision_receipt.decisions)["qwen3.qk_rope_kv_presplit"] is False
    assert dict(decision_receipt.decisions)["vllm.ir_op_priority"] == (
        ("fused_add_rms_norm", ("musa", "native")),
    )
    assert materialized_plan.value(RuntimeDecision.VLLM_IR_OP_PRIORITY) == (
        ("fused_add_rms_norm", ("musa", "native")),
    )
    assert not materialized_plan.enabled(RuntimeDecision.QWEN3_QK_ROPE_KV_PRESPLIT)
    assert config.kernel_config.ir_op_priority.fused_add_rms_norm == [
        "musa",
        "native",
        "vllm_c",
    ]
    assert receipt.selected_variant == application.selected_variant
    assert application.context_fingerprint == runtime_key_fingerprint(
        runtime_target,
        final=False,
    )
    assert receipt.context_fingerprint == runtime_key_fingerprint(
        runtime_target,
        final=True,
    )
    assert receipt.context_fingerprint != application.context_fingerprint
    _reset_engine_plugin_registry_for_test()
    runtime_plugin._PLUGIN._plans.clear()


def test_schema_v5_runtime_rejects_provider_inserted_before_sealed_fallback(
    monkeypatch,
    tmp_path,
):
    from vllm_musa.engine_plan import plugin as runtime_plugin
    from vllm_musa.engine_plan.runtime import RuntimeVariantDecision
    from vllm_musa.engine_plugins import (
        ENGINE_PLAN_ENV,
        EnginePluginActivationError,
        apply_engine_plugin_defaults,
        validate_engine_plugin_runtime,
    )
    from vllm_musa.engine_plugins.registry import (
        _reset_engine_plugin_registry_for_test,
    )

    document = build_plan_document([timing_document()], plan_id="runtime-order-drift")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    parsed = parse_plan_document(document)
    variant = dict(parsed.variants[0])
    runtime_target = timing_document()["target"]
    monkeypatch.setenv(ENGINE_PLAN_ENV, str(path))
    monkeypatch.setattr(
        runtime_plugin,
        "select_runtime_variant",
        lambda plan, config: RuntimeVariantDecision(
            variant=variant,
            runtime_target=runtime_target,
            reason="runtime_compatibility_key_early",
            differences=(),
        ),
    )
    monkeypatch.setattr(
        runtime_plugin,
        "validate_runtime_variant",
        lambda selected, config: (runtime_target, ()),
    )
    _reset_engine_plugin_registry_for_test()
    runtime_plugin._PLUGIN._plans.clear()
    runtime_plugin.register()
    config = _fake_runtime_config()
    apply_engine_plugin_defaults(config)
    config.kernel_config.ir_op_priority.fused_add_rms_norm.insert(1, "vllm_c")

    with pytest.raises(EnginePluginActivationError, match="expected prefix"):
        validate_engine_plugin_runtime(config)

    _reset_engine_plugin_registry_for_test()
    runtime_plugin._PLUGIN._plans.clear()


def test_schema_v5_runtime_no_match_is_audited_baseline_fallback(
    monkeypatch,
    tmp_path,
):
    from vllm_musa.engine_plan import plugin as runtime_plugin
    from vllm_musa.engine_plan.artifacts import runtime_key_fingerprint
    from vllm_musa.engine_plan.runtime import RuntimeVariantDecision
    from vllm_musa.engine_plugins import (
        ENGINE_PLAN_ENV,
        apply_engine_plugin_defaults,
        validate_engine_plugin_runtime,
    )
    from vllm_musa.engine_plugins.registry import (
        _reset_engine_plugin_registry_for_test,
    )

    document = build_plan_document([timing_document()], plan_id="runtime-fallback")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    runtime_target = timing_document()["target"]
    monkeypatch.setenv(ENGINE_PLAN_ENV, str(path))
    selection_phases = []

    def select_variant(plan, config, *, final=False):
        selection_phases.append(final)
        difference = (
            "hardware.multiprocessor_count: expected=64, actual=56-final"
            if final
            else "hardware.multiprocessor_count: expected=64, actual=56"
        )
        return RuntimeVariantDecision(
            variant=None,
            runtime_target=runtime_target,
            reason="no_matching_variant",
            differences=(difference,),
        )

    monkeypatch.setattr(runtime_plugin, "select_runtime_variant", select_variant)
    _reset_engine_plugin_registry_for_test()
    runtime_plugin._PLUGIN._plans.clear()
    runtime_plugin.register()
    config = _fake_runtime_config()

    application = apply_engine_plugin_defaults(config)
    receipt = validate_engine_plugin_runtime(config)

    assert application.applied_settings == ()
    assert application.selected_variant == ""
    assert application.selected_tactics == ()
    assert "no_matching_variant" in application.fallback_reason
    assert "actual=56-final" in receipt.fallback_reason
    assert selection_phases == [False, True]
    assert application.context_fingerprint == runtime_key_fingerprint(
        runtime_target,
        final=False,
    )
    assert receipt.context_fingerprint == runtime_key_fingerprint(
        runtime_target,
        final=True,
    )
    assert receipt.context_fingerprint != application.context_fingerprint
    assert config.kernel_config.ir_op_priority.fused_add_rms_norm == []
    _reset_engine_plugin_registry_for_test()
    runtime_plugin._PLUGIN._plans.clear()
