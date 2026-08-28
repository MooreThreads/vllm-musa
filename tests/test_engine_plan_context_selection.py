# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType, SimpleNamespace

import pytest

from vllm_musa.engine_plan.artifacts import (
    BenchmarkCase,
    PlanningArtifactError,
    PlanTarget,
    TacticDefinition,
    TacticKind,
    TimingCache,
    TimingCacheBuilder,
    merge_timing_cache_documents,
    runtime_key_document,
    seal_timing_cache_document,
)
from vllm_musa.engine_plan.cli import _human_inspect
from vllm_musa.engine_plan.core import EnginePlanError, parse_plan_document
from vllm_musa.engine_plan.planner import (
    BuildPolicy,
    build_plan_document,
    inspect_plan,
)
from vllm_musa.engine_plan.tuning_domains import (
    FUSED_ADD_RMS_NORM_DOMAIN_ID,
    FUSED_MOE_FP8_BLOCK_DOMAIN_ID,
    TUNING_DOMAIN_PROVENANCE_KEY,
    TuningDomain,
    get_tuning_domain,
    infer_tuning_domain,
    resolve_timing_cache_domain,
    validate_tuning_domain_target,
)

OPERATION = "musa.fused_moe.dispatch_policy"
BACKENDS = ("upstream", "gemv", "grouped_gemm")
ROUTES = ("balanced", "unique_random", "hot")
SEEDS = (11, 23, 37)


def _target() -> PlanTarget:
    return PlanTarget.from_document(
        {
            "model": {
                "profile": "qwen3.moe",
                "architecture": "Qwen3MoeForCausalLM",
                "model_id": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
                "hidden_size": 4096,
                "dtype": "bfloat16",
                "quantization": "fp8",
                "tensor_parallel_size": 8,
                "pipeline_parallel_size": 1,
            },
            "hardware": {
                "platform": "musa",
                "device_name": "MTT S5000",
                "device_uuid": "GPU-test",
                "device_capability": "3.1",
                "multiprocessor_count": 60,
                "device_count": 8,
            },
            "software": {
                "versions": {
                    "driver": "3.3.5-server",
                    "flash-attn-3": "0.2.4",
                    "flash-mla": "0.2.4",
                    "mate": "0.2.4",
                    "mthreads-ml-py": "2.4.1",
                    "musa": "5.2.0",
                    "torch": "2.11.0.post1",
                    "torch-musa": "2.11.0.post1",
                    "torchada": "0.1.83",
                    "vllm": "0.24.0",
                    "vllm-musa": "0.1.24",
                },
                "source_revisions": {
                    "vllm": "ee0da84ab",
                    "vllm-musa": "test-sha",
                },
                "image_digest": "sha256:test-image",
            },
            "workload": {
                "phase": "serving",
                "batch_size": {"min": 1, "max": 256},
                "tokens": {"min": 1, "max": 64},
                "max_model_len": 32768,
                "max_num_batched_tokens": 64,
                "max_num_seqs": 256,
                "compile_mode": "VLLM_COMPILE",
                "graph_mode": "FULL_AND_PIECEWISE",
                "cudagraph_capture_sizes": [1, 2, 4],
            },
        }
    )


def _shape(*, graph_mode: str = "eager", local_experts: int = 128) -> dict:
    return {
        "device_capability": [3, 1],
        "multiprocessor_count": 60,
        "local_experts": local_experts,
        "w1_output_size": 512,
        "w2_input_size": 256,
        "hidden_size": 4096,
        "top_k": 8,
        "block_n": 128,
        "block_k": 128,
        "activation": "silu",
        "expert_parallel": False,
        "hidden_dtype": "torch.bfloat16",
        "weight_dtype": "torch.float8_e4m3fn",
        "scale_dtype": "torch.float32",
        "w1_scale_shape": [128, 4, 32],
        "w2_scale_shape": [128, 32, 2],
        "gemv_block": "auto",
        "graph_mode": graph_mode,
    }


def _definitions() -> tuple[TacticDefinition, ...]:
    prefix = "runtime.musa.fused_moe"
    fallback_id = f"{prefix}:upstream"
    return tuple(
        TacticDefinition(
            tactic_id=f"{prefix}:{backend}",
            kind=TacticKind.RUNTIME_DECISION,
            operation=OPERATION,
            choice=backend,
            fallback_id=fallback_id,
            implementation_fingerprint=f"sha256:{backend}-impl",
            description=f"fused-MoE {backend} backend",
        )
        for backend in BACKENDS
    )


def _case(
    tokens: int,
    *,
    route: str,
    seed: int,
    graph_mode: str = "eager",
    bucket: tuple[int, int] | None = None,
) -> BenchmarkCase:
    minimum, maximum = bucket or (tokens, tokens)
    return BenchmarkCase.create_contextual(
        operation=OPERATION,
        phase="operator",
        batch_size=1,
        tokens=tokens,
        operator_shape=_shape(graph_mode=graph_mode),
        token_bucket_min=minimum,
        token_bucket_max=maximum,
        evidence_context={"route_mode": route, "seed": seed},
        dtype="bfloat16",
    )


def _timing_document(
    *,
    gemv_winners: frozenset[int] = frozenset({1, 2}),
    grouped_winners: frozenset[int] = frozenset({4}),
    gemv_regression: tuple[int, str, int] | None = None,
    routes: tuple[str, ...] = ROUTES,
    seeds: tuple[int, ...] = SEEDS,
    graph_mode: str = "eager",
    target_graph_mode: str = "FULL_AND_PIECEWISE",
    target_max_num_seqs: int = 256,
) -> dict:
    target_document = _target().to_document()
    target_document["workload"]["graph_mode"] = target_graph_mode
    target_document["workload"]["max_num_seqs"] = target_max_num_seqs
    builder = TimingCacheBuilder(
        target=PlanTarget.from_document(target_document),
        catalog=_definitions(),
        provenance={"run": "context-selection-test"},
    )
    for tokens in (1, 2, 3, 4):
        for route in routes:
            for seed in seeds:
                case = _case(tokens, route=route, seed=seed, graph_mode=graph_mode)
                values = {
                    "upstream": 1.0,
                    "gemv": 0.5 if tokens in gemv_winners else 1.2,
                    "grouped_gemm": 0.6 if tokens in grouped_winners else 1.3,
                }
                if gemv_regression == (tokens, route, seed):
                    values["gemv"] = 1.1
                for definition in _definitions():
                    builder.add_observation(
                        definition.tactic_id,
                        [values[str(definition.choice)]] * 5,
                        case=case,
                        provenance={"runner": str(definition.choice)},
                    )
    return builder.build()


def test_fused_moe_domain_accepts_deepseek_v4_fp8_runtime_name() -> None:
    target_document = _target().to_document()
    target_document["model"]["quantization"] = "deepseek_v4_fp8"

    validate_tuning_domain_target(
        FUSED_MOE_FP8_BLOCK_DOMAIN_ID,
        PlanTarget.from_document(target_document),
    )


def _bf16_timing_document() -> dict:
    target_document = _target().to_document()
    target_document["model"].update(
        {
            "model_id": "example/Future-BF16-MoE",
            "quantization": "none",
        }
    )
    target = PlanTarget.from_document(target_document)
    builder = TimingCacheBuilder(
        target=target,
        catalog=_definitions(),
        provenance={"run": "legacy-bf16-context-selection-test"},
    )
    bf16_shape = {
        **_shape(),
        "block_n": 0,
        "block_k": 0,
        "weight_dtype": "torch.bfloat16",
        "scale_dtype": "none",
        "w1_scale_shape": [],
        "w2_scale_shape": [],
    }
    for tokens in (1, 2, 3, 4):
        for route in ROUTES:
            for seed in SEEDS:
                case = BenchmarkCase.create_contextual(
                    operation=OPERATION,
                    phase="operator",
                    batch_size=1,
                    tokens=tokens,
                    operator_shape=bf16_shape,
                    token_bucket_min=tokens,
                    token_bucket_max=tokens,
                    evidence_context={"route_mode": route, "seed": seed},
                    dtype="bfloat16",
                )
                for definition in _definitions():
                    builder.add_observation(
                        definition.tactic_id,
                        [1.0 if definition.choice == "upstream" else 1.1] * 5,
                        case=case,
                        provenance={"runner": str(definition.choice)},
                    )
    return builder.build()


def test_contextual_case_round_trip_preserves_typed_dimensions() -> None:
    case = _case(7, route="hot", seed=37, bucket=(5, 8))

    parsed = BenchmarkCase.from_document(case.to_document(), "case")

    assert parsed.schema_version == "musa.engine_case.v2"
    assert parsed.operator_shape_document() == _shape()
    assert parsed.token_bucket is not None
    assert parsed.token_bucket.to_document() == {"min": 5, "max": 8}
    assert parsed.evidence_context_document() == {"route_mode": "hot", "seed": 37}


def test_contextual_case_fingerprint_tampering_fails_closed() -> None:
    document = _case(1, route="balanced", seed=11).to_document()
    document["operator_shape"]["local_experts"] = 256

    with pytest.raises(PlanningArtifactError, match="fingerprint mismatch"):
        BenchmarkCase.from_document(document, "case")


def test_contextual_timing_cache_uses_explicit_v3_schema() -> None:
    document = _timing_document()

    parsed = TimingCache.from_document(document, require_fingerprint=True)

    assert parsed.schema_version == "musa.engine_timing.v3"
    assert {item.case.schema_version for item in parsed.observations if item.case} == {
        "musa.engine_case.v2"
    }


def test_human_inspect_renders_contextual_selection() -> None:
    plan = parse_plan_document(
        build_plan_document([_timing_document()], plan_id="context-inspect-test")
    )

    rendered = _human_inspect(inspect_plan(plan))

    assert "Select musa.fused_moe.dispatch_policy: contextual" in rendered
    assert "Context context-" in rendered
    assert "graph=eager" in rendered
    assert "tokens=1-1" in rendered
    assert "runtime.musa.fused_moe:gemv" in rendered


def test_contextual_timing_cache_combines_eager_and_capture_shapes() -> None:
    eager = _timing_document(graph_mode="eager")
    capture = _timing_document(graph_mode="capture")
    combined = deepcopy(eager)
    combined.pop("fingerprint")
    combined["observations"].extend(capture["observations"])
    combined = seal_timing_cache_document(combined)

    plan = build_plan_document([combined], plan_id="mixed-graph-mode-test")

    entries = plan["variants"][0]["runtime_decisions"]["values"][OPERATION]["entries"]
    assert [entry["shape"]["graph_mode"] for entry in entries] == [
        "capture",
        "eager",
    ]
    assert all(entry["ranges"] for entry in entries)
    capture_entry = next(
        entry for entry in entries if entry["shape"]["graph_mode"] == "capture"
    )
    assert capture_entry["ranges"] == [
        {"min_tokens": 1, "max_tokens": 2, "backend": "gemv"},
        {"min_tokens": 3, "max_tokens": 64, "backend": "upstream"},
    ]
    capture_contexts = [
        context
        for context in plan["variants"][0]["selections"][0]["contexts"]
        if context["shape"]["graph_mode"] == "capture"
    ]
    assert [context["token_bucket"] for context in capture_contexts] == [
        {"min": 1, "max": 1},
        {"min": 2, "max": 2},
        {"min": 3, "max": 64},
    ]
    assert [
        {item["workload_tokens"] for item in context["evidence"]}
        for context in capture_contexts
    ] == [{1}, {2}, {4}]


def test_contextual_capture_selection_requires_graph_ladder() -> None:
    timing = _timing_document(graph_mode="capture")
    timing.pop("fingerprint")
    timing["target"]["workload"].pop("cudagraph_capture_sizes")
    timing = seal_timing_cache_document(timing)

    with pytest.raises(
        PlanningArtifactError,
        match="requires target.workload.cudagraph_capture_sizes",
    ):
        build_plan_document([timing], plan_id="missing-graph-ladder")


def test_decode_only_projection_ignores_unreachable_capture_key() -> None:
    timing = _timing_document(
        graph_mode="capture",
        target_graph_mode="FULL_DECODE_ONLY",
        target_max_num_seqs=2,
    )
    timing["target"]["workload"]["cudagraph_capture_sizes"] = [1, 2, 4]
    timing["observations"] = [
        item
        for item in timing["observations"]
        if item["case"]["workload_point"]["tokens"] != 4
    ]
    timing.pop("fingerprint")
    timing = seal_timing_cache_document(timing)

    plan = build_plan_document([timing], plan_id="decode-only-reachable")
    capture_contexts = [
        context
        for context in plan["variants"][0]["selections"][0]["contexts"]
        if context["shape"]["graph_mode"] == "capture"
    ]
    assert [context["token_bucket"] for context in capture_contexts] == [
        {"min": 1, "max": 1},
        {"min": 2, "max": 64},
    ]


def test_none_graph_target_rejects_capture_evidence() -> None:
    timing = _timing_document(
        graph_mode="capture",
        target_graph_mode="NONE",
    )
    with pytest.raises(
        PlanningArtifactError,
        match="requires a graph-enabled target",
    ):
        build_plan_document([timing], plan_id="none-capture")


def test_optional_graph_ladder_preserves_legacy_runtime_key() -> None:
    current = _target().to_document()
    legacy = deepcopy(current)
    legacy["workload"].pop("cudagraph_capture_sizes")

    legacy_key = runtime_key_document(legacy, final=True)
    current_key = runtime_key_document(current, final=True)

    assert "workload.cudagraph_capture_sizes" not in legacy_key
    assert current_key["workload.cudagraph_capture_sizes"] == [1, 2, 4]


def test_contextual_selection_emits_atomic_deterministic_winner_table() -> None:
    plan = build_plan_document(
        [_timing_document()],
        plan_id="fused-moe-context-test",
        policy=BuildPolicy(min_speedup_pct=1.0, tie_tolerance_pct=0.5),
    )

    assert plan["schema_version"] == 6
    selection = plan["variants"][0]["selections"][0]
    assert "winner" not in selection
    assert [
        (item["token_bucket"], item["winner"]) for item in selection["contexts"]
    ] == [
        ({"min": 1, "max": 1}, "runtime.musa.fused_moe:gemv"),
        ({"min": 2, "max": 2}, "runtime.musa.fused_moe:gemv"),
        ({"min": 3, "max": 3}, "runtime.musa.fused_moe:upstream"),
        ({"min": 4, "max": 4}, "runtime.musa.fused_moe:grouped_gemm"),
    ]
    assert all(len(item["evidence"]) == 9 for item in selection["contexts"])
    assert plan["variants"][0]["runtime_decisions"]["values"][OPERATION] == {
        "schema": "musa.fused_moe.dispatch_policy.v1",
        "entries": [
            {
                "shape": _shape(),
                "ranges": [
                    {"min_tokens": 1, "max_tokens": 2, "backend": "gemv"},
                    {"min_tokens": 3, "max_tokens": 3, "backend": "upstream"},
                    {
                        "min_tokens": 4,
                        "max_tokens": 4,
                        "backend": "grouped_gemm",
                    },
                ],
            }
        ],
    }
    parsed = parse_plan_document(plan)
    assert parsed.schema_version == 6
    inspected = inspect_plan(parsed)
    assert inspected["variants"][0]["selections"][0]["contexts"][0]["winner"].endswith(
        ":gemv"
    )


def test_legacy_contextual_timing_infers_domain_without_rewriting_profile() -> None:
    timing_document = _timing_document()
    timing = TimingCache.from_document(timing_document, require_fingerprint=True)

    domain, source = resolve_timing_cache_domain(timing)
    plan = build_plan_document([timing_document], plan_id="legacy-domain-compat")

    assert domain is not None
    assert domain.domain_id == FUSED_MOE_FP8_BLOCK_DOMAIN_ID
    assert source == "legacy_operation"
    variant = plan["variants"][0]
    assert variant["timing_cache"] == timing_document
    assert variant["runtime_decisions"]["profile"] == "qwen3.moe"


def test_legacy_bf16_shared_operation_remains_unknown_and_buildable() -> None:
    timing_document = _bf16_timing_document()
    timing = TimingCache.from_document(timing_document, require_fingerprint=True)

    domain, source = resolve_timing_cache_domain(timing)
    plan = build_plan_document([timing_document], plan_id="legacy-bf16-compat")

    assert domain is None
    assert source == "unknown"
    assert plan["variants"][0]["timing_cache"] == timing_document
    assert plan["variants"][0]["runtime_decisions"]["profile"] == "qwen3.moe"


def test_legacy_inference_refuses_ambiguous_shared_operation(monkeypatch) -> None:
    from vllm_musa.engine_plan import tuning_domains

    timing = TimingCache.from_document(_timing_document(), require_fingerprint=True)
    current = get_tuning_domain(FUSED_MOE_FP8_BLOCK_DOMAIN_ID)
    future = TuningDomain(
        domain_id="fused_moe.future_compatible_test",
        operations=(OPERATION,),
        evidence_mode="import",
        context_key="test",
        candidate_contract="test",
        correctness_oracle="test",
        runtime_lowering="test",
        target_requirements=("platform=musa",),
    )
    monkeypatch.setattr(
        tuning_domains,
        "_DOMAIN_BY_ID",
        MappingProxyType({current.domain_id: current, future.domain_id: future}),
    )
    monkeypatch.setattr(
        tuning_domains,
        "_DOMAINS_BY_OPERATION",
        MappingProxyType({OPERATION: (current, future)}),
    )
    monkeypatch.setattr(
        tuning_domains,
        "_TARGET_VALIDATORS",
        MappingProxyType(
            {
                current.domain_id: tuning_domains._fused_moe_target_differences,
                future.domain_id: lambda target: (),
            }
        ),
    )
    monkeypatch.setattr(
        tuning_domains,
        "_CASE_VALIDATORS",
        MappingProxyType(
            {
                current.domain_id: tuning_domains._fused_moe_case_differences,
                future.domain_id: lambda case, target: (),
            }
        ),
    )

    assert infer_tuning_domain(timing) is None


def test_partially_registered_domain_fails_closed(monkeypatch) -> None:
    from vllm_musa.engine_plan import tuning_domains

    current = get_tuning_domain(FUSED_MOE_FP8_BLOCK_DOMAIN_ID)
    partial = TuningDomain(
        domain_id="fused_moe.missing_validators_test",
        operations=(OPERATION,),
        evidence_mode="import",
        context_key="test",
        candidate_contract="test",
        correctness_oracle="test",
        runtime_lowering="test",
        target_requirements=("platform=musa",),
    )
    monkeypatch.setattr(
        tuning_domains,
        "_DOMAIN_BY_ID",
        MappingProxyType({current.domain_id: current, partial.domain_id: partial}),
    )
    timing_document = _timing_document()
    timing_document.pop("fingerprint")
    timing_document["provenance"][TUNING_DOMAIN_PROVENANCE_KEY] = partial.domain_id
    timing = TimingCache.from_document(
        seal_timing_cache_document(timing_document),
        require_fingerprint=True,
    )

    with pytest.raises(PlanningArtifactError, match="has no target validator"):
        resolve_timing_cache_domain(timing)


def test_declared_and_legacy_domain_merge_is_order_independent() -> None:
    legacy = _timing_document()
    declared = deepcopy(legacy)
    declared.pop("fingerprint")
    declared["provenance"][TUNING_DOMAIN_PROVENANCE_KEY] = FUSED_MOE_FP8_BLOCK_DOMAIN_ID
    declared = seal_timing_cache_document(declared)

    for inputs in ((legacy, declared), (declared, legacy)):
        merged = merge_timing_cache_documents(inputs)
        timing = TimingCache.from_document(merged, require_fingerprint=True)
        domain, source = resolve_timing_cache_domain(timing)

        assert merged["provenance"][TUNING_DOMAIN_PROVENANCE_KEY] == (
            FUSED_MOE_FP8_BLOCK_DOMAIN_ID
        )
        assert domain is not None
        assert domain.domain_id == FUSED_MOE_FP8_BLOCK_DOMAIN_ID
        assert source == "declared"


def test_sealed_plan_replays_domain_operation_consistency() -> None:
    from vllm_musa.engine_plan.core import seal_plan_document

    plan = build_plan_document([_timing_document()], plan_id="domain-tamper")
    tampered = deepcopy(plan)
    tampered.pop("fingerprint")
    timing = tampered["variants"][0]["timing_cache"]
    timing.pop("fingerprint")
    timing["provenance"][TUNING_DOMAIN_PROVENANCE_KEY] = FUSED_ADD_RMS_NORM_DOMAIN_ID
    tampered["variants"][0]["timing_cache"] = seal_timing_cache_document(timing)

    with pytest.raises(EnginePlanError, match="tuning domain"):
        seal_plan_document(tampered)


def test_contextual_selection_does_not_average_route_or_seed_evidence() -> None:
    plan = build_plan_document(
        [
            _timing_document(
                gemv_winners=frozenset({1}),
                grouped_winners=frozenset(),
                gemv_regression=(1, "hot", 37),
            )
        ],
        plan_id="fused-moe-robustness-test",
    )

    first = plan["variants"][0]["selections"][0]["contexts"][0]
    assert first["winner"] == "runtime.musa.fused_moe:upstream"
    assert first["rejected"] == [
        {
            "tactic_id": "runtime.musa.fused_moe:gemv",
            "reason": "evidence_context_regression_above_tie_tolerance",
        },
        {
            "tactic_id": "runtime.musa.fused_moe:grouped_gemm",
            "reason": "evidence_context_regression_above_tie_tolerance",
        },
    ]


def test_contextual_selection_rejects_one_unstable_evidence_slice() -> None:
    document = _timing_document(
        gemv_winners=frozenset({1}),
        grouped_winners=frozenset(),
    )
    document.pop("fingerprint")
    for observation in document["observations"]:
        case = observation["case"]
        if (
            observation["tactic_id"] == "runtime.musa.fused_moe:gemv"
            and case["workload_point"]["tokens"] == 1
            and case["evidence_context"] == {"route_mode": "hot", "seed": 37}
        ):
            observation["values"] = [0.1, 0.1, 0.5, 0.9, 0.9]
    document = seal_timing_cache_document(document)

    plan = build_plan_document([document], plan_id="unstable-slice-test")

    first = plan["variants"][0]["selections"][0]["contexts"][0]
    assert first["winner"] == "runtime.musa.fused_moe:upstream"
    assert {(item["tactic_id"], item["reason"]) for item in first["rejected"]} >= {
        (
            "runtime.musa.fused_moe:gemv",
            "evidence_context_unstable_iqr",
        )
    }


def test_contextual_selection_rejects_one_p95_regression_slice() -> None:
    document = _timing_document(
        gemv_winners=frozenset({1}),
        grouped_winners=frozenset(),
    )
    document.pop("fingerprint")
    for observation in document["observations"]:
        case = observation["case"]
        if (
            observation["tactic_id"] == "runtime.musa.fused_moe:gemv"
            and case["workload_point"]["tokens"] == 1
            and case["evidence_context"] == {"route_mode": "hot", "seed": 37}
        ):
            observation["values"] = [0.5] * 18 + [1.2, 1.2]
    document = seal_timing_cache_document(document)

    plan = build_plan_document([document], plan_id="p95-regression-test")

    first = plan["variants"][0]["selections"][0]["contexts"][0]
    assert first["winner"] == "runtime.musa.fused_moe:upstream"
    assert {(item["tactic_id"], item["reason"]) for item in first["rejected"]} >= {
        (
            "runtime.musa.fused_moe:gemv",
            "evidence_context_p95_regression_above_guardrail",
        )
    }


def test_contextual_selection_prefers_fallback_on_an_exact_tie() -> None:
    document = _timing_document(
        gemv_winners=frozenset(),
        grouped_winners=frozenset(),
    )
    document.pop("fingerprint")
    for observation in document["observations"]:
        if (
            observation["tactic_id"] == "runtime.musa.fused_moe:gemv"
            and observation["case"]["workload_point"]["tokens"] == 1
        ):
            observation["values"] = [1.0] * 5
    document = seal_timing_cache_document(document)

    plan = build_plan_document(
        [document],
        plan_id="fused-moe-tie-test",
        policy=BuildPolicy(min_speedup_pct=0.0, tie_tolerance_pct=0.5),
    )

    first = plan["variants"][0]["selections"][0]["contexts"][0]
    assert first["winner"] == "runtime.musa.fused_moe:upstream"
    assert first["reason"] == "fallback_fastest_or_insufficient_robust_speedup"


def test_contextual_selection_requires_full_route_seed_matrix() -> None:
    plan = build_plan_document(
        [_timing_document(routes=("balanced", "hot"))],
        plan_id="fused-moe-incomplete-evidence-test",
    )

    contexts = plan["variants"][0]["selections"][0]["contexts"]
    assert {item["winner"] for item in contexts} == {"runtime.musa.fused_moe:upstream"}
    assert {item["reason"] for item in contexts} == {
        "fallback_required:required_route_modes_missing"
    }


def test_contextual_selection_rejects_grouped_gemm_during_capture() -> None:
    plan = build_plan_document(
        [
            _timing_document(
                gemv_winners=frozenset(),
                grouped_winners=frozenset({1}),
                graph_mode="capture",
            )
        ],
        plan_id="fused-moe-capture-support-test",
    )

    first = plan["variants"][0]["selections"][0]["contexts"][0]
    assert first["shape"]["graph_mode"] == "capture"
    assert first["winner"] == "runtime.musa.fused_moe:upstream"
    assert {item["tactic_id"]: item["reason"] for item in first["rejected"]}[
        "runtime.musa.fused_moe:grouped_gemm"
    ] == ("candidate_unsupported_during_graph_capture")


def test_contextual_selection_is_deterministic_across_observation_order() -> None:
    document = _timing_document()
    reordered = deepcopy(document)
    reordered.pop("fingerprint")
    reordered["observations"].reverse()
    reordered = seal_timing_cache_document(reordered)

    first = build_plan_document([document], plan_id="deterministic")
    second = build_plan_document([reordered], plan_id="deterministic")

    assert first == second


def test_contextual_selection_rejects_overlapping_token_buckets() -> None:
    builder = TimingCacheBuilder(
        target=_target(),
        catalog=_definitions(),
        provenance={"run": "overlap-test"},
    )
    for bucket in ((1, 2), (2, 3)):
        for route in ROUTES:
            for seed in SEEDS:
                case = _case(bucket[0], route=route, seed=seed, bucket=bucket)
                for definition in _definitions():
                    builder.add_observation(
                        definition.tactic_id,
                        [1.0] * 5,
                        case=case,
                    )

    with pytest.raises(PlanningArtifactError, match="overlapping token buckets"):
        build_plan_document([builder.build()], plan_id="overlap-test")


def test_schema6_selection_tampering_fails_closed_after_reseal() -> None:
    plan = build_plan_document([_timing_document()], plan_id="tamper-test")
    unsealed = deepcopy(plan)
    unsealed.pop("fingerprint")
    unsealed["variants"][0]["selections"][0]["contexts"][0][
        "winner"
    ] = "runtime.musa.fused_moe:upstream"

    with pytest.raises(EnginePlanError, match="does not match sealed timing"):
        from vllm_musa.engine_plan.core import seal_plan_document

        seal_plan_document(unsealed)


def test_multi_tp_context_plan_rejects_unscoped_global_overrides() -> None:
    with pytest.raises(
        PlanningArtifactError,
        match="global runtime decisions.*multi-TP contextual-only",
    ):
        build_plan_document(
            [_timing_document()],
            plan_id="multi-tp-override-test",
            runtime_decisions={
                "qwen3.moe": {
                    "qwen3.qk_rope_kv_presplit": False,
                }
            },
        )


def test_schema6_context_plan_matches_exact_tp8_runtime(monkeypatch) -> None:
    from vllm_musa.engine_plan import runtime

    plan = parse_plan_document(
        build_plan_document([_timing_document()], plan_id="tp8-runtime-test")
    )
    target = _target()
    monkeypatch.setattr(
        runtime,
        "_hardware_identity",
        lambda: (
            target.hardware.device_name,
            target.hardware.device_capability,
            target.hardware.multiprocessor_count,
            dict(target.software.versions)["driver"],
            target.hardware.device_uuid,
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_software_versions",
        lambda expected, driver_version: dict(target.software.versions),
    )
    monkeypatch.setattr(
        runtime,
        "_runtime_source_revisions",
        lambda: dict(target.software.source_revisions),
    )
    monkeypatch.setattr(runtime, "_tactic_registry_differences", lambda variant: ())
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=[target.model.architecture],
                hidden_size=target.model.hidden_size,
            ),
            model=target.model.model_id,
            dtype=target.model.dtype,
            quantization=target.model.quantization,
            max_model_len=target.workload.max_model_len,
        ),
        quant_config=None,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=target.model.tensor_parallel_size,
            pipeline_parallel_size=target.model.pipeline_parallel_size,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=target.workload.max_num_batched_tokens,
            max_num_seqs=target.workload.max_num_seqs,
        ),
        compilation_config=SimpleNamespace(mode=None, cudagraph_mode=None),
    )

    decision = runtime.select_runtime_variant(plan, config)

    assert decision.is_match
    assert decision.variant is not None
    assert decision.variant["runtime_decisions"]["values"][OPERATION]["schema"] == (
        "musa.fused_moe.dispatch_policy.v1"
    )


def test_schema6_runtime_activation_reports_unique_context_winners(monkeypatch) -> None:
    from vllm_musa.engine_plan import plugin as runtime_plugin
    from vllm_musa.engine_plan.runtime import RuntimeVariantDecision
    from vllm_musa.engine_plugins import ENGINE_PLAN_VARIANT_ENV

    plan = parse_plan_document(
        build_plan_document([_timing_document()], plan_id="context-runtime-test")
    )
    variant = dict(plan.variants[0])
    runtime_target = _target().to_document()
    plugin = runtime_plugin.JsonPlanRuntimePlugin()

    monkeypatch.delenv(ENGINE_PLAN_VARIANT_ENV, raising=False)
    monkeypatch.setattr(plugin, "_load_selected_plan", lambda: plan)
    monkeypatch.setattr(
        runtime_plugin,
        "select_runtime_variant",
        lambda selected_plan, config: RuntimeVariantDecision(
            variant=variant,
            runtime_target=runtime_target,
            reason="runtime_compatibility_key_early",
            differences=(),
        ),
    )
    monkeypatch.setattr(
        runtime_plugin,
        "validate_runtime_variant",
        lambda selected_variant, config: (runtime_target, ()),
    )

    class Config:
        pass

    config = Config()
    config.kernel_config = SimpleNamespace(ir_op_priority=SimpleNamespace())

    application = plugin.apply_config_defaults(config)
    receipt = plugin.validate_runtime_config(config, application)

    assert application.selected_tactics == (
        "runtime.musa.fused_moe:gemv",
        "runtime.musa.fused_moe:grouped_gemm",
        "runtime.musa.fused_moe:upstream",
    )
    assert receipt.selected_tactics == application.selected_tactics
