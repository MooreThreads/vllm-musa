# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from vllm_musa.engine_plan.artifacts import PlanningArtifactError, PlanTarget
from vllm_musa.engine_plan.fused_moe_evidence import (
    import_fused_moe_crossover_evidence,
)
from vllm_musa.engine_plan.planner import build_plan_document
from vllm_musa.engine_plan.tuning_domains import (
    FUSED_MOE_FP8_BLOCK_DOMAIN_ID,
    TUNING_DOMAIN_PROVENANCE_KEY,
)

ROUTES = ("balanced", "unique_random", "hot")
SEEDS = (11, 23, 37)
IMAGE_DIGEST = "sha256:test-image"
SOURCE_SHA = "a" * 40


def _target(
    *,
    profile: str = "qwen3.moe",
    architecture: str = "Qwen3MoeForCausalLM",
    model_id: str = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    quantization: str = "fp8",
) -> dict:
    return {
        "model": {
            "profile": profile,
            "architecture": architecture,
            "model_id": model_id,
            "hidden_size": 4096,
            "dtype": "bfloat16",
            "quantization": quantization,
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
                "torch": "2.11.0.post1+musa5.2.0",
                "torch-musa": "2.11.0.post1+musa5.2.0",
                "torchada": "0.1.83",
                "vllm": "0.1.dev1+gtest",
                "vllm-musa": "0.1.24",
            },
            "source_revisions": {"vllm": "b" * 40, "vllm-musa": SOURCE_SHA},
            "image_digest": IMAGE_DIGEST,
        },
        "workload": {
            "phase": "serving",
            "batch_size": {"min": 1, "max": 4},
            "tokens": {"min": 1, "max": 4},
            "max_model_len": 4096,
            "max_num_batched_tokens": 4,
            "max_num_seqs": 4,
            "compile_mode": "VLLM_COMPILE",
            "graph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4],
        },
    }


def _shape(graph_mode: str = "eager") -> dict:
    return {
        "device_capability": [3, 1],
        "multiprocessor_count": 60,
        "local_experts": 128,
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


def _evidence(
    route: str,
    seed: int,
    *,
    graph_mode: str = "eager",
    tokens: tuple[int, ...] = (1, 2, 3, 4),
    dense_prefix_max: int = 4,
) -> dict:
    graph_capture = graph_mode == "capture"
    results = []
    dispatcher = {}
    graph = {}
    for token_count in tokens:
        dispatcher[str(token_count)] = {
            "passed": True,
            "identities": {
                backend: {"passed": True} for backend in ("upstream", "gemv")
            },
        }
        graph[str(token_count)] = {
            backend: {"passed": True} for backend in ("upstream", "gemv")
        }
        for backend in ("upstream", "gemv"):
            median = 0.5 if backend == "gemv" and token_count <= 2 else 1.0
            if backend == "gemv" and token_count > 2:
                median = 1.2
            gemv = backend == "gemv"
            results.append(
                {
                    "tokens": token_count,
                    "backend": backend,
                    "samples_ms": [median] * 60,
                    "round_medians_ms": [median] * 6,
                    "correctness_pass": True,
                    "error": None,
                    "finite": True,
                    "non_poison": True,
                    "relative_l2_error": 0.001 if gemv else 0.0,
                    "cosine_similarity": 0.999 if gemv else 1.0,
                    "max_row_relative_l2": 0.001 if gemv else 0.0,
                    "min_row_cosine": 0.999 if gemv else 1.0,
                    "max_abs_diff_over_reference_absmax": 0.001 if gemv else 0.0,
                }
            )
    return {
        "metadata": {
            "schema": "musa-fused-moe-crossover.v8",
            "policy_key": _shape(graph_mode),
            "route_mode": route,
            "seed": seed,
            "tokens": list(tokens),
            "backends": ["upstream", "gemv"],
            "dense_prefix_max": dense_prefix_max,
            "sweep_kind": "dense",
            "warmup": 2,
            "repeats_per_round": 10,
            "rounds": 6,
            "gemv_recommendation_requires_contiguous_prefix": True,
            "grouped_recommendation_requires_contiguous_suffix": True,
            "cold_cache": True,
            "l2_flush_mb": 512,
            "max_iqr_ratio": 0.1,
            "selection_samples": "per_round_median",
            "measurement_mode": "graph_replay" if graph_capture else "eager",
            "gemv_max_relative_l2": 0.06,
            "gemv_min_cosine": 0.998,
            "gemv_max_row_relative_l2": 0.08,
            "gemv_min_row_cosine": 0.995,
            "gemv_max_normalized_abs_diff": 0.10,
            "gemv_oracle_max_relative_l2": 0.01,
            "gemv_oracle_min_cosine": 0.9999,
            "gemv_oracle_max_row_relative_l2": 0.02,
            "gemv_oracle_min_row_cosine": 0.9998,
            "gemv_oracle_max_normalized_abs_diff": 0.05,
            "grouped_max_relative_l2": 0.01,
            "grouped_min_cosine": 0.9999,
            "grouped_max_row_relative_l2": 0.02,
            "grouped_min_row_cosine": 0.999,
            "grouped_max_normalized_abs_diff": 0.05,
            "device_capability": [3, 1],
            "multiprocessor_count": 60,
            "device_name": "MTT S5000",
            "packages": {
                "torch": "2.11.0.post1+musa5.2.0",
                "torch_musa": "2.11.0.post1+musa5.2.0",
                "torchada": "0.1.83",
                "mate": "0.2.4",
                "vllm": "0.1.dev1+gtest",
                "vllm-musa": "0.1.24",
            },
            "repo": {
                "head": SOURCE_SHA,
                "status": "",
                "overlay_matches_source": True,
            },
            "exclusive_process_lock": {
                "acquired": True,
                "path": "/tmp/source/.git/musa-fused-moe.lock",
                "pid": 1234,
                "visible_devices": "0",
            },
        },
        "qualification": {
            "passed": True,
            "device_dump_free": True,
            "cold_cache_flush_qualified": True,
        },
        "musa_device_dump_evidence": {
            "passed": True,
            "search_patterns": ["/tmp/*.mudmp"],
            "preexisting_dumps": [],
            "new_dumps": [],
            "changed_preexisting_dumps": [],
            "missing_preexisting_dumps": [],
            "scan_errors": [],
        },
        "oracle_evidence": {
            "gemv_passed": True,
            "errors": {},
            "comparisons": {
                "gemv": {
                    "relative_l2_error": 0.001,
                    "cosine_similarity": 0.99999,
                    "max_row_relative_l2": 0.001,
                    "min_row_cosine": 0.99999,
                    "max_abs_diff_over_reference_absmax": 0.001,
                }
            },
        },
        "dispatcher_smoke_by_tokens": dispatcher,
        "graph_replay_smoke_by_tokens": graph if graph_capture else {},
        "results": results,
    }


def _write_matrix(
    tmp_path,
    *,
    graph_mode: str = "eager",
    tokens: tuple[int, ...] = (1, 2, 3, 4),
    dense_prefix_max: int = 4,
) -> list[str]:
    paths = []
    for route in ROUTES:
        for seed in SEEDS:
            path = tmp_path / f"{graph_mode}-{route}-{seed}.json"
            path.write_text(
                json.dumps(
                    _evidence(
                        route,
                        seed,
                        graph_mode=graph_mode,
                        tokens=tokens,
                        dense_prefix_max=dense_prefix_max,
                    )
                )
            )
            paths.append(str(path))
    return paths


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("compile_mode", "compile"),
        ("compile_mode", "future_compiler"),
        ("graph_mode", "full"),
        ("graph_mode", "future_graph"),
    ),
)
def test_target_rejects_noncanonical_runtime_mode(field: str, value: str) -> None:
    target = _target()
    target["workload"][field] = value

    with pytest.raises(PlanningArtifactError, match=f"target.workload.{field}"):
        PlanTarget.from_document(target)


def test_import_fused_moe_matrix_builds_contextual_plan(tmp_path) -> None:
    target_path = tmp_path / "target.json"
    output_path = tmp_path / "timing.json"
    target_path.write_text(json.dumps(_target()))

    summary = import_fused_moe_crossover_evidence(
        target_path=target_path,
        evidence_paths=_write_matrix(tmp_path),
        output_path=output_path,
        image_digest=IMAGE_DIGEST,
    )
    plan = build_plan_document(
        [json.loads(output_path.read_text())],
        plan_id="fused-moe-import-test",
    )

    assert summary["schema_version"] == "musa.engine_timing.v3"
    assert summary["tuning_domain"] == FUSED_MOE_FP8_BLOCK_DOMAIN_ID
    timing = json.loads(output_path.read_text())
    assert timing["provenance"][TUNING_DOMAIN_PROVENANCE_KEY] == (
        FUSED_MOE_FP8_BLOCK_DOMAIN_ID
    )
    assert plan["schema_version"] == 6
    assert plan["variants"][0]["runtime_decisions"]["profile"] == "qwen3.moe"
    ranges = plan["variants"][0]["runtime_decisions"]["values"][
        "musa.fused_moe.dispatch_policy"
    ]["entries"][0]["ranges"]
    assert ranges == [
        {"min_tokens": 1, "max_tokens": 2, "backend": "gemv"},
        {"min_tokens": 3, "max_tokens": 4, "backend": "upstream"},
    ]


def test_plan_smooths_unrepresentable_token_one_two_backend_transition(
    tmp_path,
) -> None:
    target_path = tmp_path / "target.json"
    output_path = tmp_path / "timing.json"
    target_path.write_text(json.dumps(_target()))
    evidence_paths = _write_matrix(tmp_path)
    for evidence_path in evidence_paths:
        document = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
        for result in document["results"]:
            if result["tokens"] == 2 and result["backend"] == "gemv":
                result.update(
                    median_ms=1.5,
                    p95_ms=1.5,
                    samples_ms=[1.5] * len(result["samples_ms"]),
                    iqr_ratio=0.0,
                )
                if "round_medians_ms" in result:
                    result["round_medians_ms"] = [1.5] * len(result["round_medians_ms"])
        Path(evidence_path).write_text(json.dumps(document), encoding="utf-8")

    import_fused_moe_crossover_evidence(
        target_path=target_path,
        evidence_paths=evidence_paths,
        output_path=output_path,
        image_digest=IMAGE_DIGEST,
    )
    plan = build_plan_document(
        [json.loads(output_path.read_text())],
        plan_id="fused-moe-compile-range-smoothing-test",
    )

    ranges = plan["variants"][0]["runtime_decisions"]["values"][
        "musa.fused_moe.dispatch_policy"
    ]["entries"][0]["ranges"]
    assert ranges == [{"min_tokens": 1, "max_tokens": 4, "backend": "upstream"}]
    contexts = plan["variants"][0]["selections"][0]["contexts"]
    token_one = next(
        context for context in contexts if context["token_bucket"]["min"] == 1
    )
    token_two = next(
        context for context in contexts if context["token_bucket"]["min"] == 2
    )
    expected_reason = "fallback_required:compile_range_token_1_2_backend_transition"
    assert token_one["winner"].endswith(":upstream")
    assert token_two["winner"].endswith(":upstream")
    assert token_one["reason"] == expected_reason
    assert token_two["reason"] == expected_reason
    assert any(
        rejected["tactic_id"].endswith(":gemv")
        and rejected["reason"] == expected_reason
        for rejected in token_one["rejected"]
    )


def test_graph_evidence_covers_capture_ladder_not_serving_token_max(tmp_path) -> None:
    target = _target()
    target["workload"]["tokens"]["max"] = 8192
    target["workload"]["max_num_batched_tokens"] = 8192
    target_path = tmp_path / "target.json"
    output_path = tmp_path / "timing.json"
    target_path.write_text(json.dumps(target))
    evidence = _write_matrix(
        tmp_path,
        graph_mode="eager",
        tokens=(1, 2, 8192),
        dense_prefix_max=2,
    )
    evidence.extend(
        _write_matrix(
            tmp_path,
            graph_mode="capture",
            tokens=(1, 2, 4),
            dense_prefix_max=2,
        )
    )

    summary = import_fused_moe_crossover_evidence(
        target_path=target_path,
        evidence_paths=evidence,
        output_path=output_path,
        image_digest=IMAGE_DIGEST,
    )

    assert summary["schema_version"] == "musa.engine_timing.v3"
    timing = json.loads(output_path.read_text())
    capture_tokens = {
        observation["case"]["workload_point"]["tokens"]
        for observation in timing["observations"]
        if observation["case"]["operator_shape"]["graph_mode"] == "capture"
    }
    assert capture_tokens == {1, 2, 4}


def test_graph_evidence_rejects_missing_configured_capture_size(tmp_path) -> None:
    target = _target()
    target["workload"]["tokens"]["max"] = 8192
    target["workload"]["max_num_batched_tokens"] = 8192
    target_path = tmp_path / "target.json"
    evidence_path = tmp_path / "capture.json"
    target_path.write_text(json.dumps(target))
    evidence_path.write_text(
        json.dumps(
            _evidence(
                "balanced",
                11,
                graph_mode="capture",
                tokens=(1, 2),
                dense_prefix_max=2,
            )
        )
    )

    with pytest.raises(
        PlanningArtifactError,
        match=r"does not cover target capture sizes: missing=\[4\]",
    ):
        import_fused_moe_crossover_evidence(
            target_path=target_path,
            evidence_paths=[evidence_path],
            output_path=tmp_path / "timing.json",
            image_digest=IMAGE_DIGEST,
        )


def test_eager_evidence_still_requires_serving_token_max(tmp_path) -> None:
    target = _target()
    target["workload"]["tokens"]["max"] = 8192
    target["workload"]["max_num_batched_tokens"] = 8192
    target_path = tmp_path / "target.json"
    evidence_path = tmp_path / "eager.json"
    target_path.write_text(json.dumps(target))
    evidence_path.write_text(
        json.dumps(
            _evidence(
                "balanced",
                11,
                tokens=(1, 2),
                dense_prefix_max=2,
            )
        )
    )

    with pytest.raises(
        PlanningArtifactError,
        match="eager crossover evidence does not cover target max tokens",
    ):
        import_fused_moe_crossover_evidence(
            target_path=target_path,
            evidence_paths=[evidence_path],
            output_path=tmp_path / "timing.json",
            image_digest=IMAGE_DIGEST,
        )


def test_compatible_non_qwen_model_reuses_fused_moe_domain(tmp_path) -> None:
    target_path = tmp_path / "target.json"
    output_path = tmp_path / "timing.json"
    target_path.write_text(
        json.dumps(
            _target(
                profile="future_moe.moe",
                architecture="FutureMoeForCausalLM",
                model_id="example/Future-MoE-FP8",
            )
        )
    )

    import_fused_moe_crossover_evidence(
        target_path=target_path,
        evidence_paths=_write_matrix(tmp_path),
        output_path=output_path,
        image_digest=IMAGE_DIGEST,
    )
    timing = json.loads(output_path.read_text())
    plan = build_plan_document([timing], plan_id="future-moe-domain-reuse")

    assert timing["provenance"][TUNING_DOMAIN_PROVENANCE_KEY] == (
        FUSED_MOE_FP8_BLOCK_DOMAIN_ID
    )
    variant = plan["variants"][0]
    assert variant["timing_cache"]["target"]["model"]["architecture"] == (
        "FutureMoeForCausalLM"
    )
    assert variant["runtime_decisions"]["profile"] == "future_moe.moe"
    assert "musa.fused_moe.dispatch_policy" in variant["runtime_decisions"]["values"]


def test_fused_moe_domain_rejects_non_fp8_model_before_evidence(tmp_path) -> None:
    target_path = tmp_path / "target.json"
    target_path.write_text(json.dumps(_target(quantization="gptq")))

    with pytest.raises(
        PlanningArtifactError,
        match="model.quantization: fused-MoE domain requires fp8",
    ):
        import_fused_moe_crossover_evidence(
            target_path=target_path,
            evidence_paths=[tmp_path / "not-read.json"],
            output_path=tmp_path / "timing.json",
            image_digest=IMAGE_DIGEST,
        )


@pytest.mark.parametrize(
    "command",
    (
        ("autotune", "import", "--domain", FUSED_MOE_FP8_BLOCK_DOMAIN_ID),
        ("autotune", "import-fused-moe"),
    ),
    ids=("domain-entrypoint", "legacy-alias"),
)
def test_fused_moe_cli_domain_entrypoint_and_legacy_alias(
    tmp_path, capsys, command: tuple[str, ...]
) -> None:
    from vllm_musa.engine_plan.cli import main

    target_path = tmp_path / "target.json"
    evidence_path = tmp_path / "evidence.json"
    output_path = tmp_path / "timing.json"
    target_path.write_text(json.dumps(_target()))
    evidence_path.write_text(json.dumps(_evidence("balanced", 11)))

    result = main(
        [
            *command,
            "--target",
            str(target_path),
            "--evidence",
            str(evidence_path),
            "--image-digest",
            IMAGE_DIGEST,
            "--output",
            str(output_path),
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert result == 0
    assert summary["tuning_domain"] == FUSED_MOE_FP8_BLOCK_DOMAIN_ID
    assert (
        json.loads(output_path.read_text())["provenance"][TUNING_DOMAIN_PROVENANCE_KEY]
        == FUSED_MOE_FP8_BLOCK_DOMAIN_ID
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("block_n", 64, "operator_shape.block_n"),
        ("activation", "gelu", "operator_shape.activation"),
        ("weight_dtype", "torch.bfloat16", "operator_shape.weight_dtype"),
        ("scale_dtype", "torch.bfloat16", "operator_shape.scale_dtype"),
        ("expert_parallel", True, "operator_shape.expert_parallel"),
        ("w1_scale_shape", [128, 4, 31], "operator_shape.w1_scale_shape"),
        ("multiprocessor_count", 64, "multiprocessor_count does not match target"),
        ("device_capability", [3, 0], "device_capability does not match target"),
        ("local_experts", "bad", "local_experts must be a positive integer"),
        ("top_k", 0, "top_k must be a positive integer"),
        ("top_k", 129, "top_k must not exceed local_experts"),
        ("graph_mode", "future", "graph_mode must be eager or capture"),
        ("gemv_block", "", "gemv_block.*empty string"),
        ("gemv_block", "banana", "expected 'auto' or '<block_n>x<block_k>'"),
        ("gemv_block", "999x999", r"block_n \* block_k must be <= 512"),
        ("gemv_block", "7x7", "tile does not divide both routed GEMV"),
    ),
)
def test_fused_moe_domain_rejects_incompatible_operator_shape(
    tmp_path, field: str, value: object, message: str
) -> None:
    target_path = tmp_path / "target.json"
    evidence_path = tmp_path / "evidence.json"
    target_path.write_text(json.dumps(_target()))
    evidence = _evidence("balanced", 11)
    evidence["metadata"]["policy_key"][field] = value
    evidence_path.write_text(json.dumps(evidence))

    with pytest.raises(PlanningArtifactError, match=message):
        import_fused_moe_crossover_evidence(
            target_path=target_path,
            evidence_paths=[evidence_path],
            output_path=tmp_path / "timing.json",
            image_digest=IMAGE_DIGEST,
        )


def test_import_fused_moe_combines_eager_and_capture_in_one_timing_cache(
    tmp_path,
) -> None:
    target_path = tmp_path / "target.json"
    output_path = tmp_path / "timing.json"
    target_path.write_text(json.dumps(_target()))
    evidence_paths = [
        *_write_matrix(tmp_path, graph_mode="eager"),
        *_write_matrix(tmp_path, graph_mode="capture"),
    ]

    summary = import_fused_moe_crossover_evidence(
        target_path=target_path,
        evidence_paths=evidence_paths,
        output_path=output_path,
        image_digest=IMAGE_DIGEST,
    )
    plan = build_plan_document(
        [json.loads(output_path.read_text())],
        plan_id="fused-moe-mixed-mode-import-test",
    )

    assert summary["evidence_files"] == 18
    entries = plan["variants"][0]["runtime_decisions"]["values"][
        "musa.fused_moe.dispatch_policy"
    ]["entries"]
    assert [entry["shape"]["graph_mode"] for entry in entries] == [
        "capture",
        "eager",
    ]


def test_import_fused_moe_requires_exact_source_provenance(tmp_path) -> None:
    target_path = tmp_path / "target.json"
    evidence_path = tmp_path / "evidence.json"
    target_path.write_text(json.dumps(_target()))
    evidence = _evidence("balanced", 11)
    evidence["metadata"]["repo"] = {"head": None, "status": None}
    evidence_path.write_text(json.dumps(evidence))

    with pytest.raises(PlanningArtifactError, match="no vllm-musa source SHA"):
        import_fused_moe_crossover_evidence(
            target_path=target_path,
            evidence_paths=[evidence_path],
            output_path=tmp_path / "timing.json",
            image_digest=IMAGE_DIGEST,
        )


def test_import_fused_moe_capture_requires_graph_replay_proof(tmp_path) -> None:
    target_path = tmp_path / "target.json"
    evidence_path = tmp_path / "evidence.json"
    target_path.write_text(json.dumps(_target()))
    evidence = _evidence("balanced", 11, graph_mode="capture")
    broken = deepcopy(evidence)
    broken["graph_replay_smoke_by_tokens"]["1"]["gemv"]["passed"] = False
    evidence_path.write_text(json.dumps(broken))

    with pytest.raises(PlanningArtifactError, match="graph replay failed"):
        import_fused_moe_crossover_evidence(
            target_path=target_path,
            evidence_paths=[evidence_path],
            output_path=tmp_path / "timing.json",
            image_digest=IMAGE_DIGEST,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda evidence: evidence["musa_device_dump_evidence"].update(
                new_dumps=[{"path": "/tmp/new.mudmp"}], passed=False
            ),
            "device-dump evidence did not pass",
        ),
        (
            lambda evidence: evidence["musa_device_dump_evidence"].update(
                changed_preexisting_dumps=[{"path": "/tmp/old.mudmp"}],
                passed=False,
            ),
            "device-dump evidence did not pass",
        ),
        (
            lambda evidence: evidence["musa_device_dump_evidence"].update(
                missing_preexisting_dumps=["/tmp/old.mudmp"], passed=False
            ),
            "device-dump evidence did not pass",
        ),
        (
            lambda evidence: evidence["musa_device_dump_evidence"].update(
                scan_errors=["permission denied"], passed=False
            ),
            "device-dump evidence did not pass",
        ),
        (
            lambda evidence: evidence["musa_device_dump_evidence"].pop("scan_errors"),
            "keys do not match schema",
        ),
        (
            lambda evidence: evidence["metadata"].update(l2_flush_mb=511),
            "at least 512",
        ),
        (
            lambda evidence: evidence["metadata"].pop("exclusive_process_lock"),
            "metadata.exclusive_process_lock must be a JSON object",
        ),
        (
            lambda evidence: evidence["metadata"]["repo"].update(status=None),
            "source checkout must be clean",
        ),
        (
            lambda evidence: evidence["metadata"].update(rounds=5),
            "balance backend order",
        ),
        (
            lambda evidence: evidence["results"][0].update(round_medians_ms=[1.0] * 5),
            "round_medians_ms must contain exactly 6",
        ),
        (
            lambda evidence: evidence["results"].pop(),
            "results do not cover the declared matrix",
        ),
        (
            lambda evidence: evidence["qualification"].update(
                cold_cache_flush_qualified=False
            ),
            "canonical cold-cache flushing",
        ),
        (
            lambda evidence: evidence["metadata"].update(gemv_max_relative_l2=0.061),
            "canonical maximum",
        ),
        (
            lambda evidence: evidence["metadata"].update(gemv_min_cosine=0.997),
            "canonical minimum",
        ),
        (
            lambda evidence: evidence["results"][1].update(relative_l2_error=0.061),
            "result.relative_l2_error",
        ),
        (
            lambda evidence: evidence["results"][1].update(relative_l2_error=-1.0),
            "result.relative_l2_error must be non-negative",
        ),
    ),
)
def test_import_fused_moe_rejects_unqualified_or_weak_evidence(
    tmp_path, mutation, message
) -> None:
    target_path = tmp_path / "target.json"
    evidence_path = tmp_path / "evidence.json"
    target_path.write_text(json.dumps(_target()))
    evidence = _evidence("balanced", 11)
    mutation(evidence)
    evidence_path.write_text(json.dumps(evidence))

    with pytest.raises(PlanningArtifactError, match=message):
        import_fused_moe_crossover_evidence(
            target_path=target_path,
            evidence_paths=[evidence_path],
            output_path=tmp_path / "timing.json",
            image_digest=IMAGE_DIGEST,
        )
