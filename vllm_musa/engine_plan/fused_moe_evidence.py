# SPDX-License-Identifier: Apache-2.0

"""Normalize first-party fused-MoE crossover evidence for the AutoTuner."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifact_io import load_json_object_file, write_json_object_file
from .artifacts import (
    BenchmarkCase,
    PlanningArtifactError,
    PlanTarget,
    TimingCacheBuilder,
    compute_artifact_fingerprint,
)
from .fused_moe_contract import (
    FUSED_MOE_CROSSOVER_SCHEMA,
    FUSED_MOE_MAX_ERROR_THRESHOLDS,
    FUSED_MOE_MIN_COSINE_THRESHOLDS,
    FUSED_MOE_MIN_QUALIFIED_L2_FLUSH_MB,
)
from .tuning_domains import (
    FUSED_MOE_BACKENDS,
    FUSED_MOE_DISPATCH_OPERATION,
    FUSED_MOE_FP8_BLOCK_DOMAIN_ID,
    FUSED_MOE_ROUTES,
    TUNING_DOMAIN_PROVENANCE_KEY,
    fused_moe_tactic_definitions,
    validate_tuning_domain_case,
    validate_tuning_domain_target,
)

FUSED_MOE_OPERATION = FUSED_MOE_DISPATCH_OPERATION
_MAX_ERROR_THRESHOLDS = dict(FUSED_MOE_MAX_ERROR_THRESHOLDS)
_MIN_COSINE_THRESHOLDS = dict(FUSED_MOE_MIN_COSINE_THRESHOLDS)
FUSED_MOE_MEASUREMENT_BACKENDS = frozenset((*FUSED_MOE_BACKENDS, "deepgemm_prefill"))


def _expect_mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PlanningArtifactError(f"{field} must be a JSON object")
    return dict(value)


def _expect_positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PlanningArtifactError(f"{field} must be a positive integer")
    return value


def _expect_finite_number(value: object, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise PlanningArtifactError(f"{field} must be a finite number")
    return float(value)


def _expect_empty_list(value: object, field: str) -> None:
    if not isinstance(value, list):
        raise PlanningArtifactError(f"{field} must be a JSON array")
    if value:
        raise PlanningArtifactError(f"{field} must be empty")


def _target_from_path(path: str | Path) -> PlanTarget:
    document = load_json_object_file(path)
    return PlanTarget.from_document(document.get("target", document))


def _validate_hardware(metadata: Mapping[str, Any], target: PlanTarget) -> None:
    capability = metadata.get("device_capability")
    expected_capability = [
        int(value) for value in target.hardware.device_capability.split(".")
    ]
    if capability != expected_capability:
        raise PlanningArtifactError(
            "crossover device capability does not match the target"
        )
    if metadata.get("multiprocessor_count") != target.hardware.multiprocessor_count:
        raise PlanningArtifactError("crossover MP bin does not match the target")
    device_name = metadata.get("device_name")
    if (
        not isinstance(device_name, str)
        or target.hardware.device_name not in device_name
    ):
        raise PlanningArtifactError("crossover device name does not match the target")


def _validate_packages(metadata: Mapping[str, Any], target: PlanTarget) -> None:
    packages = _expect_mapping(metadata.get("packages"), "metadata.packages")
    target_versions = dict(target.software.versions)
    aliases = {"torch_musa": "torch-musa", "vllm-musa": "vllm-musa"}
    for evidence_name in (
        "torch",
        "torch_musa",
        "torchada",
        "mate",
        "vllm",
        "vllm-musa",
    ):
        target_name = aliases.get(evidence_name, evidence_name)
        if target_name not in target_versions:
            continue
        if packages.get(evidence_name) != target_versions[target_name]:
            raise PlanningArtifactError(
                f"crossover package {evidence_name!r} does not match the target"
            )


def _validate_source(metadata: Mapping[str, Any], target: PlanTarget) -> None:
    repo = _expect_mapping(metadata.get("repo"), "metadata.repo")
    head = repo.get("head")
    status = repo.get("status")
    expected = dict(target.software.source_revisions).get("vllm-musa")
    if not isinstance(head, str) or not head:
        raise PlanningArtifactError("crossover evidence has no vllm-musa source SHA")
    if expected != head:
        raise PlanningArtifactError("crossover source SHA does not match the target")
    if status != "":
        raise PlanningArtifactError("crossover source checkout must be clean")
    if repo.get("overlay_matches_source") is not True:
        raise PlanningArtifactError(
            "crossover runtime module does not match the clean source checkout"
        )


def _validate_device_dump_evidence(
    document: Mapping[str, Any], qualification: Mapping[str, Any]
) -> None:
    if qualification.get("device_dump_free") is not True:
        raise PlanningArtifactError(
            "crossover qualification must prove device_dump_free"
        )
    evidence = _expect_mapping(
        document.get("musa_device_dump_evidence"),
        "musa_device_dump_evidence",
    )
    required_keys = {
        "passed",
        "search_patterns",
        "preexisting_dumps",
        "new_dumps",
        "changed_preexisting_dumps",
        "missing_preexisting_dumps",
        "scan_errors",
    }
    actual_keys = set(evidence)
    if actual_keys != required_keys:
        raise PlanningArtifactError(
            "musa_device_dump_evidence keys do not match schema; "
            f"missing={sorted(required_keys - actual_keys)}, "
            f"unknown={sorted(actual_keys - required_keys)}"
        )
    if evidence.get("passed") is not True:
        raise PlanningArtifactError("crossover device-dump evidence did not pass")
    search_patterns = evidence.get("search_patterns")
    if (
        not isinstance(search_patterns, list)
        or not search_patterns
        or any(not isinstance(item, str) or not item for item in search_patterns)
    ):
        raise PlanningArtifactError(
            "musa_device_dump_evidence.search_patterns must be non-empty strings"
        )
    preexisting = evidence.get("preexisting_dumps")
    if not isinstance(preexisting, list):
        raise PlanningArtifactError(
            "musa_device_dump_evidence.preexisting_dumps must be a JSON array"
        )
    for index, item in enumerate(preexisting):
        entry = _expect_mapping(
            item, f"musa_device_dump_evidence.preexisting_dumps[{index}]"
        )
        if not isinstance(entry.get("path"), str) or not entry["path"]:
            raise PlanningArtifactError(
                "preexisting dump entries must contain a non-empty path"
            )
    for field in (
        "new_dumps",
        "changed_preexisting_dumps",
        "missing_preexisting_dumps",
        "scan_errors",
    ):
        _expect_empty_list(evidence.get(field), f"musa_device_dump_evidence.{field}")


def _validate_exclusive_process_lock(metadata: Mapping[str, Any]) -> None:
    evidence = _expect_mapping(
        metadata.get("exclusive_process_lock"),
        "metadata.exclusive_process_lock",
    )
    required_keys = {"acquired", "path", "pid", "visible_devices"}
    if set(evidence) != required_keys:
        raise PlanningArtifactError(
            "metadata.exclusive_process_lock keys do not match schema"
        )
    if evidence.get("acquired") is not True:
        raise PlanningArtifactError("exclusive benchmark process lock was not acquired")
    if not isinstance(evidence.get("path"), str) or not evidence["path"]:
        raise PlanningArtifactError("exclusive benchmark lock path must be non-empty")
    _expect_positive_int(evidence.get("pid"), "metadata.exclusive_process_lock.pid")
    if (
        not isinstance(evidence.get("visible_devices"), str)
        or not evidence["visible_devices"]
    ):
        raise PlanningArtifactError(
            "exclusive benchmark lock visible_devices must be non-empty"
        )


def _validate_correctness_thresholds(metadata: Mapping[str, Any]) -> None:
    for field, canonical_maximum in FUSED_MOE_MAX_ERROR_THRESHOLDS:
        value = _expect_finite_number(metadata.get(field), f"metadata.{field}")
        if value < 0 or value > canonical_maximum:
            raise PlanningArtifactError(
                f"crossover {field} exceeds canonical maximum; "
                f"must be in [0, {canonical_maximum}]"
            )
    for field, canonical_minimum in FUSED_MOE_MIN_COSINE_THRESHOLDS:
        value = _expect_finite_number(metadata.get(field), f"metadata.{field}")
        if value < canonical_minimum or value > 1:
            raise PlanningArtifactError(
                f"crossover {field} is below canonical minimum; "
                f"must be in [{canonical_minimum}, 1]"
            )


def _validate_measurement_design(
    metadata: Mapping[str, Any], *, dense_prefix: int
) -> tuple[tuple[str, ...], tuple[int, ...], int, int]:
    raw_backends = metadata.get("backends")
    if not isinstance(raw_backends, list) or not raw_backends:
        raise PlanningArtifactError("metadata.backends must be a non-empty list")
    if any(
        not isinstance(backend, str) or backend not in FUSED_MOE_MEASUREMENT_BACKENDS
        for backend in raw_backends
    ):
        raise PlanningArtifactError("metadata.backends contains an unsupported backend")
    backends = tuple(raw_backends)
    if len(backends) != len(set(backends)):
        raise PlanningArtifactError("metadata.backends must not contain duplicates")
    if "upstream" not in backends:
        raise PlanningArtifactError("metadata.backends must include upstream")

    raw_tokens = metadata.get("tokens")
    if not isinstance(raw_tokens, list) or not raw_tokens:
        raise PlanningArtifactError("metadata.tokens must be a non-empty list")
    tokens = tuple(
        _expect_positive_int(value, f"metadata.tokens[{index}]")
        for index, value in enumerate(raw_tokens)
    )
    if tokens != tuple(sorted(set(tokens))):
        raise PlanningArtifactError("metadata.tokens must be sorted and unique")
    required_dense_tokens = set(range(1, dense_prefix + 1))
    if not required_dense_tokens.issubset(tokens):
        raise PlanningArtifactError(
            "metadata.tokens does not cover the declared dense prefix"
        )

    warmup = _expect_positive_int(metadata.get("warmup"), "metadata.warmup")
    repeats = _expect_positive_int(
        metadata.get("repeats_per_round"), "metadata.repeats_per_round"
    )
    rounds = _expect_positive_int(metadata.get("rounds"), "metadata.rounds")
    if warmup < 1 or repeats < 3:
        raise PlanningArtifactError(
            "crossover measurements require warmup and at least three repeats"
        )
    if rounds < 3 or rounds % len(backends) != 0:
        raise PlanningArtifactError(
            "metadata.rounds must be at least three and balance backend order"
        )
    if metadata.get("gemv_recommendation_requires_contiguous_prefix") is not True:
        raise PlanningArtifactError("GEMV recommendation must require a dense prefix")
    if metadata.get("grouped_recommendation_requires_contiguous_suffix") is not True:
        raise PlanningArtifactError(
            "grouped-GEMM recommendation must require a contiguous suffix"
        )
    return backends, tokens, rounds, repeats


def _validate_timing_values(value: object, *, field: str, expected_count: int) -> None:
    if not isinstance(value, list) or len(value) != expected_count:
        raise PlanningArtifactError(
            f"{field} must contain exactly {expected_count} measurements"
        )
    if any(
        not isinstance(item, (int, float))
        or isinstance(item, bool)
        or not math.isfinite(float(item))
        or float(item) <= 0
        for item in value
    ):
        raise PlanningArtifactError(f"{field} must contain finite positive timings")


def _validate_measurement_results(
    results: Sequence[object],
    *,
    backends: tuple[str, ...],
    tokens: tuple[int, ...],
    rounds: int,
    repeats: int,
    metadata: Mapping[str, Any],
) -> None:
    expected = {(token, backend) for token in tokens for backend in backends}
    observed: set[tuple[int, str]] = set()
    for index, raw_result in enumerate(results):
        result = _expect_mapping(raw_result, f"results[{index}]")
        backend = result.get("backend")
        if backend not in backends:
            raise PlanningArtifactError(
                f"results[{index}].backend is not declared in metadata.backends"
            )
        token = _expect_positive_int(result.get("tokens"), f"results[{index}].tokens")
        key = (token, backend)
        if key not in expected:
            raise PlanningArtifactError(
                f"results[{index}] is outside the declared token/backend matrix"
            )
        if key in observed:
            raise PlanningArtifactError(
                f"duplicate crossover result for tokens={token} backend={backend}"
            )
        observed.add(key)
        if (
            result.get("error") is not None
            or result.get("correctness_pass") is not True
        ):
            raise PlanningArtifactError(
                f"crossover result failed for tokens={token} backend={backend}"
            )
        _validate_result_correctness(
            result,
            backend=backend,
            index=index,
            metadata=metadata,
        )
        _validate_timing_values(
            result.get("round_medians_ms"),
            field=f"results[{index}].round_medians_ms",
            expected_count=rounds,
        )
        _validate_timing_values(
            result.get("samples_ms"),
            field=f"results[{index}].samples_ms",
            expected_count=rounds * repeats,
        )
    if observed != expected:
        missing = sorted(expected - observed)
        raise PlanningArtifactError(
            f"crossover results do not cover the declared matrix: missing={missing}"
        )


def _validate_metric(
    result: Mapping[str, Any],
    field: str,
    *,
    maximum: float | None = None,
    minimum: float | None = None,
) -> None:
    value = _expect_finite_number(result.get(field), f"result.{field}")
    if maximum is not None and value < 0:
        raise PlanningArtifactError(f"result.{field} must be non-negative")
    if maximum is not None and value > maximum:
        raise PlanningArtifactError(
            f"result.{field} exceeds canonical maximum {maximum}"
        )
    if minimum is not None and value < minimum:
        raise PlanningArtifactError(
            f"result.{field} is below canonical minimum {minimum}"
        )


def _validate_result_correctness(
    result: Mapping[str, Any],
    *,
    backend: str,
    index: int,
    metadata: Mapping[str, Any],
) -> None:
    if result.get("finite") is not True or result.get("non_poison") is not True:
        raise PlanningArtifactError(
            f"crossover result[{index}] lacks finite/non-poison proof"
        )
    if backend == "gemv":
        _validate_metric(
            result,
            "relative_l2_error",
            maximum=min(
                _MAX_ERROR_THRESHOLDS["gemv_max_relative_l2"],
                float(metadata["gemv_max_relative_l2"]),
            ),
        )
        _validate_metric(
            result,
            "cosine_similarity",
            minimum=max(
                _MIN_COSINE_THRESHOLDS["gemv_min_cosine"],
                float(metadata["gemv_min_cosine"]),
            ),
        )
        _validate_metric(
            result,
            "max_row_relative_l2",
            maximum=min(
                _MAX_ERROR_THRESHOLDS["gemv_max_row_relative_l2"],
                float(metadata["gemv_max_row_relative_l2"]),
            ),
        )
        _validate_metric(
            result,
            "min_row_cosine",
            minimum=max(
                _MIN_COSINE_THRESHOLDS["gemv_min_row_cosine"],
                float(metadata["gemv_min_row_cosine"]),
            ),
        )
        _validate_metric(
            result,
            "max_abs_diff_over_reference_absmax",
            maximum=min(
                _MAX_ERROR_THRESHOLDS["gemv_max_normalized_abs_diff"],
                float(metadata["gemv_max_normalized_abs_diff"]),
            ),
        )
    elif backend in {"grouped_gemm", "deepgemm_prefill"}:
        _validate_metric(
            result,
            "relative_l2_error",
            maximum=min(
                _MAX_ERROR_THRESHOLDS["grouped_max_relative_l2"],
                float(metadata["grouped_max_relative_l2"]),
            ),
        )
        _validate_metric(
            result,
            "cosine_similarity",
            minimum=max(
                _MIN_COSINE_THRESHOLDS["grouped_min_cosine"],
                float(metadata["grouped_min_cosine"]),
            ),
        )
        _validate_metric(
            result,
            "max_row_relative_l2",
            maximum=min(
                _MAX_ERROR_THRESHOLDS["grouped_max_row_relative_l2"],
                float(metadata["grouped_max_row_relative_l2"]),
            ),
        )
        _validate_metric(
            result,
            "min_row_cosine",
            minimum=max(
                _MIN_COSINE_THRESHOLDS["grouped_min_row_cosine"],
                float(metadata["grouped_min_row_cosine"]),
            ),
        )
        _validate_metric(
            result,
            "max_abs_diff_over_reference_absmax",
            maximum=min(
                _MAX_ERROR_THRESHOLDS["grouped_max_normalized_abs_diff"],
                float(metadata["grouped_max_normalized_abs_diff"]),
            ),
        )


def _validate_oracle_evidence(
    document: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
    """Require the independent FP32 oracle gate whenever GEMV is present."""

    if not any(
        isinstance(item, Mapping) and item.get("backend") == "gemv"
        for item in document.get("results", [])
    ):
        return
    oracle = _expect_mapping(document.get("oracle_evidence"), "oracle_evidence")
    if oracle.get("gemv_passed") is not True:
        raise PlanningArtifactError("crossover GEMV oracle proof did not pass")
    errors = oracle.get("errors")
    if not isinstance(errors, Mapping) or errors:
        raise PlanningArtifactError("crossover GEMV oracle has errors")
    comparisons = _expect_mapping(oracle.get("comparisons"), "oracle.comparisons")
    gemv = _expect_mapping(comparisons.get("gemv"), "oracle.comparisons.gemv")
    _validate_metric(
        gemv,
        "relative_l2_error",
        maximum=min(
            _MAX_ERROR_THRESHOLDS["gemv_oracle_max_relative_l2"],
            float(metadata["gemv_oracle_max_relative_l2"]),
        ),
    )
    _validate_metric(
        gemv,
        "cosine_similarity",
        minimum=max(
            _MIN_COSINE_THRESHOLDS["gemv_oracle_min_cosine"],
            float(metadata["gemv_oracle_min_cosine"]),
        ),
    )
    _validate_metric(
        gemv,
        "max_row_relative_l2",
        maximum=min(
            _MAX_ERROR_THRESHOLDS["gemv_oracle_max_row_relative_l2"],
            float(metadata["gemv_oracle_max_row_relative_l2"]),
        ),
    )
    _validate_metric(
        gemv,
        "min_row_cosine",
        minimum=max(
            _MIN_COSINE_THRESHOLDS["gemv_oracle_min_row_cosine"],
            float(metadata["gemv_oracle_min_row_cosine"]),
        ),
    )
    _validate_metric(
        gemv,
        "max_abs_diff_over_reference_absmax",
        maximum=min(
            _MAX_ERROR_THRESHOLDS["gemv_oracle_max_normalized_abs_diff"],
            float(metadata["gemv_oracle_max_normalized_abs_diff"]),
        ),
    )


def _validate_metadata(
    document: Mapping[str, Any],
    target: PlanTarget,
) -> tuple[dict[str, Any], str, int, int]:
    metadata = _expect_mapping(document.get("metadata"), "metadata")
    if metadata.get("schema") != FUSED_MOE_CROSSOVER_SCHEMA:
        raise PlanningArtifactError(
            f"crossover schema must be {FUSED_MOE_CROSSOVER_SCHEMA!r}"
        )
    qualification = _expect_mapping(document.get("qualification"), "qualification")
    if qualification.get("passed") is not True:
        raise PlanningArtifactError("crossover qualification did not pass")
    if qualification.get("cold_cache_flush_qualified") is not True:
        raise PlanningArtifactError(
            "crossover qualification must prove canonical cold-cache flushing"
        )
    _validate_device_dump_evidence(document, qualification)
    if metadata.get("cold_cache") is not True:
        raise PlanningArtifactError("crossover evidence must use cold-cache timing")
    l2_flush_mb = _expect_positive_int(
        metadata.get("l2_flush_mb"), "metadata.l2_flush_mb"
    )
    if l2_flush_mb < FUSED_MOE_MIN_QUALIFIED_L2_FLUSH_MB:
        raise PlanningArtifactError(
            "crossover l2_flush_mb must be at least "
            f"{FUSED_MOE_MIN_QUALIFIED_L2_FLUSH_MB}"
        )
    if metadata.get("max_iqr_ratio") != 0.1:
        raise PlanningArtifactError("crossover max_iqr_ratio must be 0.1")
    if metadata.get("selection_samples") != "per_round_median":
        raise PlanningArtifactError(
            "crossover selection samples must be per-round medians"
        )
    route = metadata.get("route_mode")
    if route not in FUSED_MOE_ROUTES:
        raise PlanningArtifactError("crossover route_mode is unsupported")
    seed = metadata.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise PlanningArtifactError("crossover seed must be non-negative")
    dense_prefix = _expect_positive_int(
        metadata.get("dense_prefix_max"), "metadata.dense_prefix_max"
    )
    if metadata.get("sweep_kind") != "dense":
        raise PlanningArtifactError("crossover evidence must be a dense sweep")
    _validate_exclusive_process_lock(metadata)
    _validate_correctness_thresholds(metadata)
    _validate_hardware(metadata, target)
    _validate_packages(metadata, target)
    _validate_source(metadata, target)
    return metadata, route, seed, dense_prefix


def _token_bucket(
    token_count: int,
    *,
    dense_prefix: int,
    maximum: int,
) -> tuple[int, int]:
    if token_count <= dense_prefix:
        return token_count, token_count
    return dense_prefix + 1, maximum


def _validate_dispatch_identity(
    document: Mapping[str, Any],
    *,
    token_count: int,
    backend: str,
    graph_capture: bool,
) -> None:
    dispatch = _expect_mapping(
        _expect_mapping(
            document.get("dispatcher_smoke_by_tokens"),
            "dispatcher_smoke_by_tokens",
        ).get(str(token_count)),
        f"dispatcher_smoke_by_tokens.{token_count}",
    )
    identity = _expect_mapping(
        _expect_mapping(dispatch.get("identities"), "dispatcher.identities").get(
            backend
        ),
        f"dispatcher.identities.{backend}",
    )
    if dispatch.get("passed") is not True or identity.get("passed") is not True:
        raise PlanningArtifactError(
            f"dispatcher identity failed for tokens={token_count} backend={backend}"
        )
    if not graph_capture:
        return
    graph = _expect_mapping(
        _expect_mapping(
            document.get("graph_replay_smoke_by_tokens"),
            "graph_replay_smoke_by_tokens",
        ).get(str(token_count)),
        f"graph_replay_smoke_by_tokens.{token_count}",
    )
    if (
        _expect_mapping(graph.get(backend), f"graph.{backend}").get("passed")
        is not True
    ):
        raise PlanningArtifactError(
            f"graph replay failed for tokens={token_count} backend={backend}"
        )


def _add_document_observations(
    builder: TimingCacheBuilder,
    document: Mapping[str, Any],
    *,
    path: Path,
    target: PlanTarget,
) -> tuple[str, str, int]:
    metadata, route, seed, dense_prefix = _validate_metadata(document, target)
    policy_key = _expect_mapping(metadata.get("policy_key"), "metadata.policy_key")
    graph_capture = policy_key.get("graph_mode") == "capture"
    expected_mode = "graph_replay" if graph_capture else "eager"
    if metadata.get("measurement_mode", "eager") != expected_mode:
        raise PlanningArtifactError("crossover measurement mode does not match shape")
    results = document.get("results")
    if not isinstance(results, list) or not results:
        raise PlanningArtifactError("crossover results must be a non-empty list")
    maximum = target.workload.tokens.maximum
    if dense_prefix > maximum:
        raise PlanningArtifactError(
            "crossover dense prefix exceeds the target token maximum"
        )
    backends, declared_tokens, rounds, repeats = _validate_measurement_design(
        metadata, dense_prefix=dense_prefix
    )
    _validate_measurement_results(
        results,
        backends=backends,
        tokens=declared_tokens,
        rounds=rounds,
        repeats=repeats,
        metadata=metadata,
    )
    _validate_oracle_evidence(document, metadata)
    if graph_capture:
        capture_sizes = target.workload.cudagraph_capture_sizes
        if not capture_sizes:
            raise PlanningArtifactError(
                "graph crossover evidence requires target capture sizes"
            )
        missing_capture_sizes = sorted(set(capture_sizes) - set(declared_tokens))
        if missing_capture_sizes:
            raise PlanningArtifactError(
                "graph crossover evidence does not cover target capture sizes: "
                f"missing={missing_capture_sizes}"
            )
    elif maximum not in declared_tokens:
        raise PlanningArtifactError(
            "eager crossover evidence does not cover target max tokens"
        )
    source_fingerprint = compute_artifact_fingerprint(dict(document))
    for index, raw_result in enumerate(results):
        result = _expect_mapping(raw_result, f"results[{index}]")
        backend = result.get("backend")
        if backend not in FUSED_MOE_BACKENDS:
            continue
        tokens = _expect_positive_int(result.get("tokens"), f"results[{index}].tokens")
        if tokens > maximum:
            continue
        samples = result.get("round_medians_ms")
        _validate_dispatch_identity(
            document,
            token_count=tokens,
            backend=backend,
            graph_capture=graph_capture,
        )
        minimum, bucket_maximum = _token_bucket(
            tokens,
            dense_prefix=dense_prefix,
            maximum=maximum,
        )
        case = BenchmarkCase.create_contextual(
            operation=FUSED_MOE_OPERATION,
            phase="operator",
            batch_size=1,
            tokens=tokens,
            operator_shape=policy_key,
            token_bucket_min=minimum,
            token_bucket_max=bucket_maximum,
            evidence_context={"route_mode": route, "seed": seed},
            dtype=target.model.dtype,
        )
        validate_tuning_domain_case(FUSED_MOE_FP8_BLOCK_DOMAIN_ID, case, target)
        builder.add_observation(
            f"runtime.musa.fused_moe:{backend}",
            samples,
            case=case,
            provenance={
                "evidence.path": str(path),
                "evidence.fingerprint": source_fingerprint,
                "measurement.mode": expected_mode,
            },
        )
    return compute_artifact_fingerprint(policy_key), route, seed


def import_fused_moe_crossover_evidence(
    *,
    target_path: str | Path,
    evidence_paths: Sequence[str | Path],
    output_path: str | Path,
    image_digest: str,
) -> dict[str, Any]:
    """Validate and normalize one exact route/seed matrix into timing-v3."""

    if not evidence_paths:
        raise PlanningArtifactError("at least one fused-MoE evidence file is required")
    target = _target_from_path(target_path)
    validate_tuning_domain_target(FUSED_MOE_FP8_BLOCK_DOMAIN_ID, target)
    if target.software.image_digest != image_digest:
        raise PlanningArtifactError("provided image digest does not match the target")
    builder = TimingCacheBuilder(
        target=target,
        catalog=fused_moe_tactic_definitions(),
        provenance={
            "collector": FUSED_MOE_CROSSOVER_SCHEMA,
            "image_digest": image_digest,
            TUNING_DOMAIN_PROVENANCE_KEY: FUSED_MOE_FP8_BLOCK_DOMAIN_ID,
        },
    )
    seen: set[tuple[str, str, int]] = set()
    for raw_path in evidence_paths:
        path = Path(raw_path)
        key = _add_document_observations(
            builder,
            load_json_object_file(path),
            path=path,
            target=target,
        )
        if key in seen:
            raise PlanningArtifactError(
                "duplicate fused-MoE shape/route/seed evidence is not independent"
            )
        seen.add(key)
    document = builder.build()
    write_json_object_file(output_path, document)
    return {
        "status": "written",
        "output": str(output_path),
        "schema_version": document["schema_version"],
        "evidence_files": len(evidence_paths),
        "observations": len(document["observations"]),
        "tuning_domain": FUSED_MOE_FP8_BLOCK_DOMAIN_ID,
        "fingerprint": document["fingerprint"],
    }


__all__ = [
    "FUSED_MOE_BACKENDS",
    "FUSED_MOE_CROSSOVER_SCHEMA",
    "FUSED_MOE_MAX_ERROR_THRESHOLDS",
    "FUSED_MOE_MIN_COSINE_THRESHOLDS",
    "FUSED_MOE_MIN_QUALIFIED_L2_FLUSH_MB",
    "FUSED_MOE_OPERATION",
    "fused_moe_tactic_definitions",
    "import_fused_moe_crossover_evidence",
]
