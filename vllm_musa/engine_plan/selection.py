# SPDX-License-Identifier: Apache-2.0

"""Canonical evidence-selection logic shared by builders and validators.

Keeping the policy gate in this dependency-free module is deliberate: plan
validation must replay exactly the same deterministic decision that produced a
plan, without importing the builder (and without maintaining a second copy of
the selection algorithm).
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .artifacts import (
    BENCHMARK_CASE_SCHEMA_VERSION,
    CONTEXTUAL_BENCHMARK_CASE_SCHEMA_VERSION,
    BenchmarkCase,
    ObservationStatus,
    PlanningArtifactError,
    TacticDefinition,
    TacticKind,
    TimingCache,
    TimingObservation,
    required_power2_bound_rows,
    static_workload_scope_differences,
)
from .tuning_domains import (
    FUSED_MOE_BACKENDS,
    FUSED_MOE_DISPATCH_OPERATION,
    FUSED_MOE_ROUTES,
    validate_fused_moe_case_structure,
)

ROWS_HIDDEN_BUCKET_SCHEMA = "musa.engine_coverage.rows_power2_bounds.v1"
CONTEXTUAL_SELECTION_SCHEMA = "musa.engine_selection.contextual.v1"
FUSED_MOE_CONTEXT_SCHEMA = "musa.fused_moe.context.v1"
FUSED_MOE_DISPATCH_POLICY_SCHEMA = "musa.fused_moe.dispatch_policy.v1"
FUSED_MOE_MAX_IQR_RATIO = 0.10
_ROWS_HIDDEN_OPERATIONS = frozenset({"fused_add_rms_norm"})
_BUCKET_MEAN_RUNTIME_DECISIONS = frozenset({"musa.fused_add_rms_norm.min_rows"})
_FUSED_MOE_BACKENDS = frozenset(FUSED_MOE_BACKENDS)
_FUSED_MOE_ROUTES = FUSED_MOE_ROUTES


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class BuildPolicy:
    """Deterministic evidence gate used by the offline builder and validator."""

    metric: str = "median_ms"
    min_samples: int = 3
    min_speedup_pct: float = 1.0
    tie_tolerance_pct: float = 0.5

    def __post_init__(self) -> None:
        if self.metric != "median_ms":
            raise PlanningArtifactError("Only median_ms is supported in schema v1")
        if (
            not isinstance(self.min_samples, int)
            or isinstance(self.min_samples, bool)
            or self.min_samples <= 0
        ):
            raise PlanningArtifactError("min_samples must be a positive integer")
        for field_name in ("min_speedup_pct", "tie_tolerance_pct"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not _is_finite_number(value)
                or value < 0
            ):
                raise PlanningArtifactError(
                    f"{field_name} must be a finite non-negative number"
                )

    @classmethod
    def from_document(cls, value: Any, *, field: str = "policy") -> BuildPolicy:
        if not isinstance(value, dict):
            raise PlanningArtifactError(f"{field} must be a JSON object")
        expected = {"metric", "min_samples", "min_speedup_pct", "tie_tolerance_pct"}
        actual = set(value)
        if actual != expected:
            raise PlanningArtifactError(
                f"{field} keys do not match schema; missing={sorted(expected - actual)}, "
                f"unknown={sorted(actual - expected)}"
            )
        try:
            return cls(
                metric=value["metric"],
                min_samples=value["min_samples"],
                min_speedup_pct=value["min_speedup_pct"],
                tie_tolerance_pct=value["tie_tolerance_pct"],
            )
        except (PlanningArtifactError, TypeError, ValueError) as exc:
            if isinstance(exc, PlanningArtifactError):
                raise PlanningArtifactError(f"{field}: {exc}") from exc
            raise PlanningArtifactError(f"{field} contains invalid values") from exc

    def to_document(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "min_samples": self.min_samples,
            "min_speedup_pct": float(self.min_speedup_pct),
            "tie_tolerance_pct": float(self.tie_tolerance_pct),
        }


@dataclass(frozen=True, slots=True)
class EvidenceSummary:
    median: float | None
    p95: float | None
    iqr: float | None
    samples: int
    rejection_reason: str | None


def _quantile(ordered: Sequence[float], q: float) -> float:
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


@dataclass(frozen=True, order=True, slots=True)
class EvidenceBucket:
    """Typed operator-shape coordinate used by the production evidence gate."""

    case_fingerprint: str
    phase: str
    batch_size: int
    tokens: int
    rows: int
    hidden_size: int
    dtype: str

    def to_document(self) -> dict[str, Any]:
        return {
            "case_fingerprint": self.case_fingerprint,
            "phase": self.phase,
            "batch_size": self.batch_size,
            "tokens": self.tokens,
            "rows": self.rows,
            "hidden_size": self.hidden_size,
            "dtype": self.dtype,
        }


def _evidence_bucket_from_case(case: BenchmarkCase) -> EvidenceBucket:
    return EvidenceBucket(
        case_fingerprint=case.fingerprint,
        phase=case.phase,
        batch_size=case.batch_size,
        tokens=case.tokens,
        rows=case.rows,
        hidden_size=case.hidden_size,
        dtype=case.dtype,
    )


def _observation_bucket(
    observation: TimingObservation,
    *,
    kind: TacticKind,
    operation: str,
    timing_cache: TimingCache,
) -> EvidenceBucket:
    case = observation.case
    if case is None:
        raise PlanningArtifactError("typed benchmark case is missing")
    if kind is TacticKind.VLLM_IR_PROVIDER and operation not in _ROWS_HIDDEN_OPERATIONS:
        raise PlanningArtifactError(
            f"operation {operation!r} has no production coverage schema"
        )
    if case.operation != operation:
        raise PlanningArtifactError("bucket operation does not match tactic operation")
    if case.dtype != timing_cache.target.model.dtype:
        raise PlanningArtifactError("bucket dtype does not match target model dtype")
    if case.hidden_size != timing_cache.target.model.hidden_size:
        raise PlanningArtifactError(
            "bucket hidden_size does not match target model hidden_size"
        )
    if case.phase != "operator" or case.batch_size != 1:
        raise PlanningArtifactError(
            "fused_add_rms_norm cases require phase=operator and batch_size=1"
        )
    if case.rows != case.tokens:
        raise PlanningArtifactError(
            "fused_add_rms_norm case rows must equal the flattened token count"
        )
    workload = timing_cache.target.workload
    if not workload.tokens.minimum <= case.tokens <= workload.tokens.maximum:
        raise PlanningArtifactError("benchmark case is outside target token scope")
    return _evidence_bucket_from_case(case)


def summarize_observations(
    observations: Sequence[TimingObservation],
    policy: BuildPolicy,
) -> EvidenceSummary:
    if not observations:
        return EvidenceSummary(None, None, None, 0, "not_measured")
    values: list[float] = []
    rejection_reasons: list[str] = []
    for observation in observations:
        if observation.status is not ObservationStatus.PASSED:
            rejection_reasons.append("measurement_failed")
            continue
        if observation.correctness != "passed":
            rejection_reasons.append(f"correctness_{observation.correctness}")
            continue
        if observation.metric != policy.metric:
            rejection_reasons.append(
                f"metric_{observation.metric}_does_not_match_{policy.metric}"
            )
            continue
        values.extend(observation.values)
    if rejection_reasons:
        priority = {
            "correctness_failed": 0,
            "correctness_missing": 1,
            "measurement_failed": 2,
        }
        reason = min(
            set(rejection_reasons),
            key=lambda item: (priority.get(item, 3), item),
        )
        ordered = sorted(values)
        return EvidenceSummary(
            float(statistics.median(ordered)) if ordered else None,
            _quantile(ordered, 0.95) if ordered else None,
            (_quantile(ordered, 0.75) - _quantile(ordered, 0.25) if ordered else None),
            len(ordered),
            reason,
        )
    if len(values) < policy.min_samples:
        reason = f"samples_{len(values)}_below_{policy.min_samples}"
        return EvidenceSummary(
            float(statistics.median(values)) if values else None,
            _quantile(sorted(values), 0.95) if values else None,
            (
                _quantile(sorted(values), 0.75) - _quantile(sorted(values), 0.25)
                if values
                else None
            ),
            len(values),
            reason,
        )
    ordered = sorted(values)
    return EvidenceSummary(
        median=float(statistics.median(ordered)),
        p95=_quantile(ordered, 0.95),
        iqr=_quantile(ordered, 0.75) - _quantile(ordered, 0.25),
        samples=len(ordered),
        rejection_reason=None,
    )


def _bucket_summaries(
    observations: Sequence[TimingObservation],
    policy: BuildPolicy,
    *,
    kind: TacticKind,
    operation: str,
    timing_cache: TimingCache,
) -> dict[EvidenceBucket, EvidenceSummary]:
    buckets: dict[EvidenceBucket, list[TimingObservation]] = {}
    for observation in observations:
        bucket = _observation_bucket(
            observation,
            kind=kind,
            operation=operation,
            timing_cache=timing_cache,
        )
        buckets.setdefault(bucket, []).append(observation)
    return {
        bucket: summarize_observations(items, policy)
        for bucket, items in sorted(buckets.items())
    }


def _required_serving_buckets(
    timing_cache: TimingCache,
    *,
    operation: str,
    hidden_size: int,
) -> tuple[EvidenceBucket, ...]:
    maximum = timing_cache.target.workload.max_num_batched_tokens
    return tuple(
        _evidence_bucket_from_case(
            BenchmarkCase.create(
                operation=operation,
                phase="operator",
                batch_size=1,
                tokens=row,
                rows=row,
                hidden_size=hidden_size,
                dtype=timing_cache.target.model.dtype,
            )
        )
        for row in required_power2_bound_rows(maximum)
    )


def _coverage_and_regression_reason(
    candidate: Sequence[TimingObservation],
    fallback: Sequence[TimingObservation],
    policy: BuildPolicy,
    *,
    timing_cache: TimingCache,
    kind: TacticKind,
    operation: str,
) -> tuple[dict[str, Any], str | None]:
    activation_differences = static_workload_scope_differences(
        timing_cache.target.workload
    )
    base_manifest: dict[str, Any] = {
        "schema": ROWS_HIDDEN_BUCKET_SCHEMA,
        "case_schema": BENCHMARK_CASE_SCHEMA_VERSION,
        "status": "evidence_only" if activation_differences else "incomplete",
        "activation_differences": list(activation_differences),
        "required_buckets": [],
        "observed_buckets": [],
        "error": None,
    }
    try:
        candidate_buckets = _bucket_summaries(
            candidate,
            policy,
            kind=kind,
            operation=operation,
            timing_cache=timing_cache,
        )
        fallback_buckets = _bucket_summaries(
            fallback,
            policy,
            kind=kind,
            operation=operation,
            timing_cache=timing_cache,
        )
    except PlanningArtifactError as exc:
        reason = (
            "bucket_provenance_missing"
            if "is missing" in str(exc)
            else "bucket_provenance_invalid"
        )
        base_manifest["error"] = str(exc)
        return base_manifest, reason

    observed = sorted(candidate_buckets.keys() & fallback_buckets.keys())
    base_manifest["observed_buckets"] = [item.to_document() for item in observed]
    if candidate_buckets.keys() != fallback_buckets.keys():
        return base_manifest, "bucket_coverage_mismatch"

    hidden_sizes = {bucket.hidden_size for bucket in candidate_buckets}
    if hidden_sizes != {timing_cache.target.model.hidden_size}:
        return base_manifest, "bucket_hidden_size_mismatch"
    if not activation_differences:
        required = _required_serving_buckets(
            timing_cache,
            operation=operation,
            hidden_size=timing_cache.target.model.hidden_size,
        )
        base_manifest["required_buckets"] = [item.to_document() for item in required]
        if not set(required).issubset(candidate_buckets):
            return base_manifest, "bucket_required_coverage_missing"
        base_manifest["status"] = "complete"

    for bucket in candidate_buckets:
        candidate_summary = candidate_buckets[bucket]
        fallback_summary = fallback_buckets[bucket]
        if candidate_summary.rejection_reason is not None:
            return base_manifest, candidate_summary.rejection_reason
        if fallback_summary.rejection_reason is not None:
            return base_manifest, f"fallback_{fallback_summary.rejection_reason}"
        assert candidate_summary.median is not None
        assert fallback_summary.median is not None
        allowed = fallback_summary.median * (1.0 + policy.tie_tolerance_pct / 100.0)
        if candidate_summary.median > allowed:
            return base_manifest, "bucket_regression_above_tie_tolerance"
    return base_manifest, None


def _selection_id(timing_cache: TimingCache, operation: str, winner: str) -> str:
    payload = json.dumps(
        {
            "timing_cache": timing_cache.fingerprint,
            "operation": operation,
            "winner": winner,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"selection-{hashlib.sha256(payload).hexdigest()[:16]}"


def _selection_value(
    *,
    tactic: TacticDefinition,
    observations: Sequence[TimingObservation],
    summary: EvidenceSummary,
    policy: BuildPolicy,
    timing_cache: TimingCache,
) -> float:
    """Return the deterministic score used to rank one eligible tactic.

    Runtime decisions such as a compile-range crossover may affect only one
    representative bucket. Flattening all samples and taking one median would
    hide that improvement, so score those decisions by the equal-weight mean
    of per-bucket medians after the existing no-regression gate has passed.
    """

    assert summary.median is not None
    if tactic.operation not in _BUCKET_MEAN_RUNTIME_DECISIONS:
        return float(summary.median)
    buckets = _bucket_summaries(
        observations,
        policy,
        kind=tactic.kind,
        operation=tactic.operation,
        timing_cache=timing_cache,
    )
    medians = [
        float(bucket.median) for bucket in buckets.values() if bucket.median is not None
    ]
    if not medians:
        return float(summary.median)
    return sum(medians) / len(medians)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fused_moe_case_keys(
    case: BenchmarkCase,
    timing_cache: TimingCache,
) -> tuple[tuple[str, int, int], tuple[int, str]]:
    shape, evidence = validate_fused_moe_case_structure(case, timing_cache.target)
    assert case.token_bucket is not None
    return (
        (
            _canonical_json(shape),
            case.token_bucket.minimum,
            case.token_bucket.maximum,
        ),
        (case.tokens, _canonical_json(evidence)),
    )


def _contextual_observations(
    items: Sequence[TimingObservation],
    timing_cache: TimingCache,
) -> dict[tuple[str, int, int], dict[tuple[int, str], list[TimingObservation]]]:
    grouped: dict[
        tuple[str, int, int], dict[tuple[int, str], list[TimingObservation]]
    ] = {}
    for observation in items:
        if observation.case is None:
            raise PlanningArtifactError("contextual timing observation is missing case")
        runtime_key, evidence_key = _fused_moe_case_keys(
            observation.case,
            timing_cache,
        )
        grouped.setdefault(runtime_key, {}).setdefault(evidence_key, []).append(
            observation
        )
    return grouped


def _reachable_fused_moe_contexts(
    contexts: Mapping[
        tuple[str, int, int],
        Mapping[tuple[int, str], Sequence[TimingObservation]],
    ],
    timing_cache: TimingCache,
) -> dict[tuple[str, int, int], dict[tuple[int, str], list[TimingObservation]]]:
    """Project capture evidence onto vLLM's reachable padded graph keys."""

    capture_shapes = {
        shape_json
        for shape_json, _, _ in contexts
        if json.loads(shape_json)["graph_mode"] == "capture"
    }
    if not capture_shapes:
        return {
            key: {evidence: list(items) for evidence, items in value.items()}
            for key, value in contexts.items()
        }

    capture_sizes = timing_cache.target.workload.cudagraph_capture_sizes
    if not capture_sizes:
        raise PlanningArtifactError(
            "contextual fused-MoE capture selection requires "
            "target.workload.cudagraph_capture_sizes"
        )
    graph_mode = timing_cache.target.workload.graph_mode
    if graph_mode == "NONE":
        raise PlanningArtifactError(
            "contextual fused-MoE capture evidence requires a graph-enabled " "target"
        )
    workload = timing_cache.target.workload.tokens
    reachable_sizes = tuple(
        size for size in capture_sizes if workload.minimum <= size <= workload.maximum
    )
    if graph_mode == "FULL_DECODE_ONLY":
        max_num_seqs = timing_cache.target.workload.max_num_seqs
        reachable_sizes = tuple(
            size for size in reachable_sizes if size <= max_num_seqs
        )
    if not reachable_sizes:
        raise PlanningArtifactError(
            "cudagraph_capture_sizes contains no key inside the target token scope"
        )

    normalized = {
        key: {evidence: list(items) for evidence, items in value.items()}
        for key, value in contexts.items()
        if key[0] not in capture_shapes
    }
    for shape_json in sorted(capture_shapes):
        shape_contexts = {
            key: value for key, value in contexts.items() if key[0] == shape_json
        }
        previous = workload.minimum - 1
        for index, size in enumerate(reachable_sizes):
            matching = [key for key in shape_contexts if key[1] <= size <= key[2]]
            if len(matching) != 1:
                raise PlanningArtifactError(
                    "fused-MoE capture evidence must cover every configured "
                    f"graph key exactly once; size={size}, matches={matching}"
                )
            source_evidence = shape_contexts[matching[0]]
            evidence = {
                key: list(items)
                for key, items in source_evidence.items()
                if key[0] == size
            }
            if not evidence:
                raise PlanningArtifactError(
                    "fused-MoE capture graph key lacks exact measured evidence: "
                    f"size={size}"
                )
            maximum = workload.maximum if index == len(reachable_sizes) - 1 else size
            normalized[(shape_json, previous + 1, maximum)] = evidence
            previous = size
    return normalized


def _required_fused_moe_evidence_reason(
    evidence_keys: Sequence[tuple[int, str]],
) -> str | None:
    by_tokens: dict[int, dict[str, set[int]]] = {}
    for tokens, evidence_json in evidence_keys:
        evidence = json.loads(evidence_json)
        by_tokens.setdefault(tokens, {}).setdefault(evidence["route_mode"], set()).add(
            evidence["seed"]
        )
    for routes in by_tokens.values():
        if set(routes) != _FUSED_MOE_ROUTES:
            return "required_route_modes_missing"
        seed_sets = list(routes.values())
        if any(len(seeds) < 3 for seeds in seed_sets):
            return "required_independent_seeds_missing"
        if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
            return "route_seed_matrix_mismatch"
    return None


def _context_summaries(
    evidence: Mapping[tuple[int, str], Sequence[TimingObservation]],
    policy: BuildPolicy,
) -> dict[tuple[int, str], EvidenceSummary]:
    return {
        key: summarize_observations(items, policy)
        for key, items in sorted(evidence.items())
    }


def _context_evidence_document(
    keys: Sequence[tuple[int, str]],
    winner: Mapping[tuple[int, str], EvidenceSummary],
    fallback: Mapping[tuple[int, str], EvidenceSummary],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for key in sorted(keys):
        tokens, evidence_json = key
        winner_summary = winner[key]
        fallback_summary = fallback[key]
        assert winner_summary.median is not None
        assert winner_summary.iqr is not None
        assert fallback_summary.median is not None
        assert fallback_summary.iqr is not None
        item = {
            "workload_tokens": tokens,
            **json.loads(evidence_json),
            "winner_median": winner_summary.median,
            "winner_p95": winner_summary.p95,
            "winner_iqr": winner_summary.iqr,
            "fallback_median": fallback_summary.median,
            "fallback_p95": fallback_summary.p95,
            "fallback_iqr": fallback_summary.iqr,
            "speedup_pct": max(
                0.0,
                (fallback_summary.median - winner_summary.median)
                / fallback_summary.median
                * 100.0,
            ),
            "samples": winner_summary.samples,
        }
        result.append(item)
    return result


def _smooth_fused_moe_compile_range_boundaries(
    contexts: list[dict[str, Any]],
    *,
    fallback_id: str,
) -> None:
    """Make the token-1/2 policy representable by vLLM compile ranges.

    vLLM creates the first fused-MoE compile range for tokens one and two
    together.  A backend transition between those points cannot activate at
    config time, so prefer the registered fallback for both points instead of
    promoting either unproven tactic across the boundary.
    """

    by_shape: dict[str, list[dict[str, Any]]] = {}
    for context in contexts:
        shape_key = json.dumps(context["shape"], sort_keys=True, separators=(",", ":"))
        by_shape.setdefault(shape_key, []).append(context)

    reason = "fallback_required:compile_range_token_1_2_backend_transition"
    for shape_contexts in by_shape.values():
        token_contexts: dict[int, dict[str, Any]] = {}
        for token in (1, 2):
            matches = [
                context
                for context in shape_contexts
                if context["token_bucket"]["min"]
                <= token
                <= context["token_bucket"]["max"]
            ]
            if len(matches) == 1:
                token_contexts[token] = matches[0]
        if len(token_contexts) != 2:
            continue
        first = token_contexts[1]
        second = token_contexts[2]
        if first is second or first["winner"] == second["winner"]:
            continue
        for context in (first, second):
            original_winner = context["winner"]
            if original_winner != fallback_id:
                context["rejected"].append(
                    {
                        "tactic_id": original_winner,
                        "reason": reason,
                    }
                )
                context["rejected"].sort(
                    key=lambda item: (item["tactic_id"], item["reason"])
                )
            context["winner"] = fallback_id
            context["speedup_pct"] = 0.0
            context["reason"] = reason
            for evidence in context["evidence"]:
                evidence["winner_median"] = evidence["fallback_median"]
                evidence["winner_p95"] = evidence["fallback_p95"]
                evidence["winner_iqr"] = evidence["fallback_iqr"]
                evidence["speedup_pct"] = 0.0


def _contextual_selection_id(
    timing_cache: TimingCache,
    operation: str,
    contexts: Sequence[Mapping[str, Any]],
) -> str:
    payload = {
        "timing_cache": timing_cache.fingerprint,
        "operation": operation,
        "contexts": [
            {
                "context_id": context["context_id"],
                "winner": context["winner"],
            }
            for context in contexts
        ],
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"selection-{digest[:16]}"


def _select_contextual_operation(
    timing_cache: TimingCache,
    tactics: Sequence[TacticDefinition],
    observations: Mapping[str, Sequence[TimingObservation]],
    policy: BuildPolicy,
) -> dict[str, Any]:
    if tactics[0].operation != FUSED_MOE_DISPATCH_OPERATION:
        raise PlanningArtifactError(
            f"operation {tactics[0].operation!r} has no contextual selection schema"
        )
    if any(tactic.kind is not TacticKind.RUNTIME_DECISION for tactic in tactics):
        raise PlanningArtifactError("fused-MoE tactics must be runtime decisions")
    choices = {tactic.choice for tactic in tactics}
    if not choices.issubset(_FUSED_MOE_BACKENDS) or any(
        not isinstance(choice, str) for choice in choices
    ):
        raise PlanningArtifactError("fused-MoE tactics contain an unknown backend")
    fallback_ids = {tactic.fallback_id for tactic in tactics}
    if len(fallback_ids) != 1:
        raise PlanningArtifactError("fused-MoE tactics must share one fallback")
    fallback_id = next(iter(fallback_ids))
    by_id = {tactic.tactic_id: tactic for tactic in tactics}
    fallback = by_id.get(fallback_id)
    if fallback is None or fallback.choice != "upstream":
        raise PlanningArtifactError("fused-MoE fallback must be upstream")

    grouped = {
        tactic.tactic_id: _contextual_observations(
            observations.get(tactic.tactic_id, ()), timing_cache
        )
        for tactic in tactics
    }
    grouped = {
        tactic_id: _reachable_fused_moe_contexts(contexts, timing_cache)
        for tactic_id, contexts in grouped.items()
    }
    fallback_contexts = grouped[fallback_id]
    if not fallback_contexts:
        raise PlanningArtifactError(
            "fused-MoE contextual selection requires fallback evidence"
        )
    extra_contexts = {
        key
        for tactic_id, contexts in grouped.items()
        if tactic_id != fallback_id
        for key in contexts
        if key not in fallback_contexts
    }
    if extra_contexts:
        raise PlanningArtifactError(
            "candidate evidence contains contexts absent from the fallback"
        )
    for shape_json in {key[0] for key in fallback_contexts}:
        ranges = sorted(
            (minimum, maximum)
            for candidate_shape, minimum, maximum in fallback_contexts
            if candidate_shape == shape_json
        )
        expected_minimum = 1
        for previous, current in zip(ranges, ranges[1:]):
            if current[0] <= previous[1]:
                raise PlanningArtifactError(
                    "fused-MoE evidence contains overlapping token buckets"
                )
        for minimum, maximum in ranges:
            if minimum != expected_minimum:
                raise PlanningArtifactError(
                    "fused-MoE token buckets must start at one and be contiguous"
                )
            expected_minimum = maximum + 1

    contexts: list[dict[str, Any]] = []
    for runtime_key, fallback_evidence in sorted(fallback_contexts.items()):
        shape_json, minimum, maximum = runtime_key
        fallback_summaries = _context_summaries(fallback_evidence, policy)
        fallback_reason = _required_fused_moe_evidence_reason(tuple(fallback_summaries))
        if fallback_reason is None:
            fallback_reason = next(
                (
                    "fallback_evidence_context_unstable_iqr"
                    for summary in fallback_summaries.values()
                    if summary.median is not None
                    and summary.iqr is not None
                    and summary.iqr / summary.median > FUSED_MOE_MAX_IQR_RATIO
                ),
                None,
            )
        if fallback_reason is None:
            invalid_fallback_reason = next(
                (
                    f"fallback_{summary.rejection_reason}"
                    for summary in fallback_summaries.values()
                    if summary.rejection_reason is not None
                ),
                None,
            )
            if invalid_fallback_reason is not None:
                raise PlanningArtifactError(
                    "fused-MoE fallback evidence is invalid: "
                    f"{invalid_fallback_reason}"
                )
        eligible: list[
            tuple[float, TacticDefinition, dict[tuple[int, str], EvidenceSummary]]
        ] = []
        rejected: list[dict[str, str]] = []
        if fallback_reason is None:
            eligible.append((1.0, fallback, fallback_summaries))
        for tactic in sorted(tactics, key=lambda item: item.tactic_id):
            if tactic.tactic_id == fallback_id:
                continue
            tactic_evidence = grouped[tactic.tactic_id].get(runtime_key, {})
            reason: str | None = None
            shape = json.loads(shape_json)
            if tactic.choice == "grouped_gemm" and shape["graph_mode"] == "capture":
                reason = "candidate_unsupported_during_graph_capture"
            if tactic_evidence.keys() != fallback_evidence.keys():
                reason = reason or "evidence_context_coverage_mismatch"
            tactic_summaries = _context_summaries(tactic_evidence, policy)
            if reason is None:
                reason = next(
                    (
                        f"evidence_context_{summary.rejection_reason}"
                        for summary in tactic_summaries.values()
                        if summary.rejection_reason is not None
                    ),
                    None,
                )
            ratios: list[float] = []
            if reason is None:
                for key, summary in tactic_summaries.items():
                    fallback_summary = fallback_summaries[key]
                    assert summary.median is not None
                    assert summary.p95 is not None
                    assert summary.iqr is not None
                    assert fallback_summary.median is not None
                    assert fallback_summary.p95 is not None
                    assert fallback_summary.iqr is not None
                    if summary.iqr / summary.median > FUSED_MOE_MAX_IQR_RATIO:
                        reason = "evidence_context_unstable_iqr"
                        break
                    ratio = summary.median / fallback_summary.median
                    ratios.append(ratio)
                    if ratio > 1.0 + policy.tie_tolerance_pct / 100.0:
                        reason = "evidence_context_regression_above_tie_tolerance"
                        break
                    if summary.p95 > fallback_summary.p95 * (
                        1.0 + policy.tie_tolerance_pct / 100.0
                    ):
                        reason = "evidence_context_p95_regression_above_guardrail"
                        break
                    speedup = (1.0 - ratio) * 100.0
                    if speedup < policy.min_speedup_pct:
                        reason = "evidence_context_speedup_below_minimum"
                        break
            if reason is None:
                eligible.append((max(ratios), tactic, tactic_summaries))
            else:
                rejected.append({"tactic_id": tactic.tactic_id, "reason": reason})

        if fallback_reason is not None:
            winner = fallback
            winner_summaries = fallback_summaries
            reason = f"fallback_required:{fallback_reason}"
        else:
            eligible.sort(key=lambda item: (item[0], item[1].tactic_id))
            best_ratio = eligible[0][0]
            tied = [
                item
                for item in eligible
                if item[0] <= best_ratio * (1.0 + policy.tie_tolerance_pct / 100.0)
            ]
            fallback_tie = next(
                (item for item in tied if item[1].tactic_id == fallback_id),
                None,
            )
            _, winner, winner_summaries = fallback_tie or min(
                tied, key=lambda item: item[1].tactic_id
            )
            reason = (
                "fallback_fastest_or_insufficient_robust_speedup"
                if winner.tactic_id == fallback_id
                else "lowest_eligible_worst_context_ratio"
            )
        evidence = _context_evidence_document(
            tuple(fallback_summaries), winner_summaries, fallback_summaries
        )
        speedup_pct = min(item["speedup_pct"] for item in evidence)
        context_document = {
            "shape": json.loads(shape_json),
            "token_bucket": {"min": minimum, "max": maximum},
        }
        context_id = compute_context_fingerprint(context_document)
        contexts.append(
            {
                "context_id": context_id,
                **context_document,
                "winner": winner.tactic_id,
                "fallback": fallback_id,
                "metric": "worst_evidence_context_ratio",
                "speedup_pct": speedup_pct,
                "samples": sum(item["samples"] for item in evidence),
                "reason": reason,
                "evidence": evidence,
                "rejected": sorted(rejected, key=lambda item: item["tactic_id"]),
            }
        )
    _smooth_fused_moe_compile_range_boundaries(
        contexts,
        fallback_id=fallback_id,
    )
    return {
        "selection_schema": CONTEXTUAL_SELECTION_SCHEMA,
        "selection_id": _contextual_selection_id(
            timing_cache, tactics[0].operation, contexts
        ),
        "context_schema": FUSED_MOE_CONTEXT_SCHEMA,
        "max_iqr_ratio": FUSED_MOE_MAX_IQR_RATIO,
        "kind": TacticKind.RUNTIME_DECISION.value,
        "operation": tactics[0].operation,
        "fallback": fallback_id,
        "contexts": contexts,
    }


def compute_context_fingerprint(context: Mapping[str, Any]) -> str:
    payload = _canonical_json(context).encode("utf-8")
    return f"context-{hashlib.sha256(payload).hexdigest()[:16]}"


def select_operation(
    timing_cache: TimingCache,
    tactics: Sequence[TacticDefinition],
    observations: Mapping[str, Sequence[TimingObservation]],
    policy: BuildPolicy,
) -> dict[str, Any]:
    """Select one tactic using the canonical deterministic evidence gate."""

    case_schemas = {
        observation.case.schema_version
        for tactic in tactics
        for observation in observations.get(tactic.tactic_id, ())
        if observation.case is not None
    }
    if CONTEXTUAL_BENCHMARK_CASE_SCHEMA_VERSION in case_schemas:
        if case_schemas != {CONTEXTUAL_BENCHMARK_CASE_SCHEMA_VERSION}:
            raise PlanningArtifactError(
                "one operation cannot mix legacy and contextual benchmark cases"
            )
        return _select_contextual_operation(
            timing_cache,
            tactics,
            observations,
            policy,
        )

    fallback_ids = {tactic.fallback_id for tactic in tactics}
    if len(fallback_ids) != 1:
        raise PlanningArtifactError(
            f"Operation {tactics[0].operation!r} has multiple fallbacks: "
            f"{sorted(fallback_ids)}"
        )
    fallback_id = next(iter(fallback_ids))
    by_id = {tactic.tactic_id: tactic for tactic in tactics}
    fallback = by_id.get(fallback_id)
    if fallback is None:
        raise PlanningArtifactError(
            f"Operation {tactics[0].operation!r} fallback is outside its group"
        )

    rejected: list[dict[str, Any]] = []
    eligible: list[tuple[TacticDefinition, EvidenceSummary]] = []
    coverage_by_tactic: dict[str, dict[str, Any]] = {}
    fallback_observations = observations.get(fallback_id, ())
    fallback_coverage, _ = _coverage_and_regression_reason(
        fallback_observations,
        fallback_observations,
        policy,
        timing_cache=timing_cache,
        kind=tactics[0].kind,
        operation=tactics[0].operation,
    )
    coverage_by_tactic[fallback_id] = fallback_coverage
    for tactic in sorted(tactics, key=lambda item: item.tactic_id):
        tactic_observations = observations.get(tactic.tactic_id, ())
        summary = summarize_observations(tactic_observations, policy)
        reason = summary.rejection_reason
        if reason is None and tactic.tactic_id != fallback_id:
            coverage, reason = _coverage_and_regression_reason(
                tactic_observations,
                fallback_observations,
                policy,
                timing_cache=timing_cache,
                kind=tactic.kind,
                operation=tactic.operation,
            )
            coverage_by_tactic[tactic.tactic_id] = coverage
        if reason is None:
            eligible.append((tactic, summary))
        else:
            rejected.append(
                {
                    "tactic_id": tactic.tactic_id,
                    "reason": reason,
                    "median": summary.median,
                    "samples": summary.samples,
                }
            )

    fallback_observation = summarize_observations(fallback_observations, policy)
    fallback_reason = fallback_observation.rejection_reason
    if fallback_reason is not None:
        winner = fallback
        winner_observation = fallback_observation
        reason = f"fallback_required:{fallback_reason}"
        speedup_pct = 0.0
    else:
        comparable = [
            (
                tactic,
                observation,
                _selection_value(
                    tactic=tactic,
                    observations=observations.get(tactic.tactic_id, ()),
                    summary=observation,
                    policy=policy,
                    timing_cache=timing_cache,
                ),
            )
            for tactic, observation in eligible
            if observation.median is not None
        ]
        if not comparable:
            winner = fallback
            winner_observation = fallback_observation
            reason = "fallback_required:no_eligible_candidate"
            speedup_pct = 0.0
        else:
            comparable.sort(key=lambda item: (item[2], item[0].tactic_id))
            best_value = comparable[0][2]
            tolerance = best_value * policy.tie_tolerance_pct / 100.0
            tied = [item for item in comparable if item[2] <= best_value + tolerance]
            fallback_tie = next(
                (item for item in tied if item[0].tactic_id == fallback_id),
                None,
            )
            winner, winner_observation, winner_value = fallback_tie or min(
                tied,
                key=lambda item: item[0].tactic_id,
            )
            assert fallback_observation.median is not None
            fallback_value = next(
                item[2] for item in comparable if item[0].tactic_id == fallback_id
            )
            speedup_pct = max(
                0.0,
                (fallback_value - winner_value) / fallback_value * 100.0,
            )
            if winner.tactic_id != fallback_id and speedup_pct < policy.min_speedup_pct:
                rejected.append(
                    {
                        "tactic_id": winner.tactic_id,
                        "reason": (
                            f"speedup_{speedup_pct:.6f}_below_"
                            f"{policy.min_speedup_pct:.6f}"
                        ),
                        "median": winner_observation.median,
                        "samples": winner_observation.samples,
                    }
                )
                winner = fallback
                winner_observation = fallback_observation
                reason = "fallback_required:below_min_speedup"
                speedup_pct = 0.0
            elif winner.tactic_id == fallback_id:
                reason = "fallback_fastest_or_within_tie_tolerance"
                speedup_pct = 0.0
            else:
                reason = (
                    "lowest_eligible_bucket_mean"
                    if winner.operation in _BUCKET_MEAN_RUNTIME_DECISIONS
                    else "lowest_eligible_median"
                )

    output_winner_value = winner_observation.median
    output_fallback_value = fallback_observation.median
    output_metric = policy.metric
    if (
        winner.operation in _BUCKET_MEAN_RUNTIME_DECISIONS
        and winner_observation.rejection_reason is None
        and fallback_observation.rejection_reason is None
    ):
        output_winner_value = _selection_value(
            tactic=winner,
            observations=observations.get(winner.tactic_id, ()),
            summary=winner_observation,
            policy=policy,
            timing_cache=timing_cache,
        )
        output_fallback_value = _selection_value(
            tactic=fallback,
            observations=fallback_observations,
            summary=fallback_observation,
            policy=policy,
            timing_cache=timing_cache,
        )
        output_metric = "bucket_mean_of_medians_ms"

    return {
        "selection_id": _selection_id(
            timing_cache,
            tactics[0].operation,
            winner.tactic_id,
        ),
        "kind": winner.kind.value,
        "operation": winner.operation,
        "winner": winner.tactic_id,
        "fallback": fallback_id,
        "metric": output_metric,
        "winner_value": output_winner_value,
        "winner_p95": winner_observation.p95,
        "fallback_value": output_fallback_value,
        "fallback_p95": fallback_observation.p95,
        "speedup_pct": speedup_pct,
        "samples": winner_observation.samples,
        "reason": reason,
        "coverage": coverage_by_tactic[winner.tactic_id],
        "rejected": sorted(rejected, key=lambda item: item["tactic_id"]),
    }


def build_runtime_decisions(
    timing_cache: TimingCache,
    selections: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive one typed runtime-decision projection from sealed selections."""

    by_id = {tactic.tactic_id: tactic for tactic in timing_cache.catalog}
    priorities: dict[str, list[str]] = {}
    decisions: dict[str, Any] = {}
    for selection in selections:
        if selection.get("selection_schema") == CONTEXTUAL_SELECTION_SCHEMA:
            operation = selection["operation"]
            if operation in decisions:
                raise PlanningArtifactError(
                    f"Runtime decision {operation!r} was selected twice"
                )
            entries_by_shape: dict[str, list[dict[str, Any]]] = {}
            for context in selection["contexts"]:
                winner = by_id[context["winner"]]
                if not isinstance(winner.choice, str):
                    raise PlanningArtifactError(
                        "contextual runtime tactic choice must be a string"
                    )
                if winner.choice not in _FUSED_MOE_BACKENDS:
                    raise PlanningArtifactError(
                        f"Unknown fused-MoE backend {winner.choice!r}"
                    )
                shape_json = _canonical_json(context["shape"])
                bucket = context["token_bucket"]
                entries_by_shape.setdefault(shape_json, []).append(
                    {
                        "min_tokens": bucket["min"],
                        "max_tokens": bucket["max"],
                        "backend": winner.choice,
                    }
                )
            entries: list[dict[str, Any]] = []
            for shape_json, raw_ranges in sorted(entries_by_shape.items()):
                coalesced: list[dict[str, Any]] = []
                for item in sorted(
                    raw_ranges,
                    key=lambda value: (value["min_tokens"], value["max_tokens"]),
                ):
                    if (
                        coalesced
                        and coalesced[-1]["backend"] == item["backend"]
                        and coalesced[-1]["max_tokens"] + 1 == item["min_tokens"]
                    ):
                        coalesced[-1]["max_tokens"] = item["max_tokens"]
                    else:
                        coalesced.append(dict(item))
                entries.append({"shape": json.loads(shape_json), "ranges": coalesced})
            decisions[operation] = {
                "schema": FUSED_MOE_DISPATCH_POLICY_SCHEMA,
                "entries": entries,
            }
            continue
        winner = by_id[selection["winner"]]
        fallback = by_id[selection["fallback"]]
        if winner.kind is TacticKind.VLLM_IR_PROVIDER:
            assert isinstance(winner.choice, str)
            assert isinstance(fallback.choice, str)
            providers = [winner.choice]
            if fallback.choice not in providers:
                providers.append(fallback.choice)
            priorities[winner.operation] = providers
        elif winner.kind is TacticKind.RUNTIME_DECISION:
            if winner.operation in decisions:
                raise PlanningArtifactError(
                    f"Runtime decision {winner.operation!r} was selected twice"
                )
            decisions[winner.operation] = winner.choice

    if priorities:
        decisions["vllm.ir_op_priority"] = dict(sorted(priorities.items()))
    return {
        "profile": timing_cache.target.model.profile,
        "values": dict(sorted(decisions.items())),
    }
