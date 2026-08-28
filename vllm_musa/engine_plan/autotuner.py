# SPDX-License-Identifier: Apache-2.0

"""Runtime-neutral incremental AutoTuner orchestration.

The collector deliberately owns no torch or MUSA code.  Operation adapters
enumerate legal runners and perform measurements; this module owns cache keys,
candidate-set invalidation, correctness/status evidence, and timing-v2 output.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .artifacts import (
    BenchmarkCase,
    ObservationStatus,
    PlanningArtifactError,
    PlanTarget,
    TacticDefinition,
    TimingCache,
    TimingCacheBuilder,
    TimingObservation,
    compute_artifact_fingerprint,
    diff_runtime_targets,
)

AUTOTUNE_PROTOCOL_VERSION = "musa.engine_autotune.v1"
AUTOTUNE_PROTOCOL_KEY = "autotune.protocol"
AUTOTUNE_KEY = "autotune.key"
AUTOTUNE_CANDIDATE_SET_KEY = "autotune.candidate_set"
AUTOTUNE_TOOLCHAIN_KEY = "autotune.toolchain"
AUTOTUNE_MEASUREMENT_KEY = "autotune.measurement"
AUTOTUNE_MEASURED_TARGET_KEY = "autotune.measured_target"


@dataclass(frozen=True, slots=True)
class CandidateMeasurement:
    """One logical tactic measurement returned by an operation adapter."""

    values_ms: tuple[float, ...]
    status: ObservationStatus
    correctness: str
    provenance: tuple[tuple[str, str], ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class TuningContext:
    operation: str
    case: BenchmarkCase
    candidate_set_fingerprint: str
    toolchain_fingerprint: str
    measurement_fingerprint: str
    key: str


class TunableRunner(Protocol):
    """TRT-LLM-style runner contract implemented by MUSA operation adapters."""

    runner_id: str
    implementation_fingerprint: str

    def supports(self, case: BenchmarkCase) -> bool: ...

    def build_callable(self) -> Callable[..., object]: ...


@dataclass(slots=True)
class AutoTuneStats:
    operation: str
    case_count: int = 0
    candidate_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    invalidated: int = 0
    measured: int = 0
    failed: int = 0
    budget_skipped: int = 0

    def to_document(self) -> dict[str, int | str]:
        return {
            "operation": self.operation,
            "case_count": self.case_count,
            "candidate_count": self.candidate_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "invalidated": self.invalidated,
            "measured": self.measured,
            "failed": self.failed,
            "budget_skipped": self.budget_skipped,
        }


MeasureMissing = Callable[
    [TuningContext, tuple[TacticDefinition, ...]],
    Mapping[str, CandidateMeasurement],
]


def validate_resume_target(
    target: PlanTarget,
    existing_cache: TimingCache | None,
) -> None:
    """Reject a resume cache from a different runtime applicability envelope."""

    if existing_cache is None:
        return
    differences = diff_runtime_targets(
        target,
        existing_cache.target,
        final=True,
    )
    if differences:
        raise PlanningArtifactError(
            "AutoTuner resume target does not match the current target; "
            "refusing to reuse or measure candidates: " + "; ".join(differences)
        )


def candidate_set_fingerprint(
    definitions: Sequence[TacticDefinition],
) -> str:
    if not definitions:
        raise PlanningArtifactError("AutoTuner candidate set must not be empty")
    operations = {definition.operation for definition in definitions}
    if len(operations) != 1:
        raise PlanningArtifactError(
            "AutoTuner candidate set must name exactly one operation"
        )
    return compute_artifact_fingerprint(
        {
            "protocol": AUTOTUNE_PROTOCOL_VERSION,
            "catalog": [
                {
                    "id": definition.tactic_id,
                    "kind": definition.kind.value,
                    "operation": definition.operation,
                    "choice": definition.choice,
                    "fallback_id": definition.fallback_id,
                    "implementation_fingerprint": (
                        definition.implementation_fingerprint
                    ),
                }
                for definition in sorted(
                    definitions, key=lambda definition: definition.tactic_id
                )
            ],
        }
    )


def toolchain_fingerprint(
    target: PlanTarget,
    *,
    software_dependencies: Sequence[str],
    source_dependencies: Sequence[str],
) -> str:
    versions = dict(target.software.versions)
    sources = dict(target.software.source_revisions)
    missing_versions = sorted(set(software_dependencies) - versions.keys())
    missing_sources = sorted(set(source_dependencies) - sources.keys())
    if missing_versions or missing_sources:
        raise PlanningArtifactError(
            "AutoTuner dependency fingerprint is incomplete; "
            f"missing_versions={missing_versions}, "
            f"missing_sources={missing_sources}"
        )
    return compute_artifact_fingerprint(
        {
            "hardware_bin": {
                "platform": target.hardware.platform,
                "device_name": target.hardware.device_name,
                "device_capability": target.hardware.device_capability,
                "multiprocessor_count": target.hardware.multiprocessor_count,
            },
            "software": {
                key: versions[key] for key in sorted(set(software_dependencies))
            },
            "sources": {key: sources[key] for key in sorted(set(source_dependencies))},
            # An immutable image digest covers build-time compiler and binary
            # inputs that may not be importable in a production replay process.
            "image_digest": target.software.image_digest,
            "execution": {
                "compile_mode": target.workload.compile_mode,
                "graph_mode": target.workload.graph_mode,
            },
        }
    )


def tuning_context(
    *,
    operation: str,
    case: BenchmarkCase,
    candidate_fingerprint: str,
    toolchain: str,
    measurement: str,
) -> TuningContext:
    key = compute_artifact_fingerprint(
        {
            "protocol": AUTOTUNE_PROTOCOL_VERSION,
            "operation": operation,
            "case": case.fingerprint,
            "candidate_set": candidate_fingerprint,
            "toolchain": toolchain,
            "measurement": measurement,
        }
    )
    return TuningContext(
        operation=operation,
        case=case,
        candidate_set_fingerprint=candidate_fingerprint,
        toolchain_fingerprint=toolchain,
        measurement_fingerprint=measurement,
        key=key,
    )


def _required_provenance(context: TuningContext) -> dict[str, str]:
    return {
        AUTOTUNE_PROTOCOL_KEY: AUTOTUNE_PROTOCOL_VERSION,
        AUTOTUNE_KEY: context.key,
        AUTOTUNE_CANDIDATE_SET_KEY: context.candidate_set_fingerprint,
        AUTOTUNE_TOOLCHAIN_KEY: context.toolchain_fingerprint,
        AUTOTUNE_MEASUREMENT_KEY: context.measurement_fingerprint,
    }


def _cached_observations(
    cache: TimingCache | None,
    *,
    context: TuningContext,
    tactic_ids: frozenset[str],
) -> tuple[dict[str, TimingObservation], int]:
    if cache is None:
        return {}, 0
    required = _required_provenance(context)
    cached: dict[str, TimingObservation] = {}
    invalidated: set[str] = set()
    for observation in cache.observations:
        if observation.tactic_id not in tactic_ids:
            continue
        if (
            observation.case is None
            or observation.case.fingerprint != context.case.fingerprint
        ):
            continue
        provenance = dict(observation.provenance)
        if any(provenance.get(key) != value for key, value in required.items()):
            invalidated.add(observation.tactic_id)
            continue
        cached[observation.tactic_id] = observation
    return cached, len(invalidated)


def collect_operation(
    *,
    target: PlanTarget,
    definitions: Sequence[TacticDefinition],
    cases: Sequence[BenchmarkCase],
    measure_missing: MeasureMissing,
    software_dependencies: Sequence[str],
    source_dependencies: Sequence[str],
    existing_cache: TimingCache | None = None,
    provenance: Mapping[str, str] | None = None,
    measurement_identity: Mapping[str, object] | None = None,
    seal: bool = True,
) -> tuple[dict[str, object], AutoTuneStats]:
    """Collect one operation, reusing only observations with an exact key."""

    validate_resume_target(target, existing_cache)
    definitions = tuple(sorted(definitions, key=lambda item: item.tactic_id))
    if not definitions:
        raise PlanningArtifactError("AutoTuner operation has no candidates")
    operation = definitions[0].operation
    if any(definition.operation != operation for definition in definitions):
        raise PlanningArtifactError(
            "AutoTuner collect_operation received candidates from multiple operations"
        )
    if any(case.operation != operation for case in cases):
        raise PlanningArtifactError(
            "AutoTuner benchmark cases must match the candidate operation"
        )
    candidate_fingerprint = candidate_set_fingerprint(definitions)
    toolchain = toolchain_fingerprint(
        target,
        software_dependencies=software_dependencies,
        source_dependencies=source_dependencies,
    )
    measurement_fingerprint = compute_artifact_fingerprint(
        {
            "protocol": AUTOTUNE_PROTOCOL_VERSION,
            "policy": dict(measurement_identity or {}),
        }
    )
    builder = TimingCacheBuilder(
        target=target,
        catalog=definitions,
        provenance={
            **dict(provenance or {}),
            AUTOTUNE_PROTOCOL_KEY: AUTOTUNE_PROTOCOL_VERSION,
            AUTOTUNE_CANDIDATE_SET_KEY: candidate_fingerprint,
            AUTOTUNE_TOOLCHAIN_KEY: toolchain,
            AUTOTUNE_MEASUREMENT_KEY: measurement_fingerprint,
        },
    )
    stats = AutoTuneStats(
        operation=operation,
        case_count=len(cases),
        candidate_count=len(definitions),
    )
    tactic_ids = frozenset(definition.tactic_id for definition in definitions)
    for case in cases:
        context = tuning_context(
            operation=operation,
            case=case,
            candidate_fingerprint=candidate_fingerprint,
            toolchain=toolchain,
            measurement=measurement_fingerprint,
        )
        cached, invalidated = _cached_observations(
            existing_cache,
            context=context,
            tactic_ids=tactic_ids,
        )
        missing = tuple(
            definition
            for definition in definitions
            if definition.tactic_id not in cached
        )
        stats.cache_hits += len(cached)
        stats.cache_misses += len(missing)
        stats.invalidated += invalidated

        for definition in definitions:
            observation = cached.get(definition.tactic_id)
            if observation is None:
                continue
            builder.add_observation(
                tactic_id=observation.tactic_id,
                values=observation.values,
                status=observation.status.value,
                correctness=observation.correctness,
                case=observation.case,
                provenance=dict(observation.provenance),
            )

        if not missing:
            continue
        measured = dict(measure_missing(context, missing))
        expected = {definition.tactic_id for definition in missing}
        if set(measured) != expected:
            raise PlanningArtifactError(
                "AutoTuner adapter returned an incomplete candidate set; "
                f"missing={sorted(expected - measured.keys())}, "
                f"unknown={sorted(measured.keys() - expected)}"
            )
        required_provenance = _required_provenance(context)
        for definition in missing:
            candidate_measurement = measured[definition.tactic_id]
            item_provenance = {
                **dict(candidate_measurement.provenance),
                **required_provenance,
                AUTOTUNE_MEASURED_TARGET_KEY: target.fingerprint,
            }
            builder.add_observation(
                tactic_id=definition.tactic_id,
                values=candidate_measurement.values_ms,
                status=candidate_measurement.status.value,
                correctness=candidate_measurement.correctness,
                case=case,
                provenance=item_provenance,
            )
            stats.measured += 1
            if (
                candidate_measurement.status is ObservationStatus.FAILED
                or candidate_measurement.correctness != "passed"
            ):
                stats.failed += 1

    return builder.build(seal=seal), stats
