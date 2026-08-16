# SPDX-License-Identifier: Apache-2.0

"""Typed, runtime-neutral artifacts for MUSA engine-plan tuning.

The classes in this module deliberately use only the Python standard library.
They can be used by an offline benchmark collector or plan builder without
importing torch, vLLM, MUSA, TensorRT, or the serving plugin.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

TIMING_CACHE_SCHEMA_VERSION = "musa.engine_timing.v2"
CONTEXTUAL_TIMING_CACHE_SCHEMA_VERSION = "musa.engine_timing.v3"
LEGACY_TIMING_CACHE_SCHEMA_VERSION = "musa.engine_timing.v1"
SUPPORTED_TIMING_CACHE_SCHEMA_VERSIONS = frozenset(
    {
        LEGACY_TIMING_CACHE_SCHEMA_VERSION,
        TIMING_CACHE_SCHEMA_VERSION,
        CONTEXTUAL_TIMING_CACHE_SCHEMA_VERSION,
    }
)
BENCHMARK_CASE_SCHEMA_VERSION = "musa.engine_case.v1"
CONTEXTUAL_BENCHMARK_CASE_SCHEMA_VERSION = "musa.engine_case.v2"
SUPPORTED_BENCHMARK_CASE_SCHEMA_VERSIONS = frozenset(
    {BENCHMARK_CASE_SCHEMA_VERSION, CONTEXTUAL_BENCHMARK_CASE_SCHEMA_VERSION}
)
FINGERPRINT_PREFIX = "sha256:"
REQUIRED_SOFTWARE_VERSION_KEYS = frozenset(
    {
        "driver",
        "flash-attn-3",
        "flash-mla",
        "mate",
        "mthreads-ml-py",
        "musa",
        "torch",
        "torch-musa",
        "torchada",
        "vllm",
        "vllm-musa",
    }
)
REQUIRED_SOURCE_REVISION_KEYS = frozenset({"vllm", "vllm-musa"})
SUPPORTED_COMPILATION_MODES = frozenset(
    {"NONE", "STOCK_TORCH_COMPILE", "DYNAMO_TRACE_ONCE", "VLLM_COMPILE"}
)
SUPPORTED_CUDAGRAPH_MODES = frozenset(
    {"NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"}
)


class PlanningArtifactError(ValueError):
    """Raised when a typed planning artifact is malformed."""


def _expect_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PlanningArtifactError(f"{field} must be a JSON object")
    return value


def _expect_exact_keys(
    value: Mapping[str, Any],
    field: str,
    expected: set[str],
) -> None:
    actual = set(value)
    if actual != expected:
        raise PlanningArtifactError(
            f"{field} keys do not match schema; "
            f"missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _expect_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanningArtifactError(f"{field} must be a non-empty string")
    return value


def _expect_string_choice(value: Any, field: str, choices: frozenset[str]) -> str:
    result = _expect_string(value, field)
    if result not in choices:
        raise PlanningArtifactError(
            f"{field} must use a canonical vLLM enum name; "
            f"received={result!r}, supported={sorted(choices)!r}"
        )
    return result


def _expect_positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PlanningArtifactError(f"{field} must be a positive integer")
    return value


def _expect_string_pairs(
    value: Any,
    field: str,
) -> tuple[tuple[str, str], ...]:
    item = _expect_object(value, field)
    normalized: list[tuple[str, str]] = []
    for key, raw in item.items():
        normalized.append(
            (
                _expect_string(key, f"{field} key"),
                _expect_string(raw, f"{field}.{key}"),
            )
        )
    return tuple(sorted(normalized))


def _pairs_document(values: tuple[tuple[str, str], ...]) -> dict[str, str]:
    return dict(values)


ContextScalar = bool | int | float | str
ContextValue = ContextScalar | tuple[ContextScalar, ...]


def _expect_context_value(value: Any, field: str) -> ContextValue:
    """Validate one deterministic JSON value used in a typed case context."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PlanningArtifactError(f"{field} must be finite")
        return value
    if isinstance(value, str):
        if not value:
            raise PlanningArtifactError(f"{field} must not be an empty string")
        return value
    if isinstance(value, list):
        return tuple(
            _expect_context_scalar(item, f"{field}[{index}]")
            for index, item in enumerate(value)
        )
    raise PlanningArtifactError(
        f"{field} must be a bool, int, finite float, string, or scalar list"
    )


def _expect_context_scalar(value: Any, field: str) -> ContextScalar:
    parsed = _expect_context_value(value, field)
    if isinstance(parsed, tuple):
        raise PlanningArtifactError(f"{field} must be a scalar")
    return parsed


def _expect_context_pairs(
    value: Any,
    field: str,
    *,
    require_nonempty: bool,
) -> tuple[tuple[str, ContextValue], ...]:
    item = _expect_object(value, field)
    if require_nonempty and not item:
        raise PlanningArtifactError(f"{field} must not be empty")
    return tuple(
        sorted(
            (
                _expect_string(key, f"{field} key"),
                _expect_context_value(raw, f"{field}.{key}"),
            )
            for key, raw in item.items()
        )
    )


def _context_pairs_document(
    values: tuple[tuple[str, ContextValue], ...],
) -> dict[str, ContextScalar | list[ContextScalar]]:
    return {
        key: list(value) if isinstance(value, tuple) else value for key, value in values
    }


def canonical_payload(document: Mapping[str, Any]) -> bytes:
    """Return the stable JSON encoding used by every artifact fingerprint."""

    payload = {key: value for key, value in document.items() if key != "fingerprint"}
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PlanningArtifactError(
            "artifact fingerprint payload must contain only finite JSON values"
        ) from exc


def compute_artifact_fingerprint(document: Mapping[str, Any]) -> str:
    return (
        f"{FINGERPRINT_PREFIX}{hashlib.sha256(canonical_payload(document)).hexdigest()}"
    )


@dataclass(frozen=True, slots=True)
class IntegerRange:
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        _expect_positive_int(self.minimum, "range.min")
        _expect_positive_int(self.maximum, "range.max")
        if self.minimum > self.maximum:
            raise PlanningArtifactError("range.min must not exceed range.max")

    @classmethod
    def from_document(cls, value: Any, field: str) -> IntegerRange:
        item = _expect_object(value, field)
        _expect_exact_keys(item, field, {"min", "max"})
        return cls(
            minimum=_expect_positive_int(item["min"], f"{field}.min"),
            maximum=_expect_positive_int(item["max"], f"{field}.max"),
        )

    def to_document(self) -> dict[str, int]:
        return {"min": self.minimum, "max": self.maximum}


@dataclass(frozen=True, slots=True)
class ModelTarget:
    profile: str
    architecture: str
    model_id: str
    hidden_size: int | None
    dtype: str
    quantization: str
    tensor_parallel_size: int
    pipeline_parallel_size: int

    @classmethod
    def from_document(
        cls,
        value: Any,
        *,
        allow_missing_hidden_size: bool = False,
    ) -> ModelTarget:
        item = _expect_object(value, "target.model")
        fields = {
            "profile",
            "architecture",
            "model_id",
            "dtype",
            "quantization",
            "tensor_parallel_size",
            "pipeline_parallel_size",
        }
        if not allow_missing_hidden_size:
            fields.add("hidden_size")
        _expect_exact_keys(item, "target.model", fields)
        return cls(
            profile=_expect_string(item["profile"], "target.model.profile"),
            architecture=_expect_string(
                item["architecture"], "target.model.architecture"
            ),
            model_id=_expect_string(item["model_id"], "target.model.model_id"),
            hidden_size=(
                None
                if allow_missing_hidden_size
                else _expect_positive_int(
                    item["hidden_size"], "target.model.hidden_size"
                )
            ),
            dtype=_expect_string(item["dtype"], "target.model.dtype"),
            quantization=_expect_string(
                item["quantization"], "target.model.quantization"
            ),
            tensor_parallel_size=_expect_positive_int(
                item["tensor_parallel_size"],
                "target.model.tensor_parallel_size",
            ),
            pipeline_parallel_size=_expect_positive_int(
                item["pipeline_parallel_size"],
                "target.model.pipeline_parallel_size",
            ),
        )

    def to_document(self) -> dict[str, Any]:
        document = {
            "profile": self.profile,
            "architecture": self.architecture,
            "model_id": self.model_id,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
        }
        if self.hidden_size is not None:
            document["hidden_size"] = self.hidden_size
        return document


@dataclass(frozen=True, slots=True)
class HardwareTarget:
    platform: str
    device_name: str
    device_uuid: str
    device_capability: str
    multiprocessor_count: int
    device_count: int

    @classmethod
    def from_document(cls, value: Any) -> HardwareTarget:
        item = _expect_object(value, "target.hardware")
        fields = {
            "platform",
            "device_name",
            "device_uuid",
            "device_capability",
            "multiprocessor_count",
            "device_count",
        }
        _expect_exact_keys(item, "target.hardware", fields)
        return cls(
            platform=_expect_string(item["platform"], "target.hardware.platform"),
            device_name=_expect_string(
                item["device_name"], "target.hardware.device_name"
            ),
            device_uuid=_expect_string(
                item["device_uuid"], "target.hardware.device_uuid"
            ),
            device_capability=_expect_string(
                item["device_capability"],
                "target.hardware.device_capability",
            ),
            multiprocessor_count=_expect_positive_int(
                item["multiprocessor_count"],
                "target.hardware.multiprocessor_count",
            ),
            device_count=_expect_positive_int(
                item["device_count"], "target.hardware.device_count"
            ),
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "device_name": self.device_name,
            "device_uuid": self.device_uuid,
            "device_capability": self.device_capability,
            "multiprocessor_count": self.multiprocessor_count,
            "device_count": self.device_count,
        }


@dataclass(frozen=True, slots=True)
class SoftwareTarget:
    versions: tuple[tuple[str, str], ...]
    source_revisions: tuple[tuple[str, str], ...]
    image_digest: str

    def __post_init__(self) -> None:
        versions = dict(self.versions)
        missing = REQUIRED_SOFTWARE_VERSION_KEYS - versions.keys()
        if missing:
            raise PlanningArtifactError(
                "target.software.versions is missing cache-invalidation keys: "
                f"{sorted(missing)}"
            )
        unresolved = sorted(
            name
            for name, value in versions.items()
            if value.strip().lower() == "unknown"
        )
        if unresolved:
            raise PlanningArtifactError(
                "target.software.versions contains unresolved "
                f"cache-invalidation keys: {unresolved}"
            )
        source_revisions = dict(self.source_revisions)
        missing_sources = REQUIRED_SOURCE_REVISION_KEYS - source_revisions.keys()
        if missing_sources:
            raise PlanningArtifactError(
                "target.software.source_revisions is missing runtime keys: "
                f"{sorted(missing_sources)}"
            )
        unresolved_sources = sorted(
            name
            for name in REQUIRED_SOURCE_REVISION_KEYS
            if source_revisions[name].strip().lower() == "unknown"
        )
        if unresolved_sources:
            raise PlanningArtifactError(
                "target.software.source_revisions contains unresolved runtime "
                f"keys: {unresolved_sources}"
            )

    @classmethod
    def from_document(cls, value: Any) -> SoftwareTarget:
        item = _expect_object(value, "target.software")
        _expect_exact_keys(
            item,
            "target.software",
            {"versions", "source_revisions", "image_digest"},
        )
        return cls(
            versions=_expect_string_pairs(item["versions"], "target.software.versions"),
            source_revisions=_expect_string_pairs(
                item["source_revisions"],
                "target.software.source_revisions",
            ),
            image_digest=_expect_string(
                item["image_digest"], "target.software.image_digest"
            ),
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "versions": _pairs_document(self.versions),
            "source_revisions": _pairs_document(self.source_revisions),
            "image_digest": self.image_digest,
        }


@dataclass(frozen=True, slots=True)
class WorkloadTarget:
    phase: str
    batch_size: IntegerRange
    tokens: IntegerRange
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    compile_mode: str
    graph_mode: str
    cudagraph_capture_sizes: tuple[int, ...] | None

    @classmethod
    def from_document(cls, value: Any) -> WorkloadTarget:
        item = _expect_object(value, "target.workload")
        fields = {
            "phase",
            "batch_size",
            "tokens",
            "max_model_len",
            "max_num_batched_tokens",
            "max_num_seqs",
            "compile_mode",
            "graph_mode",
        }
        optional_fields = {"cudagraph_capture_sizes"}
        missing = fields - item.keys()
        extra = item.keys() - fields - optional_fields
        if missing or extra:
            raise PlanningArtifactError(
                "target.workload fields do not match schema: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        phase = _expect_string(item["phase"], "target.workload.phase")
        if phase not in {"prefill", "decode", "mixed", "operator", "serving"}:
            raise PlanningArtifactError(
                "target.workload.phase must be prefill, decode, mixed, operator, "
                "or serving"
            )
        max_num_batched_tokens = _expect_positive_int(
            item["max_num_batched_tokens"],
            "target.workload.max_num_batched_tokens",
        )
        raw_capture_sizes = item.get("cudagraph_capture_sizes")
        capture_sizes: tuple[int, ...] | None = None
        if raw_capture_sizes is not None:
            if not isinstance(raw_capture_sizes, list):
                raise PlanningArtifactError(
                    "target.workload.cudagraph_capture_sizes must be a list"
                )
            capture_sizes = tuple(
                _expect_positive_int(
                    size,
                    f"target.workload.cudagraph_capture_sizes[{index}]",
                )
                for index, size in enumerate(raw_capture_sizes)
            )
            if tuple(sorted(set(capture_sizes))) != capture_sizes:
                raise PlanningArtifactError(
                    "target.workload.cudagraph_capture_sizes must be strictly "
                    "increasing and unique"
                )
            if capture_sizes and capture_sizes[-1] > max_num_batched_tokens:
                raise PlanningArtifactError(
                    "target.workload.cudagraph_capture_sizes exceeds "
                    "max_num_batched_tokens"
                )
        return cls(
            phase=phase,
            batch_size=IntegerRange.from_document(
                item["batch_size"], "target.workload.batch_size"
            ),
            tokens=IntegerRange.from_document(item["tokens"], "target.workload.tokens"),
            max_model_len=_expect_positive_int(
                item["max_model_len"], "target.workload.max_model_len"
            ),
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=_expect_positive_int(
                item["max_num_seqs"], "target.workload.max_num_seqs"
            ),
            compile_mode=_expect_string_choice(
                item["compile_mode"],
                "target.workload.compile_mode",
                SUPPORTED_COMPILATION_MODES,
            ),
            graph_mode=_expect_string_choice(
                item["graph_mode"],
                "target.workload.graph_mode",
                SUPPORTED_CUDAGRAPH_MODES,
            ),
            cudagraph_capture_sizes=capture_sizes,
        )

    def to_document(self) -> dict[str, Any]:
        document = {
            "phase": self.phase,
            "batch_size": self.batch_size.to_document(),
            "tokens": self.tokens.to_document(),
            "max_model_len": self.max_model_len,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "compile_mode": self.compile_mode,
            "graph_mode": self.graph_mode,
        }
        if self.cudagraph_capture_sizes is not None:
            document["cudagraph_capture_sizes"] = list(self.cudagraph_capture_sizes)
        return document


def static_workload_scope_differences(
    workload: WorkloadTarget,
) -> tuple[str, ...]:
    """Return canonical reasons why a config-time plan cannot cover serving.

    Schema-v4 provider priority is frozen once per vLLM configuration, so a
    static plan may activate only when its evidence target covers the complete
    configured serving envelope.  Keep this check next to the typed workload
    schema so builders, offline explain/validate, and the live runtime cannot
    drift apart.
    """

    differences: list[str] = []
    if workload.phase != "serving":
        differences.append("workload.phase:static_plan_requires_serving")
    if workload.batch_size.minimum != 1:
        differences.append("workload.batch_size.min:static_plan_requires_1")
    if workload.batch_size.maximum < workload.max_num_seqs:
        differences.append("workload.batch_size.max:does_not_cover_max_num_seqs")
    if workload.tokens.minimum != 1:
        differences.append("workload.tokens.min:static_plan_requires_1")
    if workload.tokens.maximum < workload.max_num_batched_tokens:
        differences.append("workload.tokens.max:does_not_cover_max_num_batched_tokens")
    return tuple(differences)


_CONTEXTUAL_TP_OPERATIONS = frozenset({"musa.fused_moe.dispatch_policy"})


def static_topology_differences(
    model: ModelTarget,
    *,
    operations: Sequence[str] = (),
) -> tuple[str, ...]:
    if model.tensor_parallel_size == 1 and model.pipeline_parallel_size == 1:
        return ()
    operation_set = frozenset(operations)
    if (
        model.pipeline_parallel_size == 1
        and operation_set
        and operation_set.issubset(_CONTEXTUAL_TP_OPERATIONS)
    ):
        # Contextual fused-MoE evidence is bound to the exact per-rank static
        # kernel shape and the target still fingerprints TP.  Unlike a global
        # config tactic, it is therefore safe to activate on its measured TP.
        return ()
    return (
        "runtime_topology: config-time global activation supports TP1/PP1 only; "
        "multi-TP requires every selection to have an exact per-rank context; "
        f"got TP{model.tensor_parallel_size}/PP{model.pipeline_parallel_size}",
    )


def required_power2_bound_rows(maximum: int) -> tuple[int, ...]:
    """Return the code-owned representative row profile for a serving limit."""

    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0:
        raise PlanningArtifactError("serving row maximum must be a positive integer")
    rows = {1, maximum}
    power = 1
    while power < maximum:
        rows.add(power)
        power *= 2
    return tuple(sorted(rows))


@dataclass(frozen=True, slots=True)
class PlanTarget:
    model: ModelTarget
    hardware: HardwareTarget
    software: SoftwareTarget
    workload: WorkloadTarget

    @classmethod
    def from_document(
        cls,
        value: Any,
        *,
        allow_missing_model_hidden_size: bool = False,
    ) -> PlanTarget:
        item = _expect_object(value, "target")
        _expect_exact_keys(
            item,
            "target",
            {"model", "hardware", "software", "workload"},
        )
        return cls(
            model=ModelTarget.from_document(
                item["model"],
                allow_missing_hidden_size=allow_missing_model_hidden_size,
            ),
            hardware=HardwareTarget.from_document(item["hardware"]),
            software=SoftwareTarget.from_document(item["software"]),
            workload=WorkloadTarget.from_document(item["workload"]),
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "model": self.model.to_document(),
            "hardware": self.hardware.to_document(),
            "software": self.software.to_document(),
            "workload": self.workload.to_document(),
        }

    @property
    def fingerprint(self) -> str:
        return compute_artifact_fingerprint(self.to_document())


class TacticKind(str, Enum):
    VLLM_IR_PROVIDER = "vllm_ir_provider"
    RUNTIME_DECISION = "runtime_decision"


@dataclass(frozen=True, slots=True)
class TacticDefinition:
    tactic_id: str
    kind: TacticKind
    operation: str
    choice: bool | int | float | str
    fallback_id: str
    implementation_fingerprint: str
    description: str

    @classmethod
    def from_document(cls, value: Any, index: int) -> TacticDefinition:
        field = f"catalog[{index}]"
        item = _expect_object(value, field)
        fields = {
            "id",
            "kind",
            "operation",
            "choice",
            "fallback_id",
            "implementation_fingerprint",
            "description",
        }
        _expect_exact_keys(item, field, fields)
        try:
            kind = TacticKind(item["kind"])
        except (TypeError, ValueError) as exc:
            raise PlanningArtifactError(
                f"{field}.kind must be one of {[kind.value for kind in TacticKind]}"
            ) from exc
        choice = item["choice"]
        if (
            not isinstance(choice, (bool, int, float, str))
            or (isinstance(choice, str) and not choice)
            or (isinstance(choice, float) and not math.isfinite(choice))
        ):
            raise PlanningArtifactError(
                f"{field}.choice must be a finite scalar runtime-decision value"
            )
        return cls(
            tactic_id=_expect_string(item["id"], f"{field}.id"),
            kind=kind,
            operation=_expect_string(item["operation"], f"{field}.operation"),
            choice=choice,
            fallback_id=_expect_string(item["fallback_id"], f"{field}.fallback_id"),
            implementation_fingerprint=_expect_string(
                item["implementation_fingerprint"],
                f"{field}.implementation_fingerprint",
            ),
            description=_expect_string(item["description"], f"{field}.description"),
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "id": self.tactic_id,
            "kind": self.kind.value,
            "operation": self.operation,
            "choice": self.choice,
            "fallback_id": self.fallback_id,
            "implementation_fingerprint": self.implementation_fingerprint,
            "description": self.description,
        }


class ObservationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """Typed benchmark coordinate; human-readable names stay provenance only."""

    operation: str
    phase: str
    batch_size: int
    tokens: int
    rows: int
    hidden_size: int
    dtype: str
    fingerprint: str
    schema_version: str = BENCHMARK_CASE_SCHEMA_VERSION
    operator_dimensions: tuple[tuple[str, ContextValue], ...] = ()
    token_bucket: IntegerRange | None = None
    evidence_context: tuple[tuple[str, ContextValue], ...] = ()

    @classmethod
    def from_document(cls, value: Any, field: str) -> BenchmarkCase:
        item = _expect_object(value, field)
        schema_version = item.get("schema_version")
        if schema_version not in SUPPORTED_BENCHMARK_CASE_SCHEMA_VERSIONS:
            raise PlanningArtifactError(
                f"{field}.schema_version must be one of "
                f"{sorted(SUPPORTED_BENCHMARK_CASE_SCHEMA_VERSIONS)!r}"
            )
        contextual = schema_version == CONTEXTUAL_BENCHMARK_CASE_SCHEMA_VERSION
        fields = {
            "schema_version",
            "operation",
            "workload_point",
            "operator_shape",
            "dtype",
            "fingerprint",
        }
        if contextual:
            fields.update({"token_bucket", "evidence_context"})
        _expect_exact_keys(
            item,
            field,
            fields,
        )
        workload = _expect_object(item["workload_point"], f"{field}.workload_point")
        _expect_exact_keys(
            workload,
            f"{field}.workload_point",
            {"phase", "batch_size", "tokens"},
        )
        phase = _expect_string(workload["phase"], f"{field}.workload_point.phase")
        if phase not in {"prefill", "decode", "mixed", "operator"}:
            raise PlanningArtifactError(
                f"{field}.workload_point.phase must be prefill, decode, mixed, or operator"
            )
        shape = _expect_object(item["operator_shape"], f"{field}.operator_shape")
        if contextual:
            dimensions = _expect_context_pairs(
                shape,
                f"{field}.operator_shape",
                require_nonempty=True,
            )
            shape_values = dict(dimensions)
            hidden_size = _expect_positive_int(
                shape_values.get("hidden_size"),
                f"{field}.operator_shape.hidden_size",
            )
            token_bucket = IntegerRange.from_document(
                item["token_bucket"],
                f"{field}.token_bucket",
            )
            evidence_context = _expect_context_pairs(
                item["evidence_context"],
                f"{field}.evidence_context",
                require_nonempty=True,
            )
        else:
            _expect_exact_keys(
                shape,
                f"{field}.operator_shape",
                {"rows", "hidden_size"},
            )
            dimensions = ()
            hidden_size = _expect_positive_int(
                shape["hidden_size"], f"{field}.operator_shape.hidden_size"
            )
            token_bucket = None
            evidence_context = ()
        tokens = _expect_positive_int(
            workload["tokens"], f"{field}.workload_point.tokens"
        )
        if token_bucket is not None and not (
            token_bucket.minimum <= tokens <= token_bucket.maximum
        ):
            raise PlanningArtifactError(
                f"{field}.workload_point.tokens must fall inside token_bucket"
            )
        parsed = cls(
            operation=_expect_string(item["operation"], f"{field}.operation"),
            phase=phase,
            batch_size=_expect_positive_int(
                workload["batch_size"], f"{field}.workload_point.batch_size"
            ),
            tokens=tokens,
            rows=(
                tokens
                if contextual
                else _expect_positive_int(shape["rows"], f"{field}.operator_shape.rows")
            ),
            hidden_size=hidden_size,
            dtype=_expect_string(item["dtype"], f"{field}.dtype"),
            fingerprint=_expect_string(item["fingerprint"], f"{field}.fingerprint"),
            schema_version=schema_version,
            operator_dimensions=dimensions,
            token_bucket=token_bucket,
            evidence_context=evidence_context,
        )
        expected = compute_artifact_fingerprint(
            parsed.to_document(include_fingerprint=False)
        )
        if parsed.fingerprint != expected:
            raise PlanningArtifactError(
                f"{field}.fingerprint mismatch: expected {expected}, "
                f"got {parsed.fingerprint}"
            )
        return parsed

    @classmethod
    def create(
        cls,
        *,
        operation: str,
        phase: str,
        batch_size: int,
        tokens: int,
        rows: int,
        hidden_size: int,
        dtype: str,
    ) -> BenchmarkCase:
        document = {
            "schema_version": BENCHMARK_CASE_SCHEMA_VERSION,
            "operation": operation,
            "workload_point": {
                "phase": phase,
                "batch_size": batch_size,
                "tokens": tokens,
            },
            "operator_shape": {
                "rows": rows,
                "hidden_size": hidden_size,
            },
            "dtype": dtype,
        }
        document["fingerprint"] = compute_artifact_fingerprint(document)
        return cls.from_document(document, "benchmark_case")

    @classmethod
    def create_contextual(
        cls,
        *,
        operation: str,
        phase: str,
        batch_size: int,
        tokens: int,
        operator_shape: Mapping[str, Any],
        token_bucket_min: int,
        token_bucket_max: int,
        evidence_context: Mapping[str, Any],
        dtype: str,
    ) -> BenchmarkCase:
        """Create a case whose runtime bucket and robustness axes are explicit."""

        document = {
            "schema_version": CONTEXTUAL_BENCHMARK_CASE_SCHEMA_VERSION,
            "operation": operation,
            "workload_point": {
                "phase": phase,
                "batch_size": batch_size,
                "tokens": tokens,
            },
            "operator_shape": dict(operator_shape),
            "token_bucket": {
                "min": token_bucket_min,
                "max": token_bucket_max,
            },
            "evidence_context": dict(evidence_context),
            "dtype": dtype,
        }
        document["fingerprint"] = compute_artifact_fingerprint(document)
        return cls.from_document(document, "benchmark_case")

    @classmethod
    def from_2d_tensor(cls, *, operation: str, tensor: Any) -> BenchmarkCase:
        """Bind a case to the actual 2-D tensor passed to a benchmarked op."""

        shape = getattr(tensor, "shape", None)
        if shape is None or len(shape) != 2:
            raise PlanningArtifactError(
                "benchmark tensor must have exactly two dimensions"
            )
        rows = _expect_positive_int(int(shape[0]), "benchmark tensor rows")
        hidden_size = _expect_positive_int(
            int(shape[1]), "benchmark tensor hidden_size"
        )
        device = str(getattr(tensor, "device", ""))
        if not (device == "musa" or device.startswith("musa:")):
            raise PlanningArtifactError("benchmark tensor must reside on a MUSA device")
        dtype = str(getattr(tensor, "dtype", "")).removeprefix("torch.")
        if not dtype:
            raise PlanningArtifactError("benchmark tensor dtype is unavailable")
        return cls.create(
            operation=operation,
            phase="operator",
            batch_size=1,
            tokens=rows,
            rows=rows,
            hidden_size=hidden_size,
            dtype=dtype,
        )

    def to_document(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        if self.schema_version == CONTEXTUAL_BENCHMARK_CASE_SCHEMA_VERSION:
            if self.token_bucket is None:  # pragma: no cover - constructor invariant
                raise PlanningArtifactError(
                    "contextual benchmark case is missing token_bucket"
                )
            document: dict[str, Any] = {
                "schema_version": self.schema_version,
                "operation": self.operation,
                "workload_point": {
                    "phase": self.phase,
                    "batch_size": self.batch_size,
                    "tokens": self.tokens,
                },
                "operator_shape": self.operator_shape_document(),
                "token_bucket": self.token_bucket.to_document(),
                "evidence_context": self.evidence_context_document(),
                "dtype": self.dtype,
            }
            if include_fingerprint:
                document["fingerprint"] = self.fingerprint
            return document
        document: dict[str, Any] = {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "workload_point": {
                "phase": self.phase,
                "batch_size": self.batch_size,
                "tokens": self.tokens,
            },
            "operator_shape": {
                "rows": self.rows,
                "hidden_size": self.hidden_size,
            },
            "dtype": self.dtype,
        }
        if include_fingerprint:
            document["fingerprint"] = self.fingerprint
        return document

    def operator_shape_document(
        self,
    ) -> dict[str, ContextScalar | list[ContextScalar]]:
        if self.schema_version == CONTEXTUAL_BENCHMARK_CASE_SCHEMA_VERSION:
            return _context_pairs_document(self.operator_dimensions)
        return {"rows": self.rows, "hidden_size": self.hidden_size}

    def evidence_context_document(
        self,
    ) -> dict[str, ContextScalar | list[ContextScalar]]:
        return _context_pairs_document(self.evidence_context)


@dataclass(frozen=True, slots=True)
class TimingObservation:
    tactic_id: str
    status: ObservationStatus
    metric: str
    values: tuple[float, ...]
    correctness: str
    case: BenchmarkCase | None
    provenance: tuple[tuple[str, str], ...]

    @classmethod
    def from_document(
        cls,
        value: Any,
        index: int,
        *,
        require_case: bool,
    ) -> TimingObservation:
        field = f"observations[{index}]"
        item = _expect_object(value, field)
        fields = {
            "tactic_id",
            "status",
            "metric",
            "values",
            "correctness",
            "provenance",
        }
        if require_case:
            fields.add("case")
        _expect_exact_keys(item, field, fields)
        try:
            status = ObservationStatus(item["status"])
        except (TypeError, ValueError) as exc:
            raise PlanningArtifactError(
                f"{field}.status must be passed or failed"
            ) from exc
        values = _timing_values(item["values"], f"{field}.values")
        correctness = _expect_string(item["correctness"], f"{field}.correctness")
        if status is ObservationStatus.PASSED:
            if not values:
                raise PlanningArtifactError(
                    f"{field} passed observations require timing values"
                )
        return cls(
            tactic_id=_expect_string(item["tactic_id"], f"{field}.tactic_id"),
            status=status,
            metric=_expect_string(item["metric"], f"{field}.metric"),
            values=values,
            correctness=correctness,
            case=(
                BenchmarkCase.from_document(item["case"], f"{field}.case")
                if require_case
                else None
            ),
            provenance=_expect_string_pairs(item["provenance"], f"{field}.provenance"),
        )

    def to_document(self) -> dict[str, Any]:
        document = {
            "tactic_id": self.tactic_id,
            "status": self.status.value,
            "metric": self.metric,
            "values": list(self.values),
            "correctness": self.correctness,
            "provenance": _pairs_document(self.provenance),
        }
        if self.case is not None:
            document["case"] = self.case.to_document()
        return document

    @property
    def samples(self) -> int:
        return len(self.values)

    @property
    def median(self) -> float | None:
        return float(statistics.median(self.values)) if self.values else None

    @property
    def p95(self) -> float | None:
        if not self.values:
            return None
        ordered = sorted(self.values)
        return ordered[int(0.95 * (len(ordered) - 1))]


def _timing_values(value: Any, field: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise PlanningArtifactError(f"{field} must be a list")
    normalized: list[float] = []
    for index, item in enumerate(value):
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise PlanningArtifactError(
                f"{field}[{index}] must be a finite positive number"
            )
        item = float(item)
        if not math.isfinite(item) or item <= 0:
            raise PlanningArtifactError(
                f"{field}[{index}] must be a finite positive number"
            )
        normalized.append(item)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class TimingCache:
    schema_version: str
    target: PlanTarget
    catalog: tuple[TacticDefinition, ...]
    observations: tuple[TimingObservation, ...]
    provenance: tuple[tuple[str, str], ...]
    fingerprint: str

    @classmethod
    def from_document(
        cls,
        value: Any,
        *,
        require_fingerprint: bool,
    ) -> TimingCache:
        item = _expect_object(value, "timing_cache")
        fields = {
            "schema_version",
            "target",
            "catalog",
            "observations",
            "provenance",
        }
        if require_fingerprint:
            fields.add("fingerprint")
        _expect_exact_keys(item, "timing_cache", fields)
        schema_version = item["schema_version"]
        if schema_version not in SUPPORTED_TIMING_CACHE_SCHEMA_VERSIONS:
            raise PlanningArtifactError(
                "timing_cache.schema_version must be one of "
                f"{sorted(SUPPORTED_TIMING_CACHE_SCHEMA_VERSIONS)!r}"
            )
        raw_catalog = item["catalog"]
        if not isinstance(raw_catalog, list) or not raw_catalog:
            raise PlanningArtifactError("timing_cache.catalog must be a non-empty list")
        catalog = tuple(
            TacticDefinition.from_document(entry, index)
            for index, entry in enumerate(raw_catalog)
        )
        raw_observations = item["observations"]
        if not isinstance(raw_observations, list) or not raw_observations:
            raise PlanningArtifactError(
                "timing_cache.observations must be a non-empty list"
            )
        observations = tuple(
            TimingObservation.from_document(
                entry,
                index,
                require_case=schema_version != LEGACY_TIMING_CACHE_SCHEMA_VERSION,
            )
            for index, entry in enumerate(raw_observations)
        )
        expected_case_schema = {
            TIMING_CACHE_SCHEMA_VERSION: BENCHMARK_CASE_SCHEMA_VERSION,
            CONTEXTUAL_TIMING_CACHE_SCHEMA_VERSION: (
                CONTEXTUAL_BENCHMARK_CASE_SCHEMA_VERSION
            ),
        }.get(schema_version)
        if expected_case_schema is not None:
            mismatched_cases = [
                index
                for index, observation in enumerate(observations)
                if observation.case is None
                or observation.case.schema_version != expected_case_schema
            ]
            if mismatched_cases:
                raise PlanningArtifactError(
                    f"timing_cache schema {schema_version!r} requires case schema "
                    f"{expected_case_schema!r}; mismatched observations="
                    f"{mismatched_cases}"
                )
        _validate_catalog_and_observations(catalog, observations)
        fingerprint = item.get("fingerprint", "")
        if require_fingerprint:
            if not isinstance(fingerprint, str) or not fingerprint.startswith(
                FINGERPRINT_PREFIX
            ):
                raise PlanningArtifactError(
                    "timing_cache.fingerprint must use the sha256: prefix"
                )
            try:
                expected = compute_artifact_fingerprint(item)
            except PlanningArtifactError as exc:
                raise PlanningArtifactError(
                    "timing_cache fingerprint payload must contain only finite JSON values"
                ) from exc
            if fingerprint != expected:
                raise PlanningArtifactError(
                    "Timing-cache fingerprint mismatch: "
                    f"expected {expected}, got {fingerprint}"
                )
        return cls(
            schema_version=schema_version,
            target=PlanTarget.from_document(
                item["target"],
                allow_missing_model_hidden_size=(
                    schema_version == LEGACY_TIMING_CACHE_SCHEMA_VERSION
                ),
            ),
            catalog=catalog,
            observations=observations,
            provenance=_expect_string_pairs(
                item["provenance"], "timing_cache.provenance"
            ),
            fingerprint=fingerprint,
        )

    def to_document(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        sorted_catalog = sorted(self.catalog, key=lambda item: item.tactic_id)
        sorted_observations = sorted(
            self.observations,
            key=lambda item: (
                item.tactic_id,
                item.case.fingerprint if item.case is not None else "",
                item.provenance,
                item.values,
            ),
        )
        document: dict[str, Any] = {
            "schema_version": self.schema_version,
            "target": self.target.to_document(),
            "catalog": [item.to_document() for item in sorted_catalog],
            "observations": [item.to_document() for item in sorted_observations],
            "provenance": _pairs_document(self.provenance),
        }
        if include_fingerprint:
            document["fingerprint"] = self.fingerprint
        return document


@dataclass(slots=True)
class TimingCacheBuilder:
    """Mutable benchmark-side collector that emits a typed timing artifact."""

    target: PlanTarget
    catalog: tuple[TacticDefinition, ...]
    provenance: dict[str, str]
    observations: list[TimingObservation] = field(default_factory=list)

    @classmethod
    def from_documents(
        cls,
        *,
        target: Mapping[str, Any],
        catalog: Sequence[Mapping[str, Any]],
        provenance: Mapping[str, str],
    ) -> TimingCacheBuilder:
        return cls(
            target=PlanTarget.from_document(dict(target)),
            catalog=tuple(
                TacticDefinition.from_document(dict(item), index)
                for index, item in enumerate(catalog)
            ),
            provenance={
                _expect_string(key, "provenance key"): _expect_string(
                    value, f"provenance.{key}"
                )
                for key, value in provenance.items()
            },
        )

    def add_observation(
        self,
        tactic_id: str,
        values: Sequence[float],
        *,
        case: BenchmarkCase | Mapping[str, Any],
        status: str = "passed",
        metric: str = "median_ms",
        correctness: str = "passed",
        provenance: Mapping[str, str] | None = None,
    ) -> None:
        if tactic_id not in {item.tactic_id for item in self.catalog}:
            raise PlanningArtifactError(f"Unknown tactic ID {tactic_id!r}")
        self.observations.append(
            TimingObservation.from_document(
                {
                    "tactic_id": tactic_id,
                    "status": status,
                    "metric": metric,
                    "values": list(values),
                    "correctness": correctness,
                    "case": (
                        case.to_document()
                        if isinstance(case, BenchmarkCase)
                        else dict(case)
                    ),
                    "provenance": dict(provenance or {}),
                },
                len(self.observations),
                require_case=True,
            )
        )

    def build(self, *, seal: bool = True) -> dict[str, Any]:
        case_schemas = {
            observation.case.schema_version
            for observation in self.observations
            if observation.case is not None
        }
        if len(case_schemas) > 1:
            raise PlanningArtifactError(
                "timing cache cannot mix benchmark case schema versions"
            )
        schema_version = (
            CONTEXTUAL_TIMING_CACHE_SCHEMA_VERSION
            if case_schemas == {CONTEXTUAL_BENCHMARK_CASE_SCHEMA_VERSION}
            else TIMING_CACHE_SCHEMA_VERSION
        )
        timing_cache = TimingCache(
            schema_version=schema_version,
            target=self.target,
            catalog=self.catalog,
            observations=tuple(self.observations),
            provenance=tuple(sorted(self.provenance.items())),
            fingerprint="",
        )
        document = timing_cache.to_document(include_fingerprint=False)
        _validate_catalog_and_observations(
            timing_cache.catalog,
            timing_cache.observations,
        )
        return seal_timing_cache_document(document) if seal else document


def _validate_catalog_and_observations(
    catalog: Sequence[TacticDefinition],
    observations: Sequence[TimingObservation],
) -> None:
    by_id = {tactic.tactic_id: tactic for tactic in catalog}
    if len(by_id) != len(catalog):
        raise PlanningArtifactError(
            "timing_cache.catalog contains duplicate tactic IDs"
        )
    for tactic in catalog:
        fallback = by_id.get(tactic.fallback_id)
        if fallback is None:
            raise PlanningArtifactError(
                f"Tactic {tactic.tactic_id!r} references missing fallback "
                f"{tactic.fallback_id!r}"
            )
        if fallback.kind is not tactic.kind or fallback.operation != tactic.operation:
            raise PlanningArtifactError(
                f"Tactic {tactic.tactic_id!r} fallback must target the same kind "
                "and operation"
            )
        if fallback.fallback_id != fallback.tactic_id:
            raise PlanningArtifactError(
                f"Fallback tactic {fallback.tactic_id!r} must be self-fallback"
            )
        if tactic.kind is TacticKind.VLLM_IR_PROVIDER and fallback.choice != "native":
            raise PlanningArtifactError(
                f"IR operation {tactic.operation!r} must retain native fallback"
            )
        if tactic.kind is TacticKind.VLLM_IR_PROVIDER and not isinstance(
            tactic.choice, str
        ):
            raise PlanningArtifactError(
                f"IR-provider tactic {tactic.tactic_id!r} must use a string choice"
            )
    observation_ids = [item.tactic_id for item in observations]
    unknown = sorted(set(observation_ids) - set(by_id))
    if unknown:
        raise PlanningArtifactError(
            f"timing_cache.observations references unknown tactics: {unknown}"
        )
    operations = {by_id[item].operation for item in observation_ids}
    missing_operations = sorted({item.operation for item in catalog} - operations)
    if missing_operations:
        raise PlanningArtifactError(
            "timing_cache has no observations for catalog operations: "
            f"{missing_operations}"
        )


def seal_timing_cache_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate or seal one timing cache, failing closed on stale fingerprints."""

    unsealed = dict(document)
    if "fingerprint" in unsealed:
        TimingCache.from_document(unsealed, require_fingerprint=True)
        # Preserve the caller's canonical bytes and fingerprint. Reordering a
        # previously sealed document must not silently change what was signed.
        return dict(unsealed)
    unsealed.pop("fingerprint", None)
    parsed = TimingCache.from_document(unsealed, require_fingerprint=False)
    sealed = parsed.to_document(include_fingerprint=False)
    sealed["fingerprint"] = compute_artifact_fingerprint(sealed)
    TimingCache.from_document(sealed, require_fingerprint=True)
    return sealed


def merge_timing_cache_documents(
    documents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge repeated evidence for one exact target and tactic catalog."""

    if not documents:
        raise PlanningArtifactError("at least one timing cache is required to merge")
    parsed: list[TimingCache] = []
    sealed_inputs: list[dict[str, Any]] = []
    for document in documents:
        sealed = seal_timing_cache_document(document)
        sealed_inputs.append(sealed)
        parsed.append(TimingCache.from_document(sealed, require_fingerprint=True))
    first = parsed[0]
    # Import lazily to keep the typed artifact layer free of registry import
    # cycles while still making declared/legacy merges order-independent.
    from .tuning_domains import (
        TUNING_DOMAIN_PROVENANCE_KEY,
        resolve_timing_cache_domain,
    )

    resolved_domains = [resolve_timing_cache_domain(item)[0] for item in parsed]
    domain_ids = {domain.domain_id for domain in resolved_domains if domain is not None}
    if len(domain_ids) > 1 or (
        domain_ids and any(domain is None for domain in resolved_domains)
    ):
        raise PlanningArtifactError("timing caches resolve to different tuning domains")
    first_catalog = {tactic.tactic_id: tactic.to_document() for tactic in first.catalog}
    for index, timing_cache in enumerate(parsed[1:], start=1):
        if timing_cache.schema_version != first.schema_version:
            raise PlanningArtifactError(
                f"timing cache {index} uses a different schema version"
            )
        if timing_cache.target.fingerprint != first.target.fingerprint:
            raise PlanningArtifactError(
                f"timing cache {index} targets a different model/hardware/workload"
            )
        catalog = {
            tactic.tactic_id: tactic.to_document() for tactic in timing_cache.catalog
        }
        if catalog != first_catalog:
            raise PlanningArtifactError(
                f"timing cache {index} has a different tactic catalog"
            )
    provenance = dict(first.provenance)
    if domain_ids:
        provenance[TUNING_DOMAIN_PROVENANCE_KEY] = next(iter(domain_ids))
    provenance["merged_input_count"] = str(len(parsed))
    provenance["merged_input_fingerprints"] = ",".join(
        item["fingerprint"] for item in sealed_inputs
    )
    merged = TimingCache(
        schema_version=first.schema_version,
        target=first.target,
        catalog=first.catalog,
        observations=tuple(
            observation
            for timing_cache in parsed
            for observation in timing_cache.observations
        ),
        provenance=tuple(sorted(provenance.items())),
        fingerprint="",
    )
    return seal_timing_cache_document(merged.to_document(include_fingerprint=False))


# Fields known during the early vLLM platform-default phase. These are
# applicability keys, not just measurement provenance. In particular, compiled
# tactic timings are not portable across an unvalidated compiler/runtime or
# source change even when the GPU shape is identical. Image digests and device
# UUIDs remain evidence-only because the exact runtime components and source
# revisions below are the compatibility boundary.
EARLY_RUNTIME_MATCH_PATHS: tuple[tuple[str, ...], ...] = (
    ("model", "architecture"),
    ("model", "model_id"),
    ("model", "hidden_size"),
    ("model", "dtype"),
    ("model", "quantization"),
    ("model", "tensor_parallel_size"),
    ("model", "pipeline_parallel_size"),
    ("hardware", "platform"),
    ("hardware", "device_name"),
    ("hardware", "device_capability"),
    ("hardware", "multiprocessor_count"),
    ("software", "versions"),
    ("software", "source_revisions"),
    ("workload", "max_model_len"),
    ("workload", "max_num_batched_tokens"),
    ("workload", "max_num_seqs"),
)

# Compile and graph defaults may not be frozen when the plugin applies an IR
# priority, so they are rechecked during final platform validation.
FINAL_RUNTIME_MATCH_PATHS: tuple[tuple[str, ...], ...] = (
    *EARLY_RUNTIME_MATCH_PATHS,
    ("workload", "compile_mode"),
    ("workload", "graph_mode"),
)
_CUDAGRAPH_CAPTURE_SIZES_PATH = ("workload", "cudagraph_capture_sizes")


def _runtime_match_paths(
    document: Mapping[str, Any],
    *,
    final: bool,
) -> tuple[tuple[str, ...], ...]:
    paths = FINAL_RUNTIME_MATCH_PATHS if final else EARLY_RUNTIME_MATCH_PATHS
    workload = document.get("workload")
    if (
        final
        and isinstance(workload, Mapping)
        and "cudagraph_capture_sizes" in workload
    ):
        return (*paths, _CUDAGRAPH_CAPTURE_SIZES_PATH)
    return paths


EVIDENCE_ONLY_TARGET_PATHS: tuple[tuple[str, ...], ...] = (
    ("model", "profile"),
    ("hardware", "device_uuid"),
    ("hardware", "device_count"),
    ("software", "image_digest"),
    ("workload", "phase"),
    ("workload", "batch_size"),
    ("workload", "tokens"),
)


def _path_value(document: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = document
    for component in path:
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    return value


def runtime_key_document(
    target: PlanTarget | Mapping[str, Any],
    *,
    final: bool,
) -> dict[str, Any]:
    document = target.to_document() if isinstance(target, PlanTarget) else dict(target)
    paths = _runtime_match_paths(document, final=final)
    return {".".join(path): _path_value(document, path) for path in paths}


def runtime_key_fingerprint(
    target: PlanTarget | Mapping[str, Any],
    *,
    final: bool,
) -> str:
    return compute_artifact_fingerprint(runtime_key_document(target, final=final))


def diff_runtime_targets(
    expected: PlanTarget | Mapping[str, Any],
    actual: PlanTarget | Mapping[str, Any],
    *,
    final: bool,
) -> tuple[str, ...]:
    expected_document = (
        expected.to_document() if isinstance(expected, PlanTarget) else dict(expected)
    )
    actual_document = (
        actual.to_document() if isinstance(actual, PlanTarget) else dict(actual)
    )
    paths = _runtime_match_paths(expected_document, final=final)
    differences: list[str] = []
    for path in paths:
        expected_value = _path_value(expected_document, path)
        actual_value = _path_value(actual_document, path)
        if expected_value != actual_value:
            differences.append(
                f"{'.'.join(path)}: expected={expected_value!r}, "
                f"actual={actual_value!r}"
            )
    return tuple(differences)
