# SPDX-License-Identifier: Apache-2.0

"""Model-independent capability domains for engine-plan AutoTuning.

Runtime-plan profiles identify the model provider that consumes a sealed plan.
Tuning domains instead identify one operator contract: its context key,
candidate family, correctness oracle, and RuntimePlan lowering.  Keeping those
identities separate lets a new model reuse an existing tuner without teaching
the tuner the model's architecture name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Callable, Literal

from vllm_musa.tuning import FUSED_ADD_RMSNORM_TUNED_HIDDEN_SIZES

from .artifacts import (
    CONTEXTUAL_BENCHMARK_CASE_SCHEMA_VERSION,
    BenchmarkCase,
    PlanningArtifactError,
    PlanTarget,
    TacticDefinition,
    TacticKind,
    TimingCache,
)
from .tactic_fingerprints import runtime_decision_implementation

TUNING_DOMAIN_SCHEMA_VERSION = "musa.engine_tuning_domains.v1"
TUNING_DOMAIN_PROVENANCE_KEY = "autotune.domain"

FUSED_ADD_RMS_NORM_DOMAIN_ID = "fused_add_rms_norm.bf16_h5120"
FUSED_ADD_RMS_NORM_H4096_DOMAIN_ID = "fused_add_rms_norm.bf16_h4096"
FUSED_ADD_RMS_NORM_DOMAIN_HIDDEN_SIZES = MappingProxyType(
    {
        f"fused_add_rms_norm.bf16_h{hidden_size}": hidden_size
        for hidden_size in sorted(FUSED_ADD_RMSNORM_TUNED_HIDDEN_SIZES)
    }
)
FUSED_MOE_FP8_BLOCK_DOMAIN_ID = "fused_moe.fp8_block"
FUSED_MOE_FP8_QUANTIZATION_NAMES = frozenset({"fp8", "deepseek_v4_fp8"})

FUSED_ADD_RMS_NORM_OPERATION = "musa.fused_add_rms_norm.min_rows"
FUSED_MOE_DISPATCH_OPERATION = "musa.fused_moe.dispatch_policy"
FUSED_MOE_BACKENDS = ("upstream", "gemv", "grouped_gemm")
FUSED_MOE_ROUTES = frozenset({"balanced", "unique_random", "hot"})
FUSED_MOE_SHAPE_FIELDS = frozenset(
    {
        "device_capability",
        "multiprocessor_count",
        "local_experts",
        "w1_output_size",
        "w2_input_size",
        "hidden_size",
        "top_k",
        "block_n",
        "block_k",
        "activation",
        "expert_parallel",
        "hidden_dtype",
        "weight_dtype",
        "scale_dtype",
        "w1_scale_shape",
        "w2_scale_shape",
        "gemv_block",
        "graph_mode",
    }
)


@dataclass(frozen=True, slots=True)
class TuningDomain:
    """One reusable operator-level tuning contract."""

    domain_id: str
    operations: tuple[str, ...]
    evidence_mode: Literal["collect", "import"]
    context_key: str
    candidate_contract: str
    correctness_oracle: str
    runtime_lowering: str
    target_requirements: tuple[str, ...]

    def to_document(self) -> dict[str, object]:
        return {
            "id": self.domain_id,
            "operations": list(self.operations),
            "evidence_mode": self.evidence_mode,
            "context_key": self.context_key,
            "candidate_contract": self.candidate_contract,
            "correctness_oracle": self.correctness_oracle,
            "runtime_lowering": self.runtime_lowering,
            "target_requirements": list(self.target_requirements),
            "model_identity_policy": (
                "runtime profile, architecture, and model ID remain artifact "
                "applicability/provenance; none dispatches tuner code"
            ),
        }


_DOMAINS = (
    *(
        TuningDomain(
            domain_id=domain_id,
            operations=(FUSED_ADD_RMS_NORM_OPERATION,),
            evidence_mode="collect",
            context_key="rows, hidden_size, dtype, compile/graph context",
            candidate_contract="registered C-extension/JIT crossover thresholds",
            correctness_oracle="native fused-add RMSNorm numeric comparison",
            runtime_lowering=FUSED_ADD_RMS_NORM_OPERATION,
            target_requirements=(
                "platform=musa",
                "dtype=bf16",
                f"hidden_size={hidden_size}",
            ),
        )
        for domain_id, hidden_size in FUSED_ADD_RMS_NORM_DOMAIN_HIDDEN_SIZES.items()
    ),
    TuningDomain(
        domain_id=FUSED_MOE_FP8_BLOCK_DOMAIN_ID,
        operations=(FUSED_MOE_DISPATCH_OPERATION,),
        evidence_mode="import",
        context_key=(
            "MusaFusedMoeShape, token bucket, eager/capture context; route and "
            "seed are evidence axes"
        ),
        candidate_contract="upstream fallback plus supported GEMV/grouped-GEMM",
        correctness_oracle=(
            "qualified dispatcher identity, FP8 numeric metrics, and graph replay"
        ),
        runtime_lowering="musa.fused_moe.dispatch_policy.v1",
        target_requirements=(
            "platform=musa",
            "device=MTT S5000 capability 3.1 with exact active-MP bin",
            "dtype=bf16",
            "quantization=fp8",
            "live per-shape kernel support predicate",
        ),
    ),
)
_DOMAIN_BY_ID = MappingProxyType({domain.domain_id: domain for domain in _DOMAINS})
if len(_DOMAIN_BY_ID) != len(_DOMAINS):  # pragma: no cover - registry invariant
    raise RuntimeError("tuning domain IDs must be unique")
_domains_by_operation: dict[str, list[TuningDomain]] = {}
for _domain in _DOMAINS:
    for _operation in _domain.operations:
        _domains_by_operation.setdefault(_operation, []).append(_domain)
_DOMAINS_BY_OPERATION = MappingProxyType(
    {operation: tuple(domains) for operation, domains in _domains_by_operation.items()}
)


def list_tuning_domains() -> tuple[TuningDomain, ...]:
    return _DOMAINS


def fused_moe_tactic_definitions() -> tuple[TacticDefinition, ...]:
    """Return the domain-owned atomic fused-MoE candidates and fallback."""

    implementation = runtime_decision_implementation(FUSED_MOE_DISPATCH_OPERATION)
    if implementation is None or implementation.provider_key is not None:
        raise PlanningArtifactError("fused-MoE tactic identity is not registered")
    prefix = "runtime.musa.fused_moe"
    fallback_id = f"{prefix}:upstream"
    return tuple(
        TacticDefinition(
            tactic_id=f"{prefix}:{backend}",
            kind=TacticKind.RUNTIME_DECISION,
            operation=FUSED_MOE_DISPATCH_OPERATION,
            choice=backend,
            fallback_id=fallback_id,
            implementation_fingerprint=implementation.fingerprint(choice=backend),
            description=f"MUSA fused-MoE atomic backend {backend!r}",
        )
        for backend in FUSED_MOE_BACKENDS
    )


def get_tuning_domain(domain_id: str) -> TuningDomain:
    try:
        return _DOMAIN_BY_ID[domain_id]
    except KeyError as exc:
        raise PlanningArtifactError(
            f"unknown tuning domain {domain_id!r}; "
            f"registered={sorted(_DOMAIN_BY_ID)!r}"
        ) from exc


TargetValidator = Callable[[PlanTarget], tuple[str, ...]]
CaseValidator = Callable[[BenchmarkCase, PlanTarget], tuple[str, ...]]


def _platform_differences(target: PlanTarget) -> list[str]:
    differences: list[str] = []
    if target.hardware.platform != "musa":
        differences.append(
            f"hardware.platform: expected='musa', actual={target.hardware.platform!r}"
        )
    return differences


def _fused_add_target_differences(
    target: PlanTarget,
    *,
    hidden_size: int,
) -> tuple[str, ...]:
    differences = _platform_differences(target)
    if target.model.dtype not in {"bf16", "bfloat16"}:
        differences.append("model.dtype: fused-add RMSNorm domain requires bf16")
    if target.model.hidden_size != hidden_size:
        differences.append(
            "model.hidden_size: fused-add RMSNorm domain requires " f"{hidden_size}"
        )
    return tuple(differences)


def _fused_moe_target_differences(target: PlanTarget) -> tuple[str, ...]:
    differences = _platform_differences(target)
    if "S5000" not in target.hardware.device_name:
        differences.append("hardware.device_name: fused-MoE domain requires S5000")
    if target.hardware.device_capability != "3.1":
        differences.append("hardware.device_capability: fused-MoE domain requires 3.1")
    if target.model.dtype not in {"bf16", "bfloat16"}:
        differences.append("model.dtype: fused-MoE domain requires bf16")
    if target.model.quantization.lower() not in FUSED_MOE_FP8_QUANTIZATION_NAMES:
        differences.append(
            "model.quantization: fused-MoE domain requires fp8 "
            "(fp8 or deepseek_v4_fp8)"
        )
    return tuple(differences)


_TARGET_VALIDATORS: MappingProxyType[str, TargetValidator] = MappingProxyType(
    {
        **{
            domain_id: partial(
                _fused_add_target_differences,
                hidden_size=hidden_size,
            )
            for domain_id, hidden_size in FUSED_ADD_RMS_NORM_DOMAIN_HIDDEN_SIZES.items()
        },
        FUSED_MOE_FP8_BLOCK_DOMAIN_ID: _fused_moe_target_differences,
    }
)


def validate_tuning_domain_target(
    domain_id: str,
    target: PlanTarget,
) -> TuningDomain:
    """Fail closed when a target cannot use an operator tuning domain."""

    domain = get_tuning_domain(domain_id)
    try:
        validator = _TARGET_VALIDATORS[domain.domain_id]
    except KeyError as exc:
        raise PlanningArtifactError(
            f"tuning domain {domain.domain_id!r} has no target validator"
        ) from exc
    differences = validator(target)
    if differences:
        raise PlanningArtifactError(
            f"target is incompatible with tuning domain {domain.domain_id!r}: "
            + "; ".join(differences)
        )
    return domain


def validate_fused_moe_case_structure(
    case: BenchmarkCase,
    target: PlanTarget,
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate the complete model-neutral fused-MoE runtime context."""

    if case.schema_version != CONTEXTUAL_BENCHMARK_CASE_SCHEMA_VERSION:
        raise PlanningArtifactError("contextual selection requires case schema v2")
    if case.operation != FUSED_MOE_DISPATCH_OPERATION:
        raise PlanningArtifactError("contextual case operation does not match tactic")
    if case.phase != "operator" or case.batch_size != 1:
        raise PlanningArtifactError(
            "fused-MoE cases require phase=operator and batch_size=1"
        )
    if case.dtype != target.model.dtype:
        raise PlanningArtifactError("fused-MoE case dtype does not match target")
    if case.token_bucket is None:  # pragma: no cover - case parser invariant
        raise PlanningArtifactError("contextual case is missing token_bucket")
    workload = target.workload.tokens
    if (
        case.token_bucket.minimum < workload.minimum
        or case.token_bucket.maximum > workload.maximum
    ):
        raise PlanningArtifactError("fused-MoE token bucket is outside target scope")

    shape = case.operator_shape_document()
    if set(shape) != FUSED_MOE_SHAPE_FIELDS:
        raise PlanningArtifactError(
            "fused-MoE operator_shape keys do not match the runtime shape; "
            f"missing={sorted(FUSED_MOE_SHAPE_FIELDS - set(shape))}, "
            f"unknown={sorted(set(shape) - FUSED_MOE_SHAPE_FIELDS)}"
        )
    for name in (
        "multiprocessor_count",
        "local_experts",
        "w1_output_size",
        "w2_input_size",
        "hidden_size",
        "top_k",
    ):
        value = shape[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise PlanningArtifactError(
                f"fused-MoE operator_shape.{name} must be a positive integer"
            )
    if shape["top_k"] > shape["local_experts"]:
        raise PlanningArtifactError(
            "fused-MoE operator_shape.top_k must not exceed local_experts"
        )
    for name in ("block_n", "block_k"):
        value = shape[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise PlanningArtifactError(
                f"fused-MoE operator_shape.{name} must be a non-negative integer"
            )
    for name in (
        "activation",
        "hidden_dtype",
        "weight_dtype",
        "scale_dtype",
        "gemv_block",
        "graph_mode",
    ):
        if not isinstance(shape[name], str) or not shape[name]:
            raise PlanningArtifactError(
                f"fused-MoE operator_shape.{name} must be a non-empty string"
            )
    if shape["graph_mode"] not in {"eager", "capture"}:
        raise PlanningArtifactError(
            "fused-MoE operator_shape.graph_mode must be eager or capture"
        )
    if not isinstance(shape["expert_parallel"], bool):
        raise PlanningArtifactError(
            "fused-MoE operator_shape.expert_parallel must be boolean"
        )
    capability = shape["device_capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in capability
        )
    ):
        raise PlanningArtifactError(
            "fused-MoE operator_shape.device_capability must be [major, minor]"
        )
    for name in ("w1_scale_shape", "w2_scale_shape"):
        value = shape[name]
        if not isinstance(value, list) or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in value
        ):
            raise PlanningArtifactError(
                f"fused-MoE operator_shape.{name} must be a dimension list"
            )
    if shape["hidden_size"] != target.model.hidden_size:
        raise PlanningArtifactError(
            "fused-MoE operator_shape.hidden_size does not match target model"
        )
    if shape["multiprocessor_count"] != target.hardware.multiprocessor_count:
        raise PlanningArtifactError(
            "fused-MoE operator_shape.multiprocessor_count does not match target"
        )
    try:
        expected_capability = [
            int(item) for item in target.hardware.device_capability.split(".")
        ]
    except ValueError as exc:
        raise PlanningArtifactError(
            "target.hardware.device_capability must use major.minor integers"
        ) from exc
    if len(expected_capability) != 2:
        raise PlanningArtifactError(
            "target.hardware.device_capability must use major.minor integers"
        )
    if capability != expected_capability:
        raise PlanningArtifactError(
            "fused-MoE operator_shape.device_capability does not match target"
        )

    evidence = case.evidence_context_document()
    if set(evidence) != {"route_mode", "seed"}:
        raise PlanningArtifactError(
            "fused-MoE evidence_context must contain exactly route_mode and seed"
        )
    if evidence["route_mode"] not in FUSED_MOE_ROUTES:
        raise PlanningArtifactError("fused-MoE route_mode is not supported")
    seed = evidence["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise PlanningArtifactError("fused-MoE evidence seed must be non-negative")
    return shape, evidence


def _fused_moe_case_differences(
    case: BenchmarkCase,
    target: PlanTarget,
) -> tuple[str, ...]:
    shape, _ = validate_fused_moe_case_structure(case, target)
    differences: list[str] = []
    required_values = {
        "block_n": 128,
        "block_k": 128,
        "activation": "silu",
        "expert_parallel": False,
        "weight_dtype": "torch.float8_e4m3fn",
        "scale_dtype": "torch.float32",
    }
    for field, expected in required_values.items():
        if shape.get(field) != expected:
            differences.append(
                f"operator_shape.{field}: expected={expected!r}, "
                f"actual={shape.get(field)!r}"
            )
    if shape.get("hidden_dtype") not in {"torch.bfloat16", "bfloat16", "bf16"}:
        differences.append("operator_shape.hidden_dtype: expected bf16")
    if case.dtype not in {"bfloat16", "bf16"}:
        differences.append("case.dtype: expected bf16")

    local_experts = int(shape["local_experts"])
    w1_output_size = int(shape["w1_output_size"])
    w2_input_size = int(shape["w2_input_size"])
    hidden_size = int(shape["hidden_size"])
    if w1_output_size % 256:
        differences.append(
            "operator_shape.w1_output_size: expected positive multiple of 256"
        )
    if hidden_size % 128:
        differences.append(
            "operator_shape.hidden_size: expected positive multiple of 128"
        )
    if w2_input_size != w1_output_size // 2:
        differences.append("operator_shape.w2_input_size: expected w1_output_size // 2")
    expected_w1_scale = [
        local_experts,
        w1_output_size // 128,
        hidden_size // 128,
    ]
    expected_w2_scale = [
        local_experts,
        hidden_size // 128,
        w2_input_size // 128,
    ]
    if shape["w1_scale_shape"] != expected_w1_scale:
        differences.append(
            "operator_shape.w1_scale_shape: expected "
            f"{expected_w1_scale!r}, actual={shape['w1_scale_shape']!r}"
        )
    if shape["w2_scale_shape"] != expected_w2_scale:
        differences.append(
            "operator_shape.w2_scale_shape: expected "
            f"{expected_w2_scale!r}, actual={shape['w2_scale_shape']!r}"
        )
    gemv_block = str(shape["gemv_block"])
    if gemv_block != "auto":
        match = re.fullmatch(r"([1-9][0-9]*)x([1-9][0-9]*)", gemv_block)
        if match is None:
            differences.append(
                "operator_shape.gemv_block: expected 'auto' or '<block_n>x<block_k>'"
            )
        else:
            block_n, block_k = (int(item) for item in match.groups())
            vector_length = 16  # FP8 weights use 128-bit / 8-bit vector loads.
            if block_n * block_k > 512:
                differences.append(
                    "operator_shape.gemv_block: block_n * block_k must be <= 512"
                )
            if (
                (w1_output_size // 2) % block_n
                or hidden_size % (block_k * vector_length)
                or hidden_size % block_n
                or w2_input_size % (block_k * vector_length)
            ):
                differences.append(
                    "operator_shape.gemv_block: tile does not divide both routed "
                    "GEMV projections"
                )
    return tuple(differences)


def _fused_add_case_differences(
    case: BenchmarkCase,
    target: PlanTarget,
    *,
    hidden_size: int,
) -> tuple[str, ...]:
    del target
    if case.hidden_size != hidden_size or case.dtype not in {"bfloat16", "bf16"}:
        return (f"case requires BF16 H{hidden_size}",)
    return ()


_CASE_VALIDATORS: MappingProxyType[str, CaseValidator] = MappingProxyType(
    {
        **{
            domain_id: partial(
                _fused_add_case_differences,
                hidden_size=hidden_size,
            )
            for domain_id, hidden_size in FUSED_ADD_RMS_NORM_DOMAIN_HIDDEN_SIZES.items()
        },
        FUSED_MOE_FP8_BLOCK_DOMAIN_ID: _fused_moe_case_differences,
    }
)


def _validate_registry_coverage() -> None:
    domain_ids = set(_DOMAIN_BY_ID)
    missing_target = sorted(domain_ids - set(_TARGET_VALIDATORS))
    missing_case = sorted(domain_ids - set(_CASE_VALIDATORS))
    extra_target = sorted(set(_TARGET_VALIDATORS) - domain_ids)
    extra_case = sorted(set(_CASE_VALIDATORS) - domain_ids)
    if missing_target or missing_case or extra_target or extra_case:
        raise RuntimeError(
            "tuning domain validator registry mismatch: "
            f"missing_target={missing_target}, missing_case={missing_case}, "
            f"extra_target={extra_target}, extra_case={extra_case}"
        )


_validate_registry_coverage()


def validate_tuning_domain_case(
    domain_id: str,
    case: BenchmarkCase,
    target: PlanTarget,
) -> None:
    """Validate one exact operator key against its domain capability contract."""

    domain = validate_tuning_domain_target(domain_id, target)
    if case.operation not in domain.operations:
        raise PlanningArtifactError(
            f"benchmark case operation {case.operation!r} is outside tuning "
            f"domain {domain.domain_id!r}"
        )
    try:
        validator = _CASE_VALIDATORS[domain.domain_id]
    except KeyError as exc:
        raise PlanningArtifactError(
            f"tuning domain {domain.domain_id!r} has no case validator"
        ) from exc
    differences = validator(case, target)
    if differences:
        raise PlanningArtifactError(
            f"benchmark case is incompatible with tuning domain {domain.domain_id!r}: "
            + "; ".join(differences)
        )


def infer_tuning_domain(timing_cache: TimingCache) -> TuningDomain | None:
    """Infer one compatible legacy domain without consulting model identity."""

    operation_set = frozenset(tactic.operation for tactic in timing_cache.catalog)
    if not operation_set:
        return None
    candidates: set[TuningDomain] | None = None
    for operation in operation_set:
        operation_domains = set(_DOMAINS_BY_OPERATION.get(operation, ()))
        candidates = (
            operation_domains
            if candidates is None
            else candidates.intersection(operation_domains)
        )
    if not candidates:
        return None
    compatible: list[TuningDomain] = []
    for domain in sorted(candidates, key=lambda item: item.domain_id):
        try:
            validate_tuning_domain_target(domain.domain_id, timing_cache.target)
            for observation in timing_cache.observations:
                if observation.case is not None:
                    validate_tuning_domain_case(
                        domain.domain_id,
                        observation.case,
                        timing_cache.target,
                    )
        except PlanningArtifactError:
            continue
        compatible.append(domain)
    return compatible[0] if len(compatible) == 1 else None


def resolve_timing_cache_domain(
    timing_cache: TimingCache,
) -> tuple[TuningDomain | None, Literal["declared", "legacy_operation", "unknown"]]:
    """Resolve declared domains while keeping pre-domain timing caches usable."""

    operations = tuple(tactic.operation for tactic in timing_cache.catalog)
    declared = dict(timing_cache.provenance).get(TUNING_DOMAIN_PROVENANCE_KEY)
    if declared is None:
        inferred = infer_tuning_domain(timing_cache)
        if inferred is None:
            return None, "unknown"
        return inferred, "legacy_operation"

    domain = validate_tuning_domain_target(declared, timing_cache.target)
    unknown_operations = sorted(set(operations) - set(domain.operations))
    if unknown_operations:
        raise PlanningArtifactError(
            f"timing cache declares tuning domain {domain.domain_id!r} but contains "
            f"operations outside that domain: {unknown_operations}"
        )
    for observation in timing_cache.observations:
        if observation.case is not None:
            validate_tuning_domain_case(
                domain.domain_id, observation.case, timing_cache.target
            )
    return domain, "declared"


__all__ = [
    "FUSED_ADD_RMS_NORM_DOMAIN_ID",
    "FUSED_ADD_RMS_NORM_H4096_DOMAIN_ID",
    "FUSED_ADD_RMS_NORM_DOMAIN_HIDDEN_SIZES",
    "FUSED_ADD_RMS_NORM_OPERATION",
    "FUSED_MOE_DISPATCH_OPERATION",
    "FUSED_MOE_BACKENDS",
    "FUSED_MOE_FP8_BLOCK_DOMAIN_ID",
    "FUSED_MOE_ROUTES",
    "FUSED_MOE_SHAPE_FIELDS",
    "TUNING_DOMAIN_PROVENANCE_KEY",
    "TUNING_DOMAIN_SCHEMA_VERSION",
    "TuningDomain",
    "get_tuning_domain",
    "fused_moe_tactic_definitions",
    "infer_tuning_domain",
    "list_tuning_domains",
    "resolve_timing_cache_domain",
    "validate_fused_moe_case_structure",
    "validate_tuning_domain_case",
    "validate_tuning_domain_target",
]
