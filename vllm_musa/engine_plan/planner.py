# SPDX-License-Identifier: Apache-2.0

"""Offline tactic selection and sealed plan construction."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .artifacts import (
    PlanningArtifactError,
    TacticDefinition,
    TacticKind,
    TimingCache,
    TimingObservation,
    diff_runtime_targets,
    runtime_key_fingerprint,
    seal_timing_cache_document,
    static_topology_differences,
    static_workload_scope_differences,
)
from .core import PLUGIN_IDENTITY, EnginePlan, EnginePlanError, seal_plan_document
from .json_utils import loads as load_json
from .selection import BuildPolicy, build_runtime_decisions, select_operation
from .tuning_domains import resolve_timing_cache_domain


def _build_variant(
    raw_timing_cache: Mapping[str, Any],
    policy: BuildPolicy,
    runtime_decisions: Mapping[str, Any] | None,
) -> dict[str, Any]:
    sealed_timing = seal_timing_cache_document(raw_timing_cache)
    timing_cache = TimingCache.from_document(
        sealed_timing,
        require_fingerprint=True,
    )
    # New artifacts declare a model-independent tuning domain. Older artifacts
    # remain buildable by inferring that domain from the operation identity;
    # runtime profiles are deliberately never used as tuner dispatch keys.
    resolve_timing_cache_domain(timing_cache)
    workload_differences = static_workload_scope_differences(
        timing_cache.target.workload
    )
    if workload_differences:
        raise PlanningArtifactError(
            "Schema-v5 config-time plans require a complete serving workload "
            f"scope: {list(workload_differences)}"
        )
    topology_differences = static_topology_differences(
        timing_cache.target.model,
        operations=tuple(tactic.operation for tactic in timing_cache.catalog),
    )
    if topology_differences:
        raise PlanningArtifactError(
            "Schema-v5 config-time plans require a supported topology: "
            f"{list(topology_differences)}"
        )
    grouped: dict[tuple[TacticKind, str], list[TacticDefinition]] = defaultdict(list)
    for tactic in timing_cache.catalog:
        grouped[(tactic.kind, tactic.operation)].append(tactic)
    observations: dict[str, list[TimingObservation]] = defaultdict(list)
    for observation in timing_cache.observations:
        observations[observation.tactic_id].append(observation)
    selections = [
        select_operation(
            timing_cache,
            grouped[key],
            observations,
            policy,
        )
        for key in sorted(grouped, key=lambda item: (item[0].value, item[1]))
    ]
    decision_projection = build_runtime_decisions(timing_cache, selections)
    profile = timing_cache.target.model.profile
    from vllm_musa.runtime_plan.declarative import declarative_profile_identity

    profile_config = declarative_profile_identity(profile)
    if profile_config is not None:
        (
            decision_projection["profile_config_id"],
            decision_projection["profile_config_fingerprint"],
        ) = profile_config
    overrides = (
        _expect_profile_decisions(runtime_decisions, profile)
        if runtime_decisions is not None
        else {}
    )
    if overrides and (
        timing_cache.target.model.tensor_parallel_size != 1
        or timing_cache.target.model.pipeline_parallel_size != 1
    ):
        raise PlanningArtifactError(
            "Explicit global runtime decisions cannot be mixed with a multi-TP "
            "contextual-only timing variant"
        )
    values = decision_projection["values"]
    conflicts = sorted(set(values) & set(overrides))
    if conflicts:
        raise PlanningArtifactError(
            "Explicit runtime decisions must not override timing-derived "
            f"selections: {conflicts}"
        )
    values.update(copy.deepcopy(overrides))
    try:
        variant_payload = json.dumps(
            {
                "target": timing_cache.target.fingerprint,
                "timing": timing_cache.fingerprint,
                "policy": policy.to_document(),
                "runtime_decisions": decision_projection,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PlanningArtifactError(
            "runtime_decisions must contain only finite JSON values"
        ) from exc
    variant_hash = hashlib.sha256(variant_payload).hexdigest()[:16]
    return {
        "variant_id": f"variant-{variant_hash}",
        "target_fingerprint": timing_cache.target.fingerprint,
        "early_runtime_key": runtime_key_fingerprint(
            timing_cache.target,
            final=False,
        ),
        "final_runtime_key": runtime_key_fingerprint(
            timing_cache.target,
            final=True,
        ),
        "timing_cache": sealed_timing,
        "policy": policy.to_document(),
        "selections": selections,
        "runtime_decisions": decision_projection,
    }


def _expect_profile_decisions(
    runtime_decisions: Mapping[str, Any],
    profile: str,
) -> dict[str, Any]:
    if not isinstance(runtime_decisions, Mapping):
        raise PlanningArtifactError("runtime_decisions must be a profile map")
    if any(not isinstance(name, str) or not name for name in runtime_decisions):
        raise PlanningArtifactError(
            "runtime_decisions profile names must be non-empty strings"
        )
    raw = runtime_decisions.get(profile, {})
    if not isinstance(raw, Mapping):
        raise PlanningArtifactError(
            f"runtime_decisions.{profile} must be a decision-value map"
        )
    if any(not isinstance(key, str) or not key for key in raw):
        raise PlanningArtifactError(
            f"runtime_decisions.{profile} keys must be non-empty strings"
        )
    return dict(raw)


def build_plan_document(
    timing_documents: Sequence[Mapping[str, Any]],
    *,
    plan_id: str,
    policy: BuildPolicy | None = None,
    runtime_decisions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a sealed plan, preserving v5 for legacy operation-wide tactics."""

    if not isinstance(plan_id, str) or not plan_id.strip():
        raise PlanningArtifactError("plan_id must be a non-empty string")
    if not timing_documents:
        raise PlanningArtifactError("at least one timing document is required")
    policy = policy or BuildPolicy()
    variants = [
        _build_variant(document, policy, runtime_decisions)
        for document in timing_documents
    ]
    if runtime_decisions is not None:
        known_profiles = {
            TimingCache.from_document(
                variant["timing_cache"],
                require_fingerprint=True,
            ).target.model.profile
            for variant in variants
        }
        unknown_profiles = sorted(set(runtime_decisions) - known_profiles)
        if unknown_profiles:
            raise PlanningArtifactError(
                "runtime_decisions contains profiles without a timing variant: "
                f"{unknown_profiles}"
            )
    early_keys = [variant["early_runtime_key"] for variant in variants]
    if len(early_keys) != len(set(early_keys)):
        raise PlanningArtifactError(
            "Two timing inputs map to the same static runtime key. Request-shape "
            "buckets must be handled by an existing provider selector; they cannot "
            "be expressed as conflicting config-time variants."
        )

    versions: dict[str, set[str]] = {"vllm": set(), "vllm-musa": set()}
    for variant in variants:
        timing_cache = TimingCache.from_document(
            variant["timing_cache"],
            require_fingerprint=True,
        )
        target_versions = dict(timing_cache.target.software.versions)
        for distribution in versions:
            version = target_versions.get(distribution)
            if version is None:
                raise PlanningArtifactError(
                    f"target.software.versions must include {distribution!r}"
                )
            versions[distribution].add(version)

    contextual = any(
        "selection_schema" in selection
        for variant in variants
        for selection in variant["selections"]
    )
    document = {
        "schema_version": 6 if contextual else 5,
        "plan_id": plan_id,
        "plugin": {
            "name": PLUGIN_IDENTITY.name,
            "version": PLUGIN_IDENTITY.version,
            "namespace": PLUGIN_IDENTITY.namespace,
            "abi_version": PLUGIN_IDENTITY.abi_version,
        },
        "compatibility": {
            "framework": "vllm",
            "framework_version_prefixes": sorted(versions["vllm"]),
            "platform": "musa",
            "vllm_musa_version_prefixes": sorted(versions["vllm-musa"]),
        },
        "variants": sorted(variants, key=lambda item: item["variant_id"]),
    }
    return seal_plan_document(document)


def inspect_plan(plan: EnginePlan) -> dict[str, Any]:
    """Return a concise, GPU-free plan summary for CLI and tests."""

    variants: list[dict[str, Any]] = []
    for variant in plan.variants:
        timing_cache = TimingCache.from_document(
            variant["timing_cache"],
            require_fingerprint=True,
        )
        domain, domain_source = resolve_timing_cache_domain(timing_cache)
        variants.append(
            {
                "variant_id": variant["variant_id"],
                "target_fingerprint": variant["target_fingerprint"],
                "timing_fingerprint": timing_cache.fingerprint,
                "model": timing_cache.target.model.to_document(),
                "hardware": timing_cache.target.hardware.to_document(),
                "workload": timing_cache.target.workload.to_document(),
                "tuning_domain": {
                    "id": domain.domain_id if domain is not None else None,
                    "source": domain_source,
                },
                "runtime_decisions": variant["runtime_decisions"],
                "selections": [
                    _inspect_selection(selection) for selection in variant["selections"]
                ],
            }
        )
    return {
        "schema_version": plan.schema_version,
        "plan_id": plan.plan_id,
        "fingerprint": plan.fingerprint,
        "variants": variants,
    }


def _inspect_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    if "selection_schema" in selection:
        return {
            "selection_schema": selection["selection_schema"],
            "operation": selection["operation"],
            "fallback": selection["fallback"],
            "max_iqr_ratio": selection["max_iqr_ratio"],
            "contexts": copy.deepcopy(selection["contexts"]),
        }
    return {
        "operation": selection["operation"],
        "winner": selection["winner"],
        "fallback": selection["fallback"],
        "reason": selection["reason"],
        "speedup_pct": selection["speedup_pct"],
        "samples": selection["samples"],
        "winner_median_ms": selection["winner_value"],
        "winner_p95_ms": selection["winner_p95"],
        "fallback_median_ms": selection["fallback_value"],
        "fallback_p95_ms": selection["fallback_p95"],
        "coverage": selection["coverage"],
    }


def explain_plan(
    plan: EnginePlan,
    *,
    runtime_target: Mapping[str, Any] | None = None,
    final: bool = True,
) -> dict[str, Any]:
    """Explain variant matching and every recorded selection/rejection."""

    summary = inspect_plan(plan)
    decisions: list[dict[str, Any]] = []
    for variant in plan.variants:
        timing_cache = TimingCache.from_document(
            variant["timing_cache"],
            require_fingerprint=True,
        )
        domain, domain_source = resolve_timing_cache_domain(timing_cache)
        differences = (
            (
                *static_topology_differences(
                    timing_cache.target.model,
                    operations=tuple(
                        tactic.operation for tactic in timing_cache.catalog
                    ),
                ),
                *static_workload_scope_differences(timing_cache.target.workload),
                *diff_runtime_targets(
                    timing_cache.target,
                    runtime_target,
                    final=final,
                ),
            )
            if runtime_target is not None
            else (
                *static_topology_differences(
                    timing_cache.target.model,
                    operations=tuple(
                        tactic.operation for tactic in timing_cache.catalog
                    ),
                ),
                *static_workload_scope_differences(timing_cache.target.workload),
            )
        )
        decisions.append(
            {
                "variant_id": variant["variant_id"],
                "matches": not differences,
                "differences": list(differences),
                "selections": variant["selections"],
                "runtime_compatibility_scope": {
                    "architecture": timing_cache.target.model.architecture,
                    "model_id": timing_cache.target.model.model_id,
                    "hidden_size": timing_cache.target.model.hidden_size,
                    "dtype": timing_cache.target.model.dtype,
                    "quantization": timing_cache.target.model.quantization,
                    "tensor_parallel_size": (
                        timing_cache.target.model.tensor_parallel_size
                    ),
                    "pipeline_parallel_size": (
                        timing_cache.target.model.pipeline_parallel_size
                    ),
                    "software_versions": dict(timing_cache.target.software.versions),
                    "source_revisions": dict(
                        timing_cache.target.software.source_revisions
                    ),
                },
                "runtime_plan_binding_scope": {
                    "profile": timing_cache.target.model.profile,
                },
                "tuning_domain": {
                    "id": domain.domain_id if domain is not None else None,
                    "source": domain_source,
                },
                "measurement_provenance_scope": {
                    "image_digest": timing_cache.target.software.image_digest,
                    "phase": timing_cache.target.workload.phase,
                    "batch_size": timing_cache.target.workload.batch_size.to_document(),
                    "tokens": timing_cache.target.workload.tokens.to_document(),
                },
            }
        )
    matches = [item for item in decisions if item["matches"]]
    if runtime_target is None:
        status = "context_not_requested"
        reason = "showing all sealed variants"
        selected_variant = None
    elif len(matches) == 1:
        status = "selected"
        reason = "exact static runtime-key match"
        selected_variant = matches[0]["variant_id"]
    elif not matches:
        status = "fallback"
        reason = "no variant matches; retain vLLM-MUSA baseline"
        selected_variant = None
    else:
        status = "fallback"
        reason = "multiple variants match; refuse ambiguous config-time selection"
        selected_variant = None
    summary["runtime_decision"] = {
        "status": status,
        "reason": reason,
        "selected_variant": selected_variant,
        "variants": decisions,
    }
    return summary


def load_json_document(text: str, *, source: str) -> dict[str, Any]:
    try:
        value = load_json(text, source=source)
    except ValueError as exc:
        raise EnginePlanError(f"Unable to parse JSON from {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise EnginePlanError(f"{source} root must be a JSON object")
    return value
