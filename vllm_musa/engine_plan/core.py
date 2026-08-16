# SPDX-License-Identifier: Apache-2.0

"""Strict, framework-neutral engine-plan schema and fingerprint support."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_io import ArtifactFileError, load_json_object_file
from .artifacts import (
    PlanningArtifactError,
    TacticKind,
    TimingCache,
    runtime_key_fingerprint,
    static_topology_differences,
    static_workload_scope_differences,
)
from .selection import (
    CONTEXTUAL_SELECTION_SCHEMA,
    BuildPolicy,
    build_runtime_decisions,
    compute_context_fingerprint,
    select_operation,
)
from .tuning_domains import resolve_timing_cache_domain

LEGACY_PLAN_SCHEMA_VERSION = 5
PLAN_SCHEMA_VERSION = 6
SUPPORTED_PLAN_SCHEMA_VERSIONS = (LEGACY_PLAN_SCHEMA_VERSION, PLAN_SCHEMA_VERSION)
FINGERPRINT_PREFIX = "sha256:"


@dataclass(frozen=True)
class PluginIdentity:
    name: str
    version: str
    namespace: str
    abi_version: int

    def __post_init__(self) -> None:
        for field_name in ("name", "version", "namespace"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if (
            not isinstance(self.abi_version, int)
            or isinstance(self.abi_version, bool)
            or self.abi_version <= 0
        ):
            raise ValueError("abi_version must be a positive integer")


PLUGIN_IDENTITY = PluginIdentity(
    name="musa_engine_plan",
    version="0.2.0",
    namespace="musa",
    abi_version=1,
)


class EnginePlanError(ValueError):
    """Raised when a plan is malformed, incompatible, or has been modified."""


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


@dataclass(frozen=True)
class EnginePlan:
    schema_version: int
    plan_id: str
    plugin: Mapping[str, Any]
    compatibility: Mapping[str, Any]
    variants: tuple[Mapping[str, Any], ...]
    fingerprint: str


def _expect_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EnginePlanError(f"{field} must be a JSON object")
    return value


def _expect_exact_keys(
    value: Mapping[str, Any],
    field: str,
    expected: set[str],
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise EnginePlanError(
            f"{field} keys do not match schema; missing={missing}, unknown={unknown}"
        )


def _canonical_payload(document: Mapping[str, Any]) -> bytes:
    payload = {key: value for key, value in document.items() if key != "fingerprint"}
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _expect_prefixes(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise EnginePlanError(f"{field} must be a non-empty list")
    if any(not isinstance(prefix, str) or not prefix for prefix in value):
        raise EnginePlanError(f"{field} entries must be non-empty strings")
    if len(value) != len(set(value)):
        raise EnginePlanError(f"{field} must not contain duplicates")
    return tuple(value)


def _validate_runtime_decision_value(value: Any, field: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EnginePlanError(f"{field} must be finite")
        return
    if isinstance(value, str):
        if not value:
            raise EnginePlanError(f"{field} must not be an empty string")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_runtime_decision_value(item, f"{field}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise EnginePlanError(f"{field} keys must be non-empty strings")
            _validate_runtime_decision_value(item, f"{field}.{key}")
        return
    raise EnginePlanError(f"{field} must be a JSON-like runtime-decision value")


def _validate_runtime_decisions(value: Any, *, field: str) -> dict[str, Any]:
    projection = _expect_object(value, field)
    required_fields = {"profile", "values"}
    profile_config_fields = {
        "profile_config_id",
        "profile_config_fingerprint",
    }
    present_profile_config_fields = set(projection) & profile_config_fields
    if present_profile_config_fields and (
        present_profile_config_fields != profile_config_fields
    ):
        raise EnginePlanError(
            f"{field} profile-config identity fields must be set together"
        )
    _expect_exact_keys(
        projection,
        field,
        required_fields | present_profile_config_fields,
    )
    profile = projection["profile"]
    if not isinstance(profile, str) or not profile:
        raise EnginePlanError(f"{field}.profile must be a non-empty string")
    if present_profile_config_fields:
        profile_config_id = projection["profile_config_id"]
        if not isinstance(profile_config_id, str) or not profile_config_id:
            raise EnginePlanError(
                f"{field}.profile_config_id must be a non-empty string"
            )
        profile_config_fingerprint = projection["profile_config_fingerprint"]
        if (
            not isinstance(profile_config_fingerprint, str)
            or not profile_config_fingerprint.startswith("sha256:")
            or len(profile_config_fingerprint) != len("sha256:") + 64
            or any(
                character not in "0123456789abcdef"
                for character in profile_config_fingerprint.removeprefix("sha256:")
            )
        ):
            raise EnginePlanError(
                f"{field}.profile_config_fingerprint must be a canonical SHA-256"
            )
        from vllm_musa.runtime_plan.declarative import (
            DeclarativeProfileError,
            declarative_profile_identity,
        )

        try:
            live_profile_config = declarative_profile_identity(profile)
        except DeclarativeProfileError as exc:
            raise EnginePlanError(
                f"{field} cannot resolve live profile-config identity: {exc}"
            ) from exc
        artifact_profile_config = (
            profile_config_id,
            profile_config_fingerprint,
        )
        if live_profile_config != artifact_profile_config:
            raise EnginePlanError(
                f"{field} profile-config identity differs from the live source: "
                f"artifact={artifact_profile_config!r}, "
                f"live={live_profile_config!r}"
            )
    values = _expect_object(projection["values"], f"{field}.values")
    if not values:
        raise EnginePlanError(f"{field}.values must not be empty")
    requires_profile_config = False
    for key, decision_value in values.items():
        if not isinstance(key, str) or not key:
            raise EnginePlanError(
                f"{field}.values decision keys must be non-empty strings"
            )
        _validate_runtime_decision_value(
            decision_value,
            f"{field}.values.{key}",
        )
        from vllm_musa.runtime_plan.catalog import (
            RuntimeDecisionTunability,
            runtime_decision_spec,
            validate_runtime_decision_value,
        )
        from vllm_musa.runtime_plan.types import RuntimeDecision

        try:
            decision = RuntimeDecision(key)
        except ValueError as exc:
            raise EnginePlanError(
                f"{field}.values names unknown runtime decision {key!r}"
            ) from exc
        immutable_value = _immutable_runtime_decision_value(decision_value)
        try:
            validate_runtime_decision_value(
                decision,
                immutable_value,
            )
        except (TypeError, ValueError) as exc:
            raise EnginePlanError(
                f"{field}.values.{key} is not accepted by the host runtime "
                f"decision catalog: {exc}"
            ) from exc
        spec = runtime_decision_spec(decision)
        requires_profile_config = requires_profile_config or not spec.external_only
        if spec.tunability is RuntimeDecisionTunability.FIXED:
            raise EnginePlanError(
                f"{field}.values.{key} cannot project a fixed runtime decision"
            )
        if immutable_value != spec.fallback and not spec.supports_profile(profile):
            raise EnginePlanError(
                f"{field}.values.{key} is not supported by runtime profile "
                f"{profile!r}"
            )
    if requires_profile_config and not present_profile_config_fields:
        raise EnginePlanError(
            f"{field} contains profile-bound decisions without an exact "
            "profile-config identity"
        )
    if "vllm.ir_op_priority" in values:
        priorities = _expect_object(
            values["vllm.ir_op_priority"],
            f"{field}.values.vllm.ir_op_priority",
        )
        if not priorities:
            raise EnginePlanError(
                f"{field}.values.vllm.ir_op_priority must not be empty"
            )
        _validate_priorities(
            priorities,
            f"{field}.values.vllm.ir_op_priority",
        )
    return projection


def _immutable_runtime_decision_value(value: Any) -> Any:
    """Project JSON containers into the immutable RuntimePlan value codec."""

    if isinstance(value, dict):
        return tuple(
            (key, _immutable_runtime_decision_value(item))
            for key, item in sorted(value.items())
        )
    if isinstance(value, list):
        return tuple(_immutable_runtime_decision_value(item) for item in value)
    return value


def _validate_priorities(value: Mapping[str, Any], field: str) -> None:
    for op_name, providers in value.items():
        if not isinstance(op_name, str) or not op_name:
            raise EnginePlanError(f"{field} operation names must be non-empty strings")
        if not isinstance(providers, list) or not providers:
            raise EnginePlanError(f"Provider priority for {op_name} must be a list")
        if any(not isinstance(provider, str) or not provider for provider in providers):
            raise EnginePlanError(f"Providers for {op_name} must be non-empty strings")
        if len(providers) != len(set(providers)):
            raise EnginePlanError(
                f"Provider priority for {op_name} contains duplicates"
            )
        if providers[-1] != "native":
            raise EnginePlanError(
                f"Provider priority for {op_name} must retain native as final fallback"
            )


def _expect_nonnegative_number(value: Any, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not _is_finite_number(value)
        or value < 0
    ):
        raise EnginePlanError(f"{field} must be a finite non-negative number")
    return float(value)


def _validate_selection(
    value: Any,
    *,
    field: str,
    timing_cache: TimingCache,
) -> None:
    selection = _expect_object(value, field)
    _expect_exact_keys(
        selection,
        field,
        {
            "selection_id",
            "kind",
            "operation",
            "winner",
            "fallback",
            "metric",
            "winner_value",
            "winner_p95",
            "fallback_value",
            "fallback_p95",
            "speedup_pct",
            "samples",
            "reason",
            "coverage",
            "rejected",
        },
    )
    for name in (
        "selection_id",
        "kind",
        "operation",
        "winner",
        "fallback",
        "metric",
        "reason",
    ):
        if not isinstance(selection[name], str) or not selection[name]:
            raise EnginePlanError(f"{field}.{name} must be a non-empty string")
    if (
        not isinstance(selection["samples"], int)
        or isinstance(selection["samples"], bool)
        or selection["samples"] < 0
    ):
        raise EnginePlanError(f"{field}.samples must be a non-negative integer")
    for name in (
        "winner_value",
        "winner_p95",
        "fallback_value",
        "fallback_p95",
        "speedup_pct",
    ):
        value = selection[name]
        if value is not None:
            _expect_nonnegative_number(value, f"{field}.{name}")
    coverage_field = f"{field}.coverage"
    coverage = _expect_object(selection["coverage"], coverage_field)
    _expect_exact_keys(
        coverage,
        coverage_field,
        {
            "schema",
            "case_schema",
            "status",
            "activation_differences",
            "required_buckets",
            "observed_buckets",
            "error",
        },
    )
    for schema_name in ("schema", "case_schema"):
        if not isinstance(coverage[schema_name], str) or not coverage[schema_name]:
            raise EnginePlanError(
                f"{coverage_field}.{schema_name} must be a non-empty string"
            )
    if coverage["status"] not in {"complete", "incomplete", "evidence_only"}:
        raise EnginePlanError(
            f"{coverage_field}.status must be complete, incomplete, or evidence_only"
        )
    differences = coverage["activation_differences"]
    if not isinstance(differences, list) or any(
        not isinstance(item, str) or not item for item in differences
    ):
        raise EnginePlanError(
            f"{coverage_field}.activation_differences must be a list of strings"
        )
    for bucket_list_name in ("required_buckets", "observed_buckets"):
        buckets = coverage[bucket_list_name]
        if not isinstance(buckets, list):
            raise EnginePlanError(f"{coverage_field}.{bucket_list_name} must be a list")
        for index, raw_bucket in enumerate(buckets):
            bucket_field = f"{coverage_field}.{bucket_list_name}[{index}]"
            bucket = _expect_object(raw_bucket, bucket_field)
            _expect_exact_keys(
                bucket,
                bucket_field,
                {
                    "case_fingerprint",
                    "phase",
                    "batch_size",
                    "tokens",
                    "rows",
                    "hidden_size",
                    "dtype",
                },
            )
            for dimension in ("batch_size", "tokens", "rows", "hidden_size"):
                if (
                    not isinstance(bucket[dimension], int)
                    or isinstance(bucket[dimension], bool)
                    or bucket[dimension] <= 0
                ):
                    raise EnginePlanError(
                        f"{bucket_field}.{dimension} must be a positive integer"
                    )
            for string_field in ("case_fingerprint", "phase", "dtype"):
                if (
                    not isinstance(bucket[string_field], str)
                    or not bucket[string_field]
                ):
                    raise EnginePlanError(
                        f"{bucket_field}.{string_field} must be a string"
                    )
    if coverage["error"] is not None and (
        not isinstance(coverage["error"], str) or not coverage["error"]
    ):
        raise EnginePlanError(f"{coverage_field}.error must be null or a string")
    rejected = selection["rejected"]
    if not isinstance(rejected, list):
        raise EnginePlanError(f"{field}.rejected must be a list")
    for index, rejected_value in enumerate(rejected):
        rejected_field = f"{field}.rejected[{index}]"
        rejected_item = _expect_object(rejected_value, rejected_field)
        _expect_exact_keys(
            rejected_item,
            rejected_field,
            {"tactic_id", "reason", "median", "samples"},
        )
        for name in ("tactic_id", "reason"):
            if not isinstance(rejected_item[name], str) or not rejected_item[name]:
                raise EnginePlanError(
                    f"{rejected_field}.{name} must be a non-empty string"
                )
        if (
            not isinstance(rejected_item["samples"], int)
            or isinstance(rejected_item["samples"], bool)
            or rejected_item["samples"] < 0
        ):
            raise EnginePlanError(
                f"{rejected_field}.samples must be a non-negative integer"
            )
        if rejected_item["median"] is not None:
            _expect_nonnegative_number(
                rejected_item["median"], f"{rejected_field}.median"
            )
    tactics = {item.tactic_id: item for item in timing_cache.catalog}
    for name in ("winner", "fallback"):
        tactic_id = selection[name]
        if tactic_id not in tactics:
            raise EnginePlanError(
                f"{field}.{name} references unknown tactic {tactic_id!r}"
            )
    winner = tactics[selection["winner"]]
    fallback = tactics[selection["fallback"]]
    if (
        winner.operation != selection["operation"]
        or fallback.operation != selection["operation"]
    ):
        raise EnginePlanError(f"{field} tactic operation does not match selection")
    if winner.kind.value != selection["kind"] or fallback.kind is not winner.kind:
        raise EnginePlanError(f"{field} tactic kind does not match selection")
    if winner.fallback_id != fallback.tactic_id:
        raise EnginePlanError(f"{field} winner does not reference recorded fallback")


def _validate_contextual_selection(
    value: Any,
    *,
    field: str,
    timing_cache: TimingCache,
) -> None:
    selection = _expect_object(value, field)
    _expect_exact_keys(
        selection,
        field,
        {
            "selection_schema",
            "selection_id",
            "context_schema",
            "max_iqr_ratio",
            "kind",
            "operation",
            "fallback",
            "contexts",
        },
    )
    if selection["selection_schema"] != CONTEXTUAL_SELECTION_SCHEMA:
        raise EnginePlanError(
            f"{field}.selection_schema must be {CONTEXTUAL_SELECTION_SCHEMA!r}"
        )
    for name in (
        "selection_id",
        "context_schema",
        "kind",
        "operation",
        "fallback",
    ):
        if not isinstance(selection[name], str) or not selection[name]:
            raise EnginePlanError(f"{field}.{name} must be a non-empty string")
    max_iqr_ratio = selection["max_iqr_ratio"]
    if (
        not isinstance(max_iqr_ratio, (int, float))
        or isinstance(max_iqr_ratio, bool)
        or not math.isfinite(float(max_iqr_ratio))
        or not 0 < float(max_iqr_ratio) <= 1
    ):
        raise EnginePlanError(f"{field}.max_iqr_ratio must be in (0, 1]")
    tactics = {item.tactic_id: item for item in timing_cache.catalog}
    fallback_id = selection["fallback"]
    fallback = tactics.get(fallback_id)
    if fallback is None:
        raise EnginePlanError(
            f"{field}.fallback references unknown tactic {fallback_id!r}"
        )
    if (
        fallback.operation != selection["operation"]
        or fallback.kind.value != selection["kind"]
    ):
        raise EnginePlanError(f"{field} fallback does not match selection group")
    contexts = selection["contexts"]
    if not isinstance(contexts, list) or not contexts:
        raise EnginePlanError(f"{field}.contexts must be a non-empty list")
    context_ids: set[str] = set()
    for index, raw_context in enumerate(contexts):
        context_field = f"{field}.contexts[{index}]"
        context = _expect_object(raw_context, context_field)
        _expect_exact_keys(
            context,
            context_field,
            {
                "context_id",
                "shape",
                "token_bucket",
                "winner",
                "fallback",
                "metric",
                "speedup_pct",
                "samples",
                "reason",
                "evidence",
                "rejected",
            },
        )
        for name in ("context_id", "winner", "fallback", "metric", "reason"):
            if not isinstance(context[name], str) or not context[name]:
                raise EnginePlanError(
                    f"{context_field}.{name} must be a non-empty string"
                )
        context_id = context["context_id"]
        if context_id in context_ids:
            raise EnginePlanError(f"{field}.contexts contains duplicate context_id")
        context_ids.add(context_id)
        shape = _expect_object(context["shape"], f"{context_field}.shape")
        if not shape:
            raise EnginePlanError(f"{context_field}.shape must not be empty")
        bucket = _expect_object(
            context["token_bucket"], f"{context_field}.token_bucket"
        )
        _expect_exact_keys(
            bucket,
            f"{context_field}.token_bucket",
            {"min", "max"},
        )
        for name in ("min", "max"):
            if (
                not isinstance(bucket[name], int)
                or isinstance(bucket[name], bool)
                or bucket[name] <= 0
            ):
                raise EnginePlanError(
                    f"{context_field}.token_bucket.{name} must be positive"
                )
        if bucket["min"] > bucket["max"]:
            raise EnginePlanError(
                f"{context_field}.token_bucket.min must not exceed max"
            )
        expected_context_id = compute_context_fingerprint(
            {"shape": shape, "token_bucket": bucket}
        )
        if context_id != expected_context_id:
            raise EnginePlanError(
                f"{context_field}.context_id mismatch: expected "
                f"{expected_context_id}, got {context_id}"
            )
        if context["fallback"] != fallback_id:
            raise EnginePlanError(
                f"{context_field}.fallback does not match selection fallback"
            )
        winner = tactics.get(context["winner"])
        if winner is None:
            raise EnginePlanError(
                f"{context_field}.winner references an unknown tactic"
            )
        if winner.operation != fallback.operation or winner.kind is not fallback.kind:
            raise EnginePlanError(
                f"{context_field}.winner does not match selection group"
            )
        _expect_nonnegative_number(
            context["speedup_pct"], f"{context_field}.speedup_pct"
        )
        if (
            not isinstance(context["samples"], int)
            or isinstance(context["samples"], bool)
            or context["samples"] < 0
        ):
            raise EnginePlanError(
                f"{context_field}.samples must be a non-negative integer"
            )
        evidence = context["evidence"]
        if not isinstance(evidence, list) or not evidence:
            raise EnginePlanError(f"{context_field}.evidence must be non-empty")
        for evidence_index, raw_evidence in enumerate(evidence):
            evidence_field = f"{context_field}.evidence[{evidence_index}]"
            evidence_item = _expect_object(raw_evidence, evidence_field)
            _expect_exact_keys(
                evidence_item,
                evidence_field,
                {
                    "workload_tokens",
                    "route_mode",
                    "seed",
                    "winner_median",
                    "winner_p95",
                    "winner_iqr",
                    "fallback_median",
                    "fallback_p95",
                    "fallback_iqr",
                    "speedup_pct",
                    "samples",
                },
            )
            for name in ("workload_tokens", "samples"):
                if (
                    not isinstance(evidence_item[name], int)
                    or isinstance(evidence_item[name], bool)
                    or evidence_item[name] <= 0
                ):
                    raise EnginePlanError(
                        f"{evidence_field}.{name} must be a positive integer"
                    )
            if (
                not isinstance(evidence_item["seed"], int)
                or isinstance(evidence_item["seed"], bool)
                or evidence_item["seed"] < 0
            ):
                raise EnginePlanError(
                    f"{evidence_field}.seed must be a non-negative integer"
                )
            if (
                not isinstance(evidence_item["route_mode"], str)
                or not evidence_item["route_mode"]
            ):
                raise EnginePlanError(
                    f"{evidence_field}.route_mode must be a non-empty string"
                )
            for name in (
                "winner_median",
                "winner_p95",
                "winner_iqr",
                "fallback_median",
                "fallback_p95",
                "fallback_iqr",
                "speedup_pct",
            ):
                _expect_nonnegative_number(
                    evidence_item[name], f"{evidence_field}.{name}"
                )
        rejected = context["rejected"]
        if not isinstance(rejected, list):
            raise EnginePlanError(f"{context_field}.rejected must be a list")
        for rejected_index, raw_rejected in enumerate(rejected):
            rejected_field = f"{context_field}.rejected[{rejected_index}]"
            rejected_item = _expect_object(raw_rejected, rejected_field)
            _expect_exact_keys(rejected_item, rejected_field, {"tactic_id", "reason"})
            for name in ("tactic_id", "reason"):
                if not isinstance(rejected_item[name], str) or not rejected_item[name]:
                    raise EnginePlanError(
                        f"{rejected_field}.{name} must be a non-empty string"
                    )


def _validate_variants(
    value: Any,
    *,
    schema_version: int,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise EnginePlanError("variants must be a non-empty list")
    parsed: list[Mapping[str, Any]] = []
    variant_ids: set[str] = set()
    early_keys: set[str] = set()
    saw_contextual_selection = False
    for index, raw_variant in enumerate(value):
        field = f"variants[{index}]"
        variant = _expect_object(raw_variant, field)
        _expect_exact_keys(
            variant,
            field,
            {
                "variant_id",
                "target_fingerprint",
                "early_runtime_key",
                "final_runtime_key",
                "timing_cache",
                "policy",
                "selections",
                "runtime_decisions",
            },
        )
        variant_id = variant["variant_id"]
        if not isinstance(variant_id, str) or not variant_id:
            raise EnginePlanError(f"{field}.variant_id must be a non-empty string")
        if variant_id in variant_ids:
            raise EnginePlanError(f"Duplicate variant_id {variant_id!r}")
        variant_ids.add(variant_id)
        try:
            timing_cache = TimingCache.from_document(
                variant["timing_cache"],
                require_fingerprint=True,
            )
        except PlanningArtifactError as exc:
            raise EnginePlanError(f"{field}: {exc}") from exc
        try:
            resolve_timing_cache_domain(timing_cache)
        except PlanningArtifactError as exc:
            raise EnginePlanError(f"{field}: {exc}") from exc
        workload_differences = static_workload_scope_differences(
            timing_cache.target.workload
        )
        if workload_differences:
            raise EnginePlanError(
                f"{field} does not cover the static serving workload: "
                f"{list(workload_differences)}"
            )
        topology_differences = static_topology_differences(
            timing_cache.target.model,
            operations=tuple(tactic.operation for tactic in timing_cache.catalog),
        )
        if topology_differences:
            raise EnginePlanError(
                f"{field} uses an unsupported static topology: "
                f"{list(topology_differences)}"
            )
        expected_values = {
            "target_fingerprint": timing_cache.target.fingerprint,
            "early_runtime_key": runtime_key_fingerprint(
                timing_cache.target,
                final=False,
            ),
            "final_runtime_key": runtime_key_fingerprint(
                timing_cache.target,
                final=True,
            ),
        }
        for name, expected in expected_values.items():
            if variant[name] != expected:
                raise EnginePlanError(
                    f"{field}.{name} mismatch: expected {expected}, "
                    f"got {variant[name]}"
                )
        early_key = variant["early_runtime_key"]
        if early_key in early_keys:
            raise EnginePlanError(
                "variants contain an ambiguous early runtime key; dynamic request "
                "buckets require a runtime provider selector, not two config plans"
            )
        early_keys.add(early_key)
        try:
            policy = BuildPolicy.from_document(
                variant["policy"], field=f"{field}.policy"
            )
        except PlanningArtifactError as exc:
            raise EnginePlanError(str(exc)) from exc
        selections = variant["selections"]
        if not isinstance(selections, list) or not selections:
            raise EnginePlanError(f"{field}.selections must be a non-empty list")
        has_contextual_selection = any(
            isinstance(selection, dict)
            and selection.get("selection_schema") == CONTEXTUAL_SELECTION_SCHEMA
            for selection in selections
        )
        if schema_version == LEGACY_PLAN_SCHEMA_VERSION and has_contextual_selection:
            raise EnginePlanError(
                f"{field} contextual selections require plan schema 6"
            )
        saw_contextual_selection = saw_contextual_selection or has_contextual_selection
        for selection_index, selection in enumerate(selections):
            selection_field = f"{field}.selections[{selection_index}]"
            if (
                isinstance(selection, dict)
                and selection.get("selection_schema") == CONTEXTUAL_SELECTION_SCHEMA
            ):
                _validate_contextual_selection(
                    selection,
                    field=selection_field,
                    timing_cache=timing_cache,
                )
            else:
                _validate_selection(
                    selection,
                    field=selection_field,
                    timing_cache=timing_cache,
                )
        grouped: dict[tuple[TacticKind, str], list[Any]] = {}
        observations: dict[str, list[Any]] = {}
        for tactic in timing_cache.catalog:
            grouped.setdefault((tactic.kind, tactic.operation), []).append(tactic)
        for observation in timing_cache.observations:
            observations.setdefault(observation.tactic_id, []).append(observation)
        try:
            expected_selections = [
                select_operation(
                    timing_cache,
                    grouped[key],
                    observations,
                    policy,
                )
                for key in sorted(grouped, key=lambda item: (item[0].value, item[1]))
            ]
        except PlanningArtifactError as exc:
            raise EnginePlanError(
                f"{field}: unable to recompute selections: {exc}"
            ) from exc
        actual_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
        for selection in selections:
            key = (selection["kind"], selection["operation"])
            if key in actual_by_key:
                raise EnginePlanError(f"{field}.selections contains duplicate {key!r}")
            actual_by_key[key] = selection
        expected_by_key = {
            (selection["kind"], selection["operation"]): selection
            for selection in expected_selections
        }
        if set(actual_by_key) != set(expected_by_key):
            raise EnginePlanError(
                f"{field}.selections does not cover the sealed tactic catalog"
            )
        for key, expected_selection in expected_by_key.items():
            if actual_by_key[key] != expected_selection:
                raise EnginePlanError(
                    f"{field}.selection {key!r} does not match sealed timing "
                    "evidence and policy"
                )
        decisions = _validate_runtime_decisions(
            variant["runtime_decisions"],
            field=f"{field}.runtime_decisions",
        )
        if decisions["profile"] != timing_cache.target.model.profile:
            raise EnginePlanError(
                f"{field}.runtime_decisions.profile does not match the sealed "
                "timing target"
            )
        expected_decisions = build_runtime_decisions(timing_cache, selections)
        for key, expected_value in expected_decisions["values"].items():
            actual_value = decisions["values"].get(key, object())
            if actual_value != expected_value:
                raise EnginePlanError(
                    f"{field}.runtime_decisions.values.{key} does not match its "
                    "sealed tactic selections"
                )
        parsed.append(dict(variant))
    if schema_version == PLAN_SCHEMA_VERSION and not saw_contextual_selection:
        raise EnginePlanError("schema 6 plans require a contextual selection")
    return tuple(parsed)


def compute_fingerprint(document: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_payload(document)).hexdigest()
    return f"{FINGERPRINT_PREFIX}{digest}"


def _parse_document(
    document: Mapping[str, Any], *, require_fingerprint: bool
) -> EnginePlan:
    base_required = {
        "schema_version",
        "plan_id",
        "plugin",
        "compatibility",
    }
    schema_version = document.get("schema_version")
    if type(schema_version) is not int:
        raise EnginePlanError("plan.schema_version must be an integer")
    required = set(base_required)
    required.add("variants")
    if require_fingerprint:
        required.add("fingerprint")
    _expect_exact_keys(document, "plan", required)

    schema_version = document["schema_version"]
    if schema_version not in SUPPORTED_PLAN_SCHEMA_VERSIONS:
        raise EnginePlanError(
            f"Unsupported plan schema {schema_version!r}; expected one of "
            f"{SUPPORTED_PLAN_SCHEMA_VERSIONS}"
        )
    fingerprint = document.get("fingerprint", "")
    if require_fingerprint:
        if not isinstance(fingerprint, str) or not fingerprint.startswith(
            FINGERPRINT_PREFIX
        ):
            raise EnginePlanError("fingerprint must use the sha256: prefix")
        try:
            expected = compute_fingerprint(document)
        except (TypeError, ValueError) as exc:
            raise EnginePlanError(
                "plan fingerprint payload must contain only finite JSON values"
            ) from exc
        if fingerprint != expected:
            raise EnginePlanError(
                f"Plan fingerprint mismatch: expected {expected}, got {fingerprint}"
            )
    plan_id = document["plan_id"]
    if not isinstance(plan_id, str) or not plan_id:
        raise EnginePlanError("plan_id must be a non-empty string")

    plugin = _expect_object(document["plugin"], "plugin")
    _expect_exact_keys(
        plugin,
        "plugin",
        {"name", "version", "namespace", "abi_version"},
    )
    for field_name in ("name", "version", "namespace"):
        value = plugin[field_name]
        if not isinstance(value, str) or not value:
            raise EnginePlanError(f"plugin.{field_name} must be a non-empty string")
    abi_version = plugin["abi_version"]
    if (
        not isinstance(abi_version, int)
        or isinstance(abi_version, bool)
        or abi_version <= 0
    ):
        raise EnginePlanError("plugin.abi_version must be a positive integer")

    compatibility = _expect_object(document["compatibility"], "compatibility")
    _expect_exact_keys(
        compatibility,
        "compatibility",
        {
            "framework",
            "framework_version_prefixes",
            "platform",
            "vllm_musa_version_prefixes",
        },
    )
    for field_name in ("framework", "platform"):
        value = compatibility[field_name]
        if not isinstance(value, str) or not value:
            raise EnginePlanError(
                f"compatibility.{field_name} must be a non-empty string"
            )
    _expect_prefixes(
        compatibility["framework_version_prefixes"],
        "compatibility.framework_version_prefixes",
    )
    _expect_prefixes(
        compatibility["vllm_musa_version_prefixes"],
        "compatibility.vllm_musa_version_prefixes",
    )

    variants = _validate_variants(
        document["variants"],
        schema_version=schema_version,
    )

    return EnginePlan(
        schema_version=schema_version,
        plan_id=plan_id,
        plugin=dict(plugin),
        compatibility=dict(compatibility),
        variants=variants,
        fingerprint=fingerprint,
    )


def seal_plan_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and add a canonical fingerprint to an unsealed plan."""

    unsealed = dict(document)
    unsealed.pop("fingerprint", None)
    _parse_document(unsealed, require_fingerprint=False)
    try:
        sealed = json.loads(json.dumps(unsealed, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise EnginePlanError(
            "plan payload must contain only finite JSON values"
        ) from exc
    sealed["fingerprint"] = compute_fingerprint(sealed)
    return sealed


def load_plan(path: str | Path) -> EnginePlan:
    """Load a sealed plan and fail closed on schema or fingerprint drift."""

    try:
        document = load_json_object_file(path)
    except ArtifactFileError as exc:
        raise EnginePlanError(f"Unable to load engine plan {path}: {exc}") from exc
    return _parse_document(document, require_fingerprint=True)


def parse_plan_document(document: Mapping[str, Any]) -> EnginePlan:
    """Validate an already loaded sealed plan document."""

    return _parse_document(document, require_fingerprint=True)
