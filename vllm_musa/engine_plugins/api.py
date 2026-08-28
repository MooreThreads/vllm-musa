# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Internal interfaces for the vLLM-MUSA engine-plan adapter.

The split mirrors the useful part of TensorRT's V3 plugin contract: immutable
identity lives in the core metadata, offline planning is a build capability,
and serving-time configuration is a runtime capability.  The interfaces are
Python-only and do not depend on TensorRT, CUDA, torch, or MUSA runtime modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

ENGINE_PLUGIN_ABI_VERSION = 1
ENGINE_PLAN_ENV = "MUSA_ENGINE_PLAN"
# The parent process pins the content identity before spawning workers. An
# orchestrator may also set this explicitly when mounting an immutable plan.
ENGINE_PLAN_FINGERPRINT_ENV = "MUSA_ENGINE_PLAN_FINGERPRINT"
# Production deployments can require the fingerprint to come from trusted
# deployment configuration instead of allowing the parent to derive it from
# the artifact it is about to consume.
ENGINE_PLAN_REQUIRE_PINNED_FINGERPRINT_ENV = (
    "MUSA_ENGINE_PLAN_REQUIRE_PINNED_FINGERPRINT"
)
# Private spawn transport for the exact selected variant (or the baseline
# sentinel). Plan fingerprint + variant identify the full decision projection.
ENGINE_PLAN_VARIANT_ENV = "VLLM_MUSA_INTERNAL_ENGINE_PLAN_VARIANT"


@dataclass(frozen=True)
class EnginePluginMetadata:
    """Immutable identity shared by build and runtime capabilities."""

    name: str
    version: str
    namespace: str = "musa"
    abi_version: int = ENGINE_PLUGIN_ABI_VERSION

    def __post_init__(self) -> None:
        for field_name in ("name", "version", "namespace"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Engine plugin {field_name} must be a non-empty string"
                )
        if (
            not isinstance(self.abi_version, int)
            or isinstance(self.abi_version, bool)
            or self.abi_version <= 0
        ):
            raise ValueError("Engine plugin ABI version must be positive")


class EnginePluginCore(Protocol):
    """Framework-neutral identity contract shared by all capabilities."""

    name: str
    version: str
    namespace: str
    abi_version: int


@dataclass(frozen=True, slots=True)
class EngineIrProviderMetadata:
    """Stable host projection of one supported vLLM IR provider."""

    operation: str
    provider: str
    implementation_fingerprint: str

    def __post_init__(self) -> None:
        for field_name in (
            "operation",
            "provider",
            "implementation_fingerprint",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True)
class EnginePlanApplication:
    """Result of applying a plan before vLLM freezes platform defaults.

    ``context_fingerprint`` identifies the early activation context.
    """

    plugin_name: str
    plan_id: str
    plan_fingerprint: str
    applied_settings: tuple[str, ...]
    selected_variant: str = ""
    selected_tactics: tuple[str, ...] = ()
    context_fingerprint: str = ""
    fallback_reason: str = ""


@dataclass(frozen=True)
class EnginePlanReceipt:
    """Final receipt emitted after vLLM-MUSA validates the resolved config.

    ``context_fingerprint`` identifies the separately collected final context.
    """

    plugin_name: str
    plugin_version: str
    plan_id: str
    plan_fingerprint: str
    validated_settings: tuple[str, ...]
    selected_variant: str = ""
    selected_tactics: tuple[str, ...] = ()
    context_fingerprint: str = ""
    fallback_reason: str = ""


def _validate_runtime_decision_value(value: Any, field: str) -> None:
    if isinstance(value, bool | int | str):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
        return
    if not isinstance(value, tuple):
        raise ValueError(f"{field} must be an immutable JSON-like value")
    for index, item in enumerate(value):
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            _validate_runtime_decision_value(item[1], f"{field}.{item[0]}")
        else:
            _validate_runtime_decision_value(item, f"{field}[{index}]")


def _validate_profile_config_identity(
    profile_config_id: str | None,
    profile_config_fingerprint: str | None,
) -> None:
    if (profile_config_id is None) != (profile_config_fingerprint is None):
        raise ValueError("profile-config identity fields must be set together")
    if profile_config_id is None:
        return
    if not isinstance(profile_config_id, str) or not profile_config_id:
        raise ValueError("profile_config_id must be a non-empty string")
    assert profile_config_fingerprint is not None
    if not isinstance(
        profile_config_fingerprint, str
    ) or not profile_config_fingerprint.startswith("sha256:"):
        raise ValueError("profile_config_fingerprint must use the sha256: prefix")


@dataclass(frozen=True, slots=True)
class EngineRuntimeDecisions:
    """Typed decisions projected from one already selected plan variant."""

    plan_id: str
    plan_fingerprint: str
    profile: str
    decisions: tuple[tuple[str, Any], ...]
    profile_config_id: str | None = None
    profile_config_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan_id, str) or not self.plan_id:
            raise ValueError("plan_id must be a non-empty string")
        if not isinstance(self.plan_fingerprint, str) or not self.plan_fingerprint:
            raise ValueError("plan_fingerprint must be a non-empty string")
        if not isinstance(self.profile, str) or not self.profile:
            raise ValueError("profile must be a non-empty string")
        if not isinstance(self.decisions, tuple) or not self.decisions:
            raise ValueError("runtime decisions must be a non-empty tuple")
        _validate_profile_config_identity(
            self.profile_config_id,
            self.profile_config_fingerprint,
        )
        keys: list[str] = []
        for key, value in self.decisions:
            if not isinstance(key, str) or not key:
                raise ValueError("runtime decision keys must be non-empty strings")
            keys.append(key)
            _validate_runtime_decision_value(value, f"decisions.{key}")
        if len(keys) != len(set(keys)):
            raise ValueError("runtime decision keys must not contain duplicates")


@dataclass(frozen=True, slots=True)
class EngineRuntimeDecisionReceipt:
    """Auditable result of one plugin runtime-decision projection."""

    plugin_name: str
    plugin_version: str
    plan_id: str
    plan_fingerprint: str
    profile: str
    decisions: tuple[tuple[str, Any], ...]
    profile_config_id: str | None = None
    profile_config_fingerprint: str | None = None


class EnginePlanBuilder(Protocol):
    """Optional offline build/tuning capability implemented outside the host."""

    metadata: EnginePluginCore

    def build(self, request: Any) -> Any:
        """Build and seal a framework plan without mutating a live runtime."""

        ...


class EngineRuntimePlugin(Protocol):
    """Serving-time capability registered by the built-in plan adapter."""

    metadata: EnginePluginMetadata

    def is_enabled(self) -> bool:
        """Return whether this plugin has an explicitly selected plan."""

        ...

    def apply_config_defaults(self, vllm_config: Any) -> EnginePlanApplication:
        """Apply allowlisted defaults before vLLM resolves IR/fusion policy."""

        ...

    def validate_runtime_config(
        self,
        vllm_config: Any,
        application: EnginePlanApplication,
    ) -> EnginePlanReceipt:
        """Validate the final config and return a provenance receipt."""

        ...


class ContextualEngineRuntimePlanProvider(Protocol):
    """Optional typed plan projection implemented by an ABI-v1 plugin."""

    def resolve_contextual_runtime_decisions(
        self,
        application: EnginePlanApplication,
    ) -> EngineRuntimeDecisions | None:
        """Return decisions from the exact plan variant already applied."""

        ...
