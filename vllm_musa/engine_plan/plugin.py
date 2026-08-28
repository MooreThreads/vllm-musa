# SPDX-License-Identifier: Apache-2.0

"""Built-in vLLM-MUSA adapter for sealed runtime plans."""

from __future__ import annotations

import os
import weakref
from dataclasses import dataclass
from threading import RLock
from typing import Any

from vllm_musa.engine_plugins import (
    ENGINE_PLAN_ENV,
    ENGINE_PLAN_FINGERPRINT_ENV,
    ENGINE_PLAN_REQUIRE_PINNED_FINGERPRINT_ENV,
    ENGINE_PLAN_VARIANT_ENV,
    EnginePlanApplication,
    EnginePlanReceipt,
    EnginePluginMetadata,
    EngineRuntimeDecisions,
    register_engine_runtime_plugin,
)

from .artifacts import runtime_key_fingerprint
from .core import PLUGIN_IDENTITY, EnginePlan, EnginePlanError, load_plan
from .runtime import select_runtime_variant, validate_runtime_variant

_TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_ENV_VALUES = frozenset({"0", "false", "no", "off"})


def _requires_pinned_fingerprint() -> bool:
    raw_value = os.environ.get(ENGINE_PLAN_REQUIRE_PINNED_FINGERPRINT_ENV)
    if raw_value is None:
        return False
    normalized = raw_value.strip().lower()
    if normalized in _TRUE_ENV_VALUES:
        return True
    if normalized in _FALSE_ENV_VALUES:
        return False
    allowed = ", ".join(sorted(_TRUE_ENV_VALUES | _FALSE_ENV_VALUES))
    raise EnginePlanError(
        f"{ENGINE_PLAN_REQUIRE_PINNED_FINGERPRINT_ENV} must be a boolean "
        f"value ({allowed}); got {raw_value!r}"
    )


def _immutable_decision_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            (key, _immutable_decision_value(item))
            for key, item in sorted(value.items())
        )
    if isinstance(value, list):
        return tuple(_immutable_decision_value(item) for item in value)
    return value


def _selection_winners(selection: dict[str, Any]) -> tuple[str, ...]:
    if "winner" in selection:
        return (selection["winner"],)
    return tuple(context["winner"] for context in selection["contexts"])


def _selected_tactics(variant: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                winner
                for selection in variant["selections"]
                for winner in _selection_winners(selection)
            }
        )
    )


@dataclass
class _ActivePlanState:
    config_ref: weakref.ReferenceType[Any]
    plan: EnginePlan
    variant: dict[str, Any] | None
    runtime_target: dict[str, Any]
    fallback_reason: str

    def config(self) -> Any | None:
        return self.config_ref()


class JsonPlanRuntimePlugin:
    metadata = EnginePluginMetadata(
        name=PLUGIN_IDENTITY.name,
        version=PLUGIN_IDENTITY.version,
        namespace=PLUGIN_IDENTITY.namespace,
        abi_version=PLUGIN_IDENTITY.abi_version,
    )

    def __init__(self) -> None:
        self._plans: dict[int, _ActivePlanState] = {}
        self._plans_lock = RLock()

    def _discard_plan_state(
        self,
        config_id: int,
        reference: weakref.ReferenceType[Any],
    ) -> None:
        with self._plans_lock:
            state = self._plans.get(config_id)
            if state is not None and state.config_ref is reference:
                self._plans.pop(config_id, None)

    def _new_plan_state(
        self,
        vllm_config: Any,
        *,
        plan: EnginePlan,
        variant: dict[str, Any] | None,
        runtime_target: dict[str, Any],
        fallback_reason: str,
    ) -> _ActivePlanState:
        config_id = id(vllm_config)
        try:
            reference = weakref.ref(
                vllm_config,
                lambda released, identity=config_id: self._discard_plan_state(
                    identity,
                    released,
                ),
            )
        except TypeError:
            raise EnginePlanError(
                "Engine-plan state requires a weak-referenceable vLLM configuration"
            ) from None
        return _ActivePlanState(
            config_ref=reference,
            plan=plan,
            variant=variant,
            runtime_target=runtime_target,
            fallback_reason=fallback_reason,
        )

    def is_enabled(self) -> bool:
        enabled = bool(os.environ.get(ENGINE_PLAN_ENV))
        if enabled:
            _requires_pinned_fingerprint()
        return enabled

    def _load_selected_plan(self) -> EnginePlan:
        path = os.environ.get(ENGINE_PLAN_ENV)
        if not path:
            raise EnginePlanError(f"{ENGINE_PLAN_ENV} is not set")
        require_pinned_fingerprint = _requires_pinned_fingerprint()
        expected = os.environ.get(ENGINE_PLAN_FINGERPRINT_ENV)
        if require_pinned_fingerprint and not expected:
            raise EnginePlanError(
                f"{ENGINE_PLAN_REQUIRE_PINNED_FINGERPRINT_ENV}=true requires "
                f"{ENGINE_PLAN_FINGERPRINT_ENV} to be explicitly set before "
                "the engine plan is loaded"
            )
        plan = load_plan(path)
        if expected is None:
            expected = os.environ.setdefault(
                ENGINE_PLAN_FINGERPRINT_ENV,
                plan.fingerprint,
            )
        if expected != plan.fingerprint:
            raise EnginePlanError(
                f"Engine plan fingerprint {plan.fingerprint!r} does not match "
                f"the parent-pinned {ENGINE_PLAN_FINGERPRINT_ENV}={expected!r}"
            )
        return plan

    @staticmethod
    def _pin_selected_variant(variant: dict[str, Any] | None) -> None:
        selected = variant["variant_id"] if variant is not None else "baseline"
        expected = os.environ.setdefault(ENGINE_PLAN_VARIANT_ENV, selected)
        if expected != selected:
            raise EnginePlanError(
                f"Engine plan selected variant {selected!r} does not match the "
                f"parent-pinned {ENGINE_PLAN_VARIANT_ENV}={expected!r}"
            )

    @staticmethod
    def _validate_compatibility(plan: EnginePlan) -> None:
        expected_identity = {
            "name": PLUGIN_IDENTITY.name,
            "namespace": PLUGIN_IDENTITY.namespace,
            "abi_version": PLUGIN_IDENTITY.abi_version,
        }
        actual_identity = {
            key: value for key, value in plan.plugin.items() if key != "version"
        }
        if (
            actual_identity != expected_identity
            or plan.plugin["version"] != PLUGIN_IDENTITY.version
        ):
            raise EnginePlanError(
                f"Plan targets plugin identity {plan.plugin}, "
                f"but runtime identity is {expected_identity} with version "
                f"{PLUGIN_IDENTITY.version!r}"
            )
        compatibility = plan.compatibility
        if compatibility["framework"] != "vllm":
            raise EnginePlanError("This adapter accepts only framework=vllm plans")
        if compatibility["platform"] != "musa":
            raise EnginePlanError("This adapter accepts only platform=musa plans")
        # The per-variant runtime selector performs the stricter exact
        # compiler/runtime package and source-revision check. Keep that check
        # beside the typed timing target so offline explain and live activation
        # cannot drift; these top-level prefixes remain an index/inspection aid.

    @staticmethod
    def _priorities(runtime_decisions: dict[str, Any]) -> dict[str, list[str]]:
        priorities = runtime_decisions["values"].get("vllm.ir_op_priority", {})
        return {name: list(providers) for name, providers in priorities.items()}

    @staticmethod
    def _fallback_reason(reason: str, differences: tuple[str, ...]) -> str:
        if not differences:
            return reason
        detail = "; ".join(differences[:4])
        if len(differences) > 4:
            detail += f"; ... ({len(differences) - 4} more)"
        return f"{reason}: {detail}"

    def apply_config_defaults(self, vllm_config: Any) -> EnginePlanApplication:
        plan = self._load_selected_plan()
        self._validate_compatibility(plan)
        decision = select_runtime_variant(plan, vllm_config)
        variant = decision.variant
        self._pin_selected_variant(variant)
        runtime_target = decision.runtime_target
        fallback_reason = (
            ""
            if variant is not None
            else self._fallback_reason(decision.reason, decision.differences)
        )
        runtime_decisions = (
            dict(variant["runtime_decisions"])
            if variant is not None
            else {"profile": "baseline", "values": {}}
        )

        priority_config = vllm_config.kernel_config.ir_op_priority
        applied: list[str] = []
        for op_name, providers in self._priorities(runtime_decisions).items():
            if not hasattr(priority_config, op_name):
                raise EnginePlanError(f"Unknown vLLM IR operation {op_name!r}")
            current = list(getattr(priority_config, op_name))
            if current and current[: len(providers)] != providers:
                raise EnginePlanError(
                    f"Plan conflicts with explicit priority for {op_name}: "
                    f"current={current}, expected plan prefix={providers}"
                )
            if not current:
                setattr(priority_config, op_name, providers)
            applied.append(f"kernel_config.ir_op_priority.{op_name}")

        selected_tactics = _selected_tactics(variant) if variant is not None else ()
        context_fingerprint = (
            runtime_key_fingerprint(runtime_target, final=False)
            if runtime_target
            else ""
        )
        with self._plans_lock:
            self._plans[id(vllm_config)] = self._new_plan_state(
                vllm_config,
                plan=plan,
                variant=variant,
                runtime_target=runtime_target,
                fallback_reason=fallback_reason,
            )
        return EnginePlanApplication(
            plugin_name=self.metadata.name,
            plan_id=plan.plan_id,
            plan_fingerprint=plan.fingerprint,
            applied_settings=tuple(sorted(applied)),
            selected_variant=(variant["variant_id"] if variant is not None else ""),
            selected_tactics=selected_tactics,
            context_fingerprint=context_fingerprint,
            fallback_reason=fallback_reason,
        )

    def validate_runtime_config(
        self,
        vllm_config: Any,
        application: EnginePlanApplication,
    ) -> EnginePlanReceipt:
        with self._plans_lock:
            state = self._plans.get(id(vllm_config))
        if state is None or state.config() is not vllm_config:
            raise EnginePlanError("No plan state exists for this vLLM config")
        original = state.plan
        current_plan = self._load_selected_plan()
        if current_plan.fingerprint != original.fingerprint:
            raise EnginePlanError("Engine plan changed between defaults and validation")

        selected_variant = None
        if state.variant is None:
            final_decision = select_runtime_variant(
                current_plan,
                vllm_config,
                final=True,
            )
            final_runtime_target = final_decision.runtime_target
            if final_decision.variant is None:
                final_fallback_reason = self._fallback_reason(
                    final_decision.reason,
                    final_decision.differences,
                )
            else:
                final_fallback_reason = (
                    "baseline_fallback_preserved: final context matched "
                    f"unapplied variant {final_decision.variant['variant_id']!r}"
                )
            runtime_decisions = {"profile": "baseline", "values": {}}
        else:
            matching_variants = [
                dict(variant)
                for variant in current_plan.variants
                if variant["variant_id"] == state.variant["variant_id"]
            ]
            if len(matching_variants) != 1:
                raise EnginePlanError(
                    "Selected engine-plan variant disappeared before validation"
                )
            selected_variant = matching_variants[0]
            final_runtime_target, differences = validate_runtime_variant(
                selected_variant,
                vllm_config,
            )
            if differences:
                raise EnginePlanError(
                    "Final runtime context drifted after a plan variant was "
                    "applied: " + "; ".join(differences)
                )
            final_fallback_reason = ""
            runtime_decisions = dict(selected_variant["runtime_decisions"])
        final_context_fingerprint = runtime_key_fingerprint(
            final_runtime_target,
            final=True,
        )

        priority_config = vllm_config.kernel_config.ir_op_priority
        validated: list[str] = []
        for op_name, providers in self._priorities(runtime_decisions).items():
            current = list(getattr(priority_config, op_name))
            if current[: len(providers)] != providers:
                raise EnginePlanError(
                    f"Resolved priority drifted for {op_name}: "
                    f"expected prefix={providers}, got={current}"
                )
            validated.append(f"kernel_config.ir_op_priority.{op_name}")

        return EnginePlanReceipt(
            plugin_name=self.metadata.name,
            plugin_version=self.metadata.version,
            plan_id=application.plan_id,
            plan_fingerprint=application.plan_fingerprint,
            validated_settings=tuple(sorted(validated)),
            selected_variant=application.selected_variant,
            selected_tactics=application.selected_tactics,
            context_fingerprint=final_context_fingerprint,
            fallback_reason=final_fallback_reason,
        )

    def validate_resolved_cudagraph_config(
        self,
        vllm_config: Any,
        application: EnginePlanApplication,
    ) -> None:
        """Keep the selected plan bound to post-resolution graph settings."""

        with self._plans_lock:
            state = self._plans.get(id(vllm_config))
        if state is None or state.config() is not vllm_config:
            raise EnginePlanError("No plan state exists for this vLLM config")
        if state.variant is None:
            return
        timing_cache = state.variant.get("timing_cache", {})
        target = timing_cache.get("target", {})
        workload = target.get("workload", {})
        compilation_config = getattr(vllm_config, "compilation_config", None)
        mode = getattr(compilation_config, "cudagraph_mode", None)
        actual_mode = getattr(mode, "name", str(mode))
        expected_mode = workload.get("graph_mode")
        if expected_mode is not None and actual_mode != expected_mode:
            raise EnginePlanError(
                "Resolved CUDAGraph mode differs from the selected engine-plan "
                f"target: expected={expected_mode}, actual={actual_mode}"
            )
        expected_sizes = workload.get("cudagraph_capture_sizes")
        if expected_sizes is None:
            return
        actual_sizes = list(
            getattr(compilation_config, "cudagraph_capture_sizes", None) or ()
        )
        if actual_sizes != expected_sizes:
            raise EnginePlanError(
                "Resolved CUDAGraph capture ladder differs from the selected "
                f"engine-plan target: expected={expected_sizes}, "
                f"actual={actual_sizes}"
            )

    def resolve_contextual_runtime_decisions(
        self,
        application: EnginePlanApplication,
    ) -> EngineRuntimeDecisions | None:
        """Project the exact selected variant through the unified plan ABI."""

        plan = self._load_selected_plan()
        self._validate_compatibility(plan)
        if application.plan_id != plan.plan_id:
            raise EnginePlanError(
                "applied plan id does not match contextual runtime plan"
            )
        if application.plan_fingerprint != plan.fingerprint:
            raise EnginePlanError(
                "applied plan fingerprint does not match contextual runtime plan"
            )
        if not application.selected_variant:
            return None
        variants = [
            variant
            for variant in plan.variants
            if variant["variant_id"] == application.selected_variant
        ]
        if len(variants) != 1:
            raise EnginePlanError(
                "applied plan variant does not identify one runtime-decision set"
            )
        projection = variants[0]["runtime_decisions"]
        values = projection["values"]
        return EngineRuntimeDecisions(
            plan_id=plan.plan_id,
            plan_fingerprint=plan.fingerprint,
            profile=projection["profile"],
            decisions=tuple(
                (key, _immutable_decision_value(value))
                for key, value in sorted(values.items())
            ),
            profile_config_id=projection.get("profile_config_id"),
            profile_config_fingerprint=projection.get("profile_config_fingerprint"),
        )


_PLUGIN = JsonPlanRuntimePlugin()


def register() -> None:
    """Register the repository-local adapter in the current process."""

    register_engine_runtime_plugin(_PLUGIN)
