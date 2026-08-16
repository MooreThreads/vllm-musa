# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Process-local registry for the built-in vLLM-MUSA engine-plan adapter."""

from __future__ import annotations

import logging
import os
import weakref
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from .api import (
    ENGINE_PLAN_ENV,
    ENGINE_PLAN_FINGERPRINT_ENV,
    ENGINE_PLAN_VARIANT_ENV,
    ENGINE_PLUGIN_ABI_VERSION,
    EnginePlanApplication,
    EnginePlanReceipt,
    EnginePluginMetadata,
    EngineRuntimeDecisionReceipt,
    EngineRuntimeDecisions,
    EngineRuntimePlugin,
)


class EnginePluginError(RuntimeError):
    """Base error for engine-plugin registration and activation failures."""


class EnginePluginRegistrationError(EnginePluginError):
    """Raised when the engine-plan adapter violates its host ABI."""


class EnginePluginActivationError(EnginePluginError):
    """Raised when zero/one-plugin activation cannot be resolved safely."""


@dataclass
class _ConfigState:
    config_ref: weakref.ReferenceType[Any]
    plugin: EngineRuntimePlugin | None
    application: EnginePlanApplication | None
    receipt: EnginePlanReceipt | None = None
    runtime_decision_receipts: dict[str, EngineRuntimeDecisionReceipt | None] = field(
        default_factory=dict
    )

    def config(self) -> Any | None:
        return self.config_ref()


_plugins: dict[str, EngineRuntimePlugin] = {}
_config_states: dict[int, _ConfigState] = {}
_lock = RLock()
logger = logging.getLogger(__name__)


def _discard_config_state(
    config_id: int,
    reference: weakref.ReferenceType[Any],
) -> None:
    with _lock:
        state = _config_states.get(config_id)
        if state is not None and state.config_ref is reference:
            _config_states.pop(config_id, None)


def _new_config_state(
    vllm_config: Any,
    *,
    plugin: EngineRuntimePlugin | None,
    application: EnginePlanApplication | None,
) -> _ConfigState | None:
    config_id = id(vllm_config)
    try:
        reference = weakref.ref(
            vllm_config,
            lambda released, identity=config_id: _discard_config_state(
                identity,
                released,
            ),
        )
    except TypeError:
        if plugin is None and application is None:
            # A negative lookup is only a cache optimization. Do not retain an
            # unbounded strong reference merely to remember that no plugin was
            # enabled for a non-weakrefable test or compatibility object.
            return None
        raise EnginePluginActivationError(
            "Engine-plugin config state requires a weak-referenceable "
            "vLLM configuration"
        ) from None
    return _ConfigState(
        config_ref=reference,
        plugin=plugin,
        application=application,
    )


def _require_weakrefable_config(vllm_config: Any) -> None:
    try:
        weakref.ref(vllm_config)
    except TypeError:
        raise EnginePluginActivationError(
            "Engine-plugin config state requires a weak-referenceable "
            "vLLM configuration"
        ) from None


def _validate_metadata(metadata: EnginePluginMetadata) -> None:
    if metadata.abi_version != ENGINE_PLUGIN_ABI_VERSION:
        raise EnginePluginRegistrationError(
            f"Engine plugin {metadata.name!r} uses ABI {metadata.abi_version}; "
            f"vLLM-MUSA requires ABI {ENGINE_PLUGIN_ABI_VERSION}"
        )


def register_engine_runtime_plugin(plugin: EngineRuntimePlugin) -> None:
    """Register one runtime capability in the current process.

    Re-registering the exact same object is intentionally idempotent because
    frontend and worker processes may initialize the built-in adapter
    independently. A different object claiming an existing name fails closed.
    """

    metadata = getattr(plugin, "metadata", None)
    if not isinstance(metadata, EnginePluginMetadata):
        raise EnginePluginRegistrationError(
            "Engine runtime plugins must expose EnginePluginMetadata"
        )
    _validate_metadata(metadata)
    for method_name in (
        "is_enabled",
        "apply_config_defaults",
        "validate_runtime_config",
    ):
        if not callable(getattr(plugin, method_name, None)):
            raise EnginePluginRegistrationError(
                f"Engine plugin {metadata.name!r} lacks {method_name}()"
            )

    with _lock:
        existing = _plugins.get(metadata.name)
        if existing is plugin:
            return
        if existing is not None:
            raise EnginePluginRegistrationError(
                f"Engine plugin name {metadata.name!r} is already registered"
            )
        _plugins[metadata.name] = plugin


def registered_engine_plugins() -> tuple[EnginePluginMetadata, ...]:
    """Return deterministic read-only metadata for diagnostics."""

    with _lock:
        return tuple(_plugins[name].metadata for name in sorted(_plugins))


def _select_enabled_plugin() -> EngineRuntimePlugin | None:
    with _lock:
        plugins = tuple(_plugins[name] for name in sorted(_plugins))

    enabled: list[EngineRuntimePlugin] = []
    for plugin in plugins:
        try:
            if plugin.is_enabled():
                enabled.append(plugin)
        except Exception as exc:
            raise EnginePluginActivationError(
                f"Failed to probe engine plugin {plugin.metadata.name!r}"
            ) from exc

    if len(enabled) > 1:
        names = ", ".join(plugin.metadata.name for plugin in enabled)
        raise EnginePluginActivationError(
            f"Only one vLLM-MUSA engine plugin may be enabled; got: {names}"
        )
    if not enabled and os.environ.get(ENGINE_PLAN_ENV):
        raise EnginePluginActivationError(
            f"{ENGINE_PLAN_ENV} requests an engine plan, but no registered "
            "vLLM-MUSA engine plugin accepted it"
        )
    return enabled[0] if enabled else None


def _discover_engine_plugins_for_plan() -> None:
    if not os.environ.get(ENGINE_PLAN_ENV):
        return
    with _lock:
        if _plugins:
            return
    try:
        # The plan builder is part of the vLLM-MUSA package itself.  Keep the
        # env-gated discovery here so importing vLLM-MUSA does not eagerly load
        # the artifact parser, but do not require a separately installed wheel
        # or a vLLM general-plugin entry point.
        from vllm_musa.engine_plan.plugin import register

        register()
    except Exception as exc:
        raise EnginePluginActivationError(
            "Failed to discover the built-in vLLM-MUSA engine plan for the requested "
            f"{ENGINE_PLAN_ENV}: {exc}"
        ) from exc


def _get_config_state(vllm_config: Any) -> _ConfigState | None:
    state = _config_states.get(id(vllm_config))
    if state is not None and state.config() is not vllm_config:
        # Guard against CPython object-id reuse in a long-lived frontend.
        _config_states.pop(id(vllm_config), None)
        return None
    return state


def _assert_config_state_selection(
    state: _ConfigState,
    selected: EngineRuntimePlugin | None,
) -> None:
    if selected is not state.plugin:
        previous = state.plugin.metadata.name if state.plugin is not None else "none"
        current = selected.metadata.name if selected is not None else "none"
        raise EnginePluginActivationError(
            "Engine-plugin selection changed after this vLLM configuration "
            f"was initialized: previous={previous!r}, current={current!r}; "
            "create a new configuration instead of reusing cached state"
        )


def _validate_application(
    plugin: EngineRuntimePlugin,
    application: EnginePlanApplication,
) -> None:
    if not isinstance(application, EnginePlanApplication):
        raise EnginePluginActivationError(
            f"Engine plugin {plugin.metadata.name!r} returned an invalid application"
        )
    if application.plugin_name != plugin.metadata.name:
        raise EnginePluginActivationError(
            f"Engine plugin {plugin.metadata.name!r} returned application for "
            f"{application.plugin_name!r}"
        )
    if not application.plan_id or not application.plan_fingerprint:
        raise EnginePluginActivationError(
            f"Engine plugin {plugin.metadata.name!r} returned incomplete provenance"
        )


def apply_engine_plugin_defaults(
    vllm_config: Any,
) -> EnginePlanApplication | None:
    """Apply the single enabled plugin before vLLM freezes IR defaults."""

    # A config can be queried before the built-in plan adapter is loaded.
    # Discover a requested plan before honoring a cached negative selection so an early
    # plugin=None snapshot cannot permanently suppress later activation.
    _discover_engine_plugins_for_plan()
    selected_plugin = _select_enabled_plugin()
    with _lock:
        state = _get_config_state(vllm_config)
        if state is not None:
            if state.plugin is None and selected_plugin is not None:
                _config_states.pop(id(vllm_config), None)
                state = None
            else:
                _assert_config_state_selection(state, selected_plugin)
                if state.plugin is None or state.application is not None:
                    return state.application

    if state is None:
        plugin = selected_plugin
    else:
        assert state.plugin is not None
        plugin = state.plugin
    application = None
    if plugin is not None:
        _require_weakrefable_config(vllm_config)
        try:
            application = plugin.apply_config_defaults(vllm_config)
        except Exception as exc:
            raise EnginePluginActivationError(
                f"Failed to apply engine plugin {plugin.metadata.name!r} "
                f"defaults: {exc}"
            ) from exc
        _validate_application(plugin, application)

    with _lock:
        existing = _get_config_state(vllm_config)
        if existing is not None:
            if existing.plugin is not plugin:
                raise EnginePluginActivationError(
                    "Engine plugin selection changed before config defaults"
                )
            if application is not None:
                for decision_receipt in existing.runtime_decision_receipts.values():
                    if decision_receipt is None:
                        continue
                    if (
                        decision_receipt.plan_id != application.plan_id
                        or decision_receipt.plan_fingerprint
                        != application.plan_fingerprint
                    ):
                        raise EnginePluginActivationError(
                            f"Engine plugin {plugin.metadata.name!r} plan changed "
                            "between runtime-plan and config-default phases"
                        )
            existing.application = application
        else:
            new_state = _new_config_state(
                vllm_config,
                plugin=plugin,
                application=application,
            )
            if new_state is not None:
                _config_states[id(vllm_config)] = new_state
    return application


def get_engine_plugin_application(
    vllm_config: Any,
) -> EnginePlanApplication | None:
    """Return the immutable application selected for one config, if any."""

    with _lock:
        state = _get_config_state(vllm_config)
        return None if state is None else state.application


def _validate_receipt(
    plugin: EngineRuntimePlugin,
    application: EnginePlanApplication,
    receipt: EnginePlanReceipt,
) -> None:
    if not isinstance(receipt, EnginePlanReceipt):
        raise EnginePluginActivationError(
            f"Engine plugin {plugin.metadata.name!r} returned an invalid receipt"
        )
    expected = (
        plugin.metadata.name,
        plugin.metadata.version,
        application.plan_id,
        application.plan_fingerprint,
    )
    actual = (
        receipt.plugin_name,
        receipt.plugin_version,
        receipt.plan_id,
        receipt.plan_fingerprint,
    )
    if actual != expected:
        raise EnginePluginActivationError(
            f"Engine plugin {plugin.metadata.name!r} returned mismatched provenance"
        )
    if receipt.validated_settings != application.applied_settings:
        raise EnginePluginActivationError(
            f"Engine plugin {plugin.metadata.name!r} validated a different "
            "setting set than it applied"
        )
    expected_selection = (
        application.selected_variant,
        application.selected_tactics,
    )
    actual_selection = (
        receipt.selected_variant,
        receipt.selected_tactics,
    )
    if actual_selection != expected_selection:
        raise EnginePluginActivationError(
            f"Engine plugin {plugin.metadata.name!r} validated a different "
            "variant decision than it applied"
        )
    if application.context_fingerprint and not receipt.context_fingerprint:
        raise EnginePluginActivationError(
            f"Engine plugin {plugin.metadata.name!r} omitted its final "
            "context fingerprint"
        )
    if bool(receipt.fallback_reason) != bool(application.fallback_reason):
        raise EnginePluginActivationError(
            f"Engine plugin {plugin.metadata.name!r} changed its fallback state "
            "after config defaults"
        )


def validate_engine_plugin_runtime(
    vllm_config: Any,
) -> EnginePlanReceipt | None:
    """Validate the final resolved config and emit a deterministic receipt."""

    with _lock:
        state = _get_config_state(vllm_config)
        state_has_no_plugin = state is not None and state.plugin is None

    # Preserve the late-plan diagnostic for a config whose default phase
    # already recorded that no plugin was active. No plugin callback is needed
    # for that immutable state transition.
    if state_has_no_plugin:
        if os.environ.get(ENGINE_PLAN_ENV):
            raise EnginePluginActivationError(
                f"{ENGINE_PLAN_ENV} was enabled after the platform-default "
                "phase; refusing a late partial application"
            )
        return None

    selected_plugin = _select_enabled_plugin()
    with _lock:
        state = _get_config_state(vllm_config)
        if state is not None:
            _assert_config_state_selection(state, selected_plugin)
            if state.receipt is not None:
                return state.receipt

    if state is None:
        if selected_plugin is None:
            return None
        raise EnginePluginActivationError(
            f"Engine plugin {selected_plugin.metadata.name!r} was enabled after the "
            "platform-default phase; refusing a late partial application"
        )
    if state.plugin is None:
        if os.environ.get(ENGINE_PLAN_ENV):
            raise EnginePluginActivationError(
                f"{ENGINE_PLAN_ENV} was enabled after the platform-default "
                "phase; refusing a late partial application"
            )
        return None
    if state.application is None:
        raise EnginePluginActivationError(
            f"Engine plugin {state.plugin.metadata.name!r} runtime decisions "
            "were resolved, but config defaults were never applied"
        )

    try:
        receipt = state.plugin.validate_runtime_config(
            vllm_config,
            state.application,
        )
    except Exception as exc:
        raise EnginePluginActivationError(
            f"Failed to validate engine plugin "
            f"{state.plugin.metadata.name!r} plan: {exc}"
        ) from exc
    _validate_receipt(state.plugin, state.application, receipt)

    with _lock:
        state.receipt = receipt
    return receipt


def validate_engine_plugin_cudagraph_runtime(
    vllm_config: Any,
    *,
    required: bool = False,
    serialized_transport: bool = False,
) -> None:
    """Validate optional post-resolution graph invariants for the active plugin.

    A spawned worker may have no plugin object because vLLM intentionally
    serializes ``VllmConfig`` rather than Python plugin registries.  In that
    case the RuntimePlan transport has already been parsed and applied by the
    MUSA worker seam; treating that authenticated projection as the authority
    is distinct from silently accepting an unplanned missing plugin.
    """

    with _lock:
        state = _get_config_state(vllm_config)
    if state is None or state.plugin is None:
        if serialized_transport:
            return
        if required:
            raise EnginePluginActivationError(
                "Active RuntimePlan has no process-local engine plugin state "
                "after worker rehydration"
            )
        return
    if state.application is None:
        raise EnginePluginActivationError(
            "Engine plugin graph validation ran before config defaults"
        )
    validator = getattr(state.plugin, "validate_resolved_cudagraph_config", None)
    if not callable(validator):
        return
    try:
        validator(vllm_config, state.application)
    except Exception as exc:
        raise EnginePluginActivationError(
            f"Failed to validate engine plugin {state.plugin.metadata.name!r} "
            f"resolved CUDAGraph config: {exc}"
        ) from exc


def _validate_runtime_decisions(
    plugin: EngineRuntimePlugin,
    decisions: EngineRuntimeDecisions,
    application: EnginePlanApplication,
) -> None:
    if not isinstance(decisions, EngineRuntimeDecisions):
        raise EnginePluginActivationError(
            f"Engine plugin {plugin.metadata.name!r} returned invalid "
            "runtime decisions"
        )
    if (
        decisions.plan_id != application.plan_id
        or decisions.plan_fingerprint != application.plan_fingerprint
    ):
        raise EnginePluginActivationError(
            f"Engine plugin {plugin.metadata.name!r} plan changed before "
            "runtime-decision resolution"
        )


def resolve_engine_plugin_runtime_decisions(
    vllm_config: Any,
) -> EngineRuntimeDecisionReceipt | None:
    """Project typed decisions from the exact applied plan variant."""

    # Re-check the singleton selection at this phase as well.  A plugin may
    # have been disabled or a second plugin registered after validation; using
    # a stale projection would violate the application's provenance fence.
    selected_plugin = _select_enabled_plugin()
    with _lock:
        state = _get_config_state(vllm_config)
        if state is not None:
            _assert_config_state_selection(state, selected_plugin)
    if state is None:
        if os.environ.get(ENGINE_PLAN_ENV):
            raise EnginePluginActivationError(
                f"{ENGINE_PLAN_ENV} runtime decisions were requested before "
                "engine-plugin config defaults were applied"
            )
        return None
    if state.plugin is None or state.application is None:
        return None

    cache_key = (
        f"{state.application.plan_id}:"
        f"{state.application.plan_fingerprint}:"
        f"{state.application.selected_variant}"
    )
    with _lock:
        if cache_key in state.runtime_decision_receipts:
            return state.runtime_decision_receipts[cache_key]

    resolver = getattr(state.plugin, "resolve_contextual_runtime_decisions", None)
    if not callable(resolver):
        with _lock:
            existing = _get_config_state(vllm_config)
            if existing is not None:
                existing.runtime_decision_receipts[cache_key] = None
        return None
    try:
        decisions = resolver(state.application)
    except Exception as exc:
        raise EnginePluginActivationError(
            f"Failed to resolve engine plugin "
            f"{state.plugin.metadata.name!r} runtime decisions: {exc}"
        ) from exc
    if decisions is None:
        with _lock:
            existing = _get_config_state(vllm_config)
            if existing is None:
                raise EnginePluginActivationError(
                    "Engine plugin config state disappeared during "
                    "runtime-plan resolution"
                )
            existing.runtime_decision_receipts[cache_key] = None
        return None
    _validate_runtime_decisions(state.plugin, decisions, state.application)
    receipt = EngineRuntimeDecisionReceipt(
        plugin_name=state.plugin.metadata.name,
        plugin_version=state.plugin.metadata.version,
        plan_id=decisions.plan_id,
        plan_fingerprint=decisions.plan_fingerprint,
        profile=decisions.profile,
        decisions=decisions.decisions,
        profile_config_id=decisions.profile_config_id,
        profile_config_fingerprint=decisions.profile_config_fingerprint,
    )

    with _lock:
        existing = _get_config_state(vllm_config)
        if existing is None:
            raise EnginePluginActivationError(
                "Engine plugin config state disappeared during runtime-plan "
                "resolution"
            )
        if cache_key in existing.runtime_decision_receipts:
            return existing.runtime_decision_receipts[cache_key]
        existing.runtime_decision_receipts[cache_key] = receipt
    logger.info(
        "MUSA engine plugin %s resolved plan %s runtime decisions for "
        "profile %s: %s",
        receipt.plugin_name,
        receipt.plan_id,
        receipt.profile,
        receipt.decisions,
    )
    return receipt


def _reset_engine_plugin_registry_for_test() -> None:
    """Reset process-local state. Intended only for focused unit tests."""

    with _lock:
        _plugins.clear()
        _config_states.clear()
    os.environ.pop(ENGINE_PLAN_FINGERPRINT_ENV, None)
    os.environ.pop(ENGINE_PLAN_VARIANT_ENV, None)
