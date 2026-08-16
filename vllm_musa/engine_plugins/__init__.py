# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stable in-tree host ABI for vLLM-MUSA engine-plan capabilities."""

from .api import (
    ENGINE_PLAN_ENV,
    ENGINE_PLAN_FINGERPRINT_ENV,
    ENGINE_PLAN_REQUIRE_PINNED_FINGERPRINT_ENV,
    ENGINE_PLAN_VARIANT_ENV,
    ENGINE_PLUGIN_ABI_VERSION,
    ContextualEngineRuntimePlanProvider,
    EngineIrProviderMetadata,
    EnginePlanApplication,
    EnginePlanBuilder,
    EnginePlanReceipt,
    EnginePluginCore,
    EnginePluginMetadata,
    EngineRuntimeDecisionReceipt,
    EngineRuntimeDecisions,
    EngineRuntimePlugin,
)
from .ir_catalog import find_engine_ir_provider, list_engine_ir_providers
from .registry import (
    EnginePluginActivationError,
    EnginePluginError,
    EnginePluginRegistrationError,
    apply_engine_plugin_defaults,
    get_engine_plugin_application,
    register_engine_runtime_plugin,
    registered_engine_plugins,
    resolve_engine_plugin_runtime_decisions,
    validate_engine_plugin_cudagraph_runtime,
    validate_engine_plugin_runtime,
)

__all__ = [
    "ContextualEngineRuntimePlanProvider",
    "ENGINE_PLUGIN_ABI_VERSION",
    "ENGINE_PLAN_ENV",
    "ENGINE_PLAN_FINGERPRINT_ENV",
    "ENGINE_PLAN_REQUIRE_PINNED_FINGERPRINT_ENV",
    "ENGINE_PLAN_VARIANT_ENV",
    "EngineIrProviderMetadata",
    "EnginePlanApplication",
    "EnginePlanBuilder",
    "EnginePlanReceipt",
    "EnginePluginActivationError",
    "EnginePluginError",
    "EnginePluginCore",
    "EnginePluginMetadata",
    "EnginePluginRegistrationError",
    "EngineRuntimeDecisionReceipt",
    "EngineRuntimeDecisions",
    "EngineRuntimePlugin",
    "apply_engine_plugin_defaults",
    "get_engine_plugin_application",
    "find_engine_ir_provider",
    "list_engine_ir_providers",
    "register_engine_runtime_plugin",
    "registered_engine_plugins",
    "resolve_engine_plugin_runtime_decisions",
    "validate_engine_plugin_cudagraph_runtime",
    "validate_engine_plugin_runtime",
]
