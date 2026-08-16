# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import weakref
from dataclasses import dataclass

import pytest

from vllm_musa.engine_plugins import (
    ENGINE_PLAN_ENV,
    EnginePlanApplication,
    EnginePlanReceipt,
    EnginePluginActivationError,
    EnginePluginMetadata,
    EnginePluginRegistrationError,
    EngineRuntimeDecisions,
    apply_engine_plugin_defaults,
    get_engine_plugin_application,
    register_engine_runtime_plugin,
    registered_engine_plugins,
)
from vllm_musa.engine_plugins import registry as engine_registry
from vllm_musa.engine_plugins import (
    resolve_engine_plugin_runtime_decisions,
    validate_engine_plugin_cudagraph_runtime,
    validate_engine_plugin_runtime,
)
from vllm_musa.engine_plugins.registry import _reset_engine_plugin_registry_for_test


@pytest.fixture(autouse=True)
def reset_registry(monkeypatch):
    _reset_engine_plugin_registry_for_test()
    monkeypatch.delenv(ENGINE_PLAN_ENV, raising=False)
    yield
    _reset_engine_plugin_registry_for_test()


@dataclass
class FakeConfig:
    value: int = 0


class NonWeakConfig:
    __slots__ = ("value",)

    def __init__(self):
        self.value = 0


class FakePlugin:
    def __init__(
        self,
        name: str = "fake",
        *,
        enabled: bool = True,
        abi_version: int = 1,
    ):
        self.metadata = EnginePluginMetadata(
            name=name,
            version="1.2.3",
            abi_version=abi_version,
        )
        self.enabled = enabled
        self.apply_calls = 0
        self.validate_calls = 0
        self.cudagraph_validate_calls = 0

    def is_enabled(self) -> bool:
        return self.enabled

    def apply_config_defaults(self, vllm_config):
        self.apply_calls += 1
        vllm_config.value = 7
        return EnginePlanApplication(
            plugin_name=self.metadata.name,
            plan_id="fake-plan",
            plan_fingerprint="sha256:deadbeef",
            applied_settings=("value",),
        )

    def validate_runtime_config(self, vllm_config, application):
        self.validate_calls += 1
        assert vllm_config.value == 7
        return EnginePlanReceipt(
            plugin_name=self.metadata.name,
            plugin_version=self.metadata.version,
            plan_id=application.plan_id,
            plan_fingerprint=application.plan_fingerprint,
            validated_settings=application.applied_settings,
        )

    def validate_resolved_cudagraph_config(self, vllm_config, application):
        self.cudagraph_validate_calls += 1
        assert vllm_config.value == 7
        assert application.plan_id == "fake-plan"


class LockObservingDecisionPlugin(FakePlugin):
    def __init__(self):
        super().__init__()
        self.enablement_lock_states = []
        self.decision_lock_states = []

    def is_enabled(self) -> bool:
        self.enablement_lock_states.append(engine_registry._lock._is_owned())
        return super().is_enabled()

    def resolve_contextual_runtime_decisions(self, application):
        self.decision_lock_states.append(engine_registry._lock._is_owned())
        return EngineRuntimeDecisions(
            plan_id=application.plan_id,
            plan_fingerprint=application.plan_fingerprint,
            profile="fake.profile",
            decisions=(("qwen.fa3_scheduler", False),),
        )


class FinalContextPlugin(FakePlugin):
    def apply_config_defaults(self, vllm_config):
        self.apply_calls += 1
        vllm_config.value = 7
        return EnginePlanApplication(
            plugin_name=self.metadata.name,
            plan_id="fake-plan",
            plan_fingerprint="sha256:deadbeef",
            applied_settings=("value",),
            context_fingerprint="sha256:early-context",
            fallback_reason="no_matching_variant: early context",
        )

    def validate_runtime_config(self, vllm_config, application):
        self.validate_calls += 1
        assert vllm_config.value == 7
        return EnginePlanReceipt(
            plugin_name=self.metadata.name,
            plugin_version=self.metadata.version,
            plan_id=application.plan_id,
            plan_fingerprint=application.plan_fingerprint,
            validated_settings=application.applied_settings,
            context_fingerprint="sha256:final-context",
            fallback_reason="no_matching_variant: final context",
        )


class MissingFinalContextPlugin(FinalContextPlugin):
    def validate_runtime_config(self, vllm_config, application):
        receipt = super().validate_runtime_config(vllm_config, application)
        return EnginePlanReceipt(
            plugin_name=receipt.plugin_name,
            plugin_version=receipt.plugin_version,
            plan_id=receipt.plan_id,
            plan_fingerprint=receipt.plan_fingerprint,
            validated_settings=receipt.validated_settings,
            fallback_reason=receipt.fallback_reason,
        )


def test_registration_is_idempotent_for_same_object():
    plugin = FakePlugin()
    register_engine_runtime_plugin(plugin)
    register_engine_runtime_plugin(plugin)

    assert registered_engine_plugins() == (plugin.metadata,)


def test_weak_config_state_is_released_after_config_collection():
    plugin = FakePlugin()
    register_engine_runtime_plugin(plugin)
    config = FakeConfig()
    identity = id(config)
    reference = weakref.ref(config)

    apply_engine_plugin_defaults(config)
    assert identity in engine_registry._config_states

    del config
    gc.collect()

    assert reference() is None
    assert identity not in engine_registry._config_states


def test_enabled_plugin_rejects_nonweak_config_instead_of_leaking_it():
    register_engine_runtime_plugin(FakePlugin())
    config = NonWeakConfig()

    with pytest.raises(EnginePluginActivationError, match="weak-referenceable"):
        apply_engine_plugin_defaults(config)

    assert id(config) not in engine_registry._config_states


def test_disabled_plugin_does_not_cache_nonweak_negative_lookup():
    register_engine_runtime_plugin(FakePlugin(enabled=False))
    config = NonWeakConfig()

    assert apply_engine_plugin_defaults(config) is None
    assert id(config) not in engine_registry._config_states


def test_duplicate_name_from_different_plugin_fails_closed():
    register_engine_runtime_plugin(FakePlugin())

    with pytest.raises(EnginePluginRegistrationError, match="already registered"):
        register_engine_runtime_plugin(FakePlugin())


def test_incompatible_abi_is_rejected():
    with pytest.raises(EnginePluginRegistrationError, match="requires ABI 1"):
        register_engine_runtime_plugin(FakePlugin(abi_version=2))


def test_disabled_registry_is_exact_noop():
    plugin = FakePlugin(enabled=False)
    register_engine_runtime_plugin(plugin)
    config = FakeConfig()

    assert apply_engine_plugin_defaults(config) is None
    assert validate_engine_plugin_runtime(config) is None
    assert config.value == 0
    assert plugin.apply_calls == 0
    assert plugin.validate_calls == 0


def test_requested_plan_without_plugin_fails_closed(monkeypatch):
    monkeypatch.setenv(ENGINE_PLAN_ENV, "/tmp/requested-plan.json")
    monkeypatch.setattr(
        "vllm_musa.engine_plan.plugin.register",
        lambda: None,
    )

    with pytest.raises(EnginePluginActivationError, match="no registered"):
        apply_engine_plugin_defaults(FakeConfig())


def test_plan_env_triggers_builtin_plan_discovery(monkeypatch):
    plugin = FakePlugin(name="builtin-test")
    discovery_calls = 0

    def discover():
        nonlocal discovery_calls
        discovery_calls += 1
        register_engine_runtime_plugin(plugin)

    monkeypatch.setenv(ENGINE_PLAN_ENV, "/tmp/external-plan.json")
    monkeypatch.setattr(
        "vllm_musa.engine_plan.plugin.register",
        discover,
    )
    config = FakeConfig()

    application = apply_engine_plugin_defaults(config)

    assert application is not None
    assert application.plugin_name == "builtin-test"
    assert discovery_calls == 1
    assert plugin.apply_calls == 1


def test_requested_plugin_recovers_from_early_negative_config_cache(monkeypatch):
    config = FakeConfig()
    assert apply_engine_plugin_defaults(config) is None
    plugin = FakePlugin(name="discovered-after-early-query")

    monkeypatch.setenv(ENGINE_PLAN_ENV, "/tmp/external-plan.json")
    monkeypatch.setattr(
        "vllm_musa.engine_plan.plugin.register",
        lambda: register_engine_runtime_plugin(plugin),
    )

    application = apply_engine_plugin_defaults(config)

    assert application is not None
    assert application.plugin_name == plugin.metadata.name
    assert plugin.apply_calls == 1


def test_two_phase_application_and_validation_are_idempotent():
    plugin = FakePlugin()
    register_engine_runtime_plugin(plugin)
    config = FakeConfig()

    application = apply_engine_plugin_defaults(config)
    assert application is apply_engine_plugin_defaults(config)
    assert application is get_engine_plugin_application(config)
    assert config.value == 7
    assert plugin.apply_calls == 1

    receipt = validate_engine_plugin_runtime(config)
    assert receipt is validate_engine_plugin_runtime(config)
    assert receipt.plan_id == "fake-plan"
    assert plugin.validate_calls == 1


def test_post_resolution_cudagraph_validation_uses_active_application():
    plugin = FakePlugin()
    register_engine_runtime_plugin(plugin)
    config = FakeConfig()

    apply_engine_plugin_defaults(config)
    validate_engine_plugin_runtime(config)
    validate_engine_plugin_cudagraph_runtime(config)

    assert plugin.cudagraph_validate_calls == 1


def test_required_post_resolution_validation_rejects_missing_worker_state():
    with pytest.raises(
        EnginePluginActivationError,
        match="no process-local engine plugin state",
    ):
        validate_engine_plugin_cudagraph_runtime(
            FakeConfig(),
            required=True,
        )


def test_enablement_probes_run_outside_registry_lock_on_cached_paths():
    plugin = LockObservingDecisionPlugin()
    register_engine_runtime_plugin(plugin)
    config = FakeConfig()

    application = apply_engine_plugin_defaults(config)
    assert apply_engine_plugin_defaults(config) is application
    receipt = validate_engine_plugin_runtime(config)
    assert validate_engine_plugin_runtime(config) is receipt
    decisions = resolve_engine_plugin_runtime_decisions(config)
    assert resolve_engine_plugin_runtime_decisions(config) is decisions

    assert plugin.enablement_lock_states == [False] * 6
    assert plugin.decision_lock_states == [False]


def test_final_receipt_may_recompute_context_and_fallback_detail():
    plugin = FinalContextPlugin()
    register_engine_runtime_plugin(plugin)
    config = FakeConfig()

    application = apply_engine_plugin_defaults(config)
    receipt = validate_engine_plugin_runtime(config)

    assert application.context_fingerprint == "sha256:early-context"
    assert receipt.context_fingerprint == "sha256:final-context"
    assert receipt.fallback_reason == "no_matching_variant: final context"


def test_final_receipt_cannot_omit_context_after_keyed_application():
    register_engine_runtime_plugin(MissingFinalContextPlugin())
    config = FakeConfig()

    apply_engine_plugin_defaults(config)

    with pytest.raises(EnginePluginActivationError, match="final context fingerprint"):
        validate_engine_plugin_runtime(config)


def test_multiple_enabled_plugins_fail_closed():
    register_engine_runtime_plugin(FakePlugin(name="first"))
    register_engine_runtime_plugin(FakePlugin(name="second"))

    with pytest.raises(EnginePluginActivationError, match="Only one"):
        apply_engine_plugin_defaults(FakeConfig())


def test_cached_application_revalidates_singleton_selection():
    first = FakePlugin(name="first")
    second = FakePlugin(name="second", enabled=False)
    register_engine_runtime_plugin(first)
    register_engine_runtime_plugin(second)
    config = FakeConfig()
    apply_engine_plugin_defaults(config)
    second.enabled = True

    with pytest.raises(EnginePluginActivationError, match="Only one"):
        apply_engine_plugin_defaults(config)


def test_cached_receipt_revalidates_singleton_selection():
    first = FakePlugin(name="first")
    second = FakePlugin(name="second", enabled=False)
    register_engine_runtime_plugin(first)
    register_engine_runtime_plugin(second)
    config = FakeConfig()
    apply_engine_plugin_defaults(config)
    validate_engine_plugin_runtime(config)
    second.enabled = True

    with pytest.raises(EnginePluginActivationError, match="Only one"):
        validate_engine_plugin_runtime(config)


def test_late_activation_is_rejected():
    plugin = FakePlugin()
    register_engine_runtime_plugin(plugin)

    with pytest.raises(EnginePluginActivationError, match="late partial"):
        validate_engine_plugin_runtime(FakeConfig())


def test_plan_request_cannot_appear_after_disabled_defaults(monkeypatch):
    config = FakeConfig()
    assert apply_engine_plugin_defaults(config) is None
    monkeypatch.setenv(ENGINE_PLAN_ENV, "/tmp/late-plan.json")

    with pytest.raises(EnginePluginActivationError, match="late partial"):
        validate_engine_plugin_runtime(config)
