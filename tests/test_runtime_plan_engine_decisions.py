# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import pytest

from vllm_musa.engine_plugins import (
    EnginePlanApplication,
    EnginePlanReceipt,
    EnginePluginActivationError,
    EnginePluginMetadata,
    EngineRuntimeDecisions,
    apply_engine_plugin_defaults,
    register_engine_runtime_plugin,
    validate_engine_plugin_runtime,
)
from vllm_musa.engine_plugins.registry import _reset_engine_plugin_registry_for_test
from vllm_musa.runtime_plan import (
    RUNTIME_PLAN_TRANSPORT_KEY,
    RuntimeDecision,
    RuntimeDecisionError,
    publish_runtime_plan_transport,
    resolve_runtime_plan,
)
from vllm_musa.runtime_plan.declarative import declarative_profile_identity


@pytest.fixture(autouse=True)
def reset_registry(monkeypatch):
    _reset_engine_plugin_registry_for_test()
    monkeypatch.delenv("MUSA_ENGINE_PLAN", raising=False)
    yield
    _reset_engine_plugin_registry_for_test()


class WeakConfig:
    pass


def _qwen3_config() -> WeakConfig:
    text = SimpleNamespace(
        model_type="qwen3",
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=151936,
        quantization_config=None,
    )
    config = WeakConfig()
    config.model_config = SimpleNamespace(
        architectures=["Qwen3ForCausalLM"],
        hf_text_config=text,
        hf_config=text,
        dtype="bfloat16",
        quantization=None,
        enforce_eager=False,
    )
    config.parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=1,
    )
    config.cache_config = SimpleNamespace(cache_dtype="auto", block_size=64)
    config.scheduler_config = SimpleNamespace(max_num_seqs=64)
    config.speculative_config = None
    config.quant_config = None
    return config


def _deepseek_config() -> WeakConfig:
    text = SimpleNamespace(
        model_type="deepseek_v4",
        hidden_size=4096,
        num_hidden_layers=43,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        vocab_size=129280,
        n_routed_experts=256,
        num_experts_per_tok=6,
        n_shared_experts=1,
        moe_intermediate_size=2048,
        expert_dtype="fp8",
        hidden_act="silu",
        swiglu_limit=10.0,
        index_topk=512,
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    )
    config = WeakConfig()
    config.model_config = SimpleNamespace(
        architectures=["DeepseekV4ForCausalLM"],
        hf_text_config=text,
        hf_config=text,
        dtype="bfloat16",
        quantization="fp8",
        use_mla=True,
        is_hybrid=False,
        is_moe=True,
        enforce_eager=False,
    )
    config.parallel_config = SimpleNamespace(
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=1,
    )
    config.attention_config = SimpleNamespace(backend="FLASHMLA")
    config.compilation_config = SimpleNamespace(
        mode="NONE",
        cudagraph_mode="FULL_DECODE_ONLY",
    )
    config.cache_config = SimpleNamespace(cache_dtype="fp8", block_size=64)
    config.scheduler_config = SimpleNamespace(max_num_seqs=1)
    config.speculative_config = None
    config.quant_config = SimpleNamespace(weight_block_size=[128, 128])
    return config


class LegacyPlugin:
    metadata = EnginePluginMetadata(name="runtime-plan-test", version="1.0.0")

    def __init__(self) -> None:
        self.apply_calls = 0
        self.validate_calls = 0

    def is_enabled(self) -> bool:
        return True

    def apply_config_defaults(self, vllm_config) -> EnginePlanApplication:
        self.apply_calls += 1
        return EnginePlanApplication(
            plugin_name=self.metadata.name,
            plan_id="runtime-plan",
            plan_fingerprint="sha256:runtime-plan",
            applied_settings=(),
            selected_variant="variant-a",
        )

    def validate_runtime_config(
        self,
        vllm_config,
        application: EnginePlanApplication,
    ) -> EnginePlanReceipt:
        self.validate_calls += 1
        return EnginePlanReceipt(
            plugin_name=self.metadata.name,
            plugin_version=self.metadata.version,
            plan_id=application.plan_id,
            plan_fingerprint=application.plan_fingerprint,
            validated_settings=application.applied_settings,
            selected_variant=application.selected_variant,
            selected_tactics=application.selected_tactics,
        )


class DecisionPlugin(LegacyPlugin):
    def __init__(
        self,
        *,
        profile: str,
        decisions: tuple[tuple[str, object], ...] | None,
        profile_config: tuple[str, str] | None | bool = True,
    ) -> None:
        super().__init__()
        self.profile = profile
        self.decisions = decisions
        self.profile_config = profile_config
        self.decision_calls = 0

    def resolve_contextual_runtime_decisions(
        self,
        application: EnginePlanApplication,
    ) -> EngineRuntimeDecisions | None:
        self.decision_calls += 1
        if self.decisions is None:
            return None
        profile_config = (
            declarative_profile_identity(self.profile)
            if self.profile_config is True
            else self.profile_config
        )
        return EngineRuntimeDecisions(
            plan_id=application.plan_id,
            plan_fingerprint=application.plan_fingerprint,
            profile=self.profile,
            decisions=self.decisions,
            profile_config_id=(profile_config[0] if profile_config else None),
            profile_config_fingerprint=(profile_config[1] if profile_config else None),
        )


def _activate(plugin: LegacyPlugin, config: WeakConfig) -> None:
    register_engine_runtime_plugin(plugin)
    apply_engine_plugin_defaults(config)
    validate_engine_plugin_runtime(config)


def test_no_plugin_is_exact_builtin_noop() -> None:
    baseline = resolve_runtime_plan(_qwen3_config())
    config = _qwen3_config()
    assert apply_engine_plugin_defaults(config) is None

    assert resolve_runtime_plan(config) == baseline


def test_abi_v1_plugin_without_optional_decisions_is_noop() -> None:
    baseline = resolve_runtime_plan(_qwen3_config())
    config = _qwen3_config()
    plugin = LegacyPlugin()
    _activate(plugin, config)

    assert resolve_runtime_plan(config) == baseline
    assert plugin.apply_calls == 1
    assert plugin.validate_calls == 1


def test_noop_decision_projection_is_cached_by_plan_identity() -> None:
    config = _qwen3_config()
    plugin = DecisionPlugin(
        profile="qwen3.text_generation",
        decisions=None,
    )
    _activate(plugin, config)

    first = resolve_runtime_plan(config)
    second = resolve_runtime_plan(config)

    assert first == second
    assert plugin.decision_calls == 1


def test_plan_disables_supported_default_without_removing_capability() -> None:
    decision = RuntimeDecision.QWEN3_QK_ROPE_KV_PRESPLIT
    config = _qwen3_config()
    plugin = DecisionPlugin(
        profile="qwen3.text_generation",
        decisions=((decision.value, False),),
    )
    _activate(plugin, config)

    resolved = resolve_runtime_plan(config)

    assert resolved.supports(decision)
    assert not resolved.enabled(decision)
    assert resolved.decision_source(decision) == "engine_plan"
    assert resolved.decision_resolution is not None
    assert resolved.decision_resolution.source == "engine_plugin"
    assert resolved.decision_resolution.plugin_name == plugin.metadata.name
    assert resolved.decision_resolution.plan_id == "runtime-plan"
    assert dict(resolved.decision_resolution.decisions) == {decision.value: False}


def test_profile_decision_requires_exact_profile_config_identity() -> None:
    decision = RuntimeDecision.QWEN3_QK_ROPE_KV_PRESPLIT
    config = _qwen3_config()
    plugin = DecisionPlugin(
        profile="qwen3.text_generation",
        decisions=((decision.value, False),),
        profile_config=None,
    )
    _activate(plugin, config)

    with pytest.raises(RuntimeDecisionError, match="requires an exact"):
        resolve_runtime_plan(config)


def test_stale_profile_config_fingerprint_is_rejected() -> None:
    decision = RuntimeDecision.QWEN3_QK_ROPE_KV_PRESPLIT
    config = _qwen3_config()
    plugin = DecisionPlugin(
        profile="qwen3.text_generation",
        decisions=((decision.value, False),),
        profile_config=("qwen", "sha256:" + "0" * 64),
    )
    _activate(plugin, config)

    with pytest.raises(RuntimeDecisionError, match="does not match the live"):
        resolve_runtime_plan(config)


def test_plan_cannot_override_fixed_supported_decision() -> None:
    decision = RuntimeDecision.DEEPSEEK_V4_CAR_GRAPH_INPUT_CAPTURE_GUARD
    config = _deepseek_config()
    built_in = resolve_runtime_plan(_deepseek_config())
    assert built_in.supports(decision)
    assert not built_in.enabled(decision)
    plugin = DecisionPlugin(
        profile=built_in.profile,
        decisions=((decision.value, True),),
    )
    _activate(plugin, config)

    with pytest.raises(RuntimeDecisionError, match="cannot override fixed"):
        resolve_runtime_plan(config)


def test_plan_cannot_enable_unsupported_decision() -> None:
    decision = RuntimeDecision.DEEPSEEK_V4_CAR_GRAPH_INPUT_CAPTURE_GUARD
    config = _qwen3_config()
    plugin = DecisionPlugin(
        profile="qwen3.text_generation",
        decisions=((decision.value, True),),
    )
    _activate(plugin, config)

    with pytest.raises(RuntimeDecisionError, match="not supported"):
        resolve_runtime_plan(config)


def test_plan_can_pin_known_unsupported_decision_off() -> None:
    decision = RuntimeDecision.DEEPSEEK_V4_CAR_GRAPH_INPUT_CAPTURE_GUARD
    config = _qwen3_config()
    plugin = DecisionPlugin(
        profile="qwen3.text_generation",
        decisions=((decision.value, False),),
    )
    _activate(plugin, config)

    resolved = resolve_runtime_plan(config)

    assert not resolved.supports(decision)
    assert not resolved.enabled(decision)
    assert resolved.decision_resolution is not None
    assert dict(resolved.decision_resolution.decisions) == {decision.value: False}


def test_plan_can_project_registered_external_structured_decision() -> None:
    decision = RuntimeDecision.VLLM_IR_OP_PRIORITY
    value = (("fused_add_rms_norm", ("musa", "native")),)
    config = _qwen3_config()
    plugin = DecisionPlugin(
        profile="qwen3.text_generation",
        decisions=((decision.value, value),),
    )
    _activate(plugin, config)

    resolved = resolve_runtime_plan(config)

    assert resolved.supports(decision)
    assert resolved.value(decision) == value


def test_plan_rejects_invalid_fixed_dsv4_page_size() -> None:
    decision = RuntimeDecision.DEEPSEEK_V4_FLASHMLA_SPARSE_PAGE_SIZE
    config = _deepseek_config()
    built_in = resolve_runtime_plan(_deepseek_config())
    assert built_in.supports(decision)
    plugin = DecisionPlugin(
        profile=built_in.profile,
        decisions=((decision.value, 64),),
    )
    _activate(plugin, config)

    with pytest.raises(RuntimeDecisionError, match="must be one of.*256"):
        resolve_runtime_plan(config)


def test_plan_may_pin_unsupported_typed_choice_to_safe_fallback() -> None:
    decision = RuntimeDecision.DEEPSEEK_V4_FLASHMLA_SPARSE_PAGE_SIZE
    config = _qwen3_config()
    plugin = DecisionPlugin(
        profile="qwen3.text_generation",
        decisions=((decision.value, 256),),
    )
    _activate(plugin, config)

    resolved = resolve_runtime_plan(config)

    assert not resolved.supports(decision)
    assert resolved.value(decision, 64) == 64


def test_plan_rejects_wrong_registered_decision_type() -> None:
    decision = RuntimeDecision.QWEN_FA3_SCHEDULER
    config = _qwen3_config()
    plugin = DecisionPlugin(
        profile="qwen3.text_generation",
        decisions=((decision.value, "true"),),
    )
    _activate(plugin, config)

    with pytest.raises(RuntimeDecisionError, match="must be boolean"):
        resolve_runtime_plan(config)


def test_plan_cannot_name_unknown_decision() -> None:
    config = _qwen3_config()
    plugin = DecisionPlugin(
        profile="qwen3.text_generation",
        decisions=(("qwen.unknown_future_path", False),),
    )
    _activate(plugin, config)

    with pytest.raises(RuntimeDecisionError, match="unknown runtime decision"):
        resolve_runtime_plan(config)


def test_plan_profile_must_match_builtin_profile() -> None:
    config = _qwen3_config()
    plugin = DecisionPlugin(
        profile="deepseek_v4.tp8_flash_base",
        decisions=((RuntimeDecision.QWEN_FA3_SCHEDULER.value, False),),
    )
    _activate(plugin, config)

    with pytest.raises(RuntimeDecisionError, match="target profile"):
        resolve_runtime_plan(config)


def test_decision_projection_is_cached_by_applied_plan_identity() -> None:
    decision = RuntimeDecision.QWEN3_QK_ROPE_KV_PRESPLIT
    config = _qwen3_config()
    plugin = DecisionPlugin(
        profile="qwen3.text_generation",
        decisions=((decision.value, False),),
    )
    _activate(plugin, config)

    first = resolve_runtime_plan(config)
    second = resolve_runtime_plan(config)

    assert first == second
    assert plugin.decision_calls == 1


def test_runtime_decisions_require_applied_plan_provenance() -> None:
    config = _qwen3_config()
    plugin = DecisionPlugin(
        profile="qwen3.text_generation",
        decisions=((RuntimeDecision.QWEN_FA3_SCHEDULER.value, False),),
    )
    register_engine_runtime_plugin(plugin)
    application = apply_engine_plugin_defaults(config)
    assert application is not None

    plugin.apply_calls = 0
    resolved = resolve_runtime_plan(config)

    assert not resolved.enabled(RuntimeDecision.QWEN_FA3_SCHEDULER)
    assert plugin.apply_calls == 0


def test_runtime_decision_values_must_be_immutable_json_like() -> None:
    with pytest.raises(ValueError, match="immutable JSON-like"):
        EngineRuntimeDecisions(
            plan_id="runtime-plan",
            plan_fingerprint="sha256:runtime-plan",
            profile="qwen3.text_generation",
            decisions=((RuntimeDecision.QWEN_FA3_SCHEDULER.value, {"bad": True}),),
        )


def test_runtime_decision_keys_must_be_unique() -> None:
    key = RuntimeDecision.QWEN_FA3_SCHEDULER.value
    with pytest.raises(ValueError, match="duplicates"):
        EngineRuntimeDecisions(
            plan_id="runtime-plan",
            plan_fingerprint="sha256:runtime-plan",
            profile="qwen3.text_generation",
            decisions=((key, True), (key, False)),
        )


def test_plugin_plan_identity_mismatch_fails_closed() -> None:
    class BadPlugin(DecisionPlugin):
        def resolve_contextual_runtime_decisions(
            self,
            application: EnginePlanApplication,
        ) -> EngineRuntimeDecisions:
            return EngineRuntimeDecisions(
                plan_id="different-plan",
                plan_fingerprint=application.plan_fingerprint,
                profile=self.profile,
                decisions=self.decisions or (),
            )

    config = _qwen3_config()
    plugin = BadPlugin(
        profile="qwen3.text_generation",
        decisions=((RuntimeDecision.QWEN_FA3_SCHEDULER.value, False),),
    )
    _activate(plugin, config)

    with pytest.raises(EnginePluginActivationError, match="plan changed"):
        resolve_runtime_plan(config)


def test_runtime_plan_transport_roundtrip_is_registry_and_env_independent(
    monkeypatch,
) -> None:
    decision = RuntimeDecision.QWEN_FA3_SCHEDULER
    config = _qwen3_config()
    plugin = DecisionPlugin(
        profile="qwen3.text_generation",
        decisions=((decision.value, False),),
    )
    _activate(plugin, config)
    plan = resolve_runtime_plan(config)
    transport = publish_runtime_plan_transport(config, plan)
    assert transport is not None
    assert RUNTIME_PLAN_TRANSPORT_KEY in config.additional_config

    # Simulate the EngineCore -> worker pickle hop and a JSON-compatible hop
    # between two process boundaries.  Neither worker-local registry nor the
    # plan artifact environment is available on this side.
    worker_config = pickle.loads(pickle.dumps(config))
    worker_config.additional_config[RUNTIME_PLAN_TRANSPORT_KEY] = json.loads(
        json.dumps(transport)
    )
    _reset_engine_plugin_registry_for_test()
    monkeypatch.delenv("MUSA_ENGINE_PLAN", raising=False)

    worker_plan = resolve_runtime_plan(worker_config)

    assert worker_plan == plan
    assert worker_plan.decision_source(decision) == "engine_plan"


def test_runtime_plan_transport_rejects_registry_conflict(monkeypatch) -> None:
    decision = RuntimeDecision.QWEN_FA3_SCHEDULER
    config = _qwen3_config()
    plugin = DecisionPlugin(
        profile="qwen3.text_generation",
        decisions=((decision.value, False),),
    )
    _activate(plugin, config)
    plan = resolve_runtime_plan(config)
    publish_runtime_plan_transport(config, plan)

    # The transport is authoritative only when it agrees with a local
    # application.  A changed local projection must not be silently replaced.
    config.additional_config[RUNTIME_PLAN_TRANSPORT_KEY]["resolution"]["decisions"][0][
        1
    ] = True
    with pytest.raises(RuntimeDecisionError, match="transport"):
        resolve_runtime_plan(config)


def test_runtime_plan_transport_rejects_fingerprint_or_shape_drift(
    monkeypatch,
) -> None:
    decision = RuntimeDecision.QWEN_FA3_SCHEDULER
    config = _qwen3_config()
    plugin = DecisionPlugin(
        profile="qwen3.text_generation",
        decisions=((decision.value, False),),
    )
    _activate(plugin, config)
    plan = resolve_runtime_plan(config)
    publish_runtime_plan_transport(config, plan)

    config.model_config.hf_text_config.hidden_size += 1
    _reset_engine_plugin_registry_for_test()
    monkeypatch.delenv("MUSA_ENGINE_PLAN", raising=False)
    with pytest.raises(RuntimeDecisionError, match="does not match"):
        resolve_runtime_plan(config)


def test_runtime_plan_transport_allows_late_execution_defaults() -> None:
    decision = RuntimeDecision.QWEN_FA3_SCHEDULER
    config = _qwen3_config()
    plugin = DecisionPlugin(
        profile="qwen3.text_generation",
        decisions=((decision.value, False),),
    )
    _activate(plugin, config)
    early_plan = resolve_runtime_plan(config)
    early_transport = publish_runtime_plan_transport(config, early_plan)

    config.compilation_config = SimpleNamespace(mode=None, cudagraph_mode=None)
    config.compilation_config.mode = 3
    config.compilation_config.cudagraph_mode = "FULL"
    late_plan = resolve_runtime_plan(config)
    late_transport = publish_runtime_plan_transport(config, late_plan)

    assert early_plan.fingerprint != late_plan.fingerprint
    assert early_transport == late_transport
    assert config.additional_config[RUNTIME_PLAN_TRANSPORT_KEY] == late_transport


@pytest.mark.parametrize(
    "malformed",
    [
        None,
        {"schema_version": 1},
        {
            "schema_version": 99,
            "runtime_plan_fingerprint": "sha256:plan",
            "resolution": {},
        },
        {
            "schema_version": 1,
            "runtime_plan_fingerprint": "not-a-fingerprint",
            "resolution": {},
        },
    ],
)
def test_runtime_plan_transport_malformed_fails_closed(malformed) -> None:
    config = _qwen3_config()
    config.additional_config = {RUNTIME_PLAN_TRANSPORT_KEY: malformed}
    with pytest.raises(RuntimeDecisionError, match="transport"):
        resolve_runtime_plan(config)
