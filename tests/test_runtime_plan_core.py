# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_musa.runtime_plan import (
    ExecutionSignature,
    ModelFamily,
    ModelRole,
    ModelSignature,
    RuntimeDecision,
    RuntimeDecisionKind,
    RuntimeDecisionResolution,
    RuntimePlan,
    RuntimePlanPhase,
    bind_runtime_plan,
    canonical_fingerprint,
    list_runtime_decision_specs,
    runtime_decision_spec,
)


def _model() -> ModelSignature:
    return ModelSignature(
        family=ModelFamily.QWEN3,
        role=ModelRole.TEXT,
        architectures=("Qwen3ForCausalLM",),
        model_type="qwen3",
        dtype="bfloat16",
        quantization=None,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=151936,
        num_experts=None,
        num_experts_per_tok=None,
        num_shared_experts=None,
        moe_intermediate_size=None,
        expert_dtype=None,
        hidden_act="silu",
        swiglu_limit=None,
        gdn_conv_width=None,
        gdn_conv_dim=None,
        has_routed_experts=False,
        enforce_eager=False,
        index_topk=None,
    )


def _execution() -> ExecutionSignature:
    return ExecutionSignature(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=1,
        has_speculative_config=False,
        has_quant_config=False,
        is_pooling_model=False,
        has_parallel_config=True,
        cache_dtype="auto",
        cache_block_size=64,
        max_num_seqs=64,
    )


def _plan(
    values: tuple[tuple[RuntimeDecision, object], ...] = (),
) -> RuntimePlan:
    supported = frozenset(key for key, _ in values)
    return RuntimePlan(
        model=_model(),
        execution=_execution(),
        profile="test.qwen3",
        supported_decisions=supported,
        decision_values=values,
        reason="unit test",
    )


def test_plan_normalizes_decision_order_and_exposes_read_only_mapping() -> None:
    page_size = RuntimeDecision.DEEPSEEK_V4_FLASHMLA_SPARSE_PAGE_SIZE
    sampling = RuntimeDecision.QWEN_V2_SAMPLING
    plan = _plan(((sampling, True), (page_size, 256)))

    assert tuple(key for key in plan.decisions) == tuple(
        sorted((sampling, page_size), key=lambda key: key.value)
    )
    assert plan.enabled(sampling)
    assert plan.value(page_size) == 256
    assert plan.selected(page_size, 256)
    with pytest.raises(TypeError, match="not boolean"):
        plan.enabled(page_size)
    with pytest.raises(TypeError):
        plan.decisions[sampling] = False  # type: ignore[index]


def test_plan_rejects_duplicate_unsupported_and_mutable_values() -> None:
    rope = RuntimeDecision.QWEN3_QK_ROPE_KV_PRESPLIT
    with pytest.raises(ValueError, match="Duplicate"):
        RuntimePlan(
            model=_model(),
            execution=_execution(),
            profile="test",
            supported_decisions=frozenset({rope}),
            decision_values=((rope, True), (rope, False)),
        )
    with pytest.raises(ValueError, match="not supported"):
        RuntimePlan(
            model=_model(),
            execution=_execution(),
            profile="test",
            supported_decisions=frozenset(),
            decision_values=((RuntimeDecision.QWEN_V2_SAMPLING, True),),
        )
    with pytest.raises(TypeError, match="immutable"):
        _plan(((rope, ["mutable"]),))


def test_catalog_rejects_wrong_types_and_invalid_structured_choices() -> None:
    rope = RuntimeDecision.QWEN3_QK_ROPE_KV_PRESPLIT
    with pytest.raises(ValueError, match="must be boolean"):
        _plan(((rope, "true"),))

    priority = RuntimeDecision.VLLM_IR_OP_PRIORITY
    value = (("fused_add_rms_norm", ("musa", "native")),)
    plan = _plan(((priority, value),))
    assert plan.value(priority) == value
    with pytest.raises(ValueError, match="native fallback"):
        _plan(((priority, (("fused_add_rms_norm", ("musa",)),)),))


def _fused_moe_policy_value(
    *,
    graph_mode: str = "eager",
    ranges: tuple[tuple[tuple[str, object], ...], ...] | None = None,
) -> tuple[tuple[str, object], ...]:
    shape = {
        "activation": "silu",
        "block_k": 128,
        "block_n": 128,
        "device_capability": (3, 1),
        "expert_parallel": False,
        "gemv_block": "auto",
        "graph_mode": graph_mode,
        "hidden_dtype": "torch.bfloat16",
        "hidden_size": 4096,
        "local_experts": 128,
        "multiprocessor_count": 60,
        "scale_dtype": "torch.float32",
        "top_k": 8,
        "w1_output_size": 512,
        "w1_scale_shape": (128, 4, 32),
        "w2_input_size": 256,
        "w2_scale_shape": (128, 32, 2),
        "weight_dtype": "torch.float8_e4m3fn",
    }
    if ranges is None:
        ranges = (
            (
                ("backend", "gemv"),
                ("max_tokens", 4),
                ("min_tokens", 1),
            ),
            (
                ("backend", "upstream"),
                ("max_tokens", 8),
                ("min_tokens", 5),
            ),
        )
    entry = (
        ("ranges", ranges),
        ("shape", tuple(sorted(shape.items()))),
    )
    return (
        ("entries", (entry,)),
        ("schema", "musa.fused_moe.dispatch_policy.v1"),
    )


def test_catalog_validates_fused_moe_dispatch_policy_codec() -> None:
    decision = RuntimeDecision.MUSA_FUSED_MOE_DISPATCH_POLICY
    value = _fused_moe_policy_value()

    plan = _plan(((decision, value),))

    assert plan.value(decision) == value
    spec = runtime_decision_spec(decision)
    assert spec.kind is RuntimeDecisionKind.STRUCTURED
    assert spec.phase is RuntimePlanPhase.COMPILE
    assert spec.fallback == ()
    assert spec.external_only


@pytest.mark.parametrize(
    ("value", "match"),
    (
        (
            _fused_moe_policy_value(
                ranges=(
                    (
                        ("backend", "gemv"),
                        ("max_tokens", 4),
                        ("min_tokens", 2),
                    ),
                )
            ),
            "start at one",
        ),
        (
            _fused_moe_policy_value(
                ranges=(
                    (
                        ("backend", "gemv"),
                        ("max_tokens", 4),
                        ("min_tokens", 1),
                    ),
                    (
                        ("backend", "upstream"),
                        ("max_tokens", 8),
                        ("min_tokens", 6),
                    ),
                )
            ),
            "continuous",
        ),
        (
            _fused_moe_policy_value(
                ranges=(
                    (
                        ("backend", "auto"),
                        ("max_tokens", 4),
                        ("min_tokens", 1),
                    ),
                )
            ),
            "backend",
        ),
        (
            _fused_moe_policy_value(
                graph_mode="capture",
                ranges=(
                    (
                        ("backend", "grouped_gemm"),
                        ("max_tokens", 4),
                        ("min_tokens", 1),
                    ),
                ),
            ),
            "capture",
        ),
    ),
)
def test_catalog_rejects_malformed_fused_moe_dispatch_policy(
    value: tuple[tuple[str, object], ...],
    match: str,
) -> None:
    decision = RuntimeDecision.MUSA_FUSED_MOE_DISPATCH_POLICY

    with pytest.raises(ValueError, match=match):
        _plan(((decision, value),))


def test_catalog_covers_every_key_and_declares_materialization_phase() -> None:
    specs = list_runtime_decision_specs()
    assert {spec.key for spec in specs} == set(RuntimeDecision)
    page_spec = runtime_decision_spec(
        RuntimeDecision.DEEPSEEK_V4_FLASHMLA_SPARSE_PAGE_SIZE
    )
    assert page_spec.kind is RuntimeDecisionKind.INTEGER
    assert page_spec.phase is RuntimePlanPhase.CACHE_LAYOUT
    assert page_spec.fallback == 256
    assert page_spec.choices == (256,)


def test_resolution_and_fingerprint_are_deterministic() -> None:
    resolution = RuntimeDecisionResolution(
        source="engine_plugin",
        plugin_name="example",
        plugin_version="1",
        plan_id="plan",
        fingerprint="sha256:plan",
        profile="test.qwen3",
        decisions=(
            (RuntimeDecision.QWEN_V2_SAMPLING.value, True),
            (RuntimeDecision.QWEN3_QK_ROPE_KV_PRESPLIT.value, False),
        ),
    )
    assert resolution.plan_fingerprint == resolution.fingerprint
    assert resolution.enabled_decisions == frozenset({RuntimeDecision.QWEN_V2_SAMPLING})
    assert resolution.disabled_decisions == frozenset(
        {RuntimeDecision.QWEN3_QK_ROPE_KV_PRESPLIT}
    )
    assert canonical_fingerprint({"b": 2, "a": 1}) == canonical_fingerprint(
        {"a": 1, "b": 2}
    )


def test_bind_is_idempotent_and_keeps_the_first_snapshot() -> None:
    owner = SimpleNamespace()
    config = SimpleNamespace()
    first = bind_runtime_plan(owner, config)
    second = bind_runtime_plan(owner, config)
    assert first is second
    assert owner._musa_runtime_plan is first


def test_unknown_plan_is_safe_and_fingerprintable() -> None:
    plan = RuntimePlan.unknown()
    assert plan.profile == "unknown"
    assert not plan.supported_decisions
    assert not plan.enabled_decisions
    assert plan.fingerprint.startswith("sha256:")
