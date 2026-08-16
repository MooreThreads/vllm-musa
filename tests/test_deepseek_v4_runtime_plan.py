from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_musa.runtime_plan import (
    ModelFamily,
    ModelRole,
    RuntimeDecision,
    bind_runtime_plan,
    resolve_runtime_plan,
)
from vllm_musa.runtime_plan.policy import (
    deepseek_v4_mtp_async_prefill_queue_fence_enabled,
    deepseek_v4_mtp_car_graph_guard_enabled,
    deepseek_v4_mtp_car_graph_staging_plan,
    deepseek_v4_mtp_graph_registered_inputs_enabled,
    deepseek_v4_mtp_prefill_step_requires_sync,
    deepseek_v4_mtp_sparse_prefill_headroom_bytes,
)


def _flash_base_config(
    *,
    tp: int = 8,
    speculative: bool = False,
    speculative_method: str = "mtp",
    max_num_seqs: int = 1,
    max_num_batched_tokens: int = 8195,
    mtp_draft: bool = False,
    cache_dtype: str = "fp8",
    async_scheduling: bool = True,
):
    architectures = ["DeepSeekV4MTPModel" if mtp_draft else "DeepseekV4ForCausalLM"]
    model_type = "deepseek_mtp" if mtp_draft else "deepseek_v4"
    text_config = SimpleNamespace(
        architectures=architectures,
        model_type=model_type,
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
    model_config = SimpleNamespace(
        architectures=architectures,
        model_type=model_type,
        hf_text_config=text_config,
        hf_config=text_config,
        dtype="bfloat16",
        quantization="deepseek_v4_fp8",
        use_mla=True,
        is_hybrid=False,
        is_moe=True,
        enforce_eager=False,
    )
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype, block_size=64),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            async_scheduling=async_scheduling,
        ),
        attention_config=SimpleNamespace(backend="FLASHMLA"),
        compilation_config=SimpleNamespace(
            mode="NONE",
            cudagraph_mode="FULL_DECODE_ONLY",
        ),
        speculative_config=(
            SimpleNamespace(
                method=speculative_method,
                num_speculative_tokens=4,
            )
            if speculative
            else None
        ),
        quant_config=SimpleNamespace(weight_block_size=[128, 128]),
    )


def test_flash_base_tp8_resolves_validated_profile() -> None:
    plan = resolve_runtime_plan(_flash_base_config())

    assert plan.model.family is ModelFamily.DEEPSEEK_V4
    assert plan.model.role is ModelRole.TEXT
    assert plan.model.uses_mla is True
    assert plan.profile == "deepseek_v4.tp8_flash_base"
    assert plan.enabled(RuntimeDecision.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)
    assert plan.enabled(RuntimeDecision.DEEPSEEK_V4_NATIVE_SPARSE_INDEXER)
    assert plan.enabled(RuntimeDecision.DEEPSEEK_V4_MATERIALIZED_PREFILL_INDEXER)
    assert plan.value(RuntimeDecision.DEEPSEEK_V4_FLASHMLA_SPARSE_PAGE_SIZE) == 256
    assert plan.enabled(RuntimeDecision.DEEPSEEK_V4_TP8_FUSED_ADD_RMSNORM_BLOCK256)
    assert plan.supports(RuntimeDecision.DEEPSEEK_V4_CAR_GRAPH_INPUT_CAPTURE_GUARD)


def test_flash_base_tp4_is_supported_but_not_tp8_preferred() -> None:
    plan = resolve_runtime_plan(_flash_base_config(tp=4))

    assert plan.supports(RuntimeDecision.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)
    assert plan.supports(RuntimeDecision.DEEPSEEK_V4_CAR_GRAPH_INPUT_CAPTURE_GUARD)
    assert not plan.enabled(RuntimeDecision.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)
    assert plan.value(RuntimeDecision.DEEPSEEK_V4_FLASHMLA_SPARSE_PAGE_SIZE) == 256
    assert not plan.enabled(RuntimeDecision.DEEPSEEK_V4_TP8_FUSED_ADD_RMSNORM_BLOCK256)
    assert plan.enabled(RuntimeDecision.DEEPSEEK_V4_NATIVE_SPARSE_INDEXER)


def test_flash_base_multibatch_keeps_shared_mlp_path() -> None:
    config = _flash_base_config()
    config.scheduler_config.max_num_seqs = 64

    plan = resolve_runtime_plan(config)

    assert plan.profile == "deepseek_v4.unvalidated"
    assert plan.enabled(RuntimeDecision.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)
    assert plan.value(RuntimeDecision.DEEPSEEK_V4_FLASHMLA_SPARSE_PAGE_SIZE) == 256
    assert not plan.enabled(RuntimeDecision.DEEPSEEK_V4_TP8_FUSED_ADD_RMSNORM_BLOCK256)


def test_model_config_without_execution_topology_is_support_only() -> None:
    config = _flash_base_config()

    plan = resolve_runtime_plan(model_config=config.model_config)

    assert plan.supports(RuntimeDecision.DEEPSEEK_V4_NATIVE_SPARSE_INDEXER)
    assert not plan.enabled_decisions


def test_incomplete_deepseek_identity_fails_closed() -> None:
    config = _flash_base_config()
    config.model_config.architectures = []
    config.model_config.hf_text_config.architectures = []
    config.model_config.hf_config.architectures = []

    plan = resolve_runtime_plan(config)

    assert plan.model.family is ModelFamily.DEEPSEEK_V4
    assert plan.profile == "deepseek_v4.unvalidated"
    assert not plan.supported_decisions
    assert not plan.enabled_decisions


@pytest.mark.parametrize(
    ("owner", "field", "value", "expected_supported"),
    [
        ("model_config", "architectures", ["Qwen3ForCausalLM"], frozenset()),
        ("text_config", "model_type", "qwen3", frozenset()),
        (
            "model_config",
            "dtype",
            "float16",
            frozenset({RuntimeDecision.DEEPSEEK_V4_CAR_GRAPH_INPUT_CAPTURE_GUARD}),
        ),
        (
            "model_config",
            "quantization",
            "compressed-tensors",
            frozenset({RuntimeDecision.DEEPSEEK_V4_CAR_GRAPH_INPUT_CAPTURE_GUARD}),
        ),
        (
            "model_config",
            "use_mla",
            False,
            frozenset({RuntimeDecision.DEEPSEEK_V4_CAR_GRAPH_INPUT_CAPTURE_GUARD}),
        ),
    ],
)
def test_identity_and_model_traits_fail_closed(
    owner: str,
    field: str,
    value: object,
    expected_supported: frozenset[RuntimeDecision],
) -> None:
    config = _flash_base_config()
    target = (
        config.model_config.hf_text_config
        if owner == "text_config"
        else config.model_config
    )
    setattr(target, field, value)

    plan = resolve_runtime_plan(config)

    assert plan.supported_decisions == expected_supported
    assert not plan.enabled_decisions


def test_hybrid_deepseek_v4_does_not_enable_flash_base_features() -> None:
    config = _flash_base_config()
    config.model_config.is_hybrid = True

    plan = resolve_runtime_plan(config)

    assert plan.supported_decisions == frozenset(
        {
            RuntimeDecision.DEEPSEEK_V4_CAR_GRAPH_INPUT_CAPTURE_GUARD,
            RuntimeDecision.HYBRID_KV_CACHE_POOL_LAYOUT,
        }
    )
    assert not plan.enabled_decisions
    assert plan.selected(
        RuntimeDecision.HYBRID_KV_CACHE_POOL_LAYOUT,
        "separate",
    )


def test_non_block128_quantization_fails_closed() -> None:
    config = _flash_base_config()
    config.quant_config.weight_block_size = [64, 128]
    config.model_config.hf_text_config.quantization_config["weight_block_size"] = [
        64,
        128,
    ]

    plan = resolve_runtime_plan(config)

    assert plan.supported_decisions == frozenset(
        {RuntimeDecision.DEEPSEEK_V4_CAR_GRAPH_INPUT_CAPTURE_GUARD}
    )
    assert not plan.enabled_decisions


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hidden_size", 7168),
        ("num_hidden_layers", 44),
        ("num_attention_heads", 32),
        ("num_key_value_heads", 2),
        ("head_dim", 256),
        ("vocab_size", 129281),
        ("n_routed_experts", 384),
        ("num_experts_per_tok", 8),
        ("n_shared_experts", 2),
        ("moe_intermediate_size", 3072),
        ("expert_dtype", "bf16"),
        ("hidden_act", "gelu"),
        ("swiglu_limit", 0.0),
        ("index_topk", 1024),
    ],
)
def test_one_field_model_mismatch_disables_flash_base_features(
    field: str,
    value: object,
) -> None:
    config = _flash_base_config()
    setattr(config.model_config.hf_text_config, field, value)

    plan = resolve_runtime_plan(config)

    assert plan.model.family is ModelFamily.DEEPSEEK_V4
    assert plan.supported_decisions == frozenset(
        {RuntimeDecision.DEEPSEEK_V4_CAR_GRAPH_INPUT_CAPTURE_GUARD}
    )
    assert not plan.enabled_decisions


def test_model_config_use_mla_takes_precedence_over_missing_hf_fact() -> None:
    config = _flash_base_config()
    assert not hasattr(config.model_config.hf_text_config, "use_mla")
    assert not hasattr(config.model_config.hf_text_config, "kv_lora_rank")

    plan = resolve_runtime_plan(config)

    assert plan.model.uses_mla is True
    assert plan.enabled(RuntimeDecision.DEEPSEEK_V4_NATIVE_SPARSE_INDEXER)


def test_speculative_execution_keeps_support_but_disables_tp8_preferences() -> None:
    config = _flash_base_config(speculative=True)
    plan = resolve_runtime_plan(config)

    assert plan.supports(RuntimeDecision.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)
    assert not plan.enabled(RuntimeDecision.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)
    assert plan.value(RuntimeDecision.DEEPSEEK_V4_FLASHMLA_SPARSE_PAGE_SIZE) == 256
    assert plan.enabled(RuntimeDecision.DEEPSEEK_V4_MTP_CAR_GRAPH_STAGING_ARENA)
    assert deepseek_v4_mtp_graph_registered_inputs_enabled(config)
    plan = deepseek_v4_mtp_car_graph_staging_plan(config)
    assert plan is not None
    assert plan.eager_reserve_bytes == 68 * 1024 * 1024
    assert plan.car_ops_per_descriptor == 87
    assert plan.bytes_per_token == 8192
    assert plan.graph_data_capacity_bytes == 110_469_120
    assert plan.graph_meta_capacity_bytes == 117_596_160
    assert plan.max_meta_bytes_per_slot == 16 * 1024
    assert plan.communicator_buffer_bytes == 184 * 1024 * 1024
    assert plan.capture_descriptors == frozenset(
        ((5, 1), (10, 2), (20, 4), (40, 8), (80, 16))
    )
    assert plan.allows_descriptor(
        SimpleNamespace(
            num_tokens=80,
            num_reqs=16,
            uniform=True,
            has_lora=False,
            num_active_loras=0,
        )
    )
    assert not plan.allows_descriptor(
        SimpleNamespace(
            num_tokens=160,
            num_reqs=32,
            uniform=True,
            has_lora=False,
            num_active_loras=0,
        )
    )
    assert not plan.allows_descriptor(
        SimpleNamespace(
            num_tokens=80,
            num_reqs=None,
            uniform=True,
            has_lora=False,
            num_active_loras=0,
        )
    )
    assert not deepseek_v4_mtp_prefill_step_requires_sync(
        SimpleNamespace(scheduled_new_reqs=[]),
    )


def test_mtp_car_graph_staging_plan_requires_mtp4() -> None:
    config = _flash_base_config(speculative=True, max_num_seqs=64)
    config.speculative_config.num_speculative_tokens = 3

    assert deepseek_v4_mtp_car_graph_guard_enabled(config)
    assert deepseek_v4_mtp_car_graph_staging_plan(config) is None


def test_mtp4_bs1_preserves_registered_graph_path() -> None:
    config = _flash_base_config(speculative=True, max_num_seqs=1)

    assert deepseek_v4_mtp_graph_registered_inputs_enabled(config)
    assert deepseek_v4_mtp_car_graph_staging_plan(config) is not None


def test_tp8_mtp_target_prefers_sparse_prefill_fixes() -> None:
    config = _flash_base_config(
        speculative=True,
        max_num_seqs=64,
        cache_dtype="fp8_ds_mla",
    )
    plan = resolve_runtime_plan(config)

    assert plan.profile == "deepseek_v4.tp8_flash_base_mtp"
    assert plan.enabled(RuntimeDecision.DEEPSEEK_V4_TP8_MTP_SPARSE_DIRECT_OUT)
    assert plan.enabled(RuntimeDecision.DEEPSEEK_V4_TP8_MTP_SPARSE_PREFILL_HEADROOM)
    assert plan.enabled(RuntimeDecision.DEEPSEEK_V4_MTP_CAR_GRAPH_STAGING_ARENA)
    assert plan.enabled(RuntimeDecision.DEEPSEEK_V4_TP8_MTP_ASYNC_PREFILL_QUEUE_FENCE)
    assert deepseek_v4_mtp_sparse_prefill_headroom_bytes(config) == 1_073_938_432

    config.parallel_config.disable_custom_all_reduce = True
    assert deepseek_v4_mtp_sparse_prefill_headroom_bytes(config) == 537_067_520


def test_tp8_mtp_multibatch_syncs_only_prefill_or_mixed_steps() -> None:
    config = _flash_base_config(
        speculative=True,
        max_num_seqs=64,
        cache_dtype="fp8_ds_mla",
    )

    def step(*, new=(), cached_context=(), scheduled=("a", "b")):
        context_req_ids = set(cached_context)
        return SimpleNamespace(
            scheduled_new_reqs=list(new),
            scheduled_cached_reqs=SimpleNamespace(
                num_output_tokens=[
                    0 if req_id in context_req_ids else 1 for req_id in scheduled
                ]
            ),
            num_scheduled_tokens=dict.fromkeys(scheduled, 1),
            # A structured-output decode may legitimately have no drafts.
            scheduled_spec_decode_tokens={},
        )

    assert deepseek_v4_mtp_async_prefill_queue_fence_enabled(config)
    assert not deepseek_v4_mtp_prefill_step_requires_sync(step())
    assert deepseek_v4_mtp_prefill_step_requires_sync(
        step(new=("new",), scheduled=("new", "a"))
    )
    assert deepseek_v4_mtp_prefill_step_requires_sync(step(cached_context=("a",)))

    sync_config = _flash_base_config(
        speculative=True,
        max_num_seqs=64,
        cache_dtype="fp8_ds_mla",
        async_scheduling=False,
    )
    assert not resolve_runtime_plan(sync_config).enabled(
        RuntimeDecision.DEEPSEEK_V4_TP8_MTP_ASYNC_PREFILL_QUEUE_FENCE
    )
    assert not deepseek_v4_mtp_async_prefill_queue_fence_enabled(sync_config)


def test_flashmla_owner_hint_only_fills_a_missing_backend() -> None:
    config = _flash_base_config(
        speculative=True,
        max_num_seqs=64,
        cache_dtype="fp8_ds_mla",
    )
    config.attention_config.backend = None

    unhinted = resolve_runtime_plan(config)
    hinted = resolve_runtime_plan(
        config,
        attention_backend_hint="flashmla",
    )

    assert not unhinted.enabled(RuntimeDecision.DEEPSEEK_V4_TP8_MTP_SPARSE_DIRECT_OUT)
    assert hinted.enabled(RuntimeDecision.DEEPSEEK_V4_TP8_MTP_SPARSE_DIRECT_OUT)

    config.attention_config.backend = "FLASH_ATTN"
    conflicting = resolve_runtime_plan(
        config,
        attention_backend_hint="flashmla",
    )
    assert conflicting.execution.attention_backend == "flash_attn"
    assert not conflicting.enabled(
        RuntimeDecision.DEEPSEEK_V4_TP8_MTP_SPARSE_DIRECT_OUT
    )


def test_tp8_mtp_draft_prefers_sparse_prefill_fixes() -> None:
    config = _flash_base_config(
        max_num_seqs=64,
        mtp_draft=True,
        cache_dtype="fp8_ds_mla",
    )
    plan = resolve_runtime_plan(config)

    assert plan.model.role is ModelRole.MTP_DRAFT
    assert plan.profile == "deepseek_v4.tp8_flash_mtp_draft"
    assert plan.value(RuntimeDecision.DEEPSEEK_V4_FLASHMLA_SPARSE_PAGE_SIZE) == 256
    assert plan.enabled(RuntimeDecision.DEEPSEEK_V4_TP8_MTP_SPARSE_DIRECT_OUT)
    assert plan.enabled(RuntimeDecision.DEEPSEEK_V4_TP8_MTP_SPARSE_PREFILL_HEADROOM)


def test_mtp_draft_uses_page256_outside_tp8_profile() -> None:
    config = _flash_base_config(
        tp=1,
        max_num_seqs=1,
        mtp_draft=True,
        cache_dtype="fp8_ds_mla",
    )

    plan = resolve_runtime_plan(config)

    assert plan.profile == "deepseek_v4.unvalidated"
    assert plan.supports(RuntimeDecision.DEEPSEEK_V4_FLASHMLA_SPARSE_PAGE_SIZE)
    assert plan.value(RuntimeDecision.DEEPSEEK_V4_FLASHMLA_SPARSE_PAGE_SIZE) == 256


@pytest.mark.parametrize(
    ("owner", "field", "value"),
    [
        ("parallel_config", "tensor_parallel_size", 4),
        ("scheduler_config", "max_num_seqs", 1),
        ("scheduler_config", "max_num_seqs", None),
        ("cache_config", "cache_dtype", "auto"),
        ("attention_config", "backend", "FLASH_ATTN"),
        ("compilation_config", "mode", "VLLM_COMPILE"),
        ("compilation_config", "cudagraph_mode", "PIECEWISE"),
    ],
)
def test_tp8_mtp_sparse_prefill_fixes_fail_closed(
    owner: str,
    field: str,
    value: object,
) -> None:
    config = _flash_base_config(
        max_num_seqs=64,
        mtp_draft=True,
    )
    setattr(getattr(config, owner), field, value)

    plan = resolve_runtime_plan(config)

    assert not plan.enabled(RuntimeDecision.DEEPSEEK_V4_TP8_MTP_SPARSE_DIRECT_OUT)
    assert not plan.enabled(RuntimeDecision.DEEPSEEK_V4_TP8_MTP_SPARSE_PREFILL_HEADROOM)


@pytest.mark.parametrize(
    ("owner", "field", "value"),
    [
        ("cache_config", "cache_dtype", "auto"),
        ("scheduler_config", "max_num_seqs", 4),
        ("attention_config", "backend", "FLASH_ATTN"),
        ("compilation_config", "mode", "VLLM_COMPILE"),
        ("compilation_config", "cudagraph_mode", "PIECEWISE"),
    ],
)
def test_one_field_execution_mismatch_keeps_only_shared_mlp(
    owner: str,
    field: str,
    value: object,
) -> None:
    config = _flash_base_config()
    setattr(getattr(config, owner), field, value)

    plan = resolve_runtime_plan(config)

    assert plan.supports(RuntimeDecision.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)
    if owner == "scheduler_config":
        assert plan.enabled(RuntimeDecision.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)
    else:
        assert not plan.enabled(RuntimeDecision.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)
    assert (
        plan.value(
            RuntimeDecision.DEEPSEEK_V4_FLASHMLA_SPARSE_PAGE_SIZE,
            64,
        )
        == 256
    )
    assert not plan.enabled(RuntimeDecision.DEEPSEEK_V4_TP8_FUSED_ADD_RMSNORM_BLOCK256)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pipeline_parallel_size", 2),
        ("data_parallel_size", 2),
        ("decode_context_parallel_size", 2),
    ],
)
def test_non_reference_parallel_topology_disables_tp8_profile(
    field: str,
    value: int,
) -> None:
    config = _flash_base_config()
    setattr(config.parallel_config, field, value)

    plan = resolve_runtime_plan(config)

    assert (
        plan.value(
            RuntimeDecision.DEEPSEEK_V4_FLASHMLA_SPARSE_PAGE_SIZE,
            64,
        )
        == 256
    )
    assert not plan.enabled(RuntimeDecision.DEEPSEEK_V4_TP8_FUSED_ADD_RMSNORM_BLOCK256)


def test_pooling_and_missing_quant_runtime_disable_tp8_profile() -> None:
    config = _flash_base_config()
    config.quant_config = None

    plan = resolve_runtime_plan(config, is_pooling_model=True)

    assert plan.supports(RuntimeDecision.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)
    assert not plan.enabled(RuntimeDecision.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)
    assert plan.value(RuntimeDecision.DEEPSEEK_V4_FLASHMLA_SPARSE_PAGE_SIZE) == 256
    assert not plan.enabled(RuntimeDecision.DEEPSEEK_V4_TP8_FUSED_ADD_RMSNORM_BLOCK256)


def test_batch_invariant_is_snapshotted_as_an_execution_fact(monkeypatch) -> None:
    fake_vllm = ModuleType("vllm")
    fake_envs = ModuleType("vllm.envs")
    fake_envs.VLLM_BATCH_INVARIANT = True
    fake_vllm.envs = fake_envs
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.envs", fake_envs)

    plan = resolve_runtime_plan(_flash_base_config())

    assert plan.execution.batch_invariant_enabled is True
    assert plan.supports(RuntimeDecision.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)
    assert not plan.enabled(RuntimeDecision.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)


def test_plan_binds_to_runtime_owner_once_resolved() -> None:
    owner = SimpleNamespace()

    plan = bind_runtime_plan(owner, _flash_base_config())

    assert owner._musa_runtime_plan is plan
    assert plan.enabled(RuntimeDecision.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)
