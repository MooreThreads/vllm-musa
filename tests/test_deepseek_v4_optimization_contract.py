from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_musa.optimization_contract import (
    ModelFamily,
    ModelRole,
    OptimizationFeature,
    bind_optimization_contract,
    resolve_optimization_contract,
)


def _flash_base_config(*, tp: int = 8, speculative: bool = False):
    text_config = SimpleNamespace(
        architectures=["DeepseekV4ForCausalLM"],
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
    model_config = SimpleNamespace(
        architectures=["DeepseekV4ForCausalLM"],
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
        cache_config=SimpleNamespace(cache_dtype="fp8", block_size=64),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        attention_config=SimpleNamespace(backend="FLASHMLA"),
        compilation_config=SimpleNamespace(
            mode="NONE",
            cudagraph_mode="FULL_DECODE_ONLY",
        ),
        speculative_config=object() if speculative else None,
        quant_config=SimpleNamespace(weight_block_size=[128, 128]),
    )


def test_flash_base_tp8_resolves_validated_profile() -> None:
    contract = resolve_optimization_contract(_flash_base_config())

    assert contract.model.family is ModelFamily.DEEPSEEK_V4
    assert contract.model.role is ModelRole.TEXT
    assert contract.model.uses_mla is True
    assert contract.profile == "deepseek_v4.tp8_flash_base"
    assert contract.prefers(OptimizationFeature.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)
    assert contract.prefers(OptimizationFeature.DEEPSEEK_V4_NATIVE_SPARSE_INDEXER)
    assert contract.prefers(
        OptimizationFeature.DEEPSEEK_V4_MATERIALIZED_PREFILL_INDEXER
    )
    assert contract.prefers(OptimizationFeature.DEEPSEEK_V4_TP8_FLASHMLA_SPARSE_PAGE256)
    assert contract.prefers(
        OptimizationFeature.DEEPSEEK_V4_TP8_FUSED_ADD_RMSNORM_BLOCK256
    )


def test_flash_base_tp4_is_supported_but_not_tp8_preferred() -> None:
    contract = resolve_optimization_contract(_flash_base_config(tp=4))

    assert contract.supports(OptimizationFeature.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)
    assert not contract.prefers(OptimizationFeature.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)
    assert not contract.prefers(
        OptimizationFeature.DEEPSEEK_V4_TP8_FLASHMLA_SPARSE_PAGE256
    )
    assert not contract.prefers(
        OptimizationFeature.DEEPSEEK_V4_TP8_FUSED_ADD_RMSNORM_BLOCK256
    )
    assert contract.prefers(OptimizationFeature.DEEPSEEK_V4_NATIVE_SPARSE_INDEXER)


def test_incomplete_deepseek_identity_fails_closed() -> None:
    config = _flash_base_config()
    config.model_config.architectures = []
    config.model_config.hf_text_config.architectures = []
    config.model_config.hf_config.architectures = []

    contract = resolve_optimization_contract(config)

    assert contract.model.family is ModelFamily.DEEPSEEK_V4
    assert contract.profile == "deepseek_v4.unvalidated"
    assert not contract.supported_features
    assert not contract.preferred_features


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

    contract = resolve_optimization_contract(config)

    assert contract.model.family is ModelFamily.DEEPSEEK_V4
    assert not contract.supported_features
    assert not contract.preferred_features


def test_model_config_use_mla_takes_precedence_over_missing_hf_fact() -> None:
    config = _flash_base_config()
    assert not hasattr(config.model_config.hf_text_config, "use_mla")
    assert not hasattr(config.model_config.hf_text_config, "kv_lora_rank")

    contract = resolve_optimization_contract(config)

    assert contract.model.uses_mla is True
    assert contract.prefers(OptimizationFeature.DEEPSEEK_V4_NATIVE_SPARSE_INDEXER)


def test_speculative_execution_keeps_support_but_disables_tp8_preferences() -> None:
    contract = resolve_optimization_contract(_flash_base_config(speculative=True))

    assert contract.supports(OptimizationFeature.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)
    assert not contract.prefers(OptimizationFeature.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)
    assert not contract.prefers(
        OptimizationFeature.DEEPSEEK_V4_TP8_FLASHMLA_SPARSE_PAGE256
    )


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
def test_one_field_execution_mismatch_disables_tp8_profile(
    owner: str,
    field: str,
    value: object,
) -> None:
    config = _flash_base_config()
    setattr(getattr(config, owner), field, value)

    contract = resolve_optimization_contract(config)

    assert contract.supports(OptimizationFeature.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)
    assert not contract.prefers(OptimizationFeature.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)
    assert not contract.prefers(
        OptimizationFeature.DEEPSEEK_V4_TP8_FLASHMLA_SPARSE_PAGE256
    )
    assert not contract.prefers(
        OptimizationFeature.DEEPSEEK_V4_TP8_FUSED_ADD_RMSNORM_BLOCK256
    )


def test_batch_invariant_is_snapshotted_as_an_execution_fact(monkeypatch) -> None:
    fake_vllm = ModuleType("vllm")
    fake_envs = ModuleType("vllm.envs")
    fake_envs.VLLM_BATCH_INVARIANT = True
    fake_vllm.envs = fake_envs
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.envs", fake_envs)

    contract = resolve_optimization_contract(_flash_base_config())

    assert contract.execution.batch_invariant_enabled is True
    assert contract.supports(OptimizationFeature.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)
    assert not contract.prefers(OptimizationFeature.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)


def test_contract_binds_to_runtime_owner_once_resolved() -> None:
    owner = SimpleNamespace()

    contract = bind_optimization_contract(owner, _flash_base_config())

    assert owner._musa_optimization_contract is contract
    assert contract.prefers(OptimizationFeature.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)
