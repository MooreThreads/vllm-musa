from types import SimpleNamespace

import pytest

from vllm_musa.optimization_contract import (
    ModelFamily,
    ModelRole,
    OptimizationFeature,
    resolve_optimization_contract,
)


def _config(
    *,
    architecture: str,
    model_type: str,
    dtype: object = "bfloat16",
    quantization: str | None = None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    num_hidden_layers: int | None = None,
    num_attention_heads: int | None = None,
    num_key_value_heads: int | None = None,
    head_dim: int | None = None,
    num_experts: int | None = None,
    num_experts_per_tok: int | None = None,
    moe_intermediate_size: int | None = None,
    linear_num_key_heads: int | None = None,
    linear_num_value_heads: int | None = None,
    linear_key_head_dim: int | None = None,
    linear_value_head_dim: int | None = None,
    linear_conv_kernel_dim: int | None = None,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    dcp: int = 1,
    speculative: bool = False,
    cache_dtype: object = "auto",
    cache_block_size: int | None = 64,
    enforce_eager: bool = False,
):
    text_config = SimpleNamespace(
        model_type=model_type,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        linear_num_key_heads=linear_num_key_heads,
        linear_num_value_heads=linear_num_value_heads,
        linear_key_head_dim=linear_key_head_dim,
        linear_value_head_dim=linear_value_head_dim,
        linear_conv_kernel_dim=linear_conv_kernel_dim,
        quantization_config=(
            {"quant_method": quantization} if quantization is not None else None
        ),
    )
    model_config = SimpleNamespace(
        architectures=[architecture],
        hf_text_config=text_config,
        hf_config=text_config,
        dtype=dtype,
        quantization=quantization,
        enforce_eager=enforce_eager,
    )
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            data_parallel_size=dp,
            decode_context_parallel_size=dcp,
        ),
        cache_config=SimpleNamespace(
            cache_dtype=cache_dtype,
            block_size=cache_block_size,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=64),
        speculative_config=object() if speculative else None,
        quant_config=object() if quantization not in (None, "none") else None,
    )


@pytest.mark.parametrize(
    ("architecture", "model_type", "family", "role"),
    [
        ("Qwen2ForCausalLM", "qwen2", ModelFamily.QWEN2, ModelRole.TEXT),
        ("Qwen3ForCausalLM", "qwen3", ModelFamily.QWEN3, ModelRole.TEXT),
        (
            "Qwen3_5ForConditionalGeneration",
            "qwen3_5_text",
            ModelFamily.QWEN35_36,
            ModelRole.TEXT,
        ),
        (
            "Qwen3_5MoeForConditionalGeneration",
            "qwen3_5_moe_text",
            ModelFamily.QWEN35_36,
            ModelRole.TEXT,
        ),
        (
            "CosyVoice3Model",
            "cosyvoice3",
            ModelFamily.QWEN2,
            ModelRole.COSYVOICE_TALKER,
        ),
    ],
)
def test_resolves_qwen_family_without_model_name(
    architecture: str,
    model_type: str,
    family: ModelFamily,
    role: ModelRole,
) -> None:
    contract = resolve_optimization_contract(
        _config(architecture=architecture, model_type=model_type)
    )
    assert contract.model.family is family
    assert contract.model.role is role


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen3VLForConditionalGeneration",
        "Qwen3OmniMoeForConditionalGeneration",
        "DeepseekV4ForCausalLM",
        "LlamaForCausalLM",
    ],
)
def test_unknown_or_future_families_fail_closed(architecture: str) -> None:
    contract = resolve_optimization_contract(
        _config(architecture=architecture, model_type="unknown")
    )
    assert contract.model.family is ModelFamily.UNKNOWN
    assert not contract.supported_features
    assert not contract.preferred_features


def test_deepseek_v4_facts_are_normalized_without_enabling_a_provider() -> None:
    text_config = SimpleNamespace(
        model_type="deepseek_v4",
        kv_lora_rank=512,
        index_topk=2048,
        quantization_config={"weight_block_size": [128, 128]},
    )
    contract = resolve_optimization_contract(
        SimpleNamespace(
            model_config=SimpleNamespace(
                architectures=["DeepseekV4ForCausalLM"],
                hf_text_config=text_config,
                hf_config=text_config,
                dtype="bfloat16",
                quantization="fp8",
            ),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=8,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                decode_context_parallel_size=1,
            ),
            attention_config=SimpleNamespace(backend="FLASHMLA"),
            compilation_config=SimpleNamespace(
                mode="NONE",
                cudagraph_mode="FULL_DECODE_ONLY",
            ),
            quant_config=None,
        )
    )
    assert contract.model.family is ModelFamily.UNKNOWN
    assert contract.model.uses_mla is True
    assert contract.model.index_topk == 2048
    assert contract.model.quant_block_shape == (128, 128)
    assert contract.execution.attention_backend == "flashmla"
    assert contract.execution.cudagraph_mode == "full_decode_only"
    assert not contract.preferred_features


def test_qwen_runner_and_fa3_architecture_sets_preserve_current_scope() -> None:
    causal_variant = resolve_optimization_contract(
        _config(
            architecture="Qwen3_5ForCausalLM",
            model_type="qwen3_5_text",
        )
    )
    assert causal_variant.prefers(OptimizationFeature.QWEN_V2_SAMPLING)
    assert not causal_variant.prefers(OptimizationFeature.QWEN_LEGACY_SAMPLING)
    assert not causal_variant.prefers(OptimizationFeature.QWEN_FA3_SCHEDULER)

    conditional_variant = resolve_optimization_contract(
        _config(
            architecture="Qwen3_5ForConditionalGeneration",
            model_type="qwen3_5_text",
        )
    )
    assert conditional_variant.prefers(OptimizationFeature.QWEN_V2_SAMPLING)
    assert conditional_variant.prefers(OptimizationFeature.QWEN_LEGACY_SAMPLING)
    assert conditional_variant.prefers(OptimizationFeature.QWEN_FA3_SCHEDULER)

    cosyvoice = resolve_optimization_contract(
        _config(architecture="CosyVoice3Model", model_type="cosyvoice3")
    )
    assert cosyvoice.prefers(OptimizationFeature.QWEN_FA3_SCHEDULER)
    assert not cosyvoice.prefers(OptimizationFeature.QWEN_V2_SAMPLING)


def test_exact_qwen3_dense_fp8_contract() -> None:
    config = _config(
        architecture="Qwen3ForCausalLM",
        model_type="qwen3",
        quantization="fp8",
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
    )
    contract = resolve_optimization_contract(config)
    assert contract.prefers(OptimizationFeature.QWEN3_DENSE_FP8_POST_GRAD_FUSIONS)

    config.parallel_config.tensor_parallel_size = 2
    contract = resolve_optimization_contract(config)
    assert not contract.supports(OptimizationFeature.QWEN3_DENSE_FP8_POST_GRAD_FUSIONS)


@pytest.mark.parametrize(
    "geometry",
    [
        (1024, 3072, 28, 16, 8, 128),
        (4096, 12288, 36, 32, 8, 128),
    ],
)
def test_exact_qwen3_qk_rope_kv_contract(geometry: tuple[int, ...]) -> None:
    contract = resolve_optimization_contract(
        _config(
            architecture="Qwen3ForCausalLM",
            model_type="qwen3",
            hidden_size=geometry[0],
            intermediate_size=geometry[1],
            num_hidden_layers=geometry[2],
            num_attention_heads=geometry[3],
            num_key_value_heads=geometry[4],
            head_dim=geometry[5],
        )
    )
    assert contract.prefers(OptimizationFeature.QWEN3_QK_ROPE_KV_PRESPLIT)


def test_exact_qwen2_and_cosyvoice_rope_kv_contracts() -> None:
    qwen2 = _config(
        architecture="Qwen2ForCausalLM",
        model_type="qwen2",
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
    )
    assert resolve_optimization_contract(qwen2).prefers(
        OptimizationFeature.QWEN2_ROPE_KV_PRESPLIT
    )

    cosyvoice = _config(
        architecture="CosyVoice3Model",
        model_type="cosyvoice3",
        hidden_size=896,
        intermediate_size=None,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=None,
    )
    assert resolve_optimization_contract(cosyvoice).prefers(
        OptimizationFeature.QWEN2_ROPE_KV_PRESPLIT
    )


def test_qwen35_dense_gdn_and_moe_prefill_static_contracts() -> None:
    dense = _config(
        architecture="Qwen3_5ForConditionalGeneration",
        model_type="qwen3_5_text",
        hidden_size=5120,
        intermediate_size=17408,
        num_experts=None,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    dense_contract = resolve_optimization_contract(dense)
    assert dense_contract.model.gdn_conv_dim == 10240
    assert dense_contract.prefers(OptimizationFeature.QWEN35_GDN_WIDTH4_PREFILL)
    assert dense_contract.prefers(OptimizationFeature.QWEN35_INTERLEAVED_MROPE_QK)
    assert dense_contract.prefers(OptimizationFeature.HYBRID_SEPARATE_MAMBA_POOL)
    assert not dense_contract.prefers(OptimizationFeature.QWEN35_SHARED_EXPERT_FOLD)

    moe = _config(
        architecture="Qwen3_5MoeForConditionalGeneration",
        model_type="qwen3_5_moe_text",
        hidden_size=2048,
        num_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        tp=4,
    )
    moe_contract = resolve_optimization_contract(moe)
    assert moe_contract.model.gdn_conv_dim == 8192
    assert moe_contract.prefers(OptimizationFeature.QWEN35_MOE_BF16_PREFILL)
    assert moe_contract.prefers(OptimizationFeature.QWEN35_SHARED_EXPERT_FOLD)
    assert moe_contract.prefers(OptimizationFeature.QWEN35_INTERLEAVED_MROPE_QK)
    assert not moe_contract.prefers(OptimizationFeature.QWEN35_GDN_WIDTH4_PREFILL)


def test_static_contract_rejects_one_field_mismatches() -> None:
    config = _config(
        architecture="Qwen3ForCausalLM",
        model_type="qwen3",
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
    )
    config.speculative_config = object()
    contract = resolve_optimization_contract(config)
    assert not contract.prefers(OptimizationFeature.QWEN3_QK_ROPE_KV_PRESPLIT)

    config.speculative_config = None
    config.cache_config.block_size = 16
    contract = resolve_optimization_contract(config)
    assert not contract.prefers(OptimizationFeature.QWEN3_QK_ROPE_KV_PRESPLIT)
