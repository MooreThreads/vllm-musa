from __future__ import annotations

from dataclasses import replace

from .types import (
    ExecutionSignature,
    ModelFamily,
    ModelRole,
    ModelSignature,
    MusaOptimizationContract,
    OptimizationFeature,
)

QWEN_V2_SAMPLING_ARCHITECTURES = frozenset(
    {
        "Qwen2ForCausalLM",
        "Qwen2MoeForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
    }
)
QWEN_LEGACY_SAMPLING_ARCHITECTURES = frozenset(
    {
        "Qwen2ForCausalLM",
        "Qwen2MoeForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }
)
QWEN_FA3_ARCHITECTURES = frozenset(
    {*QWEN_LEGACY_SAMPLING_ARCHITECTURES, "CosyVoice3Model"}
)
_QWEN35_ARCHITECTURES = frozenset(
    {
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
    }
)


def matches_qwen35_moe_bf16_prefill_layer(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    global_num_experts: int | None,
    *,
    min_tokens: int,
) -> bool:
    """Match the layer-bound Qwen3.5/3.6-35B-A3B prefill signature.

    The caller is the generic fused-MoE custom-op boundary, so tensor and
    layout checks remain here rather than attempting to recover model config
    from global runtime state on every token.
    """

    def dtype_name(value) -> str:
        return str(value).lower().removeprefix("torch.")

    try:
        return (
            global_num_experts == 256
            and hidden_states.ndim == 2
            and dtype_name(hidden_states.dtype) == "bfloat16"
            and dtype_name(w1.dtype) == "bfloat16"
            and dtype_name(w2.dtype) == "bfloat16"
            and hidden_states.shape[0] >= min_tokens
            and hidden_states.shape[1] == 2048
            and tuple(w1.shape) == (256, 256, 2048)
            and tuple(w2.shape) == (256, 2048, 128)
            and topk_weights.ndim == 2
            and topk_ids.ndim == 2
            and topk_weights.shape == topk_ids.shape
            and topk_ids.shape[1] == 8
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        return False


def _has_architecture(model: ModelSignature, allowed: frozenset[str]) -> bool:
    architectures = model.outer_architectures or model.architectures
    return any(architecture in allowed for architecture in architectures)


def _single_device(execution: ExecutionSignature, *, include_dcp: bool) -> bool:
    sizes = (
        execution.tensor_parallel_size,
        execution.pipeline_parallel_size,
        execution.data_parallel_size,
    )
    if include_dcp:
        sizes = (*sizes, execution.decode_context_parallel_size)
    return all(size == 1 for size in sizes)


def _cache_supports_fused_qwen_attention(execution: ExecutionSignature) -> bool:
    return execution.cache_dtype in (None, "auto", "bfloat16") and (
        execution.cache_block_size in (None, 64)
    )


def _qwen2_rope_kv_preferred(
    model: ModelSignature,
    execution: ExecutionSignature,
) -> bool:
    if model.family is not ModelFamily.QWEN2:
        return False
    if model.model_type not in ("qwen2", "cosyvoice3"):
        return False
    if model.model_type == "qwen2":
        if model.num_key_value_heads != 2 or model.intermediate_size != 4864:
            return False
    elif model.num_key_value_heads not in (None, 2) or model.intermediate_size not in (
        None,
        4864,
    ):
        return False
    return (
        model.hidden_size == 896
        and model.num_hidden_layers == 24
        and model.num_attention_heads == 14
        and model.dtype == "bfloat16"
        and model.quantization in (None, "none")
        and not execution.has_quant_config
        and not execution.has_speculative_config
        and execution.has_parallel_config
        and _cache_supports_fused_qwen_attention(execution)
        and not model.enforce_eager
        and _single_device(execution, include_dcp=True)
    )


def _qwen3_rope_kv_preferred(
    model: ModelSignature,
    execution: ExecutionSignature,
) -> bool:
    geometry = (
        model.hidden_size,
        model.intermediate_size,
        model.num_hidden_layers,
        model.num_attention_heads,
        model.num_key_value_heads,
        model.head_dim,
    )
    return (
        model.architectures == ("Qwen3ForCausalLM",)
        and model.model_type == "qwen3"
        and geometry
        in {
            (1024, 3072, 28, 16, 8, 128),
            (4096, 12288, 36, 32, 8, 128),
        }
        and model.dtype == "bfloat16"
        and model.quantization in (None, "none")
        and not execution.has_quant_config
        and not execution.has_speculative_config
        and execution.has_parallel_config
        and _cache_supports_fused_qwen_attention(execution)
        and not model.enforce_eager
        and _single_device(execution, include_dcp=True)
    )


def _qwen3_dense_fp8_fusions_preferred(
    model: ModelSignature,
    execution: ExecutionSignature,
) -> bool:
    return (
        model.architectures == ("Qwen3ForCausalLM",)
        and model.model_type == "qwen3"
        and model.has_routed_experts is False
        and (
            model.hidden_size,
            model.intermediate_size,
            model.num_hidden_layers,
        )
        == (4096, 12288, 36)
        and model.quantization == "fp8"
        and model.dtype == "bfloat16"
        and not execution.has_speculative_config
        and _single_device(execution, include_dcp=False)
    )


def _qwen35_gdn_prefill_preferred(
    model: ModelSignature,
    execution: ExecutionSignature,
) -> bool:
    return (
        model.family is ModelFamily.QWEN35_36
        and model.has_routed_experts is False
        and model.dtype == "bfloat16"
        and model.gdn_conv_width == 4
        and model.gdn_conv_dim == 10240
        and execution.has_parallel_config
        and execution.tensor_parallel_size == 1
    )


def _qwen35_moe_prefill_preferred(
    model: ModelSignature,
    execution: ExecutionSignature,
) -> bool:
    return (
        model.family is ModelFamily.QWEN35_36
        and model.has_routed_experts is True
        and model.dtype == "bfloat16"
        and model.hidden_size == 2048
        and model.num_experts == 256
        and model.num_experts_per_tok == 8
        and model.moe_intermediate_size == 512
        and execution.has_parallel_config
        and execution.tensor_parallel_size == 4
        and execution.pipeline_parallel_size == 1
    )


def resolve_qwen_contract(
    model: ModelSignature,
    execution: ExecutionSignature,
) -> MusaOptimizationContract | None:
    architectures = set(model.outer_architectures or model.architectures)
    if "CosyVoice3Model" in architectures or model.model_type == "cosyvoice3":
        family = ModelFamily.QWEN2
        role = ModelRole.COSYVOICE_TALKER
    elif architectures & _QWEN35_ARCHITECTURES or model.model_type in {
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }:
        family = ModelFamily.QWEN35_36
        role = ModelRole.TEXT
    elif (
        architectures
        & {
            "Qwen3ForCausalLM",
            "Qwen3MoeForCausalLM",
        }
        or model.model_type == "qwen3"
    ):
        family = ModelFamily.QWEN3
        role = ModelRole.TEXT
    elif (
        architectures
        & {
            "Qwen2ForCausalLM",
            "Qwen2MoeForCausalLM",
        }
        or model.model_type == "qwen2"
    ):
        family = ModelFamily.QWEN2
        role = ModelRole.TEXT
    else:
        return None

    model = replace(model, family=family, role=role)
    preferred: set[OptimizationFeature] = set()
    if _has_architecture(model, QWEN_V2_SAMPLING_ARCHITECTURES):
        preferred.add(OptimizationFeature.QWEN_V2_SAMPLING)
    if (
        _has_architecture(model, QWEN_LEGACY_SAMPLING_ARCHITECTURES)
        and not execution.has_speculative_config
        and not execution.is_pooling_model
    ):
        preferred.add(OptimizationFeature.QWEN_LEGACY_SAMPLING)
    if _has_architecture(model, QWEN_FA3_ARCHITECTURES):
        preferred.add(OptimizationFeature.QWEN_FA3_SCHEDULER)
        if (
            execution.has_parallel_config
            and not execution.has_speculative_config
            and _single_device(execution, include_dcp=True)
            and execution.max_num_seqs is not None
            and execution.max_num_seqs > 0
        ):
            preferred.add(OptimizationFeature.QWEN_FA3_SINGLE_REQUEST_METADATA)
    if _qwen2_rope_kv_preferred(model, execution):
        preferred.add(OptimizationFeature.QWEN2_ROPE_KV_PRESPLIT)
    if _qwen3_rope_kv_preferred(model, execution):
        preferred.add(OptimizationFeature.QWEN3_QK_ROPE_KV_PRESPLIT)
    if _qwen3_dense_fp8_fusions_preferred(model, execution):
        preferred.add(OptimizationFeature.QWEN3_DENSE_FP8_POST_GRAD_FUSIONS)
    if _qwen35_gdn_prefill_preferred(model, execution):
        preferred.add(OptimizationFeature.QWEN35_GDN_WIDTH4_PREFILL)
    if _qwen35_moe_prefill_preferred(model, execution):
        preferred.add(OptimizationFeature.QWEN35_MOE_BF16_PREFILL)
    if model.family is ModelFamily.QWEN35_36:
        if model.has_routed_experts is True:
            preferred.add(OptimizationFeature.QWEN35_SHARED_EXPERT_FOLD)
        if model.dtype == "bfloat16":
            preferred.add(OptimizationFeature.QWEN35_INTERLEAVED_MROPE_QK)

    if model.vocab_size in (151936, 152064, 248320):
        if OptimizationFeature.QWEN_V2_SAMPLING in preferred:
            preferred.update(
                {
                    OptimizationFeature.QWEN_V2_GUMBEL,
                    OptimizationFeature.QWEN_UNIFORM_DECODE_VIEWS,
                    OptimizationFeature.QWEN_UNIFORM_SAMPLE_COUNTS,
                    OptimizationFeature.QWEN_SAMPLE_INPUT_VIEWS,
                }
            )
        if OptimizationFeature.QWEN_LEGACY_SAMPLING in preferred:
            preferred.update(
                {
                    OptimizationFeature.QWEN_LEGACY_GUMBEL,
                    OptimizationFeature.QWEN_TP_LOGITS_IPC_GATHER,
                }
            )
        if (
            OptimizationFeature.QWEN_LEGACY_SAMPLING in preferred
            and execution.tensor_parallel_size == 4
            and execution.pipeline_parallel_size == 1
        ):
            preferred.add(OptimizationFeature.QWEN_TP4_SHARDED_GUMBEL)

    profile = f"{family.value}.{'moe' if model.has_routed_experts else role.value}"
    return MusaOptimizationContract(
        model=model,
        execution=execution,
        profile=profile,
        # Keep the two sets independent even while the first Qwen rollout
        # promotes every proven feature automatically. Future providers can
        # expose a supported-but-not-yet-preferred implementation safely.
        supported_features=frozenset(preferred),
        preferred_features=frozenset(preferred),
    )
