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

_DEEPSEEK_V41_ARCHITECTURES = ("DeepseekV41ForCausalLM",)
_DEEPSEEK_V41_QUANTIZATION = frozenset({"fp8", "deepseek_v4_fp8"})
_DEEPSEEK_V41_W4AFP8_QUANTIZATION = "w4afp8"


def _claims(model: ModelSignature) -> bool:
    architectures = model.outer_architectures or model.architectures
    return (
        "DeepseekV41ForCausalLM" in architectures
        or model.model_type == "deepseek_v41"
        or model.outer_model_type == "deepseek_v41"
    )


def _has_complete_identity(model: ModelSignature) -> bool:
    architectures = model.outer_architectures or model.architectures
    return (
        architectures == _DEEPSEEK_V41_ARCHITECTURES
        and model.model_type == "deepseek_v41"
        and model.outer_model_type == "deepseek_v41"
    )


def _matches_flash_fp8_shape(model: ModelSignature) -> bool:
    return (
        model.dtype == "bfloat16"
        and model.quantization in _DEEPSEEK_V41_QUANTIZATION
        and model.hidden_size == 5120
        and model.num_hidden_layers == 40
        and model.num_attention_heads == 64
        and model.num_key_value_heads == 1
        and model.head_dim == 512
        and model.vocab_size == 129280
        and model.num_experts == 384
        and model.num_experts_per_tok == 6
        and model.num_shared_experts == 1
        and model.moe_intermediate_size == 2304
        and model.expert_dtype == "fp8"
        and model.hidden_act == "silu"
        and model.swiglu_limit == 10.0
        and model.quant_block_shape == (32, 32)
        and model.has_routed_experts is True
        and model.uses_mla is True
        and model.index_topk == 512
        and model.is_hybrid is False
    )


def _matches_w4afp8_shape(model: ModelSignature) -> bool:
    return (
        model.dtype == "bfloat16"
        and model.quantization == _DEEPSEEK_V41_W4AFP8_QUANTIZATION
        and model.hidden_size == 5120
        and model.num_hidden_layers == 40
        and model.num_attention_heads == 64
        and model.num_key_value_heads == 1
        and model.head_dim == 512
        and model.vocab_size == 129280
        and model.num_experts == 384
        and model.num_experts_per_tok == 6
        and model.num_shared_experts == 1
        and model.moe_intermediate_size == 2304
        and model.expert_dtype in ("int4", "fp4")
        and model.hidden_act == "silu"
        and model.swiglu_limit == 10.0
        and model.quant_block_shape == (32, 32)
        and model.has_routed_experts is True
        and model.uses_mla is True
        and model.index_topk == 512
        and model.is_hybrid is False
    )


def _matches_tp8(execution: ExecutionSignature) -> bool:
    if execution.has_speculative_config and execution.speculative_method != "dspark":
        return False
    return (
        execution.has_parallel_config
        and execution.tensor_parallel_size == 8
        and execution.pipeline_parallel_size == 1
        and execution.data_parallel_size == 1
        and execution.decode_context_parallel_size == 1
        and execution.has_quant_config
        and not execution.is_pooling_model
        and not execution.batch_invariant_enabled
    )


def _matches_w4afp8_overlap_execution(execution: ExecutionSignature) -> bool:
    return (
        _matches_tp8(execution)
        and execution.enable_expert_parallel
        and not execution.enable_eplb
        and not execution.enable_dbo
    )


def resolve_deepseek_v41_contract(
    model: ModelSignature,
    execution: ExecutionSignature,
) -> MusaOptimizationContract | None:
    if not _claims(model):
        return None

    model = replace(model, family=ModelFamily.DEEPSEEK_V41, role=ModelRole.TEXT)
    supported: set[OptimizationFeature] = set()
    preferred: set[OptimizationFeature] = set()
    flash_fp8 = _has_complete_identity(model) and _matches_flash_fp8_shape(model)
    w4afp8 = _has_complete_identity(model) and _matches_w4afp8_shape(model)

    if flash_fp8:
        # The folded shared expert rides the small-M FP8 MoE kernel after the
        # native expert block is refined to the 32x32 TP8 shard shape. DSpark
        # drafts build their MoE layers with the target model config, so they
        # resolve this same contract.
        supported.add(OptimizationFeature.DEEPSEEK_V41_SHARED_EXPERT_FOLD)
        if _matches_tp8(execution):
            preferred.add(OptimizationFeature.DEEPSEEK_V41_SHARED_EXPERT_FOLD)

    if w4afp8:
        feature = OptimizationFeature.DEEPSEEK_V41_W4AFP8_SHARED_EXPERT_OVERLAP
        supported.add(feature)
        if _matches_w4afp8_overlap_execution(execution):
            preferred.add(feature)

    if flash_fp8 and _matches_tp8(execution):
        profile = (
            "deepseek_v4.1.tp8_flash_dspark"
            if execution.has_speculative_config
            else "deepseek_v4.1.tp8_flash"
        )
    else:
        profile = "deepseek_v4.1.unvalidated"
    return MusaOptimizationContract(
        model=model,
        execution=execution,
        profile=profile,
        supported_features=frozenset(supported),
        preferred_features=frozenset(preferred),
    )
