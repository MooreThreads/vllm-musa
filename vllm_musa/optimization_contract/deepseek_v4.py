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

_DEEPSEEK_V4_ARCHITECTURES = ("DeepseekV4ForCausalLM",)
_DEEPSEEK_V4_QUANTIZATION = frozenset({"fp8", "deepseek_v4_fp8"})


def _has_complete_identity(model: ModelSignature) -> bool:
    architectures = model.outer_architectures or model.architectures
    return (
        architectures == _DEEPSEEK_V4_ARCHITECTURES
        and model.model_type == "deepseek_v4"
    )


def _matches_flash_base(model: ModelSignature) -> bool:
    return (
        _has_complete_identity(model)
        and model.dtype == "bfloat16"
        and model.quantization in _DEEPSEEK_V4_QUANTIZATION
        and model.hidden_size == 4096
        and model.num_hidden_layers == 43
        and model.num_attention_heads == 64
        and model.num_key_value_heads == 1
        and model.head_dim == 512
        and model.vocab_size == 129280
        and model.num_experts == 256
        and model.num_experts_per_tok == 6
        and model.num_shared_experts == 1
        and model.moe_intermediate_size == 2048
        and model.expert_dtype == "fp8"
        and model.hidden_act == "silu"
        and model.swiglu_limit == 10.0
        and model.quant_block_shape == (128, 128)
        and model.has_routed_experts is True
        and model.uses_mla is True
        and model.index_topk == 512
        and model.is_hybrid is False
    )


def _matches_tp8_reference(execution: ExecutionSignature) -> bool:
    return (
        execution.has_parallel_config
        and execution.tensor_parallel_size == 8
        and execution.pipeline_parallel_size == 1
        and execution.data_parallel_size == 1
        and execution.decode_context_parallel_size == 1
        and not execution.has_speculative_config
        and not execution.is_pooling_model
        and execution.cache_dtype == "fp8"
        and execution.max_num_seqs == 1
        and execution.attention_backend == "flashmla"
        and execution.compilation_mode == "none"
        and execution.cudagraph_mode == "full_decode_only"
    )


def resolve_deepseek_v4_contract(
    model: ModelSignature,
    execution: ExecutionSignature,
) -> MusaOptimizationContract | None:
    architectures = model.outer_architectures or model.architectures
    if (
        "DeepseekV4ForCausalLM" not in architectures
        and model.model_type != "deepseek_v4"
    ):
        return None

    model = replace(model, family=ModelFamily.DEEPSEEK_V4, role=ModelRole.TEXT)
    supported: set[OptimizationFeature] = set()
    preferred: set[OptimizationFeature] = set()

    if _matches_flash_base(model):
        supported.update(
            {
                OptimizationFeature.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8,
                OptimizationFeature.DEEPSEEK_V4_NATIVE_SPARSE_INDEXER,
                OptimizationFeature.DEEPSEEK_V4_MATERIALIZED_PREFILL_INDEXER,
                OptimizationFeature.DEEPSEEK_V4_TP8_FLASHMLA_SPARSE_PAGE256,
                OptimizationFeature.DEEPSEEK_V4_TP8_FUSED_ADD_RMSNORM_BLOCK256,
            }
        )
        if execution.has_parallel_config:
            preferred.update(
                {
                    OptimizationFeature.DEEPSEEK_V4_NATIVE_SPARSE_INDEXER,
                    OptimizationFeature.DEEPSEEK_V4_MATERIALIZED_PREFILL_INDEXER,
                }
            )
        if _matches_tp8_reference(execution):
            preferred.update(
                {
                    OptimizationFeature.DEEPSEEK_V4_TP8_FLASHMLA_SPARSE_PAGE256,
                    OptimizationFeature.DEEPSEEK_V4_TP8_FUSED_ADD_RMSNORM_BLOCK256,
                }
            )
            if not execution.batch_invariant_enabled:
                preferred.add(OptimizationFeature.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)

    profile = (
        "deepseek_v4.tp8_flash_base"
        if _matches_flash_base(model) and _matches_tp8_reference(execution)
        else "deepseek_v4.unvalidated"
    )
    return MusaOptimizationContract(
        model=model,
        execution=execution,
        profile=profile,
        supported_features=frozenset(supported),
        preferred_features=frozenset(preferred),
    )
