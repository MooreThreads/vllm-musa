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
_DEEPSEEK_V4_MTP_ARCHITECTURES = ("DeepSeekV4MTPModel",)
_DEEPSEEK_V4_QUANTIZATION = frozenset({"fp8", "deepseek_v4_fp8"})


def _has_complete_identity(model: ModelSignature) -> bool:
    architectures = model.outer_architectures or model.architectures
    return (
        architectures == _DEEPSEEK_V4_ARCHITECTURES
        and model.model_type == "deepseek_v4"
    )


def _has_complete_mtp_draft_identity(model: ModelSignature) -> bool:
    architectures = model.outer_architectures or model.architectures
    return (
        architectures == _DEEPSEEK_V4_MTP_ARCHITECTURES
        and model.model_type == "deepseek_mtp"
        and model.outer_model_type == "deepseek_mtp"
    )


def _matches_flash_shape(model: ModelSignature) -> bool:
    return (
        model.dtype == "bfloat16"
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


def _matches_flash_base(model: ModelSignature) -> bool:
    return _has_complete_identity(model) and _matches_flash_shape(model)


def _matches_mtp_draft_flash_base(model: ModelSignature) -> bool:
    return _has_complete_mtp_draft_identity(model) and _matches_flash_shape(model)


def _matches_tp8_reference(
    execution: ExecutionSignature,
    *,
    allow_multi_batch: bool = False,
) -> bool:
    max_num_seqs_matches = execution.max_num_seqs == 1
    if allow_multi_batch:
        max_num_seqs_matches = (
            execution.max_num_seqs is not None and execution.max_num_seqs > 0
        )
    return (
        execution.has_parallel_config
        and execution.tensor_parallel_size == 8
        and execution.pipeline_parallel_size == 1
        and execution.data_parallel_size == 1
        and execution.decode_context_parallel_size == 1
        and not execution.has_speculative_config
        and execution.has_quant_config
        and not execution.is_pooling_model
        and execution.cache_dtype in {"fp8", "fp8_ds_mla"}
        and max_num_seqs_matches
        and execution.attention_backend == "flashmla"
        and execution.compilation_mode == "none"
        and execution.cudagraph_mode == "full_decode_only"
    )


def _matches_tp8_mtp(execution: ExecutionSignature) -> bool:
    if not execution.has_speculative_config:
        return False
    if execution.speculative_method not in ("mtp", "deepseek_mtp"):
        return False
    if execution.max_num_seqs is None or execution.max_num_seqs <= 1:
        return False
    return _matches_tp8_reference(
        replace(execution, has_speculative_config=False),
        allow_multi_batch=True,
    )


def _matches_tp8_mtp_car(execution: ExecutionSignature) -> bool:
    """Match target DSV4 MTP CAR, including the BS1 reference profile."""
    if not execution.has_speculative_config:
        return False
    if execution.speculative_method not in ("mtp", "deepseek_mtp"):
        return False
    return _matches_tp8_reference(
        replace(execution, has_speculative_config=False),
        allow_multi_batch=True,
    )


def _matches_tp8_dspark(execution: ExecutionSignature) -> bool:
    if not execution.has_speculative_config:
        return False
    if execution.speculative_method != "dspark":
        return False
    return _matches_tp8_reference(
        replace(execution, has_speculative_config=False),
        allow_multi_batch=True,
    )


def _matches_tp8_mtp_draft(execution: ExecutionSignature) -> bool:
    # The draft-local VllmConfig no longer contains the outer
    # speculative_config. Its exact DeepSeek-V4 MTP model role is the MTP
    # proof; retain every other execution guard.
    return (
        execution.max_num_seqs is not None
        and execution.max_num_seqs > 1
        and _matches_tp8_reference(
            replace(
                execution,
                has_speculative_config=False,
                speculative_method="",
            ),
            allow_multi_batch=True,
        )
    )


def resolve_deepseek_v4_contract(
    model: ModelSignature,
    execution: ExecutionSignature,
) -> MusaOptimizationContract | None:
    architectures = model.outer_architectures or model.architectures
    if (
        "DeepseekV4ForCausalLM" not in architectures
        and model.model_type != "deepseek_v4"
        and "DeepSeekV4MTPModel" not in architectures
        and model.model_type != "deepseek_mtp"
    ):
        return None

    is_mtp_draft = _has_complete_mtp_draft_identity(model)
    model = replace(
        model,
        family=ModelFamily.DEEPSEEK_V4,
        role=ModelRole.MTP_DRAFT if is_mtp_draft else ModelRole.TEXT,
    )
    supported: set[OptimizationFeature] = set()
    preferred: set[OptimizationFeature] = set()

    if _has_complete_identity(model):
        supported.add(OptimizationFeature.DEEPSEEK_V4_CAR_GRAPH_INPUT_CAPTURE_GUARD)

    if _matches_flash_base(model):
        supported.update(
            {
                OptimizationFeature.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8,
                OptimizationFeature.DEEPSEEK_V4_NATIVE_SPARSE_INDEXER,
                OptimizationFeature.DEEPSEEK_V4_MATERIALIZED_PREFILL_INDEXER,
                OptimizationFeature.DEEPSEEK_V4_TP8_FLASHMLA_SPARSE_PAGE256,
                OptimizationFeature.DEEPSEEK_V4_TP8_FUSED_ADD_RMSNORM_BLOCK256,
                OptimizationFeature.DEEPSEEK_V4_TP8_MTP_SPARSE_DIRECT_OUT,
                OptimizationFeature.DEEPSEEK_V4_TP8_MTP_SPARSE_PREFILL_HEADROOM,
                OptimizationFeature.DEEPSEEK_V4_TP8_MTP_ASYNC_PREFILL_QUEUE_FENCE,
            }
        )
        if execution.has_parallel_config:
            preferred.update(
                {
                    OptimizationFeature.DEEPSEEK_V4_NATIVE_SPARSE_INDEXER,
                    OptimizationFeature.DEEPSEEK_V4_MATERIALIZED_PREFILL_INDEXER,
                }
            )
        # The shared-expert clamp + FP8 down-projection is shape-safe for every
        # positive max-num-seqs in the same no-MTP TP8 decode contract.  The
        # original migration accidentally restricted this already-validated
        # path to the single-sequence capture experiment, disabling it for the
        # production capture ladder (e.g. max-num-seqs=64).
        if _matches_tp8_reference(execution, allow_multi_batch=True) and not (
            execution.batch_invariant_enabled
        ):
            preferred.add(OptimizationFeature.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8)

        if _matches_tp8_mtp(execution):
            preferred.update(
                {
                    OptimizationFeature.DEEPSEEK_V4_TP8_MTP_SPARSE_DIRECT_OUT,
                    OptimizationFeature.DEEPSEEK_V4_TP8_MTP_SPARSE_PREFILL_HEADROOM,
                }
            )
            if execution.async_scheduling:
                preferred.add(
                    OptimizationFeature.DEEPSEEK_V4_TP8_MTP_ASYNC_PREFILL_QUEUE_FENCE
                )
        if _matches_tp8_mtp_car(execution):
            supported.add(OptimizationFeature.DEEPSEEK_V4_MTP_CAR_GRAPH_STAGING_ARENA)
            preferred.add(OptimizationFeature.DEEPSEEK_V4_MTP_CAR_GRAPH_STAGING_ARENA)
            # A dedicated BS1 deployment keeps registered graph inputs for its
            # single capture descriptor. Multi-batch deployments use staging.
            if execution.max_num_seqs == 1:
                supported.add(
                    OptimizationFeature.DEEPSEEK_V4_MTP_CAR_GRAPH_REGISTERED_INPUTS
                )
                preferred.add(
                    OptimizationFeature.DEEPSEEK_V4_MTP_CAR_GRAPH_REGISTERED_INPUTS
                )
        elif _matches_tp8_reference(execution):
            preferred.update(
                {
                    OptimizationFeature.DEEPSEEK_V4_TP8_FLASHMLA_SPARSE_PAGE256,
                    OptimizationFeature.DEEPSEEK_V4_TP8_FUSED_ADD_RMSNORM_BLOCK256,
                }
            )

        if _matches_tp8_dspark(execution):
            # The scheduler pads a remote-prefilled request's final prompt
            # token to the verify shape; the DSV4 sparse backends classify
            # it as a decode, so it can replay the uniform decode graph.
            supported.add(
                OptimizationFeature.DEEPSEEK_V4_FINAL_PROMPT_TOKEN_DECODE_GRAPH
            )
            preferred.add(
                OptimizationFeature.DEEPSEEK_V4_FINAL_PROMPT_TOKEN_DECODE_GRAPH
            )

    if _matches_mtp_draft_flash_base(model):
        mtp_draft_features = {
            OptimizationFeature.DEEPSEEK_V4_TP8_MTP_SPARSE_DIRECT_OUT,
            OptimizationFeature.DEEPSEEK_V4_TP8_MTP_SPARSE_PREFILL_HEADROOM,
        }
        supported.update(mtp_draft_features)
        if _matches_tp8_mtp_draft(execution):
            preferred.update(mtp_draft_features)

    if _matches_mtp_draft_flash_base(model) and _matches_tp8_mtp_draft(execution):
        profile = "deepseek_v4.tp8_flash_mtp_draft"
    elif _matches_flash_base(model) and _matches_tp8_mtp(execution):
        profile = "deepseek_v4.tp8_flash_base_mtp"
    elif _matches_flash_base(model) and _matches_tp8_reference(execution):
        profile = "deepseek_v4.tp8_flash_base"
    elif _matches_flash_base(model) and _matches_tp8_dspark(execution):
        profile = "deepseek_v4.tp8_flash_base_dspark"
    else:
        profile = "deepseek_v4.unvalidated"
    return MusaOptimizationContract(
        model=model,
        execution=execution,
        profile=profile,
        supported_features=frozenset(supported),
        preferred_features=frozenset(preferred),
    )
