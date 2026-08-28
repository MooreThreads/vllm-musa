from __future__ import annotations

from .types import ExecutionSignature, ModelSignature, RuntimePlan


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
    """Match the layer-bound Qwen3.5/3.6-35B-A3B prefill signature."""

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


def matches_qwen35_moe_bf16_decode_gemv_layer(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    global_num_experts: int | None,
    *,
    max_tokens: int,
) -> bool:
    """Match the TP4-local Qwen3.5/3.6 BF16 decode GEMV shape."""

    def dtype_name(value) -> str:
        return str(value).lower().removeprefix("torch.")

    try:
        folded = global_num_experts == 257
        unfolded = global_num_experts == 256
        expected_experts = 257 if folded else 256
        expected_top_k = 9 if folded else 8
        return (
            (folded or unfolded)
            and hidden_states.ndim == 2
            and dtype_name(hidden_states.dtype) == "bfloat16"
            and dtype_name(w1.dtype) == "bfloat16"
            and dtype_name(w2.dtype) == "bfloat16"
            and 0 < hidden_states.shape[0] <= max_tokens
            and hidden_states.shape[1] == 2048
            and tuple(w1.shape) == (expected_experts, 256, 2048)
            and tuple(w2.shape) == (expected_experts, 2048, 128)
            and topk_weights.ndim == 2
            and topk_ids.ndim == 2
            and topk_weights.shape == topk_ids.shape
            and topk_ids.shape[0] == hidden_states.shape[0]
            and topk_ids.shape[1] == expected_top_k
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        return False


def resolve_qwen_plan(
    model: ModelSignature,
    execution: ExecutionSignature,
) -> RuntimePlan | None:
    """Resolve Qwen defaults from the versioned declarative profile."""

    from .declarative import resolve_declarative_runtime_plan

    return resolve_declarative_runtime_plan(model, execution, identifier="qwen")
