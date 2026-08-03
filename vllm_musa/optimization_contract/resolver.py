from __future__ import annotations

from dataclasses import replace
from typing import Any

from .providers import CONTRACT_PROVIDERS
from .types import (
    ExecutionSignature,
    ModelFamily,
    ModelRole,
    ModelSignature,
    MusaOptimizationContract,
    OptimizationFeature,
)


def _normalize_dtype(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).lower().removeprefix("torch.")


def _normalize_name(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).lower().split(".")[-1]


def _int_tuple(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return None
    return tuple(value)


def _int_attr(config: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _float_attr(config: Any, *names: str) -> float | None:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _str_attr(config: Any, *names: str) -> str | None:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return str(value).lower()
    return None


def _text_config(model_config: Any) -> Any:
    text_config = getattr(model_config, "hf_text_config", None)
    if text_config is not None:
        return text_config
    hf_config = getattr(model_config, "hf_config", None)
    return getattr(hf_config, "text_config", hf_config)


def _architectures(model_config: Any, text_config: Any) -> tuple[str, ...]:
    values = getattr(model_config, "architectures", None) or None
    if not values:
        hf_config = getattr(model_config, "hf_config", None)
        values = getattr(hf_config, "architectures", None) or None
    if not values:
        values = getattr(text_config, "architectures", None) or ()
    return tuple(str(value) for value in values or ())


def _outer_model_type(model_config: Any, text_config: Any) -> str | None:
    value = getattr(model_config, "model_type", None)
    if value is None:
        hf_config = getattr(model_config, "hf_config", None)
        value = getattr(hf_config, "model_type", None)
    if value is None:
        value = getattr(text_config, "model_type", None)
    return str(value) if value is not None else None


def _text_architectures(text_config: Any) -> tuple[str, ...]:
    values = getattr(text_config, "architectures", None) or ()
    return tuple(str(value) for value in values or ())


def _quantization(model_config: Any, text_config: Any) -> str | None:
    value = getattr(model_config, "quantization", None)
    if value is not None:
        return str(value).lower()
    quantization_config = getattr(text_config, "quantization_config", None)
    if isinstance(quantization_config, dict):
        value = quantization_config.get("quant_method")
        return str(value).lower() if value is not None else None
    return None


def _has_routed_experts(model_config: Any, text_config: Any) -> bool | None:
    is_model_moe = getattr(model_config, "is_model_moe", None)
    if callable(is_model_moe):
        try:
            return bool(is_model_moe())
        except (AttributeError, TypeError, ValueError):
            return None
    if bool(getattr(model_config, "is_moe", False)):
        return True
    expert_values = [
        getattr(text_config, name, None)
        for name in (
            "num_experts",
            "moe_num_experts",
            "n_routed_experts",
            "num_local_experts",
        )
    ]
    if any(value is not None for value in expert_values):
        return any(bool(value) for value in expert_values)
    return False


def _gdn_conv_signature(text_config: Any) -> tuple[int | None, int | None]:
    width = _int_attr(text_config, "linear_conv_kernel_dim")
    key_heads = _int_attr(text_config, "linear_num_key_heads")
    value_heads = _int_attr(text_config, "linear_num_value_heads")
    key_dim = _int_attr(text_config, "linear_key_head_dim")
    value_dim = _int_attr(text_config, "linear_value_head_dim")
    if None in (key_heads, value_heads, key_dim, value_dim):
        return width, None
    return width, 2 * key_heads * key_dim + value_heads * value_dim


def _model_signature(model_config: Any, vllm_config: Any | None) -> ModelSignature:
    text_config = _text_config(model_config)
    gdn_width, gdn_dim = _gdn_conv_signature(text_config)
    architectures = _architectures(model_config, text_config)
    quant_config = getattr(vllm_config, "quant_config", None)
    quantization_config = getattr(text_config, "quantization_config", None)
    quant_block_shape = _int_tuple(getattr(quant_config, "weight_block_size", None))
    if quant_block_shape is None and isinstance(quantization_config, dict):
        quant_block_shape = _int_tuple(
            quantization_config.get("weight_block_size")
            or quantization_config.get("weight_block_shape")
        )
    uses_mla = getattr(model_config, "use_mla", None)
    if not isinstance(uses_mla, bool):
        uses_mla = getattr(text_config, "use_mla", None)
    if not isinstance(uses_mla, bool):
        uses_mla = getattr(text_config, "kv_lora_rank", None) is not None
    is_hybrid = getattr(model_config, "is_hybrid", None)
    if callable(is_hybrid):
        try:
            is_hybrid = bool(is_hybrid())
        except (AttributeError, TypeError, ValueError):
            is_hybrid = None
    if not isinstance(is_hybrid, bool) and gdn_dim is not None:
        is_hybrid = True
    return ModelSignature(
        family=ModelFamily.UNKNOWN,
        role=ModelRole.UNKNOWN,
        architectures=architectures,
        model_type=getattr(text_config, "model_type", None),
        dtype=_normalize_dtype(getattr(model_config, "dtype", None)),
        quantization=_quantization(model_config, text_config),
        hidden_size=_int_attr(text_config, "hidden_size"),
        intermediate_size=_int_attr(text_config, "intermediate_size"),
        num_hidden_layers=_int_attr(text_config, "num_hidden_layers"),
        num_attention_heads=_int_attr(text_config, "num_attention_heads"),
        num_key_value_heads=_int_attr(text_config, "num_key_value_heads"),
        head_dim=_int_attr(text_config, "head_dim"),
        vocab_size=_int_attr(text_config, "vocab_size"),
        num_experts=_int_attr(
            text_config,
            "num_experts",
            "moe_num_experts",
            "n_routed_experts",
            "num_local_experts",
        ),
        num_experts_per_tok=_int_attr(
            text_config,
            "num_experts_per_tok",
            "num_experts_per_token",
            "top_k",
        ),
        num_shared_experts=_int_attr(
            text_config,
            "num_shared_experts",
            "n_shared_experts",
        ),
        moe_intermediate_size=_int_attr(text_config, "moe_intermediate_size"),
        expert_dtype=_str_attr(text_config, "expert_dtype"),
        hidden_act=_str_attr(text_config, "hidden_act"),
        swiglu_limit=_float_attr(text_config, "swiglu_limit"),
        gdn_conv_width=gdn_width,
        gdn_conv_dim=gdn_dim,
        has_routed_experts=_has_routed_experts(model_config, text_config),
        enforce_eager=bool(getattr(model_config, "enforce_eager", False)),
        outer_architectures=architectures,
        text_architectures=_text_architectures(text_config),
        outer_model_type=_outer_model_type(model_config, text_config),
        uses_mla=uses_mla,
        index_topk=_int_attr(text_config, "index_topk"),
        quant_block_shape=quant_block_shape,
        is_hybrid=is_hybrid if isinstance(is_hybrid, bool) else None,
    )


def _execution_signature(
    vllm_config: Any | None,
    *,
    is_pooling_model: bool,
) -> ExecutionSignature:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    attention_config = getattr(vllm_config, "attention_config", None)
    compilation_config = getattr(vllm_config, "compilation_config", None)

    def size(name: str) -> int:
        value = getattr(parallel_config, name, 1)
        return value if isinstance(value, int) and value > 0 else 1

    block_size = getattr(cache_config, "block_size", None)
    try:
        import vllm.envs as vllm_envs

        batch_invariant_enabled = bool(vllm_envs.VLLM_BATCH_INVARIANT)
    except (AttributeError, ImportError):
        batch_invariant_enabled = False

    return ExecutionSignature(
        tensor_parallel_size=size("tensor_parallel_size"),
        pipeline_parallel_size=size("pipeline_parallel_size"),
        data_parallel_size=size("data_parallel_size"),
        decode_context_parallel_size=size("decode_context_parallel_size"),
        has_speculative_config=(
            getattr(vllm_config, "speculative_config", None) is not None
        ),
        has_quant_config=getattr(vllm_config, "quant_config", None) is not None,
        is_pooling_model=is_pooling_model,
        has_parallel_config=parallel_config is not None,
        cache_dtype=_normalize_dtype(getattr(cache_config, "cache_dtype", "auto")),
        cache_block_size=(
            block_size
            if isinstance(block_size, int) and not isinstance(block_size, bool)
            else None
        ),
        max_num_seqs=_int_attr(scheduler_config, "max_num_seqs"),
        attention_backend=_normalize_name(getattr(attention_config, "backend", None)),
        compilation_mode=_normalize_name(getattr(compilation_config, "mode", None)),
        cudagraph_mode=_normalize_name(
            getattr(compilation_config, "cudagraph_mode", None)
        ),
        batch_invariant_enabled=batch_invariant_enabled,
    )


def resolve_optimization_contract(
    vllm_config: Any | None = None,
    *,
    model_config: Any | None = None,
    is_pooling_model: bool = False,
) -> MusaOptimizationContract:
    if model_config is None:
        model_config = getattr(vllm_config, "model_config", None)
    execution = _execution_signature(
        vllm_config,
        is_pooling_model=is_pooling_model,
    )
    if model_config is None:
        model = ModelSignature(
            family=ModelFamily.UNKNOWN,
            role=ModelRole.UNKNOWN,
            architectures=(),
            model_type=None,
            dtype=None,
            quantization=None,
            hidden_size=None,
            intermediate_size=None,
            num_hidden_layers=None,
            num_attention_heads=None,
            num_key_value_heads=None,
            head_dim=None,
            vocab_size=None,
            num_experts=None,
            num_experts_per_tok=None,
            num_shared_experts=None,
            moe_intermediate_size=None,
            expert_dtype=None,
            hidden_act=None,
            swiglu_limit=None,
            gdn_conv_width=None,
            gdn_conv_dim=None,
            has_routed_experts=False,
            enforce_eager=False,
            outer_architectures=(),
            text_architectures=(),
            outer_model_type=None,
            uses_mla=None,
            index_topk=None,
            quant_block_shape=None,
            is_hybrid=None,
        )
    else:
        model = _model_signature(model_config, vllm_config)

    for provider in CONTRACT_PROVIDERS:
        contract = provider(model, execution)
        if contract is not None:
            break
    else:
        contract = MusaOptimizationContract(
            model=model,
            execution=execution,
            profile="unknown",
            supported_features=frozenset(),
            preferred_features=frozenset(),
        )

    if model.is_hybrid is True:
        feature = OptimizationFeature.HYBRID_SEPARATE_MAMBA_POOL
        contract = replace(
            contract,
            supported_features=contract.supported_features | {feature},
            preferred_features=contract.preferred_features | {feature},
        )
    return contract


def prefers_optimization(owner: Any, feature: OptimizationFeature) -> bool:
    contract = getattr(owner, "_musa_optimization_contract", None)
    return contract is not None and feature in getattr(
        contract, "preferred_features", ()
    )


def bind_optimization_contract(
    owner: Any,
    vllm_config: Any | None = None,
    *,
    model_config: Any | None = None,
    is_pooling_model: bool = False,
) -> MusaOptimizationContract:
    """Resolve once and bind an immutable contract to a runtime owner."""

    contract = (
        getattr(vllm_config, "_musa_optimization_contract", None)
        if vllm_config is not None
        else None
    )
    if not isinstance(contract, MusaOptimizationContract):
        contract = resolve_optimization_contract(
            vllm_config,
            model_config=model_config,
            is_pooling_model=is_pooling_model,
        )
        if vllm_config is not None:
            vllm_config._musa_optimization_contract = contract
    owner._musa_optimization_contract = contract
    return contract
