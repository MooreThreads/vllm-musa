from __future__ import annotations

from dataclasses import replace
from typing import Any

from .qwen import resolve_qwen_contract
from .types import (
    ExecutionSignature,
    ModelFamily,
    ModelRole,
    ModelSignature,
    MusaOptimizationContract,
    OptimizationFeature,
)


_TEXT_SIGNATURE = (896, 24, 14, 2, 64)
_VISION_SIGNATURE = (1280, 32, 16, 80)
_MROPE_SECTION = (8, 12, 12)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _mrope_section(config: Any) -> tuple[int, ...] | None:
    for owner in (config, _get(config, "text_config")):
        for name in ("mrope_section",):
            value = _get(owner, name)
            if value is not None:
                return tuple(int(v) for v in value)
        for name in ("rope_parameters", "rope_scaling"):
            value = _get(owner, name)
            if isinstance(value, dict) and value.get("mrope_section") is not None:
                return tuple(int(v) for v in value["mrope_section"])
    return None


def is_mineru_qwen2_vl_config(config: Any) -> bool:
    """Use the raw outer HF config, including its own text_config, as before."""
    text = _get(config, "text_config", config)
    vision = _get(config, "vision_config")
    if vision is None or _get(config, "model_type") != "qwen2_vl":
        return False
    text_sig = (
        _get(text, "hidden_size"),
        _get(text, "num_hidden_layers"),
        _get(text, "num_attention_heads"),
        _get(text, "num_key_value_heads"),
        _get(text, "head_dim")
        or _get(text, "hidden_size") // _get(text, "num_attention_heads"),
    )
    vision_hidden = _get(vision, "embed_dim") or _get(vision, "hidden_size")
    vision_heads = _get(vision, "num_heads") or _get(vision, "num_attention_heads")
    vision_sig = (
        vision_hidden,
        _get(vision, "depth") or _get(vision, "num_hidden_layers"),
        vision_heads,
        vision_hidden // vision_heads,
    )
    return text_sig == _TEXT_SIGNATURE and vision_sig == _VISION_SIGNATURE and (
        _mrope_section(config) == _MROPE_SECTION
    )


def resolve_mineru_contract(
    model: ModelSignature,
    execution: ExecutionSignature,
) -> MusaOptimizationContract | None:
    if not model.mineru_qwen2_vl_config_match:
        return None

    # A nested Qwen2 text schema can already receive Qwen2 features. Keep
    # those exact feature decisions while adding the MinerU-only rotary path.
    existing = resolve_qwen_contract(model, execution)
    supported = set(existing.supported_features) if existing else set()
    preferred = set(existing.preferred_features) if existing else set()
    feature = OptimizationFeature.MINERU_QWEN2_VL_ROTARY
    supported.add(feature)
    preferred.add(feature)
    return MusaOptimizationContract(
        model=replace(model, family=ModelFamily.MINERU_QWEN2_VL, role=ModelRole.TEXT),
        execution=execution,
        profile="mineru_qwen2_vl.text_generation",
        supported_features=frozenset(supported),
        preferred_features=frozenset(preferred),
    )
