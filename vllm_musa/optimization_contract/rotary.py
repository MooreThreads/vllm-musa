"""Shared adapters for model-selected MUSA rotary paths."""

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

from vllm_musa.jit_kernel import rotary_embedding
from vllm_musa.utils.environ import envs

from .types import ModelFamily, ModelRole, MusaOptimizationContract, OptimizationFeature

logger = init_logger(__name__)


class MusaVisionRotaryPositions(nn.Module):
    """Keep positions alive when a visual rotary call is graph captured."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_positions", None, persistent=False)
        self._key: tuple[int, int, torch.device] | None = None
        self._pinned: dict[tuple[int, int, torch.device], torch.Tensor] = {}

    def get(self, batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
        try:
            capturing = device.type == "musa" and bool(
                torch.cuda.is_current_stream_capturing()
            )
        except (AttributeError, RuntimeError):
            capturing = False
        key = (batch, seq_len, device)
        if key in self._pinned:
            return self._pinned[key]
        if self._positions is not None and self._key == key:
            if capturing:
                self._pinned[key] = self._positions
            return self._positions
        positions = torch.arange(seq_len, device=device).repeat(batch)
        if capturing:
            self._pinned[key] = positions
        else:
            self._positions, self._key = positions, key
        return positions


class MusaVisionApplyRotaryEmb(ApplyRotaryEmb):
    """Adapt a vision x/cos/sin call to the existing MUSA rotary kernel.

    ``inplace=True`` writes into the contiguous working tensor; for contiguous
    inputs, that tensor aliases and mutates ``x``.
    """

    def __init__(
        self,
        *,
        is_neox_style: bool,
        enable_fp32_compute: bool,
        inplace: bool = False,
        flatten: bool = False,
        positions_cache: MusaVisionRotaryPositions | None = None,
        required_bf16_neox_shape: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(
            enforce_enable=True,
            is_neox_style=is_neox_style,
            enable_fp32_compute=enable_fp32_compute,
        )
        self.inplace = inplace
        self.flatten = flatten
        self.positions_cache = positions_cache
        self.required_bf16_neox_shape = required_bf16_neox_shape

    def forward_oot(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        if x.device.type != "musa" or envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get():
            return self.forward_native(x, cos, sin)
        if self.required_bf16_neox_shape is not None:
            head_size, cos_width = self.required_bf16_neox_shape
            if (
                x.dtype != torch.bfloat16
                or x.shape[-1] != head_size
                or cos.ndim != 2
                or cos.shape[-1] != cos_width
                or sin.shape != cos.shape
                or not self.is_neox_style
                or self.enable_fp32_compute
            ):
                logger.info_once(
                    "MUSA vision rotary skipped: unsupported input geometry"
                )
                return self.forward_native(x, cos, sin)
        x, cos, sin, shape, dtype = self._pre_process(x, cos, sin)
        output = x.contiguous() if self.inplace else x.clone()
        batch, seq_len, heads, head_size = output.shape
        cache = torch.cat((cos, sin), dim=-1).to(output.dtype).contiguous()
        if self.flatten:
            positions = (
                self.positions_cache.get(batch, seq_len, output.device)
                if self.positions_cache is not None
                else torch.arange(seq_len, device=output.device).repeat(batch)
            )
            query = output.reshape(batch * seq_len, heads * head_size)
        else:
            positions = torch.arange(
                seq_len, device=output.device, dtype=torch.long
            ).expand(batch, -1)
            query = output
        rotary_embedding(positions, query, None, head_size, cache, self.is_neox_style)
        return self._post_process(output, shape, dtype)


class MusaMRotaryEmbedding(nn.Module):
    """Use upstream MRoPE except for the validated MUSA 2-D-position case."""

    def __init__(
        self, inner: nn.Module, qk_hidden_sizes: tuple[int, int] | None = None
    ) -> None:
        super().__init__()
        self.inner = inner
        self.qk_hidden_sizes = qk_hidden_sizes

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        args = (
            (positions, query, key)
            if offsets is None
            else (positions, query, key, offsets)
        )
        if envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get():
            return self.inner.forward_native(*args)
        if (
            query.device.type == "musa"
            and positions.ndim == 2
            and key is not None
            and (
                self.qk_hidden_sizes is None
                or (query.shape[-1], key.shape[-1]) == self.qk_hidden_sizes
            )
        ):
            return self.inner.forward_cuda(*args)
        return self.inner(*args)


def install_vision_rotary(
    attentions: Iterable[nn.Module | None],
    *,
    inplace: bool = False,
    flatten: bool = False,
    expected_blocks: int | None = None,
    required_bf16_neox_shape: tuple[int, int] | None = None,
) -> bool:
    """Install one adapter across a complete, already-selected vision stack."""
    active = tuple(attention for attention in attentions if attention is not None)
    if (
        not active
        or (expected_blocks is not None and len(active) != expected_blocks)
        or any(
            not isinstance(getattr(attention, "apply_rotary_emb", None), ApplyRotaryEmb)
            for attention in active
        )
    ):
        return False
    positions = MusaVisionRotaryPositions() if flatten else None
    for attention in active:
        rotary = attention.apply_rotary_emb
        attention.apply_rotary_emb = MusaVisionApplyRotaryEmb(
            is_neox_style=rotary.is_neox_style,
            enable_fp32_compute=rotary.enable_fp32_compute,
            inplace=inplace,
            flatten=flatten,
            positions_cache=positions,
            required_bf16_neox_shape=required_bf16_neox_shape,
        )
    return True


def install_mrope_rotary(
    layers: Iterable[nn.Module],
    *,
    mrope_section: tuple[int, ...] | None = None,
    qk_hidden_sizes: tuple[int, int] | None = None,
    head_size: int | None = None,
    rotary_dim: int | None = None,
    set_neox_style: bool = False,
) -> None:
    """Wrap eligible language RoPE modules with the shared MUSA adapter."""
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        rotary = getattr(attention, "rotary_emb", None)
        if (
            rotary is None
            or (
                mrope_section is not None
                and tuple(getattr(rotary, "mrope_section", ()) or ()) != mrope_section
            )
            or (
                head_size is not None
                and getattr(rotary, "head_size", None) != head_size
            )
            or (
                rotary_dim is not None
                and getattr(rotary, "rotary_dim", None) != rotary_dim
            )
        ):
            continue
        if set_neox_style:
            rotary.is_neox_style = True
        attention.rotary_emb = MusaMRotaryEmbedding(
            rotary, qk_hidden_sizes=qk_hidden_sizes
        )


def _paddleocr_vl_geometry(config: object) -> tuple[object, ...]:
    vision = getattr(config, "vision_config", None)
    text = getattr(config, "text_config", config)
    section = None
    for rope in (
        getattr(text, "rope_parameters", None),
        getattr(text, "rope_scaling", None),
        getattr(config, "rope_parameters", None),
        getattr(config, "rope_scaling", None),
    ):
        if isinstance(rope, dict) and rope.get("mrope_section") is not None:
            value = rope["mrope_section"]
            section = tuple(value) if isinstance(value, (list, tuple)) else None
            break
    return (
        getattr(vision, "hidden_size", None),
        getattr(vision, "num_hidden_layers", None),
        getattr(vision, "depth", None),
        getattr(vision, "num_attention_heads", None),
        getattr(vision, "patch_size", None),
        getattr(vision, "image_size", None),
        getattr(text, "hidden_size", None),
        getattr(text, "num_attention_heads", None),
        getattr(text, "num_key_value_heads", None),
        section,
    )


def _mineru_geometry(config: object) -> bool:
    text = getattr(config, "text_config", config)
    vision = getattr(config, "vision_config", None)
    if getattr(config, "model_type", None) != "qwen2_vl" or vision is None:
        return False
    if (
        getattr(text, "hidden_size", None),
        getattr(text, "num_hidden_layers", None),
        getattr(text, "num_attention_heads", None),
        getattr(text, "num_key_value_heads", None),
        getattr(text, "head_dim", None) or 64,
        getattr(vision, "embed_dim", None) or getattr(vision, "hidden_size", None),
        getattr(vision, "depth", None) or getattr(vision, "num_hidden_layers", None),
        getattr(vision, "num_heads", None)
        or getattr(vision, "num_attention_heads", None),
    ) != (896, 24, 14, 2, 64, 1280, 32, 16):
        return False
    section = None
    for owner in (config, text):
        section = getattr(owner, "mrope_section", None)
        if section is not None:
            break
        for name in ("rope_parameters", "rope_scaling"):
            rope = getattr(owner, name, None)
            if isinstance(rope, dict) and rope.get("mrope_section") is not None:
                section = rope["mrope_section"]
                break
        if section is not None:
            break
    try:
        return tuple(int(value) for value in section) == (8, 12, 12)
    except (TypeError, ValueError):
        return False


def _ovis_qwen35_vision_geometry(config: object, dtype: object) -> bool:
    """Match the validated OvisOCR2 Qwen3.5 vision/text geometry."""
    text = getattr(config, "text_config", None)
    vision = getattr(config, "vision_config", None)
    return (
        tuple(getattr(config, "architectures", ()) or ())
        == ("Qwen3_5ForConditionalGeneration",)
        and getattr(config, "model_type", None) == "qwen3_5"
        and getattr(text, "model_type", None) == "qwen3_5_text"
        and str(dtype) in ("bfloat16", "torch.bfloat16")
        and tuple(
            getattr(text, name, None)
            for name in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "vocab_size",
            )
        )
        == (1024, 3584, 24, 8, 2, 256, 248320)
        and tuple(
            getattr(vision, name, None)
            for name in (
                "hidden_size",
                "depth",
                "num_heads",
                "out_hidden_size",
                "patch_size",
                "spatial_merge_size",
                "temporal_patch_size",
            )
        )
        == (768, 12, 12, 1024, 16, 2, 2)
    )


def resolve_rotary_contract(
    contract: MusaOptimizationContract, model_config: object
) -> MusaOptimizationContract:
    """Select exact model rotary geometry without model rules in the resolver."""
    config = getattr(model_config, "hf_config", None)
    if getattr(config, "model_type", None) == "paddleocr_vl":
        model = replace(
            contract.model, family=ModelFamily.PADDLEOCR_VL, role=ModelRole.TEXT
        )
        geometry = _paddleocr_vl_geometry(config)
        enabled = (
            geometry[0] == 1152
            and 27 in geometry[1:3]
            and geometry[3:] == (16, 14, 384, 1024, 16, 2, (16, 24, 24))
            and getattr(torch.version, "musa", None) is not None
        )
        features = frozenset(
            {OptimizationFeature.PADDLEOCR_VL_ROTARY} if enabled else ()
        )
        return replace(
            contract,
            model=model,
            profile="paddleocr_vl",
            supported_features=features,
            preferred_features=features,
        )
    mineru = getattr(OptimizationFeature, "MINERU_QWEN2_VL_ROTARY", None)
    if (
        mineru is not None
        and contract.model.outer_model_type == "qwen2_vl"
        and contract.execution.tensor_parallel_size == 1
        and tuple(getattr(config, "architectures", ()) or ())
        == ("Qwen2VLForConditionalGeneration",)
        and _mineru_geometry(config)
    ):
        model = replace(contract.model, family=ModelFamily.QWEN2, role=ModelRole.TEXT)
        return replace(
            contract,
            model=model,
            profile=f"{ModelFamily.QWEN2.value}.{ModelRole.TEXT.value}",
            supported_features=contract.supported_features | {mineru},
            preferred_features=contract.preferred_features | {mineru},
        )
    qwen35 = getattr(OptimizationFeature, "QWEN35_VISION_ROTARY_BF16", None)
    if (
        qwen35 is None
        or contract.model.family is not ModelFamily.QWEN35_36
        or not _ovis_qwen35_vision_geometry(
            config, getattr(model_config, "dtype", None)
        )
    ):
        return contract
    return replace(
        contract,
        supported_features=contract.supported_features | {qwen35},
        preferred_features=contract.preferred_features | {qwen35},
    )


def install_model_rotary(owner: nn.Module, vllm_config: object) -> bool:
    """Select and install every supported model rotary route in one place."""
    from vllm.platforms import current_platform

    if not current_platform.is_musa():
        return False
    from . import OptimizationFeature, resolve_optimization_contract

    features = resolve_optimization_contract(vllm_config).preferred_features
    paddle = getattr(OptimizationFeature, "PADDLEOCR_VL_ROTARY", None)
    mineru = getattr(OptimizationFeature, "MINERU_QWEN2_VL_ROTARY", None)
    qwen35 = getattr(OptimizationFeature, "QWEN35_VISION_ROTARY_BF16", None)
    if paddle is not None and paddle in features:
        visual = getattr(getattr(owner, "visual", None), "vision_model", None)
        visual = getattr(getattr(visual, "encoder", None), "layers", ())
        vision = (
            (getattr(layer, "self_attn", None) for layer in visual),
            {"expected_blocks": 27},
        )
        language = (
            getattr(
                getattr(getattr(owner, "language_model", None), "model", None),
                "layers",
                (),
            ),
            {"set_neox_style": True},
        )
    elif mineru is not None and mineru in features:
        visual = getattr(owner, "visual", None)
        vision = (
            (getattr(block, "attn", None) for block in getattr(visual, "blocks", ())),
            {"expected_blocks": 32, "required_bf16_neox_shape": (80, 40)},
        )
        language = (
            getattr(
                getattr(getattr(owner, "language_model", None), "model", None),
                "layers",
                (),
            ),
            {
                "mrope_section": (8, 12, 12),
                "qk_hidden_sizes": (896, 128),
                "head_size": 64,
                "rotary_dim": 64,
            },
        )
    elif qwen35 is not None and qwen35 in features:
        visual = getattr(owner, "visual", None)
        vision = (
            (getattr(block, "attn", None) for block in getattr(visual, "blocks", ())),
            {
                "expected_blocks": 12,
                "inplace": True,
                "flatten": True,
                "required_bf16_neox_shape": (64, 32),
            },
        )
        language = None
    else:
        config = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
        if getattr(config, "model_type", None) == "qwen3_5":
            logger.info_once("MUSA Qwen3.5 vision rotary skipped: unsupported geometry")
        return False
    expected_blocks = vision[1]["expected_blocks"]
    if not install_vision_rotary(vision[0], **vision[1]):
        if qwen35 is not None and qwen35 in features:
            logger.info_once(
                "MUSA Qwen3.5 vision rotary skipped: incomplete visual blocks"
            )
        return False
    if language is not None:
        install_mrope_rotary(language[0], **language[1])
    if qwen35 is not None and qwen35 in features:
        logger.info_once(
            f"MUSA Qwen3.5 vision rotary installed on {expected_blocks} visual blocks"
        )
    return True
