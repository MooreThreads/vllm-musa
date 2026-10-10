from types import SimpleNamespace

import pytest
import torch

from vllm_musa.optimization_contract import (
    ModelFamily,
    ModelRole,
    OptimizationFeature,
    resolve_optimization_contract,
)


def _hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["Qwen3_5ForConditionalGeneration"],
        model_type="qwen3_5",
        text_config=SimpleNamespace(
            model_type="qwen3_5_text", hidden_size=1024,
            intermediate_size=3584, num_hidden_layers=24,
            num_attention_heads=8, num_key_value_heads=2,
            head_dim=256, vocab_size=248320,
        ),
        vision_config=SimpleNamespace(
            hidden_size=768, depth=12, num_heads=12,
            out_hidden_size=1024, patch_size=16,
            spatial_merge_size=2, temporal_patch_size=2,
        ),
    )


def _contract(hf_config=None, dtype=torch.bfloat16):
    hf_config = hf_config or _hf_config()
    model_config = SimpleNamespace(
        hf_config=hf_config,
        # vLLM may flatten a different text config; the feature uses raw HF.
        hf_text_config=SimpleNamespace(model_type="qwen3_5_text", hidden_size=2048),
        architectures=hf_config.architectures,
        dtype=dtype,
    )
    return resolve_optimization_contract(model_config=model_config)


def test_exact_raw_hf_geometry_adds_only_qwen35_vision_feature() -> None:
    contract = _contract()
    assert contract.model.family is ModelFamily.QWEN35_36
    assert contract.model.role is ModelRole.TEXT
    assert contract.prefers(OptimizationFeature.QWEN35_VISION_ROTARY_BF16)
    assert contract.prefers(OptimizationFeature.QWEN35_INTERLEAVED_MROPE_QK)


@pytest.mark.parametrize(
    ("owner", "name", "value"),
    [
        (None, "architectures", ["Qwen3_5MoeForConditionalGeneration"]),
        (None, "model_type", "qwen3_5_moe"),
        ("text_config", "model_type", "qwen3_5_moe_text"),
        ("text_config", "hidden_size", 2048),
        ("text_config", "intermediate_size", 4096),
        ("text_config", "num_hidden_layers", 25),
        ("text_config", "num_attention_heads", 16),
        ("text_config", "num_key_value_heads", 4),
        ("text_config", "head_dim", 128),
        ("text_config", "vocab_size", 151936),
        ("vision_config", "hidden_size", 1024),
        ("vision_config", "depth", 24),
        ("vision_config", "num_heads", 8),
        ("vision_config", "out_hidden_size", 768),
        ("vision_config", "patch_size", 14),
        ("vision_config", "spatial_merge_size", 1),
        ("vision_config", "temporal_patch_size", 1),
    ],
)
def test_each_raw_hf_mismatch_fails_closed(owner, name, value) -> None:
    config = _hf_config()
    setattr(getattr(config, owner) if owner else config, name, value)
    assert not _contract(config).prefers(
        OptimizationFeature.QWEN35_VISION_ROTARY_BF16
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, "BFLOAT16", None])
def test_non_bf16_fails_closed(dtype) -> None:
    assert not _contract(dtype=dtype).prefers(
        OptimizationFeature.QWEN35_VISION_ROTARY_BF16
    )


def test_vision_field_alias_does_not_bypass_exact_gate() -> None:
    config = _hf_config()
    config.vision_config.hidden_size = None
    config.vision_config.embed_dim = 768
    assert not _contract(config).prefers(
        OptimizationFeature.QWEN35_VISION_ROTARY_BF16
    )
