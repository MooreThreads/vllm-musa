"""Minimal fail-closed tests for the Qwen3.5 vision RoPE contract."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torchada")

from vllm_musa.optimization_contract import (  # noqa: E402
    ModelFamily,
    OptimizationFeature,
    resolve_optimization_contract,
)


def _hf_config():
    return SimpleNamespace(
        architectures=["Qwen3_5ForConditionalGeneration"],
        model_type="qwen3_5",
        text_config=SimpleNamespace(
            model_type="qwen3_5_text",
            hidden_size=1024,
            intermediate_size=3584,
            num_hidden_layers=24,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=256,
            vocab_size=248320,
        ),
        vision_config=SimpleNamespace(
            hidden_size=768,
            depth=12,
            num_heads=12,
            out_hidden_size=1024,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
        ),
    )


def _contract(hf=None, dtype="bfloat16"):
    hf = hf or _hf_config()
    return resolve_optimization_contract(
        model_config=SimpleNamespace(
            hf_config=hf,
            hf_text_config=SimpleNamespace(model_type="qwen3_5_text", hidden_size=2048),
            architectures=hf.architectures,
            dtype=dtype,
        )
    )


def test_exact_raw_geometry_enables_only_qwen35_vision_feature():
    contract = _contract()
    assert contract.model.family is ModelFamily.QWEN35_36
    assert contract.prefers(OptimizationFeature.QWEN35_VISION_ROTARY_BF16)


@pytest.mark.parametrize(
    ("owner", "field", "value"),
    [
        (None, "architectures", ["Qwen3_5MoeForConditionalGeneration"]),
        (None, "model_type", "qwen3_5_moe"),
        ("text_config", "model_type", "qwen3_5_moe_text"),
        ("text_config", "hidden_size", 2048),
        ("vision_config", "depth", 24),
        ("vision_config", "num_heads", 8),
    ],
)
def test_raw_hf_mismatch_fails_closed(owner, field, value):
    hf = _hf_config()
    setattr(getattr(hf, owner) if owner else hf, field, value)
    assert not _contract(hf).prefers(OptimizationFeature.QWEN35_VISION_ROTARY_BF16)


@pytest.mark.parametrize("dtype", ["float16", "torch.float32", None])
def test_non_bf16_fails_closed(dtype):
    assert not _contract(dtype=dtype).prefers(
        OptimizationFeature.QWEN35_VISION_ROTARY_BF16
    )


def test_vision_alias_does_not_bypass_exact_gate():
    hf = _hf_config()
    hf.vision_config.hidden_size = None
    hf.vision_config.embed_dim = 768
    assert not _contract(hf).prefers(OptimizationFeature.QWEN35_VISION_ROTARY_BF16)
