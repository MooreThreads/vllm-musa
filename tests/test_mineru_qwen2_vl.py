# SPDX-License-Identifier: Apache-2.0
"""Minimal fail-closed tests for the MinerU Qwen2-VL route."""

from types import SimpleNamespace

import pytest

from vllm_musa.optimization_contract import (
    ModelFamily,
    ModelRole,
    OptimizationFeature,
    resolve_optimization_contract,
    rotary,
)


def _raw_config():
    return SimpleNamespace(
        model_type="qwen2_vl",
        architectures=["Qwen2VLForConditionalGeneration"],
        text_config=SimpleNamespace(
            model_type="qwen2_vl",
            hidden_size=896,
            intermediate_size=4864,
            num_hidden_layers=24,
            num_attention_heads=14,
            num_key_value_heads=2,
            head_dim=64,
            rope_scaling={"mrope_section": [8, 12, 12]},
        ),
        vision_config=SimpleNamespace(embed_dim=1280, depth=32, num_heads=16),
    )


def _vllm_config(raw=None, normalized_text=None, *, tp=1):
    raw = raw or _raw_config()
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=raw,
            hf_text_config=normalized_text or raw.text_config,
            architectures=raw.architectures,
            dtype="bfloat16",
            enforce_eager=True,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, pipeline_parallel_size=1
        ),
    )


def test_exact_raw_geometry_enables_mineru_feature():
    raw = _raw_config()
    contract = resolve_optimization_contract(_vllm_config(raw))
    assert contract.model.family is ModelFamily.QWEN2
    assert contract.model.role is ModelRole.TEXT
    assert contract.prefers(OptimizationFeature.MINERU_QWEN2_VL_ROTARY)


@pytest.mark.parametrize(
    ("owner", "field", "value"),
    [
        (None, "model_type", "qwen2_5_vl"),
        ("text_config", "hidden_size", 1024),
        ("vision_config", "depth", 24),
        ("vision_config", "num_heads", 20),
    ],
)
def test_family_or_geometry_mismatch_fails_closed(owner, field, value):
    raw = _raw_config()
    setattr(getattr(raw, owner) if owner else raw, field, value)
    contract = resolve_optimization_contract(_vllm_config(raw))
    assert contract.model.family is ModelFamily.UNKNOWN
    assert not contract.prefers(OptimizationFeature.MINERU_QWEN2_VL_ROTARY)


def test_contract_requires_exact_architecture():
    raw = _raw_config()
    raw.architectures = ["OtherForConditionalGeneration"]
    contract = resolve_optimization_contract(_vllm_config(raw))
    assert contract.model.family is ModelFamily.UNKNOWN
    assert not contract.prefers(OptimizationFeature.MINERU_QWEN2_VL_ROTARY)


def test_mrope_section_mismatch_fails_closed():
    raw = _raw_config()
    raw.text_config.rope_scaling = {"mrope_section": [8, 12, 11]}
    assert not resolve_optimization_contract(_vllm_config(raw)).prefers(
        OptimizationFeature.MINERU_QWEN2_VL_ROTARY
    )


def test_raw_hf_config_wins_over_normalized_text():
    raw = _raw_config()
    normalized = SimpleNamespace(model_type="qwen2", hidden_size=1024)
    assert resolve_optimization_contract(
        _vllm_config(raw, normalized_text=normalized)
    ).prefers(OptimizationFeature.MINERU_QWEN2_VL_ROTARY)


def test_tp2_does_not_enable_unvalidated_mineru_route():
    contract = resolve_optimization_contract(_vllm_config(tp=2))
    assert not contract.prefers(OptimizationFeature.MINERU_QWEN2_VL_ROTARY)


def test_install_route_uses_shared_rotary_dispatcher(monkeypatch):
    from vllm import platforms

    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(is_musa=lambda: True)
    )
    calls = {}

    def record_vision(attentions, **kwargs):
        calls["vision"] = (tuple(attentions), kwargs)
        return True

    def record_language(layers, **kwargs):
        calls["language"] = (tuple(layers), kwargs)

    monkeypatch.setattr(rotary, "install_vision_rotary", record_vision)
    monkeypatch.setattr(rotary, "install_mrope_rotary", record_language)
    blocks = [SimpleNamespace(attn=object()) for _ in range(32)]
    layers = [SimpleNamespace(self_attn=object())]
    owner = SimpleNamespace(
        visual=SimpleNamespace(blocks=blocks),
        language_model=SimpleNamespace(model=SimpleNamespace(layers=layers)),
    )

    assert rotary.install_model_rotary(owner, _vllm_config())
    assert calls["vision"][0] == tuple(block.attn for block in blocks)
    assert calls["vision"][1]["expected_blocks"] == 32
    assert calls["language"][0] == tuple(layers)
    assert calls["language"][1]["qk_hidden_sizes"] == (896, 128)
