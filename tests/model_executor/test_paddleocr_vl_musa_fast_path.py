"""Behavioral scope checks for Paddle's model-local RoPE selection."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torchada")
import torch


@pytest.fixture
def paddle_gate(monkeypatch):
    module = pytest.importorskip("vllm.model_executor.models.paddleocr_vl")
    monkeypatch.setattr(torch.version, "musa", "5.2.0", raising=False)
    return module._musa_paddle_rotary_enabled


def _config():
    return SimpleNamespace(
        model_type="paddleocr_vl",
        vision_config=SimpleNamespace(
            hidden_size=1152,
            num_hidden_layers=27,
            num_attention_heads=16,
            patch_size=14,
            image_size=384,
        ),
        text_config=SimpleNamespace(
            hidden_size=1024,
            num_attention_heads=16,
            num_key_value_heads=2,
            rope_parameters={"mrope_section": [16, 24, 24]},
        ),
    )


def test_exact_geometry_selects_paddle_path(paddle_gate):
    assert paddle_gate(_config())


@pytest.mark.parametrize(
    ("owner", "field", "value"),
    [
        (None, "model_type", "qwen2_vl"),
        ("vision_config", "hidden_size", 1024),
        ("vision_config", "num_hidden_layers", 26),
        ("vision_config", "num_attention_heads", 12),
        ("vision_config", "patch_size", 16),
        ("vision_config", "image_size", 448),
        ("text_config", "hidden_size", 2048),
        ("text_config", "num_attention_heads", 8),
        ("text_config", "num_key_value_heads", 4),
        ("text_config", "rope_parameters", {"mrope_section": [16, 24, 23]}),
        ("text_config", "rope_parameters", {"mrope_section": None}),
    ],
)
def test_other_model_or_geometry_stays_on_original_path(
    paddle_gate, owner, field, value
):
    config = _config()
    setattr(getattr(config, owner) if owner else config, field, value)
    assert not paddle_gate(config)


def test_no_musa_stays_on_original_path(paddle_gate, monkeypatch):
    monkeypatch.setattr(torch.version, "musa", None, raising=False)
    assert not paddle_gate(_config())


def test_depth_alias_and_outer_rope_fallback(paddle_gate):
    config = _config()
    config.vision_config.num_hidden_layers = 26
    config.vision_config.depth = 27
    config.text_config.rope_parameters = None
    config.rope_parameters = {"mrope_section": (16, 24, 24)}
    assert paddle_gate(config)


def test_rope_scaling_fallback(paddle_gate):
    config = _config()
    config.text_config.rope_parameters = {"rope_type": "default"}
    config.text_config.rope_scaling = {"mrope_section": [16, 24, 24]}
    assert paddle_gate(config)
    config.text_config.rope_scaling = {"rope_type": "default"}
    config.rope_scaling = {"mrope_section": [16, 24, 24]}
    assert paddle_gate(config)
    config.text_config.rope_parameters = {"mrope_section": [16, 24, 23]}
    assert not paddle_gate(config)
