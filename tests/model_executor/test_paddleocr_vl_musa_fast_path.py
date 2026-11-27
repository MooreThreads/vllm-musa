"""Minimal fail-closed tests for PaddleOCR-VL RoPE selection."""

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_musa.optimization_contract import (
    ModelFamily,
    OptimizationFeature,
    resolve_optimization_contract,
)


@pytest.fixture(autouse=True)
def _musa_runtime(monkeypatch):
    monkeypatch.setattr(torch.version, "musa", "5.2.0", raising=False)


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


def _contract(config, text_config=None):
    return resolve_optimization_contract(
        model_config=SimpleNamespace(hf_config=config, hf_text_config=text_config)
    )


def test_exact_geometry_selects_paddle_path():
    contract = _contract(_config())
    assert contract.model.family is ModelFamily.PADDLEOCR_VL
    assert contract.prefers(OptimizationFeature.PADDLEOCR_VL_ROTARY)


def test_other_model_does_not_import_rotary(monkeypatch):
    config = _config()
    config.model_type = "llama"
    monkeypatch.setitem(sys.modules, "vllm_musa.optimization_contract.rotary", None)
    assert not _contract(config).prefers(OptimizationFeature.PADDLEOCR_VL_ROTARY)


@pytest.mark.parametrize(
    ("owner", "field", "value"),
    [
        (None, "model_type", "qwen2_vl"),
        ("vision_config", "hidden_size", 1024),
        ("vision_config", "num_hidden_layers", 26),
        ("text_config", "hidden_size", 2048),
        ("text_config", "rope_parameters", {"mrope_section": [16, 24, 23]}),
    ],
)
def test_geometry_or_family_mismatch_stays_disabled(owner, field, value):
    config = _config()
    setattr(getattr(config, owner) if owner else config, field, value)
    assert not _contract(config).prefers(OptimizationFeature.PADDLEOCR_VL_ROTARY)


def test_raw_hf_geometry_wins_over_normalized_text():
    assert _contract(_config(), SimpleNamespace(hidden_size=2048)).prefers(
        OptimizationFeature.PADDLEOCR_VL_ROTARY
    )


def test_missing_musa_stays_disabled(monkeypatch):
    monkeypatch.setattr(torch.version, "musa", None, raising=False)
    assert not _contract(_config()).prefers(OptimizationFeature.PADDLEOCR_VL_ROTARY)
