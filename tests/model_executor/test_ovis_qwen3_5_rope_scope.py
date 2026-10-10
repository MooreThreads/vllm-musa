"""Behavioral scope checks for the Qwen3.5 vision RoPE route."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torchada")
import torch  # noqa: E402

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.rotary_embedding.common import (  # noqa: E402
    ApplyRotaryEmb,
)
from vllm_musa.optimization_contract.qwen import (  # noqa: E402
    install_qwen35_vision_rotary,
)
from vllm_musa.optimization_contract.rotary import (  # noqa: E402
    MusaVisionApplyRotaryEmb,
)


def _model_config(dtype=torch.bfloat16) -> SimpleNamespace:
    hf = SimpleNamespace(
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
    return SimpleNamespace(hf_config=hf, architectures=hf.architectures, dtype=dtype)


def _visual(blocks=12) -> SimpleNamespace:
    def block() -> SimpleNamespace:
        rotary = ApplyRotaryEmb(
            enforce_enable=True,
            is_neox_style=False,
            enable_fp32_compute=True,
        )
        return SimpleNamespace(attn=SimpleNamespace(apply_rotary_emb=rotary))

    return SimpleNamespace(blocks=[block() for _ in range(blocks)])


@pytest.fixture(autouse=True)
def _vllm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


def test_installs_twelve_visual_layers_with_shared_graph_positions() -> None:
    visual = _visual()
    assert install_qwen35_vision_rotary(visual)
    layers = [block.attn.apply_rotary_emb for block in visual.blocks]
    assert all(isinstance(layer, MusaVisionApplyRotaryEmb) for layer in layers)
    assert all(layer.positions_cache is layers[0].positions_cache for layer in layers)
    assert layers[0].required_bf16_neox_shape == (64, 32)
    assert all(layer.is_neox_style is False for layer in layers)
    assert all(layer.enable_fp32_compute is True for layer in layers)


def test_wrong_block_count_preserves_all_original_layers() -> None:
    visual = _visual(11)
    original = [block.attn.apply_rotary_emb for block in visual.blocks]
    assert not install_qwen35_vision_rotary(visual)
    assert all(
        block.attn.apply_rotary_emb is rotary
        for block, rotary in zip(visual.blocks, original)
    )


@pytest.fixture
def patched_hook():
    module = pytest.importorskip("vllm.model_executor.models.qwen3_5")
    if not hasattr(module, "_enable_musa_qwen35_vision_rope"):
        pytest.skip("Ovis patch is not applied")
    return module._enable_musa_qwen35_vision_rope


def test_upstream_hook_is_disabled_on_cpu(patched_hook, monkeypatch) -> None:
    import vllm.platforms

    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(is_musa=lambda: False)
    )
    visual = _visual()
    original = visual.blocks[0].attn.apply_rotary_emb
    patched_hook(_model_config(), visual)
    assert visual.blocks[0].attn.apply_rotary_emb is original


def test_upstream_hook_respects_contract_mismatch(patched_hook, monkeypatch) -> None:
    import vllm.platforms

    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(is_musa=lambda: True)
    )
    visual = _visual()
    original = visual.blocks[0].attn.apply_rotary_emb
    patched_hook(_model_config(torch.float16), visual)
    assert visual.blocks[0].attn.apply_rotary_emb is original
