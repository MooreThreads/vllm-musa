"""The Ovis rotary route installs only on complete, selected vision stacks."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torchada")

import torch  # noqa: E402
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.rotary_embedding.common import (  # noqa: E402
    ApplyRotaryEmb,
)

import vllm_musa.optimization_contract as optimization_contract  # noqa: E402
from vllm_musa.optimization_contract import OptimizationFeature  # noqa: E402
from vllm_musa.optimization_contract.rotary import (  # noqa: E402
    MusaVisionApplyRotaryEmb,
    MusaVisionRotaryPositions,
    install_model_rotary,
    install_vision_rotary,
)


@pytest.fixture(autouse=True)
def _default_vllm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


def _visual(block_count: int = 12) -> SimpleNamespace:
    return SimpleNamespace(
        blocks=[
            SimpleNamespace(
                attn=SimpleNamespace(
                    apply_rotary_emb=ApplyRotaryEmb(enforce_enable=True)
                )
            )
            for _ in range(block_count)
        ]
    )


def _install(visual: SimpleNamespace) -> bool:
    return install_vision_rotary(
        (block.attn for block in visual.blocks),
        expected_blocks=12,
        inplace=True,
        flatten=True,
        required_bf16_neox_shape=(64, 32),
    )


def test_ovis_replacement_is_block_local_and_shares_one_cache() -> None:
    visual, unrelated = _visual(), _visual()
    assert _install(visual)
    adapters = [block.attn.apply_rotary_emb for block in visual.blocks]
    assert all(isinstance(adapter, MusaVisionApplyRotaryEmb) for adapter in adapters)
    assert all(
        adapter.positions_cache is adapters[0].positions_cache for adapter in adapters
    )
    assert all(adapter.inplace and adapter.flatten for adapter in adapters)
    assert all(adapter.required_bf16_neox_shape == (64, 32) for adapter in adapters)
    assert all(
        isinstance(block.attn.apply_rotary_emb, ApplyRotaryEmb)
        and not isinstance(block.attn.apply_rotary_emb, MusaVisionApplyRotaryEmb)
        for block in unrelated.blocks
    )


def test_partial_vision_stack_fails_closed_without_replacement() -> None:
    for visual in (_visual(11), _visual(12)):
        if len(visual.blocks) == 12:
            visual.blocks[5].attn.apply_rotary_emb = torch.nn.Identity()
        originals = [block.attn.apply_rotary_emb for block in visual.blocks]
        assert not _install(visual)
        assert all(
            block.attn.apply_rotary_emb is original
            for block, original in zip(visual.blocks, originals)
        )


def test_positions_cache_reuses_buffer_without_persisting_it() -> None:
    cache = MusaVisionRotaryPositions()
    device = torch.device("cpu")
    positions = cache.get(2, 3, device)
    assert positions.tolist() == [0, 1, 2, 0, 1, 2]
    assert cache.get(2, 3, device) is positions
    assert cache.get(2, 4, device) is not positions
    assert cache.state_dict() == {}


def test_positions_cache_reuses_captured_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    class MusaDevice:
        type = "musa"

    cache = MusaVisionRotaryPositions()
    device = MusaDevice()
    arange = torch.arange
    monkeypatch.setattr(torch, "arange", lambda end, **kwargs: arange(end))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    pinned = cache.get(2, 3, device)
    assert cache.get(2, 3, device) is pinned


def test_positions_cache_probe_failure_falls_back_to_eager_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MusaDevice:
        type = "musa"

    cache = MusaVisionRotaryPositions()
    device = MusaDevice()
    arange = torch.arange
    monkeypatch.setattr(torch, "arange", lambda end, **kwargs: arange(end))

    def raising_probe() -> bool:
        raise RuntimeError("capture probe unavailable")

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", raising_probe)
    positions = cache.get(2, 3, device)
    assert cache.get(2, 3, device) is positions
    assert cache._pinned == {}


def test_ovis_install_respects_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.platforms as platforms

    monkeypatch.setattr(
        platforms,
        "current_platform",
        SimpleNamespace(is_musa=lambda: True, is_cpu=lambda: False),
    )
    selected = SimpleNamespace(
        preferred_features={OptimizationFeature.QWEN35_VISION_ROTARY_BF16}
    )
    monkeypatch.setattr(
        optimization_contract, "resolve_optimization_contract", lambda _: selected
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_5"))
    )
    assert install_model_rotary(SimpleNamespace(visual=_visual()), config)
    incomplete = _visual()
    del incomplete.blocks[5].attn
    assert not install_model_rotary(SimpleNamespace(visual=incomplete), config)
    selected.preferred_features = frozenset()
    assert not install_model_rotary(SimpleNamespace(visual=_visual()), config)
