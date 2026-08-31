import logging

import pytest

# isort: off
import torchada  # noqa: F401
import torch
# isort: on

from vllm_musa import tuning
from vllm_musa.model_executor.layers.fused_moe import fused_moe

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="requires MUSA",
)


def _fp8_constant(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.full(shape, 0x20, dtype=torch.uint8, device="musa").view(
        torch.float8_e4m3fn
    )


def test_mp60_dsv4_m2_dispatches_stage_tactic_and_matches_legacy(
    monkeypatch,
    caplog,
):
    hardware = tuning.prime_musa_kernel_hardware(0)
    if hardware != tuning.MusaKernelHardware((3, 1), 60):
        pytest.skip(f"requires exact MP60 S5000, got {hardware.cache_key}")

    monkeypatch.delenv("VLLM_MUSA_FUSED_MOE_DISPATCH", raising=False)
    monkeypatch.delenv("VLLM_MUSA_GEMV_MOE_BLOCK", raising=False)
    torch.manual_seed(20260828)
    hidden = torch.randn((2, 4096), dtype=torch.bfloat16, device="musa") * 0.01
    w1 = _fp8_constant((256, 512, 4096))
    w2 = _fp8_constant((256, 4096, 256))
    w1_scale = torch.ones((256, 4, 32), dtype=torch.float32, device="musa")
    w2_scale = torch.ones((256, 32, 2), dtype=torch.float32, device="musa")
    topk_ids = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]],
        dtype=torch.int32,
        device="musa",
    )
    topk_weights = torch.full((2, 6), 1.0 / 6.0, dtype=torch.float32, device="musa")
    kwargs = {
        "hidden_states": hidden,
        "w1": w1,
        "w2": w2,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "activation": "silu",
        "use_fp8_w8a8": True,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "block_shape": [128, 128],
    }

    legacy = fused_moe.fused_experts_impl(
        **kwargs,
        inplace=False,
        _allow_deepgemm_prefill=False,
        _gemv_block=(16, 8),
    )
    with caplog.at_level(logging.INFO):
        production = fused_moe._musa_fused_experts_impl_dispatch(**kwargs)
    torch.musa.synchronize()

    assert production.shape == legacy.shape == (2, 4096)
    assert torch.isfinite(production).all()
    torch.testing.assert_close(production, legacy, rtol=0, atol=0)
    assert "dsv4-fp8-m2-route-policy-v1" in caplog.text
    assert "blocks=(w1=(8, 16),w2=(16, 8))" in caplog.text
