from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    torch = None

requires_musa = pytest.mark.skipif(
    torch is None or not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="requires a MUSA device",
)


@requires_musa
@pytest.mark.parametrize("tokens", [1, 257, 1030])
def test_collapse_matches_the_fp32_torch_expression(tokens) -> None:
    from vllm_musa.kernels.mhc_collapse import mhc_collapse

    torch.manual_seed(tokens)
    residual = torch.randn(tokens, 4, 5120).to(torch.bfloat16).to("musa")
    mix = (torch.rand(tokens, 4) + 0.5).to("musa")
    products = mix.unsqueeze(-1) * residual.float()
    for out, ref, scale in (
        (mhc_collapse(residual, mix), products.sum(dim=1), products.abs().sum(dim=1)),
        (
            mhc_collapse(residual),
            residual.float().mean(dim=1),
            residual.float().abs().mean(dim=1),
        ),
    ):
        assert out.dtype == torch.bfloat16 and out.shape == (tokens, 5120)
        # BF16 rounding of the FP32 sum, plus FP32 summation-order error.
        assert ((out.float() - ref).abs() <= ref.abs() * 2**-8 + scale * 2**-20).all()
        assert (out == ref.to(torch.bfloat16)).float().mean().item() > 0.99


@pytest.mark.parametrize("device", ["cpu", pytest.param("musa", marks=requires_musa)])
def test_the_off_switch_returns_the_torch_expression(device, monkeypatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm_musa.kernels import mhc_collapse as mc

    class NoLaunch:
        def __getitem__(self, grid):
            raise AssertionError("the Triton kernel ran with the switch off")

    # MUSA calls reach the kernel unless switched off; CPU calls never do.
    monkeypatch.setattr(mc, "_ENABLED", False)
    monkeypatch.setattr(mc, "_mhc_collapse_kernel", NoLaunch())
    torch.manual_seed(0)
    residual = torch.randn(7, 4, 320).to(torch.bfloat16).to(device)
    mix = (torch.rand(7, 4) + 0.5).to(device)
    out, mean = mc.mhc_collapse(residual, mix), mc.mhc_collapse(residual)
    assert out.dtype == mean.dtype == torch.bfloat16
    ref = (mix[..., None] * residual.float()).sum(1).to(residual.dtype)
    assert torch.equal(out, ref)
    assert torch.equal(mean, residual.float().mean(1).to(residual.dtype))
