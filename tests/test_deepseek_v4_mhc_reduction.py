import pytest


@pytest.mark.parametrize("device_type", ["cpu", "musa"])
@pytest.mark.parametrize("num_tokens", [1, 4, 65, 4100])
@pytest.mark.parametrize("split_k", [1, 8, 16, 64])
@pytest.mark.parametrize("hidden_size", [4096, 5120, 7168])
def test_delayed_pre_mix_reduces_independent_tokens(
    device_type, num_tokens, split_k, hidden_size, monkeypatch
):
    pytest.importorskip("torchada")
    import torch

    import vllm_musa.deepseek_v4_mhc as mhc

    if device_type == "musa" and not torch.musa.is_available():
        pytest.skip("requires a MUSA device")

    generator = torch.Generator().manual_seed(100083)
    hc_mult = 4
    sqrsum_cpu = torch.full((split_k, num_tokens), 1e11)
    small_row = min(1, num_tokens - 1)
    sqrsum_cpu[:, small_row] = torch.linspace(1000, 14000, split_k)
    mixes_cpu = torch.randn(split_k, num_tokens, 24, generator=generator) * 0.1
    mixes_cpu[:, small_row, :hc_mult] = 0.1
    scale_cpu = torch.tensor([0.7, 1.0, 1.0])
    base_cpu = torch.linspace(-0.2, 0.2, 24)
    rms_eps, hc_pre_eps = 1e-20, 1e-6
    expected_rms = torch.rsqrt(
        sqrsum_cpu.double().sum(0) / (hc_mult * hidden_size) + rms_eps
    )
    expected_logits = (
        mixes_cpu.double()[:, :, :hc_mult].sum(0)
        * expected_rms.unsqueeze(-1)
        * scale_cpu[0]
        + base_cpu[:hc_mult]
    )
    expected_pre = torch.sigmoid(expected_logits) + hc_pre_eps

    residual = torch.zeros(
        1, hc_mult, hidden_size, dtype=torch.bfloat16, device=device_type
    ).expand(num_tokens, -1, -1)
    mixes = mixes_cpu.to(device_type)
    sqrsum = sqrsum_cpu.to(device_type)
    post = torch.zeros(num_tokens, hc_mult, device=device_type)
    comb = torch.zeros(num_tokens, hc_mult * hc_mult, device=device_type)

    def prenorm_outputs(*args, **kwargs):
        return post, comb, None, mixes, sqrsum

    monkeypatch.setattr(mhc, "_mhc_pre_deepgemm_big_fuse_run", prenorm_outputs)
    _, _, actual_pre = mhc._mhc_pre_delayed_big_fuse(
        residual,
        None,
        scale_cpu.to(device_type),
        base_cpu.to(device_type),
        rms_eps,
        hc_pre_eps,
        1e-6,
        1.0,
        20,
    )
    assert torch.isfinite(actual_pre.cpu()).all()
    torch.testing.assert_close(
        actual_pre.cpu().double(), expected_pre, rtol=1e-5, atol=1e-6
    )
