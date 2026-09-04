import logging

import pytest

# isort: off
import torchada  # noqa: F401
import torch
# isort: on

from vllm_musa import deepseek_v4_mhc as mhc
from vllm_musa import tuning

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="requires MUSA",
)


def _reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x_fp32 = x.float()
    return (
        x_fp32
        * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        * weight.float()
    ).to(x.dtype)


@pytest.mark.parametrize("use_graph", [False, True])
def test_mp48_m5_weighted_rmsnorm_tactic_numeric_and_replay(
    monkeypatch,
    caplog,
    use_graph,
):
    hardware = tuning.prime_musa_kernel_hardware(0)
    if hardware != tuning.MusaKernelHardware((3, 1), 48):
        pytest.skip(f"requires exact MP48 S5000, got {hardware.cache_key}")

    torch.manual_seed(20260828)
    x = torch.randn((5, 4096), dtype=torch.bfloat16, device="musa")
    weight = torch.randn((4096,), dtype=torch.bfloat16, device="musa")
    expected = _reference(x, weight)

    # Compile the exact 256-thread variant before graph capture.
    mhc._LOGGED_WEIGHTED_RMSNORM_TACTICS.clear()
    with caplog.at_level(logging.INFO, logger=mhc.__name__):
        output = mhc._try_mhc_weighted_rms_norm_musa(x, weight, 1e-6)
    assert output is not None
    torch.musa.synchronize()
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
    assert "threads=256" in caplog.text

    if not use_graph:
        monkeypatch.setattr(
            torch.musa,
            "get_device_properties",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("provider must only read frozen hardware")
            ),
        )
        output = mhc._try_mhc_weighted_rms_norm_musa(x, weight, 1e-6)
        assert output is not None
        torch.musa.synchronize()
        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
        return

    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        output = mhc._try_mhc_weighted_rms_norm_musa(x, weight, 1e-6)
        assert output is not None
    for seed in (7, 17, 20260826):
        torch.manual_seed(seed)
        x.copy_(torch.randn_like(x))
        expected = _reference(x, weight)
        graph.replay()
        torch.musa.synchronize()
        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


def test_mp48_weighted_rmsnorm_compiling_guard(monkeypatch):
    hardware = tuning.prime_musa_kernel_hardware(0)
    if hardware != tuning.MusaKernelHardware((3, 1), 48):
        pytest.skip(f"requires exact MP48 S5000, got {hardware.cache_key}")

    x = torch.randn((5, 4096), dtype=torch.bfloat16, device="musa")
    weight = torch.randn((4096,), dtype=torch.bfloat16, device="musa")
    expected = _reference(x, weight)

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        mhc,
        "select_mhc_weighted_rmsnorm_tactic",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("Dynamo tracing must retain the legacy tactic")
        ),
    )
    output = mhc._try_mhc_weighted_rms_norm_musa(x, weight, 1e-6)
    assert output is not None
    torch.musa.synchronize()
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
