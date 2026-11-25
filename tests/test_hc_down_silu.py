# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MUSA fused Qwen4Exp mHC down projection + SiLU."""

import types

import pytest
import torch
import torch.nn.functional as F

from vllm_musa.kernels.hc_down_silu import (
    hc_down_silu,
    hc_down_silu_enabled,
    supports_hc_down_silu,
)

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="requires MUSA",
)


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    rank: int,
    hc: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    down = F.linear(x, weight).to(torch.bfloat16)
    lora = down[:, :rank]
    scaled = lora.float() / hc
    return (scaled * torch.sigmoid(scaled)).to(torch.bfloat16), down[
        :, rank : rank + hc
    ]


@pytest.mark.parametrize("num_tokens", [1])
@pytest.mark.parametrize(
    ("k", "rank", "hc", "padded_n"),
    [
        (10240, 320, 4, 336),
        (64, 8, 4, 28),
    ],
)
def test_hc_down_silu_matches_fallback(
    num_tokens: int,
    k: int,
    rank: int,
    hc: int,
    padded_n: int,
) -> None:
    torch.manual_seed(0)
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="musa")
    weight = torch.randn(padded_n, k, dtype=torch.bfloat16, device="musa")

    lora, injection = hc_down_silu(x, weight, rank, hc)
    expected_lora, expected_injection = _reference(x, weight, rank, hc)

    assert lora.shape == (num_tokens, rank)
    assert injection.shape == (num_tokens, hc)
    torch.testing.assert_close(lora, expected_lora, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(injection, expected_injection, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("num_tokens", [2, 3, 4, 5, 8, 16, 32, 48])
def test_hc_down_silu_falls_back_beyond_one_token(
    num_tokens: int,
) -> None:
    """Only M=1 is fused; the custom op must stay exact for larger M."""
    torch.manual_seed(0)
    k, rank, hc, padded_n = 10240, 320, 4, 336
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="musa")
    weight = torch.randn(padded_n, k, dtype=torch.bfloat16, device="musa")

    lora, injection = hc_down_silu(x, weight, rank, hc)
    expected_lora, expected_injection = _reference(x, weight, rank, hc)

    assert lora.shape == (num_tokens, rank)
    torch.testing.assert_close(lora, expected_lora, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(injection, expected_injection, rtol=2e-2, atol=2e-2)


def test_hc_down_silu_unpadded_output() -> None:
    """The fused output stops at rank + hc and never computes pad columns."""
    num_tokens, k, rank, hc = 1, 64, 8, 4
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="musa")
    weight = torch.randn(rank + hc + 12, k, dtype=torch.bfloat16, device="musa")
    lora, injection = hc_down_silu(x, weight, rank, hc)
    expected_lora, expected_injection = _reference(x, weight, rank, hc)
    torch.testing.assert_close(lora, expected_lora, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(injection, expected_injection, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("num_tokens", [1])
def test_hc_down_silu_float32_epilogue_reference(
    num_tokens: int,
) -> None:
    torch.manual_seed(1)
    k, rank, hc = 10240, 320, 4
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="musa")
    weight = torch.randn(rank + hc + 12, k, dtype=torch.bfloat16, device="musa")

    lora, injection = hc_down_silu(x, weight, rank, hc)

    down = (x.cpu().float() @ weight[: rank + hc].cpu().float().T).to(torch.bfloat16)
    scaled = down[:, :rank].float() / hc
    expected_lora = (scaled * torch.sigmoid(scaled)).to(torch.bfloat16)
    expected_injection = down[:, rank : rank + hc]

    torch.testing.assert_close(
        lora, expected_lora.to(lora.device), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        injection,
        expected_injection.to(injection.device),
        rtol=2e-2,
        atol=2e-2,
    )


def test_supports_hc_down_silu_rejects_fallbacks() -> None:
    x = torch.empty((1, 10240), dtype=torch.bfloat16, device="musa")
    weight = torch.empty((336, 10240), dtype=torch.bfloat16, device="musa")
    assert hc_down_silu_enabled()
    assert supports_hc_down_silu(x, weight, 320, 4)
    for num_tokens in (2, 3, 4, 8, 16, 32, 48):
        assert not supports_hc_down_silu(
            torch.empty(
                (num_tokens, 10240),
                dtype=torch.bfloat16,
                device="musa",
            ),
            weight,
            320,
            4,
        ), f"M={num_tokens} must stay on the unfused path"
    assert not supports_hc_down_silu(x.float(), weight, 320, 4)
    assert not supports_hc_down_silu(x, weight.float(), 320, 4)
    assert not supports_hc_down_silu(
        torch.empty((1, 10239), dtype=torch.bfloat16, device="musa"),
        torch.empty((336, 10239), dtype=torch.bfloat16, device="musa"),
        320,
        4,
    )


def test_gated_residual_uses_musa_fused_path() -> None:
    from vllm.models.qwen4_exp.nvidia.hyperconnection import GatedResidual

    num_tokens, k, rank, hc = 1, 64, 8, 4
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="musa")
    weight = torch.randn(rank + hc + 12, k, dtype=torch.bfloat16, device="musa")
    residual = types.SimpleNamespace(
        use_combine=True,
        _use_hc_down_silu=False,
        _use_musa_hc_down_silu=True,
        input_mix_weight_down_block_inject=types.SimpleNamespace(weight=weight),
        lora_rank=rank,
        hc_count=hc,
        pad_size=12,
    )

    lora, injection = GatedResidual._down_and_inject(residual, x)
    expected_lora, expected_injection = _reference(x, weight, rank, hc)
    torch.testing.assert_close(lora, expected_lora, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(injection, expected_injection, rtol=2e-2, atol=2e-2)
