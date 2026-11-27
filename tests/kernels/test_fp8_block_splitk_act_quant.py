# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit and MUSA hardware tests for single-pass native.act_quant in fp8_block_splitk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

try:
    import torch
except ImportError:  # pragma: no cover - host CI without torch
    torch = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITK_PATH = REPO_ROOT / "vllm_musa" / "kernels" / "fp8_block_splitk.py"


def test_fp8_block_splitk_source_contract() -> None:
    splitk_src = SPLITK_PATH.read_text(encoding="utf-8")
    assert "_can_use_native_act_quant" in splitk_src
    assert "native.act_quant(a)" in splitk_src


def _has_musa() -> bool:
    return (
        torch is not None
        and getattr(torch.version, "musa", None) is not None
        and torch.musa.is_available()
    )


@pytest.mark.skipif(not _has_musa(), reason="Requires MUSA GPU")
@pytest.mark.parametrize(
    ("n", "k", "rows"),
    [
        (1792, 5120, 32),
        (5120, 15360, 8),
        (5120, 15360, 32),
        (25600, 768, 24),
        (1024, 4096, 128),
    ],
)
def test_fp8_block_splitk_native_act_quant_bitwise_parity_and_cudagraph(
    monkeypatch: Any,
    n: int,
    k: int,
    rows: int,
) -> None:
    from vllm_musa.kernels import fp8_block_splitk as sk

    torch.manual_seed(0)
    a = torch.randn((rows, k), device="musa", dtype=torch.bfloat16)
    weight = (torch.randn((n, k), device="musa") * 0.5).to(torch.float8_e4m3fn)
    exponents = torch.randint(-8, -4, (n // 32, k // 32), device="musa")
    weight_scale = torch.pow(2.0, exponents.float())

    real_cap = torch.cuda.is_current_stream_capturing
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    cfg = sk.fp8_block_splitk_config(weight, weight_scale, rows)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", real_cap)
    if cfg is None:
        cfg = sk.fp8_block_splitk_config(weight, weight_scale, rows)
    assert cfg is not None

    out_ref = torch.empty((rows, n), device="musa", dtype=torch.bfloat16)
    monkeypatch.setattr(sk, "_NATIVE_ACT_QUANT", False)
    sk.fp8_block_splitk_mm(a, None, weight, weight_scale, out_ref, cfg)

    out_cand = torch.empty((rows, n), device="musa", dtype=torch.bfloat16)
    monkeypatch.setattr(sk, "_NATIVE_ACT_QUANT", True)
    # Force _can_use_native_act_quant eligibility in eager to check bitwise parity
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert sk._can_use_native_act_quant(a, (n + cfg[1] - 1) // cfg[1])
    sk.fp8_block_splitk_mm(a, None, weight, weight_scale, out_cand, cfg)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", real_cap)
    assert torch.equal(out_cand, out_ref)

    # Unaligned row stride falls back cleanly to Triton QUANT_A=True
    a_padded = torch.randn((rows, k + 4), device="musa", dtype=torch.bfloat16)[:, :k]
    assert a_padded.stride(0) % 8 != 0
    out_unaligned = torch.empty((rows, n), device="musa", dtype=torch.bfloat16)
    sk.fp8_block_splitk_mm(a_padded, None, weight, weight_scale, out_unaligned, cfg)
    out_unaligned_ref = torch.empty((rows, n), device="musa", dtype=torch.bfloat16)
    monkeypatch.setattr(sk, "_NATIVE_ACT_QUANT", False)
    sk.fp8_block_splitk_mm(a_padded, None, weight, weight_scale, out_unaligned_ref, cfg)
    assert torch.equal(out_unaligned, out_unaligned_ref)

    # Real CUDAGraph capture & replay bitwise parity
    if rows <= cfg[5]:
        monkeypatch.setattr(sk, "_NATIVE_ACT_QUANT", True)
        a_buf = a.clone()
        # Warmup before capture
        g_warm = torch.empty((rows, n), device="musa", dtype=torch.bfloat16)
        sk.fp8_block_splitk_mm(a_buf, None, weight, weight_scale, g_warm, cfg)
        torch.musa.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            g_out = torch.empty((rows, n), device="musa", dtype=torch.bfloat16)
            sk.fp8_block_splitk_mm(a_buf, None, weight, weight_scale, g_out, cfg)
        for step in range(4):
            a_new = torch.randn((rows, k), device="musa", dtype=torch.bfloat16)
            a_buf.copy_(a_new)
            g.replay()
            torch.musa.synchronize()
            monkeypatch.setattr(sk, "_NATIVE_ACT_QUANT", False)
            eager_check = torch.empty((rows, n), device="musa", dtype=torch.bfloat16)
            sk.fp8_block_splitk_mm(a_new, None, weight, weight_scale, eager_check, cfg)
            monkeypatch.setattr(sk, "_NATIVE_ACT_QUANT", True)
            assert torch.equal(g_out, eager_check), f"mismatch on replay {step}"
