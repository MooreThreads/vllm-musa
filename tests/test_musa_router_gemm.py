# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MUSA project
"""Contracts for the MUSA parallel FP32 router gate.

Two things are asserted here:

* the admissibility guard is a *pure predicate* - it answers ``None`` for anything
  it does not own (wrong device, shape or dtype) so the caller's own tier chain is
  untouched, and it honours ``VLLM_MUSA_ROUTER_GATE_FP32=0``;
* on MUSA the kernel is **bitwise identical** to the FP32 reference the router used
  before (``x.float() @ weight.T``), for both BF16 and FP32 activations and for
  both a BF16-valued weight (what a BF16 checkpoint produces) and a full-precision
  one.  Bitwise parity is the reason this kernel can ship without an accuracy
  discussion: the router's top-k cannot change.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllm_musa.model_executor.layers.fused_moe.router.musa_router_gemm import (  # noqa: E402
    router_gate_enabled,
    router_gate_fp32,
)

HIDDEN, EXPERTS = 2688, 128
M_VALUES = (1, 7, 115, 512)


requires_musa = pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="requires a MUSA device",
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_guard_declines_foreign_tensors(dtype: torch.dtype) -> None:
    x = torch.zeros((4, HIDDEN), dtype=dtype)
    w = torch.zeros((EXPERTS, HIDDEN), dtype=torch.float32)
    # CPU tensors belong to another platform's tier chain.
    assert router_gate_fp32(x, w) is None
    if hasattr(torch, "musa") and torch.musa.is_available():
        x_m = x.to("musa")
        # Wrong expert count / wrong hidden size / non-fp32 weight.
        assert router_gate_fp32(x_m, w.to("musa")[:64]) is None
        assert router_gate_fp32(x_m[:, :64], w.to("musa")) is None
        assert router_gate_fp32(
            x_m, w.to("musa").to(torch.bfloat16)
        ) is None
        # Narrowed dtypes are not exact, so they are not owned either.
        assert router_gate_fp32(x_m.to(torch.float16), w.to("musa")) is None


def test_guard_respects_env_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    assert router_gate_enabled()
    monkeypatch.setenv("VLLM_MUSA_ROUTER_GATE_FP32", "0")
    assert not router_gate_enabled()


@requires_musa
@pytest.mark.parametrize("m", M_VALUES)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_kernel_is_bitwise_equal_to_fp32_reference(
    m: int, x_dtype: torch.dtype, weight_dtype: torch.dtype
) -> None:
    torch.manual_seed(100037 + m)
    # A BF16 checkpoint stores gate weights that are exactly BF16-valued, so that
    # case is exercised by casting; plain randn covers the general FP32 case.
    w = torch.randn((EXPERTS, HIDDEN), device="musa", dtype=torch.float32)
    if weight_dtype == torch.bfloat16:
        w = w.to(torch.bfloat16).to(torch.float32)
    x = torch.randn((m, HIDDEN), device="musa", dtype=torch.float32).to(x_dtype)

    out = router_gate_fp32(x, w)
    assert out is not None and out.dtype == torch.float32 and out.shape == (m, EXPERTS)

    ref = x.to(torch.float32) @ w.t()
    assert torch.equal(out, ref), (
        "router logits must be bitwise identical to the FP32 reference "
        f"(m={m}, x={x_dtype}, w={weight_dtype}, "
        f"max_abs={(out - ref).abs().max().item()})"
    )


@requires_musa
@pytest.mark.parametrize("m", [1, 7, 115])
def test_topk_stream_is_unchanged(m: int) -> None:
    """Bitwise-equal logits imply the router's top-k cannot move."""
    torch.manual_seed(7_000 + m)
    w = torch.randn((EXPERTS, HIDDEN), device="musa", dtype=torch.bfloat16)
    x = torch.randn((m, HIDDEN), device="musa", dtype=torch.bfloat16)
    out = router_gate_fp32(x, w.to(torch.float32))
    ref = x.to(torch.float32) @ w.to(torch.float32).t()
    assert torch.equal(out.topk(6, dim=-1).indices, ref.topk(6, dim=-1).indices)
