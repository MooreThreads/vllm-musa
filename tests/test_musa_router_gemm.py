# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MUSA project
"""Contracts for the MUSA parallel FP32 router gate.

Three things are asserted here:

* the admissibility guard is a *pure predicate* - it answers ``None`` for anything it
  does not own (wrong device, dtype, shape or layout) so every other model's tier
  chain is untouched, it honours ``VLLM_MUSA_ROUTER_GATE_FP32=0``, and it *does*
  accept the one gate shape this kernel is tuned for (so an over-strict guard cannot
  make the other tests pass vacuously);
* the guard decides the same way for the FakeTensors Dynamo traces with as for real
  tensors.  A per-call ``x.device.type`` test passes at runtime but reads ``meta`` while
  tracing, so the tier was folded into the compiled graph as "not applicable" and the
  kernel never ran in serving: eager 348/348 accepted, compiled trace zero router-gate
  kernels with the vendor FP32 SGEMM still at 29 calls per step.  That is a silent
  perf-only failure, so it gets its own test;
* on MUSA the kernel is **bitwise identical** to the FP32 reference the router used
  before (``x.float() @ weight.T``), for both BF16 and FP32 activations and for both
  a BF16-valued weight (what a BF16 checkpoint produces) and a full-precision one.
  Bitwise parity is the reason this kernel can ship without an accuracy discussion:
  the router's top-k cannot change;
* the foreign gate shapes really used by other MoE models in this repo are declined,
  which is the regression argument for wiring the kernel into shared router code.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllm_musa.model_executor.layers.fused_moe.router import (  # noqa: E402
    musa_router_gemm as rg,
)
from vllm_musa.model_executor.layers.fused_moe.router.musa_router_gemm import (  # noqa: E402
    router_gate_enabled,
    router_gate_fp32,
)

HIDDEN, EXPERTS = 2688, 128
M_VALUES = (1, 7, 115, 512)

# (experts, hidden) of gates this kernel must NOT touch, with the model family that
# uses them: Qwen3-30B-A3B, Qwen3.5-122B-A10B, DeepSeek-V3, and a small-expert gate.
FOREIGN_GATE_SHAPES = ((128, 2048), (256, 4096), (256, 7168), (64, 2048))


requires_musa = pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="requires a MUSA device",
)


def test_guard_declines_non_musa_tensors() -> None:
    """CPU tensors belong to another platform's tier chain, whatever their shape."""
    x = torch.zeros((4, HIDDEN), dtype=torch.bfloat16)
    w = torch.zeros((EXPERTS, HIDDEN), dtype=torch.float32)
    assert router_gate_fp32(x, w) is None


def test_guard_respects_env_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    assert router_gate_enabled()
    monkeypatch.setenv("VLLM_MUSA_ROUTER_GATE_FP32", "0")
    assert not router_gate_enabled()


@requires_musa
@pytest.mark.parametrize(("experts", "hidden"), FOREIGN_GATE_SHAPES)
def test_guard_declines_foreign_gate_shapes(experts: int, hidden: int) -> None:
    x = torch.zeros((4, hidden), device="musa", dtype=torch.bfloat16)
    w = torch.zeros((experts, hidden), device="musa", dtype=torch.float32)
    assert router_gate_fp32(x, w) is None


@requires_musa
def test_guard_declines_unsupported_dtypes_and_layouts() -> None:
    x = torch.zeros((4, HIDDEN), device="musa", dtype=torch.bfloat16)
    w = torch.zeros((EXPERTS, HIDDEN), device="musa", dtype=torch.float32)
    # A BF16 or FP16 gate weight belongs to a different tier.
    assert router_gate_fp32(x, w.to(torch.bfloat16)) is None
    assert router_gate_fp32(x, w.to(torch.float16)) is None
    # Narrowed activations are declined; FP32 and BF16 are the supported inputs.
    assert router_gate_fp32(x.to(torch.float16), w) is None
    # A mismatched hidden size is declined; an empty batch is served with an empty
    # result (idle steps are legal and must not fall through to another tier).
    assert router_gate_fp32(x[:, :1024].contiguous(), w) is None
    empty = router_gate_fp32(x[:0], w)
    assert empty is not None and tuple(empty.shape) == (0, EXPERTS)
    # Non-contiguous operands are declined rather than silently reinterpreted.
    wide = torch.zeros((EXPERTS, 2 * HIDDEN), device="musa", dtype=torch.float32)
    assert router_gate_fp32(x, wide[:, ::2]) is None
    # The one supported combination is accepted, which keeps every "is None" above
    # meaningful.
    assert router_gate_fp32(x, w) is not None


@requires_musa
def test_served_calls_are_counted_so_a_dead_kernel_is_visible() -> None:
    """A kernel that never runs and a kernel that runs bitwise-identically look the
    same from the output side; only the counter (and the one-shot log) tell them apart."""
    before = rg.activation_count()
    x = torch.randn((7, HIDDEN), dtype=torch.bfloat16, device="musa")
    w = torch.randn((EXPERTS, HIDDEN), dtype=torch.float32, device="musa")
    assert rg.router_gate_fp32(x, w) is not None
    assert rg.activation_count() == before + 1
    # A declined shape must not be counted as served.
    declined = torch.randn((7, 2048), dtype=torch.float32, device="musa")
    assert rg.router_gate_fp32(declined, w[:, :2048].contiguous()) is None
    assert rg.activation_count() == before + 1


@requires_musa
def test_guard_accepts_meta_device_tensors_the_way_tracing_sees_them() -> None:
    """Under Dynamo (and under capture) the operands are not MUSA tensors: tracing sees
    FakeTensors, whose ``device.type`` is ``meta``.  A per-call device test therefore
    reads "not MUSA" exactly where it matters, folds the tier away in the compiled
    graph, and costs the speedup while leaving the output identical - a silent
    perf-only failure that no numeric test can see.  The platform decision has to be
    a module-level property instead, which is what this asserts.
    """
    meta_x = torch.zeros((7, HIDDEN), dtype=torch.bfloat16, device="meta")
    meta_w = torch.zeros((EXPERTS, HIDDEN), dtype=torch.float32, device="meta")
    assert meta_x.device.type == "meta", "the premise of this test changed"
    out = router_gate_fp32(meta_x, meta_w)
    assert out is not None and tuple(out.shape) == (7, EXPERTS)
    # A foreign shape must still be declined when the tensors are not real.
    foreign = torch.zeros((128, 2048), dtype=torch.float32, device="meta")
    assert router_gate_fp32(meta_x[:, :2048], foreign) is None


@requires_musa
@pytest.mark.parametrize("m", M_VALUES)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_kernel_is_bitwise_equal_to_fp32_reference(
    m: int, x_dtype: torch.dtype, weight_dtype: torch.dtype
) -> None:
    torch.manual_seed(100037 + m)
    # A BF16 checkpoint stores gate weights that are exactly BF16-valued, so that case
    # is exercised by casting; plain randn covers the general FP32 case.
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
