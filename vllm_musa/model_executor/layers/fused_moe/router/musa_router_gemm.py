# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MUSA project
"""Parallel FP32 router-gate GEMM for MUSA (Nemotron-H style hybrid MoE).

Why this exists
---------------
``GateLinear(..., force_fp32_compute=True)`` stores the gate weight in FP32 when no
Hopper/Blackwell router kernel exists, which is every MUSA run, so the gate falls
through to ``F.linear`` and lands on the vendor FP32 SGEMM.  For
Nemotron-3.5-Lightning (H=2688, E=128, top-6) that kernel launches with
``grid=(1,1,1)`` and costs ~292 us per call **at every M** (measured at M in
{1, 7, 512}), i.e. 19% of the decode step device time: 29 MoE layers x one call per
step.  The op is latency bound, not bandwidth bound - the FP32 weight is 1.38 MB,
which a bandwidth-bound kernel would read in about 1 us, and an empty kernel on this
device measures about 7 us.

What it does
------------
``router_gate_fp32`` computes the same ``[M, 2688] x [2688, 128]`` FP32 product with
a 2-D Triton grid (M tiles x N tiles) and keeps plain-FP32 FMA accumulation via
``tl.dot(..., input_precision="ieee")``.  The accumulation order therefore matches the
vendor kernel, and the logits are **bitwise identical** to it for every M measured -
the router's top-k, and hence the generated output, is unchanged.

Measured on S5000 (30-core, ``vllm:v0.28.0-ph1-5.2.0-torch2.11.0.post1-latest``, cold
L2, CUDA-graph replay, the tile config this module ships):

===========  =============  =============  ========  =======
M            vendor FP32    this kernel    speedup   bitwise
===========  =============  =============  ========  =======
1            296.2 us       167.0 us       1.77x     yes
7            319.1 us       165.9 us       1.92x     yes
115          322.3 us       158.0 us       2.04x     yes
512          331.0 us       324.5 us       1.02x     yes
===========  =============  =============  ========  =======

End-to-end on the same image (Nemotron-3.5-Lightning-30B-A3B, MTP6, TP1, B1,
4096-in/1000-out, 1 warmup + 3 measured): TPOT 8.1576 -> 7.49 ms/token (-8.2%),
total 9.5808 -> 8.9134 s (-7.0%), TTFT unchanged, output SHA identical to the
unpatched run.

Scope
-----
Only the shape this kernel was tuned for is enabled (``_SUPPORTED_WEIGHT_SHAPE``);
every other MoE model keeps its own tier chain untouched, which the unit tests assert
for the foreign gate shapes actually used in this repo.  Disable with
``VLLM_MUSA_ROUTER_GATE_FP32=0``.

Trace safety
------------
The guard must decide **identically for real tensors and for the FakeTensors Dynamo
traces with**: a check like ``x.device.type == "musa"`` is true at runtime but false
under tracing (FakeTensors report ``meta``), so the branch would be folded into the
compiled graph as "not applicable" and the kernel would silently never run.  That
happened in the first version of this wiring: on eager it was 348/348 accepted, in
serving the profiled decode window contained zero router-gate kernels and the vendor
FP32 SGEMM still ran 29x per step - same output, no speedup.  The platform check is
therefore a module-level Python flag, and the per-call checks only use tensor
metadata that is identical for real and fake tensors (dtype, shape, contiguity).

Two notes for reviewers
-----------------------
* An operator-level alternative is a BF16 tensor-core GEMM: both operands are exactly
  BF16-valued here (the checkpoint is BF16 and the activation arrives as BF16), so
  every product stays exact, and that path measures 37-44 us (7-8x).  It is *not*
  bitwise, because the tensor core sums K in a different order (max-abs 5e-4-7e-4 on
  logits of magnitude ~200, top-6 agreement 1.000 on random weights).  End-to-end it
  costs MTP acceptance (5.9 -> 3.7) and is therefore not what this module ships; it
  needs the model owners' accuracy sign-off.
* ``torch.mm(..., out_dtype=torch.float32)`` on this MUSA stack returns an all-zero
  tensor (see MUSA-100042): the GEMM never writes its output, for BF16 *and* FP32
  inputs.  Any FP32-router path that relies on ``out_dtype`` on MUSA is silently
  wrong, which is a further reason to compute the gate in this kernel instead.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

__all__ = ["router_gate_enabled", "router_gate_fp32"]

# The gate this module is tuned for: Nemotron-3.5-Lightning (hidden 2688, 128 routed
# experts).  Other shapes are declined so their models keep their own tier chain.
_HIDDEN, _EXPERTS = 2688, 128
_SUPPORTED_WEIGHT_SHAPE = (_EXPERTS, _HIDDEN)

# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) from a cold-cache sweep over
# BLOCK_M in {16, 32}, BLOCK_N in {32, 64, 128}, BLOCK_K in {64, 128, 256} and
# num_stages in {1, 2, 3}.  The sweep is flat within ~3% on the stage axis, so the
# kernel is not pipelining limited; the M > 128 bucket is the measured best there and
# is still only at parity with the vendor kernel.
_CFG_SMALL = (16, 32, 128, 4, 3)
_CFG_LARGE = (16, 64, 64, 4, 3)
_CFG_MAX_TOKENS = 128

# Evaluated once, at import: a per-call ``x.device.type == "musa"`` test would read
# "meta" while Dynamo traces (see "Trace safety" above) and disable the kernel in
# compiled runs.  "meta" is therefore a *supported* device here - it is how tracing and
# capture present the operands, and the op's fake impl serves it.
_ON_MUSA = current_platform.device_type == "musa"
_ACCEPTED_DEVICES = ("musa", "meta")


def router_gate_enabled() -> bool:
    """Kill switch, read per call so tests and operators can flip it."""
    return os.environ.get("VLLM_MUSA_ROUTER_GATE_FP32", "1") != "0"


def _pick_cfg(m: int) -> tuple[int, int, int, int, int]:
    return _CFG_SMALL if m <= _CFG_MAX_TOKENS else _CFG_LARGE


@triton.jit
def _router_gate_fp32_kernel(
    X,
    W,
    Y,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    X_IS_FP32: tl.constexpr,
):
    """Y[M, N] = X[M, K] @ W[N, K].T in exact FP32.

    ``W`` keeps the module's native ``[N, K]`` layout (no cached transpose), and
    ``tl.trans`` feeds ``tl.dot`` a ``[BLOCK_K, BLOCK_N]`` tile so both loads stay
    contiguous along K.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M

    x_ptrs = X + offs_m[:, None] * K + offs_k[None, :]
    w_ptrs = W + offs_n[:, None] * K + offs_k[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _k in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        if not X_IS_FP32:
            x = x.to(tl.float32)
        w = tl.load(w_ptrs)
        # input_precision="ieee" is what keeps this bitwise equal to the FP32
        # reference; the default precision lowers to a tensor-core split that changes
        # the summation order and the router's top-k with it.
        acc = tl.dot(x, tl.trans(w), acc, input_precision="ieee")
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc, mask=m_mask[:, None])


def _launch(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m = x.shape[0]
    out = torch.empty((m, _EXPERTS), dtype=torch.float32, device=x.device)
    if m == 0:
        # An empty batch is legal (idle step) but a zero-sized grid is not worth
        # launching; F.linear would return an empty tensor here too.
        return out
    block_m, block_n, block_k, num_warps, num_stages = _pick_cfg(m)
    grid = (triton.cdiv(m, block_m), _EXPERTS // block_n)
    _router_gate_fp32_kernel[grid](
        x,
        weight,
        out,
        m,
        K=_HIDDEN,
        N=_EXPERTS,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        X_IS_FP32=(x.dtype == torch.float32),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def _router_gate_fp32_impl(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _launch(x, weight)


def _router_gate_fp32_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.float32, device=x.device
    )


# Registered through the repo's usual helper so the launch is opaque to Dynamo and to
# graph capture, where the Python-side shape checks below cannot run.
direct_register_custom_op(
    op_name="musa_router_gate_fp32",
    op_func=_router_gate_fp32_impl,
    mutates_args=[],
    fake_impl=_router_gate_fp32_fake,
)


def router_gate_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor | None:
    """MUSA FP32 router gate; returns None when the caller must keep its own path.

    The checks are deliberately cheap (two shape compares, one contiguous flag): this
    runs once per MoE layer per forward step.  ``None`` means "not applicable" and
    never an error, so the caller can treat this as an optional tier.
    """
    if not _ON_MUSA or not router_gate_enabled():
        return None
    if weight.dtype != torch.float32 or weight.shape != _SUPPORTED_WEIGHT_SHAPE:
        return None
    if x.shape[-1] != _HIDDEN:
        return None
    if x.dtype not in (torch.float32, torch.bfloat16):
        return None
    if x.device.type not in _ACCEPTED_DEVICES:
        return None
    if not x.is_contiguous() or not weight.is_contiguous():
        return None
    return torch.ops.vllm.musa_router_gate_fp32.default(x, weight)
