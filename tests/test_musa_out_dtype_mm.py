"""MUSA-100042 markers: ``torch.mm(..., out_dtype=)`` must not silently return zeros.

The MUSA torch stack accepts ``out_dtype=torch.float32`` on ``torch.mm``, returns a
correctly shaped fp32 tensor, and never writes it: the result is 100% exact zeros
with no exception.  Upstream vLLM treats the kwarg as CUDA/ROCm-only and falls
back to a cast elsewhere, so a MUSA path that reaches it computes zeros instead of
logits.

What these markers are for:

* the ``xfail(strict=True)`` cases below are the executable statement of the
  contract, **and they are the enforcement**: they are what makes a silent
  behaviour change visible in CI. ``strict=True`` is deliberate - the day the
  vendor fix lands they turn into failures, which is the reminder to delete
  these markers and the ``0014-MUSA-vllm.models.deepseek_v4.attention.patch``
  workaround in the same change.
* ``vllm_musa.mm_out_dtype_guard`` is an **opt-in** convenience check for a
  future enablement, not an automatic guard: nothing in the shipped MUSA path
  calls it, and today's call sites are unreachable by construction. A PR that
  wants the bf16-in/fp32-out GEMM on MUSA (router gate, DeepSeek-V4
  ``kv_score``, an fp32 lm_head accumulation path) should call
  ``require_mm_out_dtype_semantics(...)`` at enablement time, which raises
  today.
* the last case pins the invariant that MUSA cannot satisfy the ``is_cuda()``-based
  specialized-tier gates of ``fused_moe/router/gate_linear.py``; if someone makes
  the MUSA platform claim CUDA capability, that tier starts running and these
  markers are the only warning left.

These tests only need a MUSA device; they do not start a server or load a model.
"""

from __future__ import annotations

import pytest
import torch

from vllm.platforms import current_platform
from vllm_musa.mm_out_dtype_guard import (
    mm_out_dtype_is_correct,
    require_mm_out_dtype_semantics,
)

pytestmark = pytest.mark.skipif(
    not bool(getattr(current_platform, "is_musa", lambda: False)()),
    reason="MUSA-100042 markers require the MUSA torch stack",
)

BROKEN = (
    "MUSA-100042: torch.mm(..., out_dtype=torch.float32) is accepted on MUSA and "
    "returns an all-zero tensor instead of the fp32 accumulation. Delete this xfail "
    "and the 0014 DeepSeek-V4 workaround when the torch_musa fix lands."
)


def _bf16_pair(m: int = 7, k: int = 256, n: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(100042)
    a = torch.randn(m, k, generator=gen).to("musa").to(torch.bfloat16)
    w = torch.randn(n, k, generator=gen).to("musa").to(torch.bfloat16)
    return a, w


def _reference(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # Exact for bf16-valued inputs: the products of two bf16 values are
    # representable in fp32, so only the summation order can differ.
    return a.to(torch.float32) @ w.to(torch.float32).t()


def test_control_mm_without_out_dtype_is_correct() -> None:
    """The cast reference path must work, otherwise the xfails below prove nothing."""
    a, w = _bf16_pair()
    assert torch.allclose(
        torch.mm(a, w.t()).to(torch.float32), _reference(a, w), atol=1e-2, rtol=1.6e-2
    )
    assert torch.equal(torch.mm(a.float(), w.float().t()), _reference(a, w))


@pytest.mark.xfail(strict=True, reason=BROKEN)
def test_mm_out_dtype_fp32_is_not_all_zero() -> None:
    a, w = _bf16_pair()
    out = torch.mm(a, w.t(), out_dtype=torch.float32)
    assert out.dtype is torch.float32
    assert not bool((out == 0).all()), (
        "all-zero output: the kwarg was accepted and nothing was written"
    )


@pytest.mark.xfail(strict=True, reason=BROKEN)
def test_mm_out_dtype_fp32_matches_cast_path_bf16_inputs() -> None:
    a, w = _bf16_pair()
    assert torch.allclose(
        torch.mm(a, w.t(), out_dtype=torch.float32), _reference(a, w), atol=1e-2, rtol=1.6e-2
    )


@pytest.mark.xfail(strict=True, reason=BROKEN)
def test_mm_out_dtype_fp32_matches_cast_path_fp32_inputs() -> None:
    a, w = _bf16_pair()
    assert torch.equal(
        torch.mm(a.float(), w.float().t(), out_dtype=torch.float32), _reference(a, w)
    )


def test_guard_agrees_with_the_direct_probe() -> None:
    a, w = _bf16_pair()
    out = torch.mm(a, w.t(), out_dtype=torch.float32)
    observed = (
        not bool((out == 0).all())
        and torch.allclose(out, _reference(a, w), atol=1e-2, rtol=1.6e-2)
    )
    assert mm_out_dtype_is_correct() is observed


def test_guard_raises_while_the_kwarg_is_broken() -> None:
    if mm_out_dtype_is_correct():
        pytest.skip(
            "vendor fix landed: delete the xfail markers above and the 0014 "
            "DeepSeek-V4 workaround (MUSA-100042)"
        )
    with pytest.raises(RuntimeError, match="MUSA-100042"):
        require_mm_out_dtype_semantics("test enablement")


def test_musa_cannot_satisfy_the_is_cuda_specialized_tier_gate() -> None:
    """Invariant behind 'the out_dtype tier is unreachable on MUSA today'."""
    assert not current_platform.is_cuda()
    assert not (
        current_platform.is_cuda()
        and (
            current_platform.is_device_capability((9, 0))
            or current_platform.is_device_capability_family(100)
        )
    )
