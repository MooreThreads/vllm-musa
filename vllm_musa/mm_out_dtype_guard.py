"""Fail-fast guard for ``torch.mm(..., out_dtype=...)`` on MUSA (MUSA-100042).

On this MUSA torch stack ``torch.mm(a, b, out_dtype=torch.float32)`` is accepted,
returns a correctly shaped and correctly typed tensor, and **never writes it**:
the result is 100% exact zeros with no exception.  The same call without the
kwarg is bitwise exact.  Upstream vLLM declares the kwarg as CUDA/ROCm-only and
falls back to a cast on other platforms, so silent zeros are a MUSA-specific
deviation from the documented contract.

Any future MUSA enablement that wants the bf16-in/fp32-out GEMM should call
:func:`require_mm_out_dtype_semantics` first - the router-gate tier in
``fused_moe/router/gate_linear.py``, the DeepSeek-V4 compressor/indexer
``kv_score``, or any new lm_head fp32 accumulation path.  The check is a
one-off microsecond-scale probe on a tiny tensor, so it cannot be the reason a
path is slow, and it raises ``RuntimeError`` instead of letting a platform
enablement land on zero logits.

**This function is an opt-in convenience check, not an automatic enforcement.**
Nothing in the shipped MUSA path calls it, and the MUSA call sites of the kwarg
are unreachable by construction today (the router-gate tiers sit behind
``can_use_specialized_kernels = current_platform.is_cuda() and (hopper or
blackwell)``; the DeepSeek-V4 ``kv_score`` sites are bypassed by patch
``0014``).  What actually stops a silent future enablement is the CI-visible
``tests/test_musa_out_dtype_mm.py``: its ``xfail(strict=True)`` cases fail as
soon as the semantics change, and they are the reminder to delete the xfail
markers and the ``0014`` workaround.

Call it at enablement time (model or op construction), not from inside a
CUDA-graph capture: it allocates scratch tensors.
"""

from __future__ import annotations

import threading

import torch

__all__ = ["mm_out_dtype_is_correct", "require_mm_out_dtype_semantics"]

_CHECK_LOCK = threading.Lock()
_CHECKED: dict[str, bool] = {}


def mm_out_dtype_is_correct(device: str | None = None) -> bool:
    """Return True when ``mm(..., out_dtype=fp32)`` follows its documented semantics.

    Args:
        device: Device to probe. Defaults to ``musa`` when the running torch was
            built with MUSA, else ``cpu``.

    Returns:
        True when the fp32 ``out_dtype`` result is finite, not all zeros, and
        agrees with the reference ``mm(a, b).to(torch.float32)``. False when the
        op raises, returns zeros, or disagrees.
    """
    if device is None:
        device = "musa" if getattr(torch.version, "musa", None) else "cpu"
    with _CHECK_LOCK:
        cached = _CHECKED.get(device)
    if cached is not None:
        return cached

    ok = False
    try:
        gen = torch.Generator(device="cpu").manual_seed(100042)
        a = torch.randn(7, 64, generator=gen).to(device).to(torch.bfloat16)
        w = torch.randn(16, 64, generator=gen).to(device).to(torch.bfloat16)
        with_kwarg = torch.mm(a, w.t(), out_dtype=torch.float32)
        reference = torch.mm(a, w.t()).to(torch.float32)
        if with_kwarg.dtype != torch.float32:
            ok = False
        elif bool(torch.isnan(with_kwarg).any()) or bool(torch.isinf(with_kwarg).any()):
            ok = False
        elif bool((with_kwarg == 0).all()):
            ok = False
        else:
            ok = bool(torch.allclose(with_kwarg, reference, atol=1e-2, rtol=1.6e-2))
    except Exception:  # noqa: BLE001 - raising is loud, but still not the contract.
        ok = False

    with _CHECK_LOCK:
        _CHECKED[device] = ok
    return ok


def require_mm_out_dtype_semantics(what: str) -> None:
    """Raise unless ``torch.mm(..., out_dtype=fp32)`` follows its documented semantics.

    Args:
        what: Short description of the enablement that needs the semantics, used
            in the error message (for example ``"bf16 router gate tier"``).

    Raises:
        RuntimeError: The kwarg is broken on this device (MUSA-100042): it is
            accepted and returns an all-zero tensor instead of an fp32
            accumulation. Compute in the input dtype and cast, or cast the
            operands to fp32 explicitly.
    """
    if mm_out_dtype_is_correct():
        return
    raise RuntimeError(
        f"Cannot enable {what}: torch.mm(..., out_dtype=torch.float32) does not "
        "implement its documented semantics on this device - it is accepted and "
        "returns an all-zero tensor (MUSA-100042: MUSA torch.mm(out_dtype=) "
        "returns an all-zero tensor). Compute in the input dtype and cast, or "
        "cast the operands to fp32 explicitly."
    )
