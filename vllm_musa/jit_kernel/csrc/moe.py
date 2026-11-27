from __future__ import annotations

import logging

import torch
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_musa.jit_kernel.csrc.jit import load_musa_jit
from vllm_musa.jit_kernel.utils import cache_once

logger = logging.getLogger(__name__)

# Native MUSA vec8 (int4-packed, 8 elems/load) MoE glue. The bf16 decode path otherwise runs
# silu(gate)*up and the top-k reduce as several eager kernels; the vec8 kernels do each in one
# pass and are 1:1 op replacements, so they capture into the FULL_DECODE graph unchanged. The
# correctness guards (device / contiguity / dtype / shape) fall back to the eager kernels for
# any input they do not support (e.g. FP8/int8 activations), so they are always safe to apply.

_SILU = 0
_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@cache_once
def _moe_module():
    return load_musa_jit(
        "vllm_musa_moe",
        ("moe/act_and_mul.mu", "moe/moe_sum_reduce.mu"),
        extra_musa_cflags=(
            "-Wno-error=address-of-temporary",
            "-fmusa-flush-denormals-to-zero",
            "-fno-signed-zeros",
            "-mllvm",
            "-mtgpu-opt-level=1",
            "-mllvm",
            "-mtgpu-load-store-opt=1",
            "-mllvm",
            "-mtgpu-fold-global-ldst=1",
        ),
    )


@cache_once
def _moe_available() -> bool:
    """True once the native kernels JIT-build. Cached so the guarded callers fall back to the
    eager kernels (rather than raise) when the build fails, and never recompile under capture."""
    try:
        _moe_module()
        return True
    except Exception:
        logger.warning(
            "native MUSA MoE glue unavailable; using eager kernels", exc_info=True
        )
        return False


# Single-element sentinel expert_ids per device. With expert_step = rows the kernel reads
# expert_ids[row / rows] == expert_ids[0] == 0 for every row (0 != -1 => all rows processed)
# and the bounds check ceil(rows / expert_step) == 1 holds. Pre-created so the captured path
# never does a host->device scalar copy.
_DUMMY_EIDS: dict[torch.device, torch.Tensor] = {}


def _dummy_expert_ids(device: torch.device) -> torch.Tensor:
    t = _DUMMY_EIDS.get(device)
    if t is None:
        t = torch.zeros(1, dtype=torch.int32, device=device)
        _DUMMY_EIDS[device] = t
    return t


def _silu_and_mul_custom(out: torch.Tensor, x: torch.Tensor) -> None:
    rows = x.numel() // x.shape[-1]
    _moe_module().sgl_musa_moe_act_and_mul(
        x.view(rows, x.shape[-1]),
        out.view(rows, out.shape[-1]),
        _dummy_expert_ids(x.device),
        max(rows, 1),
        _SILU,
        1,
    )


def _silu_and_mul_fake(out: torch.Tensor, x: torch.Tensor) -> None:
    return


direct_register_custom_op(
    op_name="musa_fast_silu_and_mul",
    op_func=_silu_and_mul_custom,
    mutates_args=["out"],
    fake_impl=_silu_and_mul_fake,
)


def _moe_sum_custom(out: torch.Tensor, x: torch.Tensor) -> None:
    _moe_module().sgl_musa_moe_sum_reduce(x, out, 1.0)


def _moe_sum_fake(out: torch.Tensor, x: torch.Tensor) -> None:
    return


direct_register_custom_op(
    op_name="musa_fast_moe_sum",
    op_func=_moe_sum_custom,
    mutates_args=["out"],
    fake_impl=_moe_sum_fake,
)


def maybe_fast_silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> bool:
    """out[..., d] = silu(x[..., :d]) * x[..., d:2d] via the native vec8 kernel; a drop-in for
    torch.ops._C.silu_and_mul(out, x). Returns False (eager fallback) for unsupported inputs."""
    if x.device.type != "musa":
        return False
    if not _moe_available():
        return False
    if not (x.is_contiguous() and out.is_contiguous()):
        return False
    if x.dtype not in _FAST_DTYPES:
        return False
    if out.shape[-1] * 2 != x.shape[-1]:
        return False
    torch.ops.vllm.musa_fast_silu_and_mul(out, x)
    return True


def maybe_fast_moe_sum(x: torch.Tensor, out: torch.Tensor) -> bool:
    """out[t, h] = sum_k x[t, k, h] via the native kernel; a drop-in for ops.moe_sum(x, out).
    Returns False (eager fallback) for unsupported inputs."""
    if x.device.type != "musa":
        return False
    if not _moe_available():
        return False
    if not (x.is_contiguous() and out.is_contiguous()):
        return False
    if x.dtype not in _FAST_DTYPES:
        return False
    if (
        x.dim() != 3
        or out.dim() != 2
        or x.shape[0] != out.shape[0]
        or x.shape[2] != out.shape[1]
    ):
        return False
    torch.ops.vllm.musa_fast_moe_sum(out, x)
    return True


def prewarm() -> None:
    """JIT-compile the module at load so the first call never compiles inside graph capture."""
    _moe_available()
