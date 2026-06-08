from __future__ import annotations

from pathlib import Path

import torch
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_musa.jit_kernel.csrc.jit import load_musa_jit
from vllm_musa.jit_kernel.utils import cache_once


@cache_once
def _norm_module():
    import tilelang

    tilelang_dir = Path(tilelang.__file__).resolve().parent
    return load_musa_jit(
        "vllm_musa_norm",
        ("norm/rmsnorm.mu",),
        extra_musa_cflags=(
            f"-I{(tilelang_dir / 'src').resolve()}",
            f"-I{(tilelang_dir / '3rdparty' / 'mutlass' / 'include').resolve()}",
            "-Wno-error=address-of-temporary",
            "-fmusa-flush-denormals-to-zero",
            "-fno-signed-zeros",
            "-D__MUSA_ARCH_LIST__=310",
            "-mllvm",
            "-mtgpu-opt-level=1",
            "-mllvm",
            "-mtgpu-load-store-opt=1",
            "-mllvm",
            "-mtgpu-fold-global-ldst=1",
            "-mllvm",
            "-mtgpu-load-cluster-mutation=1",
            "-mllvm",
            "-mtgpu-store-cluster-mutation=1",
            "-mllvm",
            "-mtgpu-memory-sched-mutation=1",
            "-mllvm",
            "-mtgpu-alloc-shared-memory-from-zero=1",
        ),
    )


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
    enable_pdl: bool | None = None,
) -> torch.Tensor:
    _ = enable_pdl
    if out is None:
        out = torch.empty_like(input)
    torch.ops.vllm.musa_csrc_rmsnorm(input, weight, out, float(eps), False)
    return out


def gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
    enable_pdl: bool | None = None,
) -> torch.Tensor:
    _ = enable_pdl
    if out is None:
        out = torch.empty_like(input)
    torch.ops.vllm.musa_csrc_rmsnorm(input, weight, out, float(eps), True)
    return out


def _rmsnorm_custom(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    gemma: bool,
) -> None:
    _norm_module().sgl_musa_rmsnorm(input, weight, out, float(eps), bool(gemma))


def _rmsnorm_custom_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    gemma: bool,
) -> None:
    return


direct_register_custom_op(
    op_name="musa_csrc_rmsnorm",
    op_func=_rmsnorm_custom,
    mutates_args=["out"],
    fake_impl=_rmsnorm_custom_fake,
)
