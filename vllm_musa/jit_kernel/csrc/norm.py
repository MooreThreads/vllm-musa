from __future__ import annotations

from pathlib import Path

import torch
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_musa.jit_kernel.csrc.jit import load_musa_jit
from vllm_musa.jit_kernel.utils import cache_once
from vllm_musa.tuning import (
    get_primed_musa_kernel_hardware,
    select_jit_rmsnorm_tactic,
)


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
    block_threads: int = 0,
) -> torch.Tensor:
    _ = enable_pdl
    if out is None:
        out = torch.empty_like(input)
    torch.ops.vllm.musa_csrc_rmsnorm(
        input,
        weight,
        out,
        float(eps),
        False,
        int(block_threads),
    )
    return out


def gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
    enable_pdl: bool | None = None,
    block_threads: int = 0,
) -> torch.Tensor:
    _ = enable_pdl
    if out is None:
        out = torch.empty_like(input)
    torch.ops.vllm.musa_csrc_rmsnorm(
        input,
        weight,
        out,
        float(eps),
        True,
        int(block_threads),
    )
    return out


def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    gemma: bool = False,
    block_threads: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """In-place fused residual add and RMSNorm using the JIT MUSA kernel.

    The returned tensors alias ``input`` and ``residual`` respectively. The
    vec8/cache specialization computes the variance and normalized value
    from the unrounded FP32 residual sum while storing the updated residual in
    the input dtype, matching ``vllm.ir.ops.fused_add_rms_norm``. ``weight`` may
    match the activation dtype or be FP32. Set ``gemma=True`` only for a raw
    zero-centered Gemma parameter; generic IR supplies an effective FP32 scale
    with Gemma's ``+1`` already applied and therefore uses ``gemma=False``.
    """
    torch.ops.vllm.musa_csrc_fused_add_rmsnorm(
        input,
        residual,
        weight,
        float(eps),
        bool(gemma),
        int(block_threads),
    )
    return input, residual


def _production_block_threads(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    mode: str,
    contiguous: bool,
    requested: int,
) -> int:
    if requested != 0:
        return requested
    hidden_size = input.shape[-1]
    rows = input.numel() // hidden_size
    device_index = input.device.index if input.device.index is not None else 0
    tactic = select_jit_rmsnorm_tactic(
        hardware=get_primed_musa_kernel_hardware(device_index),
        mode=mode,
        rows=rows,
        hidden_size=hidden_size,
        input_dtype=str(input.dtype),
        weight_dtype=str(weight.dtype),
        contiguous=contiguous,
    )
    return tactic.block_threads if tactic is not None else 0


def _rmsnorm_custom(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    gemma: bool,
    block_threads: int,
) -> None:
    block_threads = _production_block_threads(
        input,
        weight,
        mode="gemma" if gemma else "plain",
        contiguous=(
            input.is_contiguous() and weight.is_contiguous() and out.is_contiguous()
        ),
        requested=block_threads,
    )
    _norm_module().sgl_musa_rmsnorm(
        input,
        weight,
        out,
        float(eps),
        bool(gemma),
        int(block_threads),
    )


def _rmsnorm_custom_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    gemma: bool,
    block_threads: int,
) -> None:
    return


direct_register_custom_op(
    op_name="musa_csrc_rmsnorm",
    op_func=_rmsnorm_custom,
    mutates_args=["out"],
    fake_impl=_rmsnorm_custom_fake,
)


def _fused_add_rmsnorm_custom(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gemma: bool,
    block_threads: int,
) -> None:
    block_threads = _production_block_threads(
        input,
        weight,
        mode="fused_gemma" if gemma else "fused",
        contiguous=(
            input.is_contiguous()
            and residual.is_contiguous()
            and weight.is_contiguous()
        ),
        requested=block_threads,
    )
    _norm_module().sgl_musa_fused_add_rmsnorm(
        input,
        residual,
        weight,
        float(eps),
        bool(gemma),
        int(block_threads),
    )


def _fused_add_rmsnorm_custom_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gemma: bool,
    block_threads: int,
) -> None:
    return


direct_register_custom_op(
    op_name="musa_csrc_fused_add_rmsnorm",
    op_func=_fused_add_rmsnorm_custom,
    mutates_args=["input", "residual"],
    fake_impl=_fused_add_rmsnorm_custom_fake,
)


@cache_once
def _qk_mrope_module():
    import tilelang

    tilelang_dir = Path(tilelang.__file__).resolve().parent
    return load_musa_jit(
        "vllm_musa_norm_qk_mrope",
        ("norm/qk_mrope.mu",),
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


def fused_qk_rmsnorm_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    mrope_section_t: int,
    mrope_section_h: int,
    mrope_section_w: int,
    is_interleaved: bool,
    eps: float = 1e-6,
    gemma: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """QK-RMSNorm + MRoPE in one kernel. q/k are (tokens, heads, head_dim)."""
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    torch.ops.vllm.musa_csrc_fused_qk_rmsnorm_mrope(
        q,
        k,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        q_out,
        k_out,
        bool(is_neox),
        int(mrope_section_t),
        int(mrope_section_h),
        int(mrope_section_w),
        bool(is_interleaved),
        float(eps),
        bool(gemma),
    )
    return q_out, k_out


def fused_qk_rmsnorm_mrope_cache_out(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    is_neox: bool,
    mrope_section_t: int,
    mrope_section_h: int,
    mrope_section_w: int,
    is_interleaved: bool,
    eps: float = 1e-6,
    gemma: bool = False,
) -> None:
    """Fuse Q/K RMSNorm, RoPE, and a flat-NHD K/V cache update."""
    torch.ops.vllm.musa_csrc_fused_qk_rmsnorm_mrope_cache_out(
        q,
        k,
        v,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        q_out,
        k_out,
        key_cache,
        value_cache,
        slot_mapping,
        bool(is_neox),
        int(mrope_section_t),
        int(mrope_section_h),
        int(mrope_section_w),
        bool(is_interleaved),
        float(eps),
        bool(gemma),
    )


def _fused_qk_rmsnorm_mrope_custom(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    is_neox: bool,
    mrope_section_t: int,
    mrope_section_h: int,
    mrope_section_w: int,
    is_interleaved: bool,
    eps: float,
    gemma: bool,
) -> None:
    _qk_mrope_module().sgl_musa_fused_qk_rmsnorm_mrope(
        q,
        k,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        q_out,
        k_out,
        bool(is_neox),
        int(mrope_section_t),
        int(mrope_section_h),
        int(mrope_section_w),
        bool(is_interleaved),
        float(eps),
        bool(gemma),
    )


def _fused_qk_rmsnorm_mrope_custom_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    is_neox: bool,
    mrope_section_t: int,
    mrope_section_h: int,
    mrope_section_w: int,
    is_interleaved: bool,
    eps: float,
    gemma: bool,
) -> None:
    return


direct_register_custom_op(
    op_name="musa_csrc_fused_qk_rmsnorm_mrope",
    op_func=_fused_qk_rmsnorm_mrope_custom,
    mutates_args=["q_out", "k_out"],
    fake_impl=_fused_qk_rmsnorm_mrope_custom_fake,
)


def _fused_qk_rmsnorm_mrope_cache_out_custom(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    is_neox: bool,
    mrope_section_t: int,
    mrope_section_h: int,
    mrope_section_w: int,
    is_interleaved: bool,
    eps: float,
    gemma: bool,
) -> None:
    _qk_mrope_module().sgl_musa_fused_qk_rmsnorm_mrope_cache_out(
        q,
        k,
        v,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        q_out,
        k_out,
        key_cache,
        value_cache,
        slot_mapping,
        bool(is_neox),
        int(mrope_section_t),
        int(mrope_section_h),
        int(mrope_section_w),
        bool(is_interleaved),
        float(eps),
        bool(gemma),
    )


def _fused_qk_rmsnorm_mrope_cache_out_custom_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    is_neox: bool,
    mrope_section_t: int,
    mrope_section_h: int,
    mrope_section_w: int,
    is_interleaved: bool,
    eps: float,
    gemma: bool,
) -> None:
    return


direct_register_custom_op(
    op_name="musa_csrc_fused_qk_rmsnorm_mrope_cache_out",
    op_func=_fused_qk_rmsnorm_mrope_cache_out_custom,
    mutates_args=["q_out", "k_out", "key_cache", "value_cache"],
    fake_impl=_fused_qk_rmsnorm_mrope_cache_out_custom_fake,
)
