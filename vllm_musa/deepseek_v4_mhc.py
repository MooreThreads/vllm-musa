# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 MHC helpers for MUSA runtime and diagnostic paths."""

from __future__ import annotations

import os

import torch

_MHC_PRE_DEEPGEMM_SPLIT_K_ENV = "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_DEEPGEMM_SPLIT_K"
_MHC_PRE_BIG_FUSE_THREADS_ENV = "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_BIG_FUSE_THREADS"
_MHC_PRE_BIG_FUSE_HIDDEN_BLOCK_ENV = (
    "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_BIG_FUSE_HIDDEN_BLOCK"
)
_MHC_PRE_BIG_FUSE_PASS_CONFIG_ENV = "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_BIG_FUSE_PASS_CONFIG"


def mhc_pre_musa(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # The shape-dispatched optimized provider is the production default.  The
    # old implementation selector was only useful for A/B experiments and made
    # the selected path depend on process-global environment state.
    impl = _select_mhc_pre_auto_impl(residual)
    if impl in {"native", "musa", "mu"}:
        return _mhc_pre_native_provider(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
    if impl in {"torch", "fallback"}:
        return mhc_pre_torch_fallback(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
    if impl in {"tilelang", "jit"}:
        try:
            return _mhc_pre_tilelang_provider(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
            )
        except (ImportError, OSError, NotImplementedError, RuntimeError):
            return _mhc_pre_native_provider(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
            )
    if impl in {"deepgemm_big_fuse", "deepgemm-big-fuse"}:
        try:
            return _mhc_pre_deepgemm_big_fuse_provider(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
            )
        except (ImportError, OSError, NotImplementedError, RuntimeError):
            return _mhc_pre_native_provider(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
            )
    raise ValueError(f"unsupported DeepSeek-V4 MHC pre impl: {impl!r}")


def _apply_optional_rms_norm(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> torch.Tensor:
    if norm_weight is None:
        return x
    musa_result = _try_mhc_weighted_rms_norm_musa(x, norm_weight, norm_eps)
    if musa_result is not None:
        return musa_result
    x_float = x.to(torch.float32)
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_float = x_float * torch.rsqrt(variance + norm_eps)
    x_float = x_float * norm_weight.to(torch.float32)
    return x_float.to(x.dtype)


def _mhc_weighted_rms_norm_threads(hidden_size: int) -> int | None:
    if hidden_size % 128 == 0:
        return 128
    if hidden_size % 64 == 0:
        return 64
    return None


def _try_mhc_weighted_rms_norm_musa(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor | None:
    hidden_size = x.shape[-1]
    threads = _mhc_weighted_rms_norm_threads(hidden_size)
    supported = (
        x.device.type == "musa"
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.dim() >= 2
        and norm_weight.shape == (hidden_size,)
        and norm_weight.device == x.device
        and norm_weight.dtype == torch.bfloat16
        and norm_weight.is_contiguous()
        and threads is not None
    )
    if not supported:
        return None

    try:
        from vllm_musa.deepseek_v4_jit.tilelang_kernels import (
            mhc_weighted_rmsnorm_kernel,
            mhc_weighted_rmsnorm_mudnn_like_kernel,
        )

        x_2d = x.view(-1, hidden_size)
        out_2d = torch.empty_like(x_2d)
        kernel_factory = (
            mhc_weighted_rmsnorm_mudnn_like_kernel
            if hidden_size == 4096
            else mhc_weighted_rmsnorm_kernel
        )
        kernel_factory(hidden_size, threads=threads)(
            x_2d,
            norm_weight,
            out_2d,
            float(norm_eps),
        )
        return out_2d.view_as(x)
    except (ImportError, OSError, NotImplementedError, RuntimeError):
        return None


def mhc_pre_musa_with_norm(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    post_mix, comb_mix, layer_input = mhc_pre_musa(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    return (
        post_mix,
        comb_mix,
        _apply_optional_rms_norm(
            layer_input,
            norm_weight,
            norm_eps,
        ),
    )


def mhc_fused_post_pre_musa(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del n_splits, tile_n
    fused_result = _try_mhc_fused_post_prenorm_musa(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    if fused_result is not None:
        residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur = fused_result
        layer_input_cur = _apply_optional_rms_norm(
            layer_input_cur,
            norm_weight,
            norm_eps,
        )
        return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur

    residual_cur = mhc_post_musa(x, residual, post_layer_mix, comb_res_mix)
    post_mix_cur, comb_mix_cur, layer_input_cur = mhc_pre_musa(
        residual_cur,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    layer_input_cur = _apply_optional_rms_norm(
        layer_input_cur,
        norm_weight,
        norm_eps,
    )
    return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur


def _try_mhc_fused_post_prenorm_musa(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    num_tokens = residual.numel() // (hc_mult * hidden_size)
    hc_mult3 = hc_mult * (2 + hc_mult)
    outer_shape = residual.shape[:-2]

    supported = (
        residual.device.type == "musa"
        and 0 < num_tokens <= 16
        and hc_mult == 4
        and hidden_size == 4096
        and x.device == residual.device
        and post_layer_mix.device == residual.device
        and comb_res_mix.device == residual.device
        and fn.device == residual.device
        and hc_scale.device == residual.device
        and hc_base.device == residual.device
        and x.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and post_layer_mix.dtype == torch.float32
        and comb_res_mix.dtype == torch.float32
        and fn.dtype == torch.float32
        and hc_scale.dtype == torch.float32
        and hc_base.dtype == torch.float32
        and x.is_contiguous()
        and residual.is_contiguous()
        and post_layer_mix.is_contiguous()
        and comb_res_mix.is_contiguous()
        and fn.is_contiguous()
        and hc_scale.is_contiguous()
        and hc_base.is_contiguous()
        and x.shape == (*outer_shape, hidden_size)
        and post_layer_mix.shape
        in (
            (*outer_shape, hc_mult, 1),
            (*outer_shape, hc_mult),
        )
        and comb_res_mix.shape == (*outer_shape, hc_mult, hc_mult)
        and fn.shape == (hc_mult3, hc_mult * hidden_size)
        and hc_scale.shape == (3,)
        and hc_base.shape == (hc_mult3,)
    )
    if not supported:
        return None

    residual_flat = residual.view(num_tokens, hc_mult, hidden_size)
    x_flat = x.view(num_tokens, hidden_size)
    post_flat = post_layer_mix.view(num_tokens, hc_mult)
    comb_flat = comb_res_mix.view(num_tokens, hc_mult, hc_mult)
    tile_n = 2 if num_tokens < 8 else 3
    split_k = 8 if num_tokens < 8 else 4

    residual_cur = torch.empty_like(residual_flat)
    gemm_out_mul = torch.empty(
        split_k,
        num_tokens,
        hc_mult3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        split_k,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )

    from vllm_musa.deepseek_v4_jit.tilelang_kernels import (
        mhc_fused_post_prenorm_kernel,
        mhc_pre_big_fuse_decode_split_kernel,
    )

    mhc_fused_post_prenorm_kernel(
        hidden_size,
        n_out=hc_mult3,
        threads=256,
        tile_n=tile_n,
        split_k=split_k,
    )(
        comb_flat,
        residual_flat,
        post_flat,
        x_flat,
        fn.view(hc_mult3, hc_mult, hidden_size),
        gemm_out_mul,
        gemm_out_sqrsum,
        residual_cur,
    )

    post_mix_cur = torch.empty(
        num_tokens,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        num_tokens,
        hc_mult * hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )
    threads, hidden_block, pass_config = _resolve_mhc_pre_big_fuse_config(
        num_tokens,
        split_k,
    )
    mhc_pre_big_fuse_decode_split_kernel(
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits=split_k,
        hc_mult=hc_mult,
        threads=threads,
        hidden_block=hidden_block,
        pass_config=pass_config,
    )(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_cur,
        post_mix_cur,
        comb_mix_cur,
        layer_input_cur,
    )

    return (
        residual_cur.view(*outer_shape, hc_mult, hidden_size),
        post_mix_cur.view(*outer_shape, hc_mult, 1),
        comb_mix_cur.view(*outer_shape, hc_mult, hc_mult),
        layer_input_cur.view(*outer_shape, hidden_size),
    )


def _reshape_hc_head_input(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
) -> tuple[torch.Tensor, torch.Size, int]:
    hc_mult = hc_fn.shape[0]
    flat_hidden = hc_fn.shape[1]
    if hidden_states.shape[-1] == flat_hidden:
        if flat_hidden % hc_mult != 0:
            raise ValueError(
                f"hc_head hidden width {flat_hidden} is not divisible by hc_mult "
                f"{hc_mult}"
            )
        hidden_size = flat_hidden // hc_mult
        token_shape = hidden_states.shape[:-1]
        grouped = hidden_states.reshape(-1, hc_mult, hidden_size)
    elif (
        hidden_states.dim() >= 2
        and hidden_states.shape[-2] * hidden_states.shape[-1] == flat_hidden
    ):
        token_shape = hidden_states.shape[:-2]
        hidden_size = hidden_states.shape[-1]
        grouped = hidden_states.reshape(-1, hidden_states.shape[-2], hidden_size)
    else:
        raise ValueError(
            "hc_head hidden_states must have trailing shape "
            f"({flat_hidden},) or (*, {flat_hidden}); got "
            f"{tuple(hidden_states.shape)}"
        )
    if grouped.shape[1] != hc_mult:
        raise ValueError(
            f"hc_head grouped dimension must match hc_mult {hc_mult}, got "
            f"{grouped.shape[1]}"
        )
    return grouped, token_shape, hidden_size


def hc_head_musa(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    dtype = hidden_states.dtype
    grouped, token_shape, hidden_size = _reshape_hc_head_input(hidden_states, hc_fn)
    grouped = grouped.to(torch.float32)
    x = grouped.reshape(grouped.shape[0], -1)
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_norm_eps)
    mixes = (x @ hc_fn.to(torch.float32).t()) * rsqrt
    pre = (
        torch.sigmoid(mixes * hc_scale.to(torch.float32) + hc_base.to(torch.float32))
        + hc_eps
    )
    y = torch.sum(pre.unsqueeze(-1) * grouped, dim=1)
    return y.reshape(*token_shape, hidden_size).to(dtype)


def _select_mhc_pre_auto_impl(residual: torch.Tensor) -> str:
    max_tilelang_tokens = int(
        os.getenv("VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_TILELANG_MAX_TOKENS", "16")
    )
    if max_tilelang_tokens <= 0:
        return "native"
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    num_tokens = residual.numel() // (hc_mult * hidden_size)
    if (
        hc_mult == 4
        and hidden_size in {4096, 7168}
        and num_tokens <= max_tilelang_tokens
    ):
        return "tilelang"
    if hc_mult == 4 and hidden_size in {4096, 7168}:
        return "deepgemm_big_fuse"
    return "native"


def mhc_pre_musa_fallback(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return mhc_pre_musa(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )


def mhc_pre_torch_fallback(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape == (hc_mult3, hc_hidden_size)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]
    residual_flat = residual.reshape(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    x_float = residual_flat.reshape(num_tokens, hc_hidden_size).to(torch.float32)

    mixes = x_float @ fn.t()
    rms = torch.rsqrt(x_float.square().sum(dim=-1) / float(hc_hidden_size) + rms_eps)
    mixes = mixes * rms.unsqueeze(-1)

    pre_mix = (
        torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + hc_pre_eps
    )
    post_mix = (
        torch.sigmoid(
            mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
            + hc_base[hc_mult : 2 * hc_mult]
        )
        * hc_post_mult_value
    )
    comb_mix = mixes[:, 2 * hc_mult :].reshape(num_tokens, hc_mult, hc_mult) * hc_scale[
        2
    ] + hc_base[2 * hc_mult :].reshape(hc_mult, hc_mult)
    comb_mix = torch.softmax(comb_mix, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = (
        (pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32))
        .sum(dim=1)
        .to(torch.bfloat16)
    )

    return (
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        layer_input.reshape(*outer_shape, hidden_size),
    )


def select_mhc_prenorm_split_k(num_tokens: int, hc_hidden_size: int) -> int:
    """Return the measured default split-K for DeepSeek-V4 MHC PreNorm."""
    if hc_hidden_size == 16384:
        if num_tokens <= 64:
            return 64
        if num_tokens <= 128:
            return 16
        if num_tokens <= 256:
            return 8
        if num_tokens <= 1024:
            return 32
        if num_tokens <= 2048:
            return 16
        return 4

    return 16 if num_tokens <= 1024 else 8


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _get_mhc_pre_deepgemm_split_k(
    num_tokens: int,
    hc_hidden_size: int,
) -> int:
    split_k = _get_env_int(
        _MHC_PRE_DEEPGEMM_SPLIT_K_ENV,
        select_mhc_prenorm_split_k(num_tokens, hc_hidden_size),
    )
    if split_k <= 0:
        raise ValueError(f"{_MHC_PRE_DEEPGEMM_SPLIT_K_ENV} must be > 0, got {split_k}")
    if hc_hidden_size % split_k != 0:
        raise ValueError(
            "DeepGEMM MHC prenorm requires K divisible by split_k, "
            f"got K={hc_hidden_size}, split_k={split_k}"
        )
    return split_k


def _select_mhc_pre_big_fuse_prenorm_impl(
    num_tokens: int,
    hc_hidden_size: int,
) -> str:
    # TileLang is faster for the tiny decode shape; DeepGEMM is the measured
    # choice for larger M and prefill.  This is deliberately shape based so an
    # A/B environment variable cannot silently select an unvalidated kernel.
    if hc_hidden_size == 16384 and num_tokens <= 64:
        return "tilelang"
    return "deepgemm"


def _mhc_prenorm_gemm_sqrsum_tilelang_decode_partials(
    residual_flat: torch.Tensor,
    fn: torch.Tensor,
    *,
    split_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_flat = residual_flat.view(residual_flat.shape[0], -1).bfloat16()
    num_tokens, hc_hidden_size = x_flat.shape
    mhc_mult3 = fn.shape[0]
    if split_k <= 0:
        raise ValueError(f"TileLang MHC prenorm split_k must be > 0, got {split_k}")
    if num_tokens > 64 or hc_hidden_size != 16384:
        raise NotImplementedError(
            "TileLang decode MHC prenorm partials only support "
            f"num_tokens <= 64 and K=16384, got num_tokens={num_tokens}, "
            f"K={hc_hidden_size}"
        )
    if hc_hidden_size % split_k != 0:
        raise ValueError(
            "TileLang MHC prenorm requires K divisible by split_k, "
            f"got K={hc_hidden_size}, split_k={split_k}"
        )
    split_size = hc_hidden_size // split_k
    if split_size % 128 != 0:
        raise ValueError(
            "TileLang decode MHC prenorm partials require split_size "
            f"divisible by 128, got split_size={split_size}"
        )

    d_part = torch.empty(
        split_k,
        num_tokens,
        mhc_mult3,
        dtype=torch.float32,
        device=residual_flat.device,
    )
    s_part = torch.empty(
        split_k,
        num_tokens,
        dtype=torch.float32,
        device=residual_flat.device,
    )

    from vllm_musa.deepseek_v4_jit.tilelang_kernels import (
        mhc_prenorm_splitk_x_tme_cast_kernel,
    )

    mhc_prenorm_splitk_x_tme_cast_kernel(
        mhc_mult3,
        hc_hidden_size,
        split_k=split_k,
        token_block=32,
        hidden_block=128,
        num_stages=2,
    )(
        x_flat,
        fn.float().contiguous(),
        d_part,
        s_part,
    )
    return d_part, s_part


def _resolve_mhc_pre_big_fuse_config(
    num_tokens: int,
    n_splits: int,
) -> tuple[int, int, str]:
    is_tiny_decode = num_tokens <= 32
    is_decode_like = num_tokens <= 64
    is_mid_prefill = 128 < num_tokens <= 512

    threads = _get_env_int(_MHC_PRE_BIG_FUSE_THREADS_ENV, 0)
    if threads <= 0:
        threads = 128 if is_tiny_decode else 256 if is_decode_like else 128
    if threads not in (128, 256):
        raise ValueError(
            f"{_MHC_PRE_BIG_FUSE_THREADS_ENV} must be 128 or 256, got {threads}"
        )

    hidden_block = _get_env_int(_MHC_PRE_BIG_FUSE_HIDDEN_BLOCK_ENV, 0)
    if hidden_block <= 0:
        hidden_block = 512 if is_tiny_decode or is_mid_prefill else 1024

    pass_config = os.getenv(_MHC_PRE_BIG_FUSE_PASS_CONFIG_ENV, "auto").strip().lower()
    if pass_config == "auto":
        pass_config = (
            "aggressive_index32"
            if (is_decode_like or is_mid_prefill) and n_splits != 1
            else "safe"
        )
    return threads, hidden_block, pass_config


def _mhc_prenorm_gemm_sqrsum_deepgemm(
    residual_flat: torch.Tensor,
    fn: torch.Tensor,
    *,
    split_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from deep_gemm.interface import tf32_hc_prenorm_gemm
    except ImportError:
        from deep_gemm import tf32_hc_prenorm_gemm

    x_flat = residual_flat.view(residual_flat.shape[0], -1).bfloat16()
    num_tokens, hc_hidden_size = x_flat.shape
    mhc_mult3 = fn.shape[0]
    if split_k <= 0:
        raise ValueError(f"DeepGEMM MHC prenorm split_k must be > 0, got {split_k}")
    if hc_hidden_size % split_k != 0:
        raise ValueError(
            "DeepGEMM MHC prenorm requires K divisible by split_k, "
            f"got K={hc_hidden_size}, split_k={split_k}"
        )

    if split_k == 1:
        d_out = torch.empty(
            num_tokens,
            mhc_mult3,
            dtype=torch.float32,
            device=residual_flat.device,
        )
        s_out = torch.empty(
            num_tokens,
            dtype=torch.float32,
            device=residual_flat.device,
        )
        tf32_hc_prenorm_gemm(
            x_flat,
            fn.float().contiguous(),
            d_out,
            s_out,
            num_splits=1,
        )
        return d_out.unsqueeze(0), s_out.unsqueeze(0)

    d_part = torch.empty(
        split_k,
        num_tokens,
        mhc_mult3,
        dtype=torch.float32,
        device=residual_flat.device,
    )
    s_part = torch.empty(
        split_k,
        num_tokens,
        dtype=torch.float32,
        device=residual_flat.device,
    )
    tf32_hc_prenorm_gemm(
        x_flat,
        fn.float().contiguous(),
        d_part,
        s_part,
        num_splits=split_k,
    )
    return d_part, s_part


def _mhc_pre_deepgemm_big_fuse_provider(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    _require_contiguous("residual", residual)
    _require_contiguous("fn", fn)
    _require_contiguous("hc_scale", hc_scale)
    _require_contiguous("hc_base", hc_base)

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    if hc_mult != 4:
        raise NotImplementedError(
            f"MHC pre DeepGEMM big-fuse provider only supports hc_mult=4, "
            f"got {hc_mult}"
        )

    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * (2 + hc_mult)
    hc_hidden_size = hc_mult * hidden_size
    if hc_mult3 > 32:
        raise NotImplementedError(
            "MHC pre DeepGEMM big-fuse provider requires hc_mult3 <= 32, "
            f"got {hc_mult3}"
        )
    if fn.shape != (hc_mult3, hc_hidden_size):
        raise ValueError(
            "MHC pre DeepGEMM big-fuse provider fn mismatch: "
            f"fn={fn.shape}, expected={(hc_mult3, hc_hidden_size)}"
        )
    if hc_scale.shape != (3,) or hc_base.shape != (hc_mult3,):
        raise ValueError(
            "MHC pre DeepGEMM big-fuse provider scale/base mismatch: "
            f"hc_scale={hc_scale.shape}, hc_base={hc_base.shape}"
        )

    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    split_k = _get_mhc_pre_deepgemm_split_k(num_tokens, hc_hidden_size)
    prenorm_impl = _select_mhc_pre_big_fuse_prenorm_impl(
        num_tokens,
        hc_hidden_size,
    )
    threads, hidden_block, pass_config = _resolve_mhc_pre_big_fuse_config(
        num_tokens,
        split_k,
    )
    if hidden_block <= 0 or hidden_size % hidden_block != 0:
        raise NotImplementedError(
            "MHC pre DeepGEMM big-fuse provider requires hidden_size divisible "
            f"by hidden_block, got hidden_size={hidden_size}, "
            f"hidden_block={hidden_block}"
        )

    post_mix = torch.empty(
        (num_tokens, hc_mult),
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix = torch.empty(
        (num_tokens, hc_mult2),
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        (num_tokens, hidden_size),
        dtype=torch.bfloat16,
        device=residual.device,
    )
    if prenorm_impl == "tilelang":
        gemm_out_mul, gemm_out_sqrsum = (
            _mhc_prenorm_gemm_sqrsum_tilelang_decode_partials(
                residual_flat,
                fn,
                split_k=split_k,
            )
        )
    else:
        gemm_out_mul, gemm_out_sqrsum = _mhc_prenorm_gemm_sqrsum_deepgemm(
            residual_flat,
            fn,
            split_k=split_k,
        )

    _require_contiguous("gemm_out_mul", gemm_out_mul)
    _require_contiguous("gemm_out_sqrsum", gemm_out_sqrsum)

    from vllm_musa.deepseek_v4_jit.tilelang_kernels import (
        mhc_pre_big_fuse_decode_split_kernel,
        mhc_pre_big_fuse_kernel,
    )

    kernel_factory = (
        mhc_pre_big_fuse_decode_split_kernel
        if num_tokens <= 64
        else mhc_pre_big_fuse_kernel
    )
    kernel_factory(
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits=gemm_out_mul.shape[0],
        hc_mult=hc_mult,
        threads=threads,
        hidden_block=hidden_block,
        pass_config=pass_config,
    )(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
    )

    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def _mhc_pre_native_provider(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    _require_contiguous("residual", residual)
    _require_contiguous("fn", fn)
    _require_contiguous("hc_scale", hc_scale)
    _require_contiguous("hc_base", hc_base)

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    if hc_mult != 4:
        raise NotImplementedError(
            f"MHC pre native provider only supports hc_mult=4, got {hc_mult}"
        )
    hc_mult3 = hc_mult * (2 + hc_mult)
    hc_hidden_size = hc_mult * hidden_size
    if fn.shape != (hc_mult3, hc_hidden_size):
        raise ValueError(
            "MHC pre native provider fn mismatch: "
            f"fn={fn.shape}, expected={(hc_mult3, hc_hidden_size)}"
        )
    if hc_scale.shape != (3,) or hc_base.shape != (hc_mult3,):
        raise ValueError(
            "MHC pre native provider scale/base mismatch: "
            f"hc_scale={hc_scale.shape}, hc_base={hc_base.shape}"
        )

    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    post_mix = torch.empty(
        (num_tokens, hc_mult), dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        (num_tokens, hc_mult, hc_mult),
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device=residual.device
    )

    from vllm_musa import _custom_ops as _musa_custom_ops

    _musa_custom_ops.deepseek_v4_mhc_pre(
        residual_flat,
        fn,
        hc_scale,
        hc_base,
        post_mix,
        comb_mix,
        layer_input,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def _mhc_pre_tilelang_provider(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    _require_contiguous("residual", residual)
    _require_contiguous("fn", fn)
    _require_contiguous("hc_scale", hc_scale)
    _require_contiguous("hc_base", hc_base)

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    if hc_mult != 4:
        raise NotImplementedError(
            f"MHC pre TileLang provider only supports hc_mult=4, got {hc_mult}"
        )

    hc_mult3 = hc_mult * (2 + hc_mult)
    hc_hidden_size = hc_mult * hidden_size
    if fn.shape != (hc_mult3, hc_hidden_size):
        raise ValueError(
            "MHC pre TileLang provider fn mismatch: "
            f"fn={fn.shape}, expected={(hc_mult3, hc_hidden_size)}"
        )
    if hidden_size % 256 != 0:
        raise NotImplementedError(
            "MHC pre TileLang provider requires hidden_size divisible by 256, "
            f"got {hidden_size}"
        )

    from vllm_musa.deepseek_v4_jit.tilelang_kernels import (
        mhc_pre_split_sinkhorn_kernel,
    )

    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    x_float = residual_flat.view(num_tokens, hc_hidden_size).to(torch.float32)
    mixes = x_float @ fn.t()
    rms = torch.rsqrt(x_float.square().sum(dim=-1) / float(hc_hidden_size) + rms_eps)
    mixes = (mixes * rms.unsqueeze(-1)).contiguous()

    post_mix = torch.empty(
        (num_tokens, hc_mult), dtype=torch.float32, device=residual.device
    )
    pre_mix = torch.empty(
        (num_tokens, hc_mult), dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        (num_tokens, hc_mult * hc_mult),
        dtype=torch.float32,
        device=residual.device,
    )

    mhc_pre_split_sinkhorn_kernel(hc_mult, sinkhorn_repeat)(
        mixes,
        hc_scale,
        hc_base,
        pre_mix,
        post_mix,
        comb_mix,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
    )
    layer_input = (
        (pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32))
        .sum(dim=1)
        .to(torch.bfloat16)
    )

    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def mhc_post_musa(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    try:
        return _mhc_post_tilelang_provider(x, residual, post_layer_mix, comb_res_mix)
    except (ImportError, OSError, NotImplementedError, RuntimeError):
        return mhc_post_torch_fallback(x, residual, post_layer_mix, comb_res_mix)


def mhc_post_musa_fallback(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return mhc_post_musa(x, residual, post_layer_mix, comb_res_mix)


def mhc_post_torch_fallback(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    outer_shape = residual.shape[:-2]
    hc = residual.shape[-2]
    hidden = residual.shape[-1]
    residual_flat = residual.reshape(-1, hc, hidden).to(torch.float32)
    x_flat = x.reshape(-1, hidden).to(torch.float32)
    post_flat = post_layer_mix.reshape(-1, hc, 1).to(torch.float32)
    comb_flat = comb_res_mix.reshape(-1, hc, hc).to(torch.float32)

    out = torch.einsum("tij,tih->tjh", comb_flat, residual_flat)
    out = out + post_flat * x_flat.unsqueeze(1)
    return out.to(residual.dtype).reshape(*outer_shape, hc, hidden)


def _require_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_contiguous():
        raise NotImplementedError(f"MHC TileLang provider requires contiguous {name}")


def _mhc_post_tilelang_provider(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    assert residual.dtype == torch.bfloat16
    assert post_layer_mix.dtype == torch.float32
    assert comb_res_mix.dtype == torch.float32

    _require_contiguous("x", x)
    _require_contiguous("residual", residual)
    _require_contiguous("post_layer_mix", post_layer_mix)
    _require_contiguous("comb_res_mix", comb_res_mix)

    outer_shape = residual.shape[:-2]
    hc = residual.shape[-2]
    hidden = residual.shape[-1]
    if hc != 4:
        raise NotImplementedError(
            f"MHC post TileLang provider only supports hc_mult=4, got {hc}"
        )

    x_flat = x.view(-1, hidden)
    residual_flat = residual.view(-1, hc, hidden)
    post_flat = post_layer_mix.view(-1, hc, 1).squeeze(-1)
    comb_flat = comb_res_mix.view(-1, hc, hc)
    if x_flat.shape[0] != residual_flat.shape[0]:
        raise ValueError(
            "MHC post TileLang provider token mismatch: "
            f"x={x_flat.shape}, residual={residual_flat.shape}"
        )

    from vllm_musa.deepseek_v4_jit.tilelang_kernels import mhc_post_kernel

    out = torch.empty_like(residual_flat)
    mhc_post_kernel(hidden)(x_flat, residual_flat, post_flat, comb_flat, out)
    return out.view(*outer_shape, hc, hidden)
