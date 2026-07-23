# SPDX-License-Identifier: Apache-2.0
"""TileLang kernels for DeepSeek-V4 MUSA JIT helpers."""

import math
from functools import lru_cache

import tilelang
import tilelang.language as T

from .kernel_common import (
    _add_pass_config,
    _patch_tilelang_musa_wrapper,
    _tilelang_musa_aggressive_pass_configs,
    _tilelang_musa_burst_reduce_pass_configs,
    _tilelang_musa_pass_configs,
)

_patch_tilelang_musa_wrapper()

HIDDEN_SIZE = 512
NOPE_DIM = 448
ROPE_DIM = 64
HALF_ROPE_DIM = ROPE_DIM // 2
SCALE_DIM = NOPE_DIM // 64
TOKEN_VALUE_BYTES = NOPE_DIM + ROPE_DIM * 2
TOKEN_SCALE_BYTES = SCALE_DIM + 1
FP8_MAX = 448.0


def _warp_reduce_sum(value):
    mask = T.tvm_warp_activemask()
    value += T.tvm_warp_shuffle_down(mask, value, 16, 32, 32)
    value += T.tvm_warp_shuffle_down(mask, value, 8, 32, 32)
    value += T.tvm_warp_shuffle_down(mask, value, 4, 32, 32)
    value += T.tvm_warp_shuffle_down(mask, value, 2, 32, 32)
    value += T.tvm_warp_shuffle_down(mask, value, 1, 32, 32)
    return T.tvm_warp_shuffle(mask, value, 0, 32, 32)


def _warp_reduce_max(value):
    mask = T.tvm_warp_activemask()
    value = T.max(value, T.tvm_warp_shuffle_down(mask, value, 16, 32, 32))
    value = T.max(value, T.tvm_warp_shuffle_down(mask, value, 8, 32, 32))
    value = T.max(value, T.tvm_warp_shuffle_down(mask, value, 4, 32, 32))
    value = T.max(value, T.tvm_warp_shuffle_down(mask, value, 2, 32, 32))
    value = T.max(value, T.tvm_warp_shuffle_down(mask, value, 1, 32, 32))
    return T.tvm_warp_shuffle(mask, value, 0, 32, 32)


def _mhc_pre_big_fuse_pass_configs(tilelang_module, mode: str):
    mode = mode.strip().lower()
    if mode == "none":
        return None
    if mode == "safe":
        return _tilelang_musa_pass_configs(tilelang_module)
    if mode == "burst":
        return _tilelang_musa_burst_reduce_pass_configs(tilelang_module)
    if mode == "aggressive":
        return _tilelang_musa_aggressive_pass_configs(
            tilelang_module,
            disable_index_promotion=False,
        )
    if mode == "aggressive_index32":
        return _tilelang_musa_aggressive_pass_configs(
            tilelang_module,
            disable_index_promotion=True,
        )
    raise ValueError(
        "MHC pre big-fuse pass_config must be one of "
        "'safe', 'burst', 'aggressive', 'aggressive_index32', or 'none', "
        f"got {mode!r}"
    )


def _mhc_tme_ws_pass_configs(tilelang_module):
    pass_configs = {}
    _add_pass_config(pass_configs, tilelang_module, "TL_ENABLE_MUSA_TMA_PREFETCH", True)
    _add_pass_config(
        pass_configs,
        tilelang_module,
        "TL_DISABLE_WARP_SPECIALIZED",
        False,
    )
    _add_pass_config(pass_configs, tilelang_module, "TL_DISABLE_TMA_LOWER", False)
    _add_pass_config(
        pass_configs,
        tilelang_module,
        "TL_DISABLE_THREAD_STORAGE_SYNC",
        True,
    )
    return pass_configs or None


@lru_cache(maxsize=None)
def mhc_pre_split_sinkhorn_kernel(hc_mult: int, sinkhorn_repeat: int):
    hc_mult3 = hc_mult * (2 + hc_mult)

    @tilelang.jit(target="musa", pass_configs=_tilelang_musa_pass_configs(tilelang))
    def _mhc_pre_split_sinkhorn_kernel():
        num_tokens = T.dynamic("num_tokens")

        @T.prim_func
        def _kernel(
            mixes: T.Tensor((num_tokens, hc_mult3), T.float32),
            hc_scale: T.Tensor((3,), T.float32),
            hc_base: T.Tensor((hc_mult3,), T.float32),
            pre_mix: T.Tensor((num_tokens, hc_mult), T.float32),
            post_mix: T.Tensor((num_tokens, hc_mult), T.float32),
            comb_mix: T.Tensor((num_tokens, hc_mult * hc_mult), T.float32),
            hc_pre_eps: T.float32,
            hc_sinkhorn_eps: T.float32,
            hc_post_mult_value: T.float32,
        ):
            with T.Kernel(num_tokens, threads=64) as token_id:
                mixes_shared = T.alloc_shared((hc_mult3,), T.float32)
                T.copy(mixes[token_id, 0], mixes_shared)

                for j in T.Parallel(hc_mult):
                    pre_mix[token_id, j] = (
                        T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j])
                        + hc_pre_eps
                    )
                for j in T.Parallel(hc_mult):
                    post_mix[token_id, j] = (
                        T.sigmoid(
                            mixes_shared[j + hc_mult] * hc_scale[1]
                            + hc_base[j + hc_mult]
                        )
                        * hc_post_mult_value
                    )

                cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = (
                        mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                        + hc_base[j * hc_mult + k + hc_mult * 2]
                    )

                row_sum = T.alloc_fragment((hc_mult,), T.float32)
                col_sum = T.alloc_fragment((hc_mult,), T.float32)
                row_max = T.alloc_fragment((hc_mult,), T.float32)
                T.reduce_max(cm, row_max, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = T.exp(cm[j, k] - row_max[j])
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

                for _ in T.serial(sinkhorn_repeat - 1):
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)
                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

                for j, k in T.Parallel(hc_mult, hc_mult):
                    comb_mix[token_id, j * hc_mult + k] = cm[j, k]

        return _kernel

    return _mhc_pre_split_sinkhorn_kernel()


@lru_cache(maxsize=None)
def mhc_prenorm_splitk_x_tme_cast_kernel(
    mhc_mult3: int,
    hc_hidden_size: int,
    split_k: int,
    token_block: int = 32,
    hidden_block: int = 128,
    num_stages: int = 2,
    threads: int = 384,
):
    num_tokens = T.dynamic("num_tokens")
    assert mhc_mult3 <= 32
    assert hc_hidden_size % hidden_block == 0
    assert hc_hidden_size % split_k == 0
    split_size = hc_hidden_size // split_k
    assert split_size % hidden_block == 0
    assert hidden_block in (64, 128)
    assert num_stages == 2
    assert token_block == 32
    assert threads >= 384
    mbarrier_list = [128, 128, 128] * num_stages

    @tilelang.jit(
        target="musa",
        pass_configs=_mhc_tme_ws_pass_configs(tilelang),
    )
    def _mhc_prenorm_splitk_x_tme_cast_kernel():
        @T.prim_func
        def _kernel(
            x: T.Tensor((num_tokens, hc_hidden_size), T.bfloat16),
            fn: T.Tensor((mhc_mult3, hc_hidden_size), T.float32),
            out_partial: T.Tensor(
                (split_k, num_tokens, mhc_mult3),
                T.float32,
            ),
            sqrsum_partial: T.Tensor((split_k, num_tokens), T.float32),
        ):
            with T.Kernel(
                T.ceildiv(num_tokens, token_block),
                split_k,
                threads=threads,
            ) as (px, bz):
                x_bf16_shared = T.alloc_shared(
                    (token_block, hidden_block),
                    T.bfloat16,
                )
                x_fp32_shared = T.alloc_shared(
                    (token_block, hidden_block),
                    T.float32,
                )
                fn_shared = T.alloc_shared((32, hidden_block), T.float32)
                out_frag = T.alloc_fragment((token_block, 32), T.float32)
                sq_part4 = T.alloc_fragment((token_block, 4), T.float32)
                mbars = T.alloc_barrier(mbarrier_list)
                k_base = bz * split_size

                with T.ws(0):
                    T.clear(out_frag)
                    T.clear(sq_part4)

                for pz in T.serial(split_size // hidden_block):
                    with T.ws(1):
                        T.mbarrier_wait_parity(
                            mbarrier=mbars[2],
                            parity=(pz % 2) ^ 1,
                        )
                        T.copy(
                            x[
                                px * token_block : (px + 1) * token_block,
                                k_base
                                + pz * hidden_block : k_base
                                + (pz + 1) * hidden_block,
                            ],
                            x_bf16_shared,
                            barrier=mbars[0],
                            annotations={"musa_tma_k_major": T.int32(1)},
                        )
                        T.copy(
                            fn[
                                0:32,
                                k_base
                                + pz * hidden_block : k_base
                                + (pz + 1) * hidden_block,
                            ],
                            fn_shared,
                            barrier=mbars[1],
                            eviction_policy="evict_first",
                        )
                        T.mbarrier_arrive(mbarrier=mbars[0])
                        T.mbarrier_arrive(mbarrier=mbars[1])

                    with T.ws(0):
                        T.mbarrier_wait_parity(
                            mbarrier=mbars[0],
                            parity=pz % 2,
                        )
                        T.mbarrier_wait_parity(
                            mbarrier=mbars[1],
                            parity=pz % 2,
                        )
                        for i, k in T.Parallel(token_block, hidden_block):
                            x_fp32_shared[i, k] = T.cast(
                                x_bf16_shared[i, k],
                                T.float32,
                            )
                        for jj in T.serial(hidden_block // 4):
                            for i, j in T.Parallel(token_block, 4):
                                v = T.cast(
                                    x_bf16_shared[i, jj * 4 + j],
                                    T.float32,
                                )
                                sq_part4[i, j] += v * v
                        T.gemm(
                            x_fp32_shared,
                            fn_shared,
                            out_frag,
                            clear_accum=False,
                            transpose_B=True,
                        )
                        T.mbarrier_arrive(mbarrier=mbars[2])

                with T.ws(0):
                    sq_l = T.alloc_fragment((token_block,), T.float32)
                    T.reduce_sum(sq_part4, sq_l)
                    for i in T.Parallel(token_block):
                        t = px * token_block + i
                        if t < num_tokens:
                            sqrsum_partial[bz, t] = sq_l[i]
                    for i, j in T.Parallel(token_block, 32):
                        t = px * token_block + i
                        if t < num_tokens and j < mhc_mult3:
                            out_partial[bz, t, j] = out_frag[i, j]

        return _kernel

    return _mhc_prenorm_splitk_x_tme_cast_kernel()


@lru_cache(maxsize=None)
def mhc_fused_post_prenorm_kernel(
    hidden_size: int,
    n_out: int = 24,
    threads: int = 256,
    tile_n: int = 2,
    split_k: int = 8,
):
    """Fuse MHC post mapping with the next prenorm projection."""
    num_tokens = T.dynamic("num_tokens")
    hc_mult = 4
    hidden_per_split = hidden_size // split_k
    n_tiles = n_out // tile_n

    assert threads == 256
    assert split_k in (4, 8)
    assert hidden_size % split_k == 0
    assert hidden_per_split % threads == 0
    assert tile_n in (2, 3)
    assert n_out % tile_n == 0

    @tilelang.jit(
        target="musa",
        pass_configs=_mhc_pre_big_fuse_pass_configs(tilelang, "safe"),
    )
    def _mhc_fused_post_prenorm_kernel():
        @T.prim_func
        def _kernel(
            comb_mix: T.Tensor((num_tokens, hc_mult, hc_mult), T.float32),
            residual_in: T.Tensor(
                (num_tokens, hc_mult, hidden_size),
                T.bfloat16,
            ),
            post_mix: T.Tensor((num_tokens, hc_mult), T.float32),
            x_in: T.Tensor((num_tokens, hidden_size), T.bfloat16),
            weight: T.Tensor(
                (n_out, hc_mult, hidden_size),
                T.float32,
            ),
            out_partial: T.Tensor(
                (split_k, num_tokens, n_out),
                T.float32,
            ),
            sqrsum_partial: T.Tensor(
                (split_k, num_tokens),
                T.float32,
            ),
            residual_out: T.Tensor(
                (num_tokens, hc_mult, hidden_size),
                T.bfloat16,
            ),
        ):
            with T.Kernel(
                num_tokens,
                n_tiles,
                split_k,
                threads=threads,
            ) as (token_id, out_tile_id, split_id):
                thread_id = T.get_thread_binding()
                warp_id = thread_id // 32
                lane_id = thread_id % 32
                num_warps = threads // 32

                warp_partial = T.alloc_shared(
                    (num_warps, tile_n + 1),
                    T.float32,
                )
                post_shared = T.alloc_shared((hc_mult,), T.float32)
                comb_shared = T.alloc_shared(
                    (hc_mult, hc_mult),
                    T.float32,
                )
                post_local = T.alloc_local((hc_mult,), T.float32)
                comb_local = T.alloc_local(
                    (hc_mult, hc_mult),
                    T.float32,
                )
                accum = T.alloc_local((tile_n,), T.float32)
                sqrsum = T.alloc_local((1,), T.float32)
                new_residual = T.alloc_local((hc_mult,), T.float32)

                T.clear(accum)
                T.clear(sqrsum)
                hidden_start = split_id * hidden_per_split

                T.copy(post_mix[token_id, 0], post_shared)
                T.copy(comb_mix[token_id, 0, 0], comb_shared)
                for row in T.unroll(hc_mult):
                    post_local[row] = post_shared[row]
                    for source_row in T.unroll(hc_mult):
                        comb_local[source_row, row] = comb_shared[source_row, row]

                for hidden_iter in T.serial(hidden_per_split // threads):
                    hidden_idx = hidden_start + hidden_iter * threads + thread_id

                    for row in T.unroll(hc_mult):
                        new_residual[row] = (
                            post_local[row] * x_in[token_id, hidden_idx]
                        )
                        for source_row in T.unroll(hc_mult):
                            new_residual[row] += (
                                comb_local[source_row, row]
                                * residual_in[token_id, source_row, hidden_idx]
                            )
                        # Match the decomposed path: post stores bf16 before
                        # prenorm consumes the value.
                        new_residual[row] = T.cast(
                            new_residual[row],
                            T.bfloat16,
                        )

                    if out_tile_id == 0:
                        for row in T.unroll(hc_mult):
                            residual_out[token_id, row, hidden_idx] = T.cast(
                                new_residual[row],
                                T.bfloat16,
                            )
                            sqrsum[0] += new_residual[row] * new_residual[row]

                    for out_idx in T.unroll(tile_n):
                        weight_row = out_tile_id * tile_n + out_idx
                        for row in T.unroll(hc_mult):
                            accum[out_idx] += (
                                weight[weight_row, row, hidden_idx]
                                * new_residual[row]
                            )

                for out_idx in T.unroll(tile_n):
                    accum[out_idx] = T.warp_reduce_sum(accum[out_idx])
                if out_tile_id == 0:
                    sqrsum[0] = T.warp_reduce_sum(sqrsum[0])

                if lane_id == 0:
                    for out_idx in T.unroll(tile_n):
                        warp_partial[warp_id, out_idx] = accum[out_idx]
                    if out_tile_id == 0:
                        warp_partial[warp_id, tile_n] = sqrsum[0]
                T.sync_threads()

                if warp_id == 0:
                    if lane_id < tile_n:
                        reduced = T.alloc_var(T.float32, init=0.0)
                        for source_warp in T.unroll(num_warps):
                            reduced += warp_partial[source_warp, lane_id]
                        out_partial[
                            split_id,
                            token_id,
                            out_tile_id * tile_n + lane_id,
                        ] = reduced
                    if out_tile_id == 0 and lane_id == 0:
                        reduced_sqrsum = T.alloc_var(T.float32, init=0.0)
                        for source_warp in T.unroll(num_warps):
                            reduced_sqrsum += warp_partial[source_warp, tile_n]
                        sqrsum_partial[split_id, token_id] = reduced_sqrsum

        return _kernel

    return _mhc_fused_post_prenorm_kernel()


@lru_cache(maxsize=None)
def mhc_pre_big_fuse_kernel(
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    hc_mult: int = 4,
    threads: int = 256,
    hidden_block: int = 512,
    pass_config: str = "burst",
):
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    assert hc_mult == 4
    assert hc_mult3 <= 32
    assert threads in (128, 256)
    assert n_splits > 0
    assert sinkhorn_repeat > 0
    assert hidden_block > 0
    assert hidden_size % hidden_block == 0

    @tilelang.jit(
        target="musa",
        pass_configs=_mhc_pre_big_fuse_pass_configs(tilelang, pass_config),
    )
    def _mhc_pre_big_fuse_kernel():
        @T.prim_func
        def _kernel(
            gemm_out_mul: T.Tensor((n_splits, num_tokens, hc_mult3), T.float32),
            gemm_out_sqrsum: T.Tensor((n_splits, num_tokens), T.float32),
            hc_scale: T.Tensor((3,), T.float32),
            hc_base: T.Tensor((hc_mult3,), T.float32),
            residual: T.Tensor((num_tokens, hc_mult, hidden_size), T.bfloat16),
            post_mix: T.Tensor((num_tokens, hc_mult), T.float32),
            comb_mix: T.Tensor((num_tokens, hc_mult * hc_mult), T.float32),
            layer_input: T.Tensor((num_tokens, hidden_size), T.bfloat16),
        ):
            with T.Kernel(num_tokens, threads=threads) as token_id:
                mixes_shared = T.alloc_shared((hc_mult3,), T.float32)
                pre_mix_shared = T.alloc_shared((hc_mult,), T.float32)
                tx = T.get_thread_binding()

                if tx < 32:
                    rms = T.alloc_fragment((1,), T.float32)
                    mixes = T.alloc_fragment((hc_mult3,), T.float32)
                    T.clear(mixes)
                    if n_splits == 1:
                        rms[0] = gemm_out_sqrsum[0, token_id]
                    else:
                        rms_part = T.alloc_fragment((1,), T.float32)
                        rms_part[0] = 0.0
                        for split_base in T.serial(T.ceildiv(n_splits, 32)):
                            split_id = split_base * 32 + tx
                            rms_part[0] += T.if_then_else(
                                split_id < n_splits,
                                gemm_out_sqrsum[split_id, token_id],
                                0.0,
                            )
                        rms[0] = T.warp_reduce_sum(rms_part[0])
                    rms[0] = T.rsqrt(rms[0] / float(hc_mult * hidden_size) + rms_eps)
                    for j in T.Parallel(hc_mult3):
                        mixes[j] = 0.0
                        for split_id in T.serial(n_splits):
                            mixes[j] += gemm_out_mul[split_id, token_id, j]
                        mixes[j] *= rms[0]
                    T.copy(mixes, mixes_shared, disable_tma=True)

                T.sync_threads()

                if tx < 32:
                    cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
                    for j in T.Parallel(hc_mult):
                        pre_mix_shared[j] = (
                            T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j])
                            + hc_pre_eps
                        )
                    for j in T.Parallel(hc_mult):
                        post_mix[token_id, j] = (
                            T.sigmoid(
                                mixes_shared[j + hc_mult] * hc_scale[1]
                                + hc_base[j + hc_mult]
                            )
                            * hc_post_mult_value
                        )
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = (
                            mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                            + hc_base[j * hc_mult + k + hc_mult * 2]
                        )

                    row_sum = T.alloc_fragment((hc_mult,), T.float32)
                    col_sum = T.alloc_fragment((hc_mult,), T.float32)
                    row_max = T.alloc_fragment((hc_mult,), T.float32)
                    T.reduce_max(cm, row_max, dim=1)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = T.exp(cm[j, k] - row_max[j])
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

                    for _ in T.serial(sinkhorn_repeat - 1):
                        T.reduce_sum(cm, row_sum, dim=1)
                        for j, k in T.Parallel(hc_mult, hc_mult):
                            cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)
                        T.reduce_sum(cm, col_sum, dim=0)
                        for j, k in T.Parallel(hc_mult, hc_mult):
                            cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

                    for j, k in T.Parallel(hc_mult, hc_mult):
                        comb_mix[token_id, j * hc_mult + k] = cm[j, k]

                T.sync_threads()

                for hidden_base in T.Pipelined(
                    hidden_size // hidden_block,
                    num_stages=1,
                ):
                    layer = T.alloc_fragment((hidden_block,), T.float32)
                    T.clear(layer)

                    for hc_id in T.serial(hc_mult):
                        pre = pre_mix_shared[hc_id]
                        for i in T.Parallel(hidden_block):
                            h = hidden_base * hidden_block + i
                            layer[i] += pre * T.cast(
                                residual[token_id, hc_id, h],
                                T.float32,
                            )

                    T.copy(
                        layer,
                        layer_input[token_id, hidden_base * hidden_block],
                        disable_tma=True,
                    )

        return _kernel

    return _mhc_pre_big_fuse_kernel()


@lru_cache(maxsize=None)
def mhc_pre_big_fuse_decode_split_kernel(
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    hc_mult: int = 4,
    threads: int = 256,
    hidden_block: int = 512,
    pass_config: str = "burst",
):
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    num_hidden_tiles = hidden_size // hidden_block
    assert hc_mult == 4
    assert hc_mult3 <= 32
    assert threads in (128, 256)
    assert n_splits > 0
    assert sinkhorn_repeat > 0
    assert hidden_block > 0
    assert hidden_size % hidden_block == 0

    @tilelang.jit(
        target="musa",
        pass_configs=_mhc_pre_big_fuse_pass_configs(tilelang, pass_config),
    )
    def _mhc_pre_big_fuse_decode_split_kernel():
        @T.prim_func
        def _kernel(
            gemm_out_mul: T.Tensor((n_splits, num_tokens, hc_mult3), T.float32),
            gemm_out_sqrsum: T.Tensor((n_splits, num_tokens), T.float32),
            hc_scale: T.Tensor((3,), T.float32),
            hc_base: T.Tensor((hc_mult3,), T.float32),
            residual: T.Tensor((num_tokens, hc_mult, hidden_size), T.bfloat16),
            post_mix: T.Tensor((num_tokens, hc_mult), T.float32),
            comb_mix: T.Tensor((num_tokens, hc_mult * hc_mult), T.float32),
            layer_input: T.Tensor((num_tokens, hidden_size), T.bfloat16),
        ):
            with T.Kernel(
                num_tokens,
                num_hidden_tiles,
                threads=threads,
            ) as (token_id, hidden_tile_id):
                tx = T.get_thread_binding()
                mixes_shared = T.alloc_shared((hc_mult3,), T.float32)
                rms_for_layer = T.alloc_fragment((1,), T.float32)
                pre_mix = T.alloc_fragment((hc_mult,), T.float32)

                rms_for_layer[0] = 0.0
                for split_id in T.serial(n_splits):
                    rms_for_layer[0] += gemm_out_sqrsum[split_id, token_id]
                rms_for_layer[0] = T.rsqrt(
                    rms_for_layer[0] / float(hc_mult * hidden_size) + rms_eps
                )
                for j in T.serial(hc_mult):
                    pre_mix[j] = 0.0
                    for split_id in T.serial(n_splits):
                        pre_mix[j] += gemm_out_mul[split_id, token_id, j]
                    pre_mix[j] = (
                        T.sigmoid(
                            pre_mix[j] * rms_for_layer[0] * hc_scale[0] + hc_base[j]
                        )
                        + hc_pre_eps
                    )

                if tx < 32:
                    rms = T.alloc_fragment((1,), T.float32)
                    mixes = T.alloc_fragment((hc_mult3,), T.float32)
                    T.clear(mixes)
                    if n_splits == 1:
                        rms[0] = gemm_out_sqrsum[0, token_id]
                    else:
                        rms_part = T.alloc_fragment((1,), T.float32)
                        rms_part[0] = 0.0
                        for split_base in T.serial(T.ceildiv(n_splits, 32)):
                            split_id = split_base * 32 + tx
                            rms_part[0] += T.if_then_else(
                                split_id < n_splits,
                                gemm_out_sqrsum[split_id, token_id],
                                0.0,
                            )
                        rms[0] = T.warp_reduce_sum(rms_part[0])
                    rms[0] = T.rsqrt(rms[0] / float(hc_mult * hidden_size) + rms_eps)
                    if hidden_tile_id == 0:
                        for j in T.Parallel(hc_mult3):
                            mixes[j] = 0.0
                            for split_id in T.serial(n_splits):
                                mixes[j] += gemm_out_mul[split_id, token_id, j]
                            mixes[j] *= rms[0]
                        T.copy(mixes, mixes_shared, disable_tma=True)

                T.sync_threads()

                if tx < 32 and hidden_tile_id == 0:
                    cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
                    for j in T.Parallel(hc_mult):
                        post_mix[token_id, j] = (
                            T.sigmoid(
                                mixes_shared[j + hc_mult] * hc_scale[1]
                                + hc_base[j + hc_mult]
                            )
                            * hc_post_mult_value
                        )
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = (
                            mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                            + hc_base[j * hc_mult + k + hc_mult * 2]
                        )

                    row_sum = T.alloc_fragment((hc_mult,), T.float32)
                    col_sum = T.alloc_fragment((hc_mult,), T.float32)
                    row_max = T.alloc_fragment((hc_mult,), T.float32)
                    T.reduce_max(cm, row_max, dim=1)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = T.exp(cm[j, k] - row_max[j])
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

                    for _ in T.serial(sinkhorn_repeat - 1):
                        T.reduce_sum(cm, row_sum, dim=1)
                        for j, k in T.Parallel(hc_mult, hc_mult):
                            cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)
                        T.reduce_sum(cm, col_sum, dim=0)
                        for j, k in T.Parallel(hc_mult, hc_mult):
                            cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

                    for j, k in T.Parallel(hc_mult, hc_mult):
                        comb_mix[token_id, j * hc_mult + k] = cm[j, k]

                layer = T.alloc_fragment((hidden_block,), T.float32)
                T.clear(layer)
                for hc_id in T.serial(hc_mult):
                    pre = pre_mix[hc_id]
                    for i in T.Parallel(hidden_block):
                        h = hidden_tile_id * hidden_block + i
                        layer[i] += pre * T.cast(
                            residual[token_id, hc_id, h],
                            T.float32,
                        )

                for i in T.Parallel(hidden_block):
                    layer_input[token_id, hidden_tile_id * hidden_block + i] = T.cast(
                        layer[i],
                        T.bfloat16,
                    )

        return _kernel

    return _mhc_pre_big_fuse_decode_split_kernel()


@lru_cache(maxsize=None)
def mhc_weighted_rmsnorm_kernel(
    hidden_size: int,
    threads: int = 128,
):
    num_rows = T.dynamic("num_rows")
    warps_per_cta = threads // 32
    assert hidden_size > 0
    assert hidden_size % threads == 0
    assert threads in (64, 128, 256)

    @tilelang.jit(target="musa", pass_configs=_tilelang_musa_pass_configs(tilelang))
    def _mhc_weighted_rmsnorm_kernel():
        @T.prim_func
        def _kernel(
            x: T.Tensor((num_rows, hidden_size), T.bfloat16),
            weight: T.Tensor((hidden_size,), T.bfloat16),
            out: T.Tensor((num_rows, hidden_size), T.bfloat16),
            eps: T.float32,
        ):
            with T.Kernel(num_rows, threads=threads) as row_id:
                tx = T.get_thread_binding()
                lane = tx % 32
                warp = tx // 32
                partial_sumsq = T.alloc_local((1,), T.float32)
                warp_sumsq = T.alloc_shared((warps_per_cta,), T.float32)

                partial_sumsq[0] = 0.0
                for col_base in T.serial(0, hidden_size, threads):
                    col = col_base + tx
                    value = T.cast(x[row_id, col], T.float32)
                    partial_sumsq[0] += value * value

                partial_sumsq[0] = _warp_reduce_sum(partial_sumsq[0])
                if lane == 0:
                    warp_sumsq[warp] = partial_sumsq[0]
                T.sync_threads()

                partial_sumsq[0] = T.if_then_else(
                    tx < warps_per_cta,
                    warp_sumsq[tx],
                    0.0,
                )
                if warp == 0:
                    partial_sumsq[0] = _warp_reduce_sum(partial_sumsq[0])
                    if lane == 0:
                        warp_sumsq[0] = T.rsqrt(
                            partial_sumsq[0] / float(hidden_size) + eps
                        )
                T.sync_threads()

                for col_base in T.serial(0, hidden_size, threads):
                    col = col_base + tx
                    value = T.cast(x[row_id, col], T.float32)
                    weight_value = T.cast(weight[col], T.float32)
                    out[row_id, col] = T.cast(
                        value * warp_sumsq[0] * weight_value,
                        T.bfloat16,
                    )

        return _kernel

    return _mhc_weighted_rmsnorm_kernel()


@lru_cache(maxsize=None)
def mhc_weighted_rmsnorm_mudnn_like_kernel(
    hidden_size: int,
    threads: int = 128,
):
    """Low-latency weighted RMSNorm for DeepSeek-V4 MHC at any token M."""
    num_rows = T.dynamic("num_rows")
    elements_per_thread = 8
    chunks = hidden_size // (threads * elements_per_thread)
    assert hidden_size > 0
    assert hidden_size % (threads * elements_per_thread) == 0
    assert threads == 128

    @tilelang.jit(target="musa", pass_configs={})
    def _mhc_weighted_rmsnorm_mudnn_like_kernel():
        @T.prim_func
        def _kernel(
            x: T.Tensor((num_rows, hidden_size), T.bfloat16),
            weight: T.Tensor((hidden_size,), T.bfloat16),
            out: T.Tensor((num_rows, hidden_size), T.bfloat16),
            eps: T.float32,
        ):
            with T.Kernel(num_rows, threads=threads) as row_id:
                tx = T.get_thread_binding()
                sumsq = T.alloc_local((1,), T.float32)
                shared = T.alloc_shared((threads,), T.float32)
                rrms = T.alloc_local((1,), T.float32)
                rsqrt_estimate = T.alloc_local((1,), T.float32)

                sumsq[0] = 0.0
                for chunk in T.serial(chunks):
                    base = (
                        chunk * threads * elements_per_thread
                        + tx * elements_per_thread
                    )
                    for elem in T.serial(elements_per_thread):
                        value = T.cast(x[row_id, base + elem], T.float32)
                        sumsq[0] += value * value

                if threads >= 128:
                    if tx >= 64:
                        shared[tx] = sumsq[0]
                    T.sync_threads()
                    if tx < 64:
                        sumsq[0] += shared[tx + 64]
                if threads >= 64:
                    T.sync_threads()
                    if tx >= 32:
                        shared[tx] = sumsq[0]
                    T.sync_threads()
                    if tx < 32:
                        sumsq[0] += shared[tx + 32]
                if tx < 32:
                    sumsq[0] += T.shfl_down(sumsq[0], 16)
                    sumsq[0] += T.shfl_down(sumsq[0], 8)
                    sumsq[0] += T.shfl_down(sumsq[0], 4)
                    sumsq[0] += T.shfl_down(sumsq[0], 2)
                    sumsq[0] += T.shfl_down(sumsq[0], 1)
                if tx == 0:
                    shared[0] = sumsq[0]
                T.sync_threads()

                sumsq[0] = shared[0] / float(hidden_size) + eps
                # Match the fast muDNN sequence: hardware reciprocal sqrt
                # followed by one Newton refinement.
                rsqrt_estimate[0] = T.ieee_frsqrt(sumsq[0])
                rrms[0] = rsqrt_estimate[0] * (
                    T.cast(1.5, T.float32)
                    - T.cast(0.5, T.float32)
                    * sumsq[0]
                    * rsqrt_estimate[0]
                    * rsqrt_estimate[0]
                )

                for chunk in T.serial(chunks):
                    base = (
                        chunk * threads * elements_per_thread
                        + tx * elements_per_thread
                    )
                    for elem in T.serial(elements_per_thread):
                        value = T.cast(x[row_id, base + elem], T.float32)
                        weight_value = T.cast(weight[base + elem], T.float32)
                        out[row_id, base + elem] = T.cast(
                            (value * rrms[0]) * weight_value,
                            T.bfloat16,
                        )

        return _kernel

    return _mhc_weighted_rmsnorm_mudnn_like_kernel()


@tilelang.jit(target="musa", pass_configs=_tilelang_musa_pass_configs(tilelang))
def qnorm_rope_kernel():
    num_tokens = T.dynamic("num_tokens")
    num_heads = T.dynamic("num_heads")
    num_positions = T.dynamic("num_positions")
    threads = 256
    warps_per_cta = threads // 32

    @T.prim_func
    def _qnorm_rope_kernel(
        q: T.Tensor((num_tokens, num_heads, HIDDEN_SIZE), T.bfloat16),
        out: T.Tensor((num_tokens, num_heads, HIDDEN_SIZE), T.bfloat16),
        cos_sin_cache: T.Tensor((num_positions, ROPE_DIM), T.float32),
        positions: T.Tensor((num_tokens,), T.int64),
        eps: T.float32,
    ):
        with T.Kernel(num_tokens, num_heads, threads=threads) as (
            token_id,
            head_id,
        ):
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            partial_sumsq = T.alloc_local((1,), T.float32)
            warp_sumsq = T.alloc_shared((warps_per_cta,), T.float32)

            partial_sumsq[0] = 0.0
            for col_base in T.serial(0, HIDDEN_SIZE, threads):
                col = col_base + tx
                if col < HIDDEN_SIZE:
                    value = T.cast(q[token_id, head_id, col], T.float32)
                    partial_sumsq[0] += value * value

            partial_sumsq[0] = _warp_reduce_sum(partial_sumsq[0])
            if lane == 0:
                warp_sumsq[warp] = partial_sumsq[0]
            T.sync_threads()

            partial_sumsq[0] = T.if_then_else(tx < warps_per_cta, warp_sumsq[tx], 0.0)
            if warp == 0:
                partial_sumsq[0] = _warp_reduce_sum(partial_sumsq[0])
                if lane == 0:
                    warp_sumsq[0] = T.rsqrt(partial_sumsq[0] / float(HIDDEN_SIZE) + eps)
            T.sync_threads()

            for col_base in T.serial(0, NOPE_DIM, threads):
                col = col_base + tx
                if col < NOPE_DIM:
                    value = T.cast(q[token_id, head_id, col], T.float32)
                    out[token_id, head_id, col] = T.cast(
                        value * warp_sumsq[0], T.bfloat16
                    )

            if tx < HALF_ROPE_DIM:
                pos = positions[token_id]
                even_col = NOPE_DIM + tx * 2
                odd_col = even_col + 1
                even = T.cast(q[token_id, head_id, even_col], T.float32) * warp_sumsq[0]
                odd = T.cast(q[token_id, head_id, odd_col], T.float32) * warp_sumsq[0]
                c = cos_sin_cache[pos, tx]
                s = cos_sin_cache[pos, HALF_ROPE_DIM + tx]
                out[token_id, head_id, even_col] = T.cast(
                    even * c - odd * s, T.bfloat16
                )
                out[token_id, head_id, odd_col] = T.cast(even * s + odd * c, T.bfloat16)

    return _qnorm_rope_kernel


@tilelang.jit(target="musa", pass_configs=_tilelang_musa_pass_configs(tilelang))
def kv_rope_pack_kernel():
    num_tokens = T.dynamic("num_tokens")
    num_pages = T.dynamic("num_pages")
    page_bytes = T.dynamic("page_bytes")
    page_u32 = T.dynamic("page_u32")
    num_positions = T.dynamic("num_positions")
    threads = 256
    tile_dim = 64
    rope_pack_elems = 2

    def pow2_scale_byte_and_inv(value):
        clamped = T.max(value, 1.0e-4)
        bits = T.reinterpret("uint32", clamped)
        exp = (bits >> 23) & 0xFF
        man_bits = bits & ((1 << 23) - 1)
        exp_scale = T.Cast("int32", exp - 127 + T.if_then_else(man_bits != 0, 1, 0))
        scale_byte = T.Cast("uint8", exp_scale + 127)
        inv_scale = T.reinterpret("float32", (127 - exp_scale) << 23)
        return scale_byte, inv_scale

    def abs_f32(value):
        return T.if_then_else(value < 0.0, -value, value)

    @T.prim_func
    def _kv_rope_pack_kernel(
        kv: T.Tensor((num_tokens, HIDDEN_SIZE), T.bfloat16),
        cache_u8: T.Tensor((num_pages, page_bytes), T.uint8),
        cache_u32: T.Tensor((num_pages, page_u32), T.uint32),
        slot_mapping: T.Tensor((num_tokens,), T.int64),
        positions: T.Tensor((num_tokens,), T.int64),
        cos_sin_cache: T.Tensor((num_positions, ROPE_DIM), T.float32),
        block_size: T.int32,
    ):
        with T.Kernel(num_tokens, threads=threads) as token_id:
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            loc = slot_mapping[token_id]

            if loc >= 0:
                loc_i32 = T.Cast("int32", loc)
                page_idx = loc_i32 // block_size
                token_offset = loc_i32 % block_size

                if warp < SCALE_DIM:
                    vals = T.alloc_local((2,), dtype=T.bfloat16)
                    fvals = T.alloc_local((2,), dtype=T.float32)
                    local_amax = T.alloc_local((1,), dtype=T.float32)
                    tile_amax = T.alloc_local((1,), dtype=T.float32)
                    elem_base = warp * tile_dim + lane * 2

                    local_amax[0] = 0.0
                    for vec in T.vectorized(2):
                        vals[vec] = kv[token_id, elem_base + vec]
                        fvals[vec] = T.cast(vals[vec], T.float32)
                        local_amax[0] = T.max(local_amax[0], abs_f32(fvals[vec]))

                    tile_amax[0] = _warp_reduce_max(local_amax[0])
                    scale_byte, inv_scale = pow2_scale_byte_and_inv(
                        tile_amax[0] / FP8_MAX
                    )
                    if lane == 0:
                        cache_u8[
                            page_idx,
                            block_size * TOKEN_VALUE_BYTES
                            + token_offset * TOKEN_SCALE_BYTES
                            + warp,
                        ] = scale_byte
                    tile_offset = token_offset * TOKEN_VALUE_BYTES + elem_base
                    for vec in T.vectorized(2):
                        cache_u8[
                            page_idx,
                            tile_offset + vec,
                        ] = T.reinterpret(
                            "uint8",
                            T.Cast(
                                "float8_e4m3fn",
                                T.clamp(fvals[vec] * inv_scale, -FP8_MAX, FP8_MAX),
                            ),
                        )
                else:
                    pos = positions[token_id]
                    elem = lane * rope_pack_elems
                    even_col = NOPE_DIM + elem
                    odd_col = even_col + 1
                    even = T.cast(kv[token_id, even_col], T.float32)
                    odd = T.cast(kv[token_id, odd_col], T.float32)
                    pair_idx = elem // 2
                    c = cos_sin_cache[pos, pair_idx]
                    s = cos_sin_cache[pos, HALF_ROPE_DIM + pair_idx]
                    rope_even = T.cast(even * c - odd * s, T.bfloat16)
                    rope_odd = T.cast(even * s + odd * c, T.bfloat16)
                    lo = T.reinterpret("uint16", rope_even)
                    hi = T.reinterpret("uint16", rope_odd)
                    rope_offset_u32 = (token_offset * TOKEN_VALUE_BYTES + NOPE_DIM) // (
                        2 * rope_pack_elems
                    ) + lane
                    cache_u32[page_idx, rope_offset_u32] = T.Cast("uint32", lo) | (
                        T.Cast("uint32", hi) << 16
                    )
                    if lane == 0:
                        cache_u8[
                            page_idx,
                            block_size * TOKEN_VALUE_BYTES
                            + token_offset * TOKEN_SCALE_BYTES
                            + SCALE_DIM,
                        ] = T.Cast("uint8", 0)

    return _kv_rope_pack_kernel


@lru_cache(maxsize=None)
def mhc_post_kernel(hidden_size: int, hidden_block: int = 256, threads: int = 256):
    hidden_block = math.gcd(hidden_block, hidden_size)
    if hidden_block <= 0 or hidden_size % hidden_block != 0:
        raise ValueError(
            f"invalid MHC post hidden_block={hidden_block} for hidden={hidden_size}"
        )

    @tilelang.jit(target="musa", pass_configs=_tilelang_musa_pass_configs(tilelang))
    def _mhc_post_kernel():
        num_tokens = T.dynamic("num_tokens")

        @T.prim_func
        def _kernel(
            x: T.Tensor((num_tokens, hidden_size), T.bfloat16),
            residual: T.Tensor((num_tokens, 4, hidden_size), T.bfloat16),
            post_mix: T.Tensor((num_tokens, 4), T.float32),
            comb_mix: T.Tensor((num_tokens, 4, 4), T.float32),
            out: T.Tensor((num_tokens, 4, hidden_size), T.bfloat16),
        ):
            with T.Kernel(num_tokens, hidden_size // hidden_block, threads=threads) as (
                token_id,
                block_id,
            ):
                tx = T.get_thread_binding()
                hidden_idx = block_id * hidden_block + tx
                if tx < hidden_block:
                    x_value = T.cast(x[token_id, hidden_idx], T.float32)
                    r0 = T.cast(residual[token_id, 0, hidden_idx], T.float32)
                    r1 = T.cast(residual[token_id, 1, hidden_idx], T.float32)
                    r2 = T.cast(residual[token_id, 2, hidden_idx], T.float32)
                    r3 = T.cast(residual[token_id, 3, hidden_idx], T.float32)

                    acc0 = (
                        post_mix[token_id, 0] * x_value
                        + comb_mix[token_id, 0, 0] * r0
                        + comb_mix[token_id, 1, 0] * r1
                        + comb_mix[token_id, 2, 0] * r2
                        + comb_mix[token_id, 3, 0] * r3
                    )
                    acc1 = (
                        post_mix[token_id, 1] * x_value
                        + comb_mix[token_id, 0, 1] * r0
                        + comb_mix[token_id, 1, 1] * r1
                        + comb_mix[token_id, 2, 1] * r2
                        + comb_mix[token_id, 3, 1] * r3
                    )
                    acc2 = (
                        post_mix[token_id, 2] * x_value
                        + comb_mix[token_id, 0, 2] * r0
                        + comb_mix[token_id, 1, 2] * r1
                        + comb_mix[token_id, 2, 2] * r2
                        + comb_mix[token_id, 3, 2] * r3
                    )
                    acc3 = (
                        post_mix[token_id, 3] * x_value
                        + comb_mix[token_id, 0, 3] * r0
                        + comb_mix[token_id, 1, 3] * r1
                        + comb_mix[token_id, 2, 3] * r2
                        + comb_mix[token_id, 3, 3] * r3
                    )

                    out[token_id, 0, hidden_idx] = T.cast(acc0, T.bfloat16)
                    out[token_id, 1, hidden_idx] = T.cast(acc1, T.bfloat16)
                    out[token_id, 2, hidden_idx] = T.cast(acc2, T.bfloat16)
                    out[token_id, 3, hidden_idx] = T.cast(acc3, T.bfloat16)

        return _kernel

    return _mhc_post_kernel()
