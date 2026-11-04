# SPDX-License-Identifier: Apache-2.0
"""TileLang sparse paged GQA for the QSA prefill and decode paths.

Both stages gather K/V tile by tile with per-token head-dim-contiguous vector
loads (the sparse page addresses are constant along the dim axis; the block
stride comes off the tensor, so a packed KV slab works) and compute
QK^T / PV with T.gemm; the grouped-query tile is padded to 8 rows to satisfy the
MUSA MMA M%8 requirement. The online softmax stays in fragments.

A prefill row whose position is below the selection width selects every causal
token, so it is plain causal attention: when the caller passes the batch layout,
those rows go to the paged FlashAttention kernel and the sparse kernel skips
them.

Decode splits the tile loop over ``num_splits`` programs that each emit a
normalized partial output plus its log2-domain LSE, merged by a second kernel:
one program per row leaves the device idle at small batch sizes.

``block_n`` is tied to the thread count (``threads == block_n * _DUP``) so that
no loop in the tile body is thread-predicated. TileLang's storage-sync pass
refuses to keep a barrier inside a predicated region and hoists it out
("[ThreadSync] Hoisting sync from inside if to before if"), which drops the
write-then-read ordering on the shared gather metadata and makes the kernel
nondeterministic. Within those loops only lane ``_u == 0`` stores the gather
metadata, so each shared element has a single writer; the guard holds no
barrier.

Ported to vLLM-MUSA v0.32 with the kernel bodies unchanged; only the dispatch
in ``qwen4_exp/nvidia/qsa.py`` differs. The 32-bit index switch below stays off
until the v0.32 KV layout work lands, so the default is 64-bit addressing.
"""

import functools

import os

import tilelang
import tilelang.language as T
import torch

try:
    from flash_attn_interface import flash_attn_varlen_func
except (ImportError, ModuleNotFoundError):
    flash_attn_varlen_func = None

from vllm_musa.jit_kernel.tilelang.utils import (
    MUSA_COMMON_PASS_CONFIGS,
    MUSA_COMPILE_FLAGS,
    tilelang_dtype,
)

_PASS_CONFIGS = dict(MUSA_COMMON_PASS_CONFIGS)
# The gather stages its page metadata in shared memory before reading through
# it, so the cross-loop thread storage syncs that the common config disables for
# pure elementwise kernels must stay on.
if hasattr(tilelang.PassConfigKey, "TL_DISABLE_THREAD_STORAGE_SYNC"):
    _PASS_CONFIGS[tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC] = False
# A page of the packed KV slab is megabytes apart from the next, so a physical
# block index times that stride leaves the 2 GiB a 32-bit byte offset can reach
# and wraps to an address below the cache. Global buffer indices must then be
# 64-bit, which costs about 13% of time-to-first-token.
#
# One page per block keeps every offset well inside 32 bits, so the narrow
# indices are safe there. The width is tied to that layout rather than chosen
# independently: 32-bit indices on the packed slab would fault again.
_INDEX_BITWIDTH = 32 if os.environ.get("VLLM_MUSA_UNPACKED_KV") == "1" else 64
for _key, _value in (
    ("TL_DISABLE_SAFE_COPY_PREDICATION", True),
    ("TL_DISABLE_SAFE_ROBUST_COPY_PREDICATION", True),
    ("TL_CONFIG_INDEX_BITWIDTH", _INDEX_BITWIDTH),
):
    if hasattr(tilelang.PassConfigKey, _key):
        _PASS_CONFIGS[getattr(tilelang.PassConfigKey, _key)] = _value

__all__ = [
    "qsa_dense_layout",
    "qsa_prefill_attention_v2",
    "qsa_decode_attention",
    "prewarm_qsa_kernels",
]

# Scores are masked to a large finite negative value instead of -inf so that an
# all-invalid tile keeps the running max arithmetic well defined.
_NEG_INF = -1.0e30

_BLOCK_M = 8
_BLOCK_N = 128
_DUP = 2
_THREADS = _BLOCK_N * _DUP
_DECODE_SPLITS = 16

# Prefill runs a taller, narrower tile. The grouped-query tile height drives the
# gemm lowering: at 8 rows the vector-ALU path wastes lanes, and an 8x64 tile is
# nondeterministic, while 16x64 is deterministic and about twice as fast.
# Decode keeps the 8x128 tile because its split-k scratch is sized by the tile
# height and is preallocated for every capture size.
_PREFILL_BLOCK_M = 16
_PREFILL_BLOCK_N = 64
_PREFILL_DUP = 2
_PREFILL_THREADS = _PREFILL_BLOCK_N * _PREFILL_DUP
# bf16 elements per prefill gather load (16 B).
_GATHER_VEC = 8


@functools.lru_cache(maxsize=16)
@tilelang.jit(
    target="musa",
    pass_configs=_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _qsa_prefill_kernel(
    group_size: int,
    head_dim: int,
    topk: int,
    block_n: int,
    page_size: int,
    page_table_width: int,
    dtype: str,
    threads: int,
    dup: int,
):
    num_tiles = (topk + block_n - 1) // block_n
    block_m = _PREFILL_BLOCK_M
    # log2-scaled softmax so the loop uses exp2 and never requantizes Q.
    softmax_scale_log2 = (head_dim**-0.5) * 1.4426950408889634
    num_rows = T.dynamic("num_rows")
    num_blocks = T.dynamic("num_blocks")
    num_requests = T.dynamic("num_requests")
    # The paged cache may be a window into a packed KV slab, so the block
    # stride is read off the tensor instead of assumed to be one page.
    k_block_stride = T.dynamic("k_block_stride")
    v_block_stride = T.dynamic("v_block_stride")

    @T.prim_func
    def qsa_prefill(
        q: T.Tensor((num_rows, group_size, head_dim), dtype),
        k_cache: T.StridedTensor(
            (num_blocks, page_size, head_dim),
            (k_block_stride, head_dim, 1),
            dtype,
        ),
        v_cache: T.StridedTensor(
            (num_blocks, page_size, head_dim),
            (v_block_stride, head_dim, 1),
            dtype,
        ),
        indices: T.Tensor((num_rows, topk + 1), "int32"),
        block_table: T.Tensor((num_requests, page_table_width), "int32"),
        token_to_req: T.Tensor((num_rows,), "int32"),
        positions: T.Tensor((num_rows,), "int64"),
        dense_below: T.int32,
        out: T.Tensor((num_rows, group_size, head_dim), dtype),
    ):
        with T.Kernel(num_rows, threads=threads) as row:
            req = token_to_req[row]
            req_safe = T.min(T.max(req, 0), num_requests - 1)
            valid_count = indices[row, topk]
            # Rows below dense_below are written by the dense kernel.
            live = positions[row] >= T.cast(dense_below, "int64")

            s_q = T.alloc_shared((block_m, head_dim), dtype)
            s_k = T.alloc_shared((block_n, head_dim), dtype)
            s_v = T.alloc_shared((block_n, head_dim), dtype)
            s_pb = T.alloc_shared((block_m, block_n), dtype)
            s_idx = T.alloc_shared((block_n,), "int32")
            s_off = T.alloc_shared((block_n,), "int32")
            s_phys = T.alloc_shared((block_n,), "int32")
            s_valid = T.alloc_shared((block_n,), "float32")

            acc_frag = T.alloc_fragment((block_m, head_dim), "float32")
            scores_frag = T.alloc_fragment((block_m, block_n), "float32")
            mrun = T.alloc_fragment((block_m,), "float32")
            mprev = T.alloc_fragment((block_m,), "float32")
            alpha = T.alloc_fragment((block_m,), "float32")
            lrun = T.alloc_fragment((block_m,), "float32")
            lsum_i = T.alloc_fragment((block_m,), "float32")

            for m, d in T.Parallel(block_m, head_dim):
                if live and m < group_size:
                    s_q[m, d] = q[row, m, d]
                else:
                    s_q[m, d] = 0.0
            T.fill(acc_frag, 0.0)
            T.fill(lrun, 0.0)
            T.fill(mrun, -(2.0**30))

            for tile in T.serial(num_tiles):
                # Tiles that start beyond the row's valid count are entirely
                # masked: skip their gather and softmax work.
                if live and tile * block_n < valid_count:
                    # Keep the index-load address affine on the fast path; only
                    # the last tile of a non-multiple width needs the clamp.
                    # Clamped lanes read the trailing count column and stay
                    # invalid through the column < valid_count check below.
                    if tile < num_tiles - 1:
                        for i, _u in T.Parallel(block_n, dup):
                            if _u == 0:
                                s_idx[i] = indices[row, tile * block_n + i]
                    else:
                        for i, _u in T.Parallel(block_n, dup):
                            if _u == 0:
                                s_idx[i] = indices[row, T.min(tile * block_n + i, topk)]

                    T.sync_threads()
                    for i, _u in T.Parallel(block_n, dup):
                        tok = T.max(s_idx[i], 0)
                        raw_page = block_table[
                            req_safe,
                            T.min(tok // page_size, page_table_width - 1),
                        ]
                        ok = (
                            (req >= 0)
                            and (req < num_requests)
                            and (s_idx[i] >= 0)
                            and (tile * block_n + i < valid_count)
                            and (tok // page_size < page_table_width)
                            and (raw_page >= 0)
                            and (raw_page < num_blocks)
                        )
                        if _u == 0:
                            s_off[i] = tok % page_size
                            s_phys[i] = T.min(T.max(raw_page, 0), num_blocks - 1)
                            s_valid[i] = T.if_then_else(ok, 1.0, 0.0)

                    T.sync_threads()
                    # An explicit vector axis gives one warp per KV row and 16 B
                    # loads; the flat (row, dim) loop lowers to scalar loads with
                    # a barrier on every iteration.
                    for i, c in T.Parallel(block_n, head_dim // _GATHER_VEC):
                        for e in T.vectorized(_GATHER_VEC):
                            s_k[i, c * _GATHER_VEC + e] = k_cache[
                                s_phys[i], s_off[i], c * _GATHER_VEC + e]
                            s_v[i, c * _GATHER_VEC + e] = v_cache[
                                s_phys[i], s_off[i], c * _GATHER_VEC + e]
                    T.sync_threads()

                    for m, n in T.Parallel(block_m, block_n):
                        scores_frag[m, n] = T.if_then_else(
                            s_valid[n] > 0.5, 0.0, _NEG_INF
                        )
                    T.gemm(s_q, s_k, scores_frag, transpose_B=True)
                    T.copy(mrun, mprev)
                    T.reduce_max(scores_frag, mrun, dim=1, clear=False)
                    for m in T.Parallel(block_m):
                        alpha[m] = T.exp2((mprev[m] - mrun[m]) * softmax_scale_log2)
                    for m, n in T.Parallel(block_m, block_n):
                        scores_frag[m, n] = T.exp2(
                            scores_frag[m, n] * softmax_scale_log2
                            - mrun[m] * softmax_scale_log2
                        )
                    T.reduce_sum(scores_frag, lsum_i, dim=1)
                    for m in T.Parallel(block_m):
                        lrun[m] = lrun[m] * alpha[m] + lsum_i[m]
                    for m, d in T.Parallel(block_m, head_dim):
                        acc_frag[m, d] = acc_frag[m, d] * alpha[m]
                    T.copy(scores_frag, s_pb)
                    T.sync_threads()
                    T.gemm(s_pb, s_v, acc_frag)
                    T.sync_threads()

            for m, d in T.Parallel(block_m, head_dim):
                if live and m < group_size:
                    out[row, m, d] = T.cast(
                        T.if_then_else(
                            lrun[m] > 0.0,
                            acc_frag[m, d] / T.max(lrun[m], 1.0e-20),
                            0.0,
                        ),
                        dtype,
                    )

    return qsa_prefill


@functools.lru_cache(maxsize=16)
@tilelang.jit(
    target="musa",
    pass_configs=_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _qsa_decode_kernel(
    group_size: int,
    head_dim: int,
    topk: int,
    block_n: int,
    page_size: int,
    page_table_width: int,
    dtype: str,
    threads: int,
    dup: int,
    num_splits: int,
):
    num_tiles = (topk + block_n - 1) // block_n
    block_m = _BLOCK_M
    softmax_scale_log2 = (head_dim**-0.5) * 1.4426950408889634
    num_rows = T.dynamic("num_rows")
    num_blocks = T.dynamic("num_blocks")
    num_requests = T.dynamic("num_requests")
    # The paged cache may be a window into a packed KV slab, so the block
    # stride is read off the tensor instead of assumed to be one page.
    k_block_stride = T.dynamic("k_block_stride")
    v_block_stride = T.dynamic("v_block_stride")

    @T.prim_func
    def qsa_decode(
        q: T.Tensor((num_rows, group_size, head_dim), dtype),
        k_cache: T.StridedTensor(
            (num_blocks, page_size, head_dim),
            (k_block_stride, head_dim, 1),
            dtype,
        ),
        v_cache: T.StridedTensor(
            (num_blocks, page_size, head_dim),
            (v_block_stride, head_dim, 1),
            dtype,
        ),
        indices: T.Tensor((num_rows, topk + 1), "int32"),
        block_table: T.Tensor((num_requests, page_table_width), "int32"),
        token_to_req: T.Tensor((num_rows,), "int32"),
        partial_o: T.Tensor((num_splits, num_rows, block_m, head_dim), "float32"),
        partial_l: T.Tensor((num_splits, num_rows, block_m), "float32"),
    ):
        with T.Kernel(num_rows, num_splits, threads=threads) as (row, split):
            req = token_to_req[row]
            req_safe = T.min(T.max(req, 0), num_requests - 1)
            valid_count = indices[row, topk]

            s_q = T.alloc_shared((block_m, head_dim), dtype)
            s_k = T.alloc_shared((block_n, head_dim), dtype)
            s_v = T.alloc_shared((block_n, head_dim), dtype)
            s_pb = T.alloc_shared((block_m, block_n), dtype)
            s_idx = T.alloc_shared((block_n,), "int32")
            s_off = T.alloc_shared((block_n,), "int32")
            s_phys = T.alloc_shared((block_n,), "int32")
            s_valid = T.alloc_shared((block_n,), "float32")

            acc_frag = T.alloc_fragment((block_m, head_dim), "float32")
            scores_frag = T.alloc_fragment((block_m, block_n), "float32")
            mrun = T.alloc_fragment((block_m,), "float32")
            mprev = T.alloc_fragment((block_m,), "float32")
            alpha = T.alloc_fragment((block_m,), "float32")
            lrun = T.alloc_fragment((block_m,), "float32")
            lsum_i = T.alloc_fragment((block_m,), "float32")

            for m, d in T.Parallel(block_m, head_dim):
                if m < group_size:
                    s_q[m, d] = q[row, m, d]
                else:
                    s_q[m, d] = 0.0
            T.fill(acc_frag, 0.0)
            T.fill(lrun, 0.0)
            T.fill(mrun, -(2.0**30))

            for t_i in T.serial((num_tiles + num_splits - 1) // num_splits):
                tile = split + t_i * num_splits
                if (tile < num_tiles) and (tile * block_n < valid_count):
                    if tile < num_tiles - 1:
                        for i, _u in T.Parallel(block_n, dup):
                            if _u == 0:
                                s_idx[i] = indices[row, tile * block_n + i]
                    else:
                        for i, _u in T.Parallel(block_n, dup):
                            if _u == 0:
                                s_idx[i] = indices[row, T.min(tile * block_n + i, topk)]

                    T.sync_threads()
                    for i, _u in T.Parallel(block_n, dup):
                        tok = T.max(s_idx[i], 0)
                        raw_page = block_table[
                            req_safe,
                            T.min(tok // page_size, page_table_width - 1),
                        ]
                        ok = (
                            (req >= 0)
                            and (req < num_requests)
                            and (s_idx[i] >= 0)
                            and (tile * block_n + i < valid_count)
                            and (tok // page_size < page_table_width)
                            and (raw_page >= 0)
                            and (raw_page < num_blocks)
                        )
                        if _u == 0:
                            s_off[i] = tok % page_size
                            s_phys[i] = T.min(T.max(raw_page, 0), num_blocks - 1)
                            s_valid[i] = T.if_then_else(ok, 1.0, 0.0)

                    T.sync_threads()
                    for i, d in T.Parallel(block_n, head_dim):
                        s_k[i, d] = k_cache[s_phys[i], s_off[i], d]
                        s_v[i, d] = v_cache[s_phys[i], s_off[i], d]
                    T.sync_threads()

                    for m, n in T.Parallel(block_m, block_n):
                        scores_frag[m, n] = T.if_then_else(
                            s_valid[n] > 0.5, 0.0, _NEG_INF
                        )
                    T.gemm(s_q, s_k, scores_frag, transpose_B=True)
                    T.copy(mrun, mprev)
                    T.reduce_max(scores_frag, mrun, dim=1, clear=False)
                    for m in T.Parallel(block_m):
                        alpha[m] = T.exp2((mprev[m] - mrun[m]) * softmax_scale_log2)
                    for m, n in T.Parallel(block_m, block_n):
                        scores_frag[m, n] = T.exp2(
                            scores_frag[m, n] * softmax_scale_log2
                            - mrun[m] * softmax_scale_log2
                        )
                    T.reduce_sum(scores_frag, lsum_i, dim=1)
                    for m in T.Parallel(block_m):
                        lrun[m] = lrun[m] * alpha[m] + lsum_i[m]
                    for m, d in T.Parallel(block_m, head_dim):
                        acc_frag[m, d] = acc_frag[m, d] * alpha[m]
                    T.copy(scores_frag, s_pb)
                    T.sync_threads()
                    T.gemm(s_pb, s_v, acc_frag)
                    T.sync_threads()

            for m, d in T.Parallel(block_m, head_dim):
                partial_o[split, row, m, d] = T.if_then_else(
                    lrun[m] > 0.0, acc_frag[m, d] / T.max(lrun[m], 1.0e-20), 0.0
                )
            for m in T.Parallel(block_m):
                partial_l[split, row, m] = T.if_then_else(
                    lrun[m] > 0.0,
                    mrun[m] * softmax_scale_log2 + T.log2(lrun[m]),
                    _NEG_INF,
                )

    return qsa_decode


@functools.lru_cache(maxsize=16)
@tilelang.jit(
    target="musa",
    pass_configs=_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _qsa_merge_kernel(
    group_size: int,
    head_dim: int,
    num_splits: int,
    dtype: str,
    threads: int,
):
    block_m = _BLOCK_M
    num_rows = T.dynamic("num_rows")

    @T.prim_func
    def qsa_merge(
        partial_o: T.Tensor((num_splits, num_rows, block_m, head_dim), "float32"),
        partial_l: T.Tensor((num_splits, num_rows, block_m), "float32"),
        out: T.Tensor((num_rows, group_size, head_dim), dtype),
    ):
        with T.Kernel(num_rows, threads=threads) as row:
            acc = T.alloc_fragment((block_m, head_dim), "float32")
            lmax = T.alloc_fragment((block_m,), "float32")
            lsum = T.alloc_fragment((block_m,), "float32")
            a_old = T.alloc_fragment((block_m,), "float32")
            a_new = T.alloc_fragment((block_m,), "float32")
            T.fill(acc, 0.0)
            T.fill(lsum, 0.0)
            T.fill(lmax, _NEG_INF)
            for s_i in T.serial(num_splits):
                for m in T.Parallel(block_m):
                    lnew = T.max(lmax[m], partial_l[s_i, row, m])
                    a_old[m] = T.exp2(lmax[m] - lnew)
                    a_new[m] = T.exp2(partial_l[s_i, row, m] - lnew)
                    lsum[m] = lsum[m] * a_old[m] + a_new[m]
                    lmax[m] = lnew
                for m, d in T.Parallel(block_m, head_dim):
                    acc[m, d] = (
                        acc[m, d] * a_old[m] + partial_o[s_i, row, m, d] * a_new[m]
                    )
            for m, d in T.Parallel(block_m, head_dim):
                if m < group_size:
                    out[row, m, d] = T.cast(
                        T.if_then_else(lsum[m] > 0.0, acc[m, d] / lsum[m], 0.0),
                        dtype,
                    )

    return qsa_merge


_qsa_prefill_kernel.mode = "lazy"
_qsa_decode_kernel.mode = "lazy"
_qsa_merge_kernel.mode = "lazy"

# Split-k scratch, keyed so a captured decode graph always sees the same buffers.
_SCRATCH: dict = {}


def _shape_args(q, k_cache, logical_indices, block_table, prefill=False):
    return (
        q.shape[1] // k_cache.shape[2],
        q.shape[2],
        logical_indices.shape[1] - 1,
        _PREFILL_BLOCK_N if prefill else _BLOCK_N,
        k_cache.shape[1],
        block_table.shape[1],
        tilelang_dtype(q.dtype),
        _PREFILL_THREADS if prefill else _THREADS,
        _PREFILL_DUP if prefill else _DUP,
    )


def _compile_all(prefill_args, decode_args, num_splits=_DECODE_SPLITS):
    """Compile every kernel for this signature.

    TileLang compiles on the first call for a given signature and a first call
    inside a captured graph deadlocks the worker, so the prefill entry point
    (which always runs outside capture) also compiles the decode pair.
    The jit wrappers are memoized, so repeat calls are free.
    """
    group_size, head_dim, dtype = decode_args[0], decode_args[1], decode_args[6]
    _qsa_prefill_kernel(*prefill_args)
    _qsa_decode_kernel(*decode_args, num_splits)
    _qsa_merge_kernel(group_size, head_dim, num_splits, dtype, _THREADS)


def qsa_dense_layout(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req: torch.Tensor,
    num_rows: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Row layout of the dense-prefix split for one forward step.

    Returns per-request dense query and key counts for FlashAttention and each
    row's token index within its request. Every QSA layer of a step shares it.
    Each request's rows are consecutive positions ending at ``seq_len - 1``, so
    its dense rows are a prefix of its query rows, and bottom-right causal
    alignment of ``used_q`` rows against ``used_k`` keys gives each row exactly
    the keys up to its own position.
    """
    num_reqs = min(query_start_loc.shape[0] - 1, seq_lens.shape[0])
    query_start_loc = query_start_loc[: num_reqs + 1]
    seq_lens = seq_lens[:num_reqs]
    used_k = torch.clamp(seq_lens, max=width).to(torch.int32)
    used_q = torch.clamp(
        used_k - seq_lens + (query_start_loc[1:] - query_start_loc[:-1]), min=0
    ).to(torch.int32)
    # Token index within the request; rope positions may differ from it.
    req = token_to_req[:num_rows].long().clamp(0, num_reqs - 1)
    positions = (seq_lens.long()[req] - query_start_loc.long()[req + 1]
                 + torch.arange(num_rows, device=seq_lens.device))
    return used_q, used_k, positions


def qsa_prefill_attention_v2(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor,
    query_start_loc: torch.Tensor | None = None,
    dense_layout: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    max_query_len: int = 0,
    max_seq_len: int = 0,
) -> torch.Tensor:
    """Sparse prefill attention.

    With ``query_start_loc``, ``dense_layout`` (from ``qsa_dense_layout``) and
    the step's ``max_query_len`` / ``max_seq_len``, the rows that select their
    whole causal prefix run through FlashAttention and only the sparse rows run
    through the TileLang kernel.
    """
    num_rows = q.shape[0]
    if num_rows == 0:
        return out
    shape_args = _shape_args(q, k_cache, logical_indices, block_table,
                             prefill=True)
    _compile_all(shape_args,
                 _shape_args(q, k_cache, logical_indices, block_table))
    width = logical_indices.shape[1] - 1
    dense_below = 0
    if (
        dense_layout is not None
        and query_start_loc is not None
        and flash_attn_varlen_func is not None
    ):
        used_q, used_k, positions = dense_layout
        flash_attn_varlen_func(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q=query_start_loc[: used_q.shape[0] + 1],
            max_seqlen_q=max_query_len,
            max_seqlen_k=min(max_seq_len, width) if max_seq_len > 0 else width,
            seqused_q=used_q,
            seqused_k=used_k,
            block_table=block_table,
            softmax_scale=q.shape[-1] ** -0.5,
            causal=True,
            out=out,
        )
        if 0 < max_seq_len <= width:
            return out
        dense_below = width
    else:
        positions = torch.zeros(num_rows, dtype=torch.int64, device=q.device)
    kernel = _qsa_prefill_kernel(*shape_args)
    # The upstream forward canonicalizes the singleton KV-head stride to 0;
    # TileLang takes dense tensors, so drop the singleton dim (a view).
    kernel(q, k_cache.squeeze(2), v_cache.squeeze(2), logical_indices,
           block_table, token_to_req, positions[:num_rows], dense_below, out)
    return out


def _split_scratch(num_rows, head_dim, device, num_splits):
    key = (num_splits, num_rows, head_dim, str(device))
    buf = _SCRATCH.get(key)
    if buf is None:
        buf = (
            torch.empty(num_splits, num_rows, _BLOCK_M, head_dim,
                        device=device, dtype=torch.float32),
            torch.empty(num_splits, num_rows, _BLOCK_M,
                        device=device, dtype=torch.float32),
        )
        _SCRATCH[key] = buf
    return buf


def qsa_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor,
    num_splits: int = _DECODE_SPLITS,
) -> torch.Tensor:
    num_rows = q.shape[0]
    if num_rows == 0:
        return out
    group_size = q.shape[1] // k_cache.shape[2]
    head_dim = q.shape[2]
    shape_args = _shape_args(q, k_cache, logical_indices, block_table)
    _compile_all(_shape_args(q, k_cache, logical_indices, block_table,
                             prefill=True), shape_args, num_splits)
    partial_o, partial_l = _split_scratch(num_rows, head_dim, q.device, num_splits)
    split = _qsa_decode_kernel(*shape_args, num_splits)
    split(q, k_cache.squeeze(2), v_cache.squeeze(2), logical_indices,
          block_table, token_to_req, partial_o, partial_l)
    merge = _qsa_merge_kernel(
        group_size, head_dim, num_splits, tilelang_dtype(q.dtype), _THREADS
    )
    merge(partial_o, partial_l, out)
    return out


def prewarm_qsa_kernels(
    group_size: int,
    head_dim: int,
    topk: int,
    page_size: int,
    page_table_width: int,
    dtype: torch.dtype,
    decode_rows: tuple = (),
    num_splits: int = _DECODE_SPLITS,
) -> None:
    """Compile the kernels and allocate the split-k scratch ahead of capture.

    Both steps must happen outside CUDA-graph capture. The decode path is first
    reached while the decode graphs are being captured, so without this the JIT
    build and the split-k scratch land in the graph memory pool: with more than
    one captured size the scratch aliases another graph's tensors and the merge
    kernel reads non-finite partials.
    """
    _compile_all(
        (
            group_size, head_dim, topk, _PREFILL_BLOCK_N, page_size,
            page_table_width, tilelang_dtype(dtype), _PREFILL_THREADS,
            _PREFILL_DUP,
        ),
        (
            group_size, head_dim, topk, _BLOCK_N, page_size, page_table_width,
            tilelang_dtype(dtype), _THREADS, _DUP,
        ),
        num_splits,
    )
    if not hasattr(torch, "musa"):
        return
    device = torch.device("musa", torch.musa.current_device())
    for rows in decode_rows:
        _split_scratch(rows, head_dim, device, num_splits)
