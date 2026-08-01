"""MUSA TileLang causal conv1d forward kernel."""

import functools

import tilelang
import tilelang.language as T
import torch
from vllm.logger import init_logger

from vllm_musa.jit_kernel.tilelang.utils import (
    MUSA_COMMON_PASS_CONFIGS,
    MUSA_COMPILE_FLAGS,
    storage_window,
    tilelang_dtype,
)
from vllm_musa.utils.environ import envs

PAD_SLOT_ID = -1  # MUSA: match vllm mamba causal_conv1d PAD_SLOT_ID
NULL_BLOCK_ID = 0  # MUSA: match vllm mamba causal_conv1d NULL_BLOCK_ID


def register_custom_op(fn=None, **_kw):
    def _wrap(f):
        return f

    return _wrap if fn is None else _wrap(fn)


_LOG2E = 1.4426950408889634
_ENABLE_WIDTH4_PREFILL_SPLIT = envs.VLLM_MUSA_QWEN_GDN_WIDTH4_PREFILL_SPLIT.get()
_WIDTH4_PREFILL_SPLIT_DIMS = frozenset((10240,))
_WIDTH4_PREFILL_SPLIT_LOGGED = False
logger = init_logger(__name__)


def _should_use_width4_prefill_split(
    *,
    width: int,
    dim: int,
    dtype: torch.dtype,
    max_seq_len: int,
    batch_size: int,
    has_conv_states: bool,
    has_cache_indices: bool,
    cache_indices_stride: int,
    x_inner_stride: int,
    out_inner_stride: int,
    weight_inner_stride: int,
) -> bool:
    return (
        _ENABLE_WIDTH4_PREFILL_SPLIT
        and width == 4
        and dtype is torch.bfloat16
        # The split is beneficial only once the local channel width is large
        # enough to amortize its two launches on MP31.
        and dim in _WIDTH4_PREFILL_SPLIT_DIMS
        and max_seq_len >= 4096
        and batch_size > 1
        and has_conv_states
        and has_cache_indices
        and cache_indices_stride == 1
        and x_inner_stride == 1
        and out_inner_stride == 1
        and weight_inner_stride == 1
    )


_CAUSAL_CONV1D_PASS_CONFIGS = dict(MUSA_COMMON_PASS_CONFIGS)
for _key, _value in (
    ("TL_ENABLE_LOWER_LDGSTG", True),
    ("TL_ENABLE_LOWER_LDGSTG_PREDICATED", True),
    ("TL_DISABLE_SAFE_COPY_PREDICATION", True),
    ("TL_DISABLE_SAFE_ROBUST_COPY_PREDICATION", True),
    ("TL_CONFIG_INDEX_BITWIDTH", 32),
):
    if hasattr(tilelang.PassConfigKey, _key):
        _CAUSAL_CONV1D_PASS_CONFIGS[getattr(tilelang.PassConfigKey, _key)] = _value


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _tilelang_index_dtype(dtype: torch.dtype) -> str:
    if dtype is torch.int32:
        return "int32"
    if dtype is torch.int64:
        return "int64"
    raise TypeError(f"Unsupported index dtype for TileLang MUSA kernel: {dtype}")


@functools.lru_cache(maxsize=64)
@tilelang.jit(
    out_idx=[],
    target="musa",
    pass_configs=_CAUSAL_CONV1D_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _causal_conv1d_fwd_kernel(
    dtype: str,
    cache_indices_dtype: str,
    width: int,
    x_stride_dim: int,
    x_stride_token: int,
    w_stride_dim: int,
    w_stride_width: int,
    state_stride_seq: int,
    state_stride_dim: int,
    state_stride_token: int,
    cache_indices_stride: int,
    o_stride_dim: int,
    o_stride_token: int,
    has_bias: bool,
    has_conv_states: bool,
    has_cache_indices: bool,
    has_cache_index_mapping: bool,
    has_initial_states: bool,
    use_pad_slot: bool,
    silu_activation: bool,
    block_m: int,
    block_n: int,
):
    x_numel = T.dynamic("x_numel")
    w_numel = T.dynamic("w_numel")
    bias_numel = T.dynamic("bias_numel")
    state_numel = T.dynamic("state_numel")
    cache_numel = T.dynamic("cache_numel")
    mapping_numel = T.dynamic("mapping_numel")
    init_numel = T.dynamic("init_numel")
    query_numel = T.dynamic("query_numel")
    out_numel = T.dynamic("out_numel")
    state_len = width - 1

    @T.prim_func
    def musa_causal_conv1d_fwd(
        x: T.Tensor((x_numel,), dtype),
        weight: T.Tensor((w_numel,), dtype),
        bias: T.Tensor((bias_numel,), dtype),
        conv_states: T.Tensor((state_numel,), dtype),
        cache_indices: T.Tensor((cache_numel,), cache_indices_dtype),
        cache_index_mapping: T.Tensor((mapping_numel,), "int32"),
        has_initial_state: T.Tensor((init_numel,), "bool"),
        query_start_loc: T.Tensor((query_numel,), "int32"),
        out: T.Tensor((out_numel,), dtype),
        max_seq_len: T.int32,
        dim: T.int32,
        num_cache_lines: T.int32,
        pad_slot_id: T.int32,
    ):
        with T.Kernel(
            query_numel - 1,
            T.ceildiv(max_seq_len, block_m),
            T.ceildiv(dim, block_n),
            threads=block_n,
        ) as (seq_idx, chunk_idx, dim_block):
            tid = T.get_thread_binding()
            feat = dim_block * block_n + tid
            seq_start = T.alloc_var("int32")
            seq_end = T.alloc_var("int32")
            seq_len = T.alloc_var("int32")
            token_offset = T.alloc_var("int32")
            segment_len = T.alloc_var("int32")
            cache_idx = T.alloc_var("int32")
            load_init = T.alloc_var("bool")
            state_base = T.alloc_var("int32")
            x_base = T.alloc_var("int32")
            w_base = T.alloc_var("int32")
            out_base = T.alloc_var("int32")
            valid_seq = T.alloc_var("bool")
            col0 = T.alloc_var("float32")
            col1 = T.alloc_var("float32")
            col2 = T.alloc_var("float32")
            col3 = T.alloc_var("float32")
            w0 = T.alloc_var("float32")
            w1 = T.alloc_var("float32")
            w2 = T.alloc_var("float32")
            w3 = T.alloc_var("float32")
            w4 = T.alloc_var("float32")
            x_cur = T.alloc_var("float32")
            acc = T.alloc_var("float32")
            state_src = T.alloc_var("int32")
            state_cut = T.alloc_var("int32")

            seq_start = query_start_loc[seq_idx]
            seq_end = query_start_loc[seq_idx + 1]
            seq_len = seq_end - seq_start
            token_offset = chunk_idx * block_m
            segment_len = seq_len - token_offset
            if segment_len > block_m:
                segment_len = block_m

            cache_idx = seq_idx
            if has_cache_indices:
                cache_idx = cache_indices[seq_idx * cache_indices_stride]
                if has_cache_index_mapping:
                    cache_idx = cache_index_mapping[cache_idx]
            valid_seq = segment_len > 0
            if use_pad_slot and cache_idx == pad_slot_id:
                valid_seq = False
            if has_conv_states and cache_idx >= num_cache_lines:
                valid_seq = False

            if valid_seq and feat < dim:
                x_base = seq_start * x_stride_token + feat * x_stride_dim
                w_base = feat * w_stride_dim
                out_base = seq_start * o_stride_token + feat * o_stride_dim
                state_base = cache_idx * state_stride_seq + feat * state_stride_dim

                col0 = 0.0
                col1 = 0.0
                col2 = 0.0
                col3 = 0.0
                load_init = False
                if has_initial_states:
                    load_init = has_initial_state[seq_idx]

                if chunk_idx == 0:
                    if has_conv_states and load_init:
                        if width >= 2:
                            col0 = T.Cast(
                                "float32",
                                conv_states[
                                    state_base + (state_len - 1) * state_stride_token
                                ],
                            )
                        if width >= 3:
                            col1 = col0
                            col0 = T.Cast(
                                "float32",
                                conv_states[
                                    state_base + (state_len - 2) * state_stride_token
                                ],
                            )
                        if width >= 4:
                            col2 = col1
                            col1 = T.Cast(
                                "float32",
                                conv_states[
                                    state_base + (state_len - 2) * state_stride_token
                                ],
                            )
                            col0 = T.Cast(
                                "float32",
                                conv_states[
                                    state_base + (state_len - 3) * state_stride_token
                                ],
                            )
                        if width >= 5:
                            col3 = col2
                            col2 = T.Cast(
                                "float32",
                                conv_states[
                                    state_base + (state_len - 2) * state_stride_token
                                ],
                            )
                            col1 = T.Cast(
                                "float32",
                                conv_states[
                                    state_base + (state_len - 3) * state_stride_token
                                ],
                            )
                            col0 = T.Cast(
                                "float32",
                                conv_states[
                                    state_base + (state_len - 4) * state_stride_token
                                ],
                            )

                    if has_conv_states:
                        state_cut = state_len - seq_len
                        for state_i in T.serial(state_len):
                            if state_len <= seq_len:
                                conv_states[
                                    state_base + state_i * state_stride_token
                                ] = x[
                                    x_base
                                    + (seq_len - state_len + state_i) * x_stride_token
                                ]
                            else:
                                if load_init and state_i < state_cut:
                                    state_src = state_i + seq_len
                                    conv_states[
                                        state_base + state_i * state_stride_token
                                    ] = conv_states[
                                        state_base + state_src * state_stride_token
                                    ]
                                elif state_i >= state_cut:
                                    conv_states[
                                        state_base + state_i * state_stride_token
                                    ] = x[
                                        x_base + (state_i - state_cut) * x_stride_token
                                    ]
                                else:
                                    conv_states[
                                        state_base + state_i * state_stride_token
                                    ] = T.Cast(dtype, 0.0)
                else:
                    if width >= 2:
                        col0 = T.Cast(
                            "float32",
                            x[x_base + (token_offset - 1) * x_stride_token],
                        )
                    if width >= 3:
                        col1 = col0
                        col0 = T.Cast(
                            "float32",
                            x[x_base + (token_offset - 2) * x_stride_token],
                        )
                    if width >= 4:
                        col2 = col1
                        col1 = T.Cast(
                            "float32",
                            x[x_base + (token_offset - 2) * x_stride_token],
                        )
                        col0 = T.Cast(
                            "float32",
                            x[x_base + (token_offset - 3) * x_stride_token],
                        )
                    if width >= 5:
                        col3 = col2
                        col2 = T.Cast(
                            "float32",
                            x[x_base + (token_offset - 2) * x_stride_token],
                        )
                        col1 = T.Cast(
                            "float32",
                            x[x_base + (token_offset - 3) * x_stride_token],
                        )
                        col0 = T.Cast(
                            "float32",
                            x[x_base + (token_offset - 4) * x_stride_token],
                        )

                w0 = T.Cast("float32", weight[w_base])
                w1 = T.Cast("float32", weight[w_base + w_stride_width])
                if width >= 3:
                    w2 = T.Cast("float32", weight[w_base + 2 * w_stride_width])
                if width >= 4:
                    w3 = T.Cast("float32", weight[w_base + 3 * w_stride_width])
                if width >= 5:
                    w4 = T.Cast("float32", weight[w_base + 4 * w_stride_width])

                for token_i in T.serial(block_m):
                    if token_i < segment_len:
                        x_cur = T.Cast(
                            "float32",
                            x[x_base + (token_offset + token_i) * x_stride_token],
                        )
                        acc = 0.0
                        if has_bias:
                            acc = T.Cast("float32", bias[feat])

                        if width == 2:
                            acc += col0 * w0 + x_cur * w1
                            col0 = x_cur
                        elif width == 3:
                            acc += col0 * w0 + col1 * w1 + x_cur * w2
                            col0 = col1
                            col1 = x_cur
                        elif width == 4:
                            acc += col0 * w0 + col1 * w1 + col2 * w2 + x_cur * w3
                            col0 = col1
                            col1 = col2
                            col2 = x_cur
                        else:
                            acc += (
                                col0 * w0
                                + col1 * w1
                                + col2 * w2
                                + col3 * w3
                                + x_cur * w4
                            )
                            col0 = col1
                            col1 = col2
                            col2 = col3
                            col3 = x_cur

                        if silu_activation:
                            acc = acc / (1.0 + T.exp2(-acc * _LOG2E))
                        out[out_base + (token_offset + token_i) * o_stride_token] = (
                            T.Cast(dtype, acc)
                        )

    return musa_causal_conv1d_fwd


_causal_conv1d_fwd_kernel.mode = "lazy"


@functools.lru_cache(maxsize=64)
@tilelang.jit(
    out_idx=[],
    target="musa",
    pass_configs=_CAUSAL_CONV1D_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _causal_conv1d_fwd_width4_vec_kernel(
    dtype: str,
    cache_indices_dtype: str,
    x_stride_token: int,
    w_stride_dim: int,
    state_stride_seq: int,
    state_stride_dim: int,
    state_stride_token: int,
    o_stride_token: int,
    has_bias: bool,
    has_conv_states: bool,
    has_cache_indices: bool,
    has_cache_index_mapping: bool,
    has_initial_states: bool,
    use_pad_slot: bool,
    silu_activation: bool,
    block_m: int,
    block_feats: int,
    vec_elems: int,
):
    x_numel = T.dynamic("x_numel")
    w_numel = T.dynamic("w_numel")
    bias_numel = T.dynamic("bias_numel")
    state_numel = T.dynamic("state_numel")
    cache_numel = T.dynamic("cache_numel")
    mapping_numel = T.dynamic("mapping_numel")
    init_numel = T.dynamic("init_numel")
    query_numel = T.dynamic("query_numel")
    out_numel = T.dynamic("out_numel")
    num_threads = block_feats // vec_elems
    state_len = 3

    @T.prim_func
    def musa_causal_conv1d_fwd_width4_vec(
        x: T.Tensor((x_numel,), dtype),
        weight: T.Tensor((w_numel,), dtype),
        bias: T.Tensor((bias_numel,), dtype),
        conv_states: T.Tensor((state_numel,), dtype),
        cache_indices: T.Tensor((cache_numel,), cache_indices_dtype),
        cache_index_mapping: T.Tensor((mapping_numel,), "int32"),
        has_initial_state: T.Tensor((init_numel,), "bool"),
        query_start_loc: T.Tensor((query_numel,), "int32"),
        out: T.Tensor((out_numel,), dtype),
        max_seq_len: T.int32,
        dim: T.int32,
        num_cache_lines: T.int32,
        pad_slot_id: T.int32,
    ):
        with T.Kernel(
            query_numel - 1,
            T.ceildiv(max_seq_len, block_m),
            T.ceildiv(dim, block_feats),
            threads=num_threads,
        ) as (seq_idx, chunk_idx, dim_block):
            tid = T.get_thread_binding()
            feat_base = dim_block * block_feats + tid * vec_elems
            seq_start = T.alloc_var("int32")
            seq_end = T.alloc_var("int32")
            seq_len = T.alloc_var("int32")
            token_offset = T.alloc_var("int32")
            segment_len = T.alloc_var("int32")
            cache_idx = T.alloc_var("int32")
            load_init = T.alloc_var("bool")
            valid_seq = T.alloc_var("bool")
            state_cut = T.alloc_var("int32")
            state_base = T.alloc_local((vec_elems,), "int32")
            x_base = T.alloc_local((vec_elems,), "int32")
            w_base = T.alloc_local((vec_elems,), "int32")
            out_base = T.alloc_local((vec_elems,), "int32")
            col0 = T.alloc_local((vec_elems,), "float32")
            col1 = T.alloc_local((vec_elems,), "float32")
            col2 = T.alloc_local((vec_elems,), "float32")
            w0 = T.alloc_local((vec_elems,), "float32")
            w1 = T.alloc_local((vec_elems,), "float32")
            w2 = T.alloc_local((vec_elems,), "float32")
            w3 = T.alloc_local((vec_elems,), "float32")
            x_cur = T.alloc_local((vec_elems,), "float32")
            acc = T.alloc_local((vec_elems,), "float32")

            seq_start = query_start_loc[seq_idx]
            seq_end = query_start_loc[seq_idx + 1]
            seq_len = seq_end - seq_start
            token_offset = chunk_idx * block_m
            segment_len = seq_len - token_offset
            if segment_len > block_m:
                segment_len = block_m

            cache_idx = seq_idx
            if has_cache_indices:
                cache_idx = cache_indices[seq_idx]
                if has_cache_index_mapping:
                    cache_idx = cache_index_mapping[cache_idx]
            valid_seq = segment_len > 0
            if use_pad_slot and cache_idx == pad_slot_id:
                valid_seq = False
            if has_conv_states and cache_idx >= num_cache_lines:
                valid_seq = False
            load_init = False
            if has_initial_states:
                load_init = has_initial_state[seq_idx]

            if valid_seq:
                for v in T.vectorized(vec_elems):
                    feat = feat_base + v
                    col0[v] = 0.0
                    col1[v] = 0.0
                    col2[v] = 0.0
                    x_base[v] = seq_start * x_stride_token + feat
                    w_base[v] = feat * w_stride_dim
                    out_base[v] = seq_start * o_stride_token + feat
                    state_base[v] = (
                        cache_idx * state_stride_seq + feat * state_stride_dim
                    )
                    if feat < dim:
                        if chunk_idx == 0:
                            if has_conv_states and load_init:
                                col2[v] = T.Cast(
                                    "float32",
                                    conv_states[state_base[v] + 2 * state_stride_token],
                                )
                                col1[v] = T.Cast(
                                    "float32",
                                    conv_states[state_base[v] + state_stride_token],
                                )
                                col0[v] = T.Cast("float32", conv_states[state_base[v]])
                        else:
                            col2[v] = T.Cast(
                                "float32",
                                x[x_base[v] + (token_offset - 1) * x_stride_token],
                            )
                            col1[v] = T.Cast(
                                "float32",
                                x[x_base[v] + (token_offset - 2) * x_stride_token],
                            )
                            col0[v] = T.Cast(
                                "float32",
                                x[x_base[v] + (token_offset - 3) * x_stride_token],
                            )

                        w0[v] = T.Cast("float32", weight[w_base[v]])
                        w1[v] = T.Cast("float32", weight[w_base[v] + 1])
                        w2[v] = T.Cast("float32", weight[w_base[v] + 2])
                        w3[v] = T.Cast("float32", weight[w_base[v] + 3])

                if chunk_idx == 0 and has_conv_states:
                    state_cut = state_len - seq_len
                    for state_i in T.serial(state_len):
                        for v in T.vectorized(vec_elems):
                            feat = feat_base + v
                            if feat < dim:
                                if seq_len >= state_len:
                                    conv_states[
                                        state_base[v] + state_i * state_stride_token
                                    ] = x[
                                        x_base[v]
                                        + (seq_len - state_len + state_i)
                                        * x_stride_token
                                    ]
                                else:
                                    if load_init and state_i < state_cut:
                                        conv_states[
                                            state_base[v] + state_i * state_stride_token
                                        ] = conv_states[
                                            state_base[v]
                                            + (state_i + seq_len) * state_stride_token
                                        ]
                                    elif state_i >= state_cut:
                                        conv_states[
                                            state_base[v] + state_i * state_stride_token
                                        ] = x[
                                            x_base[v]
                                            + (state_i - state_cut) * x_stride_token
                                        ]
                                    else:
                                        conv_states[
                                            state_base[v] + state_i * state_stride_token
                                        ] = T.Cast(dtype, 0.0)

                for token_i in T.serial(block_m):
                    if token_i < segment_len:
                        for v in T.vectorized(vec_elems):
                            feat = feat_base + v
                            if feat < dim:
                                x_cur[v] = T.Cast(
                                    "float32",
                                    x[
                                        x_base[v]
                                        + (token_offset + token_i) * x_stride_token
                                    ],
                                )
                                acc[v] = 0.0
                                if has_bias:
                                    acc[v] = T.Cast("float32", bias[feat])
                                acc[v] += (
                                    col0[v] * w0[v]
                                    + col1[v] * w1[v]
                                    + col2[v] * w2[v]
                                    + x_cur[v] * w3[v]
                                )
                                col0[v] = col1[v]
                                col1[v] = col2[v]
                                col2[v] = x_cur[v]
                                if silu_activation:
                                    acc[v] = acc[v] / (1.0 + T.exp2(-acc[v] * _LOG2E))
                                out[
                                    out_base[v]
                                    + (token_offset + token_i) * o_stride_token
                                ] = T.Cast(dtype, acc[v])

    return musa_causal_conv1d_fwd_width4_vec


_causal_conv1d_fwd_width4_vec_kernel.mode = "lazy"


@functools.lru_cache(maxsize=64)
@tilelang.jit(
    out_idx=[],
    target="musa",
    pass_configs=_CAUSAL_CONV1D_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _causal_conv1d_prefill_width4_kernel(
    dtype: str,
    cache_indices_dtype: str,
    x_stride_token: int,
    w_stride_dim: int,
    state_stride_seq: int,
    state_stride_dim: int,
    state_stride_token: int,
    o_stride_token: int,
    has_bias: bool,
    has_cache_indices: bool,
    has_cache_index_mapping: bool,
    has_initial_states: bool,
    use_pad_slot: bool,
    silu_activation: bool,
    block_m: int,
    block_n: int,
):
    x_numel = T.dynamic("x_numel")
    w_numel = T.dynamic("w_numel")
    bias_numel = T.dynamic("bias_numel")
    state_numel = T.dynamic("state_numel")
    cache_numel = T.dynamic("cache_numel")
    mapping_numel = T.dynamic("mapping_numel")
    init_numel = T.dynamic("init_numel")
    query_numel = T.dynamic("query_numel")
    out_numel = T.dynamic("out_numel")

    @T.prim_func
    def musa_causal_conv1d_prefill_width4(
        x: T.Tensor((x_numel,), dtype),
        weight: T.Tensor((w_numel,), dtype),
        bias: T.Tensor((bias_numel,), dtype),
        conv_states: T.Tensor((state_numel,), dtype),
        cache_indices: T.Tensor((cache_numel,), cache_indices_dtype),
        cache_index_mapping: T.Tensor((mapping_numel,), "int32"),
        has_initial_state: T.Tensor((init_numel,), "bool"),
        query_start_loc: T.Tensor((query_numel,), "int32"),
        out: T.Tensor((out_numel,), dtype),
        max_seq_len: T.int32,
        dim: T.int32,
        num_cache_lines: T.int32,
        pad_slot_id: T.int32,
    ):
        with T.Kernel(
            query_numel - 1,
            T.ceildiv(max_seq_len, block_m),
            T.ceildiv(dim, block_n),
            threads=block_n,
        ) as (seq_idx, chunk_idx, dim_block):
            tid = T.get_thread_binding()
            feat = dim_block * block_n + tid
            seq_start = T.alloc_var("int32")
            seq_end = T.alloc_var("int32")
            seq_len = T.alloc_var("int32")
            token_offset = T.alloc_var("int32")
            segment_len = T.alloc_var("int32")
            cache_idx = T.alloc_var("int32")
            valid_seq = T.alloc_var("bool")
            load_init = T.alloc_var("bool")
            x_base = T.alloc_var("int32")
            w_base = T.alloc_var("int32")
            state_base = T.alloc_var("int32")
            out_base = T.alloc_var("int32")
            col0 = T.alloc_var("float32")
            col1 = T.alloc_var("float32")
            col2 = T.alloc_var("float32")
            w0 = T.alloc_var("float32")
            w1 = T.alloc_var("float32")
            w2 = T.alloc_var("float32")
            w3 = T.alloc_var("float32")
            x_cur = T.alloc_var("float32")
            acc = T.alloc_var("float32")

            seq_start = query_start_loc[seq_idx]
            seq_end = query_start_loc[seq_idx + 1]
            seq_len = seq_end - seq_start
            token_offset = chunk_idx * block_m
            segment_len = seq_len - token_offset
            if segment_len > block_m:
                segment_len = block_m

            cache_idx = seq_idx
            if has_cache_indices:
                cache_idx = cache_indices[seq_idx]
                if has_cache_index_mapping:
                    cache_idx = cache_index_mapping[cache_idx]
            valid_seq = segment_len > 0 and feat < dim
            if use_pad_slot and cache_idx == pad_slot_id:
                valid_seq = False
            if cache_idx >= num_cache_lines:
                valid_seq = False

            if valid_seq:
                x_base = seq_start * x_stride_token + feat
                w_base = feat * w_stride_dim
                out_base = seq_start * o_stride_token + feat
                state_base = cache_idx * state_stride_seq + feat * state_stride_dim

                col0 = 0.0
                col1 = 0.0
                col2 = 0.0
                load_init = False
                if has_initial_states:
                    load_init = has_initial_state[seq_idx]

                if chunk_idx == 0:
                    if load_init:
                        col0 = T.Cast("float32", conv_states[state_base])
                        col1 = T.Cast(
                            "float32",
                            conv_states[state_base + state_stride_token],
                        )
                        col2 = T.Cast(
                            "float32",
                            conv_states[state_base + 2 * state_stride_token],
                        )

                    if seq_len >= 3:
                        conv_states[state_base] = x[
                            x_base + (seq_len - 3) * x_stride_token
                        ]
                        conv_states[state_base + state_stride_token] = x[
                            x_base + (seq_len - 2) * x_stride_token
                        ]
                        conv_states[state_base + 2 * state_stride_token] = x[
                            x_base + (seq_len - 1) * x_stride_token
                        ]
                    else:
                        if seq_len == 1:
                            if load_init:
                                conv_states[state_base] = conv_states[
                                    state_base + state_stride_token
                                ]
                                conv_states[state_base + state_stride_token] = (
                                    conv_states[state_base + 2 * state_stride_token]
                                )
                            else:
                                conv_states[state_base] = T.Cast(dtype, 0.0)
                                conv_states[state_base + state_stride_token] = T.Cast(
                                    dtype, 0.0
                                )
                            conv_states[state_base + 2 * state_stride_token] = x[x_base]
                        else:
                            if load_init:
                                conv_states[state_base] = conv_states[
                                    state_base + 2 * state_stride_token
                                ]
                            else:
                                conv_states[state_base] = T.Cast(dtype, 0.0)
                            conv_states[state_base + state_stride_token] = x[x_base]
                            conv_states[state_base + 2 * state_stride_token] = x[
                                x_base + x_stride_token
                            ]
                else:
                    col2 = T.Cast(
                        "float32",
                        x[x_base + (token_offset - 1) * x_stride_token],
                    )
                    col1 = T.Cast(
                        "float32",
                        x[x_base + (token_offset - 2) * x_stride_token],
                    )
                    col0 = T.Cast(
                        "float32",
                        x[x_base + (token_offset - 3) * x_stride_token],
                    )

                w0 = T.Cast("float32", weight[w_base])
                w1 = T.Cast("float32", weight[w_base + 1])
                w2 = T.Cast("float32", weight[w_base + 2])
                w3 = T.Cast("float32", weight[w_base + 3])

                for token_i in T.serial(block_m):
                    if token_i < segment_len:
                        x_cur = T.Cast(
                            "float32",
                            x[x_base + (token_offset + token_i) * x_stride_token],
                        )
                        acc = 0.0
                        if has_bias:
                            acc = T.Cast("float32", bias[feat])
                        acc += col0 * w0 + col1 * w1 + col2 * w2 + x_cur * w3
                        col0 = col1
                        col1 = col2
                        col2 = x_cur
                        if silu_activation:
                            acc = acc / (1.0 + T.exp2(-acc * _LOG2E))
                        out[out_base + (token_offset + token_i) * o_stride_token] = (
                            T.Cast(dtype, acc)
                        )

    return musa_causal_conv1d_prefill_width4


_causal_conv1d_prefill_width4_kernel.mode = "lazy"


@functools.lru_cache(maxsize=64)
@tilelang.jit(
    out_idx=[],
    target="musa",
    pass_configs=_CAUSAL_CONV1D_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _causal_conv1d_prefill_width4_body_kernel(
    dtype: str,
    cache_indices_dtype: str,
    x_stride_token: int,
    w_stride_dim: int,
    o_stride_token: int,
    has_bias: bool,
    has_cache_indices: bool,
    has_cache_index_mapping: bool,
    use_pad_slot: bool,
    silu_activation: bool,
    block_m: int,
    block_n: int,
):
    x_numel = T.dynamic("x_numel")
    w_numel = T.dynamic("w_numel")
    bias_numel = T.dynamic("bias_numel")
    cache_numel = T.dynamic("cache_numel")
    mapping_numel = T.dynamic("mapping_numel")
    query_numel = T.dynamic("query_numel")
    out_numel = T.dynamic("out_numel")

    @T.prim_func
    def musa_causal_conv1d_prefill_width4_body(
        x: T.Tensor((x_numel,), dtype),
        weight: T.Tensor((w_numel,), dtype),
        bias: T.Tensor((bias_numel,), dtype),
        cache_indices: T.Tensor((cache_numel,), cache_indices_dtype),
        cache_index_mapping: T.Tensor((mapping_numel,), "int32"),
        query_start_loc: T.Tensor((query_numel,), "int32"),
        out: T.Tensor((out_numel,), dtype),
        max_seq_len: T.int32,
        dim: T.int32,
        num_cache_lines: T.int32,
        pad_slot_id: T.int32,
    ):
        with T.Kernel(
            query_numel - 1,
            T.ceildiv(max_seq_len - block_m, block_m),
            T.ceildiv(dim, block_n),
            threads=block_n,
        ) as (seq_idx, body_chunk_idx, dim_block):
            tid = T.get_thread_binding()
            feat = dim_block * block_n + tid
            seq_start = T.alloc_var("int32")
            seq_end = T.alloc_var("int32")
            seq_len = T.alloc_var("int32")
            token_offset = T.alloc_var("int32")
            segment_len = T.alloc_var("int32")
            cache_idx = T.alloc_var("int32")
            valid_seq = T.alloc_var("bool")
            x_base = T.alloc_var("int32")
            w_base = T.alloc_var("int32")
            out_base = T.alloc_var("int32")
            col0 = T.alloc_var("float32")
            col1 = T.alloc_var("float32")
            col2 = T.alloc_var("float32")
            w0 = T.alloc_var("float32")
            w1 = T.alloc_var("float32")
            w2 = T.alloc_var("float32")
            w3 = T.alloc_var("float32")
            x_cur = T.alloc_var("float32")
            acc = T.alloc_var("float32")

            seq_start = query_start_loc[seq_idx]
            seq_end = query_start_loc[seq_idx + 1]
            seq_len = seq_end - seq_start
            token_offset = (body_chunk_idx + 1) * block_m
            segment_len = seq_len - token_offset
            if segment_len > block_m:
                segment_len = block_m

            cache_idx = seq_idx
            if has_cache_indices:
                cache_idx = cache_indices[seq_idx]
                if has_cache_index_mapping:
                    cache_idx = cache_index_mapping[cache_idx]
            valid_seq = segment_len > 0 and feat < dim
            if use_pad_slot and cache_idx == pad_slot_id:
                valid_seq = False
            if cache_idx >= num_cache_lines:
                valid_seq = False

            if valid_seq:
                x_base = seq_start * x_stride_token + feat
                w_base = feat * w_stride_dim
                out_base = seq_start * o_stride_token + feat

                col2 = T.Cast(
                    "float32",
                    x[x_base + (token_offset - 1) * x_stride_token],
                )
                col1 = T.Cast(
                    "float32",
                    x[x_base + (token_offset - 2) * x_stride_token],
                )
                col0 = T.Cast(
                    "float32",
                    x[x_base + (token_offset - 3) * x_stride_token],
                )
                w0 = T.Cast("float32", weight[w_base])
                w1 = T.Cast("float32", weight[w_base + 1])
                w2 = T.Cast("float32", weight[w_base + 2])
                w3 = T.Cast("float32", weight[w_base + 3])

                if segment_len == block_m:
                    for token_i in T.serial(block_m):
                        x_cur = T.Cast(
                            "float32",
                            x[x_base + (token_offset + token_i) * x_stride_token],
                        )
                        acc = 0.0
                        if has_bias:
                            acc = T.Cast("float32", bias[feat])
                        acc += col0 * w0 + col1 * w1 + col2 * w2 + x_cur * w3
                        col0 = col1
                        col1 = col2
                        col2 = x_cur
                        if silu_activation:
                            acc = acc / (1.0 + T.exp2(-acc * _LOG2E))
                        out[out_base + (token_offset + token_i) * o_stride_token] = (
                            T.Cast(dtype, acc)
                        )
                else:
                    for token_i in T.serial(block_m):
                        if token_i < segment_len:
                            x_cur = T.Cast(
                                "float32",
                                x[x_base + (token_offset + token_i) * x_stride_token],
                            )
                            acc = 0.0
                            if has_bias:
                                acc = T.Cast("float32", bias[feat])
                            acc += col0 * w0 + col1 * w1 + col2 * w2 + x_cur * w3
                            col0 = col1
                            col1 = col2
                            col2 = x_cur
                            if silu_activation:
                                acc = acc / (1.0 + T.exp2(-acc * _LOG2E))
                            out[
                                out_base + (token_offset + token_i) * o_stride_token
                            ] = T.Cast(dtype, acc)

    return musa_causal_conv1d_prefill_width4_body


_causal_conv1d_prefill_width4_body_kernel.mode = "lazy"


@functools.lru_cache(maxsize=64)
@tilelang.jit(
    out_idx=[],
    target="musa",
    pass_configs=_CAUSAL_CONV1D_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _causal_conv1d_decode_width4_batched_kernel(
    x_dtype: str,
    weight_dtype: str,
    bias_dtype: str,
    state_dtype: str,
    out_dtype: str,
    cache_indices_dtype: str,
    x_stride_token: int,
    w_stride_dim: int,
    state_stride_seq: int,
    state_stride_dim: int,
    state_stride_token: int,
    cache_indices_stride: int,
    o_stride_token: int,
    has_bias: bool,
    has_cache_indices: bool,
    has_cache_index_mapping: bool,
    has_initial_states: bool,
    use_pad_slot: bool,
    silu_activation: bool,
    block_feats: int,
    batch_per_block: int,
):
    x_numel = T.dynamic("x_numel")
    w_numel = T.dynamic("w_numel")
    bias_numel = T.dynamic("bias_numel")
    state_numel = T.dynamic("state_numel")
    cache_numel = T.dynamic("cache_numel")
    mapping_numel = T.dynamic("mapping_numel")
    init_numel = T.dynamic("init_numel")
    out_numel = T.dynamic("out_numel")
    num_threads = block_feats * batch_per_block

    @T.prim_func
    def musa_causal_conv1d_decode_width4_batched(
        x: T.Tensor((x_numel,), x_dtype),
        weight: T.Tensor((w_numel,), weight_dtype),
        bias: T.Tensor((bias_numel,), bias_dtype),
        conv_states: T.Tensor((state_numel,), state_dtype),
        cache_indices: T.Tensor((cache_numel,), cache_indices_dtype),
        cache_index_mapping: T.Tensor((mapping_numel,), "int32"),
        has_initial_state: T.Tensor((init_numel,), "bool"),
        out: T.Tensor((out_numel,), out_dtype),
        batch: T.int32,
        dim: T.int32,
        num_cache_lines: T.int32,
        pad_slot_id: T.int32,
    ):
        with T.Kernel(
            T.ceildiv(batch, batch_per_block),
            T.ceildiv(dim, block_feats),
            threads=num_threads,
        ) as (batch_block, dim_block):
            tid = T.get_thread_binding()
            batch_lane = tid // block_feats
            feat_lane = tid - batch_lane * block_feats
            seq_idx = batch_block * batch_per_block + batch_lane
            feat = dim_block * block_feats + feat_lane
            cache_idx = T.alloc_var("int32")
            valid = T.alloc_var("bool")
            load_init = T.alloc_var("bool")
            x_base = T.alloc_var("int32")
            w_base = T.alloc_var("int32")
            state_base = T.alloc_var("int32")
            col0 = T.alloc_var("float32")
            col1 = T.alloc_var("float32")
            col2 = T.alloc_var("float32")
            x_cur = T.alloc_var("float32")
            acc = T.alloc_var("float32")

            cache_idx = seq_idx
            if has_cache_indices and seq_idx < batch:
                cache_idx = cache_indices[seq_idx * cache_indices_stride]
                if has_cache_index_mapping:
                    cache_idx = cache_index_mapping[cache_idx]
            valid = seq_idx < batch and feat < dim
            if use_pad_slot and cache_idx == pad_slot_id:
                valid = False
            if cache_idx >= num_cache_lines:
                valid = False

            # The upstream update kernel aliases its output with x and returns
            # before touching a null cache block.  Preserve that output value
            # while leaving the null block unchanged.
            if seq_idx < batch and feat < dim and not valid:
                x_base = seq_idx * x_stride_token + feat
                out[seq_idx * o_stride_token + feat] = T.Cast(out_dtype, x[x_base])

            if valid:
                load_init = False
                if has_initial_states:
                    load_init = has_initial_state[seq_idx]

                x_base = seq_idx * x_stride_token + feat
                w_base = feat * w_stride_dim
                state_base = cache_idx * state_stride_seq + feat * state_stride_dim
                x_cur = T.Cast("float32", x[x_base])
                col0 = 0.0
                col1 = 0.0
                col2 = 0.0
                if load_init:
                    col0 = T.Cast("float32", conv_states[state_base])
                    col1 = T.Cast(
                        "float32", conv_states[state_base + state_stride_token]
                    )
                    col2 = T.Cast(
                        "float32", conv_states[state_base + 2 * state_stride_token]
                    )

                acc = (
                    col0 * T.Cast("float32", weight[w_base])
                    + col1 * T.Cast("float32", weight[w_base + 1])
                    + col2 * T.Cast("float32", weight[w_base + 2])
                    + x_cur * T.Cast("float32", weight[w_base + 3])
                )
                if has_bias:
                    acc += T.Cast("float32", bias[feat])
                if silu_activation:
                    acc = acc / (1.0 + T.exp2(-acc * _LOG2E))

                out[seq_idx * o_stride_token + feat] = T.Cast(out_dtype, acc)
                if load_init:
                    conv_states[state_base] = conv_states[
                        state_base + state_stride_token
                    ]
                    conv_states[state_base + state_stride_token] = conv_states[
                        state_base + 2 * state_stride_token
                    ]
                else:
                    conv_states[state_base] = T.Cast(state_dtype, 0.0)
                    conv_states[state_base + state_stride_token] = T.Cast(
                        state_dtype, 0.0
                    )
                conv_states[state_base + 2 * state_stride_token] = T.Cast(
                    state_dtype, x_cur
                )

    return musa_causal_conv1d_decode_width4_batched


_causal_conv1d_decode_width4_batched_kernel.mode = "lazy"


def _check_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    cache_index_mapping: torch.Tensor | None,
    activation: str | bool | None,
    seq_lens_cpu: list[int],
) -> tuple[int, int, int, int, str]:
    if isinstance(activation, bool) and activation:
        activation = "silu"
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.dim() != 2:
        raise ValueError(
            "TileLang causal_conv1d_fwd expects varlen x with shape (dim, total_tokens)"
        )
    if weight.dim() != 2:
        raise ValueError("weight must be a 2D tensor")
    if query_start_loc is None or query_start_loc.dim() != 1:
        raise ValueError("query_start_loc must be a 1D tensor")
    if query_start_loc.dtype is not torch.int32:
        raise TypeError("query_start_loc must be int32")
    dim, total_tokens = x.shape
    weight_dim, width = weight.shape
    if weight_dim != dim:
        raise ValueError("weight first dimension must match x dim")
    if width < 2 or width > 5:
        raise ValueError("TileLang causal_conv1d_fwd supports width in [2, 5]")
    if bias is not None and (bias.dim() != 1 or bias.numel() != dim):
        raise ValueError("bias must have shape (dim,)")
    if cache_indices is not None:
        if cache_indices.dim() != 1 or cache_indices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError("cache_indices must be a 1D int32 or int64 tensor")
        if cache_indices.numel() != query_start_loc.numel() - 1:
            raise ValueError("cache_indices length must match batch size")
    if cache_index_mapping is not None:
        if (
            cache_index_mapping.dim() != 1
            or cache_index_mapping.dtype is not torch.int32
        ):
            raise TypeError("cache_index_mapping must be a 1D int32 tensor")
        if cache_indices is None:
            raise ValueError("cache_index_mapping requires cache_indices")
    if has_initial_state is not None:
        if has_initial_state.dim() != 1 or has_initial_state.dtype is not torch.bool:
            raise TypeError("has_initial_state must be a 1D bool tensor")
        if has_initial_state.numel() != query_start_loc.numel() - 1:
            raise ValueError("has_initial_state length must match batch size")
        if conv_states is None:
            raise ValueError("has_initial_state requires conv_states")
    if conv_states is not None:
        if conv_states.dim() != 3:
            raise ValueError(
                "conv_states must have shape (num_cache_lines, dim, state_len)"
            )
        if conv_states.size(1) != dim or conv_states.size(2) < width - 1:
            raise ValueError("conv_states shape is incompatible with x/weight")
    seq_lens_sum = sum(seq_lens_cpu)
    max_seq_len = max(seq_lens_cpu, default=0)
    if seq_lens_sum != total_tokens:
        # CUDA graph / DP padding can leave seq_lens_cpu describing the padded
        # batch while x and query_start_loc describe the actual packed tokens.
        # The kernel gets true per-sequence bounds from query_start_loc, so use
        # a conservative launch bound instead of rejecting a valid packed input.
        max_seq_len = total_tokens
    return (
        dim,
        total_tokens,
        width,
        max_seq_len,
        tilelang_dtype(x.dtype),
    )


def _causal_conv1d_fwd_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    seq_lens_cpu: list[int],
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    cache_index_mapping: torch.Tensor | None = None,
    activation: str | bool | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
) -> torch.Tensor:
    if isinstance(activation, bool) and activation:
        activation = "silu"
    dim, _total_tokens, width, max_seq_len, dtype = _check_inputs(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        cache_index_mapping,
        activation,
        seq_lens_cpu,
    )

    out = torch.empty_like(x)
    x_arg = storage_window(x)
    weight_arg = storage_window(weight)
    out_arg = storage_window(out)
    bias_arg = storage_window(bias) if bias is not None else x_arg
    conv_states_arg = storage_window(conv_states) if conv_states is not None else x_arg
    cache_indices_arg = (
        storage_window(cache_indices) if cache_indices is not None else query_start_loc
    )
    cache_index_mapping_arg = (
        cache_index_mapping if cache_index_mapping is not None else cache_indices_arg
    )
    has_initial_state_arg = (
        has_initial_state
        if has_initial_state is not None
        else torch.empty((1,), dtype=torch.bool, device=x.device)
    )

    block_n = 256 if dim >= 256 else max(32, _next_power_of_2(dim))
    if max_seq_len == 1 and query_start_loc.numel() > 128 and dim >= 256:
        block_n = 128
    block_m = 8
    if width == 4 and max_seq_len >= 128:
        if max_seq_len <= 512:
            block_m = 4
            block_n = 256
        elif max_seq_len < 4096:
            if query_start_loc.numel() > 4:
                block_m = 28
                block_n = 256
            else:
                block_m = 12
                block_n = 128 if query_start_loc.numel() > 2 else 256
        else:
            block_m = 28
            block_n = 256
    num_cache_lines = conv_states.size(0) if conv_states is not None else 0
    state_stride_seq = conv_states.stride(0) if conv_states is not None else 0
    state_stride_dim = conv_states.stride(1) if conv_states is not None else 0
    state_stride_token = conv_states.stride(2) if conv_states is not None else 0
    cache_indices_dtype = (
        _tilelang_index_dtype(cache_indices.dtype)
        if cache_indices is not None
        else "int32"
    )

    if (
        width == 4
        and max_seq_len == 1
        and query_start_loc.numel() > 256
        and conv_states is not None
        and dim >= 4096
        and x.stride(0) == 1
        and out.stride(0) == 1
        and weight.stride(1) == 1
    ):
        _causal_conv1d_decode_width4_batched_kernel(
            tilelang_dtype(x.dtype),
            tilelang_dtype(weight.dtype),
            tilelang_dtype(bias.dtype if bias is not None else x.dtype),
            tilelang_dtype(conv_states.dtype),
            tilelang_dtype(out.dtype),
            cache_indices_dtype,
            int(x.stride(1)),
            int(weight.stride(0)),
            int(state_stride_seq),
            int(state_stride_dim),
            int(state_stride_token),
            int(cache_indices.stride(0)) if cache_indices is not None else 1,
            int(out.stride(1)),
            bias is not None,
            cache_indices is not None,
            cache_index_mapping is not None,
            has_initial_state is not None,
            pad_slot_id is not None,
            activation in ("silu", "swish"),
            256,
            1,
        )(
            x_arg,
            weight_arg,
            bias_arg,
            conv_states_arg,
            cache_indices_arg,
            cache_index_mapping_arg,
            has_initial_state_arg,
            out_arg,
            int(query_start_loc.numel() - 1),
            int(dim),
            int(num_cache_lines),
            int(pad_slot_id if pad_slot_id is not None else PAD_SLOT_ID),
        )
        return out

    if _should_use_width4_prefill_split(
        width=width,
        dim=dim,
        dtype=x.dtype,
        max_seq_len=max_seq_len,
        batch_size=query_start_loc.numel() - 1,
        has_conv_states=conv_states is not None,
        has_cache_indices=cache_indices is not None,
        cache_indices_stride=(
            cache_indices.stride(0) if cache_indices is not None else 1
        ),
        x_inner_stride=x.stride(0),
        out_inner_stride=out.stride(0),
        weight_inner_stride=weight.stride(1),
    ):
        global _WIDTH4_PREFILL_SPLIT_LOGGED
        if not _WIDTH4_PREFILL_SPLIT_LOGGED:
            logger.info_once("MUSA width-4 causal-conv prefill split active")
            _WIDTH4_PREFILL_SPLIT_LOGGED = True
        _causal_conv1d_prefill_width4_kernel(
            dtype,
            cache_indices_dtype,
            int(x.stride(1)),
            int(weight.stride(0)),
            int(state_stride_seq),
            int(state_stride_dim),
            int(state_stride_token),
            int(out.stride(1)),
            bias is not None,
            cache_indices is not None,
            cache_index_mapping is not None,
            has_initial_state is not None,
            pad_slot_id is not None,
            activation in ("silu", "swish"),
            int(block_m),
            int(block_n),
        )(
            x_arg,
            weight_arg,
            bias_arg,
            conv_states_arg,
            cache_indices_arg,
            cache_index_mapping_arg,
            has_initial_state_arg,
            query_start_loc,
            out_arg,
            int(block_m),
            int(dim),
            int(num_cache_lines),
            int(pad_slot_id if pad_slot_id is not None else PAD_SLOT_ID),
        )
        if max_seq_len > block_m:
            _causal_conv1d_prefill_width4_body_kernel(
                dtype,
                cache_indices_dtype,
                int(x.stride(1)),
                int(weight.stride(0)),
                int(out.stride(1)),
                bias is not None,
                cache_indices is not None,
                cache_index_mapping is not None,
                pad_slot_id is not None,
                activation in ("silu", "swish"),
                int(block_m),
                int(block_n),
            )(
                x_arg,
                weight_arg,
                bias_arg,
                cache_indices_arg,
                cache_index_mapping_arg,
                query_start_loc,
                out_arg,
                int(max_seq_len),
                int(dim),
                int(num_cache_lines),
                int(pad_slot_id if pad_slot_id is not None else PAD_SLOT_ID),
            )
        return out

    if (
        width == 4
        and dim >= 4096
        and max_seq_len == 1
        and query_start_loc.numel() == 2
        and x.stride(0) == 1
        and out.stride(0) == 1
        and weight.stride(1) == 1
    ):
        block_feats = 256
        vec_elems = 1
        _causal_conv1d_fwd_width4_vec_kernel(
            dtype,
            cache_indices_dtype,
            int(x.stride(1)),
            int(weight.stride(0)),
            int(state_stride_seq),
            int(state_stride_dim),
            int(state_stride_token),
            int(out.stride(1)),
            bias is not None,
            conv_states is not None,
            cache_indices is not None,
            cache_index_mapping is not None,
            has_initial_state is not None,
            pad_slot_id is not None,
            activation in ("silu", "swish"),
            int(block_m),
            int(block_feats),
            int(vec_elems),
        )(
            x_arg,
            weight_arg,
            bias_arg,
            conv_states_arg,
            cache_indices_arg,
            cache_index_mapping_arg,
            has_initial_state_arg,
            query_start_loc,
            out_arg,
            int(max_seq_len),
            int(dim),
            int(num_cache_lines),
            int(pad_slot_id if pad_slot_id is not None else PAD_SLOT_ID),
        )
        return out

    _causal_conv1d_fwd_kernel(
        dtype,
        cache_indices_dtype,
        int(width),
        int(x.stride(0)),
        int(x.stride(1)),
        int(weight.stride(0)),
        int(weight.stride(1)),
        int(state_stride_seq),
        int(state_stride_dim),
        int(state_stride_token),
        int(cache_indices.stride(0)) if cache_indices is not None else 1,
        int(out.stride(0)),
        int(out.stride(1)),
        bias is not None,
        conv_states is not None,
        cache_indices is not None,
        cache_index_mapping is not None,
        has_initial_state is not None,
        pad_slot_id is not None,
        activation in ("silu", "swish"),
        int(block_m),
        int(block_n),
    )(
        x_arg,
        weight_arg,
        bias_arg,
        conv_states_arg,
        cache_indices_arg,
        cache_index_mapping_arg,
        has_initial_state_arg,
        query_start_loc,
        out_arg,
        int(max_seq_len),
        int(dim),
        int(num_cache_lines),
        int(pad_slot_id if pad_slot_id is not None else PAD_SLOT_ID),
    )
    return out


@register_custom_op(
    op_name="musa_causal_conv1d_fwd",
    mutates_args=["conv_states"],
)
def _causal_conv1d_fwd_custom(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    seq_lens_cpu: list[int],
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    cache_index_mapping: torch.Tensor | None = None,
) -> torch.Tensor:
    return _causal_conv1d_fwd_impl(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        seq_lens_cpu,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        cache_index_mapping=cache_index_mapping,
        activation=activation,
        pad_slot_id=pad_slot_id,
    )


def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    seq_lens_cpu: list[int],
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | bool | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    cache_index_mapping: torch.Tensor | None = None,
) -> torch.Tensor:
    if isinstance(seq_lens_cpu, torch.Tensor) and seq_lens_cpu.device.type != "cpu":
        # Backward-compatible positional form used by the generic mamba wrapper:
        # causal_conv1d_fwd(..., query_start_loc, cache_indices,
        #                   has_initial_state, activation, pad_slot_id)
        old_cache_indices = seq_lens_cpu
        old_has_initial_state = cache_indices
        old_activation = has_initial_state
        old_pad_slot_id = activation
        seq_lens_cpu = query_start_loc.diff().detach().cpu().tolist()
        cache_indices = old_cache_indices
        has_initial_state = old_has_initial_state
        activation = old_activation
        pad_slot_id = old_pad_slot_id
    elif isinstance(seq_lens_cpu, torch.Tensor):
        seq_lens_cpu = seq_lens_cpu.detach().cpu().tolist()
    if isinstance(activation, bool):
        activation = "silu" if activation else None
    return _causal_conv1d_fwd_impl(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        seq_lens_cpu,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        cache_index_mapping=cache_index_mapping,
        activation=activation,
        pad_slot_id=pad_slot_id,
    )


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    seq_lens_cpu: list[int],
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | bool | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    cache_index_mapping: torch.Tensor | None = None,
    **_: object,
) -> torch.Tensor:
    return causal_conv1d_fwd(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        seq_lens_cpu,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        cache_index_mapping=cache_index_mapping,
        activation=activation,
        pad_slot_id=pad_slot_id,
    )


def musa_tilelang_causal_conv1d_fn(
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    cache_indices=None,
    has_initial_state=None,
    activation="silu",
    pad_slot_id=PAD_SLOT_ID,
    cache_index_mapping=None,
    metadata=None,
    **_,
):
    """Drop-in for vllm Triton causal_conv1d_fn (prefill). Synthesizes
    seq_lens_cpu from query_start_loc and matches conv_states dtype."""
    qsl_cpu = getattr(metadata, "non_spec_query_start_loc_cpu", None)
    if qsl_cpu is not None:
        seq_lens_cpu = qsl_cpu.diff().tolist()
    else:
        seq_lens_cpu = query_start_loc.diff().cpu().tolist()
    orig_dtype = x.dtype
    if conv_states is not None and x.dtype != conv_states.dtype:
        x = x.to(conv_states.dtype)
    out = causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        seq_lens_cpu,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation=activation,
        pad_slot_id=pad_slot_id,
        cache_index_mapping=cache_index_mapping,
    )
    return out.to(orig_dtype)


# --- MUSA: single-token decode causal_conv1d (width-4) via the batched TileLang
# decode kernel. Drop-in for vllm Triton causal_conv1d_update; returns None when
# the fast path does not apply so the caller keeps the Triton path.
_DECODE_HAS_INIT_BUF = {}
_DECODE_MIXED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _decode_has_init_buffer(batch, device):
    key = str(device)
    buf = _DECODE_HAS_INIT_BUF.get(key)
    if buf is None or buf.numel() < batch:
        buf = torch.ones(max(int(batch), 2048), device=device, dtype=torch.bool)
        _DECODE_HAS_INIT_BUF[key] = buf
    return buf[:batch]


def musa_tilelang_causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    conv_state_indices=None,
    null_block_id=NULL_BLOCK_ID,
    **_,
):
    """Width-4 single-token decode conv. x:[batch,dim], conv_state:[lines,dim,>=3],
    weight:[dim,width]. Updates conv_state in place. None => use Triton fallback."""
    if conv_state is None or weight.dim() != 2 or weight.shape[1] != 4:
        return None
    if conv_state.dim() != 3 or conv_state.size(2) < 3 or x.dim() != 2:
        return None
    batch, dim = x.shape
    if batch == 0 or dim != weight.size(0) or dim != conv_state.size(1):
        return None
    if x.device != conv_state.device or x.device != weight.device:
        return None
    if x.stride(1) != 1 or weight.stride(1) != 1:
        return None
    if bias is not None and (
        bias.device != x.device
        or bias.dim() != 1
        or bias.numel() != dim
        or bias.stride(0) != 1
    ):
        return None
    if conv_state_indices is not None and (
        conv_state_indices.device != x.device
        or conv_state_indices.dim() != 1
        or conv_state_indices.numel() < batch
        or conv_state_indices.stride(0) != 1
        or conv_state_indices.dtype not in (torch.int32, torch.int64)
    ):
        return None

    same_dtype = (
        x.dtype == conv_state.dtype == weight.dtype
        and (bias is None or bias.dtype == x.dtype)
        and x.dtype in _DECODE_MIXED_DTYPES
    )
    qwen_bf16_fp32_state = (
        x.dtype == torch.bfloat16
        and conv_state.dtype == torch.float32
        and weight.dtype == torch.float32
        and (bias is None or bias.dtype == torch.float32)
    )
    if not (same_dtype or qwen_bf16_fp32_state):
        return None

    xt = x.transpose(0, 1)  # [dim, batch] VIEW, stride(0)==1
    out_bd = torch.empty_like(x)  # [batch, dim]
    outt = out_bd.transpose(0, 1)  # [dim, batch] VIEW, stride(0)==1
    if conv_state_indices is None:
        idx = torch.arange(batch, device=x.device, dtype=torch.int32)
    else:
        idx = conv_state_indices
    has_cache_indices = conv_state_indices is not None
    use_null_block = has_cache_indices and null_block_id is not None
    has_init = _decode_has_init_buffer(batch, x.device)
    silu = activation in ("silu", "swish", True)
    ker = _causal_conv1d_decode_width4_batched_kernel(
        tilelang_dtype(xt.dtype),
        tilelang_dtype(weight.dtype),
        tilelang_dtype(bias.dtype if bias is not None else xt.dtype),
        tilelang_dtype(conv_state.dtype),
        tilelang_dtype(outt.dtype),
        _tilelang_index_dtype(idx.dtype),
        int(xt.stride(1)),
        int(weight.stride(0)),
        int(conv_state.stride(0)),
        int(conv_state.stride(1)),
        int(conv_state.stride(2)),
        int(idx.stride(0)),
        int(outt.stride(1)),
        bias is not None,
        has_cache_indices,
        False,
        True,
        use_null_block,
        silu,
        256,
        1,
    )
    idx_arg = storage_window(idx)
    ker(
        storage_window(xt),
        storage_window(weight),
        storage_window(bias) if bias is not None else storage_window(xt),
        storage_window(conv_state),
        idx_arg,
        idx_arg,
        has_init,
        storage_window(outt),
        int(batch),
        int(dim),
        int(conv_state.size(0)),
        int(null_block_id if use_null_block else PAD_SLOT_ID),
    )
    return out_bd
