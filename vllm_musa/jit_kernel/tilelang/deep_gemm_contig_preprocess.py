"""TileLang MUSA DeepGEMM contiguous preprocess fast paths."""

import functools
from pathlib import Path

import tilelang
import tilelang.language as T
import torch


# MUSA: local passthrough for register_custom_op (SGLang default eager=True returns fn).
def register_custom_op(fn=None, **_kw):
    def _wrap(f):
        return f

    return _wrap if fn is None else _wrap(fn)


_ATOMIC_HELPER_H = str((Path(__file__).resolve().parent / "_atomic_helper.h").resolve())

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}
for _key, _value in (
    ("TL_ENABLE_FAST_MATH", True),
    ("TL_DISABLE_THREAD_STORAGE_SYNC", True),
    ("TL_ENABLE_MUSA_BURST", True),
    ("TL_ENABLE_REDUCE_BURST", True),
    ("TL_DISABLE_SAFE_MEMORY_ACCESS", True),
    ("TL_DISABLE_INDEX_TYPE_PROMOTION", True),
):
    if hasattr(tilelang.PassConfigKey, _key):
        _PASS_CONFIGS[getattr(tilelang.PassConfigKey, _key)] = _value

_COMPILE_FLAGS = [
    "-Od3",
    "-fno-signed-zeros",
    "-fmusa-flush-denormals-to-zero",
    "-mllvm",
    "-misched=mtgpu-max-ilp",
    "-mllvm",
    "-mtgpu-if-convert=1",
    "-mllvm",
    "-mtgpu-tiny-offset-hint=1",
    "-mllvm",
    "-mtgpu-enable-postra-sched=0",
    "-mllvm",
    "-misched-recompute-slotindex=1",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
]


@functools.lru_cache(maxsize=8)
@tilelang.jit(
    out_idx=[], target="musa", pass_configs=_PASS_CONFIGS, compile_flags=_COMPILE_FLAGS
)
def _clear_i32_kernel():
    n = T.dynamic("n")

    @T.prim_func
    def deep_gemm_contig_preprocess_clear_i32_kernel(
        ptr: T.Tensor((n,), "int32"), blocks: T.int32, total: T.int32
    ):
        with T.Kernel(blocks, threads=256) as (bid,):
            tid = T.get_thread_binding()
            idx = bid * 256 + tid
            if idx < total:
                ptr[idx] = T.int32(0)

    return deep_gemm_contig_preprocess_clear_i32_kernel


@functools.lru_cache(maxsize=8)
@tilelang.jit(
    out_idx=[], target="musa", pass_configs=_PASS_CONFIGS, compile_flags=_COMPILE_FLAGS
)
def _fill_i32_kernel():
    n = T.dynamic("n")

    @T.prim_func
    def deep_gemm_contig_preprocess_fill_i32_kernel(
        ptr: T.Tensor((n,), "int32"), blocks: T.int32, total: T.int32, value: T.int32
    ):
        with T.Kernel(blocks, threads=256) as (bid,):
            tid = T.get_thread_binding()
            idx = bid * 256 + tid
            if idx < total:
                ptr[idx] = value

    return deep_gemm_contig_preprocess_fill_i32_kernel


@functools.lru_cache(maxsize=8)
@tilelang.jit(
    out_idx=[], target="musa", pass_configs=_PASS_CONFIGS, compile_flags=_COMPILE_FLAGS
)
def _count_topk_block_hist_kernel(max_experts: int):
    n = T.dynamic("n")
    e = T.dynamic("e")
    slots_per_block = 1024

    @T.prim_func
    def deep_gemm_contig_preprocess_count_topk_block_hist_kernel(
        topk_ids: T.Tensor((n,), "int32"),
        counts: T.Tensor((e,), "int32"),
        blocks: T.int32,
        num_slots: T.int32,
        num_local_experts: T.int32,
    ):
        with T.Kernel(blocks, threads=256) as (bid,):
            tid = T.get_thread_binding()
            local_counts = T.alloc_shared((max_experts,), "int32")

            for i in T.serial(T.ceildiv(max_experts, 256)):
                expert = i * 256 + tid
                if expert < num_local_experts:
                    local_counts[expert] = T.int32(0)
            T.sync_threads()

            block_start = bid * slots_per_block
            for i in T.serial(T.ceildiv(slots_per_block, 256)):
                offset = i * 256 + tid
                slot = block_start + offset
                if slot < num_slots:
                    expert = topk_ids[slot]
                    if expert >= 0 and expert < num_local_experts:
                        T.atomic_add(local_counts[expert], T.int32(1))
            T.sync_threads()

            for i in T.serial(T.ceildiv(max_experts, 256)):
                expert = i * 256 + tid
                if expert < num_local_experts:
                    cnt = local_counts[expert]
                    if cnt != 0:
                        T.atomic_add(counts[expert], cnt)

    return deep_gemm_contig_preprocess_count_topk_block_hist_kernel


@functools.lru_cache(maxsize=8)
@tilelang.jit(
    out_idx=[], target="musa", pass_configs=_PASS_CONFIGS, compile_flags=_COMPILE_FLAGS
)
def _count_topk_single_block_kernel(max_experts: int):
    n = T.dynamic("n")
    e = T.dynamic("e")
    slots_per_block = 1024

    @T.prim_func
    def deep_gemm_contig_preprocess_count_topk_single_block_kernel(
        topk_ids: T.Tensor((n,), "int32"),
        counts: T.Tensor((e,), "int32"),
        num_slots: T.int32,
        num_local_experts: T.int32,
    ):
        with T.Kernel(1, threads=256) as (_bid,):
            tid = T.get_thread_binding()
            local_counts = T.alloc_shared((max_experts,), "int32")

            for i in T.serial(T.ceildiv(max_experts, 256)):
                expert = i * 256 + tid
                if expert < num_local_experts:
                    local_counts[expert] = T.int32(0)
            T.sync_threads()

            for i in T.serial(T.ceildiv(slots_per_block, 256)):
                slot = i * 256 + tid
                if slot < num_slots:
                    expert = topk_ids[slot]
                    if expert >= 0 and expert < num_local_experts:
                        T.atomic_add(local_counts[expert], T.int32(1))
            T.sync_threads()

            for i in T.serial(T.ceildiv(max_experts, 256)):
                expert = i * 256 + tid
                if expert < num_local_experts:
                    counts[expert] = local_counts[expert]

    return deep_gemm_contig_preprocess_count_topk_single_block_kernel


@functools.lru_cache(maxsize=8)
@tilelang.jit(
    out_idx=[], target="musa", pass_configs=_PASS_CONFIGS, compile_flags=_COMPILE_FLAGS
)
def _count_prefix_topk_single_block_kernel(max_experts: int, max_block_m: int):
    n = T.dynamic("n")
    e = T.dynamic("e")
    m = T.dynamic("m")
    slots_per_block = 1024
    scan_size = 1 << (max_experts - 1).bit_length()
    threads = max(256, scan_size)

    @T.prim_func
    def deep_gemm_contig_preprocess_count_prefix_topk_single_block_kernel(
        topk_ids: T.Tensor((n,), "int32"),
        counts: T.Tensor((e,), "int32"),
        cursor: T.Tensor((e,), "int32"),
        m_indices: T.Tensor((m,), "int32"),
        num_slots: T.int32,
        num_local_experts: T.int32,
        block_m: T.int32,
    ):
        with T.Kernel(1, threads=threads) as (_bid,):
            tid = T.get_thread_binding()
            local_counts = T.alloc_shared((max_experts,), "int32")
            prefix = T.alloc_shared((scan_size,), "int32")
            expert = T.alloc_var("int32")
            cnt = T.alloc_var("int32")
            aligned_cnt = T.alloc_var("int32")
            start = T.alloc_var("int32")
            addend = T.alloc_var("int32")

            for i in T.serial(T.ceildiv(max_experts, threads)):
                expert = i * threads + tid
                if expert < num_local_experts:
                    local_counts[expert] = T.int32(0)
            T.sync_threads()

            for i in T.serial(T.ceildiv(slots_per_block, threads)):
                slot = i * threads + tid
                if slot < num_slots:
                    expert = topk_ids[slot]
                    if expert >= 0 and expert < num_local_experts:
                        T.atomic_add(local_counts[expert], T.int32(1))
            T.sync_threads()

            if tid < scan_size:
                if tid < num_local_experts:
                    cnt = local_counts[tid]
                    counts[tid] = cnt
                    prefix[tid] = T.ceildiv(cnt, block_m) * block_m
                else:
                    prefix[tid] = T.int32(0)
            T.sync_threads()

            for offset in T.serial((scan_size.bit_length() - 1)):
                step = 1 << offset
                addend = T.int32(0)
                if tid < scan_size and tid >= step:
                    addend = prefix[tid - step]
                T.sync_threads()
                if tid < scan_size and tid >= step:
                    prefix[tid] = prefix[tid] + addend
                T.sync_threads()

            if tid < num_local_experts:
                cnt = local_counts[tid]
                aligned_cnt = T.ceildiv(cnt, block_m) * block_m
                start = prefix[tid] - aligned_cnt
                cursor[tid] = start
                for pad in T.serial(max_block_m):
                    if pad < aligned_cnt - cnt:
                        m_indices[start + cnt + pad] = tid

    return deep_gemm_contig_preprocess_count_prefix_topk_single_block_kernel


@functools.lru_cache(maxsize=8)
@tilelang.jit(
    out_idx=[], target="musa", pass_configs=_PASS_CONFIGS, compile_flags=_COMPILE_FLAGS
)
def _prefix_counts_kernel(max_experts: int, max_block_m: int):
    e = T.dynamic("e")
    m = T.dynamic("m")

    @T.prim_func
    def deep_gemm_contig_preprocess_prefix_counts(
        counts: T.Tensor((e,), "int32"),
        cursor: T.Tensor((e,), "int32"),
        m_indices: T.Tensor((m,), "int32"),
        num_local_experts: T.int32,
        block_m: T.int32,
    ):
        with T.Kernel(1, threads=256) as (_bid,):
            tid = T.get_thread_binding()
            acc = T.alloc_var("int32")
            cnt = T.alloc_var("int32")
            aligned_cnt = T.alloc_var("int32")
            start = T.alloc_var("int32")

            if tid == 0:
                acc = T.int32(0)
                for expert in T.serial(max_experts):
                    if expert < num_local_experts:
                        cursor[expert] = acc
                        cnt = counts[expert]
                        aligned_cnt = T.ceildiv(cnt, block_m) * block_m
                        acc = acc + aligned_cnt
            T.sync_threads()

            for i in T.serial(T.ceildiv(max_experts, 256)):
                expert = i * 256 + tid
                if expert < num_local_experts:
                    cnt = counts[expert]
                    aligned_cnt = T.ceildiv(cnt, block_m) * block_m
                    start = cursor[expert]
                    for pad in T.serial(max_block_m):
                        if pad < aligned_cnt - cnt:
                            m_indices[start + cnt + pad] = expert

    return deep_gemm_contig_preprocess_prefix_counts


@functools.lru_cache(maxsize=8)
@tilelang.jit(
    out_idx=[], target="musa", pass_configs=_PASS_CONFIGS, compile_flags=_COMPILE_FLAGS
)
def _prefix_counts_scan_kernel(max_experts: int, max_block_m: int):
    e = T.dynamic("e")
    m = T.dynamic("m")
    scan_size = 1 << (max_experts - 1).bit_length()
    threads = max(256, scan_size)

    @T.prim_func
    def deep_gemm_contig_preprocess_prefix_counts_scan(
        counts: T.Tensor((e,), "int32"),
        cursor: T.Tensor((e,), "int32"),
        m_indices: T.Tensor((m,), "int32"),
        num_local_experts: T.int32,
        block_m: T.int32,
    ):
        with T.Kernel(1, threads=threads) as (_bid,):
            tid = T.get_thread_binding()
            prefix = T.alloc_shared((scan_size,), "int32")
            cnt = T.alloc_var("int32")
            aligned_cnt = T.alloc_var("int32")
            start = T.alloc_var("int32")

            if tid < scan_size:
                if tid < num_local_experts:
                    cnt = counts[tid]
                    prefix[tid] = T.ceildiv(cnt, block_m) * block_m
                else:
                    prefix[tid] = T.int32(0)
            T.sync_threads()

            T.cumsum(prefix, prefix, dim=0)
            T.sync_threads()

            if tid < num_local_experts:
                cnt = counts[tid]
                aligned_cnt = T.ceildiv(cnt, block_m) * block_m
                start = prefix[tid] - aligned_cnt
                cursor[tid] = start
                for pad in T.serial(max_block_m):
                    if pad < aligned_cnt - cnt:
                        m_indices[start + cnt + pad] = tid

    return deep_gemm_contig_preprocess_prefix_counts_scan


@functools.lru_cache(maxsize=8)
@tilelang.jit(
    out_idx=[], target="musa", pass_configs=_PASS_CONFIGS, compile_flags=_COMPILE_FLAGS
)
def _prefix_counts_tree_kernel(max_experts: int, max_block_m: int):
    e = T.dynamic("e")
    m = T.dynamic("m")
    scan_size = 1 << (max_experts - 1).bit_length()
    threads = max(256, scan_size)

    @T.prim_func
    def deep_gemm_contig_preprocess_prefix_counts_tree(
        counts: T.Tensor((e,), "int32"),
        cursor: T.Tensor((e,), "int32"),
        m_indices: T.Tensor((m,), "int32"),
        num_local_experts: T.int32,
        block_m: T.int32,
    ):
        with T.Kernel(1, threads=threads) as (_bid,):
            tid = T.get_thread_binding()
            prefix = T.alloc_shared((scan_size,), "int32")
            cnt = T.alloc_var("int32")
            aligned_cnt = T.alloc_var("int32")
            start = T.alloc_var("int32")
            addend = T.alloc_var("int32")

            if tid < scan_size:
                if tid < num_local_experts:
                    cnt = counts[tid]
                    prefix[tid] = T.ceildiv(cnt, block_m) * block_m
                else:
                    prefix[tid] = T.int32(0)
            T.sync_threads()

            for offset in T.serial((scan_size.bit_length() - 1)):
                step = 1 << offset
                addend = T.int32(0)
                if tid < scan_size and tid >= step:
                    addend = prefix[tid - step]
                T.sync_threads()
                if tid < scan_size and tid >= step:
                    prefix[tid] = prefix[tid] + addend
                T.sync_threads()

            if tid < num_local_experts:
                cnt = counts[tid]
                aligned_cnt = T.ceildiv(cnt, block_m) * block_m
                start = prefix[tid] - aligned_cnt
                cursor[tid] = start
                for pad in T.serial(max_block_m):
                    if pad < aligned_cnt - cnt:
                        m_indices[start + cnt + pad] = tid

    return deep_gemm_contig_preprocess_prefix_counts_tree


def _prefix_counts_no_pad_kernel(max_experts: int):
    e = T.dynamic("e")

    @T.prim_func
    def deep_gemm_contig_preprocess_prefix_counts_no_pad(
        counts: T.Tensor((e,), "int32"),
        cursor: T.Tensor((e,), "int32"),
        num_local_experts: T.int32,
    ):
        with T.Kernel(1, threads=256) as (_bid,):
            tid = T.get_thread_binding()
            acc = T.alloc_var("int32")

            if tid == 0:
                acc = T.int32(0)
                for expert in T.serial(max_experts):
                    if expert < num_local_experts:
                        cursor[expert] = acc
                        acc = acc + counts[expert]

    return deep_gemm_contig_preprocess_prefix_counts_no_pad


@functools.lru_cache(maxsize=8)
@tilelang.jit(
    out_idx=[],
    target="musa",
    pass_configs=_PASS_CONFIGS,
    compile_flags=_COMPILE_FLAGS + ["-include", _ATOMIC_HELPER_H],
)
def _fp8_assign_compact_kernel(
    input_dtype,
    output_dtype,
    hidden_groups: int,
    groups_per_block: int,
    vec_elems: int,
    topk: int,
):
    input_numel = T.dynamic("input_numel")
    output_numel = T.dynamic("output_numel")
    scale_numel = T.dynamic("scale_numel")
    topk_ids_numel = T.dynamic("topk_ids_numel")
    src2dst_numel = T.dynamic("src2dst_numel")
    m_indices_numel = T.dynamic("m_indices_numel")
    expert_numel = T.dynamic("expert_numel")
    group_size = 128
    hidden_size = hidden_groups * group_size
    threads_per_group = group_size // vec_elems
    num_threads = threads_per_group * groups_per_block

    @T.prim_func
    def deep_gemm_contig_preprocess_fp8_assign_compact(
        hidden: T.Tensor((input_numel,), input_dtype),
        topk_ids: T.Tensor((topk_ids_numel,), "int32"),
        topk_ids_for_combine: T.Tensor((topk_ids_numel,), "int32"),
        cursor: T.Tensor((expert_numel,), "int32"),
        src2dst: T.Tensor((src2dst_numel,), "int32"),
        m_indices: T.Tensor((m_indices_numel,), "int32"),
        output_q: T.Tensor((output_numel,), output_dtype),
        output_s: T.Tensor((scale_numel,), "float32"),
        num_tokens: T.int32,
        num_local_experts: T.int32,
        eps: T.float32,
        max_8bit: T.float32,
    ):
        with T.Kernel(num_tokens, threads=num_threads) as (bt,):
            tid = T.get_thread_binding()
            subgroup = tid // threads_per_group
            lane = tid % threads_per_group
            elem_base = T.alloc_var("int32")
            hidden_group = T.alloc_var("int32")
            input_base = T.alloc_var("int64")
            local_absmax = T.alloc_local((1,), "float32")
            scale = T.alloc_local((1,), "float32")
            scale_inv = T.alloc_local((1,), "float32")
            values = T.alloc_local((vec_elems,), "float32")
            dst = T.alloc_var("int32")
            out_base = T.alloc_var("int64")
            dst_shared = T.alloc_shared((topk,), "int32")

            if tid < topk:
                slot = bt * topk + tid
                expert = topk_ids[slot]
                if expert >= 0 and expert < num_local_experts:
                    topk_ids_for_combine[slot] = expert
                    dst = T.call_extern(
                        "int32",
                        "sgl_tl_atomic_add_offset",
                        T.address_of(cursor[0]),
                        expert,
                        T.int32(1),
                    )
                    if dst < m_indices_numel:
                        src2dst[slot] = dst
                        m_indices[dst] = expert
                        dst_shared[tid] = dst
                    else:
                        topk_ids_for_combine[slot] = num_local_experts
                        src2dst[slot] = T.int32(-1)
                        dst_shared[tid] = T.int32(-1)
                else:
                    topk_ids_for_combine[slot] = num_local_experts
                    src2dst[slot] = T.int32(-1)
                    dst_shared[tid] = T.int32(-1)
            T.sync_threads()

            elem_base = lane * vec_elems
            for tile in T.serial(T.ceildiv(hidden_groups, groups_per_block)):
                hidden_group = tile * groups_per_block + subgroup
                if hidden_group < hidden_groups:
                    input_base = (
                        T.Cast("int64", bt) * hidden_size
                        + hidden_group * group_size
                        + elem_base
                    )
                    local_absmax[0] = eps
                    for i in T.vectorized(vec_elems):
                        values[i] = T.Cast("float32", hidden[input_base + i])
                        local_absmax[0] = T.max(local_absmax[0], T.abs(values[i]))

                    if threads_per_group >= 32:
                        local_absmax[0] = T.max(
                            local_absmax[0], T.shfl_xor(local_absmax[0], 16)
                        )
                    if threads_per_group >= 16:
                        local_absmax[0] = T.max(
                            local_absmax[0], T.shfl_xor(local_absmax[0], 8)
                        )
                    if threads_per_group >= 8:
                        local_absmax[0] = T.max(
                            local_absmax[0], T.shfl_xor(local_absmax[0], 4)
                        )
                    if threads_per_group >= 4:
                        local_absmax[0] = T.max(
                            local_absmax[0], T.shfl_xor(local_absmax[0], 2)
                        )
                    if threads_per_group >= 2:
                        local_absmax[0] = T.max(
                            local_absmax[0], T.shfl_xor(local_absmax[0], 1)
                        )

                    scale_inv[0] = local_absmax[0] / max_8bit
                    scale[0] = max_8bit / local_absmax[0]

                    for topk_idx in T.serial(topk):
                        dst = dst_shared[topk_idx]
                        if dst >= 0:
                            out_base = (
                                T.Cast("int64", dst) * hidden_size
                                + hidden_group * group_size
                                + elem_base
                            )
                            if vec_elems == 4:
                                T.call_extern(
                                    "handle",
                                    "sgl_tl_store_fp8e4m3x4",
                                    T.address_of(output_q[0]),
                                    out_base,
                                    T.min(
                                        T.max(values[0] * scale[0], -max_8bit), max_8bit
                                    ),
                                    T.min(
                                        T.max(values[1] * scale[0], -max_8bit), max_8bit
                                    ),
                                    T.min(
                                        T.max(values[2] * scale[0], -max_8bit), max_8bit
                                    ),
                                    T.min(
                                        T.max(values[3] * scale[0], -max_8bit), max_8bit
                                    ),
                                )
                            elif vec_elems == 8:
                                T.call_extern(
                                    "handle",
                                    "sgl_tl_store_fp8e4m3x4",
                                    T.address_of(output_q[0]),
                                    out_base,
                                    T.min(
                                        T.max(values[0] * scale[0], -max_8bit), max_8bit
                                    ),
                                    T.min(
                                        T.max(values[1] * scale[0], -max_8bit), max_8bit
                                    ),
                                    T.min(
                                        T.max(values[2] * scale[0], -max_8bit), max_8bit
                                    ),
                                    T.min(
                                        T.max(values[3] * scale[0], -max_8bit), max_8bit
                                    ),
                                )
                                T.call_extern(
                                    "handle",
                                    "sgl_tl_store_fp8e4m3x4",
                                    T.address_of(output_q[0]),
                                    out_base + 4,
                                    T.min(
                                        T.max(values[4] * scale[0], -max_8bit), max_8bit
                                    ),
                                    T.min(
                                        T.max(values[5] * scale[0], -max_8bit), max_8bit
                                    ),
                                    T.min(
                                        T.max(values[6] * scale[0], -max_8bit), max_8bit
                                    ),
                                    T.min(
                                        T.max(values[7] * scale[0], -max_8bit), max_8bit
                                    ),
                                )
                            else:
                                T.call_extern(
                                    "handle",
                                    "sgl_tl_store_fp8e4m3x4",
                                    T.address_of(output_q[0]),
                                    out_base,
                                    T.min(
                                        T.max(values[0] * scale[0], -max_8bit), max_8bit
                                    ),
                                    T.min(
                                        T.max(values[1] * scale[0], -max_8bit), max_8bit
                                    ),
                                    T.min(
                                        T.max(values[2] * scale[0], -max_8bit), max_8bit
                                    ),
                                    T.min(
                                        T.max(values[3] * scale[0], -max_8bit), max_8bit
                                    ),
                                )
                                T.call_extern(
                                    "handle",
                                    "sgl_tl_store_fp8e4m3x4",
                                    T.address_of(output_q[0]),
                                    out_base + 4,
                                    T.min(
                                        T.max(values[4] * scale[0], -max_8bit), max_8bit
                                    ),
                                    T.min(
                                        T.max(values[5] * scale[0], -max_8bit), max_8bit
                                    ),
                                    T.min(
                                        T.max(values[6] * scale[0], -max_8bit), max_8bit
                                    ),
                                    T.min(
                                        T.max(values[7] * scale[0], -max_8bit), max_8bit
                                    ),
                                )
                                T.call_extern(
                                    "handle",
                                    "sgl_tl_store_fp8e4m3x4",
                                    T.address_of(output_q[0]),
                                    out_base + 8,
                                    T.min(
                                        T.max(values[8] * scale[0], -max_8bit), max_8bit
                                    ),
                                    T.min(
                                        T.max(values[9] * scale[0], -max_8bit), max_8bit
                                    ),
                                    T.min(
                                        T.max(values[10] * scale[0], -max_8bit),
                                        max_8bit,
                                    ),
                                    T.min(
                                        T.max(values[11] * scale[0], -max_8bit),
                                        max_8bit,
                                    ),
                                )
                                T.call_extern(
                                    "handle",
                                    "sgl_tl_store_fp8e4m3x4",
                                    T.address_of(output_q[0]),
                                    out_base + 12,
                                    T.min(
                                        T.max(values[12] * scale[0], -max_8bit),
                                        max_8bit,
                                    ),
                                    T.min(
                                        T.max(values[13] * scale[0], -max_8bit),
                                        max_8bit,
                                    ),
                                    T.min(
                                        T.max(values[14] * scale[0], -max_8bit),
                                        max_8bit,
                                    ),
                                    T.min(
                                        T.max(values[15] * scale[0], -max_8bit),
                                        max_8bit,
                                    ),
                                )
                    if lane == 0:
                        for topk_idx in T.serial(topk):
                            dst = dst_shared[topk_idx]
                            if dst >= 0:
                                output_s[dst * hidden_groups + hidden_group] = (
                                    scale_inv[0]
                                )

    return deep_gemm_contig_preprocess_fp8_assign_compact


def _fp8_config(hidden_size: int) -> tuple[int, int, int] | None:
    if hidden_size <= 0 or hidden_size % 128 != 0:
        return None
    hidden_groups = hidden_size // 128
    if hidden_groups == 22:
        return hidden_groups, hidden_groups, 8
    if hidden_groups == 24:
        return hidden_groups, hidden_groups, 8
    if hidden_groups <= 32:
        return hidden_groups, hidden_groups, 4
    if hidden_groups <= 64:
        return hidden_groups, hidden_groups, 8
    if hidden_groups <= 128:
        return hidden_groups, hidden_groups, 16
    return None


def _impl_fp8(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    m_indices: torch.Tensor,
    src2dst: torch.Tensor,
    topk_ids_for_combine: torch.Tensor,
    counts: torch.Tensor,
    cursor: torch.Tensor,
    num_local_experts: int,
    block_m: int,
) -> None:
    num_slots = topk_ids.numel()
    num_tokens = hidden_states.shape[0]
    topk = topk_ids.shape[-1]
    config = _fp8_config(hidden_states.shape[1])
    if config is None:
        raise RuntimeError(
            f"unsupported TileLang DeepGEMM preprocess hidden={hidden_states.shape[1]}"
        )
    if topk > 16:
        raise RuntimeError(f"unsupported TileLang DeepGEMM fp8 preprocess topk={topk}")
    hidden_groups, groups_per_block, vec_elems = config
    num_local_experts = int(num_local_experts)
    if num_local_experts <= 0:
        raise ValueError(f"num_local_experts must be positive, got {num_local_experts}")
    clear = _clear_i32_kernel()
    fill = _fill_i32_kernel()
    single_block_count = _count_topk_single_block_kernel(num_local_experts)
    count = _count_topk_block_hist_kernel(num_local_experts)
    block_m = int(block_m)
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")
    single_block_count_prefix = _count_prefix_topk_single_block_kernel(
        num_local_experts, block_m
    )
    use_single_block_count = num_slots <= 1024
    use_single_block_count_prefix = block_m != 1 and use_single_block_count
    prefix = (
        _prefix_counts_no_pad_kernel(num_local_experts)
        if block_m == 1
        else (
            _prefix_counts_tree_kernel(num_local_experts, block_m)
            if num_tokens < 512 and num_local_experts <= 1024
            else (
                _prefix_counts_scan_kernel(num_local_experts, block_m)
                if num_local_experts <= 1024
                else _prefix_counts_kernel(num_local_experts, block_m)
            )
        )
    )
    compact = _fp8_assign_compact_kernel(
        hidden_states.dtype,
        output.dtype,
        hidden_groups,
        groups_per_block,
        vec_elems,
        topk,
    )
    if not use_single_block_count and not use_single_block_count_prefix:
        clear(counts, tilelang.cdiv(num_local_experts, 256), num_local_experts)
        count(
            topk_ids.reshape(-1),
            counts,
            tilelang.cdiv(num_slots, 1024),
            num_slots,
            num_local_experts,
        )
    if block_m == 1:
        if use_single_block_count:
            single_block_count(
                topk_ids.reshape(-1), counts, num_slots, num_local_experts
            )
        prefix(counts, cursor, num_local_experts)
    else:
        fill(
            m_indices,
            tilelang.cdiv(m_indices.numel(), 256),
            m_indices.numel(),
            num_local_experts - 1,
        )
        if use_single_block_count_prefix:
            single_block_count_prefix(
                topk_ids.reshape(-1),
                counts,
                cursor,
                m_indices,
                num_slots,
                num_local_experts,
                block_m,
            )
        else:
            prefix(counts, cursor, m_indices, num_local_experts, block_m)
    compact(
        hidden_states.reshape(-1),
        topk_ids.reshape(-1),
        topk_ids_for_combine.reshape(-1),
        cursor,
        src2dst,
        m_indices,
        output.reshape(-1),
        output_scale.reshape(-1),
        num_tokens,
        num_local_experts,
        1.0e-10,
        448.0,
    )


@register_custom_op(
    op_name="musa_deep_gemm_contig_preprocess_fp8_tilelang",
    mutates_args=[
        "output",
        "output_scale",
        "m_indices",
        "src2dst",
        "topk_ids_for_combine",
        "counts",
        "cursor",
    ],
)
def _custom_fp8(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    m_indices: torch.Tensor,
    src2dst: torch.Tensor,
    topk_ids_for_combine: torch.Tensor,
    counts: torch.Tensor,
    cursor: torch.Tensor,
    num_local_experts: int,
    block_m: int,
) -> None:
    _impl_fp8(
        hidden_states,
        topk_ids,
        output,
        output_scale,
        m_indices,
        src2dst,
        topk_ids_for_combine,
        counts,
        cursor,
        int(num_local_experts),
        int(block_m),
    )


def deep_gemm_contig_preprocess_fp8_tilelang(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    m_indices: torch.Tensor,
    src2dst: torch.Tensor,
    topk_ids_for_combine: torch.Tensor,
    counts: torch.Tensor,
    cursor: torch.Tensor,
    num_local_experts: int,
    block_m: int,
) -> None:
    _custom_fp8(
        hidden_states,
        topk_ids,
        output,
        output_scale,
        m_indices,
        src2dst,
        topk_ids_for_combine,
        counts,
        cursor,
        int(num_local_experts),
        int(block_m),
    )


def can_use_fp8_tilelang(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    num_local_experts: int,
    use_fp8_quant: bool,
) -> bool:
    return (
        use_fp8_quant
        and hidden_states.dim() == 2
        and _fp8_config(hidden_states.shape[1]) is not None
        and topk_ids.dim() == 2
        and topk_ids.shape[-1] <= 16
        and num_local_experts > 0
    )
