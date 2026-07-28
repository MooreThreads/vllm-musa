from __future__ import annotations

import functools
import os

import torch

from vllm_musa.jit_kernel.csrc.jit import load_musa_jit


def _default_threads_blocks(world_size: int) -> tuple[int, int]:
    if world_size == 8:
        return 1024, 60
    return 512, 36


def preferred_shot(world_size: int, nbytes: int) -> int:
    if world_size == 2 and nbytes >= 16 * 1024 * 1024:
        return 1
    return 2


def _compile_config(world_size: int) -> tuple[int, int, int, int, int]:
    default_threads, default_blocks = _default_threads_blocks(world_size)
    threads = int(os.environ.get("VLLM_MUSA_CUSTOM_AR_THREADS", default_threads))
    blocks = int(os.environ.get("VLLM_MUSA_CUSTOM_AR_BLOCKS", default_blocks))
    vector_load = int(os.environ.get("VLLM_MUSA_CUSTOM_AR_VECTOR_LOAD", "0"))
    atomic_barrier = int(os.environ.get("VLLM_MUSA_CUSTOM_AR_ATOMIC_BARRIER", "1"))
    max_blocks = int(
        os.environ.get("VLLM_MUSA_CUSTOM_AR_MAX_BLOCKS", str(max(120, blocks)))
    )
    max_blocks = max(max_blocks, blocks)
    return threads, blocks, vector_load, atomic_barrier, max_blocks


@functools.lru_cache(maxsize=16)
def _custom_ar_module(world_size: int):
    threads, blocks, vector_load, atomic_barrier, max_blocks = _compile_config(
        world_size
    )
    name = (
        f"vllm_musa_custom_all_reduce_t{threads}_b{blocks}"
        f"_v{vector_load}_ab{atomic_barrier}_mb{max_blocks}"
    )
    return load_musa_jit(
        name,
        ("distributed/custom_all_reduce.mu",),
        extra_musa_cflags=(
            "-Wno-error=address-of-temporary",
            "-fmusa-flush-denormals-to-zero",
            "-fno-signed-zeros",
            "-D__MUSA_ARCH_LIST__=310",
            f"-DSGL_CUSTOM_AR_THREADS={threads}",
            f"-DSGL_CUSTOM_AR_BLOCKS={blocks}",
            f"-DSGL_CUSTOM_AR_VECTOR_LOAD={vector_load}",
            f"-DSGL_CUSTOM_AR_ATOMIC_BARRIER={atomic_barrier}",
            f"-DSGL_CUSTOM_AR_MAX_BLOCKS={max_blocks}",
            "-mllvm",
            "-mtgpu-opt-level=1",
            "-mllvm",
            "-mtgpu-load-store-opt=1",
            "-mllvm",
            "-mtgpu-fold-global-ldst=1",
        ),
    )


def ensure_compiled(world_size: int) -> None:
    _custom_ar_module(int(world_size))


def meta_size(world_size: int = 8) -> int:
    return int(_custom_ar_module(int(world_size)).vllm_musa_custom_ar_meta_size())


def launch_unregistered(
    rank_data: torch.Tensor,
    signal_ptrs_cpu: torch.Tensor,
    inp: torch.Tensor,
    out: torch.Tensor,
    self_signal_ptr: int,
    self_buffer_ptr: int,
    max_size_bytes: int,
    rank: int,
    world_size: int,
    shot: int,
) -> None:
    _custom_ar_module(int(world_size)).vllm_musa_custom_ar_launch_unregistered(
        rank_data,
        signal_ptrs_cpu,
        inp,
        out,
        int(self_signal_ptr),
        int(self_buffer_ptr),
        int(max_size_bytes),
        int(rank),
        int(world_size),
        int(shot),
    )


def launch_all_gather(
    rank_data: torch.Tensor,
    signal_ptrs_cpu: torch.Tensor,
    inp: torch.Tensor,
    out: torch.Tensor,
    self_signal_ptr: int,
    self_buffer_ptr: int,
    max_size_bytes: int,
    rank: int,
    world_size: int,
) -> None:
    _custom_ar_module(int(world_size)).vllm_musa_custom_ar_launch_all_gather(
        rank_data,
        signal_ptrs_cpu,
        inp,
        out,
        int(self_signal_ptr),
        int(self_buffer_ptr),
        int(max_size_bytes),
        int(rank),
        int(world_size),
    )


def launch_all_gather_registered(
    rank_data: torch.Tensor,
    signal_ptrs_cpu: torch.Tensor,
    inp: torch.Tensor,
    out: torch.Tensor,
    rank: int,
    world_size: int,
) -> None:
    """Gather directly from persistent IPC-registered input shards."""
    _custom_ar_module(int(world_size)).vllm_musa_custom_ar_launch_all_gather_registered(
        rank_data,
        signal_ptrs_cpu,
        inp,
        out,
        int(rank),
        int(world_size),
    )


def launch_registered(
    rank_data: torch.Tensor,
    signal_ptrs_cpu: torch.Tensor,
    inp: torch.Tensor,
    out: torch.Tensor,
    rank: int,
    world_size: int,
    shot: int,
) -> None:
    _custom_ar_module(int(world_size)).vllm_musa_custom_ar_launch_registered(
        rank_data,
        signal_ptrs_cpu,
        inp,
        out,
        int(rank),
        int(world_size),
        int(shot),
    )


def launch_graph_registered(
    rank_data: torch.Tensor,
    signal_ptrs_cpu: torch.Tensor,
    inp: torch.Tensor,
    out: torch.Tensor,
    rank: int,
    world_size: int,
    shot: int,
) -> None:
    _custom_ar_module(int(world_size)).vllm_musa_custom_ar_launch_graph_registered(
        rank_data,
        signal_ptrs_cpu,
        inp,
        out,
        int(rank),
        int(world_size),
        int(shot),
    )
