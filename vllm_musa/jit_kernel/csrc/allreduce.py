from __future__ import annotations

import ctypes
import enum
import functools
import hashlib
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

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


def launch_fused_allreduce_rmsnorm_unregistered(
    rank_data: torch.Tensor,
    signal_ptrs_cpu: torch.Tensor,
    inp: torch.Tensor,
    weight: torch.Tensor,
    norm_out: torch.Tensor,
    reduced: torch.Tensor,
    self_signal_ptr: int,
    self_buffer_ptr: int,
    max_size_bytes: int,
    rank: int,
    world_size: int,
    shot: int,
    eps: float,
) -> None:
    _custom_ar_module(int(world_size)).vllm_musa_fused_ar_rmsnorm_launch_unregistered(
        rank_data,
        signal_ptrs_cpu,
        inp,
        weight,
        norm_out,
        reduced,
        int(self_signal_ptr),
        int(self_buffer_ptr),
        int(max_size_bytes),
        int(rank),
        int(world_size),
        int(shot),
        float(eps),
    )


def launch_fused_allreduce_residual_rmsnorm_unregistered(
    rank_data: torch.Tensor,
    signal_ptrs_cpu: torch.Tensor,
    inp: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    norm_out: torch.Tensor,
    residual_out: torch.Tensor,
    reduced: torch.Tensor,
    self_signal_ptr: int,
    self_buffer_ptr: int,
    max_size_bytes: int,
    rank: int,
    world_size: int,
    shot: int,
    eps: float,
) -> None:
    _custom_ar_module(
        int(world_size)
    ).vllm_musa_fused_ar_residual_rmsnorm_launch_unregistered(
        rank_data,
        signal_ptrs_cpu,
        inp,
        residual,
        weight,
        norm_out,
        residual_out,
        reduced,
        int(self_signal_ptr),
        int(self_buffer_ptr),
        int(max_size_bytes),
        int(rank),
        int(world_size),
        int(shot),
        float(eps),
    )


def launch_fused_allreduce_residual_rmsnorm_no_raw_unregistered(
    rank_data: torch.Tensor,
    signal_ptrs_cpu: torch.Tensor,
    inp: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    norm_out: torch.Tensor,
    residual_out: torch.Tensor,
    self_signal_ptr: int,
    self_buffer_ptr: int,
    max_size_bytes: int,
    rank: int,
    world_size: int,
    shot: int,
    eps: float,
) -> None:
    _custom_ar_module(
        int(world_size)
    ).vllm_musa_fused_ar_residual_rmsnorm_no_raw_launch_unregistered(
        rank_data,
        signal_ptrs_cpu,
        inp,
        residual,
        weight,
        norm_out,
        residual_out,
        int(self_signal_ptr),
        int(self_buffer_ptr),
        int(max_size_bytes),
        int(rank),
        int(world_size),
        int(shot),
        float(eps),
    )


def _check_registered_rank_data(rank_data: torch.Tensor) -> None:
    if rank_data.device.type == "cpu":
        raise ValueError(
            "registered MUSA custom-allreduce rank_data must be a device tensor"
        )


def launch_fused_allreduce_rmsnorm_registered(
    rank_data: torch.Tensor,
    signal_ptrs_cpu: torch.Tensor,
    inp: torch.Tensor,
    weight: torch.Tensor,
    norm_out: torch.Tensor,
    reduced: torch.Tensor,
    self_signal_ptr: int,
    self_buffer_ptr: int,
    max_size_bytes: int,
    rank: int,
    world_size: int,
    shot: int,
    eps: float,
) -> None:
    """Launch fused AR-RMSNorm from device-resident peer input pointers.

    The FFI entry is shared with the staging launcher. CPU rank_data selects
    the existing staging copy; MUSA rank_data selects the registered-input
    path and skips that copy.
    """
    _check_registered_rank_data(rank_data)
    launch_fused_allreduce_rmsnorm_unregistered(
        rank_data,
        signal_ptrs_cpu,
        inp,
        weight,
        norm_out,
        reduced,
        self_signal_ptr,
        self_buffer_ptr,
        max_size_bytes,
        rank,
        world_size,
        shot,
        eps,
    )


def launch_fused_allreduce_residual_rmsnorm_registered(
    rank_data: torch.Tensor,
    signal_ptrs_cpu: torch.Tensor,
    inp: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    norm_out: torch.Tensor,
    residual_out: torch.Tensor,
    reduced: torch.Tensor,
    self_signal_ptr: int,
    self_buffer_ptr: int,
    max_size_bytes: int,
    rank: int,
    world_size: int,
    shot: int,
    eps: float,
) -> None:
    _check_registered_rank_data(rank_data)
    launch_fused_allreduce_residual_rmsnorm_unregistered(
        rank_data,
        signal_ptrs_cpu,
        inp,
        residual,
        weight,
        norm_out,
        residual_out,
        reduced,
        self_signal_ptr,
        self_buffer_ptr,
        max_size_bytes,
        rank,
        world_size,
        shot,
        eps,
    )


def launch_fused_allreduce_residual_rmsnorm_no_raw_registered(
    rank_data: torch.Tensor,
    signal_ptrs_cpu: torch.Tensor,
    inp: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    norm_out: torch.Tensor,
    residual_out: torch.Tensor,
    self_signal_ptr: int,
    self_buffer_ptr: int,
    max_size_bytes: int,
    rank: int,
    world_size: int,
    shot: int,
    eps: float,
) -> None:
    _check_registered_rank_data(rank_data)
    launch_fused_allreduce_residual_rmsnorm_no_raw_unregistered(
        rank_data,
        signal_ptrs_cpu,
        inp,
        residual,
        weight,
        norm_out,
        residual_out,
        self_signal_ptr,
        self_buffer_ptr,
        max_size_bytes,
        rank,
        world_size,
        shot,
        eps,
    )


# --- v2 kernel contract (byte-identical to the upstream file) ---------
SHOT_PUSH = 0
SHOT_ONE_STAGE = 1
SHOT_TWO_STAGE = 2
SHOT_PUSH_WIDE = 3
SHOT_TWO_STAGE_512 = 4
_V2_SIGNAL_BLOCKS = 512
_V2_SEMAPHORE_BYTES = 128
# This is an ABI capability, not a runtime-performance claim.  Promote it only
# when every fused launch variant writes the pre-residual all-reduce output.
V2_FUSED_SUPPORTS_RAW = True


def allreduce_v2_meta_size() -> int:
    """Bytes of one v2 Signal region (also the push/pull controller stride)."""
    return 2 * _V2_SIGNAL_BLOCKS * _V2_SEMAPHORE_BYTES


def push_buffer_bytes(nbytes: int, world_size: int) -> int:
    """Staging bytes for the push path: two epochs times one slot per rank."""
    return int(nbytes) * 2 * int(world_size)


def _v2_default_threads_blocks(world_size: int) -> tuple[int, int]:
    # Same defaults as the v1 module so the two arms stay comparable.
    return (1024, 60) if int(world_size) == 8 else (512, 36)


def _v2_compile_config(world_size: int, shot: int) -> tuple[int, int, int, int, int, int, int]:
    default_threads, default_blocks = _v2_default_threads_blocks(world_size)
    is_push = shot in (SHOT_PUSH, SHOT_PUSH_WIDE)
    threads = int(os.environ.get("VLLM_MUSA_CUSTOM_AR_THREADS", default_threads))
    blocks = int(os.environ.get("VLLM_MUSA_CUSTOM_AR_BLOCKS", default_blocks))
    vector_load = int(os.environ.get("VLLM_MUSA_CUSTOM_AR_VECTOR_LOAD", "0"))
    atomic_barrier = int(os.environ.get("VLLM_MUSA_CUSTOM_AR_ATOMIC_BARRIER", "1"))
    max_blocks = int(
        os.environ.get("VLLM_MUSA_CUSTOM_AR_MAX_BLOCKS", str(max(120, blocks)))
    )
    max_blocks = max(max_blocks, blocks)
    # Upstream defaults: push shots use a static grid (dynamic_blocks=0) with
    # the polling wait loop, pull shots use dynamic blocks.  Only the push
    # case is compiled here because v2 pull still runs through the v1 module.
    dynamic_blocks = int(
        os.environ.get(
            "VLLM_MUSA_CUSTOM_AR_DYNAMIC_BLOCKS", "0" if is_push else "1"
        )
    )
    # `push_16b_asm` defaults to on upstream.  This port keeps it off until the
    # MUSA inline-asm store path is validated bit-exactly against the C++ store
    # path on real hardware (VLLM_MUSA_CUSTOM_AR_PUSH_16B_ASM=1 opts in).
    push_16b_asm = int(os.environ.get("VLLM_MUSA_CUSTOM_AR_PUSH_16B_ASM", "0"))
    if not is_push or push_16b_asm == 0:
        push_16b_asm = 0
    return threads, blocks, vector_load, atomic_barrier, max_blocks, dynamic_blocks, push_16b_asm


def _module_name(world_size: int, shot: int) -> str:
    threads, blocks, vector_load, atomic_barrier, max_blocks, dynamic_blocks, push_16b_asm = (
        _v2_compile_config(world_size, shot)
    )
    return (
        f"vllm_musa_custom_all_reduce_v2_s{shot}_ws{int(world_size)}"
        f"_t{threads}_b{blocks}_v{vector_load}_ab{atomic_barrier}"
        f"_mb{max_blocks}_db{dynamic_blocks}_pa{push_16b_asm}"
    )


@functools.lru_cache(maxsize=32)
def _custom_ar_v2_module(world_size: int, shot: int):
    threads, blocks, vector_load, atomic_barrier, max_blocks, dynamic_blocks, push_16b_asm = (
        _v2_compile_config(world_size, shot)
    )
    return load_musa_jit(
        _module_name(world_size, shot),
        ("distributed/custom_all_reduce_v2.mu",),
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
            f"-DSGL_CUSTOM_AR_DYNAMIC_BLOCKS={dynamic_blocks}",
            f"-DSGL_CUSTOM_AR_PUSH_POLLING={1 if int(shot) in (SHOT_PUSH, SHOT_PUSH_WIDE) else 0}",
            f"-DSGL_CUSTOM_AR_PUSH_16B_ASM={push_16b_asm}",
            "-mllvm",
            "-mtgpu-opt-level=1",
            "-mllvm",
            "-mtgpu-load-store-opt=1",
            "-mllvm",
            "-mtgpu-fold-global-ldst=1",
        ),
    )


def ensure_compiled_v2(world_size: int, shot: int = SHOT_PUSH) -> None:
    _custom_ar_v2_module(int(world_size), int(shot))


def launch_device_descriptor(
    rank_data_device: torch.Tensor,
    signal_ptrs_cpu: torch.Tensor,
    out: torch.Tensor,
    self_signal_ptr: int,
    rank: int,
    world_size: int,
    shot: int,
) -> None:
    """Launch a v2 kernel from a device-resident RankData pointer table.

    ``rank_data_device`` must be an int64 device tensor holding ``kMaxRanks``
    pointers; the kernel reads it as ``const RankData*``.  ``out`` is both
    input and output (the push path publishes from the caller's buffer).
    """
    _custom_ar_v2_module(int(world_size), int(shot)).vllm_musa_custom_ar_launch_device_descriptor(
        rank_data_device,
        signal_ptrs_cpu,
        out,
        int(self_signal_ptr),
        int(rank),
        int(world_size),
        int(shot),
    )


# --- pull launch entry points -------------------------------------------------
#
# The two ``extern "C"`` symbols below take raw pointers plus the current MUSA
# stream.  These wrappers mirror upstream's pybind11 bindings one-to-one: the
# tensor arguments are resolved to ``data_ptr()`` and ``out.numel()`` /
# ``dtype_code(out)`` are forwarded exactly as the pybind module did.

_DTYPE_CODES = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
}

_raw_libs: dict[str, ctypes.CDLL] = {}


def _raw_symbols(world_size: int, shot: int):
    """Load the compiled v2 module and return its two raw launch callables."""
    name = _module_name(int(world_size), int(shot))
    lib = _raw_libs.get(name)
    if lib is None:
        _custom_ar_v2_module(int(world_size), int(shot))  # ensures the .so exists
        cache_dir = Path(
            os.environ.get("VLLM_MUSA_JIT_CACHE_DIR")
            or (Path.home() / ".cache" / "vllm_musa_jit")
        )
        so_path = cache_dir / name / f"{name}.so"
        if not so_path.is_file():
            raise RuntimeError(
                f"MUSA custom AR v2 module was not built at {so_path}; "
                "the pull launch entry points are unavailable"
            )
        lib = ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
        unregistered = lib.vllm_musa_custom_ar_launch_unregistered_raw_nocheck
        unregistered.restype = None
        unregistered.argtypes = [
            ctypes.c_void_p,  # rank_ptrs
            ctypes.c_void_p,  # signal_ptrs
            ctypes.c_void_p,  # inp
            ctypes.c_void_p,  # out
            ctypes.c_int64,  # self_signal_ptr
            ctypes.c_int64,  # self_buffer_ptr
            ctypes.c_int64,  # push_buffer_bytes
            ctypes.c_int64,  # numel
            ctypes.c_int,  # dtype_code
            ctypes.c_int64,  # stream_value
            ctypes.c_int64,  # rank
            ctypes.c_int64,  # world_size
            ctypes.c_int64,  # shot
        ]
        registered = lib.vllm_musa_custom_ar_launch_registered_raw_nocheck
        registered.restype = None
        registered.argtypes = [
            ctypes.c_void_p,  # rank_ptrs
            ctypes.c_void_p,  # signal_ptrs
            ctypes.c_void_p,  # out
            ctypes.c_int64,  # self_signal_ptr
            ctypes.c_int64,  # numel
            ctypes.c_int,  # dtype_code
            ctypes.c_int64,  # stream_value
            ctypes.c_int64,  # rank
            ctypes.c_int64,  # world_size
            ctypes.c_int64,  # shot
        ]
        _raw_libs[name] = lib
    return (
        lib.vllm_musa_custom_ar_launch_unregistered_raw_nocheck,
        lib.vllm_musa_custom_ar_launch_registered_raw_nocheck,
    )


def _current_stream_value() -> int:
    """Raw handle of the current MUSA stream (what ``getCurrentMUSAStream`` gave)."""
    for module_name in ("musa", "cuda"):
        module = getattr(torch, module_name, None)
        current_stream = getattr(module, "current_stream", None)
        if current_stream is None:
            continue
        stream = current_stream()
        for attr in ("musa_stream", "cuda_stream"):
            value = getattr(stream, attr, None)
            if value is None:
                continue
            # 0 is the legacy/default stream handle and is a valid value here;
            # `getCurrentMUSAStream` returns exactly that outside a capture.
            return int(value)
    raise RuntimeError(
        "MUSA custom AR v2 cannot resolve the current stream handle; "
        "the pull launch path needs it"
    )


def _dtype_code(tensor: torch.Tensor) -> int:
    code = _DTYPE_CODES.get(tensor.dtype)
    if code is None:
        raise TypeError(
            "MUSA custom AR v2 only supports fp16/bf16/fp32, got "
            f"{tensor.dtype}"
        )
    return code


def launch_unregistered_func(world_size: int, shot: int):
    """Callable equivalent of upstream's ``launch_unregistered`` pybind binding."""
    symbols = _raw_symbols(int(world_size), int(shot))

    def launch(
        rank_data: torch.Tensor,
        signal_ptrs_cpu: torch.Tensor,
        input: torch.Tensor,
        out: torch.Tensor,
        self_signal_ptr: int,
        self_buffer_ptr: int,
        push_buffer_bytes: int,
        rank: int,
        world_size: int,
        shot: int,
    ) -> None:
        symbols[0](
            ctypes.c_void_p(rank_data.data_ptr()),
            ctypes.c_void_p(signal_ptrs_cpu.data_ptr()),
            ctypes.c_void_p(input.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int64(int(self_signal_ptr)),
            ctypes.c_int64(int(self_buffer_ptr)),
            ctypes.c_int64(int(push_buffer_bytes)),
            ctypes.c_int64(int(out.numel())),
            ctypes.c_int(_dtype_code(out)),
            ctypes.c_int64(_current_stream_value()),
            ctypes.c_int64(int(rank)),
            ctypes.c_int64(int(world_size)),
            ctypes.c_int64(int(shot)),
        )

    return launch


def launch_registered_func(world_size: int, shot: int):
    """Callable equivalent of upstream's ``launch_registered`` pybind binding."""
    symbols = _raw_symbols(int(world_size), int(shot))

    def launch(
        rank_data: torch.Tensor,
        signal_ptrs_cpu: torch.Tensor,
        out: torch.Tensor,
        self_signal_ptr: int,
        rank: int,
        world_size: int,
        shot: int,
    ) -> None:
        symbols[1](
            ctypes.c_void_p(rank_data.data_ptr()),
            ctypes.c_void_p(signal_ptrs_cpu.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int64(int(self_signal_ptr)),
            ctypes.c_int64(int(out.numel())),
            ctypes.c_int(_dtype_code(out)),
            ctypes.c_int64(_current_stream_value()),
            ctypes.c_int64(int(rank)),
            ctypes.c_int64(int(world_size)),
            ctypes.c_int64(int(shot)),
        )

    return launch

# Names used by the ported v2 object layer.  The object calls
# ``launch_*_pybind_func(world_size, shot)`` because upstream resolved them from
# its pybind11 module; here they resolve to the ctypes bindings above.  Aliasing
# keeps the object layer byte-identical to upstream instead of editing its launch
# call sites.
launch_unregistered_pybind_func = launch_unregistered_func
launch_registered_pybind_func = launch_registered_func


class AllReduceAlgo(enum.Enum):
    ONE_SHOT_PUSH = enum.auto()
    ONE_SHOT_PULL = enum.auto()
    TWO_SHOT_PULL = enum.auto()

    def is_push(self) -> bool:
        return self == AllReduceAlgo.ONE_SHOT_PUSH

    @property
    def shot(self) -> int:
        return 2 if self == AllReduceAlgo.TWO_SHOT_PULL else 1


# --- v2 launcher adapters used by the v2 object layer -----------------------
_PUSH_SHOTS = (SHOT_PUSH, SHOT_PUSH_WIDE)


def is_shot_forced() -> bool:
    """Whether an explicit shot override is in effect."""
    return forced_shot() is not None


def forced_shot() -> int | None:
    """Explicit algorithm override for experiments, or None for the table."""
    raw = os.environ.get("VLLM_MUSA_CUSTOM_AR_FORCE_SHOT", "")
    if not raw:
        return None
    shot = int(raw)
    if shot not in (SHOT_PUSH, SHOT_ONE_STAGE, SHOT_TWO_STAGE, SHOT_PUSH_WIDE,
                    SHOT_TWO_STAGE_512):
        raise ValueError(
            f"VLLM_MUSA_CUSTOM_AR_FORCE_SHOT={shot} is not a valid shot (0..4)"
        )
    return shot


def _bound_launch(kind, world_size, shot):
    """Wrap a concrete launch entry with a (world_size, shot) binding check."""
    world_size = int(world_size)
    shot = int(shot)
    ensure_compiled_v2(world_size, shot)

    def _check(world_size_arg, shot_arg):
        if int(world_size_arg) != world_size or int(shot_arg) != shot:
            raise RuntimeError(
                f"MUSA custom AR v2 {kind} launch is bound to "
                f"(world_size={world_size}, shot={shot}) but was called with "
                f"(world_size={world_size_arg}, shot={shot_arg})"
            )

    return world_size, shot, _check


def launch_unregistered_pybind_func(world_size: int, shot: int):
    world_size, shot, _check = _bound_launch("unregistered", world_size, shot)
    launch = launch_unregistered_func(world_size, shot)

    def _launch(rank_data, signal_ptrs_cpu, input, out, self_signal_ptr, 
                self_buffer_ptr, push_buffer_bytes, rank, world_size_arg,
                shot_arg) -> None:
        _check(world_size_arg, shot_arg)
        launch(rank_data, signal_ptrs_cpu, input, out, int(self_signal_ptr),
               int(self_buffer_ptr), int(push_buffer_bytes), int(rank),
               world_size, shot)

    return _launch


def launch_registered_pybind_func(world_size: int, shot: int):
    world_size, shot, _check = _bound_launch("registered", world_size, shot)
    launch = launch_registered_func(world_size, shot)

    def _launch(rank_data, signal_ptrs_cpu, out, self_signal_ptr, rank,
                world_size_arg, shot_arg) -> None:
        _check(world_size_arg, shot_arg)
        launch(rank_data, signal_ptrs_cpu, out, int(self_signal_ptr), int(rank),
               world_size, shot)

    return _launch


def launch_device_descriptor_func(world_size: int, shot: int):
    """Device-descriptor entry: the rank table stays on the device side."""
    shot = int(shot)
    if shot not in _PUSH_SHOTS:
        raise NotImplementedError(
            "MUSA custom AR v2 device-descriptor launch covers the push shots "
            f"only ({SHOT_PUSH}, {SHOT_PUSH_WIDE}), got shot={shot}"
        )
    world_size, shot, _check = _bound_launch("device-descriptor", world_size, shot)

    def _launch(rank_data_device, signal_ptrs_cpu, out, self_signal_ptr, rank,
                world_size_arg, shot_arg) -> None:
        _check(world_size_arg, shot_arg)
        launch_device_descriptor(rank_data_device, signal_ptrs_cpu, out,
                                int(self_signal_ptr), int(rank), world_size, shot)

    return _launch


# The fused source snapshot is documented next to its TU in custom_all_reduce_v2.mu.
# It has a distinct signal allocation and is compiled independently from pure CAR.
def _v2_fused_base_config(world_size: int, shot: int, mp_count: int) -> dict[str, int]:
    """MR577 pull defaults, isolated from the existing pure/v1 defaults."""
    from vllm_musa.optimization_contract.Fuse_CAR_RMSnorm import resolve_architecture_launch

    if shot == SHOT_TWO_STAGE_512:
        default_threads, default_blocks = 512, 120 if world_size in (4, 8) else 48
    elif world_size == 2:
        default_threads, default_blocks = (1024 if shot == SHOT_TWO_STAGE else 512), 56
    elif world_size == 4:
        default_threads, default_blocks = 1024, 40 if shot == SHOT_TWO_STAGE else 60
    elif world_size == 8:
        default_threads, default_blocks = 768, 80
    else:
        default_threads, default_blocks = 512, 36

    def env(field: str, default: int) -> int:
        return int(os.environ.get(
            f"VLLM_MUSA_CUSTOM_AR_{shot}SHOT_{field}",
            os.environ.get(f"VLLM_MUSA_CUSTOM_AR_{field}", default),
        ))

    threads, blocks = env("THREADS", default_threads), env("BLOCKS", default_blocks)
    max_blocks = env("MAX_BLOCKS", max(120, blocks))
    block_override = any(name in os.environ for name in (
        f"VLLM_MUSA_CUSTOM_AR_{shot}SHOT_BLOCKS", "VLLM_MUSA_CUSTOM_AR_BLOCKS"
    ))
    dynamic = int(not (
        world_size == 2 or (world_size == 4 and shot != SHOT_ONE_STAGE) or block_override
    ))
    return dict(
        threads=threads, blocks=blocks, max_blocks=max_blocks,
        launch_max_blocks=env("LAUNCH_MAX_BLOCKS", resolve_architecture_launch(
            mp_count=mp_count, world_size=world_size, shot=shot, max_blocks=max_blocks
        )),
        dynamic_blocks=env("DYNAMIC_BLOCKS", dynamic),
        atomic_barrier=env("ATOMIC_BARRIER", int(shot != SHOT_ONE_STAGE)),
        vector_load=int(os.environ.get("VLLM_MUSA_CUSTOM_AR_VECTOR_LOAD", "0")),
    )


def v2_fused_compile_config(
    world_size: int,
    hidden: int,
    rows: int,
    *,
    mp_count: int | None = None,
    short_rows: bool = False,
    small_rows: bool = False,
    cache_policy: int = 0,
    token_2stage_override: int = -1,
    row_warp_inv_rms_override: int = -1,
    graph_rank_data: bool = False,
) -> Any:
    """Resolve the actual fused launch flags before forming the JIT cache key."""
    from vllm_musa.optimization_contract.Fuse_CAR_RMSnorm import (
        resolve_fused_rmsnorm_compile_config,
        resolve_token_2stage,
    )

    if mp_count is None:
        device_module = getattr(torch, "musa", None)
        if device_module is None:
            raise RuntimeError("MUSA fused CAR v2 needs live device properties or mp_count")
        mp_count = int(
            device_module.get_device_properties(device_module.current_device()).multi_processor_count
        )
    token_2stage = resolve_token_2stage(world_size, hidden, token_2stage_override, rows)
    shot = forced_shot()
    if shot not in (SHOT_ONE_STAGE, SHOT_TWO_STAGE, SHOT_TWO_STAGE_512):
        shot = SHOT_TWO_STAGE if token_2stage else SHOT_ONE_STAGE
    base = _v2_fused_base_config(world_size, shot, mp_count)
    # This revision of the pure policy fixes graph_rank_data=True internally.
    # Graph-vs-staging affects the FFI entry and module key, not its rule matches.
    del graph_rank_data
    return resolve_fused_rmsnorm_compile_config(
        world_size=int(world_size),
        mp_count=int(mp_count),
        row_hidden=int(hidden),
        row_count=int(rows),
        short_rows=bool(short_rows),
        small_rows=bool(small_rows),
        cache_policy=int(cache_policy),
        token_2stage=int(token_2stage),
        row_warp_inv_rms_override=int(row_warp_inv_rms_override),
        fast_publish_override=int(os.environ.get("VLLM_MUSA_CUSTOM_AR_FUSED_RMSNORM_FAST_PUBLISH", "-1")),
        base_threads=base["threads"],
        base_blocks=base["blocks"],
        base_max_blocks=base["max_blocks"],
        base_launch_max_blocks=base["launch_max_blocks"],
        base_dynamic_blocks=base["dynamic_blocks"],
    )


def _v2_fused_cflags(
    config: Any, cache_policy: int, vector_load: int, atomic_barrier: int,
    need_raw: bool, has_residual: bool, gemma: bool = False,
) -> tuple[str, ...]:
    """Merge generic launch fields with every policy-controlled fused switch."""
    return (
        "-Wno-error=address-of-temporary",
        "-fmusa-flush-denormals-to-zero",
        "-fno-signed-zeros",
        "-D__MUSA_ARCH_LIST__=310",
        "-DSGL_CUSTOM_AR_FUSED_RMSNORM=1",
        f"-DVLLM_MUSA_FUSED_WRITE_RAW={int(need_raw)}",
        f"-DVLLM_MUSA_FUSED_HAS_RESIDUAL={int(has_residual)}",
        f"-DVLLM_MUSA_FUSED_GEMMA={int(gemma)}",
        f"-DSGL_CUSTOM_AR_THREADS={config.threads}",
        f"-DSGL_CUSTOM_AR_BLOCKS={config.blocks}",
        f"-DSGL_CUSTOM_AR_VECTOR_LOAD={vector_load}",
        f"-DSGL_CUSTOM_AR_ATOMIC_BARRIER={atomic_barrier}",
        f"-DSGL_CUSTOM_AR_MAX_BLOCKS={config.max_blocks}",
        f"-DSGL_CUSTOM_AR_DYNAMIC_BLOCKS={config.dynamic_blocks}",
        f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_CACHE_POLICY={int(cache_policy)}",
        *config.cflags(),
        "-mllvm", "-mtgpu-opt-level=1",
        "-mllvm", "-mtgpu-load-store-opt=1",
        "-mllvm", "-mtgpu-fold-global-ldst=1",
    )


@functools.lru_cache(maxsize=96)
def _custom_ar_v2_fused_module(
    world_size: int, config: Any, cflags: tuple[str, ...], graph_rank_data: bool
) -> Any:
    # The complete effective flags are cached, rather than only the shape: env
    # overrides and policy changes must never silently reuse a different module.
    digest = hashlib.sha256(repr((config, cflags)).encode()).hexdigest()[:20]
    name = (
        f"vllm_musa_car_v2_fused_ws{world_size}_t{config.threads}_b{config.blocks}"
        f"_g{int(graph_rank_data)}_{digest}"
    )
    return load_musa_jit(
        name, ("distributed/custom_all_reduce_v2.mu",), extra_musa_cflags=cflags
    )


def _v2_fused_factory(
    world_size: int,
    hidden: int,
    short_rows: bool,
    small_rows: bool,
    cache_policy: int,
    token_2stage_override: int,
    row_warp_inv_rms_override: int,
    row_count: int,
    graph_rank_data: bool,
    row_kernel: bool,
    *,
    mp_count: int | None,
    need_raw: bool,
    has_residual: bool,
    gemma: bool = False,
) -> Callable[..., None]:
    config = v2_fused_compile_config(
        world_size, hidden, row_count, mp_count=mp_count,
        short_rows=short_rows, small_rows=small_rows, cache_policy=cache_policy,
        token_2stage_override=token_2stage_override,
        row_warp_inv_rms_override=row_warp_inv_rms_override,
        graph_rank_data=graph_rank_data,
    )
    if need_raw and config.tmp_with_residual:
        # vLLM exposes raw AR to other graph users. Pre-adding the residual
        # into shared scratch destroys that value; retain the raw scratch
        # variant instead of trying to recover it by low-precision subtraction.
        config = replace(config, tmp_with_residual=0)
    if has_residual:
        # vLLM fused_add_rms_norm uses the fp32 residual sum for variance and
        # normalization, while residual_out alone rounds to T. MR577's typed
        # cache and pre-added scratch would lose that fp32 value.
        config = replace(
            config, tmp_with_residual=0,
            cache_hidden_limit=max(int(hidden), config.cache_hidden_limit),
            typed_cache_hidden_limit=0,
        )
    shot = forced_shot()
    if shot not in (SHOT_ONE_STAGE, SHOT_TWO_STAGE, SHOT_TWO_STAGE_512):
        shot = SHOT_TWO_STAGE if config.token_2stage else SHOT_ONE_STAGE
    vector_load = int(os.environ.get("VLLM_MUSA_CUSTOM_AR_VECTOR_LOAD", "0"))
    atomic_barrier = int(os.environ.get(
        f"VLLM_MUSA_CUSTOM_AR_{shot}SHOT_ATOMIC_BARRIER",
        os.environ.get("VLLM_MUSA_CUSTOM_AR_ATOMIC_BARRIER", int(shot != SHOT_ONE_STAGE)),
    ))
    flags = _v2_fused_cflags(
        config, cache_policy, vector_load, atomic_barrier, need_raw, has_residual, gemma
    )
    module = _custom_ar_v2_fused_module(int(world_size), config, flags, graph_rank_data)
    suffix = "row_" if row_kernel else ""
    suffix += "registered" if graph_rank_data else "unregistered"
    symbol = f"vllm_musa_custom_ar_v2_fused_allreduce_rmsnorm_{suffix}"
    entry = getattr(module, symbol)

    def launch_entry(*args: Any) -> None:
        entry(*args)

    # Expose the exact compiled contract to decision tracing.  Device trace is
    # still required to prove which specialized kernel the FFI entry launches.
    setattr(launch_entry, "launch_config", config)
    setattr(launch_entry, "kernel_symbol", symbol)
    setattr(launch_entry, "jit_cflags", flags)
    setattr(launch_entry, "gemma", bool(gemma))
    return launch_entry


def v2_launch_fused_allreduce_rmsnorm_unregistered_func(
    world_size: int, hidden: int = 0, short_rows: bool = False,
    small_rows: bool = False, cache_policy: int = 0,
    row_warp_inv_rms_override: int = -1, row_count: int = 0,
    token_2stage_override: int = -1, graph_rank_data: bool = False, *,
    mp_count: int | None = None, need_raw: bool = False, has_residual: bool = True,
    gemma: bool = False,
) -> Callable[..., None]:
    if graph_rank_data:
        raise ValueError("unregistered fused launcher requires graph_rank_data=False")
    return _v2_fused_factory(
        world_size, hidden, short_rows, small_rows, cache_policy,
        token_2stage_override, row_warp_inv_rms_override, row_count, False, False,
        mp_count=mp_count, need_raw=need_raw, has_residual=has_residual, gemma=gemma,
    )


def v2_launch_fused_allreduce_rmsnorm_registered_func(
    world_size: int, hidden: int = 0, short_rows: bool = False,
    small_rows: bool = False, cache_policy: int = 0,
    row_warp_inv_rms_override: int = -1, row_count: int = 0,
    token_2stage_override: int = -1, graph_rank_data: bool = True, *,
    mp_count: int | None = None, need_raw: bool = False, has_residual: bool = True,
    gemma: bool = False,
) -> Callable[..., None]:
    if not graph_rank_data:
        raise ValueError("registered fused launcher requires graph_rank_data=True")
    return _v2_fused_factory(
        world_size, hidden, short_rows, small_rows, cache_policy,
        token_2stage_override, row_warp_inv_rms_override, row_count, True, False,
        mp_count=mp_count, need_raw=need_raw, has_residual=has_residual, gemma=gemma,
    )


def v2_launch_fused_allreduce_rmsnorm_row_unregistered_func(
    world_size: int, hidden: int = 0, short_rows: bool = False,
    small_rows: bool = False, token_2stage_override: int = 1,
    row_warp_inv_rms_override: int = -1, row_count: int = 0,
    graph_rank_data: bool = False, *, mp_count: int | None = None,
    need_raw: bool = False, has_residual: bool = True,
    gemma: bool = False,
) -> Callable[..., None]:
    if graph_rank_data:
        raise ValueError("unregistered fused launcher requires graph_rank_data=False")
    return _v2_fused_factory(
        world_size, hidden, short_rows, small_rows, 0,
        token_2stage_override, row_warp_inv_rms_override, row_count, False, True,
        mp_count=mp_count, need_raw=need_raw, has_residual=has_residual, gemma=gemma,
    )


def v2_launch_fused_allreduce_rmsnorm_row_registered_func(
    world_size: int, hidden: int = 0, short_rows: bool = False,
    small_rows: bool = False, token_2stage_override: int = 1,
    row_warp_inv_rms_override: int = -1, row_count: int = 0,
    graph_rank_data: bool = True, *, mp_count: int | None = None,
    need_raw: bool = False, has_residual: bool = True,
    gemma: bool = False,
) -> Callable[..., None]:
    if not graph_rank_data:
        raise ValueError("registered fused launcher requires graph_rank_data=True")
    return _v2_fused_factory(
        world_size, hidden, short_rows, small_rows, 0,
        token_2stage_override, row_warp_inv_rms_override, row_count, True, True,
        mp_count=mp_count, need_raw=need_raw, has_residual=has_residual, gemma=gemma,
    )


def v2_fused_allreduce_rmsnorm_func(
    world_size: int,
    hidden: int,
    rows: int,
    *,
    row_kernel: bool = False,
    graph_rank_data: bool = False,
    short_rows: bool = False,
    small_rows: bool = False,
    cache_policy: int = 0,
    token_2stage_override: int = -1,
    row_warp_inv_rms_override: int = -1,
    mp_count: int | None = None,
    need_raw: bool = False,
    has_residual: bool = True,
    gemma: bool = False,
) -> Callable[..., None]:
    """Return one fixed launcher ABI for generic/row and staging/graph paths.

    The returned callable takes ``(rank_data_device, signal_ptrs_cpu, input,
    residual, residual_out, norm_out, weight, raw_out, self_signal_ptr,
    self_buffer_ptr, max_size_bytes, rank, world_size_arg, eps)``.  Both pointer
    tables are owned by the communicator; this adapter never copies or retains
    input tensors.  Graph callers use their registered device pointer table.
    """
    entry = _v2_fused_factory(
        world_size=world_size, hidden=hidden, short_rows=short_rows,
        small_rows=small_rows, cache_policy=cache_policy,
        token_2stage_override=token_2stage_override,
        row_warp_inv_rms_override=row_warp_inv_rms_override,
        row_count=rows, graph_rank_data=graph_rank_data, row_kernel=row_kernel,
        mp_count=mp_count, need_raw=need_raw, has_residual=has_residual, gemma=gemma,
    )

    def launch(
        rank_data_device: torch.Tensor,
        signal_ptrs_cpu: torch.Tensor,
        input: torch.Tensor,
        residual: torch.Tensor | None,
        residual_out: torch.Tensor,
        norm_out: torch.Tensor,
        weight: torch.Tensor,
        raw_out: torch.Tensor | None,
        self_signal_ptr: int,
        self_buffer_ptr: int,
        max_size_bytes: int,
        rank: int,
        world_size_arg: int,
        eps: float,
    ) -> None:
        if int(world_size_arg) != int(world_size):
            raise ValueError("fused CAR v2 launcher world size does not match its JIT module")
        if (residual is not None) != has_residual or (raw_out is not None) != need_raw:
            raise ValueError("fused CAR v2 outputs do not match the compiled residual/raw contract")
        # Disabled compile branches never read these placeholders. Keeping the
        # FFI tensor ABI fixed avoids optional-pointer/capture special cases.
        residual_arg = residual if has_residual else input
        raw_arg = raw_out if need_raw else norm_out
        prefix = (rank_data_device, signal_ptrs_cpu)
        if graph_rank_data:
            args = prefix + (residual_arg, residual_out, norm_out, raw_arg, weight, int(self_signal_ptr),
                             int(rank), int(world_size))
        elif row_kernel:
            args = prefix + (input, residual_arg, residual_out, norm_out, raw_arg, weight,
                             int(self_signal_ptr), int(self_buffer_ptr), int(max_size_bytes),
                             int(rank), int(world_size))
        else:
            args = prefix + (input, norm_out, residual_arg, residual_out, raw_arg, weight,
                             int(self_signal_ptr), int(self_buffer_ptr), int(max_size_bytes),
                             int(rank), int(world_size))
        if row_kernel:
            args += (int(hidden),)
        entry(*args, float(eps))

    setattr(launch, "launch_config", entry.launch_config)
    setattr(launch, "kernel_symbol", entry.kernel_symbol)
    setattr(launch, "jit_cflags", entry.jit_cflags)
    setattr(launch, "gemma", entry.gemma)
    return launch
