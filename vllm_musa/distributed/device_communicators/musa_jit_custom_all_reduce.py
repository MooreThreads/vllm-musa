from __future__ import annotations

import ctypes
import os
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, ClassVar, NamedTuple, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_musa.distributed.device_communicators import car_decision_trace as car_trace
from vllm_musa.jit_kernel.csrc import allreduce as jit_ar
from vllm_musa.jit_kernel.csrc.allreduce import AllReduceAlgo
from vllm_musa.optimization_contract import (
    ModelFamily,
    OptimizationFeature,
    resolve_optimization_contract,
)
from vllm_musa.optimization_contract.policy import (
    DeepSeekV4MtpCarGraphStagingPlan,
    deepseek_v4_mtp_car_graph_guard_enabled,
    deepseek_v4_mtp_car_graph_staging_plan,
    deepseek_v4_mtp_graph_registered_inputs_enabled,
)

logger = init_logger(__name__)
_INT32_MAX = (1 << 31) - 1


_MAX_GRAPH_RANK_DATA_BYTES = 8 * 1024 * 1024
_MAX_RANKS = 8
_MU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11
_GRAPH_STAGING_ALIGNMENT = 256

# Optional v2 (upstream SGLang) all-reduce implementation.  The switch is
# explicit and read once at construction: ``=1`` builds the v2 communicator,
# ``=0`` keeps v1.  Only when the variable is unset does the optimization
# contract contribute a default, so a policy change never shadows an explicit
# choice made by a run.
_CUSTOM_AR_V2_ENV = "VLLM_MUSA_CUSTOM_AR_V2"
_CUSTOM_AR_V2_ON = ("1", "true", "yes", "on")
_CUSTOM_AR_V2_OFF = ("0", "false", "no", "off")


def _contract_prefers_v2() -> bool:
    """Default implementation choice, asked of the optimization contract."""
    try:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
    except (ImportError, RuntimeError):
        return False
    if vllm_config is None:
        return False
    contract = resolve_optimization_contract(vllm_config)
    return contract.prefers(OptimizationFeature.MUSA_CAR_V2_ALL_REDUCE)


def _custom_ar_v2_enabled() -> bool:
    """Decide the all-reduce implementation once, at construction time.

    ``VLLM_MUSA_CUSTOM_AR_V2=1`` -> v2, ``=0`` -> v1.  Unset -> the contract
    default, which is v1 because v2 measured no throughput gain on S5000.
    """
    raw = os.environ.get(_CUSTOM_AR_V2_ENV, "").strip().lower()
    if raw in _CUSTOM_AR_V2_ON:
        return True
    if raw in _CUSTOM_AR_V2_OFF:
        return False
    return _contract_prefers_v2()


cudaError_t = ctypes.c_int


def _max_cudagraph_capture_size(vllm_config: Any) -> int | None:
    compilation_config = getattr(vllm_config, "compilation_config", None)
    capture_sizes = getattr(compilation_config, "cudagraph_capture_sizes", None)
    if not capture_sizes:
        return None
    try:
        return max(int(size) for size in capture_sizes)
    except (TypeError, ValueError):
        return None


def _use_graph_registered_inputs_for_current_model() -> bool:
    """Use direct graph inputs only when their capture residency is bounded."""
    try:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
    except (ImportError, RuntimeError):
        return True

    contract = resolve_optimization_contract(vllm_config)
    feature = OptimizationFeature.DEEPSEEK_V4_CAR_GRAPH_INPUT_CAPTURE_GUARD
    if contract.model.family is not ModelFamily.DEEPSEEK_V4:
        return True
    if not contract.supports(feature):
        return False

    compilation_config = getattr(vllm_config, "compilation_config", None)
    capture_sizes = getattr(compilation_config, "cudagraph_capture_sizes", None)
    if not capture_sizes:
        return False
    max_capture_size = _max_cudagraph_capture_size(vllm_config)
    if max_capture_size is None:
        return False
    if deepseek_v4_mtp_graph_registered_inputs_enabled(vllm_config):
        return 0 < max_capture_size <= 5
    return max_capture_size <= 1


def _graph_staging_plan_for_current_model() -> DeepSeekV4MtpCarGraphStagingPlan | None:
    try:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
    except (ImportError, RuntimeError):
        return None
    return deepseek_v4_mtp_car_graph_staging_plan(vllm_config)


def _dsv4_mtp_graph_guard_for_current_model() -> bool:
    try:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
    except (ImportError, RuntimeError):
        return False
    return deepseek_v4_mtp_car_graph_guard_enabled(vllm_config)


class cudaIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


@dataclass(frozen=True)
class _Function:
    name: str
    restype: Any
    argtypes: list[Any]


class _MusaRTLibrary:
    _exported_functions = (
        _Function(
            "cudaMalloc",
            cudaError_t,
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
        ),
        _Function("cudaFree", cudaError_t, [ctypes.c_void_p]),
        _Function(
            "cudaMemset", cudaError_t, [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        ),
        _Function("cudaGetErrorString", ctypes.c_char_p, [cudaError_t]),
        _Function(
            "cudaIpcGetMemHandle",
            cudaError_t,
            [ctypes.POINTER(cudaIpcMemHandle_t), ctypes.c_void_p],
        ),
        _Function(
            "cudaIpcOpenMemHandle",
            cudaError_t,
            [ctypes.POINTER(ctypes.c_void_p), cudaIpcMemHandle_t, ctypes.c_uint],
        ),
        _Function("cudaIpcCloseMemHandle", cudaError_t, [ctypes.c_void_p]),
    )
    _library_cache: dict[str, Any] = {}
    _func_cache: dict[str, dict[str, Any]] = {}

    def __init__(self, so_file: str = "libmusart.so") -> None:
        if so_file not in self._library_cache:
            self._library_cache[so_file] = ctypes.CDLL(so_file)
        self.lib = self._library_cache[so_file]

        if so_file not in self._func_cache:
            funcs: dict[str, Any] = {}
            for func in self._exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                funcs[func.name] = f
            self._func_cache[so_file] = funcs
        self.funcs = self._func_cache[so_file]

    def check(self, result: cudaError_t) -> None:
        if result != 0:
            error = self.funcs["cudaGetErrorString"](result)
            error_str = error.decode("utf-8") if error else str(result)
            raise RuntimeError(f"MUSART error: {error_str}")

    def malloc(self, size: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self.check(self.funcs["cudaMalloc"](ctypes.byref(ptr), size))
        return ptr

    def free(self, ptr: ctypes.c_void_p) -> None:
        self.check(self.funcs["cudaFree"](ptr))

    def memset(self, ptr: ctypes.c_void_p, value: int, count: int) -> None:
        self.check(self.funcs["cudaMemset"](ptr, value, count))

    def ipc_get_mem_handle(self, ptr: ctypes.c_void_p) -> cudaIpcMemHandle_t:
        handle = cudaIpcMemHandle_t()
        self.check(self.funcs["cudaIpcGetMemHandle"](ctypes.byref(handle), ptr))
        return handle

    def ipc_open_mem_handle(self, handle: cudaIpcMemHandle_t) -> ctypes.c_void_p:
        cuda_ipc_mem_lazy_enable_peer_access = 1
        ptr = ctypes.c_void_p()
        self.check(
            self.funcs["cudaIpcOpenMemHandle"](
                ctypes.byref(ptr), handle, cuda_ipc_mem_lazy_enable_peer_access
            )
        )
        return ptr

    def ipc_close_mem_handle(self, ptr: ctypes.c_void_p) -> None:
        self.check(self.funcs["cudaIpcCloseMemHandle"](ptr))


class _MusaDriverLibrary:
    _exported_functions = (
        _Function(
            "muPointerGetAttribute",
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64],
        ),
        _Function(
            "muGetErrorString",
            ctypes.c_int,
            [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
        ),
    )
    _library_cache: dict[str, Any] = {}
    _func_cache: dict[str, dict[str, Any]] = {}

    def __init__(self, so_file: str = "libmusa.so") -> None:
        if so_file not in self._library_cache:
            self._library_cache[so_file] = ctypes.CDLL(so_file)
        self.lib = self._library_cache[so_file]

        if so_file not in self._func_cache:
            funcs: dict[str, Any] = {}
            for func in self._exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                funcs[func.name] = f
            self._func_cache[so_file] = funcs
        self.funcs = self._func_cache[so_file]

    def check(self, result: int) -> None:
        if result == 0:
            return
        error = ctypes.c_char_p()
        self.funcs["muGetErrorString"](result, ctypes.byref(error))
        message = error.value.decode() if error.value is not None else "unknown"
        raise RuntimeError(f"MUSA driver error {result}: {message}")

    def allocation_base(self, pointer: int) -> int:
        base = ctypes.c_uint64()
        self.check(
            self.funcs["muPointerGetAttribute"](
                ctypes.byref(base),
                _MU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                ctypes.c_uint64(pointer),
            )
        )
        return int(base.value)


_COMM_REGISTRY: dict[int, Any] = {}
_NEXT_COMM_ID = 0


def _register_comm(comm: Any) -> int:
    global _NEXT_COMM_ID
    comm_id = _NEXT_COMM_ID
    _NEXT_COMM_ID += 1
    _COMM_REGISTRY[comm_id] = comm
    return comm_id


def get_musa_jit_custom_allreduce_comm(comm_id: int) -> Any:
    """Return the live JIT communicator referenced by compiled custom ops."""
    comm = _COMM_REGISTRY.get(int(comm_id))
    if comm is None:
        raise RuntimeError(f"MUSA JIT custom all-reduce communicator {comm_id} is gone")
    return comm


def _musa_jit_custom_all_reduce(input: torch.Tensor, comm_id: int) -> torch.Tensor:
    comm = get_musa_jit_custom_allreduce_comm(comm_id)
    return comm._custom_all_reduce_impl(input)


def _musa_jit_custom_all_reduce_fake(input: torch.Tensor, comm_id: int) -> torch.Tensor:
    return torch.empty_like(input)


direct_register_custom_op(
    op_name="musa_jit_custom_all_reduce",
    op_func=_musa_jit_custom_all_reduce,
    fake_impl=_musa_jit_custom_all_reduce_fake,
)


def _musa_jit_fused_allreduce_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    comm_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    comm = get_musa_jit_custom_allreduce_comm(comm_id)
    return comm._fused_allreduce_rmsnorm_impl(input, weight, float(eps))


def _musa_jit_fused_allreduce_rmsnorm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    comm_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _ = weight, eps, comm_id
    return torch.empty_like(input), torch.empty_like(input)


direct_register_custom_op(
    op_name="musa_jit_fused_allreduce_rmsnorm",
    op_func=_musa_jit_fused_allreduce_rmsnorm,
    fake_impl=_musa_jit_fused_allreduce_rmsnorm_fake,
)


def _musa_jit_fused_allreduce_residual_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    comm_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    comm = get_musa_jit_custom_allreduce_comm(comm_id)
    return comm._fused_allreduce_residual_rmsnorm_impl(
        input, residual, weight, float(eps)
    )


def _musa_jit_fused_allreduce_residual_rmsnorm_no_raw(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    comm_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    comm = get_musa_jit_custom_allreduce_comm(comm_id)
    return comm._fused_allreduce_residual_rmsnorm_no_raw_impl(
        input, residual, weight, float(eps)
    )


def _musa_jit_fused_allreduce_residual_rmsnorm_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    comm_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _ = residual, weight, eps, comm_id
    return torch.empty_like(input), torch.empty_like(input), torch.empty_like(input)


def _musa_jit_fused_allreduce_residual_rmsnorm_no_raw_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    comm_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _ = residual, weight, eps, comm_id
    return torch.empty_like(input), torch.empty_like(input)


direct_register_custom_op(
    op_name="musa_jit_fused_allreduce_residual_rmsnorm",
    op_func=_musa_jit_fused_allreduce_residual_rmsnorm,
    fake_impl=_musa_jit_fused_allreduce_residual_rmsnorm_fake,
)


direct_register_custom_op(
    op_name="musa_jit_fused_allreduce_residual_rmsnorm_no_raw",
    op_func=_musa_jit_fused_allreduce_residual_rmsnorm_no_raw,
    fake_impl=_musa_jit_fused_allreduce_residual_rmsnorm_no_raw_fake,
)


def _ipc_handle_to_bytes(handle: cudaIpcMemHandle_t) -> bytes:
    return ctypes.string_at(ctypes.byref(handle), ctypes.sizeof(handle))


def _ipc_handle_from_bytes(data: bytes) -> cudaIpcMemHandle_t:
    expected_size = ctypes.sizeof(cudaIpcMemHandle_t)
    if len(data) != expected_size:
        raise RuntimeError(
            f"Invalid MUSA IPC handle size: got {len(data)}, expected {expected_size}"
        )
    return cudaIpcMemHandle_t.from_buffer_copy(data)


@dataclass
class _SharedBuffer:
    pointers: list[int]
    opened_ipc_ptrs: list[int]


def _make_shared_buffer(
    size_in_bytes: int, group: ProcessGroup | None = None
) -> _SharedBuffer:
    lib = _MusaRTLibrary()
    pointer = lib.malloc(size_in_bytes)
    opened_ipc_ptrs: list[int] = []
    try:
        lib.memset(pointer, 0, size_in_bytes)
        handle = _ipc_handle_to_bytes(lib.ipc_get_mem_handle(pointer))

        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)

        pointers: list[int] = []
        for i, handle_i in enumerate(handles):
            if i == rank:
                pointers.append(int(pointer.value))
            else:
                assert isinstance(handle_i, bytes)
                opened = int(
                    lib.ipc_open_mem_handle(_ipc_handle_from_bytes(handle_i)).value
                )
                pointers.append(opened)
                opened_ipc_ptrs.append(opened)
        return _SharedBuffer(pointers, opened_ipc_ptrs)
    except Exception:
        for ptr in opened_ipc_ptrs:
            try:
                lib.ipc_close_mem_handle(ctypes.c_void_p(ptr))
            except Exception:
                pass
        try:
            lib.free(pointer)
        except Exception:
            pass
        raise


def _close_ipc_handles(pointers: list[int]) -> None:
    lib = _MusaRTLibrary()
    for ptr in pointers:
        try:
            lib.ipc_close_mem_handle(ctypes.c_void_p(ptr))
        except Exception:
            logger.debug("Failed to close MUSA IPC handle.", exc_info=True)


def _free_own_shared_buffer(
    pointers: list[int], group: ProcessGroup | None = None, rank: int | None = None
) -> None:
    if rank is None:
        rank = dist.get_rank(group=group)
    _MusaRTLibrary().free(ctypes.c_void_p(pointers[rank]))


class _MusaJitCustomAllreduceImpl:
    _SUPPORTED_WORLD_SIZES = (2, 4, 6, 8)
    _MAX_CAR_SIZE = 512 * 1024 * 1024
    _GRAPH_REGISTERED_INPUT_MAX_BYTES = 512 * 1024

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size: int = _MAX_CAR_SIZE,
    ) -> None:
        self.disabled = True
        self.group = group
        self.max_size = max_size
        self._IS_CAPTURING = False
        self._use_graph_registered_inputs = (
            _use_graph_registered_inputs_for_current_model()
        )
        self._dsv4_mtp_graph_guard = _dsv4_mtp_graph_guard_for_current_model()
        self._graph_staging_plan = _graph_staging_plan_for_current_model()
        if self._graph_staging_plan is not None:
            # Every eligible DSV4 MTP4 eager/graph CAR tensor is bounded by the
            # contract below. Larger tensors already fail should_custom_* and
            # use the standard collective, so allocating the global 512-MiB
            # staging pair only wastes this memory-constrained model's budget.
            self.max_size = min(
                self.max_size,
                self._graph_staging_plan.communicator_buffer_bytes,
            )
        self._use_graph_staging_arena = (
            self._graph_staging_plan is not None
            and not self._use_graph_registered_inputs
        )
        self._use_graph_collective_fallback = (
            self._dsv4_mtp_graph_guard and self._graph_staging_plan is None
        )
        self._graph_staging_eager_reserve_bytes = (
            self._graph_staging_plan.eager_reserve_bytes
            if self._graph_staging_plan is not None
            else self.max_size
        )
        self._graph_staging_data_start = self._graph_staging_eager_reserve_bytes
        self._graph_staging_meta_start = self._graph_staging_eager_reserve_bytes
        self._graph_staging_data_limit = self.max_size
        self._graph_staging_meta_limit = self.max_size
        self._comm_id: int | None = None
        self.meta_ptrs: list[int] = []
        self.meta_opened_ipc_ptrs: list[int] = []
        self.buffer_ptrs: list[int] = []
        self.buffer_opened_ipc_ptrs: list[int] = []
        self.buffer_rank_data: torch.Tensor | None = None
        self.graph_rank_data: torch.Tensor | None = None
        self.meta_size = 0
        self._pending_graph_inputs: list[torch.Tensor] = []
        self._graph_input_refs: list[torch.Tensor] = []
        self._graph_local_handles: dict[int, bytes] = {}
        self._graph_peer_bases: dict[tuple[int, bytes], int] = {}
        self._graph_opened_ptrs: list[int] = []
        self._next_graph_slot = 0
        self._graph_registered_input_enabled = self._use_graph_registered_inputs
        self._graph_staging_data_offset = self._graph_staging_data_start
        self._graph_staging_meta_offset = self._graph_staging_meta_start
        self._graph_staging_ledger: list[
            tuple[int, int, str, int, int, int, int, int]
        ] = []
        self._graph_staging_cpu_refs: list[torch.Tensor] = []
        self._graph_staging_captured_descriptors: set[tuple[int, int]] = set()
        self._graph_staging_capture_sealed = False

        if isinstance(device, int):
            device = torch.device(f"musa:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device

        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        if self.world_size not in self._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "MUSA JIT custom allreduce disabled due to unsupported world "
                "size %d. Supported world sizes: %s.",
                self.world_size,
                self._SUPPORTED_WORLD_SIZES,
            )
            return
        self._validate_graph_staging_plan_consensus()

        try:
            meta_bytes = jit_ar.meta_size(self.world_size)
            self.meta_size = meta_bytes
            self._configure_graph_staging_arena()
            meta_buffer = _make_shared_buffer(meta_bytes + self.max_size, group=group)
            self.meta_ptrs = meta_buffer.pointers
            self.meta_opened_ipc_ptrs = meta_buffer.opened_ipc_ptrs
            buffer = _make_shared_buffer(self.max_size, group=group)
            self.buffer_ptrs = buffer.pointers
            self.buffer_opened_ipc_ptrs = buffer.opened_ipc_ptrs
            jit_ar.ensure_compiled(self.world_size)
        except Exception:
            logger.exception("MUSA JIT custom allreduce initialization failed.")
            if self.meta_opened_ipc_ptrs:
                try:
                    _close_ipc_handles(self.meta_opened_ipc_ptrs)
                except Exception:
                    pass
            if self.buffer_opened_ipc_ptrs:
                try:
                    _close_ipc_handles(self.buffer_opened_ipc_ptrs)
                except Exception:
                    pass
            if self.meta_ptrs:
                try:
                    _free_own_shared_buffer(self.meta_ptrs, group=group, rank=self.rank)
                except Exception:
                    pass
            if self.buffer_ptrs:
                try:
                    _free_own_shared_buffer(
                        self.buffer_ptrs, group=group, rank=self.rank
                    )
                except Exception:
                    pass
            self.meta_ptrs = []
            self.meta_opened_ipc_ptrs = []
            self.buffer_ptrs = []
            self.buffer_opened_ipc_ptrs = []
            return

        self.signal_ptrs_cpu = torch.tensor(self.meta_ptrs, dtype=torch.int64)
        buffer_ptrs = self.buffer_ptrs + [0] * (8 - self.world_size)
        self.buffer_rank_data = torch.tensor(buffer_ptrs, dtype=torch.int64)
        if self._use_graph_registered_inputs:
            graph_slots = _MAX_GRAPH_RANK_DATA_BYTES // (
                _MAX_RANKS * torch.tensor([], dtype=torch.int64).element_size()
            )
            self.graph_rank_data = torch.zeros(
                (graph_slots, _MAX_RANKS),
                dtype=torch.int64,
                device=self.device,
            )
            logger.info_once(
                "MUSA fused CAR-RMSNorm graph registered-input path is enabled "
                "(capacity=%d, max_input_bytes=%d). Eager execution and larger "
                "graph inputs use the staging path.",
                graph_slots,
                self._GRAPH_REGISTERED_INPUT_MAX_BYTES,
            )
        elif self._use_graph_staging_arena:
            logger.info_once(
                "Using ordinal-partitioned MUSA JIT custom all-reduce staging arena "
                "for DeepSeek-V4 MTP graphs (eager_reserve_bytes=%d, "
                "data_capacity_bytes=%d, meta_capacity_bytes=%d, "
                "communicator_buffer_bytes=%d, descriptors=%s).",
                self._graph_staging_eager_reserve_bytes,
                self._graph_staging_plan.graph_data_capacity_bytes,
                self._graph_staging_plan.graph_meta_capacity_bytes,
                self._graph_staging_plan.communicator_buffer_bytes,
                tuple(sorted(self._graph_staging_plan.capture_descriptors)),
            )
        elif self._use_graph_collective_fallback:
            logger.info_once(
                "Using the standard collective fallback instead of MUSA JIT "
                "custom all-reduce for DeepSeek-V4 MTP graph capture. Eager "
                "execution retains the fixed-staging custom path."
            )
        else:
            logger.info_once(
                "Using fixed-staging MUSA JIT custom all-reduce for "
                "DeepSeek-V4 CUDA graph capture."
            )
        self._comm_id = _register_comm(self)
        self.disabled = False
        logger.info_once(
            "Using MUSA JIT custom all-reduce for world_size=%d max_size=%d.",
            self.world_size,
            self.max_size,
        )

    def _configure_graph_staging_arena(self) -> None:
        plan = self._graph_staging_plan
        if plan is None:
            return
        data_start = plan.eager_reserve_bytes
        meta_start = self._align_graph_staging_size(
            self.meta_size + plan.eager_reserve_bytes
        )
        data_limit = data_start + plan.graph_data_capacity_bytes
        meta_limit = meta_start + plan.graph_meta_capacity_bytes
        invalid_reason = None
        if self.meta_size > plan.max_meta_bytes_per_slot:
            invalid_reason = (
                f"meta_size={self.meta_size} exceeds "
                f"max_meta_bytes_per_slot={plan.max_meta_bytes_per_slot}"
            )
        elif data_limit > self.max_size or meta_limit > self.max_size:
            invalid_reason = (
                f"data_limit={data_limit}, meta_limit={meta_limit}, "
                f"capacity={self.max_size}"
            )
        if invalid_reason is not None:
            logger.warning_once(
                "Disabling the DSV4 MTP graph CAR staging arena: %s. Graph "
                "capture will use the standard collective fallback.",
                invalid_reason,
            )
            self._graph_staging_plan = None
            self._use_graph_staging_arena = False
            self._use_graph_collective_fallback = self._dsv4_mtp_graph_guard
            self._graph_staging_eager_reserve_bytes = self.max_size
            self._graph_staging_data_start = self.max_size
            self._graph_staging_meta_start = self.max_size
            self._graph_staging_data_offset = self.max_size
            self._graph_staging_meta_offset = self.max_size
            self._graph_staging_data_limit = self.max_size
            self._graph_staging_meta_limit = self.max_size
            return
        self._graph_staging_data_start = data_start
        self._graph_staging_meta_start = meta_start
        self._graph_staging_data_offset = data_start
        self._graph_staging_meta_offset = meta_start
        self._graph_staging_data_limit = data_limit
        self._graph_staging_meta_limit = meta_limit

    @contextmanager
    def capture(self):
        if self._IS_CAPTURING:
            raise RuntimeError("Nested MUSA custom-allreduce capture is unsupported")
        self._pending_graph_inputs = []
        # v0.28 captures each graph size in a separate graph_capture context.
        # Keep the arena cursor, ledger, and tensor references cumulative so
        # every later descriptor receives fresh slots without invalidating the
        # storage referenced by graphs captured in earlier contexts.
        capture_ledger_start = len(getattr(self, "_graph_staging_ledger", ()))
        capture_succeeded = False
        capture_error: BaseException | None = None
        consensus_error: BaseException | None = None
        try:
            self._IS_CAPTURING = True
            yield
            capture_succeeded = True
        except BaseException as exc:
            capture_error = exc
        finally:
            try:
                if capture_succeeded and self._pending_graph_inputs:
                    self._register_graph_buffers()
                if getattr(self, "_use_graph_staging_arena", False):
                    try:
                        self._validate_graph_staging_capture(
                            capture_succeeded,
                            capture_error,
                        )
                        if capture_succeeded:
                            new_descriptors = {
                                (entry[0], entry[1])
                                for entry in self._graph_staging_ledger[
                                    capture_ledger_start:
                                ]
                            }
                            self._graph_staging_captured_descriptors.update(
                                new_descriptors
                            )
                            plan = self._graph_staging_plan
                            self._graph_staging_capture_sealed = (
                                plan is not None
                                and self._graph_staging_captured_descriptors
                                >= plan.capture_descriptors
                            )
                    except BaseException as exc:
                        consensus_error = exc
            finally:
                self._pending_graph_inputs = []
                self._IS_CAPTURING = False
        if consensus_error is not None:
            if capture_error is not None:
                raise consensus_error from capture_error
            raise consensus_error
        if capture_error is not None:
            raise capture_error

    @staticmethod
    def _align_graph_staging_size(size: int) -> int:
        alignment = _GRAPH_STAGING_ALIGNMENT
        return (size + alignment - 1) // alignment * alignment

    def _graph_staging_plan_fingerprint(self) -> tuple[object, ...]:
        plan = self._graph_staging_plan
        plan_fingerprint: tuple[object, ...] | None = None
        if plan is not None:
            plan_fingerprint = (
                plan.eager_reserve_bytes,
                tuple(sorted(plan.capture_descriptors)),
                plan.car_ops_per_descriptor,
                plan.bytes_per_token,
                plan.graph_data_capacity_bytes,
                plan.graph_meta_capacity_bytes,
                plan.max_meta_bytes_per_slot,
                plan.communicator_buffer_bytes,
            )
        return (
            self._dsv4_mtp_graph_guard,
            self._use_graph_registered_inputs,
            self._use_graph_staging_arena,
            self._use_graph_collective_fallback,
            plan_fingerprint,
        )

    def _validate_graph_staging_plan_consensus(self) -> None:
        if self.world_size <= 1:
            return
        local_fingerprint = self._graph_staging_plan_fingerprint()
        ranks = dist.get_process_group_ranks(self.group)
        fingerprints: list[tuple[object, ...] | None] = [None] * self.world_size
        fingerprints[self.rank] = local_fingerprint
        for group_rank, src in enumerate(ranks):
            payload = [fingerprints[group_rank]]
            dist.broadcast_object_list(payload, src=src, group=self.group, device="cpu")
            fingerprints[group_rank] = payload[0]
        if any(fingerprint != local_fingerprint for fingerprint in fingerprints):
            raise RuntimeError(
                "MUSA custom AR graph contract differs across ranks: "
                f"rank={self.rank}, fingerprints={fingerprints}"
            )

    def _graph_staging_capture_active(self) -> bool:
        return (
            getattr(self, "_use_graph_staging_arena", False)
            and getattr(self, "_IS_CAPTURING", False)
            and self._graph_staging_descriptor_enabled()
            and self._is_current_stream_capturing()
        )

    def _graph_staging_descriptor_enabled(self) -> bool:
        if not (
            getattr(self, "_use_graph_staging_arena", False)
            and getattr(self, "_IS_CAPTURING", False)
        ):
            return True
        descriptor = self._current_graph_staging_descriptor()
        plan = getattr(self, "_graph_staging_plan", None)
        return plan is not None and plan.allows_descriptor(descriptor)

    def _graph_capture_requires_standard_collective(self) -> bool:
        if not getattr(self, "_IS_CAPTURING", False):
            return False
        if getattr(self, "_use_graph_collective_fallback", False):
            return True
        return (
            getattr(self, "_use_graph_staging_arena", False)
            and not self._graph_staging_descriptor_enabled()
        )

    def _require_custom_ar_graph_path(self) -> None:
        if self._graph_capture_requires_standard_collective():
            raise RuntimeError(
                "MUSA JIT custom all-reduce was invoked for a graph descriptor "
                "whose contract requires the standard collective"
            )

    @staticmethod
    def _current_graph_staging_descriptor() -> Any | None:
        try:
            from vllm.forward_context import (
                get_forward_context,
                is_forward_context_available,
            )

            if not is_forward_context_available():
                return None
            descriptor = get_forward_context().batch_descriptor
        except (ImportError, LookupError, RuntimeError):
            return None
        return descriptor

    def _graph_staging_launch_args(
        self,
        tensor: torch.Tensor,
        op_kind: str,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
        assert self.buffer_rank_data is not None
        assert self.signal_ptrs_cpu is not None
        if not self._graph_staging_capture_active():
            return (
                self.buffer_rank_data,
                self.signal_ptrs_cpu,
                self.meta_ptrs[self.rank],
                self.buffer_ptrs[self.rank],
                (
                    self._graph_staging_eager_reserve_bytes
                    if getattr(self, "_use_graph_staging_arena", False)
                    else self.max_size
                ),
            )

        descriptor = self._current_graph_staging_descriptor()
        if descriptor is None:
            raise RuntimeError(
                "MUSA custom AR graph staging capture is missing BatchDescriptor"
            )
        descriptor_key = (int(descriptor.num_tokens), int(descriptor.num_reqs))
        if descriptor_key in getattr(
            self, "_graph_staging_captured_descriptors", set()
        ):
            raise RuntimeError(
                "MUSA custom AR graph descriptor was already captured while "
                "the previous graph may still reference its staging slots: "
                f"descriptor={descriptor_key}"
            )
        input_bytes = tensor.numel() * tensor.element_size()
        data_size = self._align_graph_staging_size(input_bytes)
        meta_size = self._align_graph_staging_size(self.meta_size + input_bytes)
        data_offset = self._graph_staging_data_offset
        meta_offset = self._graph_staging_meta_offset
        data_end = data_offset + data_size
        meta_end = meta_offset + meta_size
        if (
            data_end > self._graph_staging_data_limit
            or meta_end > self._graph_staging_meta_limit
        ):
            raise RuntimeError(
                "MUSA custom AR graph staging arena is exhausted: "
                f"op={op_kind}, descriptor={descriptor_key}, "
                f"input_bytes={input_bytes}, data_end={data_end}, "
                f"meta_end={meta_end}, "
                f"data_limit={self._graph_staging_data_limit}, "
                f"meta_limit={self._graph_staging_meta_limit}, "
                f"captured_ops={len(self._graph_staging_ledger)}"
            )
        rank_data = self.buffer_rank_data + data_offset
        signal_ptrs = self.signal_ptrs_cpu + meta_offset
        self._graph_staging_cpu_refs.extend((rank_data, signal_ptrs))
        self._graph_staging_data_offset = data_end
        self._graph_staging_meta_offset = meta_end
        self._graph_staging_ledger.append(
            (
                descriptor_key[0],
                descriptor_key[1],
                op_kind,
                input_bytes,
                data_offset,
                data_size,
                meta_offset,
                meta_size,
            )
        )
        return (
            rank_data,
            signal_ptrs,
            self.meta_ptrs[self.rank] + meta_offset,
            self.buffer_ptrs[self.rank] + data_offset,
            data_size,
        )

    def _graph_staging_manifest_error(self) -> str | None:
        plan = self._graph_staging_plan
        if plan is None:
            return "graph staging plan is unavailable"
        grouped: dict[tuple[int, int], list[tuple[Any, ...]]] = {}
        for entry in self._graph_staging_ledger:
            grouped.setdefault((entry[0], entry[1]), []).append(entry)
        for descriptor, entries in grouped.items():
            if descriptor not in plan.capture_descriptors:
                return f"unsupported captured descriptor={descriptor}"
            num_tokens, _ = descriptor
            expected_input_bytes = num_tokens * plan.bytes_per_token
            if len(entries) != plan.car_ops_per_descriptor:
                return (
                    f"descriptor={descriptor} captured_ops={len(entries)} "
                    f"expected_ops={plan.car_ops_per_descriptor}"
                )
            if any(entry[3] != expected_input_bytes for entry in entries):
                return (
                    f"descriptor={descriptor} has unexpected input byte sizes: "
                    f"{sorted({entry[3] for entry in entries})}, "
                    f"expected={expected_input_bytes}"
                )
            expected_data_bytes = self._align_graph_staging_size(
                plan.expected_descriptor_data_bytes(num_tokens)
            )
            actual_data_bytes = sum(entry[5] for entry in entries)
            if actual_data_bytes != expected_data_bytes:
                return (
                    f"descriptor={descriptor} data_bytes={actual_data_bytes} "
                    f"expected={expected_data_bytes}"
                )
            expected_meta_bytes = plan.car_ops_per_descriptor * (
                self._align_graph_staging_size(self.meta_size + expected_input_bytes)
            )
            actual_meta_bytes = sum(entry[7] for entry in entries)
            if actual_meta_bytes != expected_meta_bytes:
                return (
                    f"descriptor={descriptor} meta_bytes={actual_meta_bytes} "
                    f"expected={expected_meta_bytes}"
                )
        expected_data_total = sum(
            self._align_graph_staging_size(
                plan.expected_descriptor_data_bytes(num_tokens)
            )
            for num_tokens, _ in grouped
        )
        expected_meta_total = sum(
            self._align_graph_staging_size(
                self.meta_size + num_tokens * plan.bytes_per_token
            )
            * plan.car_ops_per_descriptor
            for num_tokens, _ in grouped
        )
        if (
            self._graph_staging_data_offset - self._graph_staging_data_start
            != expected_data_total
            or self._graph_staging_meta_offset - self._graph_staging_meta_start
            != expected_meta_total
        ):
            return (
                "graph staging arena totals differ from descriptor manifest: "
                f"data={self._graph_staging_data_offset - self._graph_staging_data_start}"
                f"/{expected_data_total}, "
                f"meta={self._graph_staging_meta_offset - self._graph_staging_meta_start}"
                f"/{expected_meta_total}"
            )
        return None

    def _validate_graph_staging_capture(
        self,
        capture_succeeded: bool,
        capture_error: BaseException | None,
    ) -> None:
        manifest_error = (
            self._graph_staging_manifest_error() if capture_succeeded else None
        )
        local_payload: dict[str, Any] = {
            "success": capture_succeeded and manifest_error is None,
            "error": (
                manifest_error
                if manifest_error is not None
                else (
                    None
                    if capture_error is None
                    else f"{type(capture_error).__name__}: {capture_error}"
                )
            ),
            "ledger": self._graph_staging_ledger,
        }
        gathered: list[dict[str, Any] | None] = [None] * self.world_size
        gathered[self.rank] = local_payload
        ranks = sorted(dist.get_process_group_ranks(group=self.group))
        for group_rank, global_rank in enumerate(ranks):
            payload = [gathered[group_rank]]
            dist.broadcast_object_list(
                payload,
                src=global_rank,
                group=self.group,
                device="cpu",
            )
            gathered[group_rank] = payload[0]
        failures = [
            (rank, None if payload is None else payload["error"])
            for rank, payload in enumerate(gathered)
            if payload is None or not payload["success"]
        ]
        if failures:
            raise RuntimeError(
                "MUSA custom AR graph staging capture failed across ranks: "
                f"{failures}"
            )
        ledgers = [payload["ledger"] for payload in gathered if payload is not None]
        if any(ledger != self._graph_staging_ledger for ledger in ledgers):
            raise RuntimeError(
                "MUSA custom AR graph staging ledger differs across ranks: "
                f"local_ops={len(self._graph_staging_ledger)}, "
                f"gathered_ops={[len(ledger) for ledger in ledgers]}"
            )
        logger.info_once(
            "Captured %d disjoint MUSA custom AR graph staging slots "
            "(data_bytes=%d, meta_bytes=%d).",
            len(self._graph_staging_ledger),
            self._graph_staging_data_offset - self._graph_staging_data_start,
            self._graph_staging_meta_offset - self._graph_staging_meta_start,
        )

    def _graph_pointer_meta(self, tensor: torch.Tensor) -> tuple[bytes, int]:
        pointer = tensor.data_ptr()
        base = _MusaDriverLibrary().allocation_base(pointer)
        handle = self._graph_local_handles.get(base)
        if handle is None:
            handle = _ipc_handle_to_bytes(
                _MusaRTLibrary().ipc_get_mem_handle(ctypes.c_void_p(base))
            )
            self._graph_local_handles[base] = handle
        return handle, pointer - base

    def _register_graph_buffers(self) -> None:
        assert self.graph_rank_data is not None
        count = len(self._pending_graph_inputs)
        start = self._next_graph_slot
        end = start + count
        if end > self.graph_rank_data.shape[0]:
            raise RuntimeError(
                "MUSA custom AR exhausted graph rank-data slots: "
                f"need={end}, capacity={self.graph_rank_data.shape[0]}"
            )

        local_meta = [
            self._graph_pointer_meta(tensor) for tensor in self._pending_graph_inputs
        ]
        gathered: list[list[tuple[bytes, int]] | None] = [None] * self.world_size
        gathered[self.rank] = local_meta
        # ``all_gather_object`` is incompatible with a Gloo process group
        # under inference mode (PyTorch #126032). Match vLLM's native custom
        # AR registration and broadcast each rank's metadata on the CPU group.
        ranks = sorted(dist.get_process_group_ranks(group=self.group))
        for group_rank, global_rank in enumerate(ranks):
            payload = [gathered[group_rank]]
            dist.broadcast_object_list(
                payload,
                src=global_rank,
                group=self.group,
                device="cpu",
            )
            gathered[group_rank] = payload[0]
        if any(peer_meta is None for peer_meta in gathered):
            raise RuntimeError("MUSA custom AR graph metadata gather was incomplete")
        gathered_meta = [peer_meta for peer_meta in gathered if peer_meta is not None]
        if any(len(peer_meta) != count for peer_meta in gathered_meta):
            raise RuntimeError(
                "MUSA custom AR graph buffer count differs across ranks: "
                f"local={count}, gathered={[len(meta) for meta in gathered_meta]}"
            )

        runtime = _MusaRTLibrary()
        rows: list[list[int]] = []
        for buffer_index, local_tensor in enumerate(self._pending_graph_inputs):
            row = [0] * _MAX_RANKS
            for peer_rank, peer_meta in enumerate(gathered_meta):
                handle, offset = peer_meta[buffer_index]
                if peer_rank == self.rank:
                    pointer = local_tensor.data_ptr()
                else:
                    peer_key = (peer_rank, handle)
                    peer_base = self._graph_peer_bases.get(peer_key)
                    if peer_base is None:
                        peer_base = int(
                            runtime.ipc_open_mem_handle(
                                _ipc_handle_from_bytes(handle)
                            ).value
                        )
                        self._graph_peer_bases[peer_key] = peer_base
                        self._graph_opened_ptrs.append(peer_base)
                    pointer = peer_base + int(offset)
                row[peer_rank] = pointer
            rows.append(row)

        rank_data_cpu = torch.tensor(rows, dtype=torch.int64)
        self.graph_rank_data[start:end].copy_(rank_data_cpu.to(self.device))
        torch.musa.synchronize(self.device)
        self._graph_input_refs.extend(self._pending_graph_inputs)
        self._pending_graph_inputs = []
        self._next_graph_slot = end

    def _graph_rank_data_for_input(self, tensor: torch.Tensor) -> torch.Tensor:
        assert self.graph_rank_data is not None
        slot = self._next_graph_slot + len(self._pending_graph_inputs)
        if slot >= self.graph_rank_data.shape[0]:
            raise RuntimeError(
                "MUSA custom AR graph rank-data buffer is full: "
                f"slot={slot}, capacity={self.graph_rank_data.shape[0]}"
            )
        self._pending_graph_inputs.append(tensor)
        return self.graph_rank_data[slot]

    def _graph_registered_input_eligible(self, tensor: torch.Tensor) -> bool:
        return (
            self._graph_registered_input_enabled
            and tensor.numel() * tensor.element_size()
            <= self._GRAPH_REGISTERED_INPUT_MAX_BYTES
        )

    def _is_piecewise_cudagraph_capture(self) -> bool:
        try:
            from vllm.config import CUDAGraphMode
            from vllm.forward_context import (
                get_forward_context,
                is_forward_context_available,
            )
        except ImportError:
            return False
        if not is_forward_context_available():
            return False
        return get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE

    def _use_registered_graph_input(self, tensor: torch.Tensor) -> bool:
        return (
            self._use_graph_registered_inputs
            and self._graph_registered_input_eligible(tensor)
            and self._IS_CAPTURING
            and self._is_current_stream_capturing()
            and not self._is_piecewise_cudagraph_capture()
        )

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        use = self._should_custom_ar_gates(inp)
        car_trace.record(
            "gate",
            "v1",
            "car" if use else "fallback",
            inp.numel() * inp.element_size(),
        )
        return use

    def _should_custom_ar_gates(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if (
            self._IS_CAPTURING
            and (self._use_graph_collective_fallback or self._use_graph_staging_arena)
            and self._graph_capture_requires_standard_collective()
        ):
            return False
        if (
            self._IS_CAPTURING
            and self._dsv4_mtp_graph_guard
            and self._use_graph_registered_inputs
            and not self._graph_registered_input_eligible(inp)
        ):
            return False
        if inp.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0 or inp_size > self.max_size:
            return False
        if (
            getattr(self, "_use_graph_staging_arena", False)
            and not getattr(self, "_IS_CAPTURING", False)
            and inp_size > self._graph_staging_eager_reserve_bytes
        ):
            return False
        return inp.is_contiguous()

    def should_custom_all_gather(
        self,
        inp: torch.Tensor,
        dim: int = -1,
        output_dtype: torch.dtype | None = None,
    ) -> bool:
        if self.disabled or self.world_size not in (2, 4, 8) or inp.ndim != 2:
            return False
        if (
            self._IS_CAPTURING
            and (self._use_graph_collective_fallback or self._use_graph_staging_arena)
            and self._graph_capture_requires_standard_collective()
        ):
            return False
        if not self._is_communicator_tensor(inp):
            return False
        normalized_dim = dim if dim >= 0 else inp.ndim + dim
        if normalized_dim != inp.ndim - 1:
            return False
        if inp.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
        if output_dtype is None:
            output_dtype = inp.dtype
        if output_dtype != inp.dtype and not (
            inp.dtype == torch.bfloat16 and output_dtype == torch.float32
        ):
            return False
        if not inp.is_contiguous() or inp.shape[0] <= 0 or inp.shape[1] <= 0:
            return False
        pack = 16 // inp.element_size()
        if inp.shape[1] % pack != 0:
            return False
        inp_size = inp.numel() * inp.element_size()
        output_numel = inp.numel() * self.world_size
        output_element_size = 4 if output_dtype == torch.float32 else inp.element_size()
        output_size = output_numel * output_element_size
        if (
            getattr(self, "_use_graph_staging_arena", False)
            and output_size > self._graph_staging_eager_reserve_bytes
        ):
            return False
        return (
            inp_size <= self.max_size
            and inp_size % 16 == 0
            and output_size <= self.max_size
            and output_numel <= _INT32_MAX
        )

    def _is_communicator_tensor(self, inp: torch.Tensor) -> bool:
        return inp.device.type == "musa" and inp.device.index == self.device.index

    def _is_default_stream(self, inp: torch.Tensor) -> bool:
        try:
            current = torch.musa.current_stream(device=inp.device)
            default = torch.musa.default_stream(device=inp.device)
            return current == default
        except Exception:
            return False

    def _reject_fused_allreduce_rmsnorm_reason(
        self, inp: torch.Tensor, weight: torch.Tensor
    ) -> str | None:
        if self.disabled:
            return "communicator is disabled"
        if (
            self._IS_CAPTURING
            and (self._use_graph_collective_fallback or self._use_graph_staging_arena)
            and self._graph_capture_requires_standard_collective()
        ):
            return "active graph capture requires the standard collective"
        if inp.device.type != "musa" or weight.device.type != "musa":
            return f"device mismatch: input={inp.device} weight={weight.device}"
        if inp.dim() != 2 or weight.dim() != 1:
            return f"dim mismatch: input_dim={inp.dim()} weight_dim={weight.dim()}"
        hidden_size = inp.shape[-1]
        if hidden_size <= 0 or hidden_size != weight.numel():
            return f"hidden/weight mismatch: hidden={hidden_size} weight_numel={weight.numel()}"
        if hidden_size % 8 != 0 or hidden_size > 32768:
            return f"unsupported hidden size: hidden={hidden_size}"
        if inp.dtype not in (torch.float16, torch.bfloat16):
            return f"unsupported input dtype: {inp.dtype}"
        if weight.dtype not in (inp.dtype, torch.float32):
            return (
                "unsupported weight dtype for fused RMSNorm: "
                f"input={inp.dtype} weight={weight.dtype}"
            )
        if not inp.is_contiguous() or not weight.is_contiguous():
            return (
                "non-contiguous input/weight: "
                f"input_stride={inp.stride()} weight_stride={weight.stride()}"
            )
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0 or inp_size > self.max_size:
            return f"unsupported byte size: bytes={inp_size} max_size={self.max_size}"
        if (
            getattr(self, "_use_graph_staging_arena", False)
            and not getattr(self, "_IS_CAPTURING", False)
            and inp_size > self._graph_staging_eager_reserve_bytes
        ):
            return (
                "input exceeds the eager partition of the graph staging arena: "
                f"bytes={inp_size} reserve={self._graph_staging_eager_reserve_bytes}"
            )
        return None

    def should_fused_allreduce_rmsnorm(
        self, inp: torch.Tensor, weight: torch.Tensor
    ) -> bool:
        return self._reject_fused_allreduce_rmsnorm_reason(inp, weight) is None

    def _reject_fused_allreduce_residual_rmsnorm_reason(
        self, inp: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> str | None:
        base_reason = self._reject_fused_allreduce_rmsnorm_reason(inp, weight)
        if base_reason is not None:
            return base_reason
        if residual.device.type != "musa":
            return f"residual device mismatch: residual={residual.device}"
        if residual.dim() != 2 or residual.shape != inp.shape:
            return (
                "residual shape mismatch: "
                f"input_shape={tuple(inp.shape)} residual_shape={tuple(residual.shape)}"
            )
        if residual.dtype != inp.dtype:
            return (
                f"residual dtype mismatch: input={inp.dtype} residual={residual.dtype}"
            )
        if not residual.is_contiguous():
            return f"non-contiguous residual: residual_stride={residual.stride()}"
        return None

    def should_fused_allreduce_residual_rmsnorm(
        self, inp: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> bool:
        return (
            self._reject_fused_allreduce_residual_rmsnorm_reason(inp, residual, weight)
            is None
        )

    @staticmethod
    def _is_current_stream_capturing() -> bool:
        try:
            return bool(torch.get_device_module().is_current_stream_capturing())
        except Exception:
            return False

    @staticmethod
    def _is_torch_compiling() -> bool:
        try:
            if torch.compiler.is_compiling():
                return True
        except Exception:
            pass
        try:
            return bool(torch._dynamo.is_compiling())
        except Exception:
            return False

    def _custom_all_reduce_custom_op(self, input: torch.Tensor) -> torch.Tensor:
        assert self._comm_id is not None
        return torch.ops.vllm.musa_jit_custom_all_reduce(input, self._comm_id)

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor | None:
        if not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if not self._is_current_stream_capturing():
                # Match vLLM's CustomAllreduce warmup semantics. During graph
                # warmup no communication should run; the real graph capture
                # below records the registered-input JIT AR kernel.
                return torch.empty_like(input)
            return self._custom_all_reduce_custom_op(input)
        if self._is_torch_compiling():
            # Keep Dynamo/Inductor from tracing into the Python Tile/JIT module
            # loader. The registered op provides the compile-time fake impl and
            # launches the same fixed-staging kernel at runtime.
            return self._custom_all_reduce_custom_op(input)
        # Eager tensors are not pointer-stable, so retain the fixed staging
        # path outside graph capture.
        return self._custom_all_reduce_impl(input)

    def _custom_all_reduce_impl(self, input: torch.Tensor) -> torch.Tensor:
        assert self.buffer_rank_data is not None
        assert self.signal_ptrs_cpu is not None
        self._require_custom_ar_graph_path()
        if self._IS_CAPTURING and self._use_registered_graph_input(input):
            return self._graph_custom_all_reduce_impl(input)
        out = torch.empty_like(input)
        nbytes = input.numel() * input.element_size()
        shot = jit_ar.preferred_shot(self.world_size, nbytes)
        car_trace.record("algo", "v1", f"shot{shot}", nbytes)
        if self._use_graph_staging_arena:
            (
                rank_data,
                signal_ptrs,
                self_meta_ptr,
                self_buffer_ptr,
                max_size,
            ) = self._graph_staging_launch_args(input, "allreduce")
        else:
            rank_data = self.buffer_rank_data
            signal_ptrs = self.signal_ptrs_cpu
            self_meta_ptr = self.meta_ptrs[self.rank]
            self_buffer_ptr = self.buffer_ptrs[self.rank]
            max_size = self.max_size
        jit_ar.launch_unregistered(
            rank_data,
            signal_ptrs,
            input,
            out,
            self_meta_ptr,
            self_buffer_ptr,
            max_size,
            self.rank,
            self.world_size,
            shot,
        )
        return out

    def _graph_custom_all_reduce_impl(self, input: torch.Tensor) -> torch.Tensor:
        assert self.signal_ptrs_cpu is not None
        self._require_custom_ar_graph_path()
        rank_data = self._graph_rank_data_for_input(input)
        out = torch.empty_like(input)
        jit_ar.launch_graph_registered(
            rank_data,
            self.signal_ptrs_cpu,
            input,
            out,
            self.rank,
            self.world_size,
            jit_ar.preferred_shot(self.world_size, input.nbytes),
        )
        return out

    def custom_all_gather(
        self,
        input: torch.Tensor,
        dim: int = -1,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        if not self.should_custom_all_gather(input, dim, output_dtype):
            return None
        # Logits currently run outside the model graph. Fail closed if a future
        # runner moves this call under Dynamo or graph capture; that path needs
        # pointer-registration semantics rather than an eager FFI call.
        if (
            self._IS_CAPTURING
            or self._is_current_stream_capturing()
            or self._is_torch_compiling()
            or not self._is_default_stream(input)
        ):
            return None
        return self._custom_all_gather_impl(input, output_dtype)

    def _custom_all_gather_impl(
        self,
        input: torch.Tensor,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        assert self.buffer_rank_data is not None
        output_shape = (*input.shape[:-1], input.shape[-1] * self.world_size)
        out = torch.empty(
            output_shape,
            dtype=output_dtype if output_dtype is not None else input.dtype,
            device=input.device,
        )
        jit_ar.launch_all_gather(
            self.buffer_rank_data,
            self.signal_ptrs_cpu,
            input,
            out,
            self.meta_ptrs[self.rank],
            self.buffer_ptrs[self.rank],
            self.max_size,
            self.rank,
            self.world_size,
        )
        return out

    def _fused_allreduce_rmsnorm_custom_op(
        self, input: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._comm_id is not None
        return torch.ops.vllm.musa_jit_fused_allreduce_rmsnorm(
            input, weight, float(eps), self._comm_id
        )

    def fused_allreduce_rmsnorm(
        self, input: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.should_fused_allreduce_rmsnorm(input, weight):
            return None
        if self._IS_CAPTURING:
            if not self._is_current_stream_capturing():
                return torch.empty_like(input), torch.empty_like(input)
            return self._fused_allreduce_rmsnorm_custom_op(input, weight, eps)
        if self._is_torch_compiling():
            return self._fused_allreduce_rmsnorm_custom_op(input, weight, eps)
        return self._fused_allreduce_rmsnorm_impl(input, weight, eps)

    def _fused_allreduce_rmsnorm_impl(
        self, input: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.buffer_rank_data is not None
        self._require_custom_ar_graph_path()
        norm_out = torch.empty_like(input)
        reduced = torch.empty_like(input)
        shot = jit_ar.preferred_shot(
            self.world_size, input.numel() * input.element_size()
        )
        use_registered = self._use_registered_graph_input(input)
        if use_registered:
            rank_data = self._graph_rank_data_for_input(input)
            signal_ptrs = self.signal_ptrs_cpu
            self_meta_ptr = self.meta_ptrs[self.rank]
            self_buffer_ptr = self.buffer_ptrs[self.rank]
            max_size = self.max_size
        elif self._use_graph_staging_arena:
            (
                rank_data,
                signal_ptrs,
                self_meta_ptr,
                self_buffer_ptr,
                max_size,
            ) = self._graph_staging_launch_args(input, "fused_rmsnorm")
        else:
            rank_data = self.buffer_rank_data
            signal_ptrs = self.signal_ptrs_cpu
            self_meta_ptr = self.meta_ptrs[self.rank]
            self_buffer_ptr = self.buffer_ptrs[self.rank]
            max_size = self.max_size
        launcher = (
            jit_ar.launch_fused_allreduce_rmsnorm_registered
            if use_registered
            else jit_ar.launch_fused_allreduce_rmsnorm_unregistered
        )
        launcher(
            rank_data,
            signal_ptrs,
            input,
            weight,
            norm_out,
            reduced,
            self_meta_ptr,
            self_buffer_ptr,
            max_size,
            self.rank,
            self.world_size,
            shot,
            float(eps),
        )
        return norm_out, reduced

    def _fused_allreduce_residual_rmsnorm_custom_op(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._comm_id is not None
        return torch.ops.vllm.musa_jit_fused_allreduce_residual_rmsnorm(
            input, residual, weight, float(eps), self._comm_id
        )

    def _fused_allreduce_residual_rmsnorm_no_raw_custom_op(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._comm_id is not None
        return torch.ops.vllm.musa_jit_fused_allreduce_residual_rmsnorm_no_raw(
            input, residual, weight, float(eps), self._comm_id
        )

    def fused_allreduce_residual_rmsnorm(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not self.should_fused_allreduce_residual_rmsnorm(input, residual, weight):
            return None
        if self._IS_CAPTURING:
            if not self._is_current_stream_capturing():
                return (
                    torch.empty_like(input),
                    torch.empty_like(input),
                    torch.empty_like(input),
                )
            return self._fused_allreduce_residual_rmsnorm_custom_op(
                input, residual, weight, eps
            )
        if self._is_torch_compiling():
            return self._fused_allreduce_residual_rmsnorm_custom_op(
                input, residual, weight, eps
            )
        return self._fused_allreduce_residual_rmsnorm_impl(input, residual, weight, eps)

    def fused_allreduce_residual_rmsnorm_no_raw(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.should_fused_allreduce_residual_rmsnorm(input, residual, weight):
            return None
        if self._IS_CAPTURING:
            if not self._is_current_stream_capturing():
                return torch.empty_like(input), torch.empty_like(input)
            return self._fused_allreduce_residual_rmsnorm_no_raw_custom_op(
                input, residual, weight, eps
            )
        if self._is_torch_compiling():
            return self._fused_allreduce_residual_rmsnorm_no_raw_custom_op(
                input, residual, weight, eps
            )
        return self._fused_allreduce_residual_rmsnorm_no_raw_impl(
            input, residual, weight, eps
        )

    def _fused_allreduce_residual_rmsnorm_impl(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.buffer_rank_data is not None
        self._require_custom_ar_graph_path()
        norm_out = torch.empty_like(input)
        residual_out = torch.empty_like(input)
        reduced = torch.empty_like(input)
        shot = jit_ar.preferred_shot(
            self.world_size, input.numel() * input.element_size()
        )
        use_registered = self._use_registered_graph_input(input)
        if use_registered:
            rank_data = self._graph_rank_data_for_input(input)
            signal_ptrs = self.signal_ptrs_cpu
            self_meta_ptr = self.meta_ptrs[self.rank]
            self_buffer_ptr = self.buffer_ptrs[self.rank]
            max_size = self.max_size
        elif self._use_graph_staging_arena:
            (
                rank_data,
                signal_ptrs,
                self_meta_ptr,
                self_buffer_ptr,
                max_size,
            ) = self._graph_staging_launch_args(input, "fused_residual_rmsnorm")
        else:
            rank_data = self.buffer_rank_data
            signal_ptrs = self.signal_ptrs_cpu
            self_meta_ptr = self.meta_ptrs[self.rank]
            self_buffer_ptr = self.buffer_ptrs[self.rank]
            max_size = self.max_size
        launcher = (
            jit_ar.launch_fused_allreduce_residual_rmsnorm_registered
            if use_registered
            else jit_ar.launch_fused_allreduce_residual_rmsnorm_unregistered
        )
        launcher(
            rank_data,
            signal_ptrs,
            input,
            residual,
            weight,
            norm_out,
            residual_out,
            reduced,
            self_meta_ptr,
            self_buffer_ptr,
            max_size,
            self.rank,
            self.world_size,
            shot,
            float(eps),
        )
        return norm_out, residual_out, reduced

    def _fused_allreduce_residual_rmsnorm_no_raw_impl(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.buffer_rank_data is not None
        self._require_custom_ar_graph_path()
        norm_out = torch.empty_like(input)
        residual_out = torch.empty_like(input)
        shot = jit_ar.preferred_shot(
            self.world_size, input.numel() * input.element_size()
        )
        use_registered = self._use_registered_graph_input(input)
        if use_registered:
            rank_data = self._graph_rank_data_for_input(input)
            signal_ptrs = self.signal_ptrs_cpu
            self_meta_ptr = self.meta_ptrs[self.rank]
            self_buffer_ptr = self.buffer_ptrs[self.rank]
            max_size = self.max_size
        elif self._use_graph_staging_arena:
            (
                rank_data,
                signal_ptrs,
                self_meta_ptr,
                self_buffer_ptr,
                max_size,
            ) = self._graph_staging_launch_args(input, "fused_residual_rmsnorm_no_raw")
        else:
            rank_data = self.buffer_rank_data
            signal_ptrs = self.signal_ptrs_cpu
            self_meta_ptr = self.meta_ptrs[self.rank]
            self_buffer_ptr = self.buffer_ptrs[self.rank]
            max_size = self.max_size
        launcher = (
            jit_ar.launch_fused_allreduce_residual_rmsnorm_no_raw_registered
            if use_registered
            else jit_ar.launch_fused_allreduce_residual_rmsnorm_no_raw_unregistered
        )
        launcher(
            rank_data,
            signal_ptrs,
            input,
            residual,
            weight,
            norm_out,
            residual_out,
            self_meta_ptr,
            self_buffer_ptr,
            max_size,
            self.rank,
            self.world_size,
            shot,
            float(eps),
        )
        return norm_out, residual_out

    def close(self) -> None:
        if self.meta_opened_ipc_ptrs:
            _close_ipc_handles(self.meta_opened_ipc_ptrs)
        if self.buffer_opened_ipc_ptrs:
            _close_ipc_handles(self.buffer_opened_ipc_ptrs)
        if self._graph_opened_ptrs:
            _close_ipc_handles(self._graph_opened_ptrs)
        if self.meta_ptrs:
            try:
                _free_own_shared_buffer(self.meta_ptrs, rank=self.rank)
            except Exception:
                logger.debug(
                    "Failed to free MUSA custom AR metadata buffer.", exc_info=True
                )
        if self.buffer_ptrs:
            try:
                _free_own_shared_buffer(self.buffer_ptrs, rank=self.rank)
            except Exception:
                logger.debug(
                    "Failed to free MUSA custom AR staging buffer.", exc_info=True
                )
        if self._comm_id is not None:
            _COMM_REGISTRY.pop(self._comm_id, None)
            self._comm_id = None
        self.meta_ptrs = []
        self.meta_opened_ipc_ptrs = []
        self.buffer_ptrs = []
        self.buffer_opened_ipc_ptrs = []
        self.buffer_rank_data = None
        self.graph_rank_data = None
        self._graph_opened_ptrs = []
        self._graph_peer_bases = {}
        self._graph_local_handles = {}
        self._graph_input_refs = []
        self._pending_graph_inputs = []
        self._graph_staging_ledger = []
        self._graph_staging_cpu_refs = []
        self._graph_staging_captured_descriptors = set()
        self._graph_staging_capture_sealed = False
        self._next_graph_slot = 0
        self.disabled = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class MusaJitCustomAllreduce:
    """JIT custom all-reduce for MUSA.

    Keep a single all-reduce implementation across eager and CUDA graph paths.
    Falling back to vLLM's native CustomAllreduce on MUSA can segfault in
    graph/Dynamo startup paths, so unsupported tensors should route to the next
    communicator backend instead of a second custom-allreduce implementation.
    """

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size: int = _MusaJitCustomAllreduceImpl._MAX_CAR_SIZE,
        symm_mem_enabled: bool = False,
    ) -> None:
        self.group = group
        self.device = device
        self._jit_comm: _MusaJitCustomAllreduceImpl | Any | None = None
        del symm_mem_enabled

        # The implementation swap happens here, at construction, mirroring the
        # upstream ``dispatch_custom_allreduce()``: the environment variable
        # decides once which implementation is built, and from then on the
        # caller only sees that one set of methods -- the hot path never
        # branches between v1 and v2.
        try:
            if _custom_ar_v2_enabled():
                self._jit_comm = _MusaCustomAllreduceV2Impl(
                    group=group,
                    device=device,
                    max_size=max_size,
                )
            else:
                self._jit_comm = _MusaJitCustomAllreduceImpl(
                    group=group,
                    device=device,
                    max_size=max_size,
                )
        except Exception:
            logger.exception("Failed to initialize MUSA JIT custom all-reduce.")

        availability = torch.tensor([int(self._jit_available)], dtype=torch.int32)
        dist.all_reduce(availability, op=dist.ReduceOp.MIN, group=group)
        if not bool(availability.item()) and self._jit_comm is not None:
            self._jit_comm.close()
            self._jit_comm = None
            logger.warning_once(
                "MUSA JIT custom all-reduce disabled because a peer rank "
                "failed initialization."
            )

        if self._jit_available:
            logger.info_once(
                "Using MUSA JIT custom all-reduce (%s) for eager and eligible "
                "CUDA graph paths.",
                "v2" if _custom_ar_v2_enabled() else "v1",
            )

    @property
    def _jit_available(self) -> bool:
        return self._jit_comm is not None and not self._jit_comm.disabled

    @property
    def disabled(self) -> bool:
        return not self._jit_available

    @contextmanager
    def capture(self):
        if self._jit_available:
            with self._jit_comm.capture():
                yield
        else:
            yield

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        return self._jit_available and self._jit_comm.should_custom_ar(inp)

    def should_fused_allreduce_rmsnorm(
        self, inp: torch.Tensor, weight: torch.Tensor
    ) -> bool:
        return self._jit_available and self._jit_comm.should_fused_allreduce_rmsnorm(
            inp, weight
        )

    def should_fused_allreduce_residual_rmsnorm(
        self, inp: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> bool:
        return (
            self._jit_available
            and self._jit_comm.should_fused_allreduce_residual_rmsnorm(
                inp, residual, weight
            )
        )

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor | None:
        if self._jit_available and self._jit_comm.should_custom_ar(input):
            return self._jit_comm.custom_all_reduce(input)
        return None

    def custom_all_gather(
        self,
        input: torch.Tensor,
        dim: int = -1,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        if self._jit_available and self._jit_comm.should_custom_all_gather(
            input, dim, output_dtype
        ):
            return self._jit_comm.custom_all_gather(input, dim, output_dtype)
        return None

    def fused_allreduce_rmsnorm(
        self, input: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self._jit_available and self._jit_comm.should_fused_allreduce_rmsnorm(
            input, weight
        ):
            return self._jit_comm.fused_allreduce_rmsnorm(input, weight, eps)
        return None

    def reject_fused_allreduce_residual_rmsnorm_reason(
        self, inp: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> str | None:
        if not self._jit_available:
            return "JIT communicator is unavailable"
        return self._jit_comm._reject_fused_allreduce_residual_rmsnorm_reason(
            inp, residual, weight
        )

    def fused_allreduce_residual_rmsnorm(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if (
            self._jit_available
            and self._jit_comm.should_fused_allreduce_residual_rmsnorm(
                input, residual, weight
            )
        ):
            return self._jit_comm.fused_allreduce_residual_rmsnorm(
                input, residual, weight, eps
            )
        if self._jit_available:
            logger.warning_once(
                "MUSA JIT fused allreduce-residual-rmsnorm rejected tensor: %s",
                self._jit_comm._reject_fused_allreduce_residual_rmsnorm_reason(
                    input, residual, weight
                ),
            )
        return None

    def fused_allreduce_residual_rmsnorm_no_raw(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            self._jit_available
            and self._jit_comm.should_fused_allreduce_residual_rmsnorm(
                input, residual, weight
            )
        ):
            return self._jit_comm.fused_allreduce_residual_rmsnorm_no_raw(
                input, residual, weight, eps
            )
        if self._jit_available:
            logger.warning_once(
                "MUSA JIT fused allreduce-residual-rmsnorm no-raw rejected tensor: %s",
                self._jit_comm._reject_fused_allreduce_residual_rmsnorm_reason(
                    input, residual, weight
                ),
            )
        return None

    def close(self) -> None:
        if self._jit_comm is not None:
            self._jit_comm.close()
            self._jit_comm = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def maybe_musa_jit_logits_all_gather(
    input: torch.Tensor,
    dim: int = -1,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    """Use the MUSA IPC communicator for a last-dimension logits gather."""
    from vllm.distributed.parallel_state import get_tp_group

    device_communicator = get_tp_group().device_communicator
    custom_ar = getattr(device_communicator, "ca_comm", None)
    if not isinstance(custom_ar, MusaJitCustomAllreduce):
        return None
    output = custom_ar.custom_all_gather(input, dim, output_dtype)
    if output is not None:
        logger.info_once("Using MUSA JIT IPC all-gather for TP logits.")
    return output

# ==========================================================================
# v2 implementation: runtime bindings, ported object layer, and the
# implementation class that the facade selects at construction time.
# ==========================================================================

cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToDevice = 3


def _as_void_p(value):
    """Return ``value`` as a ``ctypes.c_void_p`` without double-wrapping."""
    if isinstance(value, ctypes.c_void_p):
        return value
    if isinstance(value, int):
        return ctypes.c_void_p(value)
    data_ptr = getattr(value, "data_ptr", None)
    if callable(data_ptr):
        return ctypes.c_void_p(int(data_ptr()))
    return ctypes.cast(value, ctypes.c_void_p)


class _MusaRTLibraryV2(_MusaRTLibrary):
    """Give the v2 object a CUDA-name runtime surface, reusing v1's bindings."""

    def __init__(self, so_file: str = "libmusart.so") -> None:
        super().__init__(so_file)
        if "cudaMemcpy" not in self.funcs:
            spec = _Function(
                "cudaMemcpy",
                cudaError_t,
                [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_uint,
                ],
            )
            func = getattr(self.lib, spec.name)
            func.restype = spec.restype
            func.argtypes = spec.argtypes
            self.funcs[spec.name] = func

    # ---- API names of the upstream v2 implementation ----------------------
    def cudaMalloc(self, size: int) -> ctypes.c_void_p:
        return self.malloc(int(size))

    def cudaFree(self, ptr) -> None:
        self.free(ptr)

    def cudaMemset(self, ptr, value: int, count: int) -> None:
        self.memset(ptr, int(value), int(count))

    def cudaIpcGetMemHandle(self, ptr) -> cudaIpcMemHandle_t:
        return self.ipc_get_mem_handle(ptr)

    def cudaIpcOpenMemHandle(self, handle: cudaIpcMemHandle_t) -> ctypes.c_void_p:
        return self.ipc_open_mem_handle(handle)

    def cudaIpcCloseMemHandle(self, ptr) -> None:
        self.ipc_close_mem_handle(ptr)

    def cudaMemcpy(self, dst, src, count: int, kind: int = cudaMemcpyHostToDevice) -> None:
        # The v2 object passes ``ctypes.c_void_p`` for both ends (it builds the
        # host staging buffer itself), while tensors and plain ints also reach
        # this shim.  Normalise instead of re-wrapping, because wrapping an
        # existing c_void_p raises "cannot be converted to pointer".
        self.check(
            self.funcs["cudaMemcpy"](
                _as_void_p(dst),
                _as_void_p(src),
                int(count),
                int(kind),
            )
        )


__all__ = [
    "_MusaRTLibraryV2",
    "cudaMemcpyHostToDevice",
    "cudaMemcpyDeviceToDevice",
    "cudaIpcMemHandle_t",
]


CudaRTLibrary = _MusaRTLibraryV2

requires_cuda_workspace_preflight = False


class AllReduceFusionPattern(IntEnum):
    kARResidualRMSNorm = 0


class MusaAllReduceFusionWorkspace:
    backend = "musa_custom_ar_v2"

    def __init__(
        self,
        coordinator,
        world_size: int,
        rank: int,
        max_token_num: int,
        hidden_dim: int,
        dtype: torch.dtype,
    ) -> None:
        self.coordinator = coordinator
        self.world_size = world_size
        self.rank = rank
        self.max_token_num = max_token_num
        self.hidden_dim = hidden_dim
        self.dtype = dtype

    def is_buffer_size_sufficient(
        self,
        *,
        tp_size: int,
        num_tokens: int,
        hidden_dim: int,
        dtype: torch.dtype,
        use_oneshot: bool | None = None,
    ) -> bool:
        del use_oneshot
        ca_comm = self.coordinator.ca_comm
        return (
            ca_comm is not None
            and not ca_comm.disabled
            and tp_size == self.world_size
            and num_tokens <= self.max_token_num
            and hidden_dim == self.hidden_dim
            and dtype == self.dtype
            and num_tokens * hidden_dim * dtype.itemsize <= ca_comm.max_size
        )

    def destroy(self) -> None:
        # The GroupCoordinator owns the communicator and its IPC allocation.
        self.coordinator = None


def _find_group_coordinator(device_group, world_size: int, rank: int):
    from sglang.srt.distributed import (
        get_attn_tp_group,
        get_moe_ep_group,
        get_moe_tp_group,
        get_tp_group,
    )

    getters = (get_attn_tp_group, get_moe_ep_group, get_moe_tp_group, get_tp_group)
    for getter in getters:
        try:
            coordinator = getter()
        except AssertionError:
            continue
        if (
            coordinator.device_group is device_group
            and coordinator.world_size == world_size
            and coordinator.rank_in_group == rank
        ):
            return coordinator
    raise RuntimeError(
        "MUSA allreduce-fusion workspace could not find the matching "
        f"GroupCoordinator for rank={rank}, world_size={world_size}"
    )


def create_allreduce_fusion_workspace(
    *,
    backend: str,
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    force_oneshot_support: bool = False,
    group=None,
    **kwargs,
) -> MusaAllReduceFusionWorkspace:
    del force_oneshot_support, kwargs
    if backend != "trtllm":
        raise ValueError(f"Unsupported MUSA allreduce-fusion backend: {backend}")
    if group is None:
        raise ValueError("MUSA allreduce-fusion workspace requires a device group")
    coordinator = _find_group_coordinator(group, world_size, rank)
    ca_comm = coordinator.ca_comm
    if ca_comm is None or ca_comm.disabled:
        raise RuntimeError("MUSA custom allreduce v2 communicator is unavailable")
    return MusaAllReduceFusionWorkspace(
        coordinator, world_size, rank, max_token_num, hidden_dim, dtype
    )


def allreduce_fusion(
    *,
    input: torch.Tensor,
    workspace: MusaAllReduceFusionWorkspace,
    pattern: AllReduceFusionPattern,
    residual_out: torch.Tensor,
    norm_out: torch.Tensor,
    residual_in: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    launch_with_pdl: bool = True,
    trigger_completion_at_end: bool = False,
    use_oneshot: bool | None = None,
    fp32_acc: bool = False,
) -> bool:
    del launch_with_pdl, trigger_completion_at_end, use_oneshot, fp32_acc
    if pattern != AllReduceFusionPattern.kARResidualRMSNorm:
        raise ValueError(f"Unsupported MUSA allreduce-fusion pattern: {pattern}")
    ca_comm = workspace.coordinator.ca_comm
    result = ca_comm.fused_allreduce_rmsnorm(
        input,
        residual_in,
        rms_gamma,
        rms_eps,
        norm_out=norm_out,
        residual_out=residual_out,
    )
    return result is not None


class ConfigResult(NamedTuple):
    num_blocks: int
    num_threads: int


def _align(value: int, alignment: int = 128) -> int:
    return (value + alignment - 1) // alignment * alignment


def _handle_to_list(handle: cudaIpcMemHandle_t) -> list[int]:
    return list(bytes(handle))


def _handle_from_list(values: list[int]) -> cudaIpcMemHandle_t:
    raw = bytes(int(value) & 0xFF for value in values)
    if len(raw) != ctypes.sizeof(cudaIpcMemHandle_t):
        raise ValueError(
            "Invalid MUSA IPC handle size: "
            f"{len(raw)} != {ctypes.sizeof(cudaIpcMemHandle_t)}"
        )
    return cudaIpcMemHandle_t.from_buffer_copy(raw)


class _AllReduceData(ctypes.Structure):
    """Host-side rank descriptor matching CUDA's ``AllReduceData`` layout."""

    _fields_ = [("input", ctypes.c_void_p * 8)]


class MusaCustomAllReduceV2Obj:
    """MUSA implementation of CUDA custom-all-reduce v2's operator ABI.

    The public ``CustomAllReduceV2`` Python control plane owns process-group
    coordination and calls this object through the same methods as the CUDA
    TVM-FFI object. Storage, MUSA IPC, and kernel selection remain backend
    details here. CUDA Graph input registration follows the shared v2 ABI.
    """

    supported_world_sizes = (2, 4, 6, 8)
    # Consumed by the shared CustomAllReduceV2._determine_algo() control plane.
    # S5000 thresholds are selected from eager-mode crossover measurements.
    # Pull covers TP2/TP4 payloads; TP4/TP6 switch to two-shot for larger
    # payloads. Push remains available through an explicit algorithm override.
    mode_configs: ClassVar[dict[int, tuple[int, int]]] = {
        2: (0, 1 << 60),
        4: (0, 256 * 1024),
        6: (0, 256 * 1024),
        8: (0, 0),
    }

    def __init__(
        self,
        rank: int,
        world_size: int,
        pull_buffer_bytes: int,
        push_buffer_bytes: int,
        graph_input_count: int,
        *,
        max_pull_blocks: int | None = None,
        max_push_blocks: int | None = None,
    ) -> None:
        del max_pull_blocks, max_push_blocks
        if world_size not in self.supported_world_sizes:
            raise ValueError(
                f"MUSA custom all-reduce v2 does not support world_size={world_size}"
            )
        if not 0 <= rank < world_size:
            raise ValueError(f"Invalid rank={rank} for world_size={world_size}")

        self.rank = int(rank)
        self.world_size = int(world_size)
        self._pull_buffer_bytes = int(pull_buffer_bytes)
        self._push_buffer_bytes = int(push_buffer_bytes)
        self._opened_ipc_ptrs: dict[bytes, int] = {}
        self._graph_input_count = int(graph_input_count)
        self._is_graph_capturing = False
        self._graph_capture_inputs: list[int] = []
        self._cum_registered_count = 0
        self._push_launchers = {}
        self._registered_launchers = {}
        self._unregistered_launchers = {}
        self._shot_cache: dict[tuple[int, AllReduceAlgo], int] = {}
        self._closed = False
        self.disabled = True

        # MUSA push uses two staging epochs and one slot per source rank.
        self._push_storage_bytes = jit_ar.push_buffer_bytes(
            self._push_buffer_bytes, self.world_size
        )
        ar_signal_bytes = jit_ar.allreduce_v2_meta_size()
        fused_signal_bytes = jit_ar.allreduce_v2_meta_size()
        # Match CUDA v2's controller layout: push counters are independent,
        # while one-shot and two-shot share one pull semaphore protocol. The
        # fused protocol gets a disjoint signal/scratch region in this same IPC
        # allocation because its barrier generations have a different lifetime.
        self._push_signal_offset = 0
        self._pull_signal_offset = _align(self._push_signal_offset + ar_signal_bytes)
        # Registered two-shot and fused kernels address scratch as
        # ``signal + 1``. Keep both reservations disjoint from each other and
        # from staging data.
        self._scratch_bytes = self._pull_buffer_bytes
        self._fused_signal_offset = _align(
            self._pull_signal_offset + ar_signal_bytes + self._scratch_bytes
        )
        self._pull_buffer_offset = _align(
            self._fused_signal_offset + fused_signal_bytes + self._scratch_bytes
        )
        # CUDA v2 keeps pull payload and push sentinel-ring storage disjoint.
        # Pull writes are arbitrary nonzero values and would otherwise look
        # like already-published push packets after an algorithm transition.
        self._push_buffer_offset = _align(
            self._pull_buffer_offset + self._pull_buffer_bytes
        )
        storage_bytes = self._push_buffer_offset + self._push_storage_bytes

        # This module is MUSA-only.  Keep the runtime-neutral current-device
        # lookup, but use the MUSA device type so allocations do not dispatch
        # to CUDA's ``aten::empty`` backend.
        device = torch.device("musa", torch.get_device_module().current_device())
        self._device = device
        self._runtime = CudaRTLibrary()
        # share_storage() has no offset field, so this must be a runtime
        # allocation whose data pointer is also its IPC allocation base.  A
        # caching-allocator tensor may be a suballocation and is not valid for
        # this ABI.
        self._storage_allocation = self._runtime.cudaMalloc(storage_bytes)
        self._runtime.cudaMemset(self._storage_allocation, 0, storage_bytes)
        self._graph_rank_data = torch.zeros(
            (self._graph_input_count, 8), dtype=torch.int64, device=device
        )
        self._storage_ptr = int(self._storage_allocation.value)
        self._self_push_signal_ptr = self._storage_ptr + self._push_signal_offset
        self._self_pull_signal_ptr = self._storage_ptr + self._pull_signal_offset
        self._self_fused_signal_ptr = self._storage_ptr + self._fused_signal_offset
        self._self_buffer_ptr = self._storage_ptr + self._pull_buffer_offset
        self._self_push_buffer_ptr = self._storage_ptr + self._push_buffer_offset
        self._rank_data_cpu: torch.Tensor | None = None
        self._rank_data_device: torch.Tensor | None = None
        self._push_rank_data_device: torch.Tensor | None = None
        self._push_signal_ptrs_cpu: torch.Tensor | None = None
        self._pull_signal_ptrs_cpu: torch.Tensor | None = None
        self._pull_config = ConfigResult(0, 0)
        # Fused RMSNorm shares the v2 pull workspace and owns only its protocol
        # state inside the same allocation. Its capacity therefore follows the
        # pull workspace even if a caller configures a larger push workspace.
        self.max_size = self._pull_buffer_bytes
        self._fused_signal_ptrs_cpu: torch.Tensor | None = None
        self._fused_signal_ptrs: list[int] = []
        self._pull_buffer_ptrs: list[int] = []
        self._fused_rmsnorm = MusaCustomAllReduceV2RMSNorm(self)
        jit_ar.ensure_compiled_v2(self.world_size, jit_ar.SHOT_ONE_STAGE)

    def share_storage(self) -> list[int]:
        handle = self._runtime.cudaIpcGetMemHandle(ctypes.c_void_p(self._storage_ptr))
        return _handle_to_list(handle)

    def post_init(self, handles: list[list[int]]) -> None:
        if len(handles) != self.world_size:
            raise ValueError(
                f"Expected {self.world_size} storage handles, got {len(handles)}"
            )
        storage_ptrs: list[int] = []
        for peer_rank, values in enumerate(handles):
            if peer_rank == self.rank:
                storage_ptrs.append(self._storage_ptr)
                continue
            key = bytes(int(value) & 0xFF for value in values)
            opened = self._opened_ipc_ptrs.get(key)
            if opened is None:
                opened = int(
                    self._runtime.cudaIpcOpenMemHandle(_handle_from_list(values)).value
                )
                self._opened_ipc_ptrs[key] = opened
            storage_ptrs.append(opened)

        buffer_ptrs = [ptr + self._pull_buffer_offset for ptr in storage_ptrs]
        push_buffer_ptrs = [ptr + self._push_buffer_offset for ptr in storage_ptrs]
        buffer_ptrs.extend([0] * (8 - self.world_size))
        push_buffer_ptrs.extend([0] * (8 - self.world_size))
        push_signal_ptrs = [ptr + self._push_signal_offset for ptr in storage_ptrs]
        pull_signal_ptrs = [ptr + self._pull_signal_offset for ptr in storage_ptrs]
        fused_signal_ptrs = [ptr + self._fused_signal_offset for ptr in storage_ptrs]
        self._rank_data_cpu = torch.tensor(buffer_ptrs, dtype=torch.int64)
        self._rank_data_device = torch.tensor(
            buffer_ptrs, dtype=torch.int64, device=self._device
        )
        # Polling push consumes a stable device descriptor. Keeping this in
        # the communicator workspace avoids passing RankData by value to every
        # kernel (64 bytes/thread of private spill on MP31).
        self._push_rank_data_device = torch.tensor(
            push_buffer_ptrs, dtype=torch.int64, device=self._device
        )
        self._push_signal_ptrs_cpu = torch.tensor(push_signal_ptrs, dtype=torch.int64)
        self._pull_signal_ptrs_cpu = torch.tensor(pull_signal_ptrs, dtype=torch.int64)
        self._fused_signal_ptrs_cpu = torch.tensor(fused_signal_ptrs, dtype=torch.int64)
        self._fused_signal_ptrs = fused_signal_ptrs
        self._pull_buffer_ptrs = buffer_ptrs[: self.world_size]
        self.disabled = False

    def set_cuda_graph_capture(self, is_capturing: bool) -> None:
        self._is_graph_capturing = bool(is_capturing)

    def _allocation_base_and_offset(self, ptr: int) -> tuple[int, int]:
        lib = ctypes.CDLL("libmusa.so")
        get_attribute = lib.muPointerGetAttribute
        get_attribute.restype = ctypes.c_int
        get_attribute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_ulonglong,
        ]
        base_ptr = ctypes.c_void_p()
        error = get_attribute(
            ctypes.byref(base_ptr),
            11,  # MU_POINTER_ATTRIBUTE_RANGE_START_ADDR
            ctypes.c_ulonglong(ptr),
        )
        if error != 0 or base_ptr.value is None:
            raise RuntimeError(f"muPointerGetAttribute failed: {error}")
        base = int(base_ptr.value)
        return base, ptr - base

    def share_graph_inputs(self):
        result = []
        handle_cache: dict[int, list[int]] = {}
        for ptr in self._graph_capture_inputs[self._cum_registered_count :]:
            base, offset = self._allocation_base_and_offset(ptr)
            handle = handle_cache.get(base)
            if handle is None:
                handle = _handle_to_list(
                    self._runtime.cudaIpcGetMemHandle(ctypes.c_void_p(base))
                )
                handle_cache[base] = handle
            result.append((offset, handle))
        return result

    def register_inputs(self, rank_inputs) -> None:
        if len(rank_inputs) != self.world_size:
            raise ValueError(
                f"Expected graph inputs from {self.world_size} ranks, "
                f"got {len(rank_inputs)}"
            )
        count = len(self._graph_capture_inputs) - self._cum_registered_count
        if count == 0:
            return
        if any(len(inputs) != count for inputs in rank_inputs):
            raise ValueError("MUSA custom AR v2 graph input counts differ across ranks")

        data = (_AllReduceData * count)()
        for index in range(count):
            row = data[index].input
            for peer_rank, inputs in enumerate(rank_inputs):
                offset, handle_values = inputs[index]
                if peer_rank == self.rank:
                    ptr = self._graph_capture_inputs[self._cum_registered_count + index]
                else:
                    key = bytes(int(value) & 0xFF for value in handle_values)
                    base = self._opened_ipc_ptrs.get(key)
                    if base is None:
                        base = int(
                            self._runtime.cudaIpcOpenMemHandle(
                                _handle_from_list(handle_values)
                            ).value
                        )
                        self._opened_ipc_ptrs[key] = base
                    ptr = base + int(offset)
                row[peer_rank] = ctypes.c_void_p(ptr)

        begin = self._cum_registered_count
        end = begin + count
        # Match CUDA's native register_graph_buffers contract.  The rank
        # table is a tiny host-to-device descriptor update; using a
        # synchronous runtime memcpy makes this copy complete before the
        # next graph replay without a device-wide synchronize.
        dst = ctypes.c_void_p(
            self._graph_rank_data.data_ptr() + begin * ctypes.sizeof(_AllReduceData)
        )
        self._runtime.cudaMemcpy(
            dst,
            ctypes.cast(data, ctypes.c_void_p),
            ctypes.sizeof(data),
            cudaMemcpyHostToDevice,
        )
        self._cum_registered_count = end

    def _check_ready(self) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._rank_data_cpu is None
            or self._rank_data_device is None
            or self._push_rank_data_device is None
            or self._push_signal_ptrs_cpu is None
            or self._pull_signal_ptrs_cpu is None
        ):
            raise RuntimeError("MUSA custom all-reduce v2 post_init() was not called")
        return self._rank_data_cpu, self._pull_signal_ptrs_cpu

    def _select_shot(self, nbytes: int, algo: AllReduceAlgo) -> int:
        if not isinstance(algo, AllReduceAlgo):
            raise TypeError(f"Unsupported custom all-reduce v2 algorithm: {algo}")
        key = (int(nbytes), algo)
        cached = self._shot_cache.get(key)
        if cached is not None:
            return cached

        if jit_ar.is_shot_forced():
            shot = jit_ar.preferred_shot(self.world_size, nbytes)
        elif algo == AllReduceAlgo.ONE_SHOT_PUSH:
            shot = jit_ar.SHOT_PUSH
        elif algo == AllReduceAlgo.ONE_SHOT_PULL:
            shot = jit_ar.SHOT_ONE_STAGE
        else:
            # SHOT_TWO_STAGE_512 is a launch-shape/kernel variant of the v2
            # TWO_SHOT_PULL algorithm, not a separate algorithm selector.
            if self.world_size == 8 or (self.world_size == 4 and nbytes <= 512 * 1024):
                shot = jit_ar.SHOT_TWO_STAGE_512
            else:
                shot = jit_ar.SHOT_TWO_STAGE
        is_push = shot in (jit_ar.SHOT_PUSH, jit_ar.SHOT_PUSH_WIDE)
        required = (
            jit_ar.push_buffer_bytes(nbytes, self.world_size) if is_push else nbytes
        )
        # Pull payload and push ring have independent capacities and storage.
        capacity = self._push_storage_bytes if is_push else self._pull_buffer_bytes
        if required > capacity:
            raise RuntimeError(
                f"MUSA custom AR v2 algorithm {algo.name} needs {required} bytes, "
                f"but its workspace has {capacity}"
            )
        self._shot_cache[key] = shot
        return shot

    def all_reduce(self, input: torch.Tensor, algo: AllReduceAlgo) -> torch.Tensor:
        rank_data, signal_ptrs = self._check_ready()
        if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(f"Unsupported MUSA custom AR dtype: {input.dtype}")
        if input.device != self._device:
            raise ValueError(
                f"MUSA custom AR v2 expected input on {self._device}, "
                f"got {input.device}"
            )
        if not input.is_contiguous():
            raise ValueError("MUSA custom AR v2 input must be contiguous")
        nbytes = input.numel() * input.element_size()
        if nbytes % 16 != 0:
            raise ValueError(
                f"MUSA custom AR v2 input bytes must be a multiple of 16: {nbytes}"
            )
        if input.data_ptr() % 16 != 0:
            raise ValueError("MUSA custom AR v2 input pointer must be 16-byte aligned")
        if input.numel() > (1 << 31) - 1:
            raise ValueError("MUSA custom AR v2 input exceeds its int32 item limit")
        if (
            self._is_graph_capturing
            and self.world_size == 4
            and 256 * 1024 < nbytes <= 512 * 1024
        ):
            # TP4 graph replay uses the registered two-shot-512 kernel only
            # for the middle payload range; smaller payloads stay one-shot
            # pull to avoid forcing a slower two-shot decode path.
            algo = AllReduceAlgo.TWO_SHOT_PULL
        shot = self._select_shot(nbytes, algo)
        graph_launch = (
            self._is_graph_capturing
            and torch.get_device_module().is_current_stream_capturing()
        )
        # Match CUDA v2: registered one-shot pull writes a distinct output,
        # while registered two-shot broadcasts reduced shards back into each
        # rank's graph input and therefore remains in-place.
        output = (
            torch.empty_like(input)
            if graph_launch and shot == jit_ar.SHOT_ONE_STAGE
            else input
        )
        if shot in (jit_ar.SHOT_PUSH, jit_ar.SHOT_PUSH_WIDE):
            assert self._push_signal_ptrs_cpu is not None
            launch_signal_ptrs = self._push_signal_ptrs_cpu
            self_signal_ptr = self._self_push_signal_ptr
            self_buffer_ptr = self._self_push_buffer_ptr
        else:
            launch_signal_ptrs = signal_ptrs
            self_signal_ptr = self._self_pull_signal_ptr
            self_buffer_ptr = self._self_buffer_ptr

        if shot in (jit_ar.SHOT_PUSH, jit_ar.SHOT_PUSH_WIDE):
            assert self._push_rank_data_device is not None
            launcher = self._push_launchers.get(shot)
            if launcher is None:
                launcher = jit_ar.launch_device_descriptor_func(self.world_size, shot)
                self._push_launchers[shot] = launcher
            launcher(
                self._push_rank_data_device,
                launch_signal_ptrs,
                output,
                self_signal_ptr,
                self.rank,
                self.world_size,
                shot,
            )
        elif graph_launch:
            index = len(self._graph_capture_inputs)
            if index >= self._graph_input_count:
                raise RuntimeError(
                    "MUSA custom AR v2 graph pointer table overflow: "
                    f"{index} >= {self._graph_input_count}"
                )
            self._graph_capture_inputs.append(int(input.data_ptr()))
            launcher = self._registered_launchers.get(shot)
            if launcher is None:
                launcher = jit_ar.launch_registered_pybind_func(self.world_size, shot)
                self._registered_launchers[shot] = launcher
            launcher(
                self._graph_rank_data[index],
                launch_signal_ptrs,
                output,
                self_signal_ptr,
                self.rank,
                self.world_size,
                shot,
            )
        else:
            launcher = self._unregistered_launchers.get(shot)
            if launcher is None:
                launcher = jit_ar.launch_unregistered_pybind_func(self.world_size, shot)
                self._unregistered_launchers[shot] = launcher
            launcher(
                rank_data,
                launch_signal_ptrs,
                input,
                output,
                self_signal_ptr,
                self_buffer_ptr,
                self._push_buffer_bytes,
                self.rank,
                self.world_size,
                shot,
            )
        return output

    def fused_allreduce_rmsnorm(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        *,
        norm_out: torch.Tensor | None = None,
        residual_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """FlashInfer-compatible AR + residual + RMSNorm operator ABI."""
        return self._fused_rmsnorm.fused_allreduce_rmsnorm(
            input,
            residual,
            weight,
            eps,
            norm_out=norm_out,
            residual_out=residual_out,
        )

    def fused_allreduce_gemma_dual_rmsnorm(
        self,
        input_: torch.Tensor,
        residual_: torch.Tensor,
        post_weight_: torch.Tensor,
        next_weight_: torch.Tensor,
        post_eps: float,
        next_eps: float,
        *,
        attention: bool,
        scalar_: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self._fused_rmsnorm.fused_allreduce_gemma_dual_rmsnorm(
            input_,
            residual_,
            post_weight_,
            next_weight_,
            post_eps,
            next_eps,
            attention=attention,
            scalar_=scalar_,
        )

    def config_pull(self, num_blocks: int = -1, num_threads: int = -1) -> ConfigResult:
        old = self._pull_config
        self._pull_config = ConfigResult(
            old.num_blocks if num_blocks == -1 else num_blocks,
            old.num_threads if num_threads == -1 else num_threads,
        )
        return old

    def free(self, group: torch.distributed.ProcessGroup) -> None:
        if self._closed:
            return
        if dist.is_initialized():
            dist.barrier(group=group)
        for ptr in self._opened_ipc_ptrs.values():
            self._runtime.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr))
        self._opened_ipc_ptrs.clear()
        self._runtime.cudaFree(self._storage_allocation)
        self._rank_data_device = None
        self._push_rank_data_device = None
        self._graph_rank_data = None
        self._push_launchers.clear()
        self._registered_launchers.clear()
        self._unregistered_launchers.clear()
        self._shot_cache.clear()
        self.disabled = True
        self._closed = True


def get_musa_custom_all_reduce_cls():
    return MusaCustomAllReduceV2Obj


# Upstream's MUSA default pull workspace (the ``_init_config`` path).
_DEFAULT_PULL_BYTES = 512 * 1024 * 1024
# Upstream hard-codes this for the graph input pointer table.
_GRAPH_INPUT_COUNT = 131072


def _is_torch_compiling() -> bool:
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:
        try:
            return bool(torch._dynamo.is_compiling())
        except Exception:
            return False


def _is_current_stream_capturing() -> bool:
    for module_name in ("musa", "cuda"):
        module = getattr(torch, module_name, None)
        check = getattr(module, "is_current_stream_capturing", None)
        if check is None:
            continue
        try:
            return bool(check())
        except Exception:
            continue
    return False


def _is_weak_contiguous(tensor: torch.Tensor) -> bool:
    """Same contract as upstream's ``is_weak_contiguous``."""
    if tensor.is_contiguous():
        return True
    storage = tensor.untyped_storage()
    return tensor.storage_offset() + tensor.numel() == (
        storage.nbytes() // tensor.element_size()
    )


class _MusaCustomAllreduceV2Impl:
    """v2 all-reduce with the interface ``MusaJitCustomAllreduce`` delegates to."""

    _MAX_CAR_SIZE = _DEFAULT_PULL_BYTES

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size: int = _DEFAULT_PULL_BYTES,
    ) -> None:
        self.disabled = True
        self.group = group
        self.device = device
        self._IS_CAPTURING = False
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)

        mode_config = MusaCustomAllReduceV2Obj.mode_configs.get(self.world_size)
        if mode_config is None:
            raise RuntimeError(
                "MUSA custom all-reduce v2 supports world sizes "
                f"{sorted(MusaCustomAllReduceV2Obj.mode_configs)} only, "
                f"got {self.world_size}"
            )
        self._push_threshold, self._pull_threshold = (
            int(mode_config[0]),
            int(mode_config[1]),
        )
        # Upstream sizes the push ring from the push threshold, which is 0 for
        # the world sizes here; push needs an explicit override to run.
        self.max_push_size = self._push_threshold
        self.max_pull_size = int(max_size)
        self.max_size = max(self.max_pull_size, self.max_push_size)
        self._algo_override: AllReduceAlgo | None = None

        self.obj = MusaCustomAllReduceV2Obj(
            rank=self.rank,
            world_size=self.world_size,
            pull_buffer_bytes=self.max_pull_size,
            push_buffer_bytes=self.max_push_size,
            graph_input_count=_GRAPH_INPUT_COUNT,
        )
        handles: list[Any] = [None] * self.world_size
        dist.all_gather_object(handles, self.obj.share_storage(), group=group)
        self.obj.post_init(handles)
        # Reuse the existing CAR custom op: ``vllm.musa_jit_custom_all_reduce``
        # calls back into ``_custom_all_reduce_impl`` through a comm_id, so
        # Dynamo only sees one opaque operator and never traces data_ptr() or
        # ctypes, which it cannot compile.
        self._comm_id = _register_comm(self)
        self.disabled = False
        logger.info_once(
            "MUSA custom all-reduce v2 enabled: world_size=%d pull_bytes=%d "
            "push_bytes=%d push_threshold=%d pull_threshold=%d "
            "(payloads above the push threshold take ONE_SHOT_PULL)",
            self.world_size,
            self.max_pull_size,
            self.max_push_size,
            self._push_threshold,
            self._pull_threshold,
        )

    # -- algorithm selection -------------------------------------------------
    def override_algo(self, algo: Optional[AllReduceAlgo]) -> None:
        """Explicit algorithm override, mirroring upstream's ``override_algo``."""
        self._algo_override = algo

    def _pick_algo(self, nbytes: int) -> AllReduceAlgo:
        push_threshold = min(self._push_threshold, self.max_push_size)
        pull_threshold = min(self._pull_threshold, self.max_pull_size)
        if self._algo_override is not None:
            algo = self._algo_override
        elif nbytes <= push_threshold:
            algo = AllReduceAlgo.ONE_SHOT_PUSH
        elif nbytes <= pull_threshold:
            algo = AllReduceAlgo.ONE_SHOT_PULL
        else:
            algo = AllReduceAlgo.TWO_SHOT_PULL
        car_trace.record("algo", "v2", algo.name.lower(), nbytes)
        return algo

    # -- public surface used by MusaJitCustomAllreduce -----------------------
    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        use = self._should_custom_ar_gates(inp)
        car_trace.record(
            "gate",
            "v2",
            "car" if use else "fallback",
            inp.numel() * inp.element_size(),
        )
        return use

    def _should_custom_ar_gates(self, inp: torch.Tensor) -> bool:
        """Rule A, kept identical to the v1 gate set.

        v1 additionally carries DeepSeek-V4-only graph guards; those are inert
        for the models measured here, so both arms accept the same calls.
        """
        if self.disabled:
            return False
        if inp.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0 or inp_size > self.max_size:
            return False
        return inp.is_contiguous()

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor:
        if self._IS_CAPTURING:
            if not _is_current_stream_capturing():
                # Graph warmup must not really communicate, matching the
                # v1/vLLM warmup semantics.
                return torch.empty_like(input)
            return torch.ops.vllm.musa_jit_custom_all_reduce(input, self._comm_id)
        if _is_torch_compiling():
            # Show Dynamo a registered operator in the graph rather than the
            # Python/ctypes implementation.
            return torch.ops.vllm.musa_jit_custom_all_reduce(input, self._comm_id)
        return self._custom_all_reduce_impl(input)

    def _custom_all_reduce_impl(self, input: torch.Tensor) -> torch.Tensor:
        nbytes = input.numel() * input.element_size()
        algo = self._pick_algo(nbytes)
        output = self.obj.all_reduce(input, algo)
        return output if isinstance(output, torch.Tensor) else torch.from_dlpack(output)

    @contextmanager
    def capture(self):
        if self.disabled:
            yield
            return
        self._IS_CAPTURING = True
        try:
            self.obj.set_cuda_graph_capture(True)
            yield
        finally:
            self.obj.set_cuda_graph_capture(False)
            self._IS_CAPTURING = False
        if _is_current_stream_capturing():
            raise RuntimeError(
                "Cannot register MUSA custom AR v2 graph inputs while capturing"
            )
        pairs = self.obj.share_graph_inputs()
        handles = [handle for _, handle in pairs]
        offsets = [offset for offset, _ in pairs]
        handles_all = self._share_list(handles)
        offsets_all = self._share_list(offsets)
        self.obj.register_inputs(
            [list(zip(o, h)) for o, h in zip(offsets_all, handles_all)]
        )
        logger.info("Registering %d MUSA custom AR v2 graph addresses", len(pairs))

    def _share_list(self, values: list[int]) -> list[list[int]]:
        tensor = torch.tensor(values, dtype=torch.int64, device="cpu")
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor, group=self.group)
        return [g.tolist() for g in gathered]

    def close(self) -> None:
        if not getattr(self, "disabled", True) and hasattr(self, "obj"):
            self.obj.free(self.group)

    # -- features the ported kernel surface does not provide -----------------
    # Fused AR+RMSNorm v2 exists upstream, but its kernel path is not part of
    # this port; running the v1 fused kernels here would mix barrier protocols.
    def should_fused_allreduce_rmsnorm(
        self, inp: torch.Tensor, weight: torch.Tensor
    ) -> bool:
        return False

    def should_fused_allreduce_residual_rmsnorm(
        self, inp: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> bool:
        return False

    def _reject_fused_allreduce_residual_rmsnorm_reason(
        self, inp: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> str:
        return "fused AR+RMSNorm is not implemented for the v2 kernel path"

    def fused_allreduce_rmsnorm(
        self, input: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        raise NotImplementedError(
            "fused all-reduce + RMSNorm is not implemented for MUSA custom AR v2"
        )

    def fused_allreduce_residual_rmsnorm(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        raise NotImplementedError(
            "fused all-reduce + residual + RMSNorm is not implemented for "
            "MUSA custom AR v2"
        )

    def fused_allreduce_residual_rmsnorm_no_raw(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        raise NotImplementedError(
            "fused all-reduce + residual + RMSNorm is not implemented for "
            "MUSA custom AR v2"
        )

    def should_custom_all_gather(
        self,
        input: torch.Tensor,
        dim: int = -1,
        output_dtype: torch.dtype | None = None,
    ) -> bool:
        return False

    def custom_all_gather(
        self,
        input: torch.Tensor,
        dim: int = -1,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        raise NotImplementedError(
            "custom all-gather is not implemented for MUSA custom AR v2"
        )


class MusaCustomAllReduceV2RMSNorm:
    """v2 backend for fused AR+RMSNorm (not ported; calling it raises)."""

    def __init__(self, owner) -> None:
        self._owner = owner

    def fused_allreduce_rmsnorm(self, *args, **kwargs):
        raise NotImplementedError(
            "v2 fused AR+residual+RMSNorm is not ported yet (it needs "
            "custom_all_reduce_rmsnorm.mu + custom_all_reduce_rmsnorm.py, which "
            "are planned for a later stage); this stage supports pure "
            "all-reduce only"
        )

    def fused_allreduce_gemma_dual_rmsnorm(self, *args, **kwargs):
        raise NotImplementedError(
            "v2 fused Gemma dual RMSNorm is not ported yet (planned for a "
            "later stage)"
        )
