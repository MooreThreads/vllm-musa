from __future__ import annotations

import ctypes
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_musa.jit_kernel.csrc import allreduce as jit_ar
from vllm_musa.runtime_plan import (
    ModelFamily,
    RuntimeDecision,
    resolve_runtime_plan,
)
from vllm_musa.runtime_plan.policy import (
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

    plan = resolve_runtime_plan(vllm_config)
    decision = RuntimeDecision.DEEPSEEK_V4_CAR_GRAPH_INPUT_CAPTURE_GUARD
    if plan.model.family is not ModelFamily.DEEPSEEK_V4:
        return True
    if not plan.supports(decision):
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
    # Keep the safe baseline available for lightweight objects constructed by
    # backend probes/tests before __init__ has populated model-specific plans.
    # A missing plan must never accidentally select a graph staging path.
    _use_graph_staging_arena = False
    _use_graph_collective_fallback = False

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

    def _use_registered_graph_input(self, tensor: torch.Tensor) -> bool:
        return (
            self._use_graph_registered_inputs
            and self._graph_registered_input_eligible(tensor)
            and self._IS_CAPTURING
            and self._is_current_stream_capturing()
        )

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
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
        shot = jit_ar.preferred_shot(
            self.world_size, input.numel() * input.element_size()
        )
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
        self._jit_comm: _MusaJitCustomAllreduceImpl | None = None
        del symm_mem_enabled

        try:
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
                "Using MUSA JIT custom all-reduce for eager and eligible CUDA "
                "graph paths."
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
