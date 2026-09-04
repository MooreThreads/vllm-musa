# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared primitives for hardware-aware kernel tactic selection.

The platform publishes the resolved graph profile before worker spawn, workers
freeze the physical MP identity, and serving selectors use exact
hardware/shape/graph keys with fallback on a miss. Keep runtime selection
deterministic and cheap. Offline benchmarks may add exact entries to a tactic
catalog, but the serving hot path must only query a stable hardware identity
and perform integer cost calculations. The wave estimator at the end of this
module is an offline diagnostic and does not select production kernels.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# Minimum rows for the qualified MUSA JIT fused-add RMSNorm profile on
# contiguous BF16 H5120 workloads.
FUSED_ADD_RMSNORM_MIN_ROWS = 64
_ENGINE_MAX_NUM_SEQS_ENV = "_VLLM_MUSA_ENGINE_MAX_NUM_SEQS"


@dataclass(frozen=True)
class MusaForwardGraphBucket:
    """Validated projection of vLLM's graph dispatch descriptor.

    ``present`` distinguishes a missing context (where a legacy static profile
    may still be safe) from a present but ineligible descriptor (which must
    fail closed).
    """

    num_tokens: int | None = None
    num_reqs: int | None = None
    uniform: bool | None = None
    runtime_mode: str | None = None
    has_lora: bool | None = None
    num_active_loras: int | None = None
    present: bool = False

    @classmethod
    def invalid(cls) -> MusaForwardGraphBucket:
        return cls(present=True)


def configure_musa_engine_scheduler_profile(max_num_seqs: int | None) -> None:
    """Publish the engine-static scheduler profile to spawned workers.

    This is written only by the MUSA platform from the resolved VllmConfig; it
    is not a user-facing override. Environment transport is used because vLLM
    spawns worker processes after platform config validation.
    """
    if isinstance(max_num_seqs, int) and max_num_seqs > 0:
        os.environ[_ENGINE_MAX_NUM_SEQS_ENV] = str(max_num_seqs)
    else:
        os.environ.pop(_ENGINE_MAX_NUM_SEQS_ENV, None)


def configure_musa_engine_scheduler_profile_from_config(vllm_config: Any) -> None:
    """Publish the resolved scheduler profile from a vLLM config object.

    A missing scheduler config or profile deliberately clears the inherited
    value so hardware selectors fail closed.
    """
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    configure_musa_engine_scheduler_profile(
        getattr(scheduler_config, "max_num_seqs", None)
    )


def query_musa_forward_graph_bucket() -> MusaForwardGraphBucket | None:
    """Read vLLM's graph-static batch key and fail closed if unavailable."""
    try:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if not is_forward_context_available():
            return None
        context = get_forward_context()
        descriptor = context.batch_descriptor
        if descriptor is None:
            return MusaForwardGraphBucket.invalid()
        num_tokens = descriptor.num_tokens
        num_reqs = descriptor.num_reqs
        uniform = descriptor.uniform
        runtime_mode = getattr(context.cudagraph_runtime_mode, "name", None)
        has_lora = descriptor.has_lora
        num_active_loras = descriptor.num_active_loras
        if (
            type(num_tokens) is not int
            or num_tokens <= 0
            or (num_reqs is not None and (type(num_reqs) is not int or num_reqs <= 0))
            or type(uniform) is not bool
            or type(runtime_mode) is not str
            or runtime_mode not in {"NONE", "PIECEWISE", "FULL"}
            or type(has_lora) is not bool
            or type(num_active_loras) is not int
            or num_active_loras < 0
        ):
            return MusaForwardGraphBucket.invalid()
        return MusaForwardGraphBucket(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            uniform=uniform,
            runtime_mode=runtime_mode,
            has_lora=has_lora,
            num_active_loras=num_active_loras,
            present=True,
        )
    except ImportError:
        # A missing ForwardContext API is dependency drift, not an absent
        # context.  Mark it present-but-invalid so engine-static profiles
        # cannot bypass the descriptor, runtime-mode, and LoRA safety checks.
        return MusaForwardGraphBucket.invalid()
    except (
        AssertionError,
        AttributeError,
        LookupError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return MusaForwardGraphBucket.invalid()


def query_musa_engine_max_num_seqs() -> int | None:
    value = os.environ.get(_ENGINE_MAX_NUM_SEQS_ENV)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


@dataclass(frozen=True, slots=True)
class MusaKernelHardware:
    """Device fields that can change a kernel's optimal launch tactic."""

    device_capability: tuple[int, int]
    multiprocessor_count: int

    @property
    def is_known(self) -> bool:
        major, minor = self.device_capability
        return major >= 0 and minor >= 0 and self.multiprocessor_count > 0

    @property
    def cache_key(self) -> str:
        """Stable key fragment for AOT maps and JIT/timing caches."""
        if not self.is_known:
            return "unknown"
        major, minor = self.device_capability
        return f"mp{major}{minor}-mps{self.multiprocessor_count}"


@dataclass(frozen=True, slots=True)
class MusaMhcWeightedRmsnormTactic:
    """Exact JIT launch geometry for one weighted-RMSNorm contract."""

    threads: int
    source: str


@dataclass(frozen=True, slots=True)
class MusaJitRMSNormTactic:
    """Exact launch geometry for one native JIT RMSNorm contract."""

    block_threads: int
    source: str


UNKNOWN_MUSA_KERNEL_HARDWARE = MusaKernelHardware(
    device_capability=(-1, -1),
    multiprocessor_count=-1,
)
_MUSA_KERNEL_HARDWARE_CACHE: dict[int, MusaKernelHardware] = {}
_FROZEN_MUSA_KERNEL_HARDWARE: dict[int, MusaKernelHardware] = {}
_MHC_WEIGHTED_RMSNORM_TACTICS: dict[
    tuple[tuple[int, int], int, int, int, str, str],
    MusaMhcWeightedRmsnormTactic,
] = {
    (
        (3, 1),
        48,
        5,
        4096,
        "torch.bfloat16",
        "torch.bfloat16",
    ): MusaMhcWeightedRmsnormTactic(
        threads=256,
        source="mp48-dsv4-mtp4-m5-wrms-256t-v1",
    )
}
_JIT_RMSNORM_TACTICS: dict[
    tuple[tuple[int, int], int, str, int, int, str, str],
    MusaJitRMSNormTactic,
] = {
    (
        (3, 1),
        60,
        mode,
        rows,
        5120,
        "torch.bfloat16",
        "torch.bfloat16",
    ): MusaJitRMSNormTactic(
        block_threads=320,
        source=("mp60-jit-rmsnorm-h5120-320t-" f"{mode}-m{rows}-v1"),
    )
    for mode, rows in (
        ("plain", 512),
        ("gemma", 512),
        ("gemma", 4096),
        ("fused_gemma", 512),
    )
} | {
    (
        (3, 1),
        56,
        mode,
        512,
        5120,
        "torch.bfloat16",
        "torch.bfloat16",
    ): MusaJitRMSNormTactic(
        block_threads=320,
        source=("mp56-jit-rmsnorm-h5120-320t-" f"{mode}-m512-v1"),
    )
    for mode in ("plain", "gemma", "fused", "fused_gemma")
}
_JIT_RMSNORM_TACTICS[
    (
        (3, 1),
        48,
        "plain",
        512,
        5120,
        "torch.bfloat16",
        "torch.bfloat16",
    )
] = MusaJitRMSNormTactic(
    block_threads=256,
    source="mp48-jit-rmsnorm-h5120-256t-plain-m512-v1",
)


def query_musa_kernel_hardware(device_index: int) -> MusaKernelHardware:
    """Return the exact MUSA kernel identity, or an explicit unknown value.

    Do not substitute a common MP count on failure: doing so can select a
    tactic qualified for a different core-count bin. The query itself is not
    cached because a call made before device initialization may return unknown;
    long-lived consumers should cache a successful result at their own boundary.
    """
    try:
        properties = _get_musa_device_properties(device_index)
        return MusaKernelHardware(
            device_capability=(int(properties.major), int(properties.minor)),
            multiprocessor_count=int(properties.multi_processor_count),
        )
    except Exception:  # noqa: BLE001 - hardware probing must fail closed
        return UNKNOWN_MUSA_KERNEL_HARDWARE


def query_cached_musa_kernel_hardware(device_index: int) -> MusaKernelHardware:
    """Cache a known device identity while allowing an early probe to recover."""
    cached = _MUSA_KERNEL_HARDWARE_CACHE.get(device_index)
    if cached is not None:
        return cached

    hardware = query_musa_kernel_hardware(device_index)
    if hardware.is_known:
        _MUSA_KERNEL_HARDWARE_CACHE[device_index] = hardware
    return hardware


def prime_musa_kernel_hardware(device_index: int) -> MusaKernelHardware:
    """Freeze one hardware fingerprint for the lifetime of a worker process.

    Unlike the retryable query cache, an unknown result is intentionally
    frozen. A worker must not change tactics after compilation or graph capture
    merely because a later device-property query succeeds.
    """
    normalized_index = int(device_index)
    hardware = _FROZEN_MUSA_KERNEL_HARDWARE.get(normalized_index)
    if hardware is None:
        hardware = query_musa_kernel_hardware(normalized_index)
        _FROZEN_MUSA_KERNEL_HARDWARE[normalized_index] = hardware
    return hardware


def get_primed_musa_kernel_hardware(device_index: int) -> MusaKernelHardware:
    """Return the frozen fingerprint, or unknown before worker priming."""
    return _FROZEN_MUSA_KERNEL_HARDWARE.get(
        int(device_index), UNKNOWN_MUSA_KERNEL_HARDWARE
    )


def select_mhc_weighted_rmsnorm_tactic(
    *,
    hardware: MusaKernelHardware,
    rows: int,
    hidden_size: int,
    input_dtype: str,
    weight_dtype: str,
    contiguous: bool,
) -> MusaMhcWeightedRmsnormTactic | None:
    """Resolve an exact production JIT tactic, with legacy fallback on miss."""
    if not hardware.is_known or not contiguous:
        return None
    return _MHC_WEIGHTED_RMSNORM_TACTICS.get(
        (
            hardware.device_capability,
            hardware.multiprocessor_count,
            rows,
            hidden_size,
            input_dtype,
            weight_dtype,
        )
    )


def select_jit_rmsnorm_tactic(
    *,
    hardware: MusaKernelHardware,
    mode: str,
    rows: int,
    hidden_size: int,
    input_dtype: str,
    weight_dtype: str,
    contiguous: bool,
) -> MusaJitRMSNormTactic | None:
    """Resolve an exact native JIT RMSNorm tactic; never use a nearest MP."""
    if not hardware.is_known or not contiguous:
        return None
    return _JIT_RMSNORM_TACTICS.get(
        (
            hardware.device_capability,
            hardware.multiprocessor_count,
            mode,
            rows,
            hidden_size,
            input_dtype,
            weight_dtype,
        )
    )


def _get_musa_device_properties(device_index: int) -> Any:
    # Import lazily so offline tactic-map tooling does not require a torch/MUSA
    # runtime merely to use the integer wave model.
    import torch

    return torch.musa.get_device_properties(device_index)


@dataclass(frozen=True, slots=True)
class WaveQuantization:
    """Integer occupancy estimate for a tiled or persistent launch."""

    total_tiles: int
    resident_slots: int
    waves: int
    tail_tiles: int
    last_wave_utilization: float
    overall_slot_utilization: float


def estimate_wave_quantization(
    total_tiles: int,
    multiprocessor_count: int,
    *,
    blocks_per_multiprocessor: int = 1,
) -> WaveQuantization:
    """Estimate tail-wave loss for one launch geometry.

    This is a ranking signal, not a performance model.  Register pressure,
    memory traffic, tensor-core efficiency, and synchronization still require
    measurement on the exact hardware bin.
    """
    if total_tiles < 0:
        raise ValueError(f"total_tiles must be non-negative, got {total_tiles}")
    if multiprocessor_count <= 0:
        raise ValueError(
            "multiprocessor_count must be positive, " f"got {multiprocessor_count}"
        )
    if blocks_per_multiprocessor <= 0:
        raise ValueError(
            "blocks_per_multiprocessor must be positive, "
            f"got {blocks_per_multiprocessor}"
        )

    resident_slots = multiprocessor_count * blocks_per_multiprocessor
    if total_tiles == 0:
        return WaveQuantization(
            total_tiles=0,
            resident_slots=resident_slots,
            waves=0,
            tail_tiles=0,
            last_wave_utilization=0.0,
            overall_slot_utilization=0.0,
        )

    waves = (total_tiles + resident_slots - 1) // resident_slots
    tail_tiles = total_tiles - (waves - 1) * resident_slots
    return WaveQuantization(
        total_tiles=total_tiles,
        resident_slots=resident_slots,
        waves=waves,
        tail_tiles=tail_tiles,
        last_wave_utilization=tail_tiles / resident_slots,
        overall_slot_utilization=total_tiles / (waves * resident_slots),
    )
