# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared primitives for hardware-aware kernel tactic selection.

Keep runtime selection deterministic and cheap.  Offline benchmarks may add
exact entries to a tactic catalog, but the serving hot path must only query a
stable hardware identity, perform integer cost calculations, and fall back for
unknown identities.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# Minimum rows where the MUSA JIT fused-add RMSNorm provider was measured to
# outperform the fallback for contiguous BF16 H5120 workloads on S5000.
FUSED_ADD_RMSNORM_MIN_ROWS = 64
_ENGINE_MAX_NUM_SEQS_ENV = "_VLLM_MUSA_ENGINE_MAX_NUM_SEQS"


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


def query_musa_forward_graph_bucket() -> tuple[int, int | None, bool] | None:
    """Read vLLM's graph-static batch key and fail closed if unavailable."""
    try:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if not is_forward_context_available():
            return None
        descriptor = get_forward_context().batch_descriptor
        if descriptor is None:
            return None
        num_tokens = descriptor.num_tokens
        num_reqs = descriptor.num_reqs
        uniform = descriptor.uniform
        if type(num_tokens) is not int or num_tokens <= 0:
            return None
        if num_reqs is not None and (type(num_reqs) is not int or num_reqs <= 0):
            return None
        if type(uniform) is not bool:
            return None
        return num_tokens, num_reqs, uniform
    except (
        AttributeError,
        ImportError,
        LookupError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return None


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


UNKNOWN_MUSA_KERNEL_HARDWARE = MusaKernelHardware(
    device_capability=(-1, -1),
    multiprocessor_count=-1,
)
_MUSA_KERNEL_HARDWARE_CACHE: dict[int, MusaKernelHardware] = {}


def query_musa_kernel_hardware(device_index: int) -> MusaKernelHardware:
    """Return the exact MUSA kernel identity, or an explicit unknown value.

    Do not substitute a common S5000 MP count on failure: doing so can select a
    tactic calibrated for a different core-count bin. The query itself is not
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
