# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared primitives for hardware-aware kernel tactic selection.

Keep runtime selection deterministic and cheap.  Offline benchmarks may add
exact entries to a tactic catalog, but the serving hot path must only query a
stable hardware identity, perform integer cost calculations, and fall back for
unknown identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Minimum rows where the MUSA JIT fused-add RMSNorm provider was measured to
# outperform the fallback for contiguous BF16 H5120 workloads on S5000.
FUSED_ADD_RMSNORM_MIN_ROWS = 64


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
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return UNKNOWN_MUSA_KERNEL_HARDWARE


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
