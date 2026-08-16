# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stable value types shared by the MUSA fused-MoE dispatcher and planner."""

from dataclasses import dataclass
from enum import Enum


class MusaFusedMoeBackend(str, Enum):
    AUTO = "auto"
    GEMV = "gemv"
    GROUPED_GEMM = "grouped_gemm"
    UPSTREAM = "upstream"


@dataclass(frozen=True)
class MusaFusedMoeShape:
    """Static, per-rank properties that determine the kernel crossover."""

    device_capability: tuple[int, int]
    multiprocessor_count: int
    local_experts: int
    w1_output_size: int
    w2_input_size: int
    hidden_size: int
    top_k: int
    block_n: int
    block_k: int
    activation: str
    expert_parallel: bool
    hidden_dtype: str
    weight_dtype: str
    scale_dtype: str
    w1_scale_shape: tuple[int, ...]
    w2_scale_shape: tuple[int, ...]
    gemv_block: str
    graph_mode: str


@dataclass(frozen=True)
class MusaFusedMoeThresholds:
    gemv_max_tokens: int | None
    grouped_gemm_min_tokens: int | None
    source: str


@dataclass(frozen=True, slots=True)
class MusaFusedMoeTokenRange:
    min_tokens: int
    max_tokens: int
    backend: MusaFusedMoeBackend


@dataclass(frozen=True, slots=True)
class MusaFusedMoeRuntimePolicyReceipt:
    plan_id: str
    plan_fingerprint: str
    profile: str
    entry_count: int


@dataclass(frozen=True, slots=True)
class MusaFusedMoeDispatchSelection:
    """One auditable hot-path selection without mutable timing state."""

    backend: MusaFusedMoeBackend
    source: str
    policy_identity: str
    plan_id: str = ""
    plan_fingerprint: str = ""
    min_tokens: int | None = None
    max_tokens: int | None = None


__all__ = [
    "MusaFusedMoeBackend",
    "MusaFusedMoeDispatchSelection",
    "MusaFusedMoeRuntimePolicyReceipt",
    "MusaFusedMoeShape",
    "MusaFusedMoeThresholds",
    "MusaFusedMoeTokenRange",
]
