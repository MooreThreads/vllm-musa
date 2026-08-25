# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model-agnostic MUSA fused-MoE backend selection.

The hot path must not benchmark, synchronize device data to the host, or key
off a model architecture name.  Offline S5000 sweeps populate exact shape
entries below; unknown shapes keep the established upstream backend.
"""

import os
from dataclasses import dataclass, replace
from enum import Enum
from typing import Final

MUSA_FUSED_MOE_DISPATCH_ENV: Final = "VLLM_MUSA_FUSED_MOE_DISPATCH"


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
    max_num_seqs: int | None = None


@dataclass(frozen=True)
class MusaFusedMoeThresholds:
    gemv_max_tokens: int | None
    grouped_gemm_min_tokens: int | None
    source: str


# Unknown shapes stay on the established upstream path until an exact S5000
# sweep is recorded.
_DEFAULT_THRESHOLDS: Final = MusaFusedMoeThresholds(
    gemv_max_tokens=None,
    grouped_gemm_min_tokens=None,
    source="uncalibrated-shape",
)


def _s5000_fp8_shape(
    *,
    local_experts: int,
    w1_output_size: int,
    w2_input_size: int,
    hidden_size: int,
    top_k: int,
    w1_scale_shape: tuple[int, ...],
    w2_scale_shape: tuple[int, ...],
    gemv_block: str,
    graph_mode: str,
) -> MusaFusedMoeShape:
    return MusaFusedMoeShape(
        device_capability=(3, 1),
        multiprocessor_count=60,
        local_experts=local_experts,
        w1_output_size=w1_output_size,
        w2_input_size=w2_input_size,
        hidden_size=hidden_size,
        top_k=top_k,
        block_n=128,
        block_k=128,
        activation="silu",
        expert_parallel=False,
        hidden_dtype="torch.bfloat16",
        weight_dtype="torch.float8_e4m3fn",
        scale_dtype="torch.float32",
        w1_scale_shape=w1_scale_shape,
        w2_scale_shape=w2_scale_shape,
        gemv_block=gemv_block,
        graph_mode=graph_mode,
    )


def _s5000_qwen35_bf16_decode_shape(
    *,
    multiprocessor_count: int,
    graph_mode: str,
    folded_shared_expert: bool,
    max_num_seqs: int | None = None,
) -> MusaFusedMoeShape:
    """TP4-local Qwen3.5/3.6 BF16 decode shape."""

    experts = 257 if folded_shared_expert else 256
    top_k = 9 if folded_shared_expert else 8

    return MusaFusedMoeShape(
        device_capability=(3, 1),
        multiprocessor_count=multiprocessor_count,
        local_experts=experts,
        w1_output_size=256,
        w2_input_size=128,
        hidden_size=2048,
        top_k=top_k,
        block_n=0,
        block_k=0,
        activation="silu",
        expert_parallel=False,
        hidden_dtype="torch.bfloat16",
        weight_dtype="torch.bfloat16",
        scale_dtype="none",
        w1_scale_shape=(),
        w2_scale_shape=(),
        gemv_block="auto",
        graph_mode=graph_mode,
        max_num_seqs=max_num_seqs,
    )


def _s5000_bf16_shape(
    *,
    local_experts: int,
    w1_output_size: int,
    w2_input_size: int,
    hidden_size: int,
    top_k: int,
    graph_mode: str,
) -> MusaFusedMoeShape:
    """Generic S5000 BF16 decode shape for independently calibrated models."""

    return MusaFusedMoeShape(
        device_capability=(3, 1),
        multiprocessor_count=60,
        local_experts=local_experts,
        w1_output_size=w1_output_size,
        w2_input_size=w2_input_size,
        hidden_size=hidden_size,
        top_k=top_k,
        block_n=0,
        block_k=0,
        activation="silu",
        expert_parallel=False,
        hidden_dtype="torch.bfloat16",
        weight_dtype="torch.bfloat16",
        scale_dtype="none",
        w1_scale_shape=(),
        w2_scale_shape=(),
        gemv_block="auto",
        graph_mode=graph_mode,
    )


def _thresholds(
    gemv_max_tokens: int | None,
    grouped_gemm_min_tokens: int | None,
    source: str,
) -> MusaFusedMoeThresholds:
    return MusaFusedMoeThresholds(
        gemv_max_tokens=gemv_max_tokens,
        grouped_gemm_min_tokens=grouped_gemm_min_tokens,
        source=source,
    )


# Exact entries are keyed by the actual per-rank kernel shape. These S5000
# entries use the worst boundary across balanced, unique-random, and hot routes
# with three independent seeds. Capture entries additionally passed eight
# bitwise-equal CUDAGraph replays. Unknown shapes remain on the established
# base path, which may itself select the large-M DeepGEMM prefill backend.
_CALIBRATED_THRESHOLDS: Final[dict[MusaFusedMoeShape, MusaFusedMoeThresholds]] = {
    _s5000_fp8_shape(
        local_experts=256,
        w1_output_size=256,
        w2_input_size=128,
        hidden_size=2048,
        top_k=8,
        w1_scale_shape=(256, 2, 16),
        w2_scale_shape=(256, 16, 1),
        gemv_block="auto",
        graph_mode=graph_mode,
    ): _thresholds(
        gemv_max_tokens=13,
        grouped_gemm_min_tokens=None,
        source=f"s5000-mp60-20260721-e256-n256-k2048-{graph_mode}-dense-v5",
    )
    for graph_mode in ("eager", "capture")
}
_CALIBRATED_THRESHOLDS.update(
    {
        _s5000_fp8_shape(
            local_experts=256,
            w1_output_size=512,
            w2_input_size=256,
            hidden_size=4096,
            top_k=6,
            w1_scale_shape=(256, 4, 32),
            w2_scale_shape=(256, 32, 2),
            gemv_block="32x8",
            graph_mode=graph_mode,
        ): _thresholds(
            gemv_max_tokens=5,
            grouped_gemm_min_tokens=None,
            source=f"s5000-mp60-20260721-e256-n512-k4096-{graph_mode}-block32-dense-v5",
        )
        for graph_mode in ("eager", "capture")
    }
)
_CALIBRATED_THRESHOLDS.update(
    {
        _s5000_qwen35_bf16_decode_shape(
            multiprocessor_count=56,
            graph_mode=graph_mode,
            folded_shared_expert=folded_shared_expert,
            max_num_seqs=max_num_seqs,
        ): _thresholds(
            # MP56 route sweep: hot routes regress at M=8 while balanced and
            # unique routes still win there. Keep the worst-route boundary
            # until a device-side route classifier is available. Register it
            # only for an engine whose graph-static max_num_seqs cannot cross
            # that boundary. Selecting from replay-time M alone can bake the
            # small-M arm into a graph reused by larger batches.
            gemv_max_tokens=4,
            grouped_gemm_min_tokens=None,
            source=(
                f"s5000-mp56-20260825-qwen35-bf16-e"
                f"{257 if folded_shared_expert else 256}-n256-k2048-v128-"
                f"{graph_mode}-maxseq{max_num_seqs}-route-worst-crossover-v2"
            ),
        )
        for graph_mode in ("eager", "capture")
        for folded_shared_expert in (False, True)
        for max_num_seqs in (1, 2, 4)
    }
)
_CALIBRATED_THRESHOLDS.update(
    {
        _s5000_fp8_shape(
            local_experts=256,
            w1_output_size=512,
            w2_input_size=256,
            hidden_size=4096,
            top_k=6,
            w1_scale_shape=(256, 4, 32),
            w2_scale_shape=(256, 32, 2),
            gemv_block="16x8",
            graph_mode=graph_mode,
        ): _thresholds(
            gemv_max_tokens=5,
            grouped_gemm_min_tokens=None,
            source=f"s5000-mp60-20260721-e256-n512-k4096-{graph_mode}-block16-dense-v5",
        )
        for graph_mode in ("eager", "capture")
    }
)
_CALIBRATED_THRESHOLDS.update(
    {
        _s5000_fp8_shape(
            local_experts=64,
            w1_output_size=2816,
            w2_input_size=1408,
            hidden_size=2048,
            top_k=6,
            w1_scale_shape=(64, 22, 16),
            w2_scale_shape=(64, 16, 11),
            gemv_block="auto",
            graph_mode=graph_mode,
        ): _thresholds(
            # Capture-mode A/B on S5000 shows a GEMV win at M=1, but a
            # regression at M=2/3.  Keep the wider eager boundary and use
            # GEMV only for the single-token decode case under capture.
            gemv_max_tokens=3 if graph_mode == "eager" else 1,
            # No production-safe grouped crossover is currently established
            # for this shape; retain upstream for the large-token regime.
            grouped_gemm_min_tokens=None,
            source=(
                f"s5000-mp60-20260721-e64-n2816-k2048-{graph_mode}-"
                f"{'m1-' if graph_mode == 'capture' else ''}serving-gated"
            ),
        )
        for graph_mode in ("eager", "capture")
    }
)
_CALIBRATED_THRESHOLDS.update(
    {
        _s5000_qwen35_bf16_decode_shape(
            multiprocessor_count=60,
            graph_mode=graph_mode,
            folded_shared_expert=folded_shared_expert,
        ): _thresholds(
            # Exact TP4-local Qwen crossover: M=1/2/4/8/12 are faster on
            # native GEMV, while M>=16 is not a stable production win.
            gemv_max_tokens=12,
            grouped_gemm_min_tokens=None,
            source=(
                f"s5000-mp60-20260806-qwen35-36-bf16-e"
                f"{257 if folded_shared_expert else 256}-"
                f"n256-k2048-v128-{graph_mode}-crossover-v1"
            ),
        )
        for graph_mode in ("eager", "capture")
        for folded_shared_expert in (False, True)
    }
)
_CALIBRATED_THRESHOLDS.update(
    {
        _s5000_bf16_shape(
            local_experts=257,
            w1_output_size=256,
            w2_input_size=128,
            hidden_size=3072,
            top_k=9,
            graph_mode=graph_mode,
        ): _thresholds(
            gemv_max_tokens=10,
            grouped_gemm_min_tokens=None,
            source=(
                f"s5000-mp60-20260806-qwen35-bf16-folded-"
                f"e257-n256-k3072-{graph_mode}"
            ),
        )
        for graph_mode in ("eager", "capture")
    }
)

_CALIBRATED_DIMENSIONS: Final = frozenset(
    (
        shape.local_experts,
        shape.w1_output_size,
        shape.w2_input_size,
        shape.hidden_size,
        shape.top_k,
    )
    for shape in _CALIBRATED_THRESHOLDS
)


def parse_dispatch_backend(value: str | None = None) -> MusaFusedMoeBackend:
    """Parse the generic force/rollback override; default is ``auto``."""

    raw_value = os.environ.get(MUSA_FUSED_MOE_DISPATCH_ENV, "auto")
    if value is not None:
        raw_value = value
    normalized = raw_value.strip().lower().replace("-", "_")
    aliases = {
        "gemm": MusaFusedMoeBackend.GROUPED_GEMM,
        "grouped": MusaFusedMoeBackend.GROUPED_GEMM,
        "native_gemv": MusaFusedMoeBackend.GEMV,
    }
    if normalized in aliases:
        return aliases[normalized]
    try:
        return MusaFusedMoeBackend(normalized)
    except ValueError as exc:
        choices = ", ".join(backend.value for backend in MusaFusedMoeBackend)
        raise ValueError(
            f"Invalid {MUSA_FUSED_MOE_DISPATCH_ENV}={raw_value!r}; "
            f"expected one of: {choices}"
        ) from exc


def thresholds_for_shape(shape: MusaFusedMoeShape) -> MusaFusedMoeThresholds:
    thresholds = _CALIBRATED_THRESHOLDS.get(shape)
    if thresholds is not None:
        return thresholds
    if shape.max_num_seqs is not None:
        # Existing hardware/shape policies predate the scheduler-profile key.
        # Fall back only to an explicitly calibrated generic entry; MP56 has
        # no such entry and therefore fails closed for engines above BS4.
        thresholds = _CALIBRATED_THRESHOLDS.get(replace(shape, max_num_seqs=None))
        if thresholds is not None:
            return thresholds
    return _DEFAULT_THRESHOLDS


def has_calibrated_dimensions(
    *,
    local_experts: int,
    w1_output_size: int,
    w2_input_size: int,
    hidden_size: int,
    top_k: int,
) -> bool:
    """Cheap hot-path rejection before capability and capture checks."""

    return (
        local_experts,
        w1_output_size,
        w2_input_size,
        hidden_size,
        top_k,
    ) in _CALIBRATED_DIMENSIONS


def select_fused_moe_backend(
    *,
    shape: MusaFusedMoeShape,
    num_tokens: int,
    can_use_gemv: bool,
    can_use_grouped_gemm: bool,
    stream_is_capturing: bool,
    requested: MusaFusedMoeBackend = MusaFusedMoeBackend.AUTO,
    thresholds: MusaFusedMoeThresholds | None = None,
) -> MusaFusedMoeBackend:
    """Choose one already-compiled backend without device synchronization.

    Forced modes are diagnostic controls, not correctness bypasses: if a
    requested backend is ineligible the established upstream path is used.
    """

    if requested == MusaFusedMoeBackend.GEMV:
        # Forced GEMV remains useful for eager diagnostics, but graph capture
        # must stay inside an explicitly calibrated capture entry and token
        # range.  Otherwise an unknown shape could be baked into a graph even
        # though the override is documented as preserving capture safety.
        if thresholds is None:
            thresholds = thresholds_for_shape(shape)
        capture_is_calibrated = bool(
            not stream_is_capturing
            or (
                thresholds.gemv_max_tokens is not None
                and num_tokens <= thresholds.gemv_max_tokens
            )
        )
        return (
            MusaFusedMoeBackend.GEMV
            if can_use_gemv and capture_is_calibrated
            else MusaFusedMoeBackend.UPSTREAM
        )
    if requested == MusaFusedMoeBackend.GROUPED_GEMM:
        return (
            MusaFusedMoeBackend.GROUPED_GEMM
            if can_use_grouped_gemm and not stream_is_capturing
            else MusaFusedMoeBackend.UPSTREAM
        )
    if requested == MusaFusedMoeBackend.UPSTREAM:
        return MusaFusedMoeBackend.UPSTREAM

    if thresholds is None:
        thresholds = thresholds_for_shape(shape)
    if (
        can_use_gemv
        and thresholds.gemv_max_tokens is not None
        and num_tokens <= thresholds.gemv_max_tokens
    ):
        return MusaFusedMoeBackend.GEMV
    if (
        can_use_grouped_gemm
        and not stream_is_capturing
        and thresholds.grouped_gemm_min_tokens is not None
        and num_tokens >= thresholds.grouped_gemm_min_tokens
    ):
        return MusaFusedMoeBackend.GROUPED_GEMM
    return MusaFusedMoeBackend.UPSTREAM


__all__ = [
    "MUSA_FUSED_MOE_DISPATCH_ENV",
    "MusaFusedMoeBackend",
    "MusaFusedMoeShape",
    "MusaFusedMoeThresholds",
    "has_calibrated_dimensions",
    "parse_dispatch_backend",
    "select_fused_moe_backend",
    "thresholds_for_shape",
]
