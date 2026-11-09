# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model-agnostic MUSA fused-MoE backend selection.

The hot path must not benchmark, synchronize device data to the host, or key
off a model architecture name.  Offline S5000 sweeps populate exact shape
entries below; unknown shapes keep the established upstream backend.
"""

import os
from dataclasses import dataclass
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
    multiprocessor_count: int = 60,
    local_experts: int,
    w1_output_size: int,
    w2_input_size: int,
    hidden_size: int,
    top_k: int,
    w1_scale_shape: tuple[int, ...],
    w2_scale_shape: tuple[int, ...],
    gemv_block: str,
    graph_mode: str,
    block_size: int = 128,
) -> MusaFusedMoeShape:
    return MusaFusedMoeShape(
        device_capability=(3, 1),
        multiprocessor_count=multiprocessor_count,
        local_experts=local_experts,
        w1_output_size=w1_output_size,
        w2_input_size=w2_input_size,
        hidden_size=hidden_size,
        top_k=top_k,
        block_n=block_size,
        block_k=block_size,
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
    *, graph_mode: str, folded_shared_expert: bool
) -> MusaFusedMoeShape:
    """TP4-local Qwen3.5/3.6 BF16 decode shape."""

    experts = 257 if folded_shared_expert else 256
    top_k = 9 if folded_shared_expert else 8

    return MusaFusedMoeShape(
        device_capability=(3, 1),
        multiprocessor_count=60,
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
            # Cold-L2 crossover calibration covers adaptive target verify
            # lengths, not only the fixed DSpark-7 M=8 shape.  The MP60
            # split-tile selector uses 32x4 through M=12; hot routing regresses
            # at M=13, so larger token batches retain the upstream path.
            gemv_max_tokens=12,
            grouped_gemm_min_tokens=None,
            source=(
                f"s5000-mp60-20260818-e256-n512-k4096-{graph_mode}-"
                "block16-split32-m12"
            ),
        )
        for graph_mode in ("eager", "capture")
    }
)
_CALIBRATED_THRESHOLDS.update(
    {
        _s5000_fp8_shape(
            multiprocessor_count=56,
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
            # Same DSV4 TP8 per-rank shape as MP48/MP60. On MP56 the native
            # W1/W2 GEMV serves up to 8 tokens; larger batches take the
            # upstream Triton path with the per-M decode configs below.
            gemv_max_tokens=8,
            grouped_gemm_min_tokens=None,
            source=f"s5000-mp56-e256-n512-k4096-{graph_mode}-block16-split32-m8",
        )
        for graph_mode in ("eager", "capture")
    }
)
_CALIBRATED_THRESHOLDS.update(
    {
        _s5000_fp8_shape(
            multiprocessor_count=48,
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
            # Preserve the MP56 contract's exact per-rank shape on MP48.
            # Match the native W1/W2 selector through target M=12.
            gemv_max_tokens=12,
            grouped_gemm_min_tokens=None,
            source=f"s5000-mp48-dsv4-tp8-{graph_mode}-block16-split32-m12",
        )
        for graph_mode in ("eager", "capture")
    }
)
_CALIBRATED_THRESHOLDS.update(
    {
        _s5000_fp8_shape(
            multiprocessor_count=multiprocessor_count,
            local_experts=128,
            w1_output_size=1536,
            w2_input_size=768,
            hidden_size=2048,
            top_k=8,
            w1_scale_shape=(128, 12, 16),
            w2_scale_shape=(128, 16, 6),
            gemv_block="auto",
            graph_mode=graph_mode,
        ): _thresholds(
            # Unsharded Qwen3-30B-A3B experts. The GEMV reads each routed
            # expert once per token, so hot routes favour upstream from M=4.
            # Grouped GEMM trailed another backend at every M swept on MP56.
            gemv_max_tokens=3,
            grouped_gemm_min_tokens=None,
            source=(
                f"s5000-mp{multiprocessor_count}-{date}-e128-n1536-k2048-"
                f"{graph_mode}-dense-v6"
            ),
        )
        for multiprocessor_count, date in ((56, "20261108"), (60, "20261109"))
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


# DeepSeek-V4.1 TP8 per-rank routed experts carry 32x32 scale blocks, which the
# native GEMV and grouped GEMM do not take: the upstream Triton path serves this
# shape with the per-M configs below. The MUSA block GEMV serves the smallest
# batches before this policy is consulted.
_DSV41_TP8_BLOCK32_SHAPES: Final = tuple(
    _s5000_fp8_shape(
        local_experts=384,
        w1_output_size=576,
        w2_input_size=288,
        hidden_size=5120,
        top_k=6,
        w1_scale_shape=(384, 18, 160),
        w2_scale_shape=(384, 160, 9),
        gemv_block="auto",
        graph_mode=graph_mode,
        block_size=32,
    )
    for graph_mode in ("eager", "capture")
)
_CALIBRATED_THRESHOLDS.update(
    {
        shape: _thresholds(
            gemv_max_tokens=None,
            grouped_gemm_min_tokens=None,
            source=f"s5000-mp60-20261012-e384-n576-k5120-{shape.graph_mode}-block32-triton",
        )
        for shape in _DSV41_TP8_BLOCK32_SHAPES
    }
)


def _triton_config(block_n: int, num_warps: int, block_m: int = 16) -> dict[str, int]:
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": num_warps,
        "num_stages": 1,
    }


# Per-M Triton configs for the DSV4 TP8 decode ladder on MP56 (target M=5R,
# DSpark draft M=4R, up to 64 requests). BLOCK_M 16 or 32 with BLOCK_K 128
# keeps these batches off the generic table's configs.
_DSV4_TP8_MP56_TRITON_CONFIGS: Final = {
    10: _triton_config(128, 8),
    12: _triton_config(64, 4),
    15: _triton_config(64, 4),
    16: _triton_config(128, 8),
    20: _triton_config(128, 8),
    24: _triton_config(64, 4),
    25: _triton_config(128, 8),
    28: _triton_config(128, 8),
    32: _triton_config(64, 4),
    35: _triton_config(128, 8),
    40: _triton_config(64, 4),
    48: _triton_config(64, 4),
    50: _triton_config(128, 8),
    52: _triton_config(64, 4),
    60: _triton_config(128, 8),
    64: _triton_config(64, 4),
    65: _triton_config(64, 4),
    80: _triton_config(64, 4),
    90: _triton_config(64, 4, block_m=32),
    100: _triton_config(64, 4, block_m=32),
    110: _triton_config(128, 8),
    120: _triton_config(64, 4, block_m=32),
    130: _triton_config(128, 8),
    140: _triton_config(64, 4),
    150: _triton_config(64, 4, block_m=32),
    160: _triton_config(64, 4, block_m=32),
    180: _triton_config(128, 8),
    200: _triton_config(128, 8),
    240: _triton_config(128, 8),
    280: _triton_config(64, 4),
    320: _triton_config(128, 8),
}


def _block32_triton_config(
    block_m: int, block_n: int, group_m: int = 1, num_stages: int = 2
) -> dict[str, int]:
    # One 32-wide scale group per K step, as the upstream kernel requires.
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": group_m,
        "num_warps": 4,
        "num_stages": num_stages,
    }


# Per-M Triton configs for the DeepSeek-V4.1 TP8 routed experts on MP60, from
# MTP/DSpark verify batches to prefill chunks (cold weights, unique-random routes).
_DSV41_TP8_MP60_TRITON_CONFIGS: Final = {
    6: _block32_triton_config(16, 64),
    8: _block32_triton_config(16, 64),
    10: _block32_triton_config(16, 64),
    12: _block32_triton_config(16, 64),
    16: _block32_triton_config(16, 64),
    20: _block32_triton_config(32, 32),
    24: _block32_triton_config(32, 32),
    32: _block32_triton_config(16, 64),
    64: _block32_triton_config(16, 64),
    96: _block32_triton_config(32, 32),
    128: _block32_triton_config(32, 32, 8),
    256: _block32_triton_config(32, 32, 8),
    512: _block32_triton_config(64, 32),
    1024: _block32_triton_config(64, 32, 8),
    2048: _block32_triton_config(64, 32, 8),
}


def _folded_config(group_m: int = 1) -> dict[str, int]:
    # 32x32 tiles with four warps round every row exactly like the default
    # config above 64 tokens, so a token's output does not depend on its batch.
    return _block32_triton_config(32, 32, group_m, num_stages=1)


# Per-M Triton configs for the DeepSeek-V4.1 TP8 routed experts with the shared
# expert folded in (385 experts, top-7) on MP56, from DSpark verify batches past
# the expert-dedup MMA range to prefill chunks: cold weights, worst boundary across
# unique-random and hot routes.
_DSV41_TP8_FOLDED_MP56_TRITON_CONFIGS: Final = {
    10: _folded_config(),
    15: _folded_config(),
    20: _folded_config(),
    25: _folded_config(),
    30: _folded_config(),
    35: _folded_config(),
    40: _folded_config(),
    50: _folded_config(),
    60: _folded_config(),
    70: _folded_config(),
    80: _folded_config(),
    100: _folded_config(8),
    120: _folded_config(),
    160: _folded_config(),
    200: _folded_config(8),
    240: _folded_config(),
    280: _folded_config(8),
    320: _folded_config(),
    512: _folded_config(),
    1024: _folded_config(),
    2048: _folded_config(),
    4096: _folded_config(8),
}

# The same for the folded DSpark draft experts (129 experts, top-4) past their
# 16-token MMA range; decode batches carry four tokens per request.
_DSV41_TP8_FOLDED_DRAFT_MP56_TRITON_CONFIGS: Final = {
    17: _folded_config(8),
    20: _folded_config(8),
    24: _folded_config(),
    28: _folded_config(),
    32: _folded_config(8),
    40: _folded_config(),
    48: _folded_config(8),
    56: _folded_config(),
    64: _folded_config(),
    80: _folded_config(),
    96: _folded_config(),
    128: _folded_config(),
    160: _folded_config(),
    192: _folded_config(8),
    256: _folded_config(8),
}

# (local experts, top-k) of the folded DeepSeek-V4.1 TP8 MoE layers.
_DSV41_TP8_FOLDED_MP56_TABLES: Final = {
    (385, 7): _DSV41_TP8_FOLDED_MP56_TRITON_CONFIGS,
    (129, 4): _DSV41_TP8_FOLDED_DRAFT_MP56_TRITON_CONFIGS,
}

# Exact-shape upstream Triton launch configs, keyed by token count. They
# replace the tuned-folder lookup only inside the recorded token range; other
# shapes and token counts keep the established lookup.
_CALIBRATED_TRITON_CONFIGS: Final[
    dict[MusaFusedMoeShape, dict[int, dict[str, int]]]
] = {
    _s5000_fp8_shape(
        multiprocessor_count=56,
        local_experts=256,
        w1_output_size=512,
        w2_input_size=256,
        hidden_size=4096,
        top_k=6,
        w1_scale_shape=(256, 4, 32),
        w2_scale_shape=(256, 32, 2),
        gemv_block="16x8",
        graph_mode=graph_mode,
    ): _DSV4_TP8_MP56_TRITON_CONFIGS
    for graph_mode in ("eager", "capture")
}
_CALIBRATED_TRITON_CONFIGS.update(
    {shape: _DSV41_TP8_MP60_TRITON_CONFIGS for shape in _DSV41_TP8_BLOCK32_SHAPES}
)
_CALIBRATED_TRITON_CONFIGS.update(
    {
        _s5000_fp8_shape(
            multiprocessor_count=56,
            local_experts=local_experts,
            w1_output_size=576,
            w2_input_size=288,
            hidden_size=5120,
            top_k=top_k,
            w1_scale_shape=(local_experts, 18, 160),
            w2_scale_shape=(local_experts, 160, 9),
            gemv_block="auto",
            graph_mode=graph_mode,
            block_size=32,
        ): table
        for (local_experts, top_k), table in _DSV41_TP8_FOLDED_MP56_TABLES.items()
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
    for shape in (*_CALIBRATED_THRESHOLDS, *_CALIBRATED_TRITON_CONFIGS)
)


# Shapes calibrated only for upstream Triton launch configs (no GEMV or grouped
# GEMM threshold); the dispatcher builds their policy key without those kernels.
_TRITON_ONLY_DIMENSIONS: Final = frozenset(
    (
        shape.local_experts,
        shape.w1_output_size,
        shape.w2_input_size,
        shape.hidden_size,
        shape.top_k,
    )
    for shape in _CALIBRATED_TRITON_CONFIGS
    if _CALIBRATED_THRESHOLDS.get(shape, _DEFAULT_THRESHOLDS).gemv_max_tokens is None
    and _CALIBRATED_THRESHOLDS.get(shape, _DEFAULT_THRESHOLDS).grouped_gemm_min_tokens
    is None
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
    return _CALIBRATED_THRESHOLDS.get(shape, _DEFAULT_THRESHOLDS)


def triton_config_for_shape(
    shape: MusaFusedMoeShape, num_tokens: int
) -> dict[str, int] | None:
    """Return the calibrated Triton config nearest ``num_tokens``, if any."""

    table = _CALIBRATED_TRITON_CONFIGS.get(shape)
    if not table or not min(table) <= num_tokens <= max(table):
        return None
    key = min(table, key=lambda tokens: (abs(tokens - num_tokens), tokens))
    return dict(table[key])


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


def has_triton_only_calibration(
    *,
    local_experts: int,
    w1_output_size: int,
    w2_input_size: int,
    hidden_size: int,
    top_k: int,
) -> bool:
    """Whether the dimensions have Triton configs but no GEMV or grouped threshold."""

    return (
        local_experts,
        w1_output_size,
        w2_input_size,
        hidden_size,
        top_k,
    ) in _TRITON_ONLY_DIMENSIONS


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
    "has_triton_only_calibration",
    "parse_dispatch_backend",
    "select_fused_moe_backend",
    "thresholds_for_shape",
]
