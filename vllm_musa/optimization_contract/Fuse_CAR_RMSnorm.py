# SPDX-License-Identifier: Apache-2.0
"""Declarative dispatch policy for the MUSA fused all-reduce RMSNorm path.

The rule set is ported from the upstream SGLang MUSA all-reduce v2 dispatch
table so it can be reviewed in one place. The kernel enum is carried as data
only: this repository launches its own fused kernels, so the enum does not
select a kernel here. The whole module is inert unless
VLLM_MUSA_CAR_FUSED_DISPATCH is set.
"""

from dataclasses import dataclass, fields, replace
import os
from enum import Enum
from typing import Any, Callable, Mapping

SHOT_PUSH = 0
SHOT_ONE_STAGE = 1
SHOT_TWO_STAGE = 2
SHOT_PUSH_WIDE = 3
SHOT_TWO_STAGE_512 = 4


class CustomARKernel(str, Enum):
    GENERIC = "generic"
    ROW_ONE = "row_one"
    TUNED = "tuned"
    SMALL_ROWS = "small_rows"
    FALLBACK = "fallback"


class Matcher:
    def matches(self, value: Any) -> bool:
        raise NotImplementedError

    @property
    def specificity(self) -> int:
        return 0


@dataclass(frozen=True)
class AnyValue(Matcher):
    def matches(self, value: Any) -> bool:
        return True


@dataclass(frozen=True)
class Exact(Matcher):
    value: Any

    def matches(self, value: Any) -> bool:
        return value == self.value

    @property
    def specificity(self) -> int:
        return 30


@dataclass(frozen=True)
class OneOf(Matcher):
    values: frozenset[Any]

    def matches(self, value: Any) -> bool:
        return value in self.values

    @property
    def specificity(self) -> int:
        return 20


@dataclass(frozen=True)
class IntRange(Matcher):
    """Integer range using the half-open interval ``[begin, end)``."""

    begin: int | None = None
    end: int | None = None
    step: int | None = None

    def matches(self, value: Any) -> bool:
        if not isinstance(value, int):
            return False
        if self.begin is not None and value < self.begin:
            return False
        if self.end is not None and value >= self.end:
            return False
        return self.step is None or (value - (self.begin or 0)) % self.step == 0

    @property
    def specificity(self) -> int:
        return 10 + int(self.begin is not None) + int(self.end is not None)


@dataclass(frozen=True)
class PowerOfTwo(Matcher):
    begin: int = 1
    end: int | None = None

    def matches(self, value: Any) -> bool:
        return (
            isinstance(value, int)
            and self.begin <= value
            and (self.end is None or value < self.end)
            and value > 0
            and (value & (value - 1)) == 0
        )

    @property
    def specificity(self) -> int:
        return 12


@dataclass(frozen=True)
class DispatchKey:
    mp_count: int
    world_size: int
    mode: str
    rows: int
    hidden_size: int
    token_2stage: bool
    graph_rank_data: bool


@dataclass(frozen=True)
class Match:
    mp_count: Matcher = AnyValue()
    world_size: Matcher = AnyValue()
    mode: Matcher = AnyValue()
    rows: Matcher = AnyValue()
    hidden_size: Matcher = AnyValue()
    token_2stage: Matcher = AnyValue()

    def matches(self, key: DispatchKey) -> bool:
        return all(
            matcher.matches(value)
            for matcher, value in (
                (self.mp_count, key.mp_count),
                (self.world_size, key.world_size),
                (self.mode, key.mode),
                (self.rows, key.rows),
                (self.hidden_size, key.hidden_size),
                (self.token_2stage, key.token_2stage),
            )
        )

    @property
    def specificity(self) -> int:
        return sum(
            matcher.specificity
            for matcher in (
                self.mp_count,
                self.world_size,
                self.mode,
                self.rows,
                self.hidden_size,
                self.token_2stage,
            )
        )


@dataclass(frozen=True)
class KernelConfig:
    """Complete launch/JIT config consumed by both dispatch call sites."""

    kernel: CustomARKernel = CustomARKernel.GENERIC
    threads: int = 512
    blocks: int = 120
    max_blocks: int = 120
    launch_max_blocks: int = 120
    dynamic_blocks: int = 1
    token_2stage: bool = True
    graph_rank_data: bool = False
    row_warp_inv_rms: int = 0
    tuned_module: bool = False
    small_rows_module: bool = False
    short_rows_module: bool = False
    row_weight_cache: bool = False
    row_weight_cache_min_rows: int = 2048
    row_shared_inv_rms: bool = False
    sums_bypass_load: int = 0
    fast_publish: bool = False
    packed_1stage: int = 0
    vec2rank_1stage: int = 0
    partial_packed_non8: int = 0
    vec2_non8: int = 0
    vec4_non8: int = 0
    regcache_2rank: int = 0
    cache_hidden_limit: int = 0
    typed_cache_hidden_limit: int = 0
    sum_blocks: int = 16
    weight_cache_hidden_limit: int = 8192
    tmp_with_residual: int = 0
    small_rows_flag: int = 0
    row_skip_end_barrier: bool = False
    row_peer_publish: bool = False
    row_skip_self_signal: bool = False
    row_push: bool = False

    def name_suffix(self) -> str:
        return (
            f"_t{self.threads}_b{self.blocks}_lb{self.launch_max_blocks}"
            f"_r2{int(self.token_2stage)}_p1{self.packed_1stage}"
            f"_v2{self.vec2rank_1stage}_pn8{self.partial_packed_non8}"
            f"_v2n{self.vec2_non8}_v4n{self.vec4_non8}"
            f"_rc{self.regcache_2rank}"
            f"_c{self.cache_hidden_limit}_tc{self.typed_cache_hidden_limit}"
            f"_h8{self.sum_blocks}_wc{self.weight_cache_hidden_limit}"
            f"_rw{int(self.row_weight_cache)}_rm{self.row_weight_cache_min_rows}"
            f"_si{int(self.row_shared_inv_rms)}_wi{self.row_warp_inv_rms}"
            f"_sb{self.sums_bypass_load}_fp{int(self.fast_publish)}_sk{int(self.row_skip_end_barrier)}"
            f"_pp{int(self.row_peer_publish)}"
            f"_ss{int(self.row_skip_self_signal)}"
            f"_rp{int(self.row_push)}"
            f"_tr{self.tmp_with_residual}_sr{self.small_rows_flag}_m2_r2s"
        )

    def cflags(self) -> tuple[str, ...]:
        # Keep macro spellings stable: they are part of the MUSA kernel ABI.
        flags = [
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_TOKEN_2STAGE={int(self.token_2stage)}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_PACKED_1STAGE={self.packed_1stage}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_VEC2RANK_1STAGE={self.vec2rank_1stage}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_PARTIAL_PACKED_NON8={self.partial_packed_non8}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_VEC2_NON8={self.vec2_non8}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_VEC4_NON8={self.vec4_non8}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_REGCACHE_2RANK={self.regcache_2rank}",
            f"-DSGL_CUSTOM_AR_RMSNORM_CACHE_HIDDEN_LIMIT={self.cache_hidden_limit}",
            f"-DSGL_CUSTOM_AR_RMSNORM_T_CACHE_HIDDEN_LIMIT={self.typed_cache_hidden_limit}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_SUM_BLOCKS={self.sum_blocks}",
            f"-DSGL_CUSTOM_AR_RMSNORM_WEIGHT_CACHE_HIDDEN_LIMIT={self.weight_cache_hidden_limit}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WEIGHT_CACHE={int(self.row_weight_cache)}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WEIGHT_CACHE_MIN_ROWS={self.row_weight_cache_min_rows}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_ROW_SHARED_INV_RMS={int(self.row_shared_inv_rms)}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WARP_INV_RMS={self.row_warp_inv_rms}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD={self.sums_bypass_load}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_FAST_PUBLISH={int(self.fast_publish)}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL={self.tmp_with_residual}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_SMALL_ROWS={self.small_rows_flag}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_ROW_SKIP_END_BARRIER={int(self.row_skip_end_barrier)}",
            "-DSGL_CUSTOM_AR_FUSED_RMSNORM_ROW_PEER_PUBLISH="
            f"{int(self.row_peer_publish)}",
            "-DSGL_CUSTOM_AR_FUSED_RMSNORM_ROW_SKIP_SELF_SIGNAL="
            f"{int(self.row_skip_self_signal)}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_ROW_PUSH={int(self.row_push)}",
            f"-DSGL_CUSTOM_AR_LAUNCH_MAX_BLOCKS={self.launch_max_blocks}",
        ]
        if self.blocks != 120:
            flags.append(f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_BLOCKS={self.blocks}")
        if self.launch_max_blocks != 120:
            flags.append(
                f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_LAUNCH_MAX_BLOCKS={self.launch_max_blocks}"
            )
        return tuple(flags)


@dataclass(frozen=True)
class DispatchRule:
    name: str
    match: Match
    config: KernelConfig | None = None
    priority: int = 0
    patch: Mapping[str, Any] | None = None

    @property
    def order(self) -> tuple[int, int]:
        return self.priority, self.match.specificity


@dataclass(frozen=True)
class ArchitectureKey:
    mp_count: int
    world_size: int
    shot: int


@dataclass(frozen=True)
class ArchitectureMatch:
    mp_count: Matcher = AnyValue()
    world_size: Matcher = AnyValue()
    shot: Matcher = AnyValue()

    def matches(self, key: ArchitectureKey) -> bool:
        return all(
            matcher.matches(value)
            for matcher, value in (
                (self.mp_count, key.mp_count),
                (self.world_size, key.world_size),
                (self.shot, key.shot),
            )
        )


@dataclass(frozen=True)
class ArchitectureRule:
    name: str
    match: ArchitectureMatch
    launch_max_blocks: int | None = None
    launch_formula: Callable[[ArchitectureKey], int] | None = None


_ARCHITECTURE_RULES = (
    ArchitectureRule(
        "tp8-56sm-two-wave",
        ArchitectureMatch(Exact(56), Exact(8), Exact(SHOT_TWO_STAGE_512)),
        launch_formula=lambda key: 2 * key.mp_count - key.world_size,
    ),
)


def resolve_architecture_launch(
    *, mp_count: int, world_size: int, shot: int, max_blocks: int
) -> int:
    """Return the architecture-specific launch cap for a custom-AR shot."""
    key = ArchitectureKey(mp_count, world_size, shot)
    for rule in _ARCHITECTURE_RULES:
        if rule.match.matches(key):
            if rule.launch_formula is not None:
                return rule.launch_formula(key)
            if rule.launch_max_blocks is not None:
                return rule.launch_max_blocks
    return max_blocks


_FUSED = Exact("fused_rmsnorm")
_POW2_HIDDEN = PowerOfTwo(1, 8193)

# Shapes whose measured fused minimum is no better than composed
# custom-AR + fused_add_rmsnorm.  Keep this list explicit and reviewable;
# every entry is derived from the checked-in min-latency matrix and can be
# removed independently after a kernel optimization is validated.
_MEASURED_FALLBACK_RULES = (
    # S5000 60-MP TP4 fused AR+RMSNorm decode kernels.  One CTA per row avoids
    # launching the generic 60-CTA reduction for the Qwen3.5 h=2048 and
    # Gemma4 h=5376 small-row shapes.  These are fused v2 kernels, not fallback
    # rules; the tuple name is historical and also contains validated patches.
    *(
        DispatchRule(
            f"validated-60-tp4-r{rows}-h2048-row-one",
            Match(
                mp_count=Exact(60),
                world_size=Exact(4),
                mode=_FUSED,
                rows=Exact(rows),
                hidden_size=Exact(2048),
            ),
            priority=11000,
            patch={
                "kernel": CustomARKernel.ROW_ONE,
                "threads": 256,
                "blocks": rows,
                "launch_max_blocks": rows,
                "small_rows_module": True,
                "small_rows_flag": 1,
            },
        )
        for rows in (2, 8, 32)
    ),
    *(
        DispatchRule(
            f"validated-60-tp4-r{rows}-h5376-row-one",
            Match(
                mp_count=Exact(60),
                world_size=Exact(4),
                mode=_FUSED,
                rows=Exact(rows),
                hidden_size=Exact(5376),
            ),
            priority=11000,
            patch={
                "kernel": CustomARKernel.ROW_ONE,
                "threads": 512,
                "blocks": rows,
                "launch_max_blocks": rows,
                "small_rows_module": False,
                "small_rows_flag": 0,
            },
        )
        for rows in (2, 4, 8, 16, 32)
    ),
    DispatchRule(
        "validated-60-tp4-r1-h5376-row-one-t672",
        Match(
            mp_count=Exact(60),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(5376),
        ),
        priority=11000,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 672,
            "blocks": 1,
            "launch_max_blocks": 1,
            "small_rows_module": False,
            "small_rows_flag": 0,
        },
    ),
    DispatchRule(
        "fallback-56-tp4-r1",
        Match(
            mp_count=Exact(56),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=OneOf(
                frozenset({1024, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192})
            ),
        ),
        priority=1000,
        patch={"kernel": CustomARKernel.FALLBACK},
    ),
    DispatchRule(
        "fallback-56-tp4-r128",
        Match(
            mp_count=Exact(56),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=OneOf(frozenset({1024, 2048})),
        ),
        priority=1000,
        patch={"kernel": CustomARKernel.FALLBACK},
    ),
    DispatchRule(
        "fallback-56-tp8-r1",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=OneOf(
                frozenset({1024, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168})
            ),
        ),
        priority=1000,
        patch={"kernel": CustomARKernel.FALLBACK},
    ),
    DispatchRule(
        "fallback-56-tp8-r128",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=OneOf(frozenset({1024, 2560, 3072, 3584, 6144, 7168})),
        ),
        priority=1000,
        patch={"kernel": CustomARKernel.FALLBACK},
    ),
    DispatchRule(
        "fallback-56-tp8-rlarge",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=IntRange(1024, 65537),
            hidden_size=OneOf(frozenset({2560, 3072, 3584, 5120, 6144, 7168})),
        ),
        priority=1000,
        patch={"kernel": CustomARKernel.FALLBACK},
    ),
)


_VALIDATED_ROW_PROMOTIONS = (
    DispatchRule(
        "validated-60-tp2-r1-h2048-special-row",
        Match(
            mp_count=Exact(60),
            world_size=Exact(2),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(2048),
        ),
        priority=10000,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 256,
            "blocks": 1,
            "launch_max_blocks": 1,
            "small_rows_module": True,
            "small_rows_flag": 1,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
        },
    ),
    DispatchRule(
        "validated-60-tp2-r4-h2048-special-row",
        Match(
            mp_count=Exact(60),
            world_size=Exact(2),
            mode=_FUSED,
            rows=Exact(4),
            hidden_size=Exact(2048),
        ),
        priority=10000,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 256,
            "blocks": 4,
            "launch_max_blocks": 4,
            "small_rows_module": True,
            "small_rows_flag": 1,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
        },
    ),
    DispatchRule(
        "validated-60-tp2-r16-h2048-special-row",
        Match(
            mp_count=Exact(60),
            world_size=Exact(2),
            mode=_FUSED,
            rows=Exact(16),
            hidden_size=Exact(2048),
        ),
        priority=10000,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 256,
            "blocks": 16,
            "launch_max_blocks": 16,
            "small_rows_module": True,
            "small_rows_flag": 1,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
        },
    ),
    DispatchRule(
        "validated-60-tp2-r64-h2048-generic-t512",
        Match(
            mp_count=Exact(60),
            world_size=Exact(2),
            mode=_FUSED,
            rows=Exact(64),
            hidden_size=Exact(2048),
        ),
        priority=10000,
        patch={
            "threads": 512,
            "blocks": 64,
            "launch_max_blocks": 64,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
        },
    ),
    DispatchRule(
        "validated-60-tp2-r128-h2048-special-row",
        Match(
            mp_count=Exact(60),
            world_size=Exact(2),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=Exact(2048),
        ),
        priority=10000,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 256,
            "blocks": 128,
            "launch_max_blocks": 128,
            "small_rows_module": False,
            "small_rows_flag": 0,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
        },
    ),
    DispatchRule(
        "validated-60-tp4-r16-h2048-special-row",
        Match(
            mp_count=Exact(60),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(16),
            hidden_size=Exact(2048),
        ),
        priority=10000,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 256,
            "blocks": 16,
            "launch_max_blocks": 16,
            "small_rows_module": True,
            "small_rows_flag": 1,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
        },
    ),
    *(
        DispatchRule(
            name,
            Match(
                mp_count=Exact(60),
                world_size=Exact(4),
                mode=_FUSED,
                rows=Exact(128),
                hidden_size=Exact(hidden),
            ),
            # Exact graph A/B: h2560/3072 on driver 3.3.5, h4096/6144/8192
            # on 3.3.8 (runtime 4.3.5). Row-local geometry beats the existing
            # generic fused route and composed AR in both execution orders.
            priority=1600,
            patch={
                "kernel": CustomARKernel.ROW_ONE,
                "threads": threads,
                "blocks": 128,
                "launch_max_blocks": 128,
                "token_2stage": True,
                "small_rows_module": False,
                "small_rows_flag": 0,
                "row_warp_inv_rms": 1,
                "row_shared_inv_rms": False,
                "sums_bypass_load": 3,
                "row_skip_end_barrier": True,
                "row_skip_self_signal": True,
                "tmp_with_residual": 0,
            },
        )
        for name, hidden, threads in (
            ("validated-60-tp4-r128-h2560-row128", 2560, 320),
            ("validated-60-tp4-r128-h3072-row128", 3072, 384),
            ("validated-60-tp4-r128-h4096-row128", 4096, 512),
            ("validated-60-tp4-r128-h6144-row128", 6144, 768),
            ("validated-60-tp4-r128-h8192-row128", 8192, 1024),
        )
    ),
    *(
        DispatchRule(
            f"validated-56-tp8-nonstandard-h{hidden}-r2048-32768-row104",
            Match(
                mp_count=Exact(56),
                world_size=Exact(8),
                mode=_FUSED,
                rows=OneOf(frozenset({2048, 32768})),
                hidden_size=Exact(hidden),
            ),
            priority=1200,
            patch={
                "kernel": CustomARKernel.ROW_ONE,
                "blocks": 104,
                "launch_max_blocks": 104,
                "row_warp_inv_rms": 1,
                "sums_bypass_load": 3,
                "row_skip_end_barrier": True,
                "row_skip_self_signal": True,
            },
        )
        for hidden in (2560, 3072, 3584)
    ),
    *(
        DispatchRule(
            name,
            Match(
                mp_count=Exact(sm),
                world_size=Exact(8),
                mode=_FUSED,
                rows=rows,
                hidden_size=Exact(hidden),
            ),
            # Same-source graph A/B against both composed and the existing fused
            # route. A shared 104-block rule regresses other 60-SM grid points.
            priority=1600,
            patch={
                "kernel": CustomARKernel.ROW_ONE,
                "threads": 512,
                "blocks": 104,
                "launch_max_blocks": 104,
                "token_2stage": True,
                "small_rows_module": False,
                "small_rows_flag": 0,
                "row_push": False,
                "row_warp_inv_rms": 1,
                "row_shared_inv_rms": False,
                "sums_bypass_load": 3,
                "row_skip_end_barrier": True,
                "row_skip_self_signal": True,
                "tmp_with_residual": 0,
            },
        )
        for name, sm, rows, hidden in (
            (
                "validated-56-tp8-grid-h8192-row104",
                56,
                OneOf(frozenset({1024, 4096, 4097, 16384, 65536})),
                8192,
            ),
            ("validated-60-tp8-grid-r4096-h3072-row104", 60, Exact(4096), 3072),
            (
                "validated-56-tp8-nonstandard-h3584-row104",
                56,
                OneOf(frozenset({1026, 4094, 8192})),
                3584,
            ),
            (
                "validated-56-tp8-nonstandard-h5120-r2048-8192-32768-row104",
                56,
                OneOf(frozenset({2048, 8192, 32768})),
                5120,
            ),
            (
                "validated-56-tp8-nonstandard-h6144-r2048-8192-32768-row104",
                56,
                OneOf(frozenset({2048, 8192, 32768})),
                6144,
            ),
            (
                "validated-56-tp8-nonstandard-h7168-r2048-8192-32768-row104",
                56,
                OneOf(frozenset({2048, 8192, 32768})),
                7168,
            ),
        )
    ),
    DispatchRule(
        "validated-56-tp8-nonstandard-r1026-4094-8192-h2560-row104",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=OneOf(frozenset({1026, 4094, 8192})),
            hidden_size=Exact(2560),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 104,
            "launch_max_blocks": 104,
            "row_warp_inv_rms": 1,
            "sums_bypass_load": 3,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
        },
    ),
    DispatchRule(
        "validated-56-tp8-nonstandard-r1026-4094-8192-h3072-row104",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=OneOf(frozenset({1026, 4094, 8192})),
            hidden_size=Exact(3072),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 104,
            "launch_max_blocks": 104,
            "row_warp_inv_rms": 1,
            "sums_bypass_load": 3,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
        },
    ),
    DispatchRule(
        "validated-tp8-r1023-h2560-3072-3584-6144-7168-row104",
        Match(
            mp_count=OneOf(frozenset({56, 60})),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1023),
            hidden_size=OneOf(frozenset({2560, 3072, 3584, 6144, 7168})),
        ),
        # Same-image graph A/B on driver 3.3.5 / runtime 4.3.5 in both
        # execution orders. Keep the exact row and hidden set.
        priority=1600,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 512,
            "blocks": 104,
            "launch_max_blocks": 104,
            "token_2stage": True,
            "small_rows_module": False,
            "small_rows_flag": 0,
            "row_push": False,
            "row_warp_inv_rms": 1,
            "row_shared_inv_rms": False,
            "sums_bypass_load": 3,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
            "tmp_with_residual": 0,
        },
    ),
    DispatchRule(
        "validated-tp8-r16385-h2560-3072-3584-6144-7168-8192-row104",
        Match(
            mp_count=OneOf(frozenset({56, 60})),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(16385),
            hidden_size=OneOf(frozenset({2560, 3072, 3584, 6144, 7168, 8192})),
        ),
        # Same-image graph A/B on driver 3.3.5 / runtime 4.3.5 in both
        # execution orders. Keep this exact row set; no range inference.
        priority=1600,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 512,
            "blocks": 104,
            "launch_max_blocks": 104,
            "token_2stage": True,
            "small_rows_module": False,
            "small_rows_flag": 0,
            "row_push": False,
            "row_warp_inv_rms": 1,
            "row_shared_inv_rms": False,
            "sums_bypass_load": 3,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
            "tmp_with_residual": 0,
        },
    ),
    DispatchRule(
        "validated-tp8-r16383-h2560-3072-3584-6144-7168-8192-row104",
        Match(
            mp_count=OneOf(frozenset({56, 60})),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(16383),
            hidden_size=OneOf(frozenset({2560, 3072, 3584, 6144, 7168, 8192})),
        ),
        # Same-image graph A/B on driver 3.3.5 / runtime 4.3.5 in both
        # execution orders. The exact row set avoids range inference.
        priority=1600,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 512,
            "blocks": 104,
            "launch_max_blocks": 104,
            "token_2stage": True,
            "small_rows_module": False,
            "small_rows_flag": 0,
            "row_push": False,
            "row_warp_inv_rms": 1,
            "row_shared_inv_rms": False,
            "sums_bypass_load": 3,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
            "tmp_with_residual": 0,
        },
    ),
    DispatchRule(
        "validated-tp8-r4097-nonpow2-row104",
        Match(
            mp_count=OneOf(frozenset({56, 60})),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(4097),
            hidden_size=OneOf(frozenset({2560, 3072, 3584, 6144, 7168})),
        ),
        # Same-image graph A/B on driver 3.3.5 / runtime 4.3.5 in both
        # execution orders. h5120 uses the separate adjacent-row rule.
        # h8192 on 60-SM retains the faster existing 120-block fused route.
        priority=1600,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 512,
            "blocks": 104,
            "launch_max_blocks": 104,
            "token_2stage": True,
            "small_rows_module": False,
            "small_rows_flag": 0,
            "row_push": False,
            "row_warp_inv_rms": 1,
            "row_shared_inv_rms": False,
            "sums_bypass_load": 3,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
            "tmp_with_residual": 0,
        },
    ),
    DispatchRule(
        "validated-tp8-adjacent-h5120-row104",
        Match(
            mp_count=OneOf(frozenset({56, 60})),
            world_size=Exact(8),
            mode=_FUSED,
            rows=OneOf(frozenset({1023, 1024, 4097, 16383, 16385})),
            hidden_size=Exact(5120),
        ),
        # Same-image graph A/B on driver 3.3.5 / runtime 4.3.5 in both
        # execution orders. Keep the row set exact; do not infer other rows.
        priority=1600,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 512,
            "blocks": 104,
            "launch_max_blocks": 104,
            "token_2stage": True,
            "small_rows_module": False,
            "small_rows_flag": 0,
            "row_push": False,
            "row_warp_inv_rms": 1,
            "row_shared_inv_rms": False,
            "sums_bypass_load": 3,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
            "tmp_with_residual": 0,
        },
    ),
    DispatchRule(
        "validated-tp8-boundary-h2560-row104",
        Match(
            mp_count=OneOf(frozenset({56, 60})),
            world_size=Exact(8),
            mode=_FUSED,
            rows=OneOf(frozenset({1025, 4095})),
            hidden_size=OneOf(frozenset({2560, 3072, 3584, 5120, 6144, 7168, 8192})),
        ),
        # Same-image graph A/B on driver 3.3.5 / runtime 4.3.5 for h2560 in
        # both execution orders with rank-distinct residuals. No range
        # inference to adjacent row counts or other hidden sizes.
        priority=1600,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 512,
            "blocks": 104,
            "launch_max_blocks": 104,
            "token_2stage": True,
            "row_warp_inv_rms": 1,
            "row_shared_inv_rms": False,
            "sums_bypass_load": 3,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
            "tmp_with_residual": 0,
        },
    ),
    DispatchRule(
        "validated-56-tp4-r1-h1024-fused-push",
        Match(
            mp_count=Exact(56),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(1024),
        ),
        priority=1600,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 128,
            "blocks": 1,
            "launch_max_blocks": 1,
            "small_rows_module": True,
            "small_rows_flag": 1,
            "token_2stage": True,
            "row_push": True,
        },
    ),
    DispatchRule(
        "validated-56-tp4-r1-h2560-fused-push",
        Match(
            mp_count=Exact(56),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(2560),
        ),
        priority=1600,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 320,
            "blocks": 1,
            "launch_max_blocks": 1,
            "token_2stage": True,
            "row_push": True,
        },
    ),
    DispatchRule(
        "validated-56-tp4-r1-h3072-fused-push",
        Match(
            mp_count=Exact(56),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(3072),
        ),
        priority=1600,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 384,
            "blocks": 1,
            "launch_max_blocks": 1,
            "token_2stage": True,
            "row_push": True,
        },
    ),
    DispatchRule(
        "validated-56-tp4-r1-h5120-fused-push",
        Match(
            mp_count=Exact(56),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(5120),
        ),
        priority=1600,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 1024,
            "blocks": 1,
            "launch_max_blocks": 1,
            "token_2stage": True,
            "row_push": True,
        },
    ),
    DispatchRule(
        "validated-56-tp4-r1-h6144-fused-push",
        Match(
            mp_count=Exact(56),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(6144),
        ),
        priority=1600,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 768,
            "blocks": 1,
            "launch_max_blocks": 1,
            "token_2stage": True,
            "row_push": True,
        },
    ),
    DispatchRule(
        "validated-56-tp4-r1-h8192-fused-push",
        Match(
            mp_count=Exact(56),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(8192),
        ),
        priority=1600,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 1024,
            "blocks": 1,
            "launch_max_blocks": 1,
            "token_2stage": True,
            "row_push": True,
        },
    ),
    DispatchRule(
        "validated-56-tp4-r1-h7168-cta1-skip-self",
        Match(
            mp_count=Exact(56),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(7168),
        ),
        # Rechecked on 56-SM / MUSA 3.3.5 graph in both execution orders.
        priority=1500,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 896,
            "blocks": 1,
            "launch_max_blocks": 1,
            "sums_bypass_load": 3,
            "row_skip_self_signal": True,
        },
    ),
    DispatchRule(
        "validated-56-tp4-r128-h1024-row128-warp",
        Match(
            mp_count=Exact(56),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=Exact(1024),
        ),
        # Rechecked on 56-SM / MUSA 3.3.5 graph in both execution orders.
        priority=1500,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 512,
            "blocks": 128,
            "launch_max_blocks": 128,
            "small_rows_module": False,
            "small_rows_flag": 0,
            "row_shared_inv_rms": False,
            "row_warp_inv_rms": 1,
            "sums_bypass_load": 3,
        },
    ),
    DispatchRule(
        "validated-56-tp4-r4-h2048-cta4",
        Match(
            mp_count=Exact(56),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(4),
            hidden_size=Exact(2048),
        ),
        # The fused h2048 TP4 kernel has a dedicated rows<=32 path. Keep one
        # CTA per row for the small decode batch instead of the generic
        # residency launch (80 CTAs for this four-row input).
        priority=1500,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 512,
            "blocks": 4,
            "launch_max_blocks": 4,
            "small_rows_module": False,
            "small_rows_flag": 0,
            "sums_bypass_load": 3,
        },
    ),
    DispatchRule(
        "validated-60-tp4-r4-h2048-cta4",
        Match(
            mp_count=Exact(60),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(4),
            hidden_size=Exact(2048),
        ),
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 512,
            "blocks": 4,
            "launch_max_blocks": 4,
            "small_rows_module": False,
            "small_rows_flag": 0,
            "sums_bypass_load": 3,
        },
        priority=1500,
    ),
    DispatchRule(
        "validated-56-tp4-r1-h3584-cta1-peer-publish",
        Match(
            mp_count=Exact(56),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(3584),
        ),
        # Rechecked on 56-SM / MUSA 3.3.5 graph in both execution orders.
        priority=1500,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 448,
            "blocks": 1,
            "launch_max_blocks": 1,
            "sums_bypass_load": 3,
            "row_peer_publish": True,
        },
    ),
    DispatchRule(
        "validated-56-tp4-r1-h4096-cta1-t1024",
        Match(
            mp_count=Exact(56),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(4096),
        ),
        # Rechecked on 56-SM / MUSA 3.3.5 graph in both execution orders.
        priority=1500,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 1024,
            "blocks": 1,
            "launch_max_blocks": 1,
            "sums_bypass_load": 3,
        },
    ),
    DispatchRule(
        "validated-56-tp4-r1-h2048-cta1-small-shared",
        Match(
            mp_count=Exact(56),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(2048),
        ),
        # Rechecked on 56-SM / MUSA 3.3.5 graph in both execution orders.
        priority=1500,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 256,
            "blocks": 1,
            "launch_max_blocks": 1,
            "small_rows_module": True,
            "small_rows_flag": 1,
            "row_shared_inv_rms": True,
            "sums_bypass_load": 3,
        },
    ),
    DispatchRule(
        "validated-56-tp8-r16384-65536-h7168-row104",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=OneOf(frozenset({16384, 65536})),
            hidden_size=Exact(7168),
        ),
        # Rechecked on 56-SM / MUSA 3.3.5 graph in both execution orders.
        priority=1500,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 896,
            "blocks": 104,
            "launch_max_blocks": 104,
            "fast_publish": False,
        },
    ),
    # Rechecked on 56-SM / MUSA 5.2.0 graph in both execution orders with
    # rank-distinct random inputs.  Keep the unmeasured rows=128 h1024 point
    # in the explicit fallback table below.
    *(
        DispatchRule(
            f"validated-56-tp8-r1-h{hidden}-fused-push-52",
            Match(
                mp_count=Exact(56),
                world_size=Exact(8),
                mode=_FUSED,
                rows=Exact(1),
                hidden_size=Exact(hidden),
            ),
            priority=1600,
            patch={
                "kernel": CustomARKernel.ROW_ONE,
                "threads": threads,
                "blocks": 1,
                "launch_max_blocks": 1,
                "token_2stage": True,
                "small_rows_module": True,
                "small_rows_flag": 1,
                "row_push": True,
                "row_warp_inv_rms": 1,
                "sums_bypass_load": 3,
                "row_skip_end_barrier": True,
                "tmp_with_residual": 0,
            },
        )
        for hidden, threads in (
            (1024, 128),
            (2048, 256),
            (2560, 320),
            (3072, 384),
            (5120, 640),
            (6144, 768),
            (7168, 896),
        )
    ),
    DispatchRule(
        "validated-56-tp8-r128-h1024-direct-52",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=Exact(1024),
        ),
        # Rechecked on 56-SM / MUSA 5.2.0 graph in both orders with random
        # inputs. The small-row one-CTA push candidate regressed here; use
        # the direct 128-CTA row route instead.
        priority=1600,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 128,
            "blocks": 128,
            "launch_max_blocks": 128,
            "token_2stage": True,
            "small_rows_module": False,
            "small_rows_flag": 0,
            "row_push": False,
            "row_warp_inv_rms": 1,
            "sums_bypass_load": 3,
            "row_skip_end_barrier": True,
            "tmp_with_residual": 0,
        },
    ),
    DispatchRule(
        "validated-56-tp8-r128-h1536-small-1stage-52",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=Exact(1536),
        ),
        # Rechecked on 56-SM / MUSA 5.2.0 with graph batch=8, random
        # rank-distinct inputs, both execution orders, and a second seed.
        # The small-row path avoids the generic 512-thread route, while the
        # one-stage token path removes an unnecessary second synchronization.
        priority=1600,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 192,
            "blocks": 128,
            "launch_max_blocks": 128,
            "token_2stage": False,
            "small_rows_module": True,
            "small_rows_flag": 1,
            "row_push": False,
            "row_warp_inv_rms": 1,
            "row_shared_inv_rms": False,
            "sums_bypass_load": 3,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
            "tmp_with_residual": 0,
        },
    ),
    DispatchRule(
        "validated-56-tp4-r1-h1536-small-push-52",
        Match(
            mp_count=Exact(56),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(1536),
        ),
        # Rechecked on 56-SM / MUSA 5.2.0, both graph orders, random input.
        priority=1600,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 192,
            "blocks": 1,
            "launch_max_blocks": 1,
            "token_2stage": True,
            "small_rows_module": True,
            "small_rows_flag": 1,
            "row_push": True,
            "row_warp_inv_rms": 1,
            "row_shared_inv_rms": False,
            "sums_bypass_load": 3,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
            "tmp_with_residual": 0,
        },
    ),
    DispatchRule(
        "validated-56-tp4-r128-h1536-small-1stage-52",
        Match(
            mp_count=Exact(56),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=Exact(1536),
        ),
        # Rechecked on 56-SM / MUSA 5.2.0, both graph orders, random input.
        priority=1600,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 192,
            "blocks": 128,
            "launch_max_blocks": 128,
            "token_2stage": False,
            "small_rows_module": True,
            "small_rows_flag": 1,
            "row_push": False,
            "row_warp_inv_rms": 1,
            "row_shared_inv_rms": False,
            "sums_bypass_load": 3,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
            "tmp_with_residual": 0,
        },
    ),
    DispatchRule(
        "validated-56-tp8-r1024-4096-16384-h1536-row104",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=OneOf(frozenset({1024, 4096, 16384})),
            hidden_size=Exact(1536),
        ),
        # 1024/4096 rechecked on 5.2.0; all three shapes also checked on
        # driver 3.3.8 / runtime 4.3.5 in both graph orders with random input.
        priority=1600,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 512,
            "blocks": 104,
            "launch_max_blocks": 104,
            "token_2stage": True,
            "small_rows_module": False,
            "small_rows_flag": 0,
            "row_push": False,
            "row_warp_inv_rms": 1,
            "row_shared_inv_rms": False,
            "sums_bypass_load": 3,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
            "tmp_with_residual": 0,
        },
    ),
    DispatchRule(
        "validated-56-tp8-r128-h5120-row128-52",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=Exact(5120),
        ),
        # Rechecked on 56-SM / MUSA 5.2.0 with graph batch=8, both orders,
        # seeds 41 and 23.  The 128-CTA row route avoids the generic
        # 64-CTA configuration, which was ~31% slower than composed.
        priority=1600,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 512,
            "blocks": 128,
            "launch_max_blocks": 128,
            "token_2stage": True,
            "small_rows_module": False,
            "small_rows_flag": 0,
            "row_push": False,
            "row_warp_inv_rms": 1,
            "row_shared_inv_rms": False,
            "sums_bypass_load": 3,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
            "tmp_with_residual": 0,
        },
    ),
    DispatchRule(
        "validated-60-tp4-r1-h1536-small-push-335",
        Match(
            mp_count=Exact(60),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(1536),
        ),
        # Rechecked on 60-SM / MUSA 3.3.5, graph batch=8, both orders.
        priority=1600,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 192,
            "blocks": 1,
            "launch_max_blocks": 1,
            "token_2stage": True,
            "small_rows_module": True,
            "small_rows_flag": 1,
            "row_push": True,
            "row_warp_inv_rms": 1,
            "sums_bypass_load": 3,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
            "tmp_with_residual": 0,
        },
    ),
    *(
        DispatchRule(
            name,
            Match(
                mp_count=Exact(60),
                world_size=tp,
                mode=_FUSED,
                rows=rows,
                hidden_size=Exact(1536),
            ),
            priority=1600,
            patch=patch,
        )
        for name, tp, rows, patch in (
            (
                "validated-60-tp4-r128-h1536-small-335",
                Exact(4),
                Exact(128),
                {
                    "kernel": CustomARKernel.ROW_ONE,
                    "threads": 192,
                    "blocks": 128,
                    "launch_max_blocks": 128,
                    "token_2stage": False,
                    "small_rows_module": True,
                    "small_rows_flag": 1,
                    "row_push": False,
                    "row_warp_inv_rms": 1,
                    "sums_bypass_load": 3,
                    "row_skip_end_barrier": True,
                    "row_skip_self_signal": True,
                    "tmp_with_residual": 0,
                },
            ),
            (
                "validated-60-tp8-r128-h1536-small-335",
                Exact(8),
                Exact(128),
                {
                    "kernel": CustomARKernel.ROW_ONE,
                    "threads": 192,
                    "blocks": 128,
                    "launch_max_blocks": 128,
                    "token_2stage": False,
                    "small_rows_module": True,
                    "small_rows_flag": 1,
                    "row_push": False,
                    "row_warp_inv_rms": 1,
                    "sums_bypass_load": 3,
                    "row_skip_end_barrier": True,
                    "row_skip_self_signal": True,
                    "tmp_with_residual": 0,
                },
            ),
        )
    ),
    DispatchRule(
        "validated-60-tp8-r1024-4096-16384-h1536-row104",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=OneOf(frozenset({1024, 4096, 16384})),
            hidden_size=Exact(1536),
        ),
        # Rechecked on driver 3.3.8 / runtime 4.3.5, graph batch=8,
        # both orders with random input. Only measured rows are promoted.
        priority=1600,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 512,
            "blocks": 104,
            "launch_max_blocks": 104,
            "token_2stage": True,
            "small_rows_module": False,
            "small_rows_flag": 0,
            "row_push": False,
            "row_warp_inv_rms": 1,
            "sums_bypass_load": 3,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
            "tmp_with_residual": 0,
        },
    ),
    DispatchRule(
        "validated-56-tp8-r16384-65536-h6144-row104-weight-shared",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=OneOf(frozenset({16384, 65536})),
            hidden_size=Exact(6144),
        ),
        # Rechecked on 56-SM / MUSA 3.3.5 graph in both execution orders.
        priority=1500,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 768,
            "blocks": 104,
            "launch_max_blocks": 104,
            "row_weight_cache": True,
            "row_shared_inv_rms": True,
            "fast_publish": False,
        },
    ),
    DispatchRule(
        "validated-56-tp8-r128-h3584-cta104-warp-skip",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=Exact(3584),
        ),
        # Rechecked on 56-SM / MUSA 3.3.5 graph in both execution orders.
        priority=1500,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 448,
            "blocks": 104,
            "launch_max_blocks": 104,
            "row_warp_inv_rms": 1,
            "row_skip_end_barrier": True,
            "fast_publish": False,
        },
    ),
    DispatchRule(
        "validated-56-tp8-r128-h6144-cta56-weight-shared-skip",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=Exact(6144),
        ),
        # Rechecked on 56-SM / MUSA 3.3.5 graph in both execution orders.
        priority=1500,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 768,
            "blocks": 56,
            "launch_max_blocks": 56,
            "row_weight_cache": True,
            "row_shared_inv_rms": True,
            "row_skip_end_barrier": True,
            "fast_publish": False,
        },
    ),
    DispatchRule(
        "validated-56-tp8-r128-h2560-cta104-warp-skip",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=Exact(2560),
        ),
        # Rechecked on 56-SM / MUSA 3.3.5 graph in both execution orders.
        priority=1500,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 320,
            "blocks": 104,
            "launch_max_blocks": 104,
            "row_warp_inv_rms": 1,
            "row_skip_end_barrier": True,
            "fast_publish": False,
        },
    ),
    DispatchRule(
        "validated-56-tp8-r1-h4096-cta1-t512-skip",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(4096),
        ),
        # Rechecked on 56-SM / MUSA 3.3.5 graph in both execution orders.
        priority=1500,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 512,
            "blocks": 1,
            "launch_max_blocks": 1,
            "row_skip_end_barrier": True,
            "fast_publish": False,
        },
    ),
    DispatchRule(
        "validated-56-tp8-r1-h3584-cta1-t448-one-stage-skip",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(3584),
        ),
        # Rechecked on 56-SM / MUSA 3.3.5 graph in both execution orders.
        # One packed CTA avoids idle lanes and the one-stage path removes the
        # unnecessary token barrier for this decode shape.
        priority=1500,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 448,
            "blocks": 1,
            "launch_max_blocks": 1,
            "token_2stage": False,
            "row_skip_end_barrier": True,
            "fast_publish": False,
        },
    ),
    DispatchRule(
        "validated-60-tp8-r1-h2560-small-warp-bypass-skip",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(2560),
        ),
        # Rechecked with 32 calls per graph and 300 samples in both orders
        # on 60-SM / MUSA 3.3.5. Push removes the non-push row regression.
        priority=1500,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 320,
            "blocks": 1,
            "launch_max_blocks": 1,
            "small_rows_module": True,
            "small_rows_flag": 1,
            "row_warp_inv_rms": 1,
            "sums_bypass_load": 3,
            "row_skip_end_barrier": True,
            "row_push": True,
        },
    ),
    DispatchRule(
        "validated-60-tp8-r1-h1024-fused-push",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(1024),
        ),
        priority=1300,
        patch={
            "threads": 128,
            "blocks": 1,
            "launch_max_blocks": 1,
            "row_push": True,
        },
    ),
    DispatchRule(
        "validated-60-tp8-r1-h2048-small-warp-bypass-skip",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(2048),
        ),
        # Rechecked on 60-SM / MUSA 3.3.5 with 32 calls per graph, 300
        # samples, and both execution orders. Push avoids the measured
        # short-row regression; keep one packed row per thread block.
        priority=1300,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 256,
            "blocks": 1,
            "launch_max_blocks": 1,
            "small_rows_module": True,
            "small_rows_flag": 1,
            "row_warp_inv_rms": 1,
            "sums_bypass_load": 3,
            "row_skip_end_barrier": True,
            "row_push": True,
        },
    ),
    DispatchRule(
        "validated-60-tp8-r1-h3072-fused-push",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(3072),
        ),
        # Rechecked on 60-SM / MUSA 3.3.5 graph with 1500 iterations in both
        # execution orders.  Match the packed row width and publish the local
        # reduction directly to peers to keep fused ahead of composed.
        priority=1300,
        patch={
            "threads": 384,
            "blocks": 1,
            "launch_max_blocks": 1,
            "row_shared_inv_rms": True,
            "row_skip_end_barrier": True,
            "row_push": True,
        },
    ),
    DispatchRule(
        "validated-60-tp8-r1-h8192-shared-inv-skip-end",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(8192),
        ),
        priority=1300,
        patch={
            "threads": 1024,
            "row_push": True,
            "row_shared_inv_rms": True,
            "row_skip_end_barrier": True,
        },
    ),
    DispatchRule(
        "validated-60-tp8-r1-h4096-fused-push",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(4096),
        ),
        priority=1300,
        patch={
            "threads": 1024,
            "blocks": 1,
            "launch_max_blocks": 1,
            "row_skip_end_barrier": True,
            "row_push": True,
        },
    ),
    DispatchRule(
        "validated-60-tp8-r1-h6144-small-warp-bypass-skip",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(6144),
        ),
        priority=1300,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 768,
            "blocks": 1,
            "launch_max_blocks": 1,
            "small_rows_module": True,
            "small_rows_flag": 1,
            "row_warp_inv_rms": 1,
            "sums_bypass_load": 3,
            "row_skip_end_barrier": True,
            "row_push": True,
        },
    ),
    DispatchRule(
        "validated-60-tp8-r1-h3584-small-warp-bypass-skip",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(3584),
        ),
        priority=1300,
        patch={
            "small_rows_module": True,
            "small_rows_flag": 1,
            "row_warp_inv_rms": 1,
            "sums_bypass_load": 3,
            "row_shared_inv_rms": True,
            "row_skip_end_barrier": True,
            "row_push": True,
        },
    ),
    DispatchRule(
        "validated-60-tp8-r1-h5120-fused-push",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(5120),
        ),
        priority=1310,
        patch={
            "threads": 1024,
            "blocks": 1,
            "launch_max_blocks": 1,
            "row_shared_inv_rms": False,
            "row_warp_inv_rms": 1,
            "row_skip_end_barrier": True,
            "sums_bypass_load": 3,
            "row_push": True,
        },
    ),
    DispatchRule(
        "validated-60-tp8-r1-h7168-fused-push",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(7168),
        ),
        priority=1300,
        patch={
            "threads": 1024,
            "blocks": 1,
            "launch_max_blocks": 1,
            "row_skip_end_barrier": True,
            "row_push": True,
        },
    ),
    DispatchRule(
        "validated-60-tp8-r1024-h1024-no-small-skip-end",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1024),
            hidden_size=Exact(1024),
        ),
        priority=1300,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 120,
            "launch_max_blocks": 120,
            "small_rows_module": False,
            "small_rows_flag": 0,
            "row_skip_end_barrier": True,
        },
    ),
    # Exact shapes validated against production composed AR in both graph
    # execution orders. Other rows/hidden sizes retain their existing route.
    *(
        DispatchRule(
            f"validated-60-tp4-r1-h{hidden}-push-row",
            Match(
                mp_count=Exact(60),
                world_size=Exact(4),
                mode=_FUSED,
                rows=Exact(1),
                hidden_size=Exact(hidden),
            ),
            priority=1300,
            patch={
                "kernel": CustomARKernel.ROW_ONE,
                "threads": threads,
                "token_2stage": True,
                "blocks": 1,
                "launch_max_blocks": 1,
                "small_rows_module": True,
                "small_rows_flag": 1,
                "row_warp_inv_rms": 1,
                "sums_bypass_load": 3,
                "row_skip_end_barrier": True,
                "row_push": True,
                # Push scratch stores the reduction input only. Each rank
                # adds its own residual after reading the local scratch.
                "tmp_with_residual": 0,
            },
        )
        for hidden, threads in (
            (1024, 128),
            (2048, 256),
            (2560, 320),
            (3072, 384),
            (3584, 448),
            (4096, 512),
            (5120, 640),
            (6144, 768),
            (7168, 896),
            (8192, 1024),
        )
    ),
    DispatchRule(
        "validated-60-tp2-r1-h1024-push-row",
        Match(
            mp_count=Exact(60),
            world_size=Exact(2),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(1024),
        ),
        priority=1300,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 128,
            "token_2stage": True,
            "blocks": 1,
            "launch_max_blocks": 1,
            "small_rows_module": True,
            "small_rows_flag": 1,
            "row_warp_inv_rms": 1,
            "sums_bypass_load": 3,
            "row_skip_end_barrier": True,
            "row_push": True,
            "tmp_with_residual": 0,
        },
    ),
    DispatchRule(
        "validated-60-tp8-r4096-h3072-row",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(4096),
            hidden_size=Exact(3072),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 64,
            "launch_max_blocks": 64,
        },
    ),
    DispatchRule(
        "candidate-56-tp4-r128-h2048-small-row",
        Match(
            mp_count=Exact(56),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=Exact(2048),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 128,
            "launch_max_blocks": 128,
            "small_rows_module": True,
            "small_rows_flag": 1,
        },
    ),
    DispatchRule(
        "candidate-56-tp8-r1024-4096-h6144-7168-row104",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=OneOf(frozenset({1024, 4096})),
            hidden_size=OneOf(frozenset({6144, 7168})),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 104,
            "launch_max_blocks": 104,
        },
    ),
    DispatchRule(
        "candidate-60-tp8-r1024-h1024-small-row",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1024),
            hidden_size=Exact(1024),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 120,
            "launch_max_blocks": 120,
            "small_rows_module": True,
            "small_rows_flag": 1,
        },
    ),
    DispatchRule(
        "candidate-60-tp8-r1024-h2560-3072-row",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1024),
            hidden_size=OneOf(frozenset({2560, 3072})),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 120,
            "launch_max_blocks": 120,
        },
    ),
    DispatchRule(
        "candidate-60-tp8-r1024-h3584-5120-6144-7168-row",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1024),
            hidden_size=OneOf(frozenset({3584, 5120, 6144, 7168})),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 120,
            "launch_max_blocks": 120,
        },
    ),
    DispatchRule(
        "candidate-60-tp8-r128-h3072-4096-row",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=OneOf(frozenset({3072, 4096})),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 128,
            "launch_max_blocks": 128,
        },
    ),
    DispatchRule(
        "validated-60-tp8-r128-h6144-weight-shared-skip",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=Exact(6144),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 768,
            "blocks": 56,
            "launch_max_blocks": 56,
            "row_weight_cache": True,
            "row_weight_cache_min_rows": 128,
            "row_shared_inv_rms": True,
            "row_skip_end_barrier": True,
        },
    ),
    DispatchRule(
        "candidate-60-tp8-r128-h8192-row-cta60",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=Exact(8192),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 60,
            "launch_max_blocks": 60,
        },
    ),
    DispatchRule(
        "candidate-60-tp8-r128-h5120-row-cta60",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=Exact(5120),
        ),
        priority=1200,
        # CTA56 with a shared inverse-RMS value and a skipped end barrier
        # removes the small-row synchronization tail on 60-SM / MUSA 3.3.5.
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 56,
            "launch_max_blocks": 56,
            "row_shared_inv_rms": 1,
            "row_skip_end_barrier": True,
        },
    ),
    DispatchRule(
        "candidate-60-tp8-r128-h7168-row-cta60",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=Exact(7168),
        ),
        priority=1200,
        # On 60-SM / MUSA 3.3.5 graph runs, 60 CTAs avoids the idle CTA
        # tail of the generic 128-CTA row kernel for this hidden size.
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 60,
            "launch_max_blocks": 60,
        },
    ),
    DispatchRule(
        "validated-60-tp8-r128-h2560-cta64-warp-skip",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=Exact(2560),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 320,
            "blocks": 64,
            "launch_max_blocks": 64,
            "row_warp_inv_rms": 1,
            "row_skip_end_barrier": True,
            "sums_bypass_load": 3,
        },
    ),
    DispatchRule(
        "candidate-60-tp4-r128-h2048-small-row",
        Match(
            mp_count=Exact(60),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=Exact(2048),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 128,
            "launch_max_blocks": 128,
            "small_rows_module": True,
            "small_rows_flag": 1,
        },
    ),
    DispatchRule(
        "candidate-60-tp2-r1-row-local",
        Match(
            mp_count=Exact(60),
            world_size=Exact(2),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=OneOf(
                frozenset({1024, 2048, 2560, 3072, 3584, 5120, 6144, 7168, 8192})
            ),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 1,
            "launch_max_blocks": 1,
            "small_rows_module": True,
            "small_rows_flag": 1,
        },
    ),
    DispatchRule(
        "candidate-60-tp4-r1-h3584-7168-row",
        Match(
            mp_count=Exact(60),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=OneOf(frozenset({3584, 7168})),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 1,
            "launch_max_blocks": 1,
        },
    ),
    DispatchRule(
        "candidate-60-tp4-r1-h1024-2048-small-row",
        Match(
            mp_count=Exact(60),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=OneOf(frozenset({1024, 2048})),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 1,
            "launch_max_blocks": 1,
            "small_rows_module": True,
            "small_rows_flag": 1,
        },
    ),
    DispatchRule(
        "candidate-60-tp4-r1-h4096-5120-6144-row",
        Match(
            mp_count=Exact(60),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=OneOf(frozenset({4096, 5120, 6144})),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 1,
            "launch_max_blocks": 1,
        },
    ),
    DispatchRule(
        "candidate-60-tp4-tp8-r1-h8192-row-local",
        Match(
            mp_count=Exact(60),
            world_size=OneOf(frozenset({4, 8})),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(8192),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 1,
            "launch_max_blocks": 1,
        },
    ),
    DispatchRule(
        "candidate-60-tp8-r1-row-local",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=OneOf(
                frozenset({1024, 2048, 3072, 3584, 4096, 5120, 6144, 7168})
            ),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 1,
            "launch_max_blocks": 1,
            "small_rows_module": True,
            "small_rows_flag": 1,
        },
    ),
    DispatchRule(
        "validated-60-tp8-r128-h1024-row128-warp-skip",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=Exact(1024),
        ),
        priority=1200,
        # One CTA per row with no idle packed lanes is faster in both graph
        # execution orders on 60-SM / MUSA 3.3.5.
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 128,
            "blocks": 128,
            "launch_max_blocks": 128,
            "small_rows_module": False,
            "small_rows_flag": 0,
            "row_shared_inv_rms": False,
            "tmp_with_residual": 0,
            "row_warp_inv_rms": 1,
            "row_skip_end_barrier": True,
            "sums_bypass_load": 3,
        },
    ),
    DispatchRule(
        "candidate-60-tp8-r128-h2048-row-cta120",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=Exact(2048),
        ),
        priority=1200,
        # 120 CTAs with the regular row kernel is faster than the small-row
        # route for this shape on the 60-SM / MUSA 3.3.5 graph path.
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 120,
            "launch_max_blocks": 120,
            "small_rows_module": False,
            "small_rows_flag": 0,
            "row_shared_inv_rms": False,
            "tmp_with_residual": 0,
            "sums_bypass_load": 0,
        },
    ),
    DispatchRule(
        "validated-60-tp8-r128-h3584-cta64-warp-skip",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=Exact(3584),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "threads": 448,
            "blocks": 64,
            "launch_max_blocks": 64,
            "row_warp_inv_rms": 1,
            "row_skip_end_barrier": True,
        },
    ),
    DispatchRule(
        "validated-60-tp8-r4096-h3584-5120-6144-7168-row",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(4096),
            hidden_size=OneOf(frozenset({3584, 5120, 6144, 7168})),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 120,
            "launch_max_blocks": 120,
        },
    ),
    DispatchRule(
        "validated-60-tp8-r4096-16384-65536-h6144-row120-weight-cache",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=OneOf(frozenset({4096, 16384, 65536})),
            hidden_size=Exact(6144),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 120,
            "launch_max_blocks": 120,
            "row_weight_cache": 1,
            "row_shared_inv_rms": 1,
        },
    ),
    DispatchRule(
        "validated-60-tp8-r4096-16384-65536-h8192-row120-sums3",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=OneOf(frozenset({4096, 16384, 65536})),
            hidden_size=Exact(8192),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 120,
            "launch_max_blocks": 120,
            "sums_bypass_load": 3,
        },
    ),
    DispatchRule(
        "candidate-60-tp8-r16384-65536-h3072-row120",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=OneOf(frozenset({16384, 65536})),
            hidden_size=Exact(3072),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 120,
            "launch_max_blocks": 120,
        },
    ),
    DispatchRule(
        "validated-60-tp8-r4096-16384-65536-h2560-3072-3584-4096-5120-7168-row120",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=OneOf(frozenset({4096, 16384, 65536})),
            hidden_size=OneOf(frozenset({2560, 3584, 4096, 5120, 7168})),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 120,
            "launch_max_blocks": 120,
        },
    ),
    DispatchRule(
        "validated-56-tp8-r128-h3072-4096-7168-8192-row104",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=OneOf(frozenset({3072, 4096, 7168, 8192})),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 104,
            "launch_max_blocks": 104,
        },
    ),
    DispatchRule(
        "validated-56-tp8-r1024-4096-16384-65536-h2560-3072-3584-4096-5120-row104",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=OneOf(frozenset({1024, 4096, 16384, 65536})),
            hidden_size=OneOf(frozenset({2560, 3072, 3584, 4096, 5120})),
        ),
        priority=1200,
        patch={
            "kernel": CustomARKernel.ROW_ONE,
            "blocks": 104,
            "launch_max_blocks": 104,
        },
    ),
)


# Kernel selection rules. A config selects the implementation; a patch only
# changes an orthogonal launch/JIT knob. The final result always has one kernel.
CUSTOM_AR_DISPATCH_RULES = (
    *_MEASURED_FALLBACK_RULES,
    *_VALIDATED_ROW_PROMOTIONS,
    DispatchRule(
        "tp8-row",
        Match(
            world_size=Exact(8), mode=_FUSED, rows=IntRange(1), hidden_size=_POW2_HIDDEN
        ),
        KernelConfig(CustomARKernel.ROW_ONE),
    ),
    DispatchRule(
        "tp8-medium-row-cta64-56sm",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=OneOf(frozenset({2560, 3072, 3584, 5120, 6144, 7168})),
        ),
        priority=300,
        patch={"blocks": 64, "launch_max_blocks": 64},
    ),
    DispatchRule(
        "tp8-medium-row-cta80-h7168-56sm",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=Exact(7168),
        ),
        priority=390,
        patch={"blocks": 80, "launch_max_blocks": 80},
    ),
    DispatchRule(
        "tp8-large-row-cta56-nonpow-56sm",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1024),
            hidden_size=OneOf(frozenset({2560, 3072, 3584, 4096, 5120, 6144, 7168})),
        ),
        priority=350,
        patch={"blocks": 56, "launch_max_blocks": 56},
    ),
    DispatchRule(
        "tp8-max-row-cta56-nonpow-56sm",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(65536),
            hidden_size=OneOf(frozenset({2560, 3072, 3584, 5120, 6144, 7168})),
        ),
        priority=350,
        patch={"blocks": 56, "launch_max_blocks": 56},
    ),
    DispatchRule(
        "tp8-large-row-cta56-nonpow-r16384-56sm",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(16384),
            hidden_size=OneOf(frozenset({2560, 3072, 3584, 5120, 6144, 7168})),
        ),
        priority=350,
        patch={"blocks": 56, "launch_max_blocks": 56},
    ),
    DispatchRule(
        "tp8-large-row-cta56-nonpow-r4096-56sm",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(4096),
            hidden_size=OneOf(frozenset({2560, 3072, 7168})),
        ),
        priority=350,
        patch={"blocks": 56, "launch_max_blocks": 56},
    ),
    DispatchRule(
        "tp4-decode-cta80-h5120-56sm",
        Match(
            mp_count=Exact(56),
            world_size=Exact(4),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(5120),
        ),
        priority=350,
        patch={"blocks": 80, "launch_max_blocks": 80},
    ),
    DispatchRule(
        "tp8-row-no-vec2-h6144-56sm",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(6144),
        ),
        priority=350,
        patch={"vec2_non8": 0},
    ),
    DispatchRule(
        "tp8-row-no-vec2-h5120-56sm",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(5120),
        ),
        priority=350,
        patch={"vec2_non8": 0},
    ),
    DispatchRule(
        "tp8-decode-cta64-h4096-60sm",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(4096),
        ),
        priority=350,
        patch={"blocks": 64, "launch_max_blocks": 64},
    ),
    DispatchRule(
        "tp8-decode-cta64-h7168-60sm",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(7168),
        ),
        priority=350,
        patch={"blocks": 64, "launch_max_blocks": 64},
    ),
    DispatchRule(
        "tp2-decode-shared-inv-rms-60sm",
        Match(
            mp_count=Exact(60),
            world_size=Exact(2),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=OneOf(frozenset({2048, 3072, 3584, 5120, 6144})),
        ),
        priority=300,
        patch={"row_shared_inv_rms": 1},
    ),
    DispatchRule(
        "tp2-decode-blocks8-h1024-60sm",
        Match(
            mp_count=Exact(60),
            world_size=Exact(2),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(1024),
        ),
        priority=350,
        patch={"blocks": 8, "launch_max_blocks": 8},
    ),
    DispatchRule(
        "tp2-decode-blocks8-56sm",
        Match(
            mp_count=Exact(56),
            world_size=Exact(2),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=OneOf(
                frozenset({1024, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192})
            ),
        ),
        priority=350,
        patch={"blocks": 8, "launch_max_blocks": 8},
    ),
    # 56-SM TP8 has fewer resident SMs than the 60-SM tuning target.  The
    # generic 120-CTA launch leaves a long tail for h8192 small-row traffic;
    # use the measured 56-SM CTA/signal configuration instead.
    DispatchRule(
        "tp8-h8192-r1-56sm-skip",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(8192),
        ),
        priority=1800,
        patch={
            "blocks": 56,
            "launch_max_blocks": 56,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
            "sums_bypass_load": 3,
        },
    ),
    # On 56-SM TP2, the default two-stage reduction leaves a long tail for
    # the Qwen3.5 local-hidden=2048 decode shapes.  The measured one-stage
    # packed/vec2rank path is faster for rows 1/4/64 and remains correct.
    DispatchRule(
        "tp2-h2048-r1-56sm-one-stage",
        Match(
            mp_count=Exact(56),
            world_size=Exact(2),
            mode=_FUSED,
            rows=Exact(1),
            hidden_size=Exact(2048),
        ),
        priority=1900,
        patch={
            "token_2stage": 0,
            "packed_1stage": 1,
            "vec2rank_1stage": 1,
        },
    ),
    DispatchRule(
        "tp2-h2048-r4-56sm-one-stage",
        Match(
            mp_count=Exact(56),
            world_size=Exact(2),
            mode=_FUSED,
            rows=Exact(4),
            hidden_size=Exact(2048),
        ),
        priority=1900,
        patch={
            "token_2stage": 0,
            "packed_1stage": 1,
            "vec2rank_1stage": 1,
        },
    ),
    DispatchRule(
        "tp2-h2048-r64-56sm-one-stage",
        Match(
            mp_count=Exact(56),
            world_size=Exact(2),
            mode=_FUSED,
            rows=Exact(64),
            hidden_size=Exact(2048),
        ),
        priority=1900,
        patch={
            "token_2stage": 0,
            "packed_1stage": 1,
            "vec2rank_1stage": 1,
        },
    ),
    DispatchRule(
        "tp8-h8192-r4-56sm-skip",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(4),
            hidden_size=Exact(8192),
        ),
        priority=1800,
        patch={
            "blocks": 56,
            "launch_max_blocks": 56,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
            "sums_bypass_load": 3,
        },
    ),
    DispatchRule(
        "tp8-h8192-r16-56sm-skip",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(16),
            hidden_size=Exact(8192),
        ),
        priority=1800,
        patch={
            "blocks": 56,
            "launch_max_blocks": 56,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
            "sums_bypass_load": 3,
        },
    ),
    DispatchRule(
        "tp8-h8192-r64-56sm-skip",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(64),
            hidden_size=Exact(8192),
        ),
        priority=1800,
        patch={
            "blocks": 48,
            "launch_max_blocks": 48,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
            "sums_bypass_load": 3,
        },
    ),
    DispatchRule(
        "tp8-h8192-r128-56sm-skip",
        Match(
            mp_count=Exact(56),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(128),
            hidden_size=Exact(8192),
        ),
        priority=1800,
        patch={
            "blocks": 56,
            "launch_max_blocks": 56,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
            "sums_bypass_load": 3,
        },
    ),
    DispatchRule(
        "tp8-h8192-r64-60sm-skip",
        Match(
            mp_count=Exact(60),
            world_size=Exact(8),
            mode=_FUSED,
            rows=Exact(64),
            hidden_size=Exact(8192),
        ),
        priority=1800,
        patch={
            "blocks": 56,
            "launch_max_blocks": 56,
            "row_skip_end_barrier": True,
            "row_skip_self_signal": True,
            "sums_bypass_load": 3,
        },
    ),
)


_TOKEN_STAGE_RULES = (
    (Match(world_size=Exact(4), rows=Exact(1), hidden_size=Exact(1024)), 0),
    # TP2 large non-power-of-two rows are bandwidth-bound; the two-stage
    # fused kernel avoids the long one-stage reduction tail on MUSA.
    (Match(world_size=Exact(2), hidden_size=Exact(7168)), 1),
    (
        Match(
            world_size=Exact(2),
            rows=IntRange(1, 129),
            hidden_size=OneOf(frozenset((2048, 4096))),
        ),
        1,
    ),
    (Match(world_size=Exact(4), hidden_size=Exact(3072)), 0),
    (Match(world_size=OneOf(frozenset((4, 8)))), 1),
)


def resolve_token_2stage(
    world_size: int, row_hidden: int, override: int = -1, row_count: int = 1
) -> int:
    """Resolve the fused reduction stage from the complete dispatch key."""
    if override >= 0:
        return int(override)
    key = DispatchKey(
        0, world_size, "fused_rmsnorm", max(1, int(row_count)), row_hidden, True, False
    )
    for match, value in _TOKEN_STAGE_RULES:
        if match.matches(key):
            return value
    return 0


def _valid_key(key: DispatchKey) -> bool:
    return (
        key.mode == "fused_rmsnorm"
        and key.world_size in (2, 4, 6, 8)
        and 1 <= key.rows <= 131072
        and 0 < key.hidden_size <= 16384
        and key.hidden_size % 8 == 0
    )


def _base_config(key: DispatchKey) -> KernelConfig:
    return KernelConfig(
        token_2stage=key.token_2stage,
        graph_rank_data=key.graph_rank_data,
        fast_publish=key.mp_count == 56 and key.world_size == 8,
    )


def resolve_custom_ar_dispatch(key: DispatchKey) -> KernelConfig | None:
    if not _valid_key(key):
        return None
    config = _base_config(key)
    for rule in sorted(CUSTOM_AR_DISPATCH_RULES, key=lambda item: item.order):
        if not rule.match.matches(key):
            continue
        if rule.config is not None:
            # A config is a complete declarative override.  Keep base values
            # (notably architecture-derived fast_publish) unless the rule
            # explicitly differs from KernelConfig defaults.
            defaults = KernelConfig()
            updates = {
                field.name: getattr(rule.config, field.name)
                for field in fields(KernelConfig)
                if getattr(rule.config, field.name) != getattr(defaults, field.name)
            }
            config = replace(config, **updates)
        if rule.patch:
            config = replace(config, **dict(rule.patch))
    return config


def resolve_fused_rmsnorm_compile_config(
    *,
    world_size: int,
    mp_count: int,
    row_hidden: int,
    short_rows: bool,
    small_rows: bool,
    cache_policy: int,
    token_2stage: int,
    row_warp_inv_rms_override: int,
    fast_publish_override: int,
    row_count: int = 0,
    base_threads: int = 512,
    base_blocks: int = 120,
    base_max_blocks: int = 120,
    base_launch_max_blocks: int = 120,
    base_dynamic_blocks: int = 1,
) -> KernelConfig:
    """Resolve compile-only switches using the same dispatch vocabulary."""
    key = DispatchKey(
        mp_count=mp_count,
        world_size=world_size,
        mode="fused_rmsnorm",
        rows=(int(row_count) if row_count > 0 else (1 if short_rows else 1024)),
        hidden_size=row_hidden,
        token_2stage=bool(token_2stage),
        graph_rank_data=True,
    )
    config = resolve_custom_ar_dispatch(key) or KernelConfig(
        token_2stage=bool(token_2stage)
    )
    no_cache = cache_policy == 1
    is_power_of_two = row_hidden > 0 and (row_hidden & (row_hidden - 1)) == 0
    if (
        small_rows
        and world_size in (2, 4, 8)
        and (world_size == 2 or token_2stage)
        and row_hidden in (512, 1024, 2048, 4096)
    ):
        threads = row_hidden // 8
    elif (
        world_size == 2
        and short_rows
        and 512 <= row_hidden <= 8192
        and row_hidden % 8 == 0
        and not is_power_of_two
    ):
        threads = row_hidden // 8
    elif (
        world_size == 2
        and 6144 <= row_hidden <= 8192
        and row_hidden % 8 == 0
        and not is_power_of_two
    ):
        threads = 512
    elif world_size == 2 and row_hidden == 512:
        threads = 256
    elif world_size == 4 and row_hidden % 8 != 0:
        threads = 512
    elif world_size == 2 and row_hidden == 1024 and short_rows:
        threads = 512
    elif world_size == 4 and token_2stage and row_hidden in (4096, 8192):
        threads = 1024
    elif world_size == 8 and row_hidden == 1536 and not token_2stage and not short_rows:
        threads = 512
    elif world_size == 2 and row_hidden in (1024, 2048):
        threads = row_hidden // 8
    elif world_size == 2 and short_rows:
        threads = 512
    elif world_size == 2:
        threads = 1024
    elif world_size in (4, 8) and token_2stage:
        threads = 512
    else:
        threads = base_threads

    if world_size == 2 and row_hidden == 1024 and not short_rows:
        blocks = 80
    elif world_size == 2 and row_hidden == 2048:
        blocks = 64
    elif world_size == 8 and token_2stage:
        blocks = (
            min(base_launch_max_blocks, 104)
            if base_launch_max_blocks < base_max_blocks
            else 180 if row_hidden == 1536 else 120
        )
    elif world_size == 4 and token_2stage:
        blocks = (
            40
            if row_hidden % 8 != 0
            else (
                120
                if row_hidden == 1536
                else (
                    240
                    if row_hidden == 1024 and short_rows
                    else (
                        80
                        if row_hidden == 4096 and short_rows
                        else (
                            56
                            if row_hidden == 4096
                            else (
                                80
                                if row_hidden in (1024, 2048, 4096, 8192)
                                and not short_rows
                                else 56
                            )
                        )
                    )
                )
            )
        )
    elif world_size == 8 and row_hidden == 1536 and not short_rows:
        blocks = 60
    else:
        blocks = base_blocks
    dispatch_threads = config.threads if config.threads != 512 else None
    dispatch_blocks = config.blocks if config.blocks != 120 else None
    dispatch_max_blocks = config.max_blocks if config.max_blocks != 120 else None
    dispatch_launch_max_blocks = (
        config.launch_max_blocks if config.launch_max_blocks != 120 else None
    )
    dispatch_dynamic_blocks = (
        config.dynamic_blocks if config.dynamic_blocks != 1 else None
    )
    threads = dispatch_threads if dispatch_threads is not None else threads
    blocks = dispatch_blocks if dispatch_blocks is not None else blocks
    max_blocks = (
        dispatch_max_blocks
        if dispatch_max_blocks is not None
        else max(base_max_blocks, blocks)
    )
    dynamic_blocks = (
        dispatch_dynamic_blocks
        if dispatch_dynamic_blocks is not None
        else (
            0
            if (world_size == 8 and token_2stage)
            or (
                world_size == 8
                and row_hidden == 1536
                and not token_2stage
                and not short_rows
            )
            else base_dynamic_blocks
        )
    )
    result = replace(
        config,
        threads=threads,
        blocks=blocks,
        launch_max_blocks=(
            dispatch_launch_max_blocks
            if dispatch_launch_max_blocks is not None
            else blocks
        ),
        max_blocks=max_blocks,
        dynamic_blocks=dynamic_blocks,
        fast_publish=(
            config.fast_publish
            if fast_publish_override < 0
            else bool(fast_publish_override)
        ),
        packed_1stage=int(world_size == 2 and row_hidden == 8192 and not token_2stage),
        vec2rank_1stage=int(
            config.vec2rank_1stage or (world_size == 2 and not token_2stage)
        ),
        partial_packed_non8=int(
            token_2stage and row_hidden > 0 and (row_hidden & (row_hidden - 1)) != 0
        ),
        vec2_non8=int(
            token_2stage and row_hidden > 0 and (row_hidden & (row_hidden - 1)) != 0
        ),
        vec4_non8=int(
            token_2stage
            and row_hidden > 0
            and (row_hidden & (row_hidden - 1)) != 0
            and row_hidden % 4 == 0
        ),
        cache_hidden_limit=(
            0
            if no_cache
            else (
                1536
                if world_size == 8
                and row_hidden == 1536
                and not token_2stage
                and not short_rows
                else (
                    4096
                    if world_size == 2 or (world_size in (4, 8) and token_2stage)
                    else 2048
                )
            )
        ),
        typed_cache_hidden_limit=0 if no_cache else 8192,
        weight_cache_hidden_limit=(
            0
            if (
                world_size == 8
                and row_hidden == 1536
                and not token_2stage
                and not short_rows
            )
            or (world_size == 4 and row_hidden == 8192 and (short_rows or small_rows))
            else 8192
        ),
        row_weight_cache=bool(
            world_size == 8
            and token_2stage
            and (row_hidden == 6144 or (mp_count == 56 and row_hidden == 4096))
        ),
        row_weight_cache_min_rows=(
            4096
            if world_size == 8 and token_2stage and row_hidden == 6144
            else 512 if world_size == 8 and token_2stage else 2048
        ),
        row_shared_inv_rms=bool(
            token_2stage
            and (
                (world_size == 4 and row_hidden == 2048)
                or (world_size == 8 and row_hidden == 1024)
            )
        ),
        row_warp_inv_rms=(
            int(row_warp_inv_rms_override)
            if row_warp_inv_rms_override >= 0
            else config.row_warp_inv_rms
        ),
        sums_bypass_load=(
            3
            if (world_size == 4 and token_2stage)
            or (world_size == 8 and token_2stage and row_hidden in (1536, 8192))
            else 0
        ),
        tmp_with_residual=int(
            token_2stage
            and (
                (
                    world_size == 4
                    and (
                        row_hidden in (512, 1024)
                        or (row_hidden == 4096 and not short_rows)
                    )
                )
                or (
                    world_size == 8
                    and (
                        row_hidden == 1536
                        or (row_hidden in (1024, 2048) and small_rows)
                    )
                )
            )
        ),
        small_rows_flag=int(
            small_rows
            and world_size in (2, 4, 8)
            and (world_size == 2 or token_2stage)
            and row_hidden in (512, 1024, 2048, 4096, 8192)
        ),
    )
    # KernelConfig selects the implementation; only explicit rule patches
    # override the shape-derived compile settings above.  Copying every field
    # from a default KernelConfig here would silently erase settings such as
    # sums_bypass_load, caches, and tmp_with_residual.
    for rule in sorted(CUSTOM_AR_DISPATCH_RULES, key=lambda item: item.order):
        if rule.match.matches(key) and rule.patch:
            result = replace(result, **dict(rule.patch))
    return result


# ---------------------------------------------------------------------------
# Runtime adapter: environment gate and the fused RMSNorm decision entry point.
# ---------------------------------------------------------------------------

DISPATCH_ENV = "VLLM_MUSA_CAR_FUSED_DISPATCH"
FUSED_MODE = "fused_rmsnorm"

_ENABLED_VALUES = frozenset(("1", "true", "yes", "on"))


def dispatch_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return True only when the operator opted into rule-table dispatch."""
    source = os.environ if environ is None else environ
    raw = source.get(DISPATCH_ENV, "")
    return raw.strip().lower() in _ENABLED_VALUES


def resolve_fused_car_rmsnorm(
    *,
    mp_count: int,
    world_size: int,
    rows: int,
    hidden_size: int,
    graph_rank_data: bool = False,
    token_2stage: bool = True,
    environ: Mapping[str, str] | None = None,
) -> KernelConfig | None:
    """Resolve the fused-RMSNorm launch policy, or None when dispatch is off.

    Returning None keeps every caller on its previous code path, so the default
    configuration is unchanged.
    """
    if not dispatch_enabled(environ):
        return None
    key = DispatchKey(
        mp_count=int(mp_count),
        world_size=int(world_size),
        mode=FUSED_MODE,
        rows=int(rows),
        hidden_size=int(hidden_size),
        token_2stage=bool(token_2stage),
        graph_rank_data=bool(graph_rank_data),
    )
    return resolve_custom_ar_dispatch(key)


def decision_label(config: KernelConfig | None) -> str:
    """Stable, compact label used by decision tracing.

    Callers only build a label once dispatch is enabled, so ``None`` here means
    no rule matched the runtime key.
    """
    if config is None:
        return "no-rule"
    return f"{config.kernel.value}{config.name_suffix()}"
