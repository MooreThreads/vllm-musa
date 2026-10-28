# SPDX-License-Identifier: Apache-2.0
"""The inputs a tuning decision is a pure function of.

A *tuning problem* is everything that can change which kernel tactic is fastest,
frozen so that the answer cannot depend on when it was asked:

* :class:`LayerShape` -- what the layer *is*: the projection's N and K, the
  activation dtype, the weight format, the quantisation group, how many experts
  share the kernel.
* :class:`ComputeSemantics` -- what the caller *requires* of it: dense or MoE,
  whether the accumulation dtype is fixed, whether a split-K reduction is
  allowed at all.
* :class:`DeviceProfile` -- what the *card* is, in the same terms the kernel's
  resource model uses (multiprocessor count, warp width, shared-memory and
  register limits).

Nothing here imports torch or torchada.  ``DeviceProfile.from_spec()`` takes any
object with the matching attributes, which is what lets the whole selection layer
be tested without a GPU and enumerated offline for a bin that is not in the
building machine.

The split is deliberate and is the one thing this module exists to enforce: a
device field never appears in :class:`LayerShape`, and a layer field never
appears in :class:`DeviceProfile`.  Mixing them makes two problems incomparable
across cards above the fold, which is how a "tuned" table ends up encoding the
machine it was measured on instead of the machine it runs on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ComputeSemantics",
    "DeviceProfile",
    "LayerShape",
    "TuningProblem",
]

#: GEMM kinds a tactic can be asked for.  A string rather than an enum on purpose:
#: the tables are JSON and a value read back from them is already a string, so an
#: enum would only add a conversion that can disagree with the file.
DENSE = "dense"
MOE_CONTIGUOUS = "moe_contiguous"
MOE_MASKED = "moe_masked"


@dataclass(frozen=True)
class LayerShape:
    """What one layer is, independent of the card it runs on."""

    #: Output features of the projection (the GEMM's N).
    shape_n: int
    #: Reduction size (the GEMM's K).
    shape_k: int
    #: Activation dtype: ``"bf16"``, ``"fp16"``.
    dtype: str
    #: Weight format the kernel consumes: ``"bf16"``, ``"fp8"``, ``"int4"``.
    weight_dtype: str
    #: Quantisation group size along K, or ``None`` for an unquantised weight.
    group_size: int | None = None
    #: Experts sharing this kernel; ``0`` for a dense layer.
    num_experts: int = 0
    #: Top-k routing width, which decides the per-expert row count; ``0`` for dense.
    top_k: int = 0

    def __post_init__(self) -> None:
        if self.shape_n <= 0 or self.shape_k <= 0:
            raise ValueError(
                f"layer shape must be positive, got N={self.shape_n} K={self.shape_k}"
            )
        if self.num_experts < 0 or self.top_k < 0:
            raise ValueError(
                f"expert counts cannot be negative, got E={self.num_experts} "
                f"top_k={self.top_k}"
            )
        if self.num_experts and self.top_k > self.num_experts:
            raise ValueError(
                f"top_k={self.top_k} exceeds num_experts={self.num_experts}"
            )

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    def expert_rows(self, m: int) -> int:
        """Rows per expert for a batch of ``m`` tokens, rounded up.

        The MoE kernels tile per expert, so this -- not ``m`` -- is the M their
        tile and pipeline depth were chosen for.
        """
        if not self.num_experts:
            return m
        routed = m * max(self.top_k, 1)
        return max(1, -(-routed // self.num_experts))


@dataclass(frozen=True)
class ComputeSemantics:
    """What the caller requires of the kernel, also independent of the card."""

    #: One of :data:`DENSE`, :data:`MOE_CONTIGUOUS`, :data:`MOE_MASKED`.
    gemm_type: str = DENSE
    #: Whether the reduction must accumulate in the activation dtype.  The
    #: batch-invariant path requires it, and it forbids some split-K tactics.
    use_f16_accum: bool = False
    #: Whether a split-K reduction (a second launch) is acceptable at all.
    split_k_allowed: bool = True
    #: How many ranks shard K; the kernel sees K // tp_shard_k.
    tp_shard_k: int = 1

    def __post_init__(self) -> None:
        if self.tp_shard_k < 1:
            raise ValueError(f"tp_shard_k must be >= 1, got {self.tp_shard_k}")
        if self.gemm_type not in (DENSE, MOE_CONTIGUOUS, MOE_MASKED):
            raise ValueError(
                f"unknown gemm_type {self.gemm_type!r}; expected one of "
                f"{DENSE}, {MOE_CONTIGUOUS}, {MOE_MASKED}"
            )


@dataclass(frozen=True)
class DeviceProfile:
    """The device facts a resource model needs, named as torchada names them.

    Field names intentionally mirror ``torchada.musa.device.MusaDeviceSpec`` so a
    reader can trace each one to the property it came from; this class does not
    reinterpret them (in particular it does not collapse the two shared-memory
    limits into one "max", because which one binds depends on whether the kernel
    opts into dynamic shared memory).
    """

    arch_name: str
    mp_count: int
    warp_size: int
    shared_memory_per_block: int
    shared_memory_per_multiprocessor: int
    regs_per_multiprocessor: int
    max_threads_per_block: int
    max_threads_per_multiprocessor: int

    @classmethod
    def from_spec(cls, spec: Any) -> DeviceProfile:
        """Build from a ``MusaDeviceSpec`` (or any object with the same attributes)."""
        try:
            return cls(
                arch_name=spec.arch_name,
                mp_count=int(spec.mp_count),
                warp_size=int(spec.warp_size),
                shared_memory_per_block=int(spec.shared_memory_per_block),
                shared_memory_per_multiprocessor=int(
                    spec.shared_memory_per_multiprocessor
                ),
                regs_per_multiprocessor=int(spec.regs_per_multiprocessor),
                max_threads_per_block=int(spec.max_threads_per_block),
                max_threads_per_multiprocessor=int(spec.max_threads_per_multiprocessor),
            )
        except AttributeError as exc:
            raise TypeError(
                "DeviceProfile.from_spec() needs a device spec with arch_name, "
                "mp_count, warp_size, shared_memory_per_block, "
                "shared_memory_per_multiprocessor, regs_per_multiprocessor, "
                "max_threads_per_block and max_threads_per_multiprocessor; "
                f"{type(spec).__name__} is missing {exc}"
            ) from None


@dataclass(frozen=True)
class TuningProblem:
    """A layer, a batch size and a card: everything a tactic may depend on."""

    layer: LayerShape
    m: int
    compute: ComputeSemantics = field(default_factory=ComputeSemantics)
    device: DeviceProfile | None = None
    #: Whether the launch happens inside a graph capture.  The best tactic can
    #: differ from the eager one (a split-K reduction is one more launch to
    #: capture, and a deeper pipeline costs registers the capture cannot hide),
    #: so it is part of the problem rather than a global the resolver reads.
    graph_capture: bool = False

    def __post_init__(self) -> None:
        if self.m <= 0:
            raise ValueError(f"m must be positive, got {self.m}")

    @property
    def tuning_m(self) -> int:
        """The M a MoE tactic is chosen for: tokens per expert, not tokens."""
        return self.layer.expert_rows(self.m)

    @property
    def arch_name(self) -> str:
        """``""`` when no device was supplied, which matches no arch entry."""
        return self.device.arch_name if self.device is not None else ""

    @property
    def mp_count(self) -> int | None:
        return self.device.mp_count if self.device is not None else None
