# SPDX-License-Identifier: Apache-2.0
"""The *values* a tactic may hold, one frozen dataclass per kernel.

The table on disk is JSON, so every value arrives as an untyped ``dict``.  A typo
in a key -- ``block_N`` for ``block_n`` -- used to survive the loader and only
show up as "this kernel is mysteriously 2x slower", because the launch-time code
reads the keys it knows and silently gets a default for the rest.  Each kernel
therefore declares its fields once, here, and the loader builds the dataclass
with :meth:`Tactic.from_values`, which rejects a key it does not know and a key
that is missing.

Adding a tactic for a new kernel is a subclass plus one :data:`REGISTRY` entry;
the loader, the ladder and the resolver never change.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, ClassVar, TypeVar

__all__ = [
    "MoeTactic",
    "REGISTRY",
    "RmsnormTactic",
    "Tactic",
    "tactic_class",
]

T = TypeVar("T", bound="Tactic")


class Tactic:
    """Base class: one kernel's numeric tactic, frozen and self-validating."""

    #: The kernel name this tactic belongs to, matching the table's ``kernel``.
    KERNEL: ClassVar[str] = ""

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(field.name for field in fields(cls))

    @classmethod
    def from_values(cls: type[T], values: dict[str, Any], *, where: str = "") -> T:
        """Build from a table's ``values`` mapping, refusing anything unexpected.

        ``where`` is the table location used in the error, so a bad entry names
        the file rather than only the key.
        """
        if not isinstance(values, dict):
            raise ValueError(
                f"{where}: 'values' must be an object, got {type(values).__name__}"
            )
        known = cls.field_names()
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ValueError(
                f"{where}: unknown value key(s) {unknown} for kernel "
                f"{cls.KERNEL!r}; known keys are {list(known)}"
            )
        missing = [name for name in known if name not in values]
        if missing:
            raise ValueError(
                f"{where}: missing value key(s) {missing} for kernel {cls.KERNEL!r}"
            )
        for name in known:
            raw = values[name]
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise ValueError(
                    f"{where}: {name} must be an integer, got {raw!r} "
                    f"({type(raw).__name__})"
                )
        return cls(**{name: values[name] for name in known})

    def to_values(self) -> dict[str, int]:
        """The mapping the table holds, in field order."""
        return {name: getattr(self, name) for name in self.field_names()}

    def __str__(self) -> str:  # pragma: no cover - logging aid
        return str(self.to_values())


def _require_positive(instance: Tactic) -> None:
    for name, value in instance.to_values().items():
        if value <= 0:
            cls = type(instance).__name__
            raise ValueError(f"{cls}.{name} must be positive, got {value}")


@dataclass(frozen=True)
class MoeTactic(Tactic):
    """Expert-block tiling for the fused MoE kernel on one M band."""

    KERNEL: ClassVar[str] = "moe"

    block_m: int
    block_n: int
    block_k: int
    split_k: int
    num_stages: int

    def __post_init__(self) -> None:
        _require_positive(self)


@dataclass(frozen=True)
class RmsnormTactic(Tactic):
    """Row-block and vector width for the RMSNorm kernel."""

    KERNEL: ClassVar[str] = "rmsnorm"

    rows_per_block: int
    vector_width: int

    def __post_init__(self) -> None:
        _require_positive(self)


#: ``kernel name -> tactic class``.  The loader reads it, so a new kernel is a
#: subclass plus this entry and nothing else.
REGISTRY: dict[str, type[Tactic]] = {
    MoeTactic.KERNEL: MoeTactic,
    RmsnormTactic.KERNEL: RmsnormTactic,
}


def tactic_class(kernel: str) -> type[Tactic]:
    """The tactic class for ``kernel``, or raise with the known names."""
    try:
        return REGISTRY[kernel]
    except KeyError:
        raise KeyError(
            f"no tactic class registered for kernel {kernel!r}; known kernels are "
            f"{sorted(REGISTRY)}. Add a Tactic subclass and a REGISTRY entry."
        ) from None
