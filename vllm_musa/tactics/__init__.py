# SPDX-License-Identifier: Apache-2.0
"""Data-driven kernel tactics.

A *tactic* is a choice that is correct on every arch but fastest on one: tile
shape, block size, split-K factor, pipeline depth.  The code that applies a
tactic is the same everywhere; only the numbers differ.  Keeping the numbers in
``tactics/*.json`` (keyed by arch and multiprocessor count) instead of in ``if``
chains means supporting a new arch is a data change, and a missing entry is a
loud lookup failure rather than a slow default inherited by accident.

The lookup ladder, in order, is:

1. ``(arch_name, mp_count)`` -- the exact bin
2. ``(arch_name, None)``      -- the arch
3. ``(None, None)``           -- the table's own fallback
4. the caller's ``default=``  -- explicit opt-in
5. :class:`TacticNotFound`

Anything past 3 is the caller saying "a generic tactic is fine here"; a caller
that does not say so gets an error rather than an unvalidated guess.  This is
deliberate: on S5000 a wrong tiling is a 2x regression that looks like a
hardware limit, and nobody re-reads a tuning table for an arch they just
enabled.

The numbers are typed per kernel (:class:`MoeTactic`, :class:`RmsnormTactic`).
A mapping of raw values accepts a misspelled key and hands back a default the
caller never chose, which is the same class of bug as a silent fallback, one
level down.  A kernel with no registered type cannot be loaded at all: the type
*is* the schema, and registering one is how a kernel states which knobs it has.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

TACTICS_DIR = Path(__file__).resolve().parent
SCHEMA_VERSION = 1

#: Value an entry may name instead of an arch to mean "last resort".  Only ``null``
#: is written; ``"default"`` is still read so an older table keeps working.
LEGACY_FALLBACK_ARCH = "default"


class TacticNotFound(LookupError):
    """No tactic for the requested device and the caller offered no default."""


@dataclass(frozen=True)
class Tactic:
    """Base class for one kernel's numbers.

    Subclasses declare the knobs as typed fields, and :meth:`from_values` is the
    only way in: an unknown key, a missing knob or a non-integer value fails
    rather than being ignored or coerced.
    """

    @classmethod
    def from_values(cls, values: Mapping[str, Any]) -> Tactic:
        """Build from a table's ``values``, rejecting anything unexpected."""
        expected = [field.name for field in fields(cls)]
        unknown = sorted(set(values) - set(expected))
        if unknown:
            knobs = "knob" if len(unknown) == 1 else "knobs"
            raise ValueError(
                f"{cls.__name__} has no {knobs} "
                f"{', '.join(repr(name) for name in unknown)}; "
                f"it takes [{', '.join(expected)}]"
            )
        missing = sorted(set(expected) - set(values))
        if missing:
            raise ValueError(
                f"{cls.__name__} needs [{', '.join(repr(name) for name in missing)}], "
                "missing from this tactic"
            )
        for name in expected:
            value = values[name]
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"{name}={value!r}: every {cls.__name__} knob is an integer"
                )
        return cls(**{name: values[name] for name in expected})

    def as_dict(self) -> dict[str, int]:
        """The values as a plain mapping, for JSON and for generic callers."""
        return asdict(self)


@dataclass(frozen=True)
class MoeTactic(Tactic):
    """Expert-block tiling for the fused MoE kernel."""

    block_m: int
    block_n: int
    block_k: int
    split_k: int
    num_stages: int


@dataclass(frozen=True)
class RmsnormTactic(Tactic):
    """Row-block size and vector width for the RMSNorm JIT kernel."""

    rows_per_block: int
    vector_width: int


#: Kernel name -> the type its ``values`` must fit.  Registering here is what lets
#: :func:`load` refuse a typo instead of handing the caller a raw mapping.
TACTIC_TYPES: dict[str, type[Tactic]] = {
    "moe": MoeTactic,
    "rmsnorm": RmsnormTactic,
}


@dataclass(frozen=True)
class TacticEntry:
    """One row of a table: a key (arch, bin) and the tactic it selects."""

    arch: str | None
    mp_count: int | None
    tactic: Tactic
    note: str = ""

    @property
    def values(self) -> dict[str, int]:
        """The numbers as a mapping, for a caller that does not want the type."""
        return self.tactic.as_dict()


def _table_path(kernel: str, directory: Path | None) -> Path:
    return (TACTICS_DIR if directory is None else directory) / f"{kernel}.json"


def tactic_type(kernel: str) -> type[Tactic]:
    """The type ``kernel``'s table must fit, or a ``ValueError`` naming the registered ones."""
    try:
        return TACTIC_TYPES[kernel]
    except KeyError:
        raise ValueError(
            "no tactic type registered for kernel {!r}; registered kernels: {}. Add one "
            "to TACTIC_TYPES so this table's values are checked instead of trusted.".format(
                kernel, ", ".join(sorted(TACTIC_TYPES)) or "none"
            )
        ) from None


def _ladder(arch_name: str, mp_count: int | None) -> list[tuple[str | None, int | None]]:
    """The keys to try, most specific first."""
    keys: list[tuple[str | None, int | None]] = [(arch_name, mp_count)]
    if mp_count is not None:
        keys.append((arch_name, None))
    keys.append((None, None))
    return keys


def _labelled(key: tuple[str | None, int | None]) -> str:
    """``(arch, None)`` -> ``"mp_31"``; ``(None, None)`` -> ``"default"``, for messages."""
    arch, mp = key
    if arch is None:
        return LEGACY_FALLBACK_ARCH
    return arch if mp is None else f"{arch}/mp{mp}"


def load(
    kernel: str, directory: Path | None = None
) -> dict[tuple[str | None, int | None], TacticEntry]:
    """Load ``<kernel>.json`` into a ``(arch, mp_count) -> entry`` map."""
    directory = TACTICS_DIR if directory is None else directory
    kind = tactic_type(kernel)
    path = _table_path(kernel, directory)
    if not path.is_file():
        raise TacticNotFound(f"no tactic table for kernel {kernel!r}: {path} is missing")

    with path.open() as handle:
        raw = json.load(handle)

    if raw.get("_schema") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported tactic schema {raw.get('_schema')!r}, "
            f"expected {SCHEMA_VERSION}"
        )
    if raw.get("kernel") != kernel:
        raise ValueError(
            f"{path}: table says kernel {raw.get('kernel')!r}, " f"expected {kernel!r}"
        )

    entries = raw.get("tactics", [])
    if not isinstance(entries, list):
        raise ValueError(f"{path}: 'tactics' must be a list, got {type(entries).__name__}")

    table: dict[tuple[str | None, int | None], TacticEntry] = {}
    for index, item in enumerate(entries):
        try:
            arch = item.get("arch")
            entry = TacticEntry(
                arch=None if arch == LEGACY_FALLBACK_ARCH else arch,
                mp_count=item.get("mp_count"),
                tactic=kind.from_values(item["values"]),
                note=item.get("note", ""),
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(
                f"{path}: tactic #{index} is malformed ({exc!r}); each entry needs "
                "an 'arch' (or null) and a 'values' mapping"
            ) from exc
        except ValueError as exc:
            raise ValueError(f"{path}: tactic #{index}: {exc}") from exc
        key = (entry.arch, entry.mp_count)
        if key in table:
            raise ValueError(
                f"{path}: duplicate tactic for arch={key[0]} " f"mp_count={key[1]}"
            )
        table[key] = entry
    if not table:
        raise ValueError(f"{path}: table defines no tactics")
    return table


def _lookup(
    kernel: str,
    *,
    arch_name: str,
    mp_count: int | None,
    directory: Path | None,
) -> tuple[TacticEntry | None, list[str]]:
    """Walk the ladder; return the hit (or ``None``) and the keys that were tried."""
    table = load(kernel, directory)
    tried: list[str] = []
    for key in _ladder(arch_name, mp_count):
        entry = table.get(key)
        if entry is not None:
            return entry, tried
        tried.append(_labelled(key))
    return None, tried


def _coerce(kernel: str, value: Any) -> Tactic:
    """Accept a registered tactic, or a mapping to build one from."""
    if isinstance(value, Tactic):
        return value
    return tactic_type(kernel).from_values(value)


def resolve_tactic(
    kernel: str,
    *,
    arch_name: str,
    mp_count: int | None = None,
    default: Tactic | Mapping[str, Any] | None = None,
    directory: Path | None = None,
) -> Tactic:
    """Return the typed tactic for this arch and bin, or raise.

    ``default`` is the caller stating that a generic tactic is acceptable; pass it
    only where that is true, because it is what silences the failure for an arch
    nobody has tuned.
    """
    entry, tried = _lookup(kernel, arch_name=arch_name, mp_count=mp_count, directory=directory)
    if entry is not None:
        return entry.tactic
    if default is not None:
        return _coerce(kernel, default)
    raise TacticNotFound(
        f"no tactic for kernel {kernel!r} on {arch_name} "
        f"(mp_count={mp_count}); tried {', '.join(tried)}. Add an entry to "
        f"{_table_path(kernel, directory)} with the measured numbers for this "
        "arch, or pass default= where a generic tactic is known to be fine."
    )


def resolve_tactic_for_device(
    kernel: str,
    *,
    default: Tactic | Mapping[str, Any] | None = None,
    directory: Path | None = None,
) -> Tactic:
    """``resolve_tactic`` for the device this process is running on.

    ``arch_name`` and ``mp_count`` come from the device, not from an environment
    variable or from ``torch_musa.get_arch_list()``, which reports the arch the
    wheel was built for.  The multiprocessor count is what tells two bins of one
    arch apart, which is why it is part of the key.
    """
    from vllm_musa.device_spec import current_device_spec

    spec = current_device_spec()
    return resolve_tactic(
        kernel,
        arch_name=spec.arch_name,
        mp_count=spec.mp_count,
        default=default,
        directory=directory,
    )


__all__ = [
    "LEGACY_FALLBACK_ARCH",
    "SCHEMA_VERSION",
    "TACTICS_DIR",
    "TACTIC_TYPES",
    "MoeTactic",
    "RmsnormTactic",
    "Tactic",
    "TacticEntry",
    "TacticNotFound",
    "load",
    "resolve_tactic",
    "resolve_tactic_for_device",
    "tactic_type",
]
