# SPDX-License-Identifier: Apache-2.0
"""Loading, validating and looking up the tactic tables.

A table is ``tactics/<kernel>.json``: a list of entries, each keyed by the three
things that can change the answer -- the arch, the multiprocessor count, and the
band of M it was measured for -- and holding the tactic's values.

Why a band and not a ``M -> config`` grid like vLLM and SGLang use: those two
compile with Triton on first call, so a grid of twenty M rows costs nothing until
one is used.  MUSA has no runtime compiler, so every distinct tactic is a variant
that has to be built ahead of time.  A band *is* one compiled variant, so the
table doubles as the list of variants to build; a per-M grid would silently imply
thousands.

Why the arch and the multiprocessor count are both in the key: S5000 ships 48,
52, 56 and 60 multiprocessor bins (MUSA-100006), and the tuned numbers differ
between them while ``arch_name`` stays ``mp_31``.  vLLM's config filename carries
only a device *name*, which is why it has to rewrite H200 family names by hand
(``fused_moe.py:1003-1005``); a key that omits a dimension has to be patched
later.

Validation happens at load time, not at launch: a duplicate key, an overlapping
band, an unknown value key or a schema that does not match is an error here,
where the file and the index can be named.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .problem import TuningProblem
from .values import Tactic, tactic_class

__all__ = [
    "DEFAULT_DIR",
    "SCHEMA_VERSION",
    "TacticEntry",
    "TacticNotFound",
    "load",
    "lookup",
    "table_path",
]

DEFAULT_DIR = Path(__file__).resolve().parent

#: Bumped when an entry's shape changes.  A table from another schema is refused
#: rather than half-read: silently ignoring an unknown key is how a "tuned" table
#: becomes a generic one without anyone noticing.
SCHEMA_VERSION = 2


class TacticNotFound(LookupError):
    """No tactic for the requested problem and the caller offered no default."""


@dataclass(frozen=True)
class TacticEntry:
    """One row of a table: the key it answers for, and the values it holds."""

    arch: str | None
    mp_count: int | None
    m_range: tuple[int, int] | None
    tactic: Tactic
    note: str = ""

    @property
    def key(self) -> tuple[str | None, int | None, tuple[int, int] | None]:
        return (self.arch, self.mp_count, self.m_range)

    def covers(self, m: int) -> bool:
        if self.m_range is None:
            return True
        low, high = self.m_range
        return low <= m <= high

    def describe(self) -> str:
        arch = self.arch if self.arch is not None else "*"
        mp = f"mp{self.mp_count}" if self.mp_count is not None else "any-mp"
        band = f"m{self.m_range[0]}..{self.m_range[1]}" if self.m_range else "any-m"
        return f"{arch}/{mp}/{band}"


def table_path(kernel: str, directory: Path | None = None) -> Path:
    return (DEFAULT_DIR if directory is None else Path(directory)) / f"{kernel}.json"


def _parse_band(raw: Any, where: str) -> tuple[int, int] | None:
    if raw is None:
        return None
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != 2
        or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in raw
        )
    ):
        raise ValueError(
            f"{where}: 'm_range' must be null or a [low, high] pair of integers, "
            f"got {raw!r}"
        )
    low, high = int(raw[0]), int(raw[1])
    if low < 1 or high < low:
        raise ValueError(
            f"{where}: 'm_range' must satisfy 1 <= low <= high, got [{low}, {high}]"
        )
    return (low, high)


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def load(
    kernel: str, directory: Path | None = None
) -> list[TacticEntry]:
    """Read and validate ``<kernel>.json``, returning its entries in file order."""
    path = table_path(kernel, directory)
    if not path.is_file():
        raise TacticNotFound(
            f"no tactic table for kernel {kernel!r}: {path} is missing"
        )
    with path.open() as handle:
        raw = json.load(handle)

    where = str(path)
    if raw.get("_schema") != SCHEMA_VERSION:
        raise ValueError(
            f"{where}: unsupported tactic schema {raw.get('_schema')!r}, expected "
            f"{SCHEMA_VERSION}. A table written for another schema is rewritten, "
            "not read: an unknown key would otherwise be ignored silently."
        )
    if raw.get("kernel") != kernel:
        raise ValueError(
            f"{where}: table says kernel {raw.get('kernel')!r}, expected {kernel!r}"
        )

    entries = raw.get("tactics", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{where}: 'tactics' must be a non-empty list")

    cls = tactic_class(kernel)
    parsed: list[TacticEntry] = []
    for index, item in enumerate(entries):
        at = f"{where}: tactic #{index}"
        if not isinstance(item, dict):
            raise ValueError(f"{at} must be an object, got {type(item).__name__}")
        arch = item.get("arch")
        if arch is not None and not isinstance(arch, str):
            raise ValueError(f"{at}: 'arch' must be a string or null, got {arch!r}")
        if "default" == arch:
            raise ValueError(
                f"{at}: 'arch' is null for the table-wide fallback; the string "
                "'default' was the schema-1 spelling and is not an arch name"
            )
        mp_count = item.get("mp_count")
        if mp_count is not None and (
            not isinstance(mp_count, int) or isinstance(mp_count, bool)
        ):
            raise ValueError(
                f"{at}: 'mp_count' must be an integer or null, got {mp_count!r}"
            )
        band = _parse_band(item.get("m_range"), at)
        tactic = cls.from_values(item.get("values"), where=at)
        note = item.get("note", "")
        if not isinstance(note, str):
            raise ValueError(f"{at}: 'note' must be a string")
        if arch is not None and not note.strip():
            raise ValueError(
                f"{at}: a row for arch {arch!r} must carry a non-empty 'note' saying "
                "how it was measured. Only the table-wide (null, null) row may be "
                "generic: a measured-looking row without provenance is how a table "
                "starts lying."
            )
        entry = TacticEntry(
            arch=arch, mp_count=mp_count, m_range=band, tactic=tactic, note=note
        )
        for earlier in parsed:
            if earlier.key == entry.key:
                raise ValueError(
                    f"{where}: duplicate tactic for {entry.describe()}"
                )
            if (
                earlier.arch == entry.arch
                and earlier.mp_count == entry.mp_count
                and earlier.m_range is not None
                and entry.m_range is not None
                and _overlaps(earlier.m_range, entry.m_range)
            ):
                raise ValueError(
                    f"{where}: {earlier.describe()} and {entry.describe()} overlap; "
                    "two rows cannot answer the same M"
                )
            if (earlier.arch, earlier.mp_count) == (entry.arch, entry.mp_count) and (
                earlier.m_range is None or entry.m_range is None
            ):
                raise ValueError(
                    f"{where}: {entry.describe()} is redundant with "
                    f"{earlier.describe()}; an entry without a band already covers "
                    "every M at that specificity"
                )
        parsed.append(entry)

    return parsed


def _keys_for(problem: TuningProblem) -> list[tuple[str | None, int | None]]:
    """The ``(arch, mp_count)`` rungs to try, most specific first."""
    arch, mp = problem.arch_name or None, problem.mp_count
    rungs: list[tuple[str | None, int | None]] = []
    if arch is not None:
        if mp is not None:
            rungs.append((arch, mp))
        rungs.append((arch, None))
    rungs.append((None, None))
    return rungs


def lookup(
    kernel: str,
    problem: TuningProblem,
    directory: Path | None = None,
) -> Tactic:
    """The table's answer for ``problem``, or raise :class:`TacticNotFound`."""
    entries = load(kernel, directory)
    m = problem.tuning_m
    tried: list[str] = []
    for arch, mp in _keys_for(problem):
        candidates = [
            entry for entry in entries if (entry.arch, entry.mp_count) == (arch, mp)
        ]
        covering = [entry for entry in candidates if entry.covers(m)]
        if covering:
            # The band is part of the key, so at most one entry can cover a given
            # M after validation; sort by specificity anyway so the error is not
            # the only thing that is deterministic.
            covering.sort(
                key=lambda entry: (entry.m_range is None, entry.m_range or (0, 0))
            )
            return covering[0].tactic
        tried.extend(entry.describe() for entry in candidates)
    tried.append(f"table-wide fallback for m={m}")
    raise TacticNotFound(
        f"no tactic for kernel {kernel!r} on arch={problem.arch_name!r} "
        f"mp_count={problem.mp_count} m={m} (tuning_m={m}); the table at "
        f"{table_path(kernel, directory)} has {', '.join(tried) or 'no entries'}. "
        "Add a measured entry for this bin, or pass default= where a generic "
        "tactic is known to be fine."
    )
