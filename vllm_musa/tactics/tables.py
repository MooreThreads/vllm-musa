# SPDX-License-Identifier: Apache-2.0
"""Loading, validating and looking up the tactic tables.

A table is ``tactics/<kernel>.json``: a list of entries, each an optional
*pattern* (``match``) over the things that can change the answer, plus the values
that pattern answers with.

The pattern exists because the dimensions that matter are not the same for every
kernel.  A MoE table needs the arch, the multiprocessor count and a band of M; the
split-K table needs the projection's ``(N, K)`` and whether the launch is inside a
graph capture, and does not care about the arch yet.  Rather than add a column and
a rung of the ladder for each new kernel, an entry lists only the dimensions it
really constrains and every other dimension is a wildcard.  The entry whose
pattern is the most specific about a given problem wins, which is what lets a
table say "this bin, for these M" and still keep one generic row underneath --
or, in the split-K case, no generic row at all, so an unseen shape fails closed.

Why a band and not a ``M -> config`` grid like vLLM and SGLang use: those two
compile with Triton on first call, so a grid of twenty M rows costs nothing until
one is used.  MUSA has no runtime compiler, so every distinct tactic is a variant
that has to be built ahead of time.  A band *is* one compiled variant, so the
table doubles as the list of variants to build; a per-M grid would silently imply
thousands.

Why the arch and the multiprocessor count are both available: S5000 ships 48, 52,
56 and 60 multiprocessor bins (MUSA-100006), and the tuned numbers differ between
them while ``arch_name`` stays ``mp_31``.  vLLM's config filename carries only a
device *name*, which is why it has to rewrite H200 family names by hand
(``fused_moe.py:1003-1005``); a key that omits a dimension has to be patched
later.

Validation happens at load time, not at launch: an ambiguous pair (two patterns
that can answer the same problem with the same specificity), an unknown match
key, a malformed band, an unknown value key, a schema that does not match -- all
are errors here, where the file and the row index can be named.
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
    "MATCH_KEYS",
    "SCHEMA_VERSION",
    "TacticEntry",
    "TacticMatch",
    "TacticNotFound",
    "load",
    "lookup",
    "lookup_entry",
    "table_path",
]

DEFAULT_DIR = Path(__file__).resolve().parent

#: Bumped when an entry's shape changes.  A table from another schema is refused
#: rather than half-read: silently ignoring an unknown key is how a "tuned" table
#: becomes a generic one without anyone noticing.
SCHEMA_VERSION = 3

#: The match keys a table may use.  Each one is a question the resolver can ask
#: of a :class:`~.problem.TuningProblem`; a kernel that needs a new question adds
#: it here and in :meth:`TacticMatch.matches`.
MATCH_KEYS = ("arch", "mp_count", "graph_capture", "shape", "m_range")


class TacticNotFound(LookupError):
    """No tactic matches the problem and the caller offered no default."""


@dataclass(frozen=True)
class TacticMatch:
    """The pattern an entry answers for.  ``None`` on a field means "any"."""

    arch: str | None = None
    mp_count: int | None = None
    graph_capture: bool | None = None
    #: ``(N, K)`` of the projection, for kernels whose numbers depend on it.
    shape: tuple[int, int] | None = None
    #: Inclusive band of per-expert rows (``TuningProblem.tuning_m``).
    m_range: tuple[int, int] | None = None

    @property
    def specificity(self) -> int:
        """How many dimensions this pattern pins down."""
        return sum(
            value is not None
            for value in (
                self.arch,
                self.mp_count,
                self.graph_capture,
                self.shape,
                self.m_range,
            )
        )

    def matches(self, problem: TuningProblem) -> bool:
        if self.arch is not None and self.arch != problem.arch_name:
            return False
        if self.mp_count is not None and self.mp_count != problem.mp_count:
            return False
        if self.graph_capture is not None:
            if self.graph_capture != problem.graph_capture:
                return False
        if self.shape is not None:
            if self.shape != (problem.layer.shape_n, problem.layer.shape_k):
                return False
        if self.m_range is not None:
            low, high = self.m_range
            if not low <= problem.tuning_m <= high:
                return False
        return True

    def intersects(self, other: TacticMatch) -> bool:
        """Whether some problem could match both patterns."""
        for left, right in (
            (self.arch, other.arch),
            (self.mp_count, other.mp_count),
            (self.graph_capture, other.graph_capture),
            (self.shape, other.shape),
        ):
            if left is not None and right is not None and left != right:
                return False
        if self.m_range is not None and other.m_range is not None:
            return (
                self.m_range[0] <= other.m_range[1]
                and other.m_range[0] <= self.m_range[1]
            )
        return True

    def describe(self) -> str:
        parts = []
        if self.arch is not None:
            parts.append(self.arch)
        if self.mp_count is not None:
            parts.append(f"mp{self.mp_count}")
        if self.graph_capture is not None:
            parts.append("capture" if self.graph_capture else "eager")
        if self.shape is not None:
            parts.append(f"{self.shape[0]}x{self.shape[1]}")
        if self.m_range is not None:
            parts.append(f"m{self.m_range[0]}..{self.m_range[1]}")
        return "/".join(parts) if parts else "any"


@dataclass(frozen=True)
class TacticEntry:
    """One row of a table: the pattern it answers, and the values it holds."""

    match: TacticMatch
    tactic: Tactic
    note: str = ""


def table_path(kernel: str, directory: Path | None = None) -> Path:
    return (DEFAULT_DIR if directory is None else Path(directory)) / f"{kernel}.json"


def _parse_band(raw: Any, where: str) -> tuple[int, int] | None:
    if raw is None:
        return None
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != 2
        or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in raw
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


def _parse_shape(raw: Any, where: str) -> tuple[int, int] | None:
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
            f"{where}: 'shape' must be null or a [N, K] pair of integers, got {raw!r}"
        )
    shape = (int(raw[0]), int(raw[1]))
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"{where}: 'shape' must be positive, got {shape}")
    return shape


def _parse_match(raw: Any, where: str) -> TacticMatch:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{where}: 'match' must be an object, got {type(raw).__name__}"
        )
    unknown = sorted(set(raw) - set(MATCH_KEYS))
    if unknown:
        raise ValueError(
            f"{where}: unknown match key(s) {unknown}; known keys are "
            f"{list(MATCH_KEYS)}"
        )
    arch = raw.get("arch")
    if arch is not None:
        if not isinstance(arch, str):
            raise ValueError(f"{where}: 'arch' must be a string or null, got {arch!r}")
        if arch == "default":
            raise ValueError(
                f"{where}: 'arch' is null for the table-wide fallback; the string "
                "'default' was the schema-1 spelling and is not an arch name"
            )
    mp_count = raw.get("mp_count")
    if mp_count is not None and (
        not isinstance(mp_count, int) or isinstance(mp_count, bool)
    ):
        raise ValueError(f"{where}: 'mp_count' must be an integer or null")
    graph_capture = raw.get("graph_capture")
    if graph_capture is not None and not isinstance(graph_capture, bool):
        raise ValueError(f"{where}: 'graph_capture' must be a boolean or null")
    return TacticMatch(
        arch=arch,
        mp_count=mp_count,
        graph_capture=graph_capture,
        shape=_parse_shape(raw.get("shape"), where),
        m_range=_parse_band(raw.get("m_range"), where),
    )


def load(kernel: str, directory: Path | None = None) -> list[TacticEntry]:
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
        match = _parse_match(item.get("match"), at)
        tactic = cls.from_values(item.get("values"), where=at)
        note = item.get("note", "")
        if not isinstance(note, str):
            raise ValueError(f"{at}: 'note' must be a string")
        if match.specificity and not note.strip():
            raise ValueError(
                f"{at}: a row that constrains {match.describe()} must carry a "
                "non-empty 'note' saying how it was measured. Only an all-wildcard "
                "row may be generic: a measured-looking row without provenance is "
                "how a table starts lying."
            )
        entry = TacticEntry(match=match, tactic=tactic, note=note)
        for earlier in parsed:
            if earlier.match.intersects(entry.match):
                if earlier.match.specificity == entry.match.specificity:
                    raise ValueError(
                        f"{where}: {entry.match.describe()} and "
                        f"{earlier.match.describe()} can both answer the same "
                        "problem with the same specificity; one of them must "
                        "constrain another dimension"
                    )
        parsed.append(entry)
    return parsed


def lookup_entry(
    kernel: str,
    problem: TuningProblem,
    directory: Path | None = None,
) -> TacticEntry:
    """The most specific entry matching ``problem``, or raise :class:`TacticNotFound`.

    Callers that need the pattern (to read the M band it was measured for) use
    this; callers that only need the numbers use :func:`lookup`.
    """
    entries = load(kernel, directory)
    matching = [entry for entry in entries if entry.match.matches(problem)]
    if not matching:
        known = ", ".join(entry.match.describe() for entry in entries)
        raise TacticNotFound(
            f"no tactic for kernel {kernel!r} on arch={problem.arch_name!r} "
            f"mp_count={problem.mp_count} shape="
            f"{problem.layer.shape_n}x{problem.layer.shape_k} "
            f"capture={problem.graph_capture} m={problem.tuning_m}; the table at "
            f"{table_path(kernel, directory)} covers {known}. Add a measured entry "
            "for this key, or pass default= where a generic tactic is known to be "
            "fine."
        )
    # Validation forbids two matches of equal specificity, so the maximum is
    # unique; picking it is the whole precedence rule.
    return max(matching, key=lambda entry: entry.match.specificity)


def lookup(
    kernel: str,
    problem: TuningProblem,
    directory: Path | None = None,
) -> Tactic:
    """The most specific tactic matching ``problem``, or raise ``TacticNotFound``."""
    return lookup_entry(kernel, problem, directory).tactic
