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
3. ``("default", None)``      -- the table's own fallback
4. the caller's ``default=``  -- explicit opt-in
5. :class:`TacticNotFound`

Anything past 4 is the caller saying "a generic tactic is fine here"; a caller
that does not say so gets an error rather than an unvalidated guess.  This is
deliberate: on S5000 a wrong tiling is a 2x regression that looks like a
hardware limit, and nobody re-reads a tuning table for an arch they just
enabled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

TACTICS_DIR = Path(__file__).resolve().parent
SCHEMA_VERSION = 1

#: Arch whose numbers are the fallback when a table has no ``default`` entry.
DEFAULT_ARCH = "default"


class TacticNotFound(LookupError):
    """No tactic for the requested device and the caller offered no default."""


@dataclass(frozen=True)
class TacticEntry:
    arch: str
    mp_count: Optional[int]
    values: dict[str, Any]
    note: str = ""


def _ladder(arch_name: str, mp_count: Optional[int]) -> list[tuple[str, Optional[int]]]:
    keys: list[tuple[str, Optional[int]]] = [(arch_name, mp_count)]
    if mp_count is not None:
        keys.append((arch_name, None))
    keys.append((DEFAULT_ARCH, None))
    return keys


def _table_path(kernel: str, directory: Path | None) -> Path:
    return (TACTICS_DIR if directory is None else directory) / f"{kernel}.json"


def load(
    kernel: str, directory: Path | None = None
) -> dict[tuple[str, Optional[int]], TacticEntry]:
    """Load ``<kernel>.json`` into a ``(arch, mp_count) -> entry`` map."""
    directory = TACTICS_DIR if directory is None else directory
    path = directory / f"{kernel}.json"
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
            f"{path}: table says kernel {raw.get('kernel')!r}, "
            f"expected {kernel!r}"
        )

    table: dict[tuple[str, Optional[int]], TacticEntry] = {}
    entries = raw.get("tactics", [])
    if not isinstance(entries, list):
        raise ValueError(f"{path}: 'tactics' must be a list, got {type(entries).__name__}")
    for index, item in enumerate(entries):
        try:
            key = (item["arch"], item.get("mp_count"))
            entry = TacticEntry(
                arch=item["arch"],
                mp_count=item.get("mp_count"),
                values=dict(item["values"]),
                note=item.get("note", ""),
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(
                f"{path}: tactic #{index} is malformed ({exc!r}); each entry needs "
                "'arch' and a 'values' mapping"
            ) from exc
        if key in table:
            raise ValueError(
                f"{path}: duplicate tactic for arch={key[0]} "
                f"mp_count={key[1]}"
            )
        table[key] = entry
    if not table:
        raise ValueError(f"{path}: table defines no tactics")
    return table


def select(
    kernel: str,
    *,
    arch_name: str,
    mp_count: Optional[int] = None,
    default: dict[str, Any] | None = None,
    directory: Path | None = None,
) -> dict[str, Any]:
    """Return the tactic values for this device, or raise.

    ``default`` is the caller stating that a generic tactic is acceptable; pass
    it only where that is true, because it is what silences the failure for an
    arch nobody has tuned.
    """
    table = load(kernel, directory)
    tried: list[str] = []
    for arch, mp in _ladder(arch_name, mp_count):
        entry = table.get((arch, mp))
        if entry is not None:
            return entry.values
        tried.append(f"{arch}/mp{mp}" if mp is not None else f"{arch}")

    if default is not None:
        return dict(default)

    raise TacticNotFound(
        f"no tactic for kernel {kernel!r} on {arch_name} "
        f"(mp_count={mp_count}); tried {', '.join(tried)}. Add an entry to "
        f"{_table_path(kernel, directory)} with the measured numbers for this "
        "arch, or pass default= where a generic tactic is known to be fine."
    )


def describe(kernel: str, directory: Path | None = None) -> str:
    """Human-readable table, for logs and for ``python -m`` troubleshooting."""
    table = load(kernel, directory)
    lines = [f"{kernel} ({len(table)} tactics)"]
    for (arch, mp), entry in sorted(
        table.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)
    ):
        bin_name = f"{arch}/mp{mp}" if mp is not None else arch
        note = f"  # {entry.note}" if entry.note else ""
        lines.append(f"  {bin_name:<16} {entry.values}{note}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    """``python -m vllm_musa.tactics <kernel>...``: dump the tables.

    Kept as a function so ``__main__.py`` can call it; a bare
    ``if __name__ == "__main__"`` guard here would never fire, because ``-m``
    dispatches to the package's ``__main__`` module.
    """
    import sys

    names = list(sys.argv[1:] if argv is None else argv)
    if not names:
        available = sorted(p.stem for p in TACTICS_DIR.glob("*.json"))
        print(f"usage: python -m vllm_musa.tactics <kernel> [{', '.join(available)}]")
        return 2
    for name in names:
        print(describe(name))
    return 0


if __name__ == "__main__":  # pragma: no cover - direct script execution
    raise SystemExit(main())
