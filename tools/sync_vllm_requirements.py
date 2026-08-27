#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synchronize the pinned upstream vLLM runtime requirements snapshot.

The outer vllm-musa project needs a checked-in snapshot because setuptools
reads project metadata before the custom build commands can prepare the
third-party checkout. This script keeps that snapshot reproducible when the
upstream pin changes; it never mutates the upstream checkout.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "third_party" / "vllm" / "requirements" / "common.txt"
DEFAULT_OUTPUT = ROOT / "requirements" / "vllm_common.txt"
GENERATED_HEADER = """# DO NOT EDIT: generated from the pinned upstream vLLM checkout.
# Run `python tools/sync_vllm_requirements.py` after changing third_party/PINS.
"""


def render_snapshot(source: Path) -> str:
    """Render the checked-in snapshot from an upstream requirements file."""
    if not source.is_file():
        raise FileNotFoundError(
            f"upstream requirements file is missing: {source}\n"
            "Run `make -f Makefile.sync checkout` first."
        )
    contents = source.read_text(encoding="utf-8")
    if not contents.endswith("\n"):
        contents += "\n"
    return GENERATED_HEADER + contents


def sync_snapshot(source: Path, output: Path, *, check: bool = False) -> bool:
    """Write or validate the snapshot; return whether it is up to date."""
    expected = render_snapshot(source)
    actual = output.read_text(encoding="utf-8") if output.is_file() else None
    if check:
        if actual == expected:
            return True
        print(f"generated requirements snapshot is stale: {output}", file=sys.stderr)
        if actual is not None:
            diff = difflib.unified_diff(
                actual.splitlines(keepends=True),
                expected.splitlines(keepends=True),
                fromfile=str(output),
                tofile=str(source),
            )
            sys.stderr.writelines(diff)
        return False

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(expected, encoding="utf-8")
    print(f"wrote generated requirements snapshot: {output}")
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the checked-in snapshot differs from the source",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        up_to_date = sync_snapshot(args.source, args.output, check=args.check)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0 if up_to_date else 1


if __name__ == "__main__":
    raise SystemExit(main())
