#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""thin alias for ``musa_sync verify`` — the validator entry point the
strategy doc promised. Runs the offline pre-bump gate (no MUSA hardware).

    python tools/patch_validate.py [--target TAG] [--repo PATH]
    python tools/patch_validate.py <subcommand> [args…]

A known subcommand passes through unchanged (``… patch_validate.py check-series``
is not ``verify check-series``); bare flags default to ``verify``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from musa_sync import SUBCOMMANDS, main  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv or argv[0] not in SUBCOMMANDS:
        argv = ["verify", *argv]
    sys.exit(main(argv))
