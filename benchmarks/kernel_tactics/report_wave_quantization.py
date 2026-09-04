#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report MP-count wave discontinuities before running silicon benchmarks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from vllm_musa.tuning import estimate_wave_quantization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--multiprocessor-counts",
        type=int,
        nargs="+",
        default=(48, 52, 56, 60, 64),
        help="Exact runtime multi_processor_count values to compare.",
    )
    parser.add_argument(
        "--tiles",
        type=int,
        nargs="+",
        required=True,
        help="Total logical tile counts emitted by candidate launch shapes.",
    )
    parser.add_argument("--blocks-per-multiprocessor", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for total_tiles in args.tiles:
        for multiprocessor_count in args.multiprocessor_counts:
            estimate = estimate_wave_quantization(
                total_tiles,
                multiprocessor_count,
                blocks_per_multiprocessor=args.blocks_per_multiprocessor,
            )
            rows.append(
                {
                    "multiprocessor_count": multiprocessor_count,
                    "blocks_per_multiprocessor": args.blocks_per_multiprocessor,
                    **asdict(estimate),
                }
            )
    print(json.dumps({"schema": "musa-wave-quantization.v1", "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
