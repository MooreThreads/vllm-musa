#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a MATE benchmark script with MUSA-event timing instead of Kineto."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

# isort: off
import torchada  # noqa: F401  # patch before torch ecosystem imports
from mate.testing.utils import bench_gpu_time_with_musa_event
# isort: on


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--dry-runs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--l2-flush-mb", type=int, default=8000)
    return parser.parse_known_args()


def main() -> int:
    args, target_args = parse_args()
    target = args.target.resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    if min(args.dry_runs, args.repeats, args.l2_flush_mb) <= 0:
        raise ValueError("timing parameters must be positive")

    sys.path.insert(0, str(target.parent))
    module_name = f"musa_event_benchmark_{target.stem}"
    spec = importlib.util.spec_from_file_location(module_name, target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load benchmark: {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    def bench_with_events(fn, **_ignored) -> float:
        samples_ms = [
            float(value)
            for value in bench_gpu_time_with_musa_event(
                fn,
                dry_run_iters=args.dry_runs,
                repeat_iters=args.repeats,
                l2_flush=True,
                l2_flush_size_mb=args.l2_flush_mb,
            )
        ]
        print(
            "MUSA_EVENT_SAMPLES "
            + json.dumps(
                {
                    "samples_ms": samples_ms,
                    "median_ms": statistics.median(samples_ms),
                    "dry_runs": args.dry_runs,
                    "repeats": args.repeats,
                    "l2_flush_mb": args.l2_flush_mb,
                },
                sort_keys=True,
            )
        )
        return statistics.median(samples_ms) / 1000.0

    if not hasattr(module, "bench_kineto"):
        raise AttributeError(f"target has no bench_kineto binding: {target}")
    module.bench_kineto = bench_with_events
    sys.argv = [str(target), *target_args]
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
