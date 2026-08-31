#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired cold-cache DSV4 weighted-RMSNorm JIT thread sweep."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from _benchmark_utils import (
    emit_payload,
    percentile,
    provenance,
    verify_lease_device_fence,
)

PRODUCTION_THREADS = 128
VALID_THREADS = (64, 128, 256)


def parse_threads(value: str) -> int:
    try:
        threads = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"threads must be an integer: {value!r}"
        ) from exc
    if threads not in VALID_THREADS:
        raise argparse.ArgumentTypeError(f"threads must be one of {VALID_THREADS}")
    return threads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[20, 80, 320])
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument(
        "--threads",
        type=parse_threads,
        nargs="+",
        default=list(VALID_THREADS),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--l2-flush-mb", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--expected-physical-device", type=int, required=True)
    parser.add_argument("--expected-device-uuid", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_runtime() -> tuple[Any, Any]:
    # Import order is part of the MUSA compatibility contract.
    # isort: off
    import torchada  # noqa: F401
    import torch
    from vllm_musa.deepseek_v4_jit.tilelang_kernels import (
        mhc_weighted_rmsnorm_mudnn_like_kernel,
    )
    # isort: on

    return torch, mhc_weighted_rmsnorm_mudnn_like_kernel


@dataclass
class _Run:
    threads: int
    kernel: Any
    output: Any
    compile_seconds: float
    candidate_samples: list[float] = field(default_factory=list)
    baseline_samples: list[float] = field(default_factory=list)


def _invoke(run: _Run, x: Any, weight: Any) -> None:
    run.kernel(x, weight, run.output, 1e-6)


def _timed_invoke(
    torch: Any,
    flush: Any,
    run: _Run,
    x: Any,
    weight: Any,
) -> float:
    run.output.fill_(float("nan"))
    flush.zero_()
    torch.musa.synchronize()
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    _invoke(run, x, weight)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def main() -> int:
    args = parse_args()
    if min(args.tokens) <= 0 or args.hidden_size != 4096:
        raise ValueError("production DSV4 sweep requires positive M and H=4096")
    if min(args.warmup, args.repeats, args.l2_flush_mb) <= 0:
        raise ValueError("timing arguments must be positive")

    lease_device_fence = verify_lease_device_fence(
        args.expected_physical_device,
        args.expected_device_uuid,
    )
    torch, kernel_factory = _load_runtime()
    if not torch.musa.is_available() or torch.musa.device_count() != 1:
        raise RuntimeError(
            "weighted-RMSNorm benchmark requires one visible MUSA device"
        )
    device = torch.device("musa")
    properties = torch.musa.get_device_properties(0)
    flush = torch.empty(
        args.l2_flush_mb * 1024 * 1024 // 4,
        dtype=torch.int32,
        device=device,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    configs = list(dict.fromkeys([PRODUCTION_THREADS, *args.threads]))
    rows: list[dict[str, Any]] = []

    for num_tokens in args.tokens:
        x = torch.randn(
            (num_tokens, args.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        ).to(device)
        weight = torch.randn(
            (args.hidden_size,), generator=generator, dtype=torch.bfloat16
        ).to(device)
        runs: list[_Run] = []
        for threads in configs:
            started = time.perf_counter()
            kernel = kernel_factory(args.hidden_size, threads=threads)
            runs.append(
                _Run(
                    threads=threads,
                    kernel=kernel,
                    output=torch.empty_like(x),
                    compile_seconds=time.perf_counter() - started,
                )
            )
        production = next(run for run in runs if run.threads == PRODUCTION_THREADS)
        for run in runs:
            for _ in range(args.warmup):
                _invoke(run, x, weight)
        torch.musa.synchronize()

        for repeat in range(args.repeats):
            for candidate in (run for run in runs if run is not production):
                pair = (
                    (production, candidate)
                    if repeat % 2 == 0
                    else (candidate, production)
                )
                first_ms = _timed_invoke(torch, flush, pair[0], x, weight)
                second_ms = _timed_invoke(torch, flush, pair[1], x, weight)
                if repeat % 2 == 0:
                    candidate.baseline_samples.append(first_ms)
                    candidate.candidate_samples.append(second_ms)
                else:
                    candidate.candidate_samples.append(first_ms)
                    candidate.baseline_samples.append(second_ms)

        _invoke(production, x, weight)
        torch.musa.synchronize()
        reference = production.output.detach().cpu().float()
        for run in runs:
            if run is production:
                run.baseline_samples = [
                    _timed_invoke(torch, flush, run, x, weight)
                    for _ in range(args.repeats)
                ]
                run.candidate_samples = list(run.baseline_samples)
                ratios = [1.0] * args.repeats
            else:
                ratios = [
                    candidate / baseline
                    for candidate, baseline in zip(
                        run.candidate_samples, run.baseline_samples
                    )
                ]
            run.output.fill_(float("nan"))
            _invoke(run, x, weight)
            torch.musa.synchronize()
            max_abs_diff = float(
                (run.output.detach().cpu().float() - reference).abs().max().item()
            )
            output_absmax = float(run.output.float().abs().max().item())
            median_ratio = statistics.median(ratios)
            rows.append(
                {
                    "tokens": num_tokens,
                    "hidden_size": args.hidden_size,
                    "dtype": "torch.bfloat16",
                    "threads": run.threads,
                    "production_threads": PRODUCTION_THREADS,
                    "is_production_config": run is production,
                    "median_ms": float(statistics.median(run.candidate_samples)),
                    "baseline_median_ms": float(
                        statistics.median(run.baseline_samples)
                    ),
                    "median_ratio": median_ratio,
                    "ratio_p95": percentile(ratios, 0.95),
                    "ratio_p99": percentile(ratios, 0.99),
                    "speedup_pct": (1.0 / median_ratio - 1.0) * 100.0,
                    "candidate_samples_ms": list(run.candidate_samples),
                    "baseline_samples_ms": list(run.baseline_samples),
                    "compile_seconds": run.compile_seconds,
                    "output_absmax": output_absmax,
                    "max_abs_diff_vs_production": max_abs_diff,
                    "correctness_pass": bool(
                        torch.isfinite(run.output).all().item()
                        and output_absmax > 1e-12
                        and max_abs_diff <= 5e-2
                    ),
                    "cache_policy": "cold-l2-per-sample",
                }
            )

    emit_payload(
        {
            "schema": "musa-dsv4-weighted-rmsnorm-jit-paired-ab.v1",
            "device_name": torch.musa.get_device_name(0),
            "device_capability": [int(properties.major), int(properties.minor)],
            "multiprocessor_count": int(properties.multi_processor_count),
            "lease_device_fence": lease_device_fence,
            "benchmark": {
                "warmup": args.warmup,
                "repeats": args.repeats,
                "l2_flush_mb": args.l2_flush_mb,
                "flush_before_every_timed_launch": True,
                "paired_alternating_order": True,
                "inner_iters": 1,
                "production_threads": PRODUCTION_THREADS,
            },
            "kernel_source": (
                "vllm_musa/deepseek_v4_jit/tilelang_kernels.py::"
                "mhc_weighted_rmsnorm_mudnn_like_kernel"
            ),
            "provenance": provenance(Path(__file__)),
            "rows": rows,
            "skipped": [],
        },
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
