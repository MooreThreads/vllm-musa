#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired cold-cache benchmark for the production DSV4 MHC-post JIT kernel."""

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

PRODUCTION_CONFIG = (256, 256)


def parse_config(value: str) -> tuple[int, int]:
    """Parse ``hidden_block x threads`` and reject partial-write configs."""
    try:
        hidden_block, threads = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"config must be HIDDEN_BLOCKxTHREADS, got {value!r}"
        ) from exc
    if hidden_block <= 0 or threads <= 0 or threads < hidden_block:
        raise argparse.ArgumentTypeError(
            "config requires positive values and threads >= hidden_block"
        )
    return hidden_block, threads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[20, 80, 320],
        help="production standalone post buckets after the M<=16 fused path",
    )
    parser.add_argument(
        "--configs",
        type=parse_config,
        nargs="+",
        default=[(64, 64), (128, 128), PRODUCTION_CONFIG, (512, 512)],
        metavar="BLOCKxTHREADS",
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
    from vllm_musa.deepseek_v4_jit.tilelang_kernels import mhc_post_kernel
    # isort: on

    return torch, mhc_post_kernel


@dataclass
class _Run:
    config: tuple[int, int]
    kernel: Any
    output: Any
    compile_seconds: float
    candidate_samples: list[float] = field(default_factory=list)
    baseline_samples: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class _Inputs:
    x: Any
    residual: Any
    post_mix: Any
    comb_mix: Any


def _invoke(run: _Run, inputs: _Inputs) -> None:
    run.kernel(
        inputs.x,
        inputs.residual,
        inputs.post_mix,
        inputs.comb_mix,
        run.output,
    )


def _timed_invoke(torch: Any, flush: Any, run: _Run, inputs: _Inputs) -> float:
    flush.zero_()
    torch.musa.synchronize()
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    _invoke(run, inputs)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def main() -> int:
    args = parse_args()
    if min(args.tokens) <= 0 or args.hidden_size <= 0:
        raise ValueError("tokens and hidden size must be positive")
    if min(args.warmup, args.repeats, args.l2_flush_mb) <= 0:
        raise ValueError("timing arguments must be positive")

    lease_device_fence = verify_lease_device_fence(
        args.expected_physical_device,
        args.expected_device_uuid,
    )
    torch, kernel_factory = _load_runtime()
    if not torch.musa.is_available() or torch.musa.device_count() != 1:
        raise RuntimeError("MHC-post benchmark requires one visible MUSA device")
    device = torch.device("musa")
    properties = torch.musa.get_device_properties(0)
    hidden_size = int(args.hidden_size)
    configs = list(dict.fromkeys([PRODUCTION_CONFIG, *args.configs]))
    flush = torch.empty(
        args.l2_flush_mb * 1024 * 1024 // 4,
        dtype=torch.int32,
        device=device,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for num_tokens in args.tokens:
        inputs = _Inputs(
            x=torch.randn(
                (num_tokens, hidden_size),
                generator=generator,
                dtype=torch.bfloat16,
            ).to(device),
            residual=torch.randn(
                (num_tokens, 4, hidden_size),
                generator=generator,
                dtype=torch.bfloat16,
            ).to(device),
            post_mix=torch.randn(
                (num_tokens, 4), generator=generator, dtype=torch.float32
            ).to(device),
            comb_mix=torch.randn(
                (num_tokens, 4, 4), generator=generator, dtype=torch.float32
            ).to(device),
        )
        compiled: list[_Run] = []
        for hidden_block, threads in configs:
            config = (hidden_block, threads)
            if hidden_size % hidden_block:
                skipped.append(
                    {
                        "tokens": num_tokens,
                        "config": list(config),
                        "reason": "hidden size is not divisible by hidden block",
                    }
                )
                continue
            started = time.perf_counter()
            try:
                kernel = kernel_factory(
                    hidden_size,
                    hidden_block=hidden_block,
                    threads=threads,
                )
                run = _Run(
                    config=config,
                    kernel=kernel,
                    output=torch.empty(
                        (num_tokens, 4, hidden_size),
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    compile_seconds=time.perf_counter() - started,
                )
                _invoke(run, inputs)
                torch.musa.synchronize()
            except Exception as exc:  # pragma: no cover - device-side contract
                skipped.append(
                    {
                        "tokens": num_tokens,
                        "config": list(config),
                        "reason": str(exc),
                    }
                )
                continue
            compiled.append(run)

        production = next(
            (run for run in compiled if run.config == PRODUCTION_CONFIG), None
        )
        if production is None:
            raise RuntimeError("production MHC-post config failed to compile")
        for run in compiled:
            for _ in range(args.warmup):
                _invoke(run, inputs)
        torch.musa.synchronize()

        for repeat in range(args.repeats):
            for candidate in (run for run in compiled if run is not production):
                pair = (
                    (production, candidate)
                    if repeat % 2 == 0
                    else (candidate, production)
                )
                first_ms = _timed_invoke(torch, flush, pair[0], inputs)
                second_ms = _timed_invoke(torch, flush, pair[1], inputs)
                if repeat % 2 == 0:
                    candidate.baseline_samples.append(first_ms)
                    candidate.candidate_samples.append(second_ms)
                else:
                    candidate.candidate_samples.append(first_ms)
                    candidate.baseline_samples.append(second_ms)

        _invoke(production, inputs)
        torch.musa.synchronize()
        reference = production.output.detach().cpu().float()
        for run in compiled:
            if run is production:
                run.baseline_samples = [
                    _timed_invoke(torch, flush, run, inputs)
                    for _ in range(args.repeats)
                ]
                run.candidate_samples = list(run.baseline_samples)
                ratios = [1.0] * args.repeats
                max_abs_diff = 0.0
            else:
                _invoke(run, inputs)
                torch.musa.synchronize()
                max_abs_diff = float(
                    (run.output.detach().cpu().float() - reference).abs().max().item()
                )
                ratios = [
                    candidate / baseline
                    for candidate, baseline in zip(
                        run.candidate_samples, run.baseline_samples
                    )
                ]
            output_absmax = float(run.output.float().abs().max().item())
            median_ratio = statistics.median(ratios)
            rows.append(
                {
                    "tokens": num_tokens,
                    "hidden_block": run.config[0],
                    "threads": run.config[1],
                    "production_config": list(PRODUCTION_CONFIG),
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
            "schema": "musa-dsv4-mhc-post-jit-paired-ab.v1",
            "device_name": torch.musa.get_device_name(0),
            "device_capability": [int(properties.major), int(properties.minor)],
            "multiprocessor_count": int(properties.multi_processor_count),
            "lease_device_fence": lease_device_fence,
            "hidden_size": hidden_size,
            "benchmark": {
                "warmup": args.warmup,
                "repeats": args.repeats,
                "l2_flush_mb": args.l2_flush_mb,
                "flush_before_every_timed_launch": True,
                "paired_alternating_order": True,
                "production_config": list(PRODUCTION_CONFIG),
            },
            "kernel_source": (
                "vllm_musa/deepseek_v4_jit/tilelang_kernels.py::mhc_post_kernel"
            ),
            "provenance": provenance(Path(__file__)),
            "rows": rows,
            "skipped": skipped,
        },
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
