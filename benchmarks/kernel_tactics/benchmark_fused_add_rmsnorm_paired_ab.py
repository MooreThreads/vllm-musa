#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired same-card A/B timing for existing fused-add-RMSNorm AOT blocks."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

# isort: off
import torchada  # noqa: F401  # patch before torch ecosystem imports
import torch
# isort: on

from _benchmark_utils import (
    emit_payload,
    percentile,
    provenance,
    verify_lease_device_fence,
)

from vllm_musa import _custom_ops as musa_ops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=(16, 56, 112, 120))
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--pairs", type=int, nargs=2, action="append")
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--dry-runs", type=int, default=8)
    parser.add_argument("--inner-iters", type=int, default=1)
    parser.add_argument("--l2-flush-mb", type=int, default=8000)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--expected-physical-device", type=int, required=True)
    parser.add_argument("--expected-device-uuid", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = args.pairs or [[0, 256], [0, 512]]
    valid_blocks = {0, 128, 256, 512, 1024}
    if not args.rows or min(args.rows) <= 0:
        raise ValueError("rows must contain positive integers")
    if any(set(pair) - valid_blocks for pair in pairs):
        raise ValueError(f"pairs must use blocks from {sorted(valid_blocks)}")
    if args.hidden_size <= 0 or args.hidden_size % 8:
        raise ValueError("hidden-size must be positive and divisible by 8")
    if min(args.repeats, args.dry_runs, args.inner_iters, args.l2_flush_mb) <= 0:
        raise ValueError("timing parameters must be positive")
    if args.inner_iters != 1:
        raise ValueError(
            "cold-L2 evidence requires --inner-iters 1; batching launches after "
            "one flush measures warm-cache work"
        )

    lease_device_fence = verify_lease_device_fence(
        args.expected_physical_device,
        args.expected_device_uuid,
    )
    properties = torch.musa.get_device_properties(0)
    flush_buffer = torch.empty(
        args.l2_flush_mb * 1024 * 1024 // 4, dtype=torch.int32, device="musa"
    )
    output_rows = []

    for row_count in args.rows:
        torch.manual_seed(args.seed + row_count)
        base_x = torch.randn(
            (row_count, args.hidden_size), dtype=torch.bfloat16, device="musa"
        )
        base_residual = torch.randn_like(base_x)
        weight = torch.randn((args.hidden_size,), dtype=torch.bfloat16, device="musa")
        work = {
            block: (
                torch.empty(
                    (args.inner_iters, *base_x.shape),
                    dtype=base_x.dtype,
                    device=base_x.device,
                ),
                torch.empty(
                    (args.inner_iters, *base_residual.shape),
                    dtype=base_residual.dtype,
                    device=base_residual.device,
                ),
            )
            for pair in pairs
            for block in pair
        }

        def prepare(block: int) -> None:
            x, residual = work[block]
            x.copy_(base_x)
            residual.copy_(base_residual)

        def launch(block: int) -> None:
            x, residual = work[block]
            for inner in range(args.inner_iters):
                musa_ops.musa_fused_add_rms_norm(
                    x[inner], residual[inner], weight, args.eps, block_x=block
                )

        for block in sorted(work):
            prepare(block)
            launch(block)
        torch.musa.synchronize()

        for pair in pairs:
            baseline, candidate = pair
            prepare(baseline)
            launch(baseline)
            prepare(candidate)
            launch(candidate)
            torch.musa.synchronize()
            torch.testing.assert_close(
                work[candidate][0].float(),
                work[baseline][0].float(),
                rtol=2e-2,
                atol=2e-2,
            )
            torch.testing.assert_close(
                work[candidate][1].float(),
                work[baseline][1].float(),
                rtol=0.0,
                atol=0.0,
            )
            baseline_samples: list[float] = []
            candidate_samples: list[float] = []
            correctness = True
            for dry in range(args.dry_runs):
                block = pair[dry % 2]
                prepare(block)
                launch(block)
            torch.musa.synchronize()
            for repeat in range(args.repeats):
                order = (
                    (baseline, candidate) if repeat % 2 == 0 else (candidate, baseline)
                )
                measured: dict[int, float] = {}
                for block in order:
                    prepare(block)
                    flush_buffer.zero_()
                    start = torch.musa.Event(enable_timing=True)
                    end = torch.musa.Event(enable_timing=True)
                    start.record()
                    launch(block)
                    end.record()
                    end.synchronize()
                    measured[block] = float(start.elapsed_time(end)) / args.inner_iters
                    x, residual = work[block]
                    correctness = correctness and bool(
                        torch.isfinite(x).all().item()
                        and torch.isfinite(residual).all().item()
                    )
                baseline_samples.append(measured[baseline])
                candidate_samples.append(measured[candidate])

            prepare(candidate)
            launch(candidate)
            torch.musa.synchronize()
            candidate_x = work[candidate][0].float()
            output_absmax = float(candidate_x.abs().max().item())
            # A constant normalized row is valid; use only all-zero output as
            # the uninitialized-output guard.
            output_std = float(candidate_x.std().item())
            poison_output = output_absmax <= 1e-12
            correctness = correctness and not poison_output
            ratios = [c / b for b, c in zip(baseline_samples, candidate_samples)]
            baseline_p50 = statistics.median(baseline_samples)
            candidate_p50 = statistics.median(candidate_samples)
            candidate_seconds = candidate_p50 / 1000.0
            estimated_flops = row_count * args.hidden_size * 4
            algorithmic_bytes = row_count * args.hidden_size * 5 * 2
            output_rows.append(
                {
                    "rows": row_count,
                    "hidden_size": args.hidden_size,
                    "baseline_block_x": baseline,
                    "candidate_block_x": candidate,
                    "baseline_samples_ms": baseline_samples,
                    "candidate_samples_ms": candidate_samples,
                    "ratio_samples": ratios,
                    "baseline_median_ms": baseline_p50,
                    "candidate_median_ms": candidate_p50,
                    "baseline_p90_ms": percentile(baseline_samples, 0.90),
                    "baseline_p99_ms": percentile(baseline_samples, 0.99),
                    "candidate_p90_ms": percentile(candidate_samples, 0.90),
                    "candidate_p99_ms": percentile(candidate_samples, 0.99),
                    "median_ratio": statistics.median(ratios),
                    "geomean_ratio": math.exp(
                        sum(math.log(x) for x in ratios) / len(ratios)
                    ),
                    "ratio_p95": percentile(ratios, 0.95),
                    "ratio_p99": percentile(ratios, 0.99),
                    "ratio_iqr": percentile(ratios, 0.75) - percentile(ratios, 0.25),
                    "speedup_pct": (baseline_p50 / candidate_p50 - 1.0) * 100.0,
                    "theory": {
                        "estimated_flops": estimated_flops,
                        "algorithmic_bytes": algorithmic_bytes,
                        "candidate_tflops": estimated_flops / candidate_seconds / 1e12,
                        "candidate_gbps": algorithmic_bytes / candidate_seconds / 1e9,
                    },
                    "output_absmax": output_absmax,
                    "output_std": output_std,
                    "poison_output": poison_output,
                    "correctness_pass": correctness,
                }
            )

    payload = {
        "schema": "musa-fused-add-rmsnorm-aot-paired-ab.v2",
        "device_name": torch.musa.get_device_name(0),
        "device_capability": [int(properties.major), int(properties.minor)],
        "multiprocessor_count": int(properties.multi_processor_count),
        "dtype": "bfloat16",
        "benchmark": {
            "repeats": args.repeats,
            "dry_runs": args.dry_runs,
            "inner_iters": args.inner_iters,
            "paired_alternating_order": True,
            "l2_flush_mb": args.l2_flush_mb,
            "l2_flush_bytes": flush_buffer.numel() * flush_buffer.element_size(),
            "flush_before_every_timed_launch": True,
            "cache_policy": "cold-l2-per-sample",
        },
        "roofline": {
            "memory_gbps": 1600,
            "bf16_tflops": 500,
        },
        "kernel_source": ("csrc/musa/fused_add_rmsnorm.mu::fused_add_rmsnorm_kernel"),
        "lease_device_fence": lease_device_fence,
        "provenance": provenance(Path(__file__)),
        "rows": output_rows,
    }
    emit_payload(payload, args.output)
    return 0 if all(row["correctness_pass"] for row in output_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
