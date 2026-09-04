#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired cold-cache requested-thread sweep for native-JIT RMSNorm kernels."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

from _benchmark_utils import (
    emit_payload,
    percentile,
    provenance,
    verify_lease_device_fence,
)

SUPPORTED_THREADS = (32, 64, 96, 128, 192, 256, 320, 512, 640, 896, 1024)


def parse_threads(value: str) -> int:
    try:
        threads = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"threads must be an integer: {value!r}"
        ) from exc
    if threads not in SUPPORTED_THREADS:
        raise argparse.ArgumentTypeError(
            f"threads must be one of {SUPPORTED_THREADS}, got {threads}"
        )
    return threads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("plain", "gemma", "fused", "fused_gemma"),
        required=True,
    )
    parser.add_argument("--rows", type=int, nargs="+", required=True)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", required=True)
    parser.add_argument(
        "--threads", type=parse_threads, nargs="+", default=SUPPORTED_THREADS
    )
    parser.add_argument("--weight-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--dry-runs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--l2-flush-mb", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--expected-physical-device", type=int, required=True)
    parser.add_argument("--expected-device-uuid", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def vec8_threads(hidden_size: int) -> int:
    rounded = ((hidden_size // 8 + 31) // 32) * 32
    return min(rounded, 1024)


def production_threads(
    mode: str,
    rows: int,
    hidden_size: int,
    multiprocessor_count: int | None = None,
) -> int | None:
    if (
        hidden_size == 5120
        and multiprocessor_count == 48
        and mode == "plain"
        and rows == 512
    ):
        return 256
    exact_h5120_mp_tactic = (
        multiprocessor_count == 56
        and rows == 512
        and mode in {"plain", "gemma", "fused", "fused_gemma"}
    ) or (
        multiprocessor_count == 60
        and (mode, rows)
        in {
            ("plain", 512),
            ("gemma", 512),
            ("gemma", 4096),
            ("fused_gemma", 512),
        }
    )
    if hidden_size == 5120 and exact_h5120_mp_tactic:
        return 320
    if mode in ("fused", "fused_gemma"):
        return vec8_threads(hidden_size)
    if rows <= 16 and hidden_size in (1024, 2048):
        return None
    if hidden_size <= 512:
        return 64
    if hidden_size <= 4096:
        if rows <= 16:
            return min(vec8_threads(hidden_size), 512)
        if rows <= 256:
            return min(vec8_threads(hidden_size), 256)
        return 128
    if hidden_size <= 8192:
        return min(vec8_threads(hidden_size), 512)
    return min(vec8_threads(hidden_size), 896)


def _load_runtime() -> tuple[Any, Any, Any]:
    # Import order is part of the MUSA compatibility contract.
    # isort: off
    import torchada  # noqa: F401
    import torch
    from vllm_musa.jit_kernel.csrc import norm
    from vllm_musa.tuning import prime_musa_kernel_hardware
    # isort: on

    return torch, norm, prime_musa_kernel_hardware


def main() -> int:
    args = parse_args()
    if min(args.rows) <= 0 or min(args.hidden_sizes) <= 0:
        raise ValueError("rows and hidden sizes must be positive")
    if any(hidden_size % 8 for hidden_size in args.hidden_sizes):
        raise ValueError("requested-thread sweep requires hidden sizes divisible by 8")
    if max(args.hidden_sizes) > 32768:
        raise ValueError("requested-thread sweep supports hidden sizes <=32768")
    if min(args.dry_runs, args.repeats, args.l2_flush_mb) <= 0:
        raise ValueError("timing arguments must be positive")
    if args.mode in ("plain", "gemma") and args.weight_dtype != "bf16":
        raise ValueError("plain RMSNorm requires activation-matching BF16 weight")

    lease_device_fence = verify_lease_device_fence(
        args.expected_physical_device,
        args.expected_device_uuid,
    )
    torch, norm, prime_hardware = _load_runtime()
    if not torch.musa.is_available() or torch.musa.device_count() != 1:
        raise RuntimeError("JIT RMSNorm benchmark requires one visible MUSA GPU")
    device = torch.device("musa")
    properties = torch.musa.get_device_properties(0)
    primed_hardware = prime_hardware(0)
    if primed_hardware.multiprocessor_count != int(properties.multi_processor_count):
        raise RuntimeError("primed hardware does not match runtime device")
    torch.manual_seed(args.seed)
    flush = torch.empty(
        args.l2_flush_mb * 1024 * 1024 // 4,
        dtype=torch.int32,
        device=device,
    )
    rows_out: list[dict[str, Any]] = []

    for hidden_size in args.hidden_sizes:
        for rows in args.rows:
            original_input = torch.randn(
                (rows, hidden_size), dtype=torch.bfloat16, device=device
            )
            original_residual = torch.randn_like(original_input)
            weight_dtype = (
                torch.bfloat16 if args.weight_dtype == "bf16" else torch.float32
            )
            weight = torch.randn((hidden_size,), dtype=weight_dtype, device=device)
            input_tensor = original_input.clone()
            residual = original_residual.clone()
            output = torch.empty_like(input_tensor)

            def reset_mutable_inputs() -> None:
                if args.mode in ("fused", "fused_gemma"):
                    input_tensor.copy_(original_input)
                    residual.copy_(original_residual)

            def launch(threads: int) -> tuple[Any, ...]:
                if args.mode in ("plain", "gemma"):
                    op = norm.gemma_rmsnorm if args.mode == "gemma" else norm.rmsnorm
                    op(
                        input_tensor,
                        weight,
                        1e-6,
                        out=output,
                        block_threads=threads,
                    )
                    return (output,)
                norm.fused_add_rmsnorm(
                    input_tensor,
                    residual,
                    weight,
                    1e-6,
                    gemma=args.mode == "fused_gemma",
                    block_threads=threads,
                )
                return (input_tensor, residual)

            reset_mutable_inputs()
            launch(0)
            torch.musa.synchronize()
            reset_mutable_inputs()
            baseline_outputs = tuple(tensor.clone() for tensor in launch(0))
            torch.musa.synchronize()

            for candidate_threads in args.threads:
                reset_mutable_inputs()
                candidate_outputs = launch(candidate_threads)
                torch.musa.synchronize()
                for candidate, baseline in zip(candidate_outputs, baseline_outputs):
                    torch.testing.assert_close(
                        candidate.float(), baseline.float(), rtol=2e-2, atol=2e-2
                    )
                correctness = all(
                    bool(torch.isfinite(tensor).all().item())
                    and float(tensor.float().abs().max().item()) > 1e-12
                    for tensor in candidate_outputs
                )

                for dry_run in range(args.dry_runs):
                    reset_mutable_inputs()
                    launch(0 if dry_run % 2 == 0 else candidate_threads)
                torch.musa.synchronize()

                baseline_samples: list[float] = []
                candidate_samples: list[float] = []
                for repeat in range(args.repeats):
                    order = (
                        (0, candidate_threads)
                        if repeat % 2 == 0
                        else (candidate_threads, 0)
                    )
                    measured: dict[int, float] = {}
                    for threads in order:
                        reset_mutable_inputs()
                        flush.zero_()
                        start = torch.musa.Event(enable_timing=True)
                        end = torch.musa.Event(enable_timing=True)
                        start.record()
                        launch(threads)
                        end.record()
                        end.synchronize()
                        measured[threads] = float(start.elapsed_time(end))
                    baseline_samples.append(measured[0])
                    candidate_samples.append(measured[candidate_threads])

                ratios = [
                    candidate / baseline
                    for baseline, candidate in zip(baseline_samples, candidate_samples)
                ]
                baseline_median = statistics.median(baseline_samples)
                candidate_median = statistics.median(candidate_samples)
                rows_out.append(
                    {
                        "mode": args.mode,
                        "rows": rows,
                        "hidden_size": hidden_size,
                        "activation_dtype": "torch.bfloat16",
                        "weight_dtype": f"torch.{args.weight_dtype}",
                        "baseline_threads": 0,
                        "baseline_effective_threads": production_threads(
                            args.mode,
                            rows,
                            hidden_size,
                            primed_hardware.multiprocessor_count,
                        ),
                        "baseline_special_kernel": (
                            production_threads(
                                args.mode,
                                rows,
                                hidden_size,
                                primed_hardware.multiprocessor_count,
                            )
                            is None
                        ),
                        "candidate_threads": candidate_threads,
                        "requested_threads_applied": True,
                        "baseline_samples_ms": baseline_samples,
                        "candidate_samples_ms": candidate_samples,
                        "baseline_median_ms": baseline_median,
                        "candidate_median_ms": candidate_median,
                        "median_ratio": statistics.median(ratios),
                        "ratio_p95": percentile(ratios, 0.95),
                        "ratio_p99": percentile(ratios, 0.99),
                        "speedup_pct": (baseline_median / candidate_median - 1.0)
                        * 100.0,
                        "correctness_pass": correctness,
                        "cache_policy": "cold-l2-per-sample",
                    }
                )

    emit_payload(
        {
            "schema": "musa-jit-rmsnorm-threads-paired-ab.v1",
            "mode": args.mode,
            "device_name": torch.musa.get_device_name(0),
            "device_capability": [int(properties.major), int(properties.minor)],
            "multiprocessor_count": int(properties.multi_processor_count),
            "primed_hardware": {
                "device_capability": list(primed_hardware.device_capability),
                "multiprocessor_count": primed_hardware.multiprocessor_count,
            },
            "lease_device_fence": lease_device_fence,
            "benchmark": {
                "dry_runs": args.dry_runs,
                "repeats": args.repeats,
                "inner_iters": 1,
                "l2_flush_mb": args.l2_flush_mb,
                "l2_flush_bytes": flush.numel() * flush.element_size(),
                "flush_before_every_timed_launch": True,
                "paired_alternating_order": True,
            },
            "kernel_source": "vllm_musa/jit_kernel/csrc/norm/rmsnorm.mu",
            "provenance": provenance(Path(__file__)),
            "rows": rows_out,
            "skipped": [],
        },
        args.output,
    )
    return 0 if all(row["correctness_pass"] for row in rows_out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
