#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired A/B sweep of existing MUSA dense FP8 GEMV AOT blocks."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics

# isort: off
import torchada  # noqa: F401  # patch before torch ecosystem imports
import torch
# isort: on

from vllm_musa import _custom_ops as musa_ops

OVERRIDE_ENV = "VLLM_MUSA_GEMV_BLOCK"
SHAPES = {
    "dsv4_o_proj": (1024, 4096),
    "dsv4_shared_gate": (512, 4096),
}


def parse_block(value: str) -> tuple[int, int]:
    try:
        block_n, block_k = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"invalid block {value!r}; expected NxK"
        ) from exc
    if block_n <= 0 or block_k <= 0:
        raise argparse.ArgumentTypeError("block dimensions must be positive")
    return block_n, block_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=sorted(SHAPES), required=True)
    parser.add_argument("--tokens", type=int, nargs="+", required=True)
    parser.add_argument(
        "--blocks",
        type=parse_block,
        nargs="+",
        default=((4, 32), (8, 16), (16, 8), (32, 4), (32, 8)),
    )
    parser.add_argument("--dry-runs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--inner-iters", type=int, default=16)
    parser.add_argument("--l2-flush-mb", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def set_override(block: tuple[int, int] | None) -> None:
    if block is None:
        os.environ.pop(OVERRIDE_ENV, None)
    else:
        os.environ[OVERRIDE_ENV] = f"{block[0]}x{block[1]}"


def main() -> int:
    args = parse_args()
    if not args.tokens or min(args.tokens) <= 0:
        raise ValueError("tokens must contain positive integers")
    if min(args.dry_runs, args.repeats, args.inner_iters, args.l2_flush_mb) <= 0:
        raise ValueError("timing arguments must be positive")

    output_size, reduction_size = SHAPES[args.route]
    valid_blocks = [
        block
        for block in args.blocks
        if output_size % block[0] == 0 and reduction_size % (block[1] * 16) == 0
    ]
    skipped = [block for block in args.blocks if block not in valid_blocks]

    device = torch.device("musa")
    torch.manual_seed(args.seed)
    properties = torch.musa.get_device_properties(0)
    weight = torch.full(
        (output_size, reduction_size), 0x20, dtype=torch.uint8, device=device
    ).view(torch.float8_e4m3fn)
    weight_scale = torch.ones(
        (output_size, reduction_size // 128), dtype=torch.float32, device=device
    )
    flush_buffer = torch.empty(
        args.l2_flush_mb * 1024 * 1024 // 4, dtype=torch.float32, device=device
    )
    original_override = os.environ.get(OVERRIDE_ENV)
    rows = []

    try:
        for tokens in args.tokens:
            activation = (
                torch.randn(
                    (tokens, reduction_size), dtype=torch.bfloat16, device=device
                )
                * 0.01
            )
            output = torch.empty(
                (tokens, output_size), dtype=torch.bfloat16, device=device
            )

            def launch(block: tuple[int, int] | None) -> None:
                set_override(block)
                for _ in range(args.inner_iters):
                    musa_ops.musa_fused_gemv(
                        activation,
                        weight,
                        None,
                        weight_scale,
                        output=output,
                    )

            launch(None)
            torch.musa.synchronize()
            baseline_output = output.clone()

            for candidate in valid_blocks:
                launch(candidate)
                torch.musa.synchronize()
                candidate_output = output.clone()
                torch.testing.assert_close(
                    candidate_output.float(),
                    baseline_output.float(),
                    rtol=2e-2,
                    atol=2e-2,
                )
                correctness = bool(torch.isfinite(candidate_output).all().item())

                for dry in range(args.dry_runs):
                    launch(None if dry % 2 == 0 else candidate)
                torch.musa.synchronize()

                baseline_samples: list[float] = []
                candidate_samples: list[float] = []
                for repeat in range(args.repeats):
                    order = (None, candidate) if repeat % 2 == 0 else (candidate, None)
                    measured: dict[tuple[int, int] | None, float] = {}
                    for block in order:
                        set_override(block)
                        flush_buffer.add_(1)
                        start = torch.musa.Event(enable_timing=True)
                        end = torch.musa.Event(enable_timing=True)
                        start.record()
                        for _ in range(args.inner_iters):
                            musa_ops.musa_fused_gemv(
                                activation,
                                weight,
                                None,
                                weight_scale,
                                output=output,
                            )
                        end.record()
                        end.synchronize()
                        measured[block] = (
                            float(start.elapsed_time(end)) / args.inner_iters
                        )
                    baseline_samples.append(measured[None])
                    candidate_samples.append(measured[candidate])

                ratios = [
                    candidate_time / baseline_time
                    for baseline_time, candidate_time in zip(
                        baseline_samples, candidate_samples
                    )
                ]
                rows.append(
                    {
                        "route": args.route,
                        "tokens": tokens,
                        "activation_shape": list(activation.shape),
                        "weight_shape": list(weight.shape),
                        "candidate_block": list(candidate),
                        "baseline_samples_ms": baseline_samples,
                        "candidate_samples_ms": candidate_samples,
                        "ratio_samples": ratios,
                        "baseline_median_ms": statistics.median(baseline_samples),
                        "candidate_median_ms": statistics.median(candidate_samples),
                        "median_ratio": statistics.median(ratios),
                        "geomean_ratio": math.exp(
                            sum(math.log(value) for value in ratios) / len(ratios)
                        ),
                        "ratio_iqr": percentile(ratios, 0.75)
                        - percentile(ratios, 0.25),
                        "ratio_p95": percentile(ratios, 0.95),
                        "correctness_pass": correctness,
                    }
                )
    finally:
        if original_override is None:
            set_override(None)
        else:
            os.environ[OVERRIDE_ENV] = original_override

    print(
        json.dumps(
            {
                "schema": "musa-dense-gemv-aot-paired-ab.v1",
                "route": args.route,
                "device_name": torch.musa.get_device_name(0),
                "device_capability": [int(properties.major), int(properties.minor)],
                "multiprocessor_count": int(properties.multi_processor_count),
                "benchmark": {
                    "dry_runs": args.dry_runs,
                    "repeats": args.repeats,
                    "inner_iters": args.inner_iters,
                    "l2_flush_mb": args.l2_flush_mb,
                    "paired_alternating_order": True,
                },
                "skipped_blocks": [list(block) for block in skipped],
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if all(row["correctness_pass"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
