#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interleaved cold-cache sweep of existing fused-add-RMSNorm AOT blocks."""

from __future__ import annotations

import argparse
import json
import os
import statistics

# isort: off
import torchada  # noqa: F401  # patch before torch ecosystem imports
import torch
# isort: on

from vllm_musa import _custom_ops as musa_ops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=(1, 16, 32, 56, 60, 64, 112, 120, 128, 256, 512),
    )
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument(
        "--blocks", type=int, nargs="+", default=(0, 128, 256, 512, 1024)
    )
    parser.add_argument("--dry-runs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--l2-flush-mb", type=int, default=8000)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_fp32 = x.float() + residual.float()
    residual_out = residual_fp32.to(x.dtype)
    normalized = residual_fp32 * torch.rsqrt(
        residual_fp32.pow(2).mean(dim=-1, keepdim=True) + eps
    )
    return (normalized * weight.float()).to(x.dtype), residual_out


def main() -> int:
    args = parse_args()
    manual_override = os.environ.get("VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X")
    if manual_override is not None:
        raise RuntimeError(
            "unset VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X for an unbiased block sweep"
        )
    valid_blocks = {0, 128, 256, 512, 1024}
    if not args.rows or min(args.rows) <= 0:
        raise ValueError("rows must contain positive integers")
    if set(args.blocks) - valid_blocks:
        raise ValueError(f"blocks must be drawn from {sorted(valid_blocks)}")
    if args.hidden_size <= 0 or args.hidden_size % 8:
        raise ValueError("hidden-size must be positive and divisible by 8")
    if min(args.dry_runs, args.repeats, args.l2_flush_mb) <= 0:
        raise ValueError("timing parameters must be positive")

    properties = torch.musa.get_device_properties(0)
    flush_elements = args.l2_flush_mb * 1024 * 1024 // 4
    flush_buffer = torch.empty(flush_elements, dtype=torch.float32, device="musa")
    rows_out = []

    for num_rows in args.rows:
        torch.manual_seed(args.seed + num_rows)
        base_x = torch.randn(
            (num_rows, args.hidden_size), dtype=torch.bfloat16, device="musa"
        )
        base_residual = torch.randn_like(base_x)
        weight = torch.randn((args.hidden_size,), dtype=torch.bfloat16, device="musa")
        expected, expected_residual = reference(base_x, base_residual, weight, args.eps)
        working = {
            block: (torch.empty_like(base_x), torch.empty_like(base_residual))
            for block in args.blocks
        }

        def reset_and_run(block: int) -> None:
            x, residual = working[block]
            x.copy_(base_x)
            residual.copy_(base_residual)
            musa_ops.musa_fused_add_rms_norm(
                x, residual, weight, args.eps, block_x=block
            )

        correctness = {}
        for block in args.blocks:
            reset_and_run(block)
            torch.musa.synchronize()
            output, residual_out = working[block]
            torch.testing.assert_close(
                output.float(), expected.float(), rtol=2e-2, atol=2e-2
            )
            torch.testing.assert_close(
                residual_out.float(),
                expected_residual.float(),
                rtol=0.0,
                atol=0.0,
            )
            correctness[block] = bool(
                torch.isfinite(output).all().item()
                and torch.isfinite(residual_out).all().item()
            )

        for dry_run in range(args.dry_runs):
            order = (
                args.blocks[dry_run % len(args.blocks) :]
                + args.blocks[: dry_run % len(args.blocks)]
            )
            for block in order:
                flush_buffer.add_(1)
                reset_and_run(block)
        torch.musa.synchronize()

        samples = {block: [] for block in args.blocks}
        for repeat in range(args.repeats):
            offset = repeat % len(args.blocks)
            order = args.blocks[offset:] + args.blocks[:offset]
            for block in order:
                x, residual = working[block]
                x.copy_(base_x)
                residual.copy_(base_residual)
                flush_buffer.add_(1)
                start = torch.musa.Event(enable_timing=True)
                end = torch.musa.Event(enable_timing=True)
                start.record()
                musa_ops.musa_fused_add_rms_norm(
                    x, residual, weight, args.eps, block_x=block
                )
                end.record()
                end.synchronize()
                samples[block].append(float(start.elapsed_time(end)))

        for block in args.blocks:
            block_samples = samples[block]
            rows_out.append(
                {
                    "rows": num_rows,
                    "hidden_size": args.hidden_size,
                    "block_x": block,
                    "samples_ms": block_samples,
                    "median_ms": statistics.median(block_samples),
                    "p95_ms": percentile(block_samples, 0.95),
                    "iqr_ms": percentile(block_samples, 0.75)
                    - percentile(block_samples, 0.25),
                    "correctness_pass": correctness[block],
                }
            )

    print(
        json.dumps(
            {
                "schema": "musa-fused-add-rmsnorm-aot-block-sweep.v1",
                "device_name": torch.musa.get_device_name(0),
                "device_capability": [int(properties.major), int(properties.minor)],
                "multiprocessor_count": int(properties.multi_processor_count),
                "dtype": "bfloat16",
                "eps": args.eps,
                "blocks": args.blocks,
                "manual_override": manual_override,
                "benchmark": {
                    "dry_runs": args.dry_runs,
                    "repeats": args.repeats,
                    "interleaved": True,
                    "l2_flush_mb": args.l2_flush_mb,
                },
                "rows": rows_out,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if all(row["correctness_pass"] for row in rows_out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
