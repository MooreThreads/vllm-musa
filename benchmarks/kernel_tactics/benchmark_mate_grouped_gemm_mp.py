#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure MATE grouped-GEMM sensitivity to its runtime MP-count hint."""

from __future__ import annotations

import argparse
import json
import statistics

# isort: off
import torchada  # noqa: F401  # patch before torch ecosystem imports
import torch
# isort: on

from mate.gemm import ragged_m_moe_gemm_16bit
from mate.testing.utils import bench_gpu_time_with_musa_event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--tokens-per-expert", type=int, default=128)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--mp-counts", type=int, nargs="+", default=(48, 52, 56, 60))
    parser.add_argument("--dry-runs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--l2-flush-mb", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def main() -> int:
    args = parse_args()
    if min(args.experts, args.tokens_per_expert, args.n, args.k) <= 0:
        raise ValueError("shape dimensions must be positive")
    if not args.mp_counts or min(args.mp_counts) <= 0:
        raise ValueError("mp-counts must contain positive integers")

    device = torch.device("musa")
    properties = torch.musa.get_device_properties(0)
    physical_mps = int(properties.multi_processor_count)
    if max(args.mp_counts) > physical_mps:
        raise ValueError(
            f"requested MP count exceeds physical count: {args.mp_counts} > {physical_mps}"
        )

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    total_tokens = args.experts * args.tokens_per_expert
    input_a = torch.randn(
        (total_tokens, args.k),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    input_b = torch.randn(
        (args.experts, args.n, args.k),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    expert_ids = torch.arange(args.experts, dtype=torch.int32, device=device)
    expert_ids = expert_ids.repeat_interleave(args.tokens_per_expert)
    output = torch.empty((total_tokens, args.n), dtype=torch.bfloat16, device=device)

    def run(num_mps: int) -> torch.Tensor:
        return ragged_m_moe_gemm_16bit(
            input_a,
            input_b,
            expert_ids,
            output,
            gemm_mode="per_token",
            num_mp=num_mps,
            alignment_m=args.tokens_per_expert,
            backend="mutlass",
        )

    reference = run(physical_mps).clone()
    torch.musa.synchronize()
    rows = []
    for num_mps in args.mp_counts:
        candidate = run(num_mps).clone()
        torch.musa.synchronize()
        difference = (candidate.float() - reference.float()).abs()
        samples = [
            float(value)
            for value in bench_gpu_time_with_musa_event(
                lambda: run(num_mps),
                dry_run_iters=args.dry_runs,
                repeat_iters=args.repeats,
                l2_flush=True,
                l2_flush_size_mb=args.l2_flush_mb,
            )
        ]
        rows.append(
            {
                "num_mps": num_mps,
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
                "p95_ms": percentile(samples, 0.95),
                "iqr_ms": percentile(samples, 0.75) - percentile(samples, 0.25),
                "max_abs_diff": float(difference.max().item()),
                "correctness_pass": bool(torch.equal(candidate, reference)),
            }
        )

    payload = {
        "schema": "musa-mate-grouped-gemm-mp-sensitivity.v1",
        "device_name": torch.musa.get_device_name(0),
        "device_capability": [int(properties.major), int(properties.minor)],
        "physical_multiprocessor_count": physical_mps,
        "shape": {
            "experts": args.experts,
            "tokens_per_expert": args.tokens_per_expert,
            "total_tokens": total_tokens,
            "n": args.n,
            "k": args.k,
            "dtype": "bfloat16",
        },
        "benchmark": {
            "dry_runs": args.dry_runs,
            "repeats": args.repeats,
            "l2_flush": True,
            "l2_flush_mb": args.l2_flush_mb,
        },
        "rows": rows,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(row["correctness_pass"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
