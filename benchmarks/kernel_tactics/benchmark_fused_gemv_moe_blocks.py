#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired A/B sweep of existing MUSA fused-MoE GEMV AOT blocks."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass

# isort: off
import torchada  # noqa: F401  # patch before torch ecosystem imports
import torch
# isort: on

from vllm_musa import _custom_ops as musa_ops


@dataclass(frozen=True)
class Family:
    name: str
    experts: int
    hidden_size: int
    intermediate_size: int
    topk: int
    weight_dtype: torch.dtype


FAMILIES = {
    "dsv4_fp8": Family("dsv4_fp8", 256, 4096, 256, 6, torch.float8_e4m3fn),
    "qwen_bf16": Family("qwen_bf16", 256, 2048, 128, 8, torch.bfloat16),
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
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--tokens", type=int, nargs="+", required=True)
    parser.add_argument(
        "--routes", choices=("balanced", "hot"), nargs="+", default=("balanced",)
    )
    parser.add_argument(
        "--stages", choices=("w1", "w2"), nargs="+", default=("w1", "w2")
    )
    parser.add_argument(
        "--blocks",
        type=parse_block,
        nargs="+",
        default=((4, 32), (8, 16), (16, 8), (32, 4), (32, 8)),
    )
    parser.add_argument("--dry-runs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=24)
    parser.add_argument("--inner-iters", type=int, default=8)
    parser.add_argument("--l2-flush-mb", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def fp8_constant(shape: tuple[int, ...], *, device: torch.device) -> torch.Tensor:
    # E4M3 byte 0x20 is a small finite positive value. Creating the payload as
    # uint8 avoids a multi-gigabyte float32 staging allocation for DSV4 weights.
    return torch.full(shape, 0x20, dtype=torch.uint8, device=device).view(
        torch.float8_e4m3fn
    )


def make_weight(
    shape: tuple[int, ...], dtype: torch.dtype, *, device: torch.device
) -> torch.Tensor:
    if dtype == torch.float8_e4m3fn:
        return fp8_constant(shape, device=device)
    return torch.randn(shape, dtype=dtype, device=device) * 0.01


def make_route_ids(
    tokens: int,
    topk: int,
    experts: int,
    route: str,
    *,
    device: torch.device,
) -> torch.Tensor:
    if route == "hot":
        return torch.zeros((tokens, topk), dtype=torch.int32, device=device)
    return (
        torch.arange(tokens * topk, dtype=torch.int32, device=device)
        .reshape(tokens, topk)
        .remainder(experts)
    )


def main() -> int:
    args = parse_args()
    if not args.tokens or min(args.tokens) <= 0:
        raise ValueError("tokens must contain positive integers")
    if min(args.dry_runs, args.repeats, args.inner_iters, args.l2_flush_mb) <= 0:
        raise ValueError("timing arguments must be positive")

    family = FAMILIES[args.family]
    device = torch.device("musa")
    torch.manual_seed(args.seed)
    properties = torch.musa.get_device_properties(0)

    w1 = make_weight(
        (family.experts, 2 * family.intermediate_size, family.hidden_size),
        family.weight_dtype,
        device=device,
    )
    w2 = make_weight(
        (family.experts, family.hidden_size, family.intermediate_size),
        family.weight_dtype,
        device=device,
    )
    if family.weight_dtype == torch.float8_e4m3fn:
        w1_scale = torch.ones(
            (
                family.experts,
                (2 * family.intermediate_size) // 128,
                family.hidden_size // 128,
            ),
            dtype=torch.float32,
            device=device,
        )
        w2_scale = torch.ones(
            (
                family.experts,
                family.hidden_size // 128,
                family.intermediate_size // 128,
            ),
            dtype=torch.float32,
            device=device,
        )
    else:
        w1_scale = None
        w2_scale = None

    flush_buffer = torch.empty(
        args.l2_flush_mb * 1024 * 1024 // 4, dtype=torch.float32, device=device
    )
    rows = []

    for tokens in args.tokens:
        hidden = (
            torch.randn(
                (tokens, family.hidden_size), dtype=torch.bfloat16, device=device
            )
            * 0.01
        )
        intermediate = (
            torch.randn(
                (tokens * family.topk, family.intermediate_size),
                dtype=torch.bfloat16,
                device=device,
            )
            * 0.01
        )

        for route in args.routes:
            topk_ids = make_route_ids(
                tokens, family.topk, family.experts, route, device=device
            )
            topk_weights = torch.full(
                (tokens, family.topk),
                1.0 / family.topk,
                dtype=torch.float32,
                device=device,
            )

            for stage in args.stages:
                if stage == "w1":
                    activation = hidden
                    weight = w1
                    weight_scale = w1_scale
                    output = torch.empty(
                        (tokens * family.topk, family.intermediate_size),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    kernel_topk = family.topk
                    use_swigelu = True
                    mul_routed_weight = False
                else:
                    activation = intermediate
                    weight = w2
                    weight_scale = w2_scale
                    output = torch.empty(
                        (tokens, family.topk, family.hidden_size),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    kernel_topk = 1
                    use_swigelu = False
                    mul_routed_weight = True

                def launch(block: tuple[int, int]) -> None:
                    block_n, block_k = block
                    for _ in range(args.inner_iters):
                        musa_ops.musa_fused_gemv_moe(
                            activation,
                            weight,
                            output,
                            None,
                            weight_scale,
                            topk_weights,
                            topk_ids,
                            mul_routed_weight,
                            kernel_topk,
                            False,
                            use_swigelu,
                            block_n=block_n,
                            block_k=block_k,
                        )

                baseline_block = (0, 0)
                launch(baseline_block)
                torch.musa.synchronize()
                baseline_output = output.clone()

                for candidate_block in args.blocks:
                    launch(candidate_block)
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
                        launch(baseline_block if dry % 2 == 0 else candidate_block)
                    torch.musa.synchronize()

                    baseline_samples: list[float] = []
                    candidate_samples: list[float] = []
                    for repeat in range(args.repeats):
                        order = (
                            (baseline_block, candidate_block)
                            if repeat % 2 == 0
                            else (candidate_block, baseline_block)
                        )
                        measured: dict[tuple[int, int], float] = {}
                        for block in order:
                            flush_buffer.add_(1)
                            start = torch.musa.Event(enable_timing=True)
                            end = torch.musa.Event(enable_timing=True)
                            start.record()
                            launch(block)
                            end.record()
                            end.synchronize()
                            measured[block] = (
                                float(start.elapsed_time(end)) / args.inner_iters
                            )
                        baseline_samples.append(measured[baseline_block])
                        candidate_samples.append(measured[candidate_block])

                    ratios = [
                        candidate / baseline
                        for baseline, candidate in zip(
                            baseline_samples, candidate_samples
                        )
                    ]
                    rows.append(
                        {
                            "family": family.name,
                            "tokens": tokens,
                            "route": route,
                            "stage": stage,
                            "topk": family.topk,
                            "activation_shape": list(activation.shape),
                            "weight_shape": list(weight.shape),
                            "baseline_block": list(baseline_block),
                            "candidate_block": list(candidate_block),
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

    print(
        json.dumps(
            {
                "schema": "musa-fused-gemv-moe-aot-paired-ab.v1",
                "family": family.name,
                "device_name": torch.musa.get_device_name(0),
                "device_capability": [int(properties.major), int(properties.minor)],
                "multiprocessor_count": int(properties.multi_processor_count),
                "weight_dtype": str(family.weight_dtype),
                "benchmark": {
                    "dry_runs": args.dry_runs,
                    "repeats": args.repeats,
                    "inner_iters": args.inner_iters,
                    "l2_flush_mb": args.l2_flush_mb,
                    "paired_alternating_order": True,
                },
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if all(row["correctness_pass"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
