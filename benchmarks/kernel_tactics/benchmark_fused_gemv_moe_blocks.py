#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired A/B sweep of existing MUSA fused-MoE GEMV AOT blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

# isort: off
import torchada  # noqa: F401  # patch before torch ecosystem imports
import torch
# isort: on

from _benchmark_utils import (
    effective_gemv_block,
    emit_payload,
    percentile,
    provenance,
    verify_lease_device_fence,
)

from vllm_musa import _custom_ops as musa_ops


@dataclass(frozen=True)
class Family:
    name: str
    experts: int
    hidden_size: int
    intermediate_size: int
    topk: int
    weight_dtype: torch.dtype
    folded_shared_expert: bool = False


FAMILIES = {
    "dsv4_fp8": Family("dsv4_fp8", 256, 4096, 256, 6, torch.float8_e4m3fn),
    "qwen_bf16": Family("qwen_bf16", 32, 2048, 128, 8, torch.bfloat16),
    "qwen35_folded_bf16": Family(
        "qwen35_folded_bf16", 33, 2048, 128, 9, torch.bfloat16, True
    ),
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


def parse_baseline_block(value: str) -> tuple[int, int]:
    if value.lower() == "auto":
        return (0, 0)
    return parse_block(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--tokens", type=int, nargs="+", required=True)
    parser.add_argument(
        "--routes",
        choices=("balanced", "hot", "unique_random"),
        nargs="+",
        default=("balanced",),
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
    parser.add_argument(
        "--baseline-block",
        type=parse_baseline_block,
        default=(0, 0),
        help="paired baseline block, or 'auto'",
    )
    parser.add_argument("--dry-runs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=24)
    parser.add_argument("--inner-iters", type=int, default=1)
    parser.add_argument("--l2-flush-mb", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--execution-mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--expected-physical-device", type=int, required=True)
    parser.add_argument("--expected-device-uuid", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


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
    folded_shared_expert: bool,
    seed: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    routed_experts = experts - 1 if folded_shared_expert else experts
    routed_topk = topk - 1 if folded_shared_expert else topk
    offsets = torch.arange(routed_topk, dtype=torch.int64).reshape(1, -1)
    if route == "balanced":
        rows = torch.arange(tokens, dtype=torch.int64).reshape(-1, 1)
        routed = (rows * routed_topk + offsets).remainder(routed_experts)
    elif route == "hot":
        routed = offsets.expand(tokens, -1)
    else:
        generator = torch.Generator().manual_seed(seed)
        scores = torch.rand((tokens, routed_experts), generator=generator)
        routed = scores.topk(routed_topk, dim=-1, sorted=False).indices
    if folded_shared_expert:
        shared = torch.full((tokens, 1), experts - 1, dtype=torch.int64)
        routed = torch.cat((routed, shared), dim=1)
    return routed.to(device=device, dtype=torch.int32)


def route_evidence(topk_ids: torch.Tensor, experts: int) -> dict[str, object]:
    ids = topk_ids.detach().cpu().to(torch.int64)
    histogram = torch.bincount(ids.flatten(), minlength=experts).tolist()
    encoded = json.dumps(ids.tolist(), separators=(",", ":")).encode()
    nonzero = [index for index, count in enumerate(histogram) if count]
    return {
        "topk_ids_sha256": hashlib.sha256(encoded).hexdigest(),
        "expert_load_histogram": histogram,
        "nonzero_experts": nonzero,
        "max_expert_load": max(histogram),
        "min_nonzero_expert_load": min(histogram[index] for index in nonzero),
    }


def workload_theory(family: Family, tokens: int, stage: str) -> tuple[int, int]:
    assignments = tokens * family.topk
    hidden = family.hidden_size
    intermediate = family.intermediate_size
    weight_bytes = 1 if family.weight_dtype == torch.float8_e4m3fn else 2
    if stage == "w1":
        flops = assignments * 4 * hidden * intermediate
        matrix_elements = 2 * hidden * intermediate
        input_elements = hidden
        output_elements = intermediate
    else:
        flops = assignments * 2 * hidden * intermediate
        matrix_elements = hidden * intermediate
        input_elements = intermediate
        output_elements = hidden
    algorithmic_bytes = assignments * (
        matrix_elements * weight_bytes + input_elements * 2 + output_elements * 2
    )
    if family.weight_dtype == torch.float8_e4m3fn:
        algorithmic_bytes += assignments * matrix_elements // (128 * 128) * 4
    return flops, algorithmic_bytes


def effective_block(
    family: Family, tokens: int, stage: str, requested: tuple[int, int]
) -> tuple[tuple[int, int], bool, str | None]:
    """Return the block the checked-in C++ selector will actually launch."""
    # This one-token DSV4 split-tile arm runs before the per-call diagnostic
    # override.  Do not promote a requested block that the kernel ignored.
    return effective_gemv_block(family.name, tokens, stage, requested)


def main() -> int:
    args = parse_args()
    if not args.tokens or min(args.tokens) <= 0:
        raise ValueError("tokens must contain positive integers")
    if min(args.dry_runs, args.repeats, args.inner_iters, args.l2_flush_mb) <= 0:
        raise ValueError("timing arguments must be positive")
    if args.inner_iters != 1:
        raise ValueError(
            "cold-L2 evidence requires --inner-iters 1; batching launches after "
            "one flush measures warm-cache work"
        )

    lease_device_fence = verify_lease_device_fence(
        args.expected_physical_device,
        args.expected_device_uuid,
    )
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
        args.l2_flush_mb * 1024 * 1024 // 4, dtype=torch.int32, device=device
    )
    rows = []
    skipped = []
    vector_length = 16 if family.weight_dtype == torch.float8_e4m3fn else 8

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
                tokens,
                family.topk,
                family.experts,
                route,
                family.folded_shared_expert,
                args.seed + tokens,
                device=device,
            )
            route_metadata = route_evidence(topk_ids, family.experts)
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
                    output_size = family.intermediate_size
                    reduction_size = family.hidden_size
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
                    output_size = family.hidden_size
                    reduction_size = family.intermediate_size

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

                def capture_runner(block: tuple[int, int]):
                    launch(block)
                    torch.musa.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        launch(block)
                    return graph.replay

                baseline_block = args.baseline_block
                if baseline_block != (0, 0):
                    baseline_n, baseline_k = baseline_block
                    baseline_load_size = baseline_k * vector_length
                    if output_size % baseline_n or reduction_size % baseline_load_size:
                        raise ValueError(
                            f"invalid baseline block {baseline_n}x{baseline_k} "
                            f"for stage={stage}, output_size={output_size}, "
                            f"reduction_size={reduction_size}, "
                            f"vector_length={vector_length}"
                        )
                if args.execution_mode == "eager":
                    launch(baseline_block)
                    torch.musa.synchronize()
                    baseline_output = output.clone()
                else:
                    baseline_output = None

                for candidate_block in args.blocks:
                    block_n, block_k = candidate_block
                    load_size = block_k * vector_length
                    if output_size % block_n or reduction_size % load_size:
                        skipped.append(
                            {
                                "family": family.name,
                                "tokens": tokens,
                                "route": route,
                                "stage": stage,
                                "candidate_block": list(candidate_block),
                                "reason": (
                                    f"invalid for output_size={output_size}, "
                                    f"reduction_size={reduction_size}, "
                                    f"vector_length={vector_length}"
                                ),
                            }
                        )
                        continue
                    if args.execution_mode == "graph":
                        baseline_runner = capture_runner(baseline_block)
                        candidate_runner = capture_runner(candidate_block)
                        baseline_runner()
                        torch.musa.synchronize()
                        baseline_output = output.clone()
                        candidate_runner()
                        torch.musa.synchronize()
                        candidate_output = output.clone()
                    else:

                        def baseline_runner() -> None:
                            launch(baseline_block)

                        def candidate_runner() -> None:
                            launch(candidate_block)

                        candidate_runner()
                        torch.musa.synchronize()
                        candidate_output = output.clone()
                    assert baseline_output is not None
                    torch.testing.assert_close(
                        candidate_output.float(),
                        baseline_output.float(),
                        rtol=2e-2,
                        atol=2e-2,
                    )
                    correctness = bool(torch.isfinite(candidate_output).all().item())
                    output_absmax = float(candidate_output.float().abs().max().item())
                    # A valid tiny/quantized GEMV result may have zero variance
                    # (notably one-token FP8 rows).  Only classify an all-zero
                    # result as poison; variance is not a correctness oracle.
                    poison_output = output_absmax <= 1e-12
                    correctness = correctness and not poison_output

                    for dry in range(args.dry_runs):
                        if dry % 2 == 0:
                            baseline_runner()
                        else:
                            candidate_runner()
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
                            flush_buffer.zero_()
                            start = torch.musa.Event(enable_timing=True)
                            end = torch.musa.Event(enable_timing=True)
                            start.record()
                            if block == baseline_block:
                                baseline_runner()
                            else:
                                candidate_runner()
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
                    baseline_p50 = statistics.median(baseline_samples)
                    candidate_p50 = statistics.median(candidate_samples)
                    flops, algorithmic_bytes = workload_theory(family, tokens, stage)
                    candidate_seconds = candidate_p50 / 1000.0
                    effective, request_applied, override_reason = effective_block(
                        family, tokens, stage, tuple(candidate_block)
                    )
                    baseline_effective, baseline_applied, baseline_override_reason = (
                        effective_block(family, tokens, stage, tuple(baseline_block))
                    )
                    rows.append(
                        {
                            "family": family.name,
                            "tokens": tokens,
                            "route": route,
                            "route_evidence": route_metadata,
                            "stage": stage,
                            "topk": family.topk,
                            "activation_shape": list(activation.shape),
                            "weight_shape": list(weight.shape),
                            "baseline_block": list(baseline_block),
                            "baseline_effective_block": list(baseline_effective),
                            "baseline_block_applied": baseline_applied,
                            "baseline_block_override_reason": baseline_override_reason,
                            "candidate_block": list(candidate_block),
                            "effective_block": list(effective),
                            "requested_block_applied": request_applied,
                            "block_override_reason": override_reason,
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
                                sum(math.log(value) for value in ratios) / len(ratios)
                            ),
                            "ratio_iqr": percentile(ratios, 0.75)
                            - percentile(ratios, 0.25),
                            "ratio_p95": percentile(ratios, 0.95),
                            "ratio_p99": percentile(ratios, 0.99),
                            "speedup_pct": (baseline_p50 / candidate_p50 - 1.0) * 100.0,
                            "theory": {
                                "flops": flops,
                                "algorithmic_bytes": algorithmic_bytes,
                                "candidate_tflops": flops / candidate_seconds / 1e12,
                                "candidate_gbps": algorithmic_bytes
                                / candidate_seconds
                                / 1e9,
                            },
                            "output_absmax": output_absmax,
                            "output_std": float(candidate_output.float().std().item()),
                            "poison_output": poison_output,
                            "correctness_pass": correctness,
                        }
                    )

    payload = {
        "schema": "musa-fused-gemv-moe-aot-paired-ab.v2",
        "family": family.name,
        "execution_mode": args.execution_mode,
        "folded_shared_expert": family.folded_shared_expert,
        "device_name": torch.musa.get_device_name(0),
        "device_capability": [int(properties.major), int(properties.minor)],
        "multiprocessor_count": int(properties.multi_processor_count),
        "lease_device_fence": lease_device_fence,
        "weight_dtype": str(family.weight_dtype),
        "benchmark": {
            "dry_runs": args.dry_runs,
            "repeats": args.repeats,
            "inner_iters": args.inner_iters,
            "l2_flush_mb": args.l2_flush_mb,
            "l2_flush_bytes": flush_buffer.numel() * flush_buffer.element_size(),
            "flush_before_every_timed_launch": True,
            "cache_policy": "cold-l2-per-sample",
            "paired_alternating_order": True,
        },
        "roofline": {
            "memory_gbps": 1600,
            "bf16_tflops": 500,
            "fp8_tflops": 1000,
        },
        "kernel_source": "csrc/musa/gemv.mu::musa_gemv_kernel",
        "baseline_block": list(args.baseline_block),
        "provenance": provenance(Path(__file__)),
        "rows": rows,
        "skipped": skipped,
    }
    emit_payload(payload, args.output)
    return 0 if all(row["correctness_pass"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
