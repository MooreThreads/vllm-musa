#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired cold-cache block sweep for the production dense FP8 GEMV AOT op."""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _benchmark_utils import (
    emit_payload,
    percentile,
    provenance,
    verify_lease_device_fence,
)

SUPPORTED_BLOCKS = (
    (4, 32),
    (8, 16),
    (8, 32),
    (16, 8),
    (16, 16),
    (32, 1),
    (32, 4),
    (32, 8),
    (128, 1),
)


@dataclass(frozen=True)
class Family:
    name: str
    output_size: int
    reduction_size: int
    production_tokens: tuple[int, ...]


FAMILIES = {
    "dsv4_o_proj": Family(
        name="dsv4_o_proj",
        output_size=1024,
        reduction_size=4096,
        production_tokens=(1, 2, 4, 8, 16, 32, 64),
    ),
    "dsv4_shared_gate_up": Family(
        name="dsv4_shared_gate_up",
        output_size=512,
        reduction_size=4096,
        production_tokens=(1,),
    ),
}


def parse_block(value: str) -> tuple[int, int]:
    try:
        block_n, block_k = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"invalid block {value!r}; expected NxK"
        ) from exc
    block = (block_n, block_k)
    if block not in SUPPORTED_BLOCKS:
        raise argparse.ArgumentTypeError(
            f"unsupported dense GEMV block {value!r}; expected one of "
            + ", ".join(f"{n}x{k}" for n, k in SUPPORTED_BLOCKS)
        )
    return block


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--tokens", type=int, nargs="+")
    parser.add_argument(
        "--blocks",
        type=parse_block,
        nargs="+",
        default=SUPPORTED_BLOCKS,
    )
    parser.add_argument("--dry-runs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--inner-iters", type=int, default=1)
    parser.add_argument("--l2-flush-mb", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--expected-physical-device", type=int, required=True)
    parser.add_argument("--expected-device-uuid", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def production_block(
    family: str, tokens: int, multiprocessor_count: int | None = None
) -> tuple[int, int] | None:
    if family != "dsv4_o_proj":
        return None
    if multiprocessor_count == 48:
        if tokens == 1:
            return (8, 32)
        if tokens == 8:
            return (16, 8)
        if tokens in (32, 64):
            return (32, 4)
    elif multiprocessor_count == 56:
        if tokens == 8:
            return (16, 16)
        if tokens == 64:
            return (32, 4)
    elif multiprocessor_count == 60:
        if tokens == 8:
            return (8, 16)
        if tokens == 64:
            return (32, 4)
    if tokens in (1, 2, 8):
        return (4, 32)
    if tokens in (4, 32, 64):
        return (8, 16)
    if tokens == 16:
        return (32, 4)
    return None


def _load_runtime() -> tuple[Any, Any]:
    # Import order is part of the MUSA compatibility contract.
    # isort: off
    import torchada  # noqa: F401
    import torch
    from vllm_musa import _custom_ops as musa_ops
    # isort: on

    return torch, musa_ops


def _fp8_constant(torch: Any, shape: tuple[int, ...], device: Any) -> Any:
    return torch.full(shape, 0x20, dtype=torch.uint8, device=device).view(
        torch.float8_e4m3fn
    )


def main() -> int:
    args = parse_args()
    if min(args.dry_runs, args.repeats, args.inner_iters, args.l2_flush_mb) <= 0:
        raise ValueError("timing arguments must be positive")
    if args.inner_iters != 1:
        raise ValueError("cold-L2 evidence requires --inner-iters 1")

    family = FAMILIES[args.family]
    tokens_list = args.tokens or list(family.production_tokens)
    if min(tokens_list) <= 0:
        raise ValueError("tokens must be positive")
    if any(tokens not in family.production_tokens for tokens in tokens_list):
        raise ValueError(
            f"tokens must be production buckets {family.production_tokens} for "
            f"{family.name}"
        )

    lease_device_fence = verify_lease_device_fence(
        args.expected_physical_device,
        args.expected_device_uuid,
    )
    torch, musa_ops = _load_runtime()
    if not torch.musa.is_available() or torch.musa.device_count() != 1:
        raise RuntimeError("dense GEMV benchmark requires exactly one visible MUSA GPU")
    device = torch.device("musa")
    properties = torch.musa.get_device_properties(0)
    torch.manual_seed(args.seed)

    weight = _fp8_constant(
        torch,
        (family.output_size, family.reduction_size),
        device,
    )
    weight_scale = torch.ones(
        (family.output_size // 128, family.reduction_size // 128),
        dtype=torch.float32,
        device=device,
    )
    flush = torch.empty(
        args.l2_flush_mb * 1024 * 1024 // 4,
        dtype=torch.int32,
        device=device,
    )
    rows: list[dict[str, Any]] = []

    for tokens in tokens_list:
        activation = (
            torch.randn(
                (tokens, family.reduction_size),
                dtype=torch.bfloat16,
                device=device,
            )
            * 0.01
        )
        output = torch.empty(
            (tokens, family.output_size),
            dtype=torch.bfloat16,
            device=device,
        )

        def launch(block: tuple[int, int]) -> None:
            for _ in range(args.inner_iters):
                musa_ops.musa_fused_gemv(
                    activation,
                    weight,
                    None,
                    weight_scale,
                    use_swigelu=False,
                    use_rms_norm=False,
                    output=output,
                    block_n=block[0],
                    block_k=block[1],
                )

        baseline = (0, 0)
        launch(baseline)
        torch.musa.synchronize()
        baseline_output = output.clone()

        for candidate in args.blocks:
            block_n, block_k = candidate
            if family.output_size % block_n or family.reduction_size % (block_k * 16):
                continue
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
            output_absmax = float(candidate_output.float().abs().max().item())
            correctness = correctness and output_absmax > 1e-12

            for dry_run in range(args.dry_runs):
                launch(baseline if dry_run % 2 == 0 else candidate)
            torch.musa.synchronize()

            baseline_samples: list[float] = []
            candidate_samples: list[float] = []
            for repeat in range(args.repeats):
                order = (
                    (baseline, candidate) if repeat % 2 == 0 else (candidate, baseline)
                )
                measured: dict[tuple[int, int], float] = {}
                for block in order:
                    flush.zero_()
                    start = torch.musa.Event(enable_timing=True)
                    end = torch.musa.Event(enable_timing=True)
                    start.record()
                    launch(block)
                    end.record()
                    end.synchronize()
                    measured[block] = float(start.elapsed_time(end))
                baseline_samples.append(measured[baseline])
                candidate_samples.append(measured[candidate])

            ratios = [
                candidate_ms / baseline_ms
                for baseline_ms, candidate_ms in zip(
                    baseline_samples, candidate_samples
                )
            ]
            baseline_median = statistics.median(baseline_samples)
            candidate_median = statistics.median(candidate_samples)
            candidate_seconds = candidate_median / 1000.0
            flops = 2 * tokens * family.output_size * family.reduction_size
            algorithmic_bytes = (
                family.output_size * family.reduction_size
                + tokens * (family.output_size + family.reduction_size) * 2
                + weight_scale.numel() * 4
            )
            rows.append(
                {
                    "family": family.name,
                    "tokens": tokens,
                    "activation_shape": list(activation.shape),
                    "weight_shape": list(weight.shape),
                    "weight_scale_shape": list(weight_scale.shape),
                    "baseline_block": [0, 0],
                    "baseline_effective_block": (
                        list(
                            production_block(
                                family.name,
                                tokens,
                                int(properties.multi_processor_count),
                            )
                        )
                        if production_block(
                            family.name,
                            tokens,
                            int(properties.multi_processor_count),
                        )
                        is not None
                        else None
                    ),
                    "candidate_block": list(candidate),
                    "requested_block_applied": True,
                    "baseline_samples_ms": baseline_samples,
                    "candidate_samples_ms": candidate_samples,
                    "ratio_samples": ratios,
                    "baseline_median_ms": baseline_median,
                    "candidate_median_ms": candidate_median,
                    "median_ratio": statistics.median(ratios),
                    "geomean_ratio": math.exp(
                        sum(math.log(value) for value in ratios) / len(ratios)
                    ),
                    "ratio_p95": percentile(ratios, 0.95),
                    "ratio_p99": percentile(ratios, 0.99),
                    "speedup_pct": (baseline_median / candidate_median - 1.0) * 100.0,
                    "theory": {
                        "flops": flops,
                        "algorithmic_bytes": algorithmic_bytes,
                        "candidate_tflops": flops / candidate_seconds / 1e12,
                        "candidate_gbps": algorithmic_bytes / candidate_seconds / 1e9,
                    },
                    "output_absmax": output_absmax,
                    "correctness_pass": correctness,
                    "cache_policy": "cold-l2-per-sample",
                }
            )

    payload = {
        "schema": "musa-dense-fp8-gemv-aot-paired-ab.v1",
        "family": family.name,
        "device_name": torch.musa.get_device_name(0),
        "device_capability": [int(properties.major), int(properties.minor)],
        "multiprocessor_count": int(properties.multi_processor_count),
        "lease_device_fence": lease_device_fence,
        "benchmark": {
            "dry_runs": args.dry_runs,
            "repeats": args.repeats,
            "inner_iters": args.inner_iters,
            "l2_flush_mb": args.l2_flush_mb,
            "l2_flush_bytes": flush.numel() * flush.element_size(),
            "flush_before_every_timed_launch": True,
            "paired_alternating_order": True,
        },
        "kernel_source": "csrc/musa/gemv.mu::musa_fused_gemv",
        "provenance": provenance(Path(__file__)),
        "rows": rows,
        "skipped": [],
    }
    emit_payload(payload, args.output)
    return 0 if all(row["correctness_pass"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
