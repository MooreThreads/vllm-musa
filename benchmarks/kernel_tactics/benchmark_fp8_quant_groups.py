#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold-L2 paired sweep for local group-128 FP8 quantization kernels."""

from __future__ import annotations

import argparse
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


@dataclass(frozen=True)
class Mode:
    name: str
    input_multiplier: int
    clamp: bool


MODES = {
    "quant": Mode("quant", 1, False),
    "silu": Mode("silu", 2, False),
    "silu_clamp": Mode("silu_clamp", 2, True),
}
PRODUCTION_TOKENS = (
    1,
    2,
    4,
    5,
    6,
    8,
    12,
    16,
    20,
    24,
    30,
    32,
    36,
    48,
    64,
    80,
    96,
    120,
    128,
    192,
    256,
    288,
    320,
    384,
    480,
    512,
    768,
    1536,
    1920,
    2048,
    4096,
)
PRODUCTION_HIDDEN_SIZES = (
    128,
    256,
    512,
    1024,
    1536,
    2048,
    2560,
    3072,
    3584,
    4096,
    5120,
    7168,
    8192,
)
SUPPORTED_GROUPS_PER_BLOCK = (1, 2, 4, 8, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", choices=tuple(MODES), nargs="+", required=True)
    parser.add_argument("--tokens", type=int, nargs="+", required=True)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", required=True)
    parser.add_argument(
        "--groups-per-block",
        type=int,
        nargs="+",
        default=list(SUPPORTED_GROUPS_PER_BLOCK),
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--dry-runs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--l2-flush-mb", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--swiglu-limit", type=float, default=7.0)
    parser.add_argument("--expected-physical-device", type=int, required=True)
    parser.add_argument("--expected-device-uuid", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def production_groups_per_block(num_groups: int) -> int:
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    for candidate in reversed(SUPPORTED_GROUPS_PER_BLOCK):
        if num_groups % candidate == 0:
            return candidate
    return 1


def _load_runtime() -> Any:
    # Import order is part of the MUSA compatibility contract.
    # isort: off
    import torchada  # noqa: F401
    import torch
    from vllm_musa import _custom_ops as _musa_ops  # noqa: F401
    # isort: on

    return torch


def _allocate_outputs(torch: Any, tokens: int, hidden_size: int, device: Any):
    output_q = torch.empty(
        (tokens, hidden_size), dtype=torch.float8_e4m3fn, device=device
    )
    output_s = torch.empty(
        (tokens, hidden_size // 128), dtype=torch.float32, device=device
    )
    return output_q, output_s


def main() -> int:
    args = parse_args()
    if min(args.dry_runs, args.repeats, args.l2_flush_mb) <= 0:
        raise ValueError("timing arguments must be positive")
    if args.group_size != 128:
        raise ValueError("local vec kernels require --group-size 128")
    if any(value not in PRODUCTION_TOKENS for value in args.tokens):
        raise ValueError(f"tokens must be production buckets {PRODUCTION_TOKENS}")
    if any(value not in PRODUCTION_HIDDEN_SIZES for value in args.hidden_sizes):
        raise ValueError(
            f"hidden sizes must be production buckets {PRODUCTION_HIDDEN_SIZES}"
        )
    if any(value not in SUPPORTED_GROUPS_PER_BLOCK for value in args.groups_per_block):
        raise ValueError("groups per block must be in " f"{SUPPORTED_GROUPS_PER_BLOCK}")

    lease_device_fence = verify_lease_device_fence(
        args.expected_physical_device,
        args.expected_device_uuid,
    )
    torch = _load_runtime()
    if not torch.musa.is_available() or torch.musa.device_count() != 1:
        raise RuntimeError("FP8 quant benchmark requires exactly one visible GPU")
    device = torch.device("musa")
    properties = torch.musa.get_device_properties(0)
    torch.manual_seed(args.seed)
    flush = torch.empty(
        args.l2_flush_mb * 1024 * 1024, dtype=torch.uint8, device=device
    )
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for mode_name in args.modes:
        mode = MODES[mode_name]
        for hidden_size in args.hidden_sizes:
            groups_per_token = hidden_size // args.group_size
            for tokens in args.tokens:
                num_groups = tokens * groups_per_token
                baseline_groups = production_groups_per_block(num_groups)
                input_tensor = torch.randn(
                    (tokens, mode.input_multiplier * hidden_size),
                    dtype=torch.bfloat16,
                    device=device,
                )
                input_tensor.mul_(0.5)
                baseline_q, baseline_s = _allocate_outputs(
                    torch, tokens, hidden_size, device
                )
                candidate_q, candidate_s = _allocate_outputs(
                    torch, tokens, hidden_size, device
                )

                def launch(groups_per_block: int, output_q: Any, output_s: Any) -> None:
                    if mode.name == "quant":
                        torch.ops._C_musa_ops.per_token_group_quant_8bit_vec(
                            input_tensor,
                            output_q,
                            output_s,
                            args.group_size,
                            1e-10,
                            fp8_info.min,
                            fp8_info.max,
                            groups_per_block,
                        )
                    elif mode.clamp:
                        torch.ops._C_musa_ops.silu_and_mul_clamp_per_token_group_fp8_quant(
                            input_tensor,
                            output_q,
                            output_s,
                            args.group_size,
                            1e-10,
                            fp8_info.min,
                            fp8_info.max,
                            args.swiglu_limit,
                            groups_per_block,
                        )
                    else:
                        torch.ops._C_musa_ops.silu_and_mul_per_token_group_fp8_quant(
                            input_tensor,
                            output_q,
                            output_s,
                            args.group_size,
                            1e-10,
                            fp8_info.min,
                            fp8_info.max,
                            groups_per_block,
                        )

                launch(0, baseline_q, baseline_s)
                torch.musa.synchronize()
                reference_q = baseline_q.clone()
                reference_s = baseline_s.clone()

                for candidate_groups in args.groups_per_block:
                    if mode.name == "quant" and num_groups % candidate_groups:
                        skipped.append(
                            {
                                "mode": mode.name,
                                "tokens": tokens,
                                "hidden_size": hidden_size,
                                "candidate_groups_per_block": candidate_groups,
                                "reason": "num_groups_not_divisible",
                            }
                        )
                        continue

                    candidate_q.fill_(fp8_info.min)
                    candidate_s.fill_(float("nan"))
                    launch(candidate_groups, candidate_q, candidate_s)
                    torch.musa.synchronize()
                    q_equal = bool(
                        torch.equal(reference_q.float(), candidate_q.float())
                    )
                    scale_equal = bool(torch.equal(reference_s, candidate_s))
                    finite = bool(torch.isfinite(candidate_s).all().item())
                    correctness_pass = q_equal and scale_equal and finite

                    for dry_run in range(args.dry_runs):
                        if dry_run % 2:
                            launch(candidate_groups, candidate_q, candidate_s)
                        else:
                            launch(0, baseline_q, baseline_s)
                    torch.musa.synchronize()

                    baseline_samples: list[float] = []
                    candidate_samples: list[float] = []
                    paired_ratios: list[float] = []
                    for repeat in range(args.repeats):
                        order = (0, candidate_groups)
                        if repeat % 2:
                            order = (candidate_groups, 0)
                        measured: dict[int, float] = {}
                        for groups in order:
                            flush.zero_()
                            start = torch.musa.Event(enable_timing=True)
                            end = torch.musa.Event(enable_timing=True)
                            start.record()
                            if groups == 0:
                                launch(groups, baseline_q, baseline_s)
                            else:
                                launch(groups, candidate_q, candidate_s)
                            end.record()
                            end.synchronize()
                            measured[groups] = float(start.elapsed_time(end))
                        baseline_ms = measured[0]
                        candidate_ms = measured[candidate_groups]
                        baseline_samples.append(baseline_ms)
                        candidate_samples.append(candidate_ms)
                        paired_ratios.append(candidate_ms / baseline_ms)

                    baseline_median = statistics.median(baseline_samples)
                    candidate_median = statistics.median(candidate_samples)
                    rows.append(
                        {
                            "mode": mode.name,
                            "tokens": tokens,
                            "hidden_size": hidden_size,
                            "groups_per_token": groups_per_token,
                            "num_groups": num_groups,
                            "baseline_groups_per_block": baseline_groups,
                            "candidate_groups_per_block": candidate_groups,
                            "correctness_pass": correctness_pass,
                            "q_equal": q_equal,
                            "scale_equal": scale_equal,
                            "finite": finite,
                            "baseline_median_ms": baseline_median,
                            "candidate_median_ms": candidate_median,
                            "median_ratio": statistics.median(paired_ratios),
                            "ratio_p95": percentile(paired_ratios, 0.95),
                            "speedup_pct": (baseline_median / candidate_median - 1.0)
                            * 100.0,
                            "baseline_samples_ms": baseline_samples,
                            "candidate_samples_ms": candidate_samples,
                            "paired_ratios": paired_ratios,
                        }
                    )

    payload = {
        "schema": "vllm-musa-fp8-quant-groups.v1",
        "benchmark": "fp8-quant-groups",
        "device_name": torch.musa.get_device_name(0),
        "device_capability": list(torch.musa.get_device_capability(0)),
        "multiprocessor_count": int(properties.multi_processor_count),
        "lease_device_fence": lease_device_fence,
        "group_size": args.group_size,
        "timing": {
            "dry_runs": args.dry_runs,
            "repeats": args.repeats,
            "inner_iters": 1,
            "l2_flush_mb": args.l2_flush_mb,
            "paired_alternating_order": True,
        },
        "kernel_sources": [
            "csrc/musa/quantization/per_token_group_quant_8bit_vec.cu",
            "csrc/musa/quantization/silu_and_mul_per_token_group_fp8_quant.cu",
        ],
        "provenance": provenance(Path(__file__)),
        "rows": rows,
        "skipped": skipped,
    }
    emit_payload(payload, args.output)
    return 0 if all(row["correctness_pass"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
