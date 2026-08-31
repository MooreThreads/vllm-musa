#!/usr/bin/env python3
"""Cold-L2 paired A/B for native Inductor versus an exploratory RMSNorm HOP."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any, Callable

from _benchmark_utils import (
    emit_payload,
    percentile,
    provenance,
    verify_lease_device_fence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--num-warps", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--chain", choices=("op", "scale"), default="op")
    parser.add_argument("--dry-runs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--l2-flush-mb", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--expected-physical-device", type=int)
    parser.add_argument("--expected-device-uuid")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_runtime() -> tuple[Any, Any, Any, Any]:
    # isort: off
    import torchada  # noqa: F401
    import torch
    import torch.nn.functional as functional
    # isort: on

    from _rmsnorm_inductor_candidate import rms_norm_triton

    return torch, functional, rms_norm_triton, torch._inductor.config


def _compile_functions(
    torch: Any,
    functional: Any,
    rms_norm_triton: Any,
    *,
    hidden_size: int,
    chain: str,
    candidates: list[int],
) -> tuple[Callable[..., Any], dict[int, Callable[..., Any]]]:
    def finish(output: Any, post_scale: Any) -> Any:
        return output if chain == "op" else output * post_scale

    def native(input_tensor: Any, weight: Any, post_scale: Any) -> Any:
        output = functional.rms_norm(
            input_tensor,
            (hidden_size,),
            weight,
            eps=1e-6,
        )
        return finish(output, post_scale)

    native_compiled = torch.compile(native, fullgraph=True, dynamic=False)
    candidate_compiled: dict[int, Callable[..., Any]] = {}
    for num_warps in candidates:

        def candidate(
            input_tensor: Any,
            weight: Any,
            post_scale: Any,
            *,
            _num_warps: int = num_warps,
        ) -> Any:
            output = rms_norm_triton(
                input_tensor,
                weight,
                1e-6,
                num_warps=_num_warps,
            )
            return finish(output, post_scale)

        candidate_compiled[num_warps] = torch.compile(
            candidate,
            fullgraph=True,
            dynamic=False,
        )
    return native_compiled, candidate_compiled


def main() -> int:
    args = parse_args()
    if args.rows <= 0 or args.hidden_size <= 0:
        raise ValueError("rows and hidden-size must be positive")
    if args.repeats <= 0 or args.dry_runs < 0:
        raise ValueError("repeats must be positive and dry-runs non-negative")
    if any(item not in (1, 2, 4, 8) for item in args.num_warps):
        raise ValueError("num-warps candidates must be chosen from 1, 2, 4, 8")

    lease_device_fence = verify_lease_device_fence(
        expected_physical_device=args.expected_physical_device,
        expected_device_uuid=args.expected_device_uuid,
    )
    torch, functional, rms_norm_triton, inductor_config = _load_runtime()
    torch.manual_seed(args.seed)
    torch.musa.set_device(0)
    inductor_config.force_disable_caches = True

    from vllm_musa.tuning import prime_musa_kernel_hardware

    hardware = prime_musa_kernel_hardware(0)
    properties = torch.musa.get_device_properties(0)
    input_tensor = torch.randn(
        (args.rows, args.hidden_size),
        device="musa",
        dtype=torch.bfloat16,
    )
    weight = torch.randn(
        (args.hidden_size,),
        device="musa",
        dtype=torch.bfloat16,
    )
    post_scale = torch.randn_like(input_tensor)
    flush_elements = max(1, args.l2_flush_mb * 1024 * 1024 // 4)
    flush = torch.empty(flush_elements, device="musa", dtype=torch.float32)

    native, candidates = _compile_functions(
        torch,
        functional,
        rms_norm_triton,
        hidden_size=args.hidden_size,
        chain=args.chain,
        candidates=args.num_warps,
    )
    expected = native(input_tensor, weight, post_scale)
    torch.musa.synchronize()

    rows_out = []
    for num_warps, candidate in candidates.items():
        actual = candidate(input_tensor, weight, post_scale)
        torch.musa.synchronize()
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=2e-2, atol=2e-2
        )
        correctness = bool(torch.isfinite(actual).all().item())

        for dry_run in range(args.dry_runs):
            (native if dry_run % 2 == 0 else candidate)(
                input_tensor, weight, post_scale
            )
        torch.musa.synchronize()

        baseline_samples: list[float] = []
        candidate_samples: list[float] = []
        for repeat in range(args.repeats):
            order = (
                (("baseline", native), ("candidate", candidate))
                if repeat % 2 == 0
                else (("candidate", candidate), ("baseline", native))
            )
            measured: dict[str, float] = {}
            for name, launch in order:
                flush.zero_()
                start = torch.musa.Event(enable_timing=True)
                end = torch.musa.Event(enable_timing=True)
                start.record()
                launch(input_tensor, weight, post_scale)
                end.record()
                end.synchronize()
                measured[name] = float(start.elapsed_time(end))
            baseline_samples.append(measured["baseline"])
            candidate_samples.append(measured["candidate"])

        ratios = [
            candidate_ms / baseline_ms
            for baseline_ms, candidate_ms in zip(
                baseline_samples, candidate_samples, strict=True
            )
        ]
        baseline_median = statistics.median(baseline_samples)
        candidate_median = statistics.median(candidate_samples)
        rows_out.append(
            {
                "rows": args.rows,
                "hidden_size": args.hidden_size,
                "chain": args.chain,
                "candidate_num_warps": num_warps,
                "baseline": "torch.compile(F.rms_norm)",
                "candidate": "torch.compile(benchmark-local wrap_triton RMSNorm)",
                "baseline_samples_ms": baseline_samples,
                "candidate_samples_ms": candidate_samples,
                "baseline_median_ms": baseline_median,
                "candidate_median_ms": candidate_median,
                "median_ratio": statistics.median(ratios),
                "ratio_p95": percentile(ratios, 0.95),
                "ratio_p99": percentile(ratios, 0.99),
                "speedup_pct": (baseline_median / candidate_median - 1.0) * 100.0,
                "correctness_pass": correctness,
                "cache_policy": "cold-l2-per-sample",
            }
        )

    emit_payload(
        {
            "schema": "musa-rmsnorm-inductor-paired-ab.v2",
            "candidate_kind": "exploratory-rejected",
            "device_name": torch.musa.get_device_name(0),
            "device_capability": [int(properties.major), int(properties.minor)],
            "multiprocessor_count": int(properties.multi_processor_count),
            "primed_hardware": {
                "device_capability": list(hardware.device_capability),
                "multiprocessor_count": hardware.multiprocessor_count,
            },
            "lease_device_fence": lease_device_fence,
            "benchmark": {
                "dry_runs": args.dry_runs,
                "repeats": args.repeats,
                "l2_flush_mb": args.l2_flush_mb,
                "l2_flush_bytes": flush.numel() * flush.element_size(),
                "flush_before_every_timed_launch": True,
                "paired_alternating_order": True,
                "inductor_cache_disabled": True,
            },
            "kernel_source": (
                "benchmarks/kernel_tactics/_rmsnorm_inductor_candidate.py"
            ),
            "provenance": provenance(Path(__file__)),
            "rows": rows_out,
        },
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
