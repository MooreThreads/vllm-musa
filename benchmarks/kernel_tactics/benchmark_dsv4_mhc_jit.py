#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold-cache microbenchmark for the DSV4 MHC pre-fuse TileLang JIT.

The benchmark deliberately times the already-compiled fuse kernel only.  JIT
compile time is reported separately, and the output records the production
resolver's configuration alongside every forced alternative.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from _benchmark_utils import (
    emit_payload,
    percentile,
    provenance,
    verify_lease_device_fence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument(
        "--production-path",
        choices=("standalone", "fused_post_prenorm"),
        default="standalone",
    )
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 8, 32, 64, 128, 256]
    )
    parser.add_argument(
        "--splits",
        type=int,
        nargs="*",
        default=[],
        help="extra diagnostic splits; the production split is always included",
    )
    parser.add_argument("--threads", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--hidden-blocks", type=int, nargs="+", default=[512, 1024])
    parser.add_argument(
        "--pass-configs",
        nargs="+",
        default=["safe", "burst", "aggressive_index32"],
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--l2-flush-mb", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--expected-physical-device", type=int, required=True)
    parser.add_argument("--expected-device-uuid", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_runtime() -> tuple[Any, Any, Any, Any]:
    # Import order is part of the MUSA compatibility contract.
    # isort: off
    import torchada  # noqa: F401
    import torch
    from vllm_musa.deepseek_v4_jit.tilelang_kernels import (
        mhc_pre_big_fuse_decode_split_kernel,
        mhc_pre_big_fuse_kernel,
    )
    from vllm_musa.deepseek_v4_mhc import (
        _get_mhc_pre_deepgemm_split_k,
        _resolve_mhc_pre_big_fuse_config,
    )
    # isort: on

    return (
        torch,
        mhc_pre_big_fuse_decode_split_kernel,
        mhc_pre_big_fuse_kernel,
        (_get_mhc_pre_deepgemm_split_k, _resolve_mhc_pre_big_fuse_config),
    )


def production_split_for_path(
    production_path: str,
    num_tokens: int,
    hidden_size: int,
    split_resolver: Any,
) -> int:
    if production_path == "fused_post_prenorm":
        if num_tokens > 16:
            raise ValueError("fused_post_prenorm supports M<=16")
        return 8 if num_tokens < 8 else 4
    if production_path == "standalone":
        return int(split_resolver(num_tokens, hidden_size * 4))
    raise ValueError(f"unsupported production path: {production_path}")


def _invoke(
    kernel: Any,
    gemm_out_mul: Any,
    gemm_out_sqrsum: Any,
    hc_scale: Any,
    hc_base: Any,
    residual: Any,
    post_mix: Any,
    comb_mix: Any,
    layer_input: Any,
) -> None:
    kernel(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
    )


@dataclass
class _CompiledConfig:
    config: tuple[int, int, str]
    kernel: Any
    post_mix: Any
    comb_mix: Any
    layer_input: Any
    compile_seconds: float
    candidate_samples: list[float] = field(default_factory=list)
    baseline_samples: list[float] = field(default_factory=list)
    max_abs_diff_vs_production: float | None = None
    output_absmax: float = 0.0


@dataclass(frozen=True)
class _InvocationArgs:
    d_part: Any
    s_part: Any
    hc_scale: Any
    hc_base: Any
    residual: Any


def _timed_invoke(torch: Any, flush: Any, run: _CompiledConfig, args: Any) -> float:
    flush.zero_()
    torch.musa.synchronize()
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    _invoke(
        run.kernel,
        args.d_part,
        args.s_part,
        args.hc_scale,
        args.hc_base,
        args.residual,
        run.post_mix,
        run.comb_mix,
        run.layer_input,
    )
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def main() -> int:
    args = parse_args()
    if min(args.tokens) <= 0 or (args.splits and min(args.splits) <= 0):
        raise ValueError("tokens and splits must be positive")
    if min(args.warmup, args.repeats, args.l2_flush_mb) <= 0:
        raise ValueError("timing arguments must be positive")
    lease_device_fence = verify_lease_device_fence(
        args.expected_physical_device,
        args.expected_device_uuid,
    )
    torch, decode_factory, big_factory, resolver = _load_runtime()
    if not torch.musa.is_available() or torch.musa.device_count() != 1:
        raise RuntimeError("MHC benchmark requires one visible MUSA device")
    device = torch.device("musa")
    properties = torch.musa.get_device_properties(0)
    mp = int(properties.multi_processor_count)
    hidden_size = int(args.hidden_size)
    hc_mult3 = 24
    flush = torch.empty(
        args.l2_flush_mb * 1024 * 1024 // 4,
        dtype=torch.int32,
        device=device,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    split_resolver, config_resolver = resolver

    for num_tokens in args.tokens:
        if hidden_size != 4096:
            skipped.append(
                {"tokens": num_tokens, "reason": "current MHC JIT contract is H4096"}
            )
            continue
        try:
            production_split = production_split_for_path(
                args.production_path,
                num_tokens,
                hidden_size,
                split_resolver,
            )
            config_resolver(num_tokens, production_split)
        except Exception as exc:  # pragma: no cover - device-side contract
            skipped.append({"tokens": num_tokens, "reason": str(exc)})
            continue

        split_values = sorted(set(args.splits) | {production_split})
        # The production provider passes residual_flat [M, 4, 4096] and a
        # [split_k, M, 24] partial.  Keep the same layout for direct fuse A/B.
        residual = torch.randn(
            (num_tokens, 4, hidden_size), generator=generator, dtype=torch.bfloat16
        ).to(device)
        fn_mul = torch.randn(
            (max(split_values), num_tokens, hc_mult3),
            generator=generator,
            dtype=torch.float32,
        ).to(device)
        sqrsum = torch.rand(
            (max(split_values), num_tokens),
            generator=generator,
            dtype=torch.float32,
        ).to(device)
        hc_scale = torch.randn((3,), generator=generator, dtype=torch.float32).to(
            device
        )
        hc_base = torch.randn((hc_mult3,), generator=generator, dtype=torch.float32).to(
            device
        )
        for split_k in split_values:
            if hidden_size * 4 % split_k or (hidden_size * 4 // split_k) % 128:
                skipped.append(
                    {
                        "tokens": num_tokens,
                        "production_path": args.production_path,
                        "split_k": split_k,
                        "reason": "invalid split size",
                    }
                )
                continue
            # Slice the common allocation so each config receives its own
            # correctly shaped partial tensor.
            d_part = fn_mul[:split_k]
            s_part = sqrsum[:split_k]
            split_production_config = config_resolver(num_tokens, split_k)
            requested_configs = {
                (threads, hidden_block, pass_config)
                for threads in args.threads
                for hidden_block in args.hidden_blocks
                if hidden_size * 4 % hidden_block == 0
                for pass_config in args.pass_configs
            }
            # Always include the production resolver arm even when a quick
            # campaign intentionally requested a reduced candidate set.
            requested_configs.add(tuple(split_production_config))
            configs = list(requested_configs)
            configs.sort(key=lambda item: item != tuple(split_production_config))
            reference_config = tuple(split_production_config)
            compiled: list[_CompiledConfig] = []
            for threads, hidden_block, pass_config in configs:
                is_decode = num_tokens <= 64
                factory = decode_factory if is_decode else big_factory
                config = (threads, hidden_block, pass_config)
                compile_started = time.perf_counter()
                try:
                    kernel = factory(
                        hidden_size,
                        1e-6,
                        1e-6,
                        1e-6,
                        1.0,
                        20,
                        n_splits=split_k,
                        hc_mult=4,
                        threads=threads,
                        hidden_block=hidden_block,
                        pass_config=pass_config,
                    )
                    post_mix = torch.empty(
                        (num_tokens, 4), dtype=torch.float32, device=device
                    )
                    comb_mix = torch.empty(
                        (num_tokens, 16), dtype=torch.float32, device=device
                    )
                    layer_input = torch.empty(
                        (num_tokens, hidden_size),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    _invoke(
                        kernel,
                        d_part,
                        s_part,
                        hc_scale,
                        hc_base,
                        residual,
                        post_mix,
                        comb_mix,
                        layer_input,
                    )
                    torch.musa.synchronize()
                except Exception as exc:  # pragma: no cover - device-side contract
                    skipped.append(
                        {
                            "tokens": num_tokens,
                            "split_k": split_k,
                            "config": config,
                            "reason": str(exc),
                        }
                    )
                    continue
                compiled.append(
                    _CompiledConfig(
                        config=config,
                        kernel=kernel,
                        post_mix=post_mix,
                        comb_mix=comb_mix,
                        layer_input=layer_input,
                        compile_seconds=time.perf_counter() - compile_started,
                    )
                )

            production = next(
                (run for run in compiled if run.config == reference_config), None
            )
            if production is None:
                for run in compiled:
                    skipped.append(
                        {
                            "tokens": num_tokens,
                            "split_k": split_k,
                            "config": run.config,
                            "reason": "production resolver config failed to compile",
                        }
                    )
                continue

            for run in compiled:
                for _ in range(args.warmup):
                    _invoke(
                        run.kernel,
                        d_part,
                        s_part,
                        hc_scale,
                        hc_base,
                        residual,
                        run.post_mix,
                        run.comb_mix,
                        run.layer_input,
                    )
            torch.musa.synchronize()

            invocation_args = _InvocationArgs(
                d_part=d_part,
                s_part=s_part,
                hc_scale=hc_scale,
                hc_base=hc_base,
                residual=residual,
            )
            candidates = [run for run in compiled if run is not production]
            for repeat in range(args.repeats):
                for candidate in candidates:
                    pair = (
                        (production, candidate)
                        if repeat % 2 == 0
                        else (candidate, production)
                    )
                    first_ms = _timed_invoke(torch, flush, pair[0], invocation_args)
                    second_ms = _timed_invoke(torch, flush, pair[1], invocation_args)
                    if repeat % 2 == 0:
                        candidate.baseline_samples.append(first_ms)
                        candidate.candidate_samples.append(second_ms)
                    else:
                        candidate.candidate_samples.append(first_ms)
                        candidate.baseline_samples.append(second_ms)

            torch.musa.synchronize()
            _invoke(
                production.kernel,
                d_part,
                s_part,
                hc_scale,
                hc_base,
                residual,
                production.post_mix,
                production.comb_mix,
                production.layer_input,
            )
            torch.musa.synchronize()
            reference_outputs = (
                production.layer_input.detach().cpu().clone(),
                production.post_mix.detach().cpu().clone(),
                production.comb_mix.detach().cpu().clone(),
            )
            for run in compiled:
                if run is not production:
                    _invoke(
                        run.kernel,
                        d_part,
                        s_part,
                        hc_scale,
                        hc_base,
                        residual,
                        run.post_mix,
                        run.comb_mix,
                        run.layer_input,
                    )
                    torch.musa.synchronize()
                output = run.layer_input.float()
                output_absmax = float(output.abs().max().item())
                max_abs_diff = max(
                    float(
                        (
                            run.layer_input.detach().cpu().float()
                            - reference_outputs[0].float()
                        )
                        .abs()
                        .max()
                        .item()
                    ),
                    float(
                        (
                            run.post_mix.detach().cpu().float()
                            - reference_outputs[1].float()
                        )
                        .abs()
                        .max()
                        .item()
                    ),
                    float(
                        (
                            run.comb_mix.detach().cpu().float()
                            - reference_outputs[2].float()
                        )
                        .abs()
                        .max()
                        .item()
                    ),
                )
                if run is production:
                    run.baseline_samples = [
                        _timed_invoke(torch, flush, production, invocation_args)
                        for _ in range(args.repeats)
                    ]
                    run.candidate_samples = list(run.baseline_samples)
                rows.append(
                    {
                        "tokens": num_tokens,
                        "production_path": args.production_path,
                        "split_k": split_k,
                        "threads": run.config[0],
                        "hidden_block": run.config[1],
                        "pass_config": run.config[2],
                        "production_split_k": production_split,
                        "is_production_split": split_k == production_split,
                        "production_config": list(split_production_config),
                        "is_production_config": run is production,
                        "median_ms": float(statistics.median(run.candidate_samples)),
                        "p90_ms": percentile(run.candidate_samples, 0.90),
                        "p99_ms": percentile(run.candidate_samples, 0.99),
                        "baseline_median_ms": float(
                            statistics.median(run.baseline_samples)
                        ),
                        "candidate_samples_ms": list(run.candidate_samples),
                        "baseline_samples_ms": list(run.baseline_samples),
                        "compile_seconds": run.compile_seconds,
                        "output_absmax": output_absmax,
                        "max_abs_diff_vs_production": None
                        if run is production
                        else max_abs_diff,
                        "correctness_pass": bool(
                            torch.isfinite(output).all().item()
                            and output_absmax > 1e-12
                            and (run is production or max_abs_diff <= 5e-2)
                        ),
                        "cache_policy": "cold-l2-per-sample",
                    }
                )

    # Paired samples are already aligned above; compute the ratio distribution
    # per row after all launches for a key have completed.
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["tokens"], row["split_k"]), []).append(row)
    for group in grouped.values():
        reference = next((row for row in group if row["is_production_config"]), None)
        if reference is None:
            for row in group:
                row["correctness_pass"] = False
                row["median_ratio"] = None
                row["speedup_pct"] = None
            continue
        for row in group:
            if row["is_production_config"]:
                ratios = [1.0] * args.repeats
            else:
                # Candidate/base medians are paired at launch time; the raw
                # pair list is retained in the row for audit and reduction.
                candidate_ms = row.get("candidate_samples_ms")
                baseline_samples = row.get("baseline_samples_ms")
                ratios = (
                    [c / b for c, b in zip(candidate_ms, baseline_samples)]
                    if candidate_ms and baseline_samples
                    else [row["median_ms"] / row["baseline_median_ms"]]
                )
            row["median_ratio"] = statistics.median(ratios)
            row["ratio_p95"] = percentile(ratios, 0.95)
            row["ratio_p99"] = percentile(ratios, 0.99)
            row["speedup_pct"] = (1.0 / row["median_ratio"] - 1.0) * 100.0

    payload = {
        "schema": "musa-dsv4-mhc-jit-aot-paired-ab.v1",
        "device_name": torch.musa.get_device_name(0),
        "multiprocessor_count": mp,
        "lease_device_fence": lease_device_fence,
        "hidden_size": hidden_size,
        "production_path": args.production_path,
        "benchmark": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "l2_flush_mb": args.l2_flush_mb,
            "flush_before_every_timed_launch": True,
            "cache_policy": "cold-l2-per-sample",
            "production_resolver": "vllm_musa.deepseek_v4_mhc._resolve_mhc_pre_big_fuse_config",
        },
        "kernel_sources": [
            "vllm_musa/deepseek_v4_jit/tilelang_kernels.py::mhc_pre_big_fuse_decode_split_kernel",
            "vllm_musa/deepseek_v4_jit/tilelang_kernels.py::mhc_pre_big_fuse_kernel",
        ],
        "provenance": provenance(Path(__file__)),
        "rows": rows,
        "skipped": skipped,
    }
    emit_payload(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
