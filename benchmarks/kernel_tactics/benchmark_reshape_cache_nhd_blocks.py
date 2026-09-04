#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired cold-L2 BLOCK_X sweep for the native NHD cache-write kernel."""

from __future__ import annotations

import argparse
import os
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

PRODUCTION_DEFAULT_BLOCK_X = 512
SUPPORTED_BLOCK_X = (128, 256, 512, 1024)
CACHE_BLOCK_SIZE = 64
PRODUCTION_TOKEN_BUCKETS = (
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    16,
    32,
    48,
    64,
    128,
    192,
    256,
    2048,
    4096,
)
TOKEN_BUCKET_SOURCES = (
    "benchmarks/kernel_tactics/qwen_dsv4_rmsnorm_shapes.json::row_shapes",
    "git:1fe15be47::benchmarks/op_perf/bench_reshape_and_cache_flash.py",
)
PRODUCTION_CONSUMERS = (
    "vllm_musa/v1/attention/backends/fa_utils.py::reshape_and_cache_flash",
    "vllm_musa/v1/attention/backends/flash_attn.py::do_kv_cache_update",
    "vllm_musa/v1/attention/backends/tree_attn.py::do_kv_cache_update",
)


@dataclass(frozen=True)
class CacheShape:
    num_kv_heads: int
    head_size: int
    source: str

    @property
    def vecs_per_token(self) -> int:
        return self.num_kv_heads * self.head_size // 8


# These are current production envelopes from the Qwen2/Qwen3 FlashAttention
# consumers. Keep the cache dimensions separate from query-head count: this op
# receives K/V after they have been viewed as [tokens, num_kv_heads, head_size].
PRODUCTION_SHAPES = {
    "qwen2-kv2-d64": CacheShape(
        2,
        64,
        "vllm_musa/v1/attention/backends/flash_attn.py::"
        "fused_rope_kvcache_supported",
    ),
    "qwen3-kv8-d128": CacheShape(
        8,
        128,
        "vllm_musa/v1/attention/backends/flash_attn.py::"
        "qwen3_qk_rope_kvcache_supported",
    ),
    "qwen3-kv1-d256": CacheShape(
        1,
        256,
        "vllm_musa/v1/attention/backends/flash_attn.py::"
        "qwen3_qk_rope_kvcache_supported",
    ),
    "qwen3-kv2-d256": CacheShape(
        2,
        256,
        "vllm_musa/v1/attention/backends/flash_attn.py::"
        "qwen3_qk_rope_kvcache_supported",
    ),
}


def parse_block_x(value: str) -> int:
    try:
        block_x = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("BLOCK_X must be an integer") from exc
    if block_x not in SUPPORTED_BLOCK_X:
        raise argparse.ArgumentTypeError(
            f"BLOCK_X must be one of {SUPPORTED_BLOCK_X}, got {block_x}"
        )
    return block_x


def parse_production_tokens(value: str) -> int:
    try:
        tokens = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tokens must be an integer") from exc
    if tokens not in PRODUCTION_TOKEN_BUCKETS:
        raise argparse.ArgumentTypeError(
            "tokens must be a checked production bucket in "
            f"{PRODUCTION_TOKEN_BUCKETS}, got {tokens}"
        )
    return tokens


def production_tokens_per_block(shape: CacheShape) -> int:
    if shape.vecs_per_token <= 64:
        return 8
    if shape.vecs_per_token <= 128:
        return 4
    if shape.vecs_per_token <= 256:
        return 2
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        choices=tuple(PRODUCTION_SHAPES),
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--tokens",
        type=parse_production_tokens,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--blocks",
        type=parse_block_x,
        nargs="+",
        default=list(SUPPORTED_BLOCK_X),
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--dry-runs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--l2-flush-mb", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--expected-physical-device", type=int, required=True)
    parser.add_argument("--expected-device-uuid", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_runtime() -> tuple[Any, Any]:
    # Import order is part of the MUSA compatibility contract.
    # isort: off
    import torchada  # noqa: F401
    import torch
    from vllm_musa import _custom_ops as musa_ops
    # isort: on

    return torch, musa_ops


def _launch(
    musa_ops: Any,
    key: Any,
    value: Any,
    key_cache: Any,
    value_cache: Any,
    slot_mapping: Any,
    block_x: int,
) -> None:
    musa_ops.musa_reshape_and_cache_flash_nhd(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        block_x=block_x,
    )


def _reference_cache_write(
    torch: Any,
    key: Any,
    value: Any,
    key_cache: Any,
    value_cache: Any,
    slot_mapping: Any,
) -> None:
    valid = slot_mapping >= 0
    valid_slots = slot_mapping[valid]
    block_indices = torch.div(valid_slots, CACHE_BLOCK_SIZE, rounding_mode="floor")
    block_offsets = torch.remainder(valid_slots, CACHE_BLOCK_SIZE)
    key_cache[block_indices, block_offsets] = key[valid]
    value_cache[block_indices, block_offsets] = value[valid]


def _check_correctness(
    torch: Any,
    musa_ops: Any,
    key: Any,
    value: Any,
    initial_key_cache: Any,
    initial_value_cache: Any,
    slot_mapping: Any,
    block_x: int,
) -> dict[str, bool]:
    correctness_slots = slot_mapping.clone()
    if correctness_slots.numel() > 1:
        correctness_slots[-1] = -1
    expected_key = initial_key_cache.clone()
    expected_value = initial_value_cache.clone()
    _reference_cache_write(
        torch,
        key,
        value,
        expected_key,
        expected_value,
        correctness_slots,
    )

    actual_key = initial_key_cache.clone()
    actual_value = initial_value_cache.clone()
    _launch(
        musa_ops,
        key,
        value,
        actual_key,
        actual_value,
        correctness_slots,
        block_x,
    )
    torch.musa.synchronize()
    key_equal = bool(torch.equal(expected_key, actual_key))
    value_equal = bool(torch.equal(expected_value, actual_value))
    finite = bool(
        torch.isfinite(actual_key).all().item()
        and torch.isfinite(actual_value).all().item()
    )
    changed = bool(
        not torch.equal(initial_key_cache, actual_key)
        and not torch.equal(initial_value_cache, actual_value)
    )
    negative_key = initial_key_cache.clone()
    negative_value = initial_value_cache.clone()
    negative_slots = torch.full_like(slot_mapping, -1)
    _launch(
        musa_ops,
        key,
        value,
        negative_key,
        negative_value,
        negative_slots,
        block_x,
    )
    torch.musa.synchronize()
    negative_slot_unchanged = bool(
        torch.equal(initial_key_cache, negative_key)
        and torch.equal(initial_value_cache, negative_value)
    )
    return {
        "passed": (
            key_equal and value_equal and finite and changed and negative_slot_unchanged
        ),
        "key_bit_equal": key_equal,
        "value_bit_equal": value_equal,
        "finite": finite,
        "changed_from_poison": changed,
        "negative_slot_unchanged": negative_slot_unchanged,
    }


def _measure_ms(
    torch: Any,
    flush: Any,
    launch: Any,
) -> float:
    flush.zero_()
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    launch()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _benchmark_pair(
    torch: Any,
    flush: Any,
    launch_baseline: Any,
    launch_candidate: Any,
    dry_runs: int,
    repeats: int,
) -> dict[str, Any]:
    for dry_run in range(dry_runs):
        (launch_candidate if dry_run % 2 else launch_baseline)()
    torch.musa.synchronize()

    baseline_samples: list[float] = []
    candidate_samples: list[float] = []
    paired_ratios: list[float] = []
    for repeat in range(repeats):
        order = ("baseline", "candidate")
        if repeat % 2:
            order = tuple(reversed(order))
        measured: dict[str, float] = {}
        for variant in order:
            launch = launch_baseline if variant == "baseline" else launch_candidate
            measured[variant] = _measure_ms(torch, flush, launch)
        baseline_ms = measured["baseline"]
        candidate_ms = measured["candidate"]
        if baseline_ms <= 0 or candidate_ms <= 0:
            raise RuntimeError(
                "MUSA event timing must be positive, got "
                f"baseline={baseline_ms}, candidate={candidate_ms}"
            )
        baseline_samples.append(baseline_ms)
        candidate_samples.append(candidate_ms)
        paired_ratios.append(candidate_ms / baseline_ms)

    return {
        "baseline_samples_ms": baseline_samples,
        "candidate_samples_ms": candidate_samples,
        "paired_ratios": paired_ratios,
        "baseline_median_ms": statistics.median(baseline_samples),
        "candidate_median_ms": statistics.median(candidate_samples),
        "median_ratio": statistics.median(paired_ratios),
        "ratio_p95": percentile(paired_ratios, 0.95),
    }


def main() -> int:
    args = parse_args()
    if min(args.dry_runs, args.repeats, args.l2_flush_mb) <= 0:
        raise ValueError("timing arguments must be positive")
    if len(set(args.shapes)) != len(args.shapes):
        raise ValueError("shapes must not contain duplicates")
    if len(set(args.tokens)) != len(args.tokens):
        raise ValueError("tokens must not contain duplicates")
    if len(set(args.blocks)) != len(args.blocks):
        raise ValueError("blocks must not contain duplicates")
    tokens_per_block_override = os.environ.get(
        "VLLM_MUSA_RESHAPE_CACHE_TOKENS_PER_BLOCK"
    )
    if tokens_per_block_override is not None:
        raise RuntimeError(
            "unset VLLM_MUSA_RESHAPE_CACHE_TOKENS_PER_BLOCK for an unbiased "
            "production-selector comparison"
        )

    lease_device_fence = verify_lease_device_fence(
        args.expected_physical_device,
        args.expected_device_uuid,
    )
    torch, musa_ops = _load_runtime()
    if not torch.musa.is_available() or torch.musa.device_count() != 1:
        raise RuntimeError("NHD cache benchmark requires exactly one visible GPU")

    device = torch.device("musa")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    properties = torch.musa.get_device_properties(0)
    flush = torch.empty(
        args.l2_flush_mb * 1024 * 1024,
        dtype=torch.uint8,
        device=device,
    )
    rows: list[dict[str, Any]] = []

    for shape_index, shape_name in enumerate(args.shapes):
        shape = PRODUCTION_SHAPES[shape_name]
        for tokens in args.tokens:
            torch.manual_seed(args.seed + shape_index * 10000 + tokens)
            key = torch.randn(
                (tokens, shape.num_kv_heads, shape.head_size),
                dtype=dtype,
                device=device,
            )
            value = torch.randn_like(key)
            slot_offset = 17
            num_blocks = (
                slot_offset + tokens + CACHE_BLOCK_SIZE - 1
            ) // CACHE_BLOCK_SIZE
            initial_key_cache = torch.randn(
                (num_blocks, CACHE_BLOCK_SIZE, shape.num_kv_heads, shape.head_size),
                dtype=dtype,
                device=device,
            )
            initial_value_cache = torch.randn_like(initial_key_cache)
            slot_mapping = (
                torch.arange(tokens, dtype=torch.int64, device=device) + slot_offset
            )

            baseline_key_cache = initial_key_cache.clone()
            baseline_value_cache = initial_value_cache.clone()
            baseline_correctness = _check_correctness(
                torch,
                musa_ops,
                key,
                value,
                initial_key_cache,
                initial_value_cache,
                slot_mapping,
                0,
            )

            for candidate_block_x in args.blocks:
                candidate_key_cache = initial_key_cache.clone()
                candidate_value_cache = initial_value_cache.clone()
                candidate_correctness = _check_correctness(
                    torch,
                    musa_ops,
                    key,
                    value,
                    initial_key_cache,
                    initial_value_cache,
                    slot_mapping,
                    candidate_block_x,
                )

                def launch_baseline() -> None:
                    _launch(
                        musa_ops,
                        key,
                        value,
                        baseline_key_cache,
                        baseline_value_cache,
                        slot_mapping,
                        0,
                    )

                def launch_candidate() -> None:
                    _launch(
                        musa_ops,
                        key,
                        value,
                        candidate_key_cache,
                        candidate_value_cache,
                        slot_mapping,
                        candidate_block_x,
                    )

                timing = _benchmark_pair(
                    torch,
                    flush,
                    launch_baseline,
                    launch_candidate,
                    args.dry_runs,
                    args.repeats,
                )
                io_bytes = 4 * tokens * shape.num_kv_heads * shape.head_size * 2
                baseline_seconds = timing["baseline_median_ms"] / 1000.0
                candidate_seconds = timing["candidate_median_ms"] / 1000.0
                rows.append(
                    {
                        "shape": shape_name,
                        "shape_source": shape.source,
                        "tokens": tokens,
                        "num_kv_heads": shape.num_kv_heads,
                        "head_size": shape.head_size,
                        "cache_block_size": CACHE_BLOCK_SIZE,
                        "vecs_per_token": shape.vecs_per_token,
                        "tokens_per_block": production_tokens_per_block(shape),
                        "baseline_block_x": PRODUCTION_DEFAULT_BLOCK_X,
                        "candidate_block_x": candidate_block_x,
                        "baseline_correctness": baseline_correctness,
                        "candidate_correctness": candidate_correctness,
                        "io_bytes": io_bytes,
                        "baseline_gbps": io_bytes / baseline_seconds / 1e9,
                        "candidate_gbps": io_bytes / candidate_seconds / 1e9,
                        "speedup_pct": (
                            timing["baseline_median_ms"] / timing["candidate_median_ms"]
                            - 1.0
                        )
                        * 100.0,
                        **timing,
                    }
                )

    payload = {
        "schema": "vllm-musa-reshape-cache-nhd-block-x.v1",
        "benchmark": "reshape-cache-nhd-block-x",
        "kernel_name": "reshape_and_cache_flash_nhd_kernel",
        "kernel_source": "csrc/musa/cache_kernels.mu",
        "production_consumers": list(PRODUCTION_CONSUMERS),
        "consumer_scope": (
            "generic NHD fallback and prefill cache writes; successful fused "
            "Qwen2/Qwen3 cache-out paths can bypass this standalone op"
        ),
        "production_default_block_x": PRODUCTION_DEFAULT_BLOCK_X,
        "supported_candidate_block_x": list(SUPPORTED_BLOCK_X),
        "token_bucket_sources": list(TOKEN_BUCKET_SOURCES),
        "device_name": torch.musa.get_device_name(0),
        "device_capability": list(torch.musa.get_device_capability(0)),
        "multiprocessor_count": int(properties.multi_processor_count),
        "lease_device_fence": lease_device_fence,
        "timing": {
            "dry_runs": args.dry_runs,
            "repeats": args.repeats,
            "inner_iters": 1,
            "l2_flush_mb": args.l2_flush_mb,
            "paired_alternating_order": True,
        },
        "provenance": provenance(Path(__file__)),
        "rows": rows,
    }
    emit_payload(payload, args.output)
    return (
        0
        if all(
            row["baseline_correctness"]["passed"]
            and row["candidate_correctness"]["passed"]
            for row in rows
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
