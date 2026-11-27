#!/usr/bin/env python3
# ruff: noqa: I001

"""Cold-cache A/B/A gate for top-k=50 probability renormalization."""

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from collections.abc import Callable

import torchada  # noqa: F401
import torch

# Importing the Python custom-op shim loads vllm_musa._C and registers both the
# pinned native op and the candidate op. The direct calls below intentionally
# bypass its dispatch wrapper so A/B timing compares the two registered kernels.
import vllm_musa._custom_ops  # noqa: F401,E402


TOP_K = 50


def _native(probs: torch.Tensor, output: torch.Tensor) -> None:
    torch.ops._C_musa_ops.top_k_renorm_probs.default(probs, output, None, TOP_K)


def _candidate(probs: torch.Tensor, output: torch.Tensor) -> None:
    torch.ops._C_musa_ops.musa_rubymine_top_k_renorm_probs.default(probs, output, TOP_K)


def _flush_cache(flush: torch.Tensor) -> None:
    flush.add_(1.0)
    torch.musa.synchronize()


def _event_us(call: Callable[[], None]) -> float:
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    call()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000.0)


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def _paired_bootstrap_ci(
    a1: list[float], b: list[float], a2: list[float], samples: int = 2000
) -> tuple[float, float]:
    rng = random.Random(765)
    paired = [((x + z) / 2.0, y) for x, y, z in zip(a1, b, a2, strict=True)]
    speedups: list[float] = []
    for _ in range(samples):
        draw = [paired[rng.randrange(len(paired))] for _ in paired]
        native = statistics.median(item[0] for item in draw)
        candidate = statistics.median(item[1] for item in draw)
        speedups.append(100.0 * (native - candidate) / native)
    return _quantile(speedups, 0.025), _quantile(speedups, 0.975)


def _capture_replay_us(
    call: Callable[[], None], warmups: int, samples: int
) -> dict[str, float]:
    warmup_stream = torch.musa.Stream()
    warmup_stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(warmup_stream):
        for _ in range(warmups):
            call()
    torch.musa.current_stream().wait_stream(warmup_stream)
    torch.musa.synchronize()

    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        call()
    values = [_event_us(graph.replay) for _ in range(samples)]
    return {
        "median_us": statistics.median(values),
        "p20_us": _quantile(values, 0.20),
        "p80_us": _quantile(values, 0.80),
    }


def _bench_shape(
    rows: int,
    vocab: int,
    warmups: int,
    samples: int,
    graph_samples: int,
    flush: torch.Tensor,
) -> dict[str, object]:
    generator = torch.Generator(device="musa")
    generator.manual_seed(765000 + rows + vocab)
    logits = torch.randn(
        (rows, vocab), generator=generator, device="musa", dtype=torch.float32
    )
    probs = torch.softmax(logits, dim=-1)
    native_output = torch.empty_like(probs)
    candidate_output = torch.empty_like(probs)

    def native_call() -> None:
        _native(probs, native_output)

    def candidate_call() -> None:
        _candidate(probs, candidate_output)

    for _ in range(warmups):
        native_call()
        candidate_call()
    torch.musa.synchronize()
    native_call()
    candidate_call()
    torch.musa.synchronize()
    max_abs_error = float((candidate_output - native_output).abs().max().item())
    support_equal = bool(torch.equal(candidate_output != 0, native_output != 0))
    row_sum_error = float((candidate_output.sum(dim=-1) - 1.0).abs().max().item())
    if not support_equal or not math.isfinite(max_abs_error):
        raise AssertionError(
            f"support/error mismatch at rows={rows}, vocab={vocab}: "
            f"support={support_equal}, max_abs_error={max_abs_error}"
        )
    torch.testing.assert_close(candidate_output, native_output, rtol=2e-5, atol=2e-7)

    native_a1: list[float] = []
    candidate: list[float] = []
    native_a2: list[float] = []
    for _ in range(samples):
        _flush_cache(flush)
        native_a1.append(_event_us(native_call))
        _flush_cache(flush)
        candidate.append(_event_us(candidate_call))
        _flush_cache(flush)
        native_a2.append(_event_us(native_call))

    native_median = statistics.median(
        [(a + z) / 2.0 for a, z in zip(native_a1, native_a2, strict=True)]
    )
    candidate_median = statistics.median(candidate)
    ci_low, ci_high = _paired_bootstrap_ci(native_a1, candidate, native_a2)
    return {
        "rows": rows,
        "vocab": vocab,
        "top_k": TOP_K,
        "correctness": {
            "support_equal": support_equal,
            "max_abs_error": max_abs_error,
            "max_row_sum_error": row_sum_error,
        },
        "cold_cache": {
            "samples": samples,
            "native_a1_median_us": statistics.median(native_a1),
            "candidate_median_us": candidate_median,
            "native_a2_median_us": statistics.median(native_a2),
            "native_paired_median_us": native_median,
            "speedup_percent": 100.0
            * (native_median - candidate_median)
            / native_median,
            "speedup_percent_ci95": [ci_low, ci_high],
        },
        "graph_replay": {
            "native": _capture_replay_us(native_call, warmups, graph_samples),
            "candidate": _capture_replay_us(candidate_call, warmups, graph_samples),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--graph-samples", type=int, default=100)
    parser.add_argument("--flush-mib", type=int, default=8192)
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        help="ROWSxVOCAB; repeatable. Defaults to production and dashboard grids.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shape_args = args.shape or [
        "1x151936",
        "4x151936",
        "16x151936",
        "64x151936",
        "1x152064",
        "8x152064",
        "64x152064",
        "256x152064",
    ]
    shapes = [tuple(map(int, item.lower().split("x", 1))) for item in shape_args]
    flush = torch.empty(
        args.flush_mib * 1024 * 1024 // 4, device="musa", dtype=torch.float32
    )
    flush.zero_()
    torch.musa.synchronize()

    result = {
        "schema_version": 1,
        "torch": torch.__version__,
        "torch_musa": getattr(torch.version, "musa", None),
        "device": torch.musa.get_device_name(0),
        "flush_mib": args.flush_mib,
        "warmups": args.warmups,
        "rows": [],
    }
    for rows, vocab in shapes:
        row = _bench_shape(
            rows,
            vocab,
            args.warmups,
            args.samples,
            args.graph_samples,
            flush,
        )
        result["rows"].append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
