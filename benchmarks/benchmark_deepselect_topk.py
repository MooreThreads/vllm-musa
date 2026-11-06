"""Compare DeepSelect with the production DeepSeek-V4 MUSA radix top-k."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import torchada
import torch

from mate.testing.utils import bench_gpu_time_with_musa_event, bench_kineto

import vllm._deepselect_C as deepselect_extension
import vllm_musa
from vllm_musa import _custom_ops

LOGGER = logging.getLogger(__name__)
PRODUCTION_BATCHES = (1, 4, 16, 64)
PRODUCTION_WIDTHS = (2048, 8192, 32768)
README_BATCHES = (6, 512, 4096)
README_WIDTHS = (16384, 65536, 131072, 262144, 524288, 1048576)


def percentile(samples: list[float], quantile: float) -> float:
    position = (len(samples) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    ordered = sorted(samples)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def check_output(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    output: torch.Tensor,
    expected_values: torch.Tensor,
) -> None:
    topk = output.shape[1]
    ranks = torch.arange(topk, device=logits.device).expand_as(output)
    valid = ranks < lengths[:, None].clamp(max=topk)
    if not bool(torch.all(output[~valid] == -1)):
        raise AssertionError("Out-of-bounds entries must be padded with -1")
    if not bool(torch.all((output[valid] >= 0))):
        raise AssertionError("Unwritten or negative valid index")
    if not bool(torch.all(output[valid] < lengths[:, None].expand_as(output)[valid])):
        raise AssertionError("Selected index exceeds its row length")
    sorted_indices = output.sort(dim=1).values
    duplicate = (sorted_indices[:, 1:] == sorted_indices[:, :-1]) & (
        sorted_indices[:, 1:] >= 0
    )
    if bool(duplicate.any()):
        raise AssertionError("Duplicate selected indices")
    values = logits.gather(1, output.clamp(min=0).long())
    values.masked_fill_(~valid, float("-inf"))
    if not torch.equal(values.sort(dim=1, descending=True).values, expected_values):
        raise AssertionError("Selected values differ from the exact top-k multiset")


def profile_kernels(
    runners: dict[str, Callable[[], None]], output: Path
) -> dict[str, list[str]]:
    result = {}
    for name, runner in runners.items():
        runner()
        torch.musa.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.MUSA]
        ) as profiler:
            runner()
            torch.musa.synchronize()
        trace_path = output / f"{name}.trace.json"
        profiler.export_chrome_trace(str(trace_path))
        trace = json.loads(trace_path.read_text())
        kernels = sorted(
            {
                event["name"]
                for event in trace["traceEvents"]
                if "kernel" in event.get("cat", "").lower()
            }
        )
        if not kernels:
            raise RuntimeError(f"No device kernels captured for {name}")
        if name == "vllm_radix" and not any(
            "sparse_indexer_topk_radix_kernel" in kernel for kernel in kernels
        ):
            raise RuntimeError(f"Production radix kernel not observed: {kernels}")
        result[name] = kernels
    return result


def run_case(batch: int, width: int, args: argparse.Namespace, profile: bool) -> dict:
    generator = torch.Generator(device="musa").manual_seed(20261029 + batch + width)
    logits = torch.randn(
        batch, width, dtype=torch.float32, device="musa", generator=generator
    )
    lengths = torch.full((batch,), width, dtype=torch.int32, device="musa")
    if args.lengths == "ragged":
        lengths = (torch.arange(batch, dtype=torch.int32, device="musa") * 997) % (
            width + 1
        )
    outputs = {
        name: torch.full((batch, args.topk), -99, dtype=torch.int32, device="musa")
        for name in ("vllm_radix", "deepselect")
    }

    def radix() -> None:
        _custom_ops.deepseek_v4_sparse_indexer_topk_decode(
            logits, lengths, outputs["vllm_radix"], args.topk
        )

    def deepselect() -> None:
        torch.ops.deep_select.topk(
            logits,
            args.topk,
            None,
            lengths,
            False,
            False,
            None,
            outputs["deepselect"],
            None,
            -1,
            float("-inf"),
            False,
            False,
        )

    runners = {"vllm_radix": radix, "deepselect": deepselect}
    torch_values = torch.empty(batch, args.topk, dtype=torch.float32, device="musa")
    torch_indices = torch.empty(batch, args.topk, dtype=torch.int64, device="musa")
    oracle_input = logits
    if args.lengths == "ragged":
        oracle_input = logits.masked_fill(
            torch.arange(width, device="musa")[None, :] >= lengths[:, None],
            float("-inf"),
        )
    torch.topk(
        oracle_input, args.topk, dim=1, sorted=True, out=(torch_values, torch_indices)
    )
    for runner in runners.values():
        runner()
    torch.musa.synchronize()
    for output in outputs.values():
        check_output(logits, lengths, output, torch_values)
    if args.include_torch:
        if args.lengths != "full":
            raise ValueError("torch.topk reference timing requires --lengths full")

        def torch_reference() -> None:
            torch.topk(
                logits,
                args.topk,
                dim=1,
                sorted=False,
                out=(torch_values, torch_indices),
            )

        runners["torch_reference"] = torch_reference

    kernels = profile_kernels(runners, args.output.parent) if profile else None
    samples = {name: [] for name in runners}
    order = list(runners)
    for round_index in range(args.rounds):
        round_order = order if round_index % 2 == 0 else list(reversed(order))
        for name in round_order:
            measurements = bench_gpu_time_with_musa_event(
                runners[name],
                dry_run_iters=3,
                repeat_iters=args.iters,
                l2_flush=True,
                l2_flush_size_mb=8000,
            )
            samples[name].extend(measurements)
            torch.musa.synchronize()
    kernel_seconds = {}
    kernel_names = {
        "vllm_radix": "sparse_indexer_topk_radix_kernel",
        "deepselect": "topk_fp32_kernel",
    }
    kernel_order = list(kernel_names)
    if batch % 2 == 0:
        kernel_order.reverse()
    for name in kernel_order:
        trace_path = (
            str(args.output.parent / f"{name}.cold.trace.json") if profile else None
        )
        kernel_seconds[name] = bench_kineto(
            runners[name],
            kernel_names[name],
            num_tests=args.iters,
            suppress_kineto_output=True,
            flush_l2=True,
            trace_path=trace_path,
        )
        if kernel_seconds[name] <= 0:
            raise RuntimeError(f"Missing kernel timing for {name}")
        torch.musa.synchronize()
    metrics = {}
    effective_bytes = (
        logits.numel() * logits.element_size() + outputs["deepselect"].numel() * 4
    )
    for name, timings in samples.items():
        median_ms = statistics.median(timings)
        metrics[name] = {
            "median_us": median_ms * 1000,
            "p95_us": percentile(timings, 0.95) * 1000,
            "iqr_us": (percentile(timings, 0.75) - percentile(timings, 0.25)) * 1000,
            "effective_tb_s": effective_bytes / (median_ms * 1e-3) / 1e12,
            "samples_ms": timings,
        }
        if name in kernel_seconds:
            metrics[name]["kernel_mean_us"] = kernel_seconds[name] * 1e6
            metrics[name]["kernel_effective_tb_s"] = (
                effective_bytes / kernel_seconds[name] / 1e12
            )
    result = {
        "batch": batch,
        "width": width,
        "topk": args.topk,
        "dtype": "float32",
        "index_dtype": "int32",
        "lengths": args.lengths,
        "correctness": "PASS",
        "metrics": metrics,
        "speedup_vs_vllm_radix": metrics["vllm_radix"]["median_us"]
        / metrics["deepselect"]["median_us"],
        "speedup_kernel_vs_vllm_radix": kernel_seconds["vllm_radix"]
        / kernel_seconds["deepselect"],
        "kernels": kernels,
    }
    del logits, outputs, torch_values, torch_indices, oracle_input, lengths
    torch.musa.empty_cache()
    return result


def provenance() -> dict:
    sources = [
        Path(__file__),
        Path(vllm_musa.__file__).parent.parent
        / "csrc/musa/attention/deepseek_v4_sparse_indexer_topk.mu",
        Path(deepselect_extension.__file__),
    ]
    return {
        "hostname": platform.node(),
        "python": sys.version,
        "device": torch.musa.get_device_name(0),
        "torch": torch.__version__,
        "musa": torch.version.musa,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "vllm-musa", "torchada", "mate")
        },
        "sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
        "command": sys.argv,
        "status": "exploratory: user-provided container; node exclusivity not established",
        "timing": "mate MUSA events, round-interleaved, 8000 MiB flush before every timed call",
        "flush_bytes": 8000 * 1024 * 1024,
        "kernel_timing": "mate bench_kineto: 2N-1 active calls, 8000000000-byte flush per call",
        "bandwidth": "logical FP32 input bytes plus int32 output bytes; not measured DRAM traffic",
        "torch_reference": "extra values/int64 output; reference only, not the production baseline",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("production", "readme", "all"), default="production"
    )
    parser.add_argument("--batches", type=int, nargs="+")
    parser.add_argument("--widths", type=int, nargs="+")
    parser.add_argument("--topk", type=int, choices=range(8, 513, 8), default=512)
    parser.add_argument("--lengths", choices=("full", "ragged"), default="full")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--include-torch", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 2 or args.iters < 2:
        parser.error("At least two rounds and iterations are required")
    if args.lengths != "full" and args.include_torch:
        parser.error("torch.topk has no per-row lengths; omit --include-torch")
    if (args.batches is None) != (args.widths is None):
        parser.error("Specify both --batches and --widths")
    shapes = []
    if args.batches is not None:
        shapes = [(batch, width) for batch in args.batches for width in args.widths]
    else:
        if args.suite in ("production", "all"):
            shapes.extend(
                (batch, width)
                for batch in PRODUCTION_BATCHES
                for width in PRODUCTION_WIDTHS
            )
        if args.suite in ("readme", "all"):
            shapes.extend(
                (batch, width) for batch in README_BATCHES for width in README_WIDTHS
            )
    if any(batch <= 0 or width < args.topk or width % 8 for batch, width in shapes):
        parser.error("Positive batches and FP32-aligned widths >= topk are required")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    vllm_musa.register_custom_ops()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"provenance": provenance(), "results": []}
    for shape_index, (batch, width) in enumerate(shapes):
        result = run_case(batch, width, args, profile=shape_index == 0)
        report["results"].append(result)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        LOGGER.info(
            "B=%d N=%d radix=%.3fus deepselect=%.3fus speedup=%.3fx correctness=PASS",
            batch,
            width,
            result["metrics"]["vllm_radix"]["median_us"],
            result["metrics"]["deepselect"]["median_us"],
            result["speedup_vs_vllm_radix"],
        )


if __name__ == "__main__":
    main()
