#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-worker eager E2E run for one MP60 DSV4 GEMV tactic state."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

GEMV_BLOCK_ENV = "VLLM_MUSA_GEMV_MOE_BLOCK"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    parser.add_argument(
        "--tactic",
        choices=("gemv-mp60",),
        required=True,
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    return parser.parse_args()


def _set_mode(tactic: str, mode: str) -> None:
    os.environ.pop(GEMV_BLOCK_ENV, None)
    if mode == "baseline":
        if tactic == "gemv-mp60":
            os.environ[GEMV_BLOCK_ENV] = "16x8"
        else:  # pragma: no cover - argparse owns this contract
            raise ValueError(f"unknown tactic: {tactic}")
    elif mode == "candidate":
        pass
    else:  # pragma: no cover - internal contract
        raise ValueError(f"unknown mode: {mode}")


def _token_ids(outputs: list[Any]) -> list[list[int]]:
    return [list(output.outputs[0].token_ids) for output in outputs]


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main() -> int:
    args = parse_args()
    if args.repeats <= 0 or args.max_tokens <= 0:
        raise ValueError("repeats and max tokens must be positive")

    # Must precede LLM worker spawn. Driver-side environment changes made
    # afterward are not visible to existing tensor-parallel worker processes.
    _set_mode(args.tactic, args.mode)

    # Import order is part of the MUSA compatibility contract.
    # isort: off
    import torchada  # noqa: F401
    import torch
    from vllm import LLM, SamplingParams
    # isort: on

    properties = torch.musa.get_device_properties(0)
    if torch.musa.device_count() != args.tensor_parallel_size:
        raise RuntimeError("visible MUSA devices must equal tensor parallel size")
    expected_mp = 60
    if int(properties.multi_processor_count) != expected_mp:
        raise RuntimeError(f"this exact E2E tactic requires MP{expected_mp}")

    batch_size = 2
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        tokenizer_mode="deepseek_v4",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        kv_cache_dtype="fp8",
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        max_num_seqs=batch_size,
        seed=20260828,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        min_tokens=args.max_tokens,
        ignore_eos=True,
    )
    prompts = ["The future of artificial intelligence is"] * batch_size

    def run() -> dict[str, Any]:
        torch.musa.synchronize()
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        torch.musa.synchronize()
        elapsed = time.perf_counter() - started
        token_ids = _token_ids(outputs)
        output_tokens = sum(len(ids) for ids in token_ids)
        return {
            "mode": args.mode,
            "elapsed_seconds": elapsed,
            "output_tokens": output_tokens,
            "output_tokens_per_second": output_tokens / elapsed,
            "token_ids": token_ids,
            "texts": [output.outputs[0].text for output in outputs],
        }

    warmups = [run(), run()]
    samples = [run() for _ in range(args.repeats)]
    elapsed = [sample["elapsed_seconds"] for sample in samples]
    variants = {tuple(tuple(ids) for ids in sample["token_ids"]) for sample in samples}
    payload = {
        "schema": "musa-dsv4-e2e-kernel-tactic-fixed-worker.v1",
        "source_sha": os.environ.get("MUSA100006_SOURCE_SHA", "unknown"),
        "model": args.model,
        "device": {
            "name": torch.musa.get_device_name(0),
            "capability": [int(properties.major), int(properties.minor)],
            "multiprocessor_count": int(properties.multi_processor_count),
            "count": torch.musa.device_count(),
        },
        "execution": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "enforce_eager": True,
            "batch_size": len(prompts),
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
            "mode": args.mode,
            "tactic": args.tactic,
            "baseline_block": "16x8",
            "candidate_stage_blocks": {"w1": "8x16", "w2": "16x8"},
            "baseline_weighted_rmsnorm_threads": 128,
            "candidate_weighted_rmsnorm_threads": 256,
        },
        "warmups": warmups,
        "samples": samples,
        "summary": {
            "median_elapsed_seconds": statistics.median(elapsed),
            "p95_elapsed_seconds": _percentile(elapsed, 0.95),
            "min_elapsed_seconds": min(elapsed),
            "max_elapsed_seconds": max(elapsed),
            "unique_output_variants": len(variants),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
