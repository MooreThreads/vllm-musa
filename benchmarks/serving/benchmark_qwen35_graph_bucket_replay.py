"""Replay mixed decode buckets through one compiled Qwen3.5 engine.

This is a graph-contamination gate, not an HTTP latency benchmark. It keeps one
engine alive while replaying small and fallback buckets in an adversarial order.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark_qwen35_mp_ab import (
    DEFAULT_MODEL,
    make_prompt_ids,
)


def _parse_sequence(value: str) -> list[int]:
    sequence = [int(item) for item in value.split(",") if item.strip()]
    if not sequence or any(batch < 1 for batch in sequence):
        raise argparse.ArgumentTypeError(
            "batch sequence must contain positive integers"
        )
    return sequence


def _run_batch(
    *,
    llm: Any,
    prompts: list[Any],
    sampling_params: Any,
    batch_size: int,
    expected_output_tokens: int,
) -> dict[str, Any]:
    import torch

    batch = prompts[:batch_size]
    torch.musa.synchronize()
    started = time.perf_counter()
    outputs = llm.generate(batch, sampling_params, use_tqdm=False)
    torch.musa.synchronize()
    elapsed_s = time.perf_counter() - started

    token_counts = [len(output.outputs[0].token_ids) for output in outputs]
    texts = [output.outputs[0].text for output in outputs]
    semantic_pass = all("beijing" in text.lower() for text in texts)
    exact_output_tokens = all(count == expected_output_tokens for count in token_counts)
    if not exact_output_tokens:
        raise RuntimeError(
            f"expected {expected_output_tokens} output tokens per request, "
            f"got {token_counts}"
        )
    if not semantic_pass:
        raise RuntimeError("semantic output check failed: expected Beijing")

    total_output_tokens = sum(token_counts)
    return {
        "batch_size": batch_size,
        "elapsed_s": elapsed_s,
        "output_tokens": total_output_tokens,
        "output_tokens_per_s": total_output_tokens / elapsed_s,
        "exact_output_tokens": exact_output_tokens,
        "semantic_pass": semantic_pass,
        "texts": texts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--policy", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument(
        "--batch-sequence",
        type=_parse_sequence,
        default="1,16,4,16,1,2,16",
    )
    parser.add_argument("--warmup-sequence", type=_parse_sequence, default="1,16,4")
    parser.add_argument("--input-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--cudagraph-capture-sizes", type=_parse_sequence)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    max_batch = max(args.batch_sequence + args.warmup_sequence)
    if args.max_num_seqs < max_batch:
        raise ValueError("max_num_seqs must cover every replay batch")

    import torchada  # noqa: F401 - activate MUSA compatibility first
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    token_ids = make_prompt_ids(tokenizer, args.input_tokens)
    prompts = [TokensPrompt(prompt_token_ids=token_ids) for _ in range(max_batch)]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_tokens,
        min_tokens=args.output_tokens,
        ignore_eos=True,
    )

    llm = LLM(
        model=args.model,
        tensor_parallel_size=4,
        trust_remote_code=True,
        enforce_eager=False,
        dtype="bfloat16",
        seed=args.seed,
        max_model_len=max(8192, args.input_tokens + args.output_tokens + 128),
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=0.9,
        compilation_config=(
            {"cudagraph_capture_sizes": args.cudagraph_capture_sizes}
            if args.cudagraph_capture_sizes is not None
            else None
        ),
    )

    for batch_size in args.warmup_sequence:
        _run_batch(
            llm=llm,
            prompts=prompts,
            sampling_params=sampling_params,
            batch_size=batch_size,
            expected_output_tokens=args.output_tokens,
        )

    records = []
    for step, batch_size in enumerate(args.batch_sequence):
        record = _run_batch(
            llm=llm,
            prompts=prompts,
            sampling_params=sampling_params,
            batch_size=batch_size,
            expected_output_tokens=args.output_tokens,
        )
        record["step"] = step
        records.append(record)

    grouped: dict[int, list[float]] = defaultdict(list)
    for record in records:
        grouped[record["batch_size"]].append(record["output_tokens_per_s"])
    per_batch = {
        str(batch_size): {
            "samples": values,
            "median_output_tokens_per_s": statistics.median(values),
        }
        for batch_size, values in sorted(grouped.items())
    }

    result = {
        "policy": args.policy,
        "model": args.model,
        "tensor_parallel_size": 4,
        "max_num_seqs": args.max_num_seqs,
        "batch_sequence": args.batch_sequence,
        "warmup_sequence": args.warmup_sequence,
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "cudagraph_capture_sizes": args.cudagraph_capture_sizes,
        "seed": args.seed,
        "enforce_eager": False,
        "compiled_or_captured": True,
        "semantic_pass": all(record["semantic_pass"] for record in records),
        "exact_output_tokens": all(record["exact_output_tokens"] for record in records),
        "per_batch": per_batch,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    del llm
    gc.collect()


if __name__ == "__main__":
    main()
