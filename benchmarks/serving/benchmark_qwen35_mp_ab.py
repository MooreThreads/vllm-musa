"""Compiled/captured Qwen3.5 TP4 baseline-vs-candidate benchmark.

The caller controls the production policy file before starting this process.
Each process should use a fresh container/cache so graph and JIT compilation
costs cannot leak from one policy arm into the other.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torchada  # noqa: F401
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/dist/models/Qwen3.5-35B-A3B-BF16")
    parser.add_argument("--policy", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return parser.parse_args()


def _instruction(tokenizer: AutoTokenizer) -> list[int]:
    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    "What is the capital of China? Answer in English with one word."
                ),
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def make_prompt_ids(tokenizer: AutoTokenizer, input_tokens: int) -> list[int]:
    if input_tokens < 32:
        raise ValueError("input_tokens must be at least 32")
    suffix = _instruction(tokenizer)
    filler = tokenizer.encode(
        "The following context is neutral and contains no additional instructions. ",
        add_special_tokens=False,
    )
    ids: list[int] = []
    while len(ids) + len(suffix) < input_tokens:
        ids.extend(filler)
    ids.extend(suffix)
    return ids[-input_tokens:]


def run_batch(
    llm: LLM,
    prompt_ids: list[int],
    batch_size: int,
    sampling: SamplingParams,
) -> tuple[float, list[str], int]:
    prompts = [TokensPrompt(prompt_token_ids=prompt_ids) for _ in range(batch_size)]
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.perf_counter() - start
    texts = [item.outputs[0].text for item in outputs]
    generated = sum(len(item.outputs[0].token_ids) for item in outputs)
    return elapsed, texts, generated


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.repeats < 1 or args.warmup < 0:
        raise ValueError("batch_size/repeats must be positive and warmup non-negative")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_ids = make_prompt_ids(tokenizer, args.input_tokens)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=4,
        trust_remote_code=True,
        enforce_eager=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.input_tokens + args.output_tokens + 256,
        max_num_seqs=args.batch_size,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_tokens,
        ignore_eos=True,
    )

    warmup_records = []
    for _ in range(args.warmup):
        elapsed, _, generated = run_batch(llm, prompt_ids, args.batch_size, sampling)
        warmup_records.append({"elapsed_s": elapsed, "generated_tokens": generated})

    records: list[dict[str, Any]] = []
    for index in range(args.repeats):
        elapsed, texts, generated = run_batch(
            llm, prompt_ids, args.batch_size, sampling
        )
        if generated != args.batch_size * args.output_tokens:
            raise AssertionError(
                f"run {index} generated {generated}, expected "
                f"{args.batch_size * args.output_tokens}"
            )
        records.append(
            {
                "index": index,
                "elapsed_s": elapsed,
                "generated_tokens": generated,
                "output_tokens_per_s": generated / elapsed,
                "semantic_text_sample": texts[0][:256],
            }
        )

    semantic_sampling = SamplingParams(temperature=0.0, max_tokens=32)
    _, semantic_texts, _ = run_batch(llm, _instruction(tokenizer), 1, semantic_sampling)
    semantic_text = semantic_texts[0]
    if "beijing" not in semantic_text.lower():
        raise AssertionError(
            f"semantic check did not contain Beijing: {semantic_text!r}"
        )

    throughputs = [record["output_tokens_per_s"] for record in records]
    result = {
        "policy": args.policy,
        "model": args.model,
        "tensor_parallel_size": 4,
        "compiled_or_captured": True,
        "enforce_eager": False,
        "batch_size": args.batch_size,
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "warmup": warmup_records,
        "records": records,
        "median_output_tokens_per_s": statistics.median(throughputs),
        "mean_output_tokens_per_s": statistics.mean(throughputs),
        "semantic_text": semantic_text,
        "semantic_pass": True,
        "argv": list(os.sys.argv),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
