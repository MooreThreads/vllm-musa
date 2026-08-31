"""Compiled/captured Qwen3.5 TP4 same-source selector attribution.

This diagnostic compares the explicit upstream rollback with AUTO on one source
tree. Exact base-vs-head serving gates must keep AUTO in both source arms.
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

# Keep the benchmark portable: model storage is fleet-specific and must be
# supplied by the caller rather than baked into a published script.
DEFAULT_MODEL = None
DISPATCH_ENV = "VLLM_MUSA_FUSED_MOE_DISPATCH"


def inspect_compile_state(llm: Any) -> dict[str, Any] | None:
    """Return client intent plus every worker's resolved dispatcher state."""
    engine = getattr(llm, "llm_engine", None)
    vllm_config = getattr(engine, "vllm_config", None)
    compilation_config = getattr(vllm_config, "compilation_config", None)
    if compilation_config is None:
        return None
    mode = getattr(compilation_config, "cudagraph_mode", None)
    mode_name = getattr(mode, "name", None) or str(mode)
    capture_sizes = getattr(compilation_config, "cudagraph_capture_sizes", None)
    worker_states = llm.collective_rpc("get_musa_cudagraph_runtime_state", timeout=60)
    worker_runtime_modes = {state["resolved_cudagraph_mode"] for state in worker_states}
    workers_active = bool(
        worker_states
        and len(worker_states) == 4
        and len(worker_runtime_modes) == 1
        and worker_runtime_modes != {"NONE"}
        and all(state["keys_initialized"] for state in worker_states)
        and all(state["capture_descriptors"] for state in worker_states)
    )
    return {
        "client_cudagraph_mode": mode_name,
        "client_cudagraph_capture_sizes": list(capture_sizes or []),
        "worker_states": worker_states,
        "compile_active": bool(
            mode_name.upper() not in {"NONE", "NONE_MODE"} and workers_active
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="local model path or ID")
    parser.add_argument("--policy", choices=("upstream", "auto"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--input-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return parser.parse_args()


def _instruction(tokenizer: Any) -> list[int]:
    encoded = tokenizer.apply_chat_template(
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
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    return list(encoded)


def make_prompt_ids(tokenizer: Any, input_tokens: int) -> list[int]:
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
    llm: Any,
    prompt_ids: list[int],
    batch_size: int,
    sampling: Any,
) -> tuple[float, list[str], int]:
    from vllm import TokensPrompt

    prompts = [TokensPrompt(prompt_token_ids=prompt_ids) for _ in range(batch_size)]
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.perf_counter() - start
    texts = [item.outputs[0].text for item in outputs]
    generated = sum(len(item.outputs[0].token_ids) for item in outputs)
    return elapsed, texts, generated


def main() -> int:
    args = parse_args()
    if (
        args.batch_size < 1
        or args.repeats < 1
        or args.warmup < 0
        or (args.max_num_seqs is not None and args.max_num_seqs < args.batch_size)
    ):
        raise ValueError(
            "batch_size/repeats must be positive, warmup non-negative, and "
            "max_num_seqs must cover batch_size"
        )

    if args.input_tokens > 512:
        raise ValueError(
            "same-source selector attribution is limited to <=512 input tokens; "
            "use exact base-vs-head AUTO serving arms for long-prefill gates"
        )
    max_num_seqs = args.max_num_seqs or args.batch_size
    dispatch_policy = args.policy
    os.environ[DISPATCH_ENV] = dispatch_policy

    import torchada  # noqa: F401 - activate MUSA compatibility first
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_ids = make_prompt_ids(tokenizer, args.input_tokens)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=4,
        trust_remote_code=True,
        enforce_eager=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.input_tokens + args.output_tokens + 256,
        max_num_seqs=max_num_seqs,
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

    compile_state = inspect_compile_state(llm)
    if compile_state is None or not compile_state["compile_active"]:
        raise RuntimeError(
            "vLLM resolved cudagraph mode is unavailable or disabled; "
            "refusing to record a compiled A/B result"
        )

    throughputs = [record["output_tokens_per_s"] for record in records]
    result = {
        "policy": args.policy,
        "model": args.model,
        "tensor_parallel_size": 4,
        "compile_state": compile_state,
        "compiled_or_captured": True,
        "enforce_eager": False,
        "batch_size": args.batch_size,
        "max_num_seqs": max_num_seqs,
        "dispatch_policy": dispatch_policy,
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
