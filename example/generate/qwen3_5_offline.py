# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline chat inference for Qwen3.5-35B-A3B-FP8 on four S5000 cards.

Uses the cookbook profile (TP4, MTP3) with at most four concurrent requests.
The model is downloaded from Hugging Face unless --model points at a local
checkpoint.

    MUSA_VISIBLE_DEVICES=0,1,2,3 python qwen3_5_offline.py
    MUSA_VISIBLE_DEVICES=0,1,2,3 python qwen3_5_offline.py --model /path/to/model
"""

import argparse
import os

# MUSA plugins and runtime settings; must be set before vllm is imported.
os.environ.setdefault("VLLM_PLUGINS", "musa,musa_custom_ops")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")

from vllm import LLM, SamplingParams  # noqa: E402

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Write a Python function that checks whether a number is prime.",
    "Explain mixture-of-experts models in two sentences.",
    "Give three tips for writing clear technical documentation.",
]


def initialize_engine(model: str) -> LLM:
    return LLM(
        model=model,
        trust_remote_code=True,
        tensor_parallel_size=4,
        max_model_len=8192,
        max_num_seqs=4,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.85,
        mamba_ssm_cache_dtype="float32",
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        attention_config={"backend": "FLASH_ATTN"},
        # With MTP3 each request verifies 4 tokens per decode step, so 1-4
        # requests need graphs for 4, 8, 12 and 16 tokens.
        compilation_config={
            "mode": "NONE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [4, 8, 12, 16],
        },
        speculative_config={"method": "qwen3_5_mtp", "num_speculative_tokens": 3},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    args = parser.parse_args()

    llm = initialize_engine(args.model)
    # Model card sampling for general tasks without thinking.
    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=256
    )
    conversations = [[{"role": "user", "content": p}] for p in PROMPTS]
    outputs = llm.chat(
        conversations,
        sampling_params,
        chat_template_kwargs={"enable_thinking": False},
    )
    for prompt, output in zip(PROMPTS, outputs):
        print(f"\n### {prompt}\n{output.outputs[0].text.strip()}")


if __name__ == "__main__":
    main()
