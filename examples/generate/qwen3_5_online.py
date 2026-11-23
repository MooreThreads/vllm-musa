# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chat with a Qwen3.5-35B-A3B-FP8 server through the OpenAI API.

Start the server first (cookbook profile, at most four concurrent requests;
replace the model ID with a local path to skip the Hugging Face download):

    export VLLM_PLUGINS=musa,musa_custom_ops
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export SAFETENSORS_FAST_GPU=1
    MUSA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \\
        --served-model-name Qwen3.5-35B-A3B-FP8 \\
        --trust-remote-code \\
        --tensor-parallel-size 4 \\
        --max-model-len 8192 \\
        --max-num-seqs 4 \\
        --max-num-batched-tokens 8192 \\
        --gpu-memory-utilization 0.85 \\
        --mamba-ssm-cache-dtype float32 \\
        --no-enable-prefix-caching \\
        --no-enable-chunked-prefill \\
        --attention-config '{"backend":"FLASH_ATTN"}' \\
        --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16]}' \\
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'

Then run:

    python qwen3_5_online.py
"""

import os

from openai import OpenAI

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
MODEL = os.environ.get("VLLM_MODEL", "Qwen3.5-35B-A3B-FP8")

# Model card sampling for general tasks without thinking; top_k and the chat
# template switch are vLLM extensions passed through extra_body.
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5}
EXTRA_BODY = {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}


def main() -> None:
    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="EMPTY")

    print("=== Chat ===")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        max_tokens=256,
        extra_body=EXTRA_BODY,
        **SAMPLING,
    )
    print(response.choices[0].message.content)

    print("\n=== Streaming chat ===")
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Explain mixture-of-experts models in two sentences.",
            }
        ],
        max_tokens=256,
        stream=True,
        extra_body=EXTRA_BODY,
        **SAMPLING,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
