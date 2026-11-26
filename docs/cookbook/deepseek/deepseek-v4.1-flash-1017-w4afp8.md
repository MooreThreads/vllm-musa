# DeepSeek-V4.1-Flash-1017-W4AFP8

## Overview

DeepSeek-V4.1-Flash-1017-W4AFP8 is served as a single-node TP8 deployment on
eight S5000 GPUs with expert parallelism. This recipe covers two DSpark
profiles for the same deployment shape:

- **Adaptive-5**: five draft tokens with adaptive verification enabled.
- **Fixed-4**: four draft tokens with adaptive verification disabled.

> [!TIP]
> Use Adaptive-5 when adaptive verification is desired. Keep Fixed-4 as the
> four-token comparison profile.

## Model source

The MUSA-converted W4AFP8 checkpoint is available from
[ModelScope](https://modelscope.cn/models/MooreThreads/DeepSeek-V4.1-Flash-1017-W4AFP8-mt).

## At a glance

| Setting | Value |
|---|---|
| Hardware | 8x S5000 |
| Deployment shape | Single node |
| Precision | W4AFP8 |
| Architecture | MoE |
| Tensor parallelism | TP8 |
| Expert parallelism | EP8 |
| Attention backend | FlashMLA |
| KV cache | FP8 |
| Speculative decoding | DSpark Adaptive-5 or Fixed-4 |
| Maximum context | 6,144 tokens |
| Maximum batched tokens | 8,195 |
| Maximum sequences | 64 |
| Prefix caching | Off |
| CUDA graphs | `FULL_DECODE_ONLY` |

## Prerequisites

- vLLM-MUSA from the `v0.32.0-dev` branch.
- Eight S5000 GPUs available to one TP8 server.
- The W4AFP8 checkpoint mounted at
  `/models/DeepSeek-V4.1-Flash-1017-W4AFP8-mt`, or an equivalent path
  substituted in both commands.
- FlashMLA and the MUSA custom-op plugin available in the environment.

## Launching the server

### DSpark Adaptive-5

```bash
export VLLM_PLUGINS=musa,musa_custom_ops

vllm serve /models/DeepSeek-V4.1-Flash-1017-W4AFP8-mt \
  --trust-remote-code \
  --tokenizer-mode deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --limit-mm-per-prompt '{"image":0}' \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --served-model-name DeepSeek-V4.1-Flash \
  --max-model-len 6144 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8195 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --attention-backend FLASHMLA \
  --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":64,"cudagraph_capture_sizes":[1,2,4,8,16,32,64],"cudagraph_copy_inputs":false}' \
  --async-scheduling \
  --port 8000 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy","enable_adaptive_verification":true}'
```

### DSpark Fixed-4

```bash
export VLLM_PLUGINS=musa,musa_custom_ops

vllm serve /models/DeepSeek-V4.1-Flash-1017-W4AFP8-mt \
  --trust-remote-code \
  --tokenizer-mode deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --limit-mm-per-prompt '{"image":0}' \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --served-model-name DeepSeek-V4.1-Flash \
  --max-model-len 6144 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8195 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --attention-backend FLASHMLA \
  --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":64,"cudagraph_capture_sizes":[1,2,4,8,16,32,64],"cudagraph_copy_inputs":false}' \
  --async-scheduling \
  --port 8000 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":4,"draft_sample_method":"greedy","enable_adaptive_verification":false}'
```

## Verify the endpoint

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"What is 2+3? Reply with only the digit."}],"temperature":0,"max_tokens":16,"chat_template_kwargs":{"thinking":false}}'
```

The recorded semantic response is `5`.

Return to the [cookbook index](../README.md).
