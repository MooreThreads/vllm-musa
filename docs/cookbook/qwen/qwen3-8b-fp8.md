# Qwen3-8B-FP8

## Overview

FP8 dense-model recipe for a single S5000 GPU.

> [!TIP]
> Use TP1 and keep speculative decoding disabled for this checkpoint.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 1x S5000 |
| Precision | FP8 |
| Tensor parallelism | TP1 |
| Speculative decoding | Off |
| Maximum context | 5,192 tokens |
| Maximum sequences | 64 |

## Validation

| Item | Value |
|---|---|
| Checkpoint | [`Qwen/Qwen3-8B-FP8`](https://modelscope.cn/models/Qwen/Qwen3-8B-FP8) (ModelScope) @ 82f329a83a9f75ab4d922e2528e403bc649062c3 |
| vLLM-MUSA | vLLM 0.28.0 @ 2cf0a69, vllm-musa 0.1.28 |
| MUSA driver | 5.2.0 |
| Hardware | 1x S5000 |
| Served request | 2K/1K, 3K/1K, 3.5K/1K, 4K/1K, 4K/1.5K, 500 prompts each |

## Launching the server

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SAFETENSORS_FAST_GPU=1

vllm serve /models/Qwen3-8B-FP8 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --served-model-name qwen3-8b-fp8 \
  --max-model-len 5192 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 4096 \
  --long-prefill-token-threshold 2048 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --generation-config vllm \
  --async-scheduling \
  --attention-config '{"backend":"FLASH_ATTN"}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,4,16,64],"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

## Configuration notes

- Chunked prefill and asynchronous scheduling are enabled.
- Prefix caching is disabled.
- Replace `/models/Qwen3-8B-FP8` if the checkpoint is mounted elsewhere.

Return to the [Qwen recipe index](README.md).
