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
  --max-num-partial-prefills 1 \
  --max-long-partial-prefills 1 \
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
