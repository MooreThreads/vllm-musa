# Qwen3.5-27B-FP8

## Overview

FP8 dense-model recipe for two S5000 GPUs.

> [!TIP]
> Use this TP2 profile when serving the FP8 checkpoint.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 2x S5000 |
| Precision | FP8 |
| Tensor parallelism | TP2 |
| Speculative decoding | Off |
| Maximum context | 7,168 tokens |
| Maximum sequences | 64 |

## Launching the server

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_DISABLE_COMPILE_CACHE=1
export SAFETENSORS_FAST_GPU=1

vllm serve /models/Qwen3.5-27B-FP8 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --served-model-name Qwen3.5-27B-FP8 \
  --max-model-len 7168 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --attention-config '{"backend":"FLASH_ATTN"}' \
  --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,64]}'
```

## Configuration notes

- Replace `/models/Qwen3.5-27B-FP8` if needed.

Return to the [Qwen recipe index](README.md).
