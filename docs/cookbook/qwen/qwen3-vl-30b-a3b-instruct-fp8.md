# Qwen3-VL-30B-A3B-Instruct-FP8

## Overview

FP8 vision-language MoE recipe for four S5000 GPUs.

> [!TIP]
> Use this TP4 profile when serving the FP8 checkpoint.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 4x S5000 |
| Precision | FP8 |
| Architecture | Vision-Language MoE, 30B-A3B |
| Tensor parallelism | TP4 |
| Speculative decoding | Off |
| Maximum context | 8,192 tokens |
| Maximum sequences | 64 |

## Launching the server

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export SAFETENSORS_FAST_GPU=1

vllm serve /models/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --served-model-name Qwen3-VL-30B-A3B-Instruct-FP8 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --attention-backend FLASH_ATTN \
  --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,64]}'
```

## Configuration notes

- The memory-utilization target is 0.85 for this vision-language checkpoint.
- Chunked prefill and asynchronous scheduling are disabled.
- Replace `/models/Qwen3-VL-30B-A3B-Instruct-FP8` if needed.

Return to the [Qwen recipe index](README.md).
