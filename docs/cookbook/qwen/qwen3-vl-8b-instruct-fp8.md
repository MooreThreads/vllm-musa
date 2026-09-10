# Qwen3-VL-8B-Instruct-FP8

## Overview

FP8 vision-language recipe for a single S5000 GPU.

> [!TIP]
> Use this TP1 profile when serving the FP8 checkpoint.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 1x S5000 |
| Precision | FP8 |
| Architecture | Vision-Language |
| Tensor parallelism | TP1 |
| Speculative decoding | Off |
| Maximum context | 8,192 tokens |
| Maximum sequences | 256 |

## Launching the server

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export VLLM_MUSA_QWEN3_VL_VISION_CUDAGRAPH=1
export SAFETENSORS_FAST_GPU=1

CAPTURE_SIZES="$(seq -s, 1 256)"

vllm serve /models/Qwen3-VL-8B-Instruct-FP8 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --served-model-name Qwen3-VL-8B-Instruct-FP8 \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  --mamba-ssm-cache-dtype float32 \
  --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --attention-backend FLASH_ATTN \
  --mm-encoder-attn-backend FLASH_ATTN \
  --compilation-config "{\"mode\":\"NONE\",\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${CAPTURE_SIZES}]}"
```

## Configuration notes

- The memory-utilization target is 0.9 for this vision-language checkpoint.
- Chunked prefill and asynchronous scheduling are disabled.
- Replace `/models/Qwen3-VL-8B-Instruct-FP8` if needed.

Return to the [Qwen recipe index](README.md).
