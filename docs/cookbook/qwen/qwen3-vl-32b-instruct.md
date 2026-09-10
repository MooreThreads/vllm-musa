# Qwen3-VL-32B-Instruct

## Overview

Vision-language recipe for four S5000 GPUs.

> [!TIP]
> Use TP4 and keep speculative decoding disabled for this checkpoint.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 4x S5000 |
| Precision | BF16 |
| Architecture | Vision-Language |
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

vllm serve /models/Qwen3-VL-32B-Instruct \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --served-model-name Qwen3-VL-32B-Instruct \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill
```

## Configuration notes

- The memory-utilization target is 0.85 for this vision-language checkpoint.
- Chunked prefill is enabled and prefix caching is disabled.
- Replace `/models/Qwen3-VL-32B-Instruct` if needed.

Return to the [Qwen recipe index](README.md).
