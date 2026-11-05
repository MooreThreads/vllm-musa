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

## Validation

| Item | Value |
|---|---|
| Checkpoint | [`Qwen/Qwen3-VL-32B-Instruct`](https://modelscope.cn/models/Qwen/Qwen3-VL-32B-Instruct) (ModelScope) @ 8d6f46b05db6abe0cb3a5ca8c817929cc0fa2d5a |
| vLLM-MUSA | vLLM 0.28.0 @ 2cf0a69, vllm-musa 0.1.28 |
| MUSA driver | 5.2.0 |
| Hardware | 4x S5000 |
| Served request | 2K/1K, 3K/1K, 3.5K/1K, 4K/1K, 4K/1.5K, 500 prompts each |

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
