# Qwen3-VL-32B-Instruct-FP8

## Overview

FP8 vision-language recipe for four S5000 GPUs.

> [!TIP]
> Use this TP4 profile when serving the FP8 checkpoint.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 4x S5000 |
| Precision | FP8 |
| Architecture | Vision-Language |
| Tensor parallelism | TP4 |
| Speculative decoding | Off |
| Maximum context | 8,192 tokens |
| Maximum sequences | 64 |

## Validation

| Item | Value |
|---|---|
| Checkpoint | [`Qwen/Qwen3-VL-32B-Instruct-FP8`](https://modelscope.cn/models/Qwen/Qwen3-VL-32B-Instruct-FP8) (ModelScope) @ 1417e71e89cc11576e9eec131bb4507337d195a9 |
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

vllm serve /models/Qwen3-VL-32B-Instruct-FP8 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --served-model-name Qwen3-VL-32B-Instruct-FP8 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --max-num-partial-prefills 1 \
  --max-long-partial-prefills 1 \
  --long-prefill-token-threshold 4096 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":1}' \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --generation-config vllm \
  --attention-config '{"backend":"FLASH_ATTN"}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,4,16,64],"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

## Configuration notes

- The memory-utilization target is 0.90 for this vision-language checkpoint.
- Chunked prefill is enabled and prefix caching is disabled.
- Replace `/models/Qwen3-VL-32B-Instruct-FP8` if needed.

Return to the [Qwen recipe index](README.md).
