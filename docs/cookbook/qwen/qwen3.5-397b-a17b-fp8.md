# Qwen3.5-397B-A17B-FP8

## Overview

FP8 MoE recipe for eight S5000 GPUs.

> [!TIP]
> Use this TP8 profile when serving the FP8 checkpoint.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 8x S5000 |
| Precision | FP8 |
| Architecture | MoE, 397B-A17B |
| Tensor parallelism | TP8 |
| Speculative decoding | Off |
| Maximum context | 8,192 tokens |
| Maximum sequences | 64 |

## Launching the server

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export SAFETENSORS_FAST_GPU=1

vllm serve /models/Qwen3.5-397B-A17B-FP8 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --served-model-name Qwen3.5-397B-A17B-FP8 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 32768 \
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

- The memory-utilization target is 0.90 for this FP8 MoE checkpoint.
- Chunked prefill and asynchronous scheduling are enabled.
- Replace `/models/Qwen3.5-397B-A17B-FP8` if needed.

Return to the [Qwen recipe index](README.md).
