# Qwen3.5-35B-A3B-BF16

## Overview

BF16 MoE recipe for a single S5000 GPU.

> [!TIP]
> Use TP1 and keep speculative decoding disabled for this checkpoint.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 1x S5000 |
| Precision | BF16 |
| Architecture | MoE, 35B-A3B |
| Tensor parallelism | TP1 |
| Speculative decoding | Off |
| Maximum context | 5,192 tokens |

## Launching the server

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SAFETENSORS_FAST_GPU=1

vllm serve /mnt/models/Qwen3.5-35B-A3B-BF16 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --served-model-name qwen35-35b-a3b-bf16 \
  --max-model-len 5192 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --max-num-partial-prefills 1 \
  --max-long-partial-prefills 1 \
  --long-prefill-token-threshold 2048 \
  --gpu-memory-utilization 0.95 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --generation-config vllm \
  --async-scheduling \
  --attention-config '{"backend":"FLASH_ATTN"}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,4,16,64],"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

## Configuration notes

- The memory-utilization target is 0.95 for this BF16 MoE checkpoint.
- Chunked prefill and asynchronous scheduling are enabled.
- Replace `/mnt/models/Qwen3.5-35B-A3B-BF16` if needed.

Return to the [Qwen recipe index](README.md).
