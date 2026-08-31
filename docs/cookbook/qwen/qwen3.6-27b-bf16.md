# Qwen3.6-27B-BF16

## Overview

BF16 dense-model recipe for a single S5000 GPU.

> [!TIP]
> Use TP1 and keep speculative decoding disabled for this checkpoint.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 1x S5000 |
| Precision | BF16 |
| Tensor parallelism | TP1 |
| Speculative decoding | Off |
| Maximum context | 5,192 tokens |
| Maximum sequences | 64 |

## Launching the server

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SAFETENSORS_FAST_GPU=1

vllm serve /mnt/models/Qwen3.6-27B \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --served-model-name qwen36-27b-bf16 \
  --max-model-len 5192 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 2048 \
  --long-prefill-token-threshold 4096 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --generation-config vllm \
  --async-scheduling \
  --attention-config '{"backend":"FLASH_ATTN"}' \
  --compilation-config '{"compile_sizes":[1,4,16,64],"cudagraph_capture_sizes":[1,4,16,64],"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

## Configuration notes

- The 2,048-token scheduler budget favors predictable prefill admission.
- Chunked prefill and asynchronous scheduling are enabled.
- Replace `/mnt/models/Qwen3.6-27B` if needed.

Return to the [Qwen recipe index](README.md).
