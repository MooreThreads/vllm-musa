# Qwen3.5-27B-FP8

## Overview

FP8 dense-model recipe with one-token MTP speculative decoding.

> [!TIP]
> Use this TP1 profile when serving the FP8 checkpoint.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 1x S5000 |
| Precision | FP8 |
| Tensor parallelism | TP1 |
| Speculative decoding | MTP1 |
| Maximum context | 5,192 tokens |
| Maximum sequences | 64 |

## Launching the server

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SAFETENSORS_FAST_GPU=1

vllm serve /mnt/models/Qwen3.5-27B-FP8 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --served-model-name qwen35-27b-fp8 \
  --max-model-len 5192 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --long-prefill-token-threshold 4096 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --generation-config vllm \
  --async-scheduling \
  --attention-config '{"backend":"FLASH_ATTN"}' \
  --compilation-config '{"cudagraph_capture_sizes":[2,8,32,128],"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --speculative-config '{"attention_backend":"FLASH_ATTN","method":"mtp","num_speculative_tokens":1}'
```

## Configuration notes

- MTP1 uses the FlashAttention backend for draft attention.
- Graph capture sizes account for the additional draft token.
- Replace `/mnt/models/Qwen3.5-27B-FP8` if needed.

Return to the [Qwen recipe index](README.md).
