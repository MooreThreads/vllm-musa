# Qwen3.5-122B-A10B-BF16

## Overview

BF16 MoE recipe for eight S5000 GPUs.

> [!TIP]
> Use TP8 and enable MTP3 speculative decoding for this checkpoint.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 8x S5000 |
| Precision | BF16 |
| Architecture | MoE, 122B-A10B |
| Tensor parallelism | TP8 |
| Speculative decoding | MTP3 |
| Maximum context | 8,192 tokens |
| Maximum sequences | 128 |

## Launching the server

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export VLLM_MUSA_QWEN3_VL_VISION_CUDAGRAPH=1
export SAFETENSORS_FAST_GPU=1

CAPTURE_SIZES="$(seq -s, 4 4 512)"

vllm serve /models/Qwen3.5-122B-A10B \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --served-model-name Qwen3.5-122B-A10B \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.8 \
  --mamba-ssm-cache-dtype float32 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --attention-backend FLASH_ATTN \
  --mm-encoder-attn-backend FLASH_ATTN \
  --compilation-config "{\"mode\":\"NONE\",\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${CAPTURE_SIZES}]}" \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":3}'
```

## Configuration notes

- The memory-utilization target is 0.8 for this BF16 MoE checkpoint.
- Chunked prefill and asynchronous scheduling are disabled.
- Replace `/models/Qwen3.5-122B-A10B` if needed.

Return to the [Qwen recipe index](README.md).
