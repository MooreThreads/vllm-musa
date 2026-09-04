# Qwen3.6-35B-A3B-BF16

## Overview

BF16 MoE recipe for four S5000 GPUs.

> [!TIP]
> Use TP4 and enable MTP2 speculative decoding for this checkpoint.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 4x S5000 |
| Precision | BF16 |
| Architecture | MoE, 35B-A3B |
| Tensor parallelism | TP4 |
| Speculative decoding | MTP2 |
| Maximum context | 8,192 tokens |

## Launching the server

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export SAFETENSORS_FAST_GPU=1

vllm serve /models/Qwen3.6-35B-A3B \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --served-model-name Qwen3.6-35B-A3B \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --mamba-ssm-cache-dtype float32 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --attention-config '{"backend":"FLASH_ATTN"}' \
  --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78,81,84,87,90,93,96,120,144,168,192]}' \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":2}'
```

## Configuration notes

- The memory-utilization target is 0.85 for this BF16 MoE checkpoint.
- Chunked prefill and asynchronous scheduling are disabled.
- Replace `/models/Qwen3.6-35B-A3B` if needed.

Return to the [Qwen recipe index](README.md).
