# Qwen3.6-27B-BF16

## Overview

BF16 dense-model recipe for two S5000 GPUs.

> [!TIP]
> Use TP2 and enable MTP3 speculative decoding for this checkpoint.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 2x S5000 |
| Precision | BF16 |
| Tensor parallelism | TP2 |
| Speculative decoding | MTP3 |
| Maximum context | 8,192 tokens |
| Maximum sequences | 128 |

## Launching the server

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export SAFETENSORS_FAST_GPU=1

CAPTURE_SIZES="$(seq -s, 4 4 512)"

vllm serve /models/Qwen3.6-27B \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --served-model-name Qwen3.6-27B \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  --mamba-ssm-cache-dtype float32 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --attention-config '{"backend":"FLASH_ATTN"}' \
  --compilation-config "{\"mode\":\"NONE\",\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${CAPTURE_SIZES}]}" \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
```

## Configuration notes

- The 8,192-token scheduler budget favors predictable prefill admission.
- Chunked prefill and asynchronous scheduling are disabled.
- Replace `/models/Qwen3.6-27B` if needed.

Return to the [Qwen recipe index](README.md).
