# Qwen3.5-35B-A3B-BF16

## Overview

BF16 MoE recipe for four S5000 GPUs.

> [!TIP]
> Use TP4 and enable MTP3 speculative decoding for this checkpoint.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 4x S5000 |
| Precision | BF16 |
| Architecture | MoE, 35B-A3B |
| Tensor parallelism | TP4 |
| Speculative decoding | MTP3 |
| Maximum context | 8,192 tokens |
| Maximum sequences | 192 |

## Validation

| Item | Value |
|---|---|
| Checkpoint | [`Qwen/Qwen3.5-35B-A3B`](https://modelscope.cn/models/Qwen/Qwen3.5-35B-A3B) (ModelScope) @ 712cf74392b05026a6db2bf213d343747d1f6d45 |
| vLLM-MUSA | vLLM 0.28.0 @ 2cf0a69, vllm-musa 0.1.28 |
| MUSA driver | 5.2.0 |
| Hardware | 4x S5000 |
| Served request | 2K/1K, 3K/1K, 3.5K/1K, 4K/1K, 4K/1.5K, 500 prompts each |

## Launching the server

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/usr/local/mtshmem/lib:${LD_LIBRARY_PATH:-}
export SAFETENSORS_FAST_GPU=1

CAPTURE_SIZES="$(seq -s, 4 4 768)"

vllm serve /models/Qwen3.5-35B-A3B \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --served-model-name Qwen3.5-35B-A3B \
  --max-model-len 8192 \
  --max-num-seqs 192 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --mamba-ssm-cache-dtype float32 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --attention-config '{"backend":"FLASH_ATTN"}' \
  --compilation-config "{\"mode\":\"NONE\",\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${CAPTURE_SIZES}]}" \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
```

## Configuration notes

- The memory-utilization target is 0.85 for this BF16 MoE checkpoint.
- Chunked prefill and asynchronous scheduling are disabled.
- Replace `/models/Qwen3.5-35B-A3B` if needed.

Return to the [Qwen recipe index](README.md).
