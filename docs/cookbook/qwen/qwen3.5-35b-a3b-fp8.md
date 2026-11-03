# Qwen3.5-35B-A3B-FP8

## Overview

FP8 MoE recipe with three-token MTP speculative decoding.

> [!TIP]
> Use this TP4 profile for small-batch serving of the FP8 checkpoint.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 4x S5000 |
| Precision | FP8 |
| Architecture | MoE, 35B-A3B |
| Tensor parallelism | TP4 |
| Speculative decoding | MTP3 |
| Maximum context | 8,192 tokens |
| Maximum sequences | 192 |

## Validation

| Item | Value |
|---|---|
| Checkpoint | [`Qwen/Qwen3.5-35B-A3B-FP8`](https://modelscope.cn/models/Qwen/Qwen3.5-35B-A3B-FP8) (ModelScope) @ 80ef06163dc47ac7755a2a4cf7e398dc7b198500 |
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

vllm serve /models/Qwen3.5-35B-A3B-FP8 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --served-model-name Qwen3.5-35B-A3B-FP8 \
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

- MTP3 uses graph sizes scaled for three draft tokens.
- The 8,192-token scheduler budget supports the selected profile.
- Replace `/models/Qwen3.5-35B-A3B-FP8` if needed.

Return to the [Qwen recipe index](README.md).
