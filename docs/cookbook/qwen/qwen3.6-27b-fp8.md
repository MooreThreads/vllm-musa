# Qwen3.6-27B-FP8

## Overview

FP8 dense-model recipe with three-token MTP speculative decoding.

> [!TIP]
> Use this TP2 profile when serving the FP8 checkpoint.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 2x S5000 |
| Precision | FP8 |
| Tensor parallelism | TP2 |
| Speculative decoding | MTP3 |
| Maximum context | 8,192 tokens |
| Maximum sequences | 128 |

## Validation

| Item | Value |
|---|---|
| Checkpoint | [`Qwen/Qwen3.6-27B-FP8`](https://modelscope.cn/models/Qwen/Qwen3.6-27B-FP8) (ModelScope) @ 2585979d2934d3b55c58a5c570d3f180d793e56d |
| vLLM-MUSA | vLLM 0.28.0 @ 2cf0a69, vllm-musa 0.1.28 |
| MUSA driver | 5.2.0 |
| Hardware | 2x S5000 |
| Served request | 2K/1K, 3K/1K, 3.5K/1K, 4K/1K, 4K/1.5K, 500 prompts each |

## Launching the server

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export SAFETENSORS_FAST_GPU=1

CAPTURE_SIZES="$(seq -s, 4 4 512)"

vllm serve /models/Qwen3.6-27B-FP8 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --served-model-name Qwen3.6-27B-FP8 \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.80 \
  --mamba-ssm-cache-dtype float32 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --attention-config '{"backend":"FLASH_ATTN"}' \
  --compilation-config "{\"mode\":\"NONE\",\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${CAPTURE_SIZES}]}" \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
```

## Configuration notes

- MTP3 uses the FlashAttention backend for draft attention.
- Graph capture sizes account for the additional draft tokens.
- Replace `/models/Qwen3.6-27B-FP8` if needed.

Return to the [Qwen recipe index](README.md).
