# Qwen3.5-0.8B

## Overview

BF16 small-model recipe for a single S5000 GPU with MTP1 speculative decoding
and decode-only CUDA graphs.

> [!TIP]
> Use TP1 and enable MTP1 speculative decoding for this checkpoint.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 1x S5000 |
| Precision | BF16 |
| Tensor parallelism | TP1 |
| Speculative decoding | MTP1 |
| Maximum context | 8,192 tokens |
| Maximum sequences | 128 |
| CUDAGraph mode | `FULL_DECODE_ONLY` |
| Prefix caching | Off |

## Validation

| Item | Value |
|---|---|
| Checkpoint | [`Qwen/Qwen3.5-0.8B`](https://modelscope.cn/models/Qwen/Qwen3.5-0.8B) (ModelScope) @ 4d58a7b524cd33ed843d5125be8cd8f0a452d9bf |
| vLLM-MUSA | vLLM 0.28.0 @ 2cf0a69, vllm-musa 0.1.28 |
| MUSA driver | 5.2.0 |
| Hardware | 1x S5000 |
| Served request | 2K/1K, 3K/1K, 3.5K/1K, 4K/1K, 4K/1.5K, 500 prompts each |

## Launching the server

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export VLLM_DISABLE_COMPILE_CACHE=1
export SAFETENSORS_FAST_GPU=1

vllm serve /models/Qwen3.5-0.8B \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --served-model-name Qwen3.5-0.8B \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.85 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --attention-config '{"backend":"FLASH_ATTN"}' \
  --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,24,32,48,64,96,128]}' \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'
```

## Configuration notes

- Capture sizes are dense from 1 to 16 and sparse up to 128, so one-request
  and small-batch decode steps stay fully captured while larger batches still
  find a matching graph.
- The profile compiles in `NONE` mode and relies on CUDA graphs for decode
  only; chunked prefill stays enabled for long prompts.
- `VLLM_DISABLE_COMPILE_CACHE=1` matches the validated QA launch script.
- Replace `/models/Qwen3.5-0.8B` if needed.

Return to the [Qwen recipe index](README.md).
