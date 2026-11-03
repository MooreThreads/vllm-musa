# Qwen3.5-27B-BF16

## Overview

BF16 dense-model recipe for two S5000 GPUs.

> [!TIP]
> Use TP2 and enable MTP1 speculative decoding for this checkpoint.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 2x S5000 |
| Precision | BF16 |
| Tensor parallelism | TP2 |
| Speculative decoding | MTP1 |
| Maximum context | 7,168 tokens |
| Maximum sequences | 64 |

## Validation

| Item | Value |
|---|---|
| Checkpoint | [`Qwen/Qwen3.5-27B`](https://modelscope.cn/models/Qwen/Qwen3.5-27B) (ModelScope) @ 8ee5435f4c700c6d8d4bb69efbf2e60c46b3e23e |
| vLLM-MUSA | vLLM 0.28.0 @ 2cf0a69, vllm-musa 0.1.28 |
| MUSA driver | 5.2.0 |
| Hardware | 2x S5000 |
| Served request | 2K/1K, 3K/1K, 3.5K/1K, 4K/1K, 4K/1.5K, 500 prompts each |

## Launching the server

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_DISABLE_COMPILE_CACHE=1
export SAFETENSORS_FAST_GPU=1

vllm serve /models/Qwen3.5-27B \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --served-model-name Qwen3.5-27B \
  --max-model-len 7168 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --attention-config '{"backend":"FLASH_ATTN"}' \
  --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,64]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

## Configuration notes

- Chunked prefill and asynchronous scheduling are disabled.
- Replace `/models/Qwen3.5-27B` if needed.

Return to the [Qwen recipe index](README.md).
