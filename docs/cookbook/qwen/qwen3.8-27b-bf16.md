# Qwen3.8-27B

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

## Validation

| Item | Value |
|---|---|
| Checkpoint | [`Qwen/Qwen3.8-27B`](https://modelscope.cn/models/Qwen/Qwen3.8-27B) (ModelScope) @ e823e888ae179eb3be02c1a48899c4f828371376 |
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

vllm serve /models/Qwen3.8-27B \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --served-model-name Qwen3.8-27B \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --attention-config '{"backend":"FLASH_ATTN"}' \
  --compilation-config "{\"mode\":\"NONE\",\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${CAPTURE_SIZES}]}" \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
```

## Configuration notes

- MTP3 drafts three tokens per step through the `qwen3_5_mtp` backend
  method; keep the method name exactly as written.
- The profile compiles in `NONE` mode and captures decode-only CUDA graphs
  from 4 to 512 in steps of four.
- Chunked prefill and asynchronous scheduling are disabled.
- Replace `/models/Qwen3.8-27B` if needed.

Return to the [Qwen recipe index](README.md).
