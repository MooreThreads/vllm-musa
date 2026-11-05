# Qwen3.5-122B-A10B-FP8

## Overview

FP8 MoE recipe for eight S5000 GPUs, with launch commands for text-only and
multimodal deployments.

> [!TIP]
> Use TP8 and enable MTP3 speculative decoding for this checkpoint, and pick
> the profile that matches your workload: text-only for plain chat
> completions, multimodal when requests carry image or video inputs.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 8x S5000 |
| Precision | FP8 |
| Architecture | MoE, 122B-A10B |
| Tensor parallelism | TP8 |
| Speculative decoding | MTP3 |
| Maximum context | 8,192 tokens |
| Maximum sequences | 128 |

## Validation

| Item | Value |
|---|---|
| Checkpoint | [`Qwen/Qwen3.5-122B-A10B-FP8`](https://modelscope.cn/models/Qwen/Qwen3.5-122B-A10B-FP8) (ModelScope) @ cb193be84ec5f1ba121e7c558c6ee6885c03cffc |
| vLLM-MUSA | vLLM 0.28.0 @ 2cf0a69, vllm-musa 0.1.28 |
| MUSA driver | 5.2.0 |
| Hardware | 8x S5000 |
| Served request | 2K/1K, 3K/1K, 3.5K/1K, 4K/1K, 4K/1.5K, 500 prompts each |

## Text-only profile

### Launching the server

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export SAFETENSORS_FAST_GPU=1

CAPTURE_SIZES="$(seq -s, 4 4 512)"

vllm serve /models/Qwen3.5-122B-A10B-FP8 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --served-model-name Qwen3.5-122B-A10B-FP8 \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.8 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --attention-backend FLASH_ATTN \
  --compilation-config "{\"mode\":\"NONE\",\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${CAPTURE_SIZES}]}" \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":3}'
```

## Multimodal profile

### Launching the server

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export VLLM_MUSA_QWEN3_VL_VISION_CUDAGRAPH=1
export SAFETENSORS_FAST_GPU=1

CAPTURE_SIZES="$(seq -s, 4 4 512)"

vllm serve /models/Qwen3.5-122B-A10B-FP8 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --served-model-name Qwen3.5-122B-A10B-FP8 \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.8 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --attention-backend FLASH_ATTN \
  --mm-encoder-attn-backend FLASH_ATTN \
  --mm-processor-cache-type shm \
  --mm-processor-cache-gb 8 \
  --mm-shm-cache-max-object-size-mb 128 \
  --compilation-config "{\"mode\":\"NONE\",\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${CAPTURE_SIZES}]}" \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":3}'
```

## Configuration notes

- The memory-utilization target is 0.8, matching the BF16 sibling profile.
- `--mm-encoder-attn-backend FLASH_ATTN` routes the multimodal encoder's
  attention through FlashAttention; the text-only profile omits it because
  no encoder runs in that scenario.
- `--mm-processor-cache-gb 8` gives the multimodal processor an 8 GiB cache
  of already-processed inputs, so repeated images skip reprocessing.
- `--mm-processor-cache-type shm` keeps that cache in shared memory so all
  processes share a single copy.
- `--mm-shm-cache-max-object-size-mb 128` bypasses the cache for objects
  larger than 128 MiB, so one large input cannot fill it.
- The `qwen3_next_mtp` value is the backend method name; keep it exactly as
  written.
- Chunked prefill and asynchronous scheduling are disabled.
- Replace `/models/Qwen3.5-122B-A10B-FP8` if needed.

Return to the [Qwen recipe index](README.md).
