# Qwen3-Next-80B-A3B-Instruct

## Overview

BF16 hybrid Mamba/attention MoE recipe for eight S5000 GPUs with piecewise
CUDA graphs.

> [!TIP]
> Use TP8 for this checkpoint. The validated profile ships without MTP
> speculative decoding.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 8x S5000 |
| Precision | BF16 |
| Architecture | Hybrid Mamba/attention MoE, 80B-A3B |
| Tensor parallelism | TP8 |
| Speculative decoding | Off |
| Maximum context | 8,192 tokens |
| Maximum sequences | 128 |
| CUDAGraph mode | `PIECEWISE` |
| Prefix caching | Off |

## Validation

| Item | Value |
|---|---|
| Checkpoint | [`Qwen/Qwen3-Next-80B-A3B-Instruct`](https://modelscope.cn/models/Qwen/Qwen3-Next-80B-A3B-Instruct) (ModelScope) @ 02ac308d71328bf7c19dc77a3db266903894f84d |
| vLLM-MUSA | vLLM 0.28.0 @ 2cf0a69, vllm-musa 0.1.28 |
| MUSA driver | 5.2.0 |
| Hardware | 8x S5000 |
| Served request | 2K/1K, 3K/1K, 3.5K/1K, 4K/1K, 4K/1.5K, 500 prompts each |

## Launching the server

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export SAFETENSORS_FAST_GPU=1

vllm serve /models/Qwen3-Next-80B-A3B-Instruct \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --served-model-name Qwen3-Next-80B-A3B-Instruct \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --mamba-ssm-cache-dtype float32 \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,64,80,96,112,128]}'
```

## Configuration notes

- Piecewise CUDA graphs cover prefill and decode with the listed capture
  sizes; the compilation mode stays at its default.
- The validated profile ships without MTP speculative decoding.
- Replace `/models/Qwen3-Next-80B-A3B-Instruct` if needed.

Return to the [Qwen recipe index](README.md).
