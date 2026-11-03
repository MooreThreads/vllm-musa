# NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16

## Overview

BF16 hybrid Mamba-2/attention MoE recipe for the S5000 with two validated
deployment profiles:

- **Single-GPU (TP1)** — serves one request at a time with MTP6 and is tuned
  for per-token decode latency.
- **Dual-GPU (TP2)** — admits up to 98 concurrent sequences with MTP4 for
  throughput-oriented workloads.

> [!TIP]
> Pick the profile that matches your concurrency target, and copy its launch
> command as a unit. Do not raise `max-num-seqs` on the single-GPU profile
> without revalidation.

## Prerequisites

- vLLM-MUSA v0.28.0-dev with Nemotron-3.5 support. The two log lines under
  Configuration notes confirm that the image carries it.
- torchada 0.1.95, as pinned by this branch. It ships the S5000 fused-MoE
  configuration for this checkpoint's expert shape
  (`E=128,N=1856,device_name=MTT_S5000.json`).
- The checkpoint mounted at
  `/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, or an equivalent path
  substituted in the command.

## Single-GPU profile (TP1)

### At a glance

| Setting | Value |
|---|---|
| Hardware | 1x S5000 |
| Precision | BF16 |
| Architecture | Hybrid Mamba-2/attention MoE, 30B-A3B |
| Tensor parallelism | TP1 |
| Speculative decoding | MTP6 |
| Maximum context | 6,144 tokens |
| Maximum sequences | 1 |
| Maximum batched tokens | 512 tokens |
| Mamba backend | Triton |
| Compilation mode | `3` (`VLLM_COMPILE`) |
| CUDAGraph mode | `FULL_AND_PIECEWISE` |
| Prefix caching | Off |

### Launching the server

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export SAFETENSORS_FAST_GPU=1

vllm serve /models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --served-model-name Nemotron-3.5-Lightning-30B-A3B \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 6144 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.98 \
  --no-enable-prefix-caching \
  --async-scheduling \
  --mamba-backend triton \
  --speculative-config '{"method":"mtp","num_speculative_tokens":6}' \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128]}'
```

### Configuration notes

- MTP6 drafts six tokens per step with the checkpoint's MTP head. On the
  reference 4K-input/1K-output request it had lower per-token latency than
  MTP1, MTP2, and MTP4. Acceptance depends on how predictable the output is,
  so compare MTP4 on your own traffic if it differs.
- `max-num-batched-tokens=512` splits a long prompt into 512-token prefill
  chunks. With MTP6, vLLM rounds the capture sizes up to multiples of seven, so
  the one-sequence decode step (one token plus six drafts) runs as a full
  CUDA graph. Keep the compiled path and the decode graph for serving.
- The server log should contain
  `Using configuration from .../E=128,N=1856,device_name=MTT_S5000.json` and
  `MUSA parallel FP32 router-gate GEMM is serving`. `Using default MoE config`
  means the installed torchada is older than this branch's pin; a missing
  router-gate line means the image predates Nemotron-3.5 support.
- Two startup warnings are expected with this profile:
  `max_num_scheduled_tokens is set to 512 based on the speculative decoding settings`
  and `Using default Mamba SSU config`. Keep `max-num-batched-tokens` at 512.
- Keep prefix caching disabled; vLLM enables it by default for this model. With
  prefix caching on, this one-sequence MTP6 profile can stop making progress
  on a request whose decode crosses a Mamba cache block boundary: the request
  is preempted and recomputed repeatedly and does not finish.
- A fresh container compiles MATE attention kernels during startup and on the
  first requests; the first 4K-token request can take a few minutes. MATE
  stores them under `$MATE_WORKSPACE_BASE/.cache/mate` (default
  `~/.cache/mate`). Mount a persistent volume at that path, or set
  `MATE_WORKSPACE_BASE` to a persistent directory, so a new container reuses
  them.
- Responses begin with the model's reasoning, ending in `</think>`, inside
  `message.content`; the profile sets no reasoning parser. Requests that omit
  sampling parameters use the checkpoint's defaults (temperature 1.0,
  top-p 0.95).
- The memory-utilization target of 0.98 assumes the GPU is otherwise idle.
  Lower it if other processes use the same card, and wait for a previous
  server's memory to be released before restarting.

## Dual-GPU profile (TP2)

Use this profile when the workload needs more than one concurrent request.

### At a glance

| Setting | Value |
|---|---|
| Hardware | 2x S5000 |
| Precision | BF16 |
| Architecture | Hybrid Mamba-2/attention MoE, 30B-A3B |
| Tensor parallelism | TP2 |
| Speculative decoding | MTP4 |
| Maximum context | 8,192 tokens |
| Maximum sequences | 98 |
| Maximum batched tokens | 8,192 tokens |
| Compilation mode | `3` (`VLLM_COMPILE`) |
| CUDAGraph mode | `FULL_AND_PIECEWISE` |
| Prefix caching | Off |

### Validation

| Item | Value |
|---|---|
| Checkpoint | [`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) (HF) @ a9904d24bcc1d289a1950fa9d2b978c47cf903b9 |
| vLLM-MUSA | vLLM 0.28.0 @ 2cf0a69, vllm-musa 0.1.28 |
| MUSA driver | 5.2.0 |
| Hardware | 2x S5000 |
| Served request | 2K/1K, 3K/1K, 3.5K/1K, 4K/1K, 4K/1.5K, 500 prompts each |

### Launching the server

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/usr/local/mtshmem/lib:${LD_LIBRARY_PATH:-}
export SAFETENSORS_FAST_GPU=1

CAPTURE_SIZES="$(seq -s, 5 5 490)"

vllm serve /models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --served-model-name Nemotron-3.5-Lightning-30B-A3B \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 98 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  --no-enable-chunked-prefill \
  --no-enable-prefix-caching \
  --attention-config '{"backend":"FLASH_ATTN"}' \
  --trust-remote-code \
  --mamba-ssm-cache-dtype float32 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --compilation-config "{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${CAPTURE_SIZES}]}"
```

### Configuration notes

- MTP4 drafts four tokens per step; the dual-GPU profile validated MTP4,
  while the single-GPU profile validated MTP6. Compare both on your traffic
  if it differs.
- Chunked prefill is disabled, and the 8,192-token batched budget admits a
  full 4K-token prompt in one prefill step.
- Capture sizes are multiples of five, matching the MTP4 draft width so
  decode steps run fully captured.
- The MATE kernel cache, expected log lines, and prefix-caching caveats
  described under the single-GPU profile apply here as well.

Return to the [cookbook index](../README.md).
