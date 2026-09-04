# DeepSeek-V4-Flash

## Overview

DeepSeek-V4-Flash is served on eight S5000 GPUs with tensor parallelism. This
recipe provides a fixed-MTP4 profile for decode-latency-sensitive traffic and
an MTP-off profile for deployments that do not use speculative decoding.

> [!TIP]
> Start with **MTP4** when decode latency is the priority, especially at small
> batches.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 8x S5000 |
| Tensor parallelism | TP8 |
| Attention backend | FlashMLA |
| KV cache | FP8 |
| Maximum context (MTP4) | 8,192 tokens |
| Maximum context (MTP-off) | 6,144 tokens |
| Maximum batched tokens (MTP4) | 8,192 tokens |
| Maximum batched tokens (MTP-off) | 8,195 tokens |
| Maximum sequences | 64 |
| Recommended profile | Fixed MTP4 |
| Alternative | MTP-off |

## Prerequisites

- vLLM-MUSA v0.24.0.
- Eight S5000 GPUs available to the server process.
- The checkpoint mounted at `/models/DeepSeek-V4-Flash-Base`, or an
  equivalent path substituted in the command.
- Both launch profiles use the `/models` mount and the same served model name,
  `DeepSeek-V4-Flash-Base`, so the verification request can be reused.

## Launching the server

### MTP4

This profile uses a fixed four-token MTP draft.

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SAFETENSORS_FAST_GPU=1
export PYTHONUNBUFFERED=1
export VLLM_USE_DEEP_GEMM=1
export VLLM_USE_DEEP_GEMM_E8M0=0
export VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export VLLM_MUSA_DEEPSEEK_V4_TP8_PROFILE=aggressive_long_prefill
export VLLM_MUSA_DEEPSEEK_V4_MOE_DEEPGEMM_PREFILL=1
export VLLM_MUSA_FUSED_AR_RMSNORM=0

CUDAGRAPH_CAPTURE_SIZES="$(seq -s, 5 5 320)"

vllm serve /models/DeepSeek-V4-Flash-Base \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --served-model-name DeepSeek-V4-Flash-Base \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.92 \
  --kv-cache-dtype fp8 \
  --no-enable-prefix-caching \
  --attention-backend FLASHMLA \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --compilation-config "{\"mode\":\"NONE\",\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${CUDAGRAPH_CAPTURE_SIZES}]}" \
  --async-scheduling
```

### MTP-off

Use this profile when speculative decoding is not desired. At higher
concurrency, the requested context and graph footprint can exceed available
memory on some 8x S5000 deployments. Reduce `max-num-seqs` and the graph
capture ceiling together, then revalidate before production use.

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SAFETENSORS_FAST_GPU=1
export VLLM_USE_DEEP_GEMM=1
export VLLM_USE_DEEP_GEMM_E8M0=0
export VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0

vllm serve /models/DeepSeek-V4-Flash-Base \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --served-model-name DeepSeek-V4-Flash-Base \
  --max-model-len 6144 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8195 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --attention-backend FLASHMLA \
  --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":64,"cudagraph_capture_sizes":[1,2,4,8,16,32,64],"cudagraph_copy_inputs":false}' \
  --async-scheduling
```

## Verifying the server

Both profiles use the same served model name:

| Profile | Checkpoint path in the command | Served model name |
|---|---|---|
| MTP4 | `/models/DeepSeek-V4-Flash-Base` | `DeepSeek-V4-Flash-Base` |
| MTP-off | `/models/DeepSeek-V4-Flash-Base` | `DeepSeek-V4-Flash-Base` |

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
served_model_name = "DeepSeek-V4-Flash-Base"
response = client.chat.completions.create(
    model=served_model_name,
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0,
)
print(response.choices[0].message.content)
```

## Configuration notes

- Keep TP8 for this checkpoint.
- For MTP4, `max-num-batched-tokens=8192` preserves the intended 4K-input
  workload envelope. The MTP-off profile uses `8195` as shown in its command.
- `async-scheduling` keeps the scheduler off the critical path. If combining
  this profile with pipeline parallelism or structured outputs, revalidate the
  exact deployment workload.
- The MTP4 capture ladder accounts for its four-token draft; MTP-off uses
  ordinary request batch sizes. `FULL_DECODE_ONLY` keeps graph capture focused
  on decode.
- On lower-memory S5000 variants, reduce `max-num-seqs` and the matching graph
  capture ceiling before lowering `max-num-batched-tokens`.

Return to the [cookbook index](../README.md).
