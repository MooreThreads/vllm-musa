# DeepSeek-V4-Flash

## Overview

DeepSeek-V4-Flash is served on eight S5000 GPUs with tensor parallelism. This
recipe provides a fixed-MTP4 profile for decode-latency-sensitive traffic and
an MTP-off profile for deployments that do not use speculative decoding.

> [!TIP]
> Start with **MTP4** when decode latency is the priority, especially at small
> batches. Keep chunked prefill enabled so long prefills do not block active
> decode requests for their full duration.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 8x S5000 |
| Tensor parallelism | TP8 |
| Attention backend | FlashMLA |
| KV cache | FP8 |
| Maximum context | 6,144 tokens |
| Maximum sequences | 64 |
| Recommended profile | Fixed MTP4 |
| Alternative | MTP-off |

## Prerequisites

- vLLM-MUSA v0.24.0.
- Eight S5000 GPUs available to the server process.
- The checkpoint mounted at `/mnt/models/DeepSeek-V4-Flash-Base`, or an
  equivalent path substituted in the command.

## Launching the server

### MTP4

This profile uses a fixed four-token MTP draft.

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SAFETENSORS_FAST_GPU=1
export VLLM_USE_DEEP_GEMM=1
export VLLM_USE_DEEP_GEMM_E8M0=0
export VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0

vllm serve /mnt/models/DeepSeek-V4-Flash-Base \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --served-model-name deepseek-v4-flash-base \
  --max-model-len 6144 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8195 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --attention-backend FLASHMLA \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":320,"cudagraph_capture_sizes":[5,10,20,40,80,160,320],"cudagraph_copy_inputs":false}' \
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

vllm serve /mnt/models/DeepSeek-V4-Flash-Base \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --served-model-name deepseek-v4-flash-base \
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

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="deepseek-v4-flash-base",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0,
)
print(response.choices[0].message.content)
```

## Configuration notes

- Keep TP8 for this checkpoint.
- Keep chunked prefill enabled. Disabling it can reduce isolated prefill
  latency, but long prefills may then cause severe decode inter-token stalls.
- `max-num-batched-tokens=8195` preserves the intended 4K-input workload
  envelope.
- `async-scheduling` keeps the scheduler off the critical path. If combining
  this profile with pipeline parallelism or structured outputs, revalidate the
  exact deployment workload.
- MTP4 uses speculative batch sizes while MTP-off uses ordinary request batch
  sizes. `FULL_DECODE_ONLY` keeps graph capture focused on decode.
- On lower-memory S5000 variants, reduce `max-num-seqs` and the matching graph
  capture ceiling before lowering `max-num-batched-tokens`.

Return to the [cookbook index](../README.md).
