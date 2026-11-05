# DeepSeek-OCR

## Overview

BF16 vision-language OCR recipe for a single S5000 GPU.

> [!TIP]
> Use TP1 for this checkpoint. Prefix caching and the multimodal processor
> cache stay off for OCR traffic.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 1x S5000 |
| Precision | BF16 |
| Architecture | Vision-Language (OCR) |
| Tensor parallelism | TP1 |
| Speculative decoding | Off |
| Maximum context | 8,192 tokens |
| Maximum sequences | 64 |
| Prefix caching | Off |

## Validation

| Item | Value |
|---|---|
| Checkpoint | [`deepseek-ai/DeepSeek-OCR`](https://modelscope.cn/models/deepseek-ai/DeepSeek-OCR) (ModelScope) @ 99e7e2063f5efd86e50e69c41d37f53b1438caad |
| vLLM-MUSA | vLLM 0.28.0 @ 2cf0a69, vllm-musa 0.1.28 |
| MUSA driver | 5.2.0 |
| Hardware | 1x S5000 |
| Served request | OmniBench |

## Launching the server

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export SAFETENSORS_FAST_GPU=1

vllm serve /models/DeepSeek-OCR \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name DeepSeek-OCR \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --mm-processor-cache-gb 0 \
  --no-enable-prefix-caching \
  --logits-processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor
```

## Configuration notes

- `--mm-processor-cache-gb 0` disables the multimodal processor cache, which
  keeps memory predictable for image-heavy OCR traffic.
- The NGram per-request logits processor is part of the validated DeepSeek-OCR
  decoding path; keep it in the command.
- The 8,192-token batched budget admits a full document page in one prefill.
- Vision attention routes through the Triton backend by default on this
  branch; no backend flag is required.
- Replace `/models/DeepSeek-OCR` if needed.

Return to the [OCR cookbook index](README.md).
