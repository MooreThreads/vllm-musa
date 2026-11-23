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

## Prerequisites

- vLLM-MUSA with vLLM v0.32.0 and
  `vllm.v1.worker.gpu.sample.no_repeat_ngram:NoRepeatNGramState` available.
- Model Runner V2. This processor does not support speculative decoding.

## Validation

| Item | Value |
|---|---|
| Checkpoint | [`deepseek-ai/DeepSeek-OCR`](https://modelscope.cn/models/deepseek-ai/DeepSeek-OCR) (ModelScope) @ 99e7e2063f5efd86e50e69c41d37f53b1438caad |
| vLLM-MUSA | vLLM 0.32.0 @ 5df63a7, vllm-musa 0.1.32 |
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
  --logits-processors vllm.v1.worker.gpu.sample.no_repeat_ngram:NoRepeatNGramState \
  --no-enable-prefix-caching
```

## Send an image

Load the processor once at server startup, then enable it for each request.
The model's [official example](https://github.com/deepseek-ai/DeepSeek-OCR/blob/09eaf526153e7a01ed16c9dea8c96282aaea29c0/README.md#L151-L155)
uses N=30, a 90-token window, and the table-token whitelist below.

```python
import base64
from pathlib import Path

from openai import OpenAI

image = base64.b64encode(Path("<IMAGE_PATH>").read_bytes()).decode()
client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8000/v1")
response = client.chat.completions.create(
    model="DeepSeek-OCR",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{image}",
            }},
            {"type": "text", "text": "Free OCR."},
        ],
    }],
    temperature=0.0,
    extra_body={
        "skip_special_tokens": False,
        "vllm_xargs": {
            "ngram_size": 30,
            "window_size": 90,
            "whitelist_token_ids": [128821, 128822],
        },
    },
)
print(response.choices[0].message.content)
```

## Configuration notes

- `--mm-processor-cache-gb 0` disables the multimodal processor cache, which
  keeps memory predictable for image-heavy OCR traffic.
- The native V2 processor is selected explicitly by `--logits-processors`.
  Requests without `ngram_size` or `no_repeat_ngram_size` pass through without
  NGram masking. Do not supply both field names in one request.
- The original upstream V1 processor remains available for Model Runner V1;
  this recipe does not select or change the runner.
- The 8,192-token batched budget admits a full document page in one prefill.
- Vision attention routes through the Triton backend by default on this
  branch; no backend flag is required.
- Replace `/models/DeepSeek-OCR` if needed.

Return to the [OCR cookbook index](README.md).
