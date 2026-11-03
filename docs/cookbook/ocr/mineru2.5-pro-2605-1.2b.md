# MinerU2.5-Pro-2605-1.2B

## Overview

Run this vision-language OCR model with vLLM-MUSA on one S5000 GPU. Start the
HTTP server with `vllm serve`, then use MinerU's client for the two-step
extraction pipeline.

> [!TIP]
> `MinerUClient(backend="http-client")` sends both extraction stages to the
> vLLM HTTP endpoint. A plain chat completion does not run this pipeline.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 1x S5000 |
| Precision | BF16 |
| Tensor parallelism | TP1 |
| Maximum context | 8,192 tokens |
| Maximum sequences | 128 |
| Maximum batched tokens | 2,048 |
| GPU memory utilization | 0.80 |
| Prefix cache | On |
| Chunked prefill | On |
| Compilation and CUDA graphs | `VLLM_COMPILE`, `FULL_AND_PIECEWISE` |

## Prerequisites

- vLLM-MUSA v0.28.0-dev with its patched vLLM installed.
- The MinerU2.5-Pro-2605-1.2B checkpoint available at `MODEL_PATH`.
- `mineru-vl-utils` installed wherever the vLLM engine runs for
  `MinerULogitsProcessor`, and wherever the MinerU client runs.

## Validation

| Item | Value |
|---|---|
| Checkpoint | [`opendatalab/MinerU2.5-Pro-2605-1.2B`](https://modelscope.cn/models/opendatalab/MinerU2.5-Pro-2605-1.2B) (ModelScope) @ a479e052d985bddc3d3fd7a9c9a4dac1edb94797 |
| vLLM-MUSA | vLLM 0.28.0 @ 2cf0a69, vllm-musa 0.1.28 |
| MUSA driver | 5.2.0 |
| Hardware | 1x S5000 |
| Served request | OmniBench |

## HTTP server

This entry point exposes the vLLM engine through HTTP. Replace `<MODEL_PATH>`
with the checkpoint directory.

```bash
export MODEL_PATH="<MODEL_PATH>"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name MinerU2.5-Pro-2605-1.2B \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.80 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --hf-overrides '{"tie_word_embeddings":true}' \
  --logits-processors mineru_vl_utils:MinerULogitsProcessor \
  --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE"}'
```


## Send an image

Use MinerU's asynchronous HTTP client to send an image through the two-step
pipeline and render the resulting blocks as Markdown:

```python
import asyncio
from pathlib import Path

from PIL import Image
from mineru_vl_utils import MinerUClient
from mineru_vl_utils.post_process import json2md

client = MinerUClient(
    backend="http-client",
    server_url="http://127.0.0.1:8000",
    model_name="MinerU2.5-Pro-2605-1.2B",
    image_analysis=False,
    use_tqdm=False,
)

async def request() -> None:
    with Image.open(Path("<IMAGE_PATH>")) as image:
        blocks = await client.aio_two_step_extract(image.convert("RGB"))
    print(json2md(blocks))
    if hasattr(client, "aclose"):
        await client.aclose()

asyncio.run(request())
```

## Configuration notes

- The HTTP client calls MinerU's two-step extraction method. A direct VLM
  completion does not run the complete pipeline.
- MUSA selects `FLASH_ATTN` for ViT and `MMEncoderAttention` automatically.
  Do not add the language-model `--attention-config` option to select it.
- The image processor uses the checkpoint's own defaults. Do not add a pixel
  override to this profile.
- Keep BF16, TP1, 0.80 GPU memory utilization, context 8,192, 2,048 batched
  tokens, prefix caching on, chunked prefill on, and
  `VLLM_COMPILE`/`FULL_AND_PIECEWISE` together.

Return to the [OCR recipe index](README.md).
