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

- vLLM-MUSA with vLLM v0.32.0 and
  `vllm.v1.worker.gpu.sample.no_repeat_ngram.NoRepeatNGramState` available.
- Model Runner V2 with speculative decoding off.
- The MinerU2.5-Pro-2605-1.2B checkpoint available at `MODEL_PATH`.
- `mineru-vl-utils` installed wherever the MinerU client runs.

## Validation

| Item | Value |
|---|---|
| Checkpoint | [`opendatalab/MinerU2.5-Pro-2605-1.2B`](https://modelscope.cn/models/opendatalab/MinerU2.5-Pro-2605-1.2B) (ModelScope) @ a479e052d985bddc3d3fd7a9c9a4dac1edb94797 |
| vLLM-MUSA | vLLM 0.32.0 @ 5df63a7203cdb9328d824a27b682e77c7b9928be, vllm-musa 0.1.32 |
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
  --logits-processors vllm.v1.worker.gpu.sample.no_repeat_ngram:NoRepeatNGramState \
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

# Preserve MinerU's complete generated-history search within this server's
# 8,192-token context. Adapt only this client instance's outgoing requests.
build_request_body = client.client.build_request_body

def build_request_with_ngram_window(*args, **kwargs):
    body = build_request_body(*args, **kwargs)
    extra = body.get("vllm_xargs", {})
    if extra.get("no_repeat_ngram_size") is not None:
        extra["window_size"] = 8192
    return body

client.client.build_request_body = build_request_with_ngram_window

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
- MinerU's [official client defaults](https://github.com/opendatalab/mineru-vl-utils/blob/c3b3446901697a1c9e3b98ea6186ccf08834441f/mineru_vl_utils/mineru_client.py#L36-L46)
  use `no_repeat_ngram_size=100`. Its original processor retains generated
  history in a cache. The shared V2 processor defaults to a 100-token search
  window, so the example explicitly sends `window_size=8192` to cover all
  generated history within this recipe's context. This window is an adapter
  setting, not an additional official MinerU model parameter. Increase it if
  you increase the context limit.
- Keep MinerU's own NGram field and sampling defaults.
- MUSA selects `FLASH_ATTN` for ViT and `MMEncoderAttention` automatically.
  Do not add the language-model `--attention-config` option to select it.
- The image processor uses the checkpoint's own defaults. Do not add a pixel
  override to this profile.
- Keep BF16, TP1, 0.80 GPU memory utilization, context 8,192, 2,048 batched
  tokens, prefix caching on, chunked prefill on, and
  `VLLM_COMPILE`/`FULL_AND_PIECEWISE` together.

Return to the [OCR recipe index](README.md).
