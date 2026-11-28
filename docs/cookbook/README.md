# vLLM-MUSA serving recipes

Copy-ready configurations for serving supported models with vLLM-MUSA on
S5000 GPUs. Revalidate the exact image, model checkpoint, and workload before
treating a profile as a production recommendation.

> [!TIP]
> Start with the recipe unchanged. Adjust the model path and served-model name
> for your deployment before changing scheduler, memory, or graph options.

The profiles use the documented 4K-input/1K-output workload shape and bounded
concurrency. Recheck first-token latency, per-token latency, and decode
interference when your traffic shape or concurrency target differs.
Recipes written for the v0.32 line follow its scheduler interface; options
removed from that line are intentionally not carried over.

## Before you start

- Run inside a release image of the same line as the page, e.g.
  `registry.mthreads.com/mcconline/inference/vllm/vllm-openai:v0.32.0` for the
  v0.32 recipes, or an equivalent installation, unless the recipe names its
  own image.
- Apart from the image a recipe names, the registry tags are published
  independently of this branch and may carry a different dependency revision;
  verify package versions, or build the local image described in the
  installation guide for exact branch parity.
- Mount the checkpoint under `/models` or update the path in the command.
- Use the tensor-parallelism size documented in each recipe; increase TP only
  when memory requires it.

## OCR Models

See the [OCR cookbook index](ocr/README.md) for the four model recipes.

| Model | Precision | Hardware | Recipe |
|---|---|---|---|
| DeepSeek-OCR | BF16 | 1x S5000, TP1 | [Open recipe](ocr/deepseek-ocr-bf16.md) |
| MinerU2.5-Pro-2605-1.2B | BF16 | 1x S5000, TP1 | [Open recipe](ocr/mineru2.5-pro-2605-1.2b.md) |
| OvisOCR2 | BF16 | 1x S5000, TP1 | [Open recipe](ocr/ovisocr2-bf16.md) |
| PaddleOCR-VL-1.6 | BF16 | 1x S5000 recognition, TP1 | [Open recipe](ocr/paddleocr-vl-1.6-bf16.md) |

## Qwen

| Model | Precision | Hardware | Speculative decoding | Recipe |
|---|---|---|---|---|
| Qwen3-8B | FP8 | 1x S5000, TP1 | Off | [Open recipe](qwen/qwen3-8b-fp8.md) |
| Qwen3.5-0.8B | BF16 | 1x S5000, TP1 | MTP1 | [Open recipe](qwen/qwen3.5-0.8b.md) |
| Qwen3.5-2B | BF16 | 1x S5000, TP1 | MTP1 | [Open recipe](qwen/qwen3.5-2b.md) |
| Qwen3.5-4B | BF16 | 1x S5000, TP1 | MTP1 | [Open recipe](qwen/qwen3.5-4b.md) |
| Qwen3.5-9B | BF16 | 2x S5000, TP2 | MTP1 | [Open recipe](qwen/qwen3.5-9b.md) |
| Qwen3-VL-8B-Instruct | BF16 | 1x S5000, TP1 | Off | [Open recipe](qwen/qwen3-vl-8b-instruct.md) |
| Qwen3-VL-8B-Instruct | FP8 | 1x S5000, TP1 | Off | [Open recipe](qwen/qwen3-vl-8b-instruct-fp8.md) |
| Qwen3-VL-30B-A3B-Instruct | BF16 | 4x S5000, TP4 | Off | [Open recipe](qwen/qwen3-vl-30b-a3b-instruct.md) |
| Qwen3-VL-30B-A3B-Instruct | FP8 | 4x S5000, TP4 | Off | [Open recipe](qwen/qwen3-vl-30b-a3b-instruct-fp8.md) |
| Qwen3-VL-32B-Instruct | BF16 | 4x S5000, TP4 | Off | [Open recipe](qwen/qwen3-vl-32b-instruct.md) |
| Qwen3-VL-32B-Instruct | FP8 | 4x S5000, TP4 | Off | [Open recipe](qwen/qwen3-vl-32b-instruct-fp8.md) |
| Qwen3.5-27B | BF16 | 2x S5000, TP2 | MTP1 | [Open recipe](qwen/qwen3.5-27b-bf16.md) |
| Qwen3.5-27B | FP8 | 2x S5000, TP2 | MTP1 | [Open recipe](qwen/qwen3.5-27b-fp8.md) |
| Qwen3.5-35B-A3B | BF16 | 4x S5000, TP4 | MTP3 | [Open recipe](qwen/qwen3.5-35b-a3b-bf16.md) |
| Qwen3.5-35B-A3B | FP8 | 4x S5000, TP4 | MTP3 | [Open recipe](qwen/qwen3.5-35b-a3b-fp8.md) |
| Qwen3-Next-80B-A3B-Instruct | BF16 | 8x S5000, TP8 | Off | [Open recipe](qwen/qwen3-next-80b-a3b-instruct.md) |
| Qwen3.5-122B-A10B | BF16 | 8x S5000, TP8 | MTP3 | [Open recipe](qwen/qwen3.5-122b-a10b-bf16.md) |
| Qwen3.5-122B-A10B | FP8 | 8x S5000, TP8 | MTP3 | [Open recipe](qwen/qwen3.5-122b-a10b-fp8.md) |
| Qwen3.5-397B-A17B | FP8 | 8x S5000, TP8 | Off | [Open recipe](qwen/qwen3.5-397b-a17b-fp8.md) |
| Qwen3.6-27B | BF16 | 2x S5000, TP2 | MTP3 | [Open recipe](qwen/qwen3.6-27b-bf16.md) |
| Qwen3.6-27B | FP8 | 2x S5000, TP2 | MTP3 | [Open recipe](qwen/qwen3.6-27b-fp8.md) |
| Qwen3.6-35B-A3B | BF16 | 4x S5000, TP4 | MTP2 | [Open recipe](qwen/qwen3.6-35b-a3b-bf16.md) |
| Qwen3.6-35B-A3B | FP8 | 4x S5000, TP4 | MTP2 | [Open recipe](qwen/qwen3.6-35b-a3b-fp8.md) |
| Qwen3.8-27B | BF16 | 2x S5000, TP2 | MTP3 | [Open recipe](qwen/qwen3.8-27b-bf16.md) |
| Qwen3.8-27B | FP8 | 2x S5000, TP2 | MTP3 | [Open recipe](qwen/qwen3.8-27b-fp8.md) |

## DeepSeek

| Model | Hardware | Recommended profile | Alternative |
|---|---|---|---|
| [DeepSeek-V4-Flash](deepseek/deepseek-v4-flash.md) | 8x S5000, TP8 | Fixed MTP4 | MTP-off |
| [DeepSeek-V4-Flash-0731](deepseek/deepseek-v4-flash-731.md) | 8x S5000 per node, TP8 | Single node, DSpark4 | 1P1D with Mooncake store |

## GLM

| Model | Precision | Hardware | Recommended profile | Recipe |
|---|---|---|---|---|
| GLM-5.3 | FP8 | 16x S5000, TP8 + PP2 | FULL_DECODE_ONLY | [Open recipe](glm/glm-5.3-fp8.md) |
| GLM-5.2 | FP8 | 16x S5000, TP8 + PP2 | Same as GLM-5.3 | [Open recipe](glm/glm-5.3-fp8.md) |

GLM-5.2 and GLM-5.3 share the serving topology and graph profile. Their
no-thinking request handling is different: GLM-5.3 needs the documented
template workaround, while GLM-5.2 uses its checkpoint template with
`chat_template_kwargs: {"enable_thinking": false}`.

## Nemotron

| Model | Precision | Hardware | Speculative decoding | Recipe |
|---|---|---|---|---|
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | BF16 | 1x S5000, TP1 | MTP6 | [Open recipe](nemotron/nemotron-3.5-lightning-30b-a3b-bf16.md) |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | BF16 | 2x S5000, TP2 | MTP4 | [Open recipe](nemotron/nemotron-3.5-lightning-30b-a3b-bf16.md) |

The TP1 profile serves one request at a time with MTP6 and targets per-token
decode latency; the TP2 profile serves up to 98 concurrent sequences with
MTP4.

## DiffusionGemma

| Model | Precision | Hardware | Speculative decoding | Recipe |
|---|---|---|---|---|
| DiffusionGemma-26B-A4B-it | BF16 | 1x S5000, TP1 | Off | [Open recipe](diffusiongemma/diffusiongemma-26b-a4b-it.md) |

The recipe uses the checkpoint's own canvas (256 tokens, 48 steps) and the
`/v1/systemone` endpoint that vLLM v0.32 ships, so one server answers both
`/v1/chat/completions` and structured reads.

## Qwen-Image

Text-to-image recipes served through vLLM-Omni. See the
[Qwen-Image cookbook index](qwen-image/README.md) for the four model recipes.

| Model | Precision | Steps | Hardware | Recipe |
|---|---|---|---|---|
| Qwen-Image-2.1 | BF16 | 20 | 1x S5000, TP1 | [Open recipe](qwen-image/qwen-image-2.1-bf16.md) |
| Qwen-Image-2.1 | FP8 | 20 | 1x S5000, TP1 | [Open recipe](qwen-image/qwen-image-2.1-fp8.md) |
| Qwen-Image-2.1-Turbo | BF16 | 8 | 1x S5000, TP1 | [Open recipe](qwen-image/qwen-image-2.1-turbo-bf16.md) |
| Qwen-Image-2.1-Turbo | FP8 | 8 | 1x S5000, TP1 | [Open recipe](qwen-image/qwen-image-2.1-turbo-fp8.md) |

These differ from every other recipe here: the server takes `--omni`, requests go
to `/v1/images/generations`, and the image is built rather than pulled as a
`vllm-openai` tag. All four use the same
[Omni image build](qwen-image/README.md#building-the-omni-image) and
[VAE warmup procedure](qwen-image/README.md#warming-up-the-vae-decoder).

## Verify a server

The recipes expose an OpenAI-compatible endpoint on port 8000.

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="<served-model-name>",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0,
)
print(response.choices[0].message.content)
```

## Adding a recipe

One Markdown file per recipe at `<vendor>/<slug>.md` (lowercase, usually
`<model>-<precision>`), plus a row in the tables above — a file that no table
links is not reachable from this page.

A recipe carries four things, and the build fails without them:

- `# <model title>` — also the card title and the page `<title>`;
- `## Overview` — the first sentence becomes the card summary;
- `## At a glance` — a two-column table whose rows become the sidebar and the
  catalog filters (`Precision`, `Hardware`, `Tensor parallelism`,
  `Speculative decoding`, `Architecture`, ...);
- `## Launching the server` with a fenced block that starts the server —
  `vllm serve ...`, or `docker run ...` for the container-based profiles. The
  block that actually launches is what the copy button hands out.

The pull request that adds the file runs the same build the site uses and fails
on a missing section, a table it cannot read, a recipe the index does not link,
or an internal link that resolves to nothing. Merging publishes
<https://moorethreads.github.io/vllm-musa/> from the default branch, with the
footer naming the commit it was built from.

Preview the page locally:

```console
$ python tools/cookbook-site/build_site.py --source docs/cookbook --out _site
$ python3 -m http.server -d _site 8000
```

Build options and the complete list of checks are in
[tools/cookbook-site/README.md](https://github.com/MooreThreads/vllm-musa/blob/HEAD/tools/cookbook-site/README.md).
