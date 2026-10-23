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
- The registry tag is published independently of this branch and may carry a
  different dependency revision; verify package versions, or build the local
  image described in the installation guide for exact branch parity.
- Mount the checkpoint under `/models` or update the path in the command.
- Use the tensor-parallelism size documented in each recipe; increase TP only
  when memory requires it.

## OCR Models

See the [OCR cookbook index](ocr/README.md) for the three model recipes.

| Model | Precision | Hardware | Recipe |
|---|---|---|---|
| MinerU2.5-Pro-2605-1.2B | BF16 | 1x S5000, TP1 | [Open recipe](ocr/mineru2.5-pro-2605-1.2b.md) |
| OvisOCR2 | BF16 | 1x S5000, TP1 | [Open recipe](ocr/ovisocr2-bf16.md) |
| PaddleOCR-VL-1.6 | BF16 | 1x S5000 recognition, TP1 | [Open recipe](ocr/paddleocr-vl-1.6-bf16.md) |

## Qwen

| Model | Precision | Hardware | Speculative decoding | Recipe |
|---|---|---|---|---|
| Qwen3-8B | FP8 | 1x S5000, TP1 | Off | [Open recipe](qwen/qwen3-8b-fp8.md) |
| Qwen3-VL-8B-Instruct | BF16 | 1x S5000, TP1 | Off | [Open recipe](qwen/qwen3-vl-8b-instruct.md) |
| Qwen3-VL-8B-Instruct | FP8 | 1x S5000, TP1 | Off | [Open recipe](qwen/qwen3-vl-8b-instruct-fp8.md) |
| Qwen3-VL-30B-A3B-Instruct | BF16 | 4x S5000, TP4 | Off | [Open recipe](qwen/qwen3-vl-30b-a3b-instruct.md) |
| Qwen3-VL-30B-A3B-Instruct | FP8 | 4x S5000, TP4 | Off | [Open recipe](qwen/qwen3-vl-30b-a3b-instruct-fp8.md) |
| Qwen3-VL-32B-Instruct | BF16 | 4x S5000, TP4 | Off | [Open recipe](qwen/qwen3-vl-32b-instruct.md) |
| Qwen3-VL-32B-Instruct | FP8 | 4x S5000, TP4 | Off | [Open recipe](qwen/qwen3-vl-32b-instruct-fp8.md) |
| Qwen3.5-27B | BF16 | 2x S5000, TP2 | MTP1 | [Open recipe](qwen/qwen3.5-27b-bf16.md) |
| Qwen3.5-27B | FP8 | 2x S5000, TP2 | Off | [Open recipe](qwen/qwen3.5-27b-fp8.md) |
| Qwen3.5-35B-A3B | BF16 | 4x S5000, TP4 | MTP3 | [Open recipe](qwen/qwen3.5-35b-a3b-bf16.md) |
| Qwen3.5-35B-A3B | FP8 | 4x S5000, TP4 | MTP3 | [Open recipe](qwen/qwen3.5-35b-a3b-fp8.md) |
| Qwen3.5-122B-A10B | BF16 | 8x S5000, TP8 | MTP3 | [Open recipe](qwen/qwen3.5-122b-a10b-bf16.md) |
| Qwen3.5-397B-A17B | FP8 | 8x S5000, TP8 | Off | [Open recipe](qwen/qwen3.5-397b-a17b-fp8.md) |
| Qwen3.6-27B | BF16 | 2x S5000, TP2 | MTP3 | [Open recipe](qwen/qwen3.6-27b-bf16.md) |
| Qwen3.6-27B | FP8 | 2x S5000, TP2 | MTP3 | [Open recipe](qwen/qwen3.6-27b-fp8.md) |
| Qwen3.6-35B-A3B | BF16 | 4x S5000, TP4 | MTP2 | [Open recipe](qwen/qwen3.6-35b-a3b-bf16.md) |
| Qwen3.6-35B-A3B | FP8 | 4x S5000, TP4 | MTP2 | [Open recipe](qwen/qwen3.6-35b-a3b-fp8.md) |

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

The Nemotron profile serves one request at a time and targets per-token decode
latency.

## DiffusionGemma

| Model | Precision | Hardware | Speculative decoding | Recipe |
|---|---|---|---|---|
| DiffusionGemma-26B-A4B-it | BF16 | 1x S5000, TP1 | Off | [Open recipe](diffusiongemma/diffusiongemma-26b-a4b-it.md) |

The recipe uses the checkpoint's own canvas (256 tokens, 48 steps) and the
`/v1/systemone` endpoint that vLLM v0.32 ships, so one server answers both
`/v1/chat/completions` and structured reads.

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
