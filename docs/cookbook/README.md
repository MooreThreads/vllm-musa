# vLLM-MUSA serving recipes

Starting configurations for serving the listed models with vLLM-MUSA v0.28.0
on S5000 GPUs. Revalidate the exact image, model checkpoint, and workload
before treating a profile as a production recommendation.

> [!TIP]
> Start with the recipe unchanged. Adjust the model path and served-model name
> for your deployment before changing scheduler, memory, or graph options.

The profiles use the documented 4K-input/1K-output workload shape and bounded
concurrency. Recheck first-token latency, per-token latency, and decode
interference when your traffic shape or concurrency target differs.
The commands follow the v0.28 scheduler interface; options removed from this
release line are intentionally not carried over.

## Before you start

- Run inside the v0.28.0 release image
  `registry.mthreads.com/mcconline/inference/vllm/vllm-openai:v0.28.0`, or an
  equivalent installation.
- The registry tag is the planned release image name; if it is not published
  yet, build and use the local image described in the installation guide.
- Mount the checkpoint under `/mnt/models` or update the path in the command.
- Keep TP1 when a model fits on one GPU; increase TP only when memory requires
  it.

## Qwen

The listed Qwen checkpoints fit on one S5000 and use TP1.

| Model | Precision | Speculative decoding | Recipe |
|---|---|---|---|
| Qwen3-8B | FP8 | Off | [Open recipe](qwen/qwen3-8b-fp8.md) |
| Qwen3.5-27B | BF16 | Off | [Open recipe](qwen/qwen3.5-27b-bf16.md) |
| Qwen3.5-27B | FP8 | MTP1 | [Open recipe](qwen/qwen3.5-27b-fp8.md) |
| Qwen3.5-35B-A3B | BF16 | Off | [Open recipe](qwen/qwen3.5-35b-a3b-bf16.md) |
| Qwen3.5-35B-A3B | FP8 | MTP3 | [Open recipe](qwen/qwen3.5-35b-a3b-fp8.md) |
| Qwen3.6-27B | BF16 | Off | [Open recipe](qwen/qwen3.6-27b-bf16.md) |
| Qwen3.6-27B | FP8 | MTP1 | [Open recipe](qwen/qwen3.6-27b-fp8.md) |
| Qwen3.6-35B-A3B | BF16 | Off | [Open recipe](qwen/qwen3.6-35b-a3b-bf16.md) |
| Qwen3.6-35B-A3B | FP8 | MTP3 | [Open recipe](qwen/qwen3.6-35b-a3b-fp8.md) |

## DeepSeek

| Model | Hardware | Recommended profile | Alternative |
|---|---|---|---|
| [DeepSeek-V4-Flash](deepseek/deepseek-v4-flash.md) | 8x S5000, TP8 | Fixed MTP4 | MTP-off |

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
