# Qwen serving recipes

Qwen recipes use one S5000 GPU whenever the checkpoint fits. Choose the page
that matches both the model version and weight precision.

> [!NOTE]
> FP8 and BF16 checkpoints have different memory and speculative-decoding
> settings. Do not reuse a command across precisions without updating the
> model-specific options.

## Dense models

| Model | Precision | Speculative decoding | Recipe |
|---|---|---|---|
| Qwen3-8B | FP8 | Off | [Open recipe](qwen3-8b-fp8.md) |
| Qwen3.5-27B | BF16 | Off | [Open recipe](qwen3.5-27b-bf16.md) |
| Qwen3.5-27B | FP8 | MTP1 | [Open recipe](qwen3.5-27b-fp8.md) |
| Qwen3.6-27B | BF16 | Off | [Open recipe](qwen3.6-27b-bf16.md) |
| Qwen3.6-27B | FP8 | MTP1 | [Open recipe](qwen3.6-27b-fp8.md) |

## MoE models

| Model | Precision | Speculative decoding | Recipe |
|---|---|---|---|
| Qwen3.5-35B-A3B | BF16 | Off | [Open recipe](qwen3.5-35b-a3b-bf16.md) |
| Qwen3.5-35B-A3B | FP8 | MTP3 | [Open recipe](qwen3.5-35b-a3b-fp8.md) |
| Qwen3.6-35B-A3B | BF16 | Off | [Open recipe](qwen3.6-35b-a3b-bf16.md) |
| Qwen3.6-35B-A3B | FP8 | MTP3 | [Open recipe](qwen3.6-35b-a3b-fp8.md) |

## Common deployment choices

| Setting | Recommended value |
|---|---|
| Hardware | 1x S5000 |
| Tensor parallelism | TP1 |
| API | OpenAI-compatible server on port 8000 |
| Model path | `/mnt/models/<checkpoint>` |

See the [cookbook index](../README.md) for client usage and the DeepSeek
recipe.
