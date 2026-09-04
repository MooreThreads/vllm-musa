# Qwen serving recipes

Qwen recipes use the hardware and tensor-parallel size listed on each page.
The current validated profiles use TP1 for Qwen3-8B, TP2 for Qwen3.5/3.6-27B,
and TP4 for Qwen3.5/3.6-35B-A3B. Choose the page that matches both the model
version and weight precision.

> [!NOTE]
> FP8 and BF16 checkpoints have different memory and speculative-decoding
> settings. Do not reuse a command across precisions without updating the
> model-specific options.

## Dense models

| Model | Precision | Speculative decoding | Recipe |
|---|---|---|---|
| Qwen3-8B | FP8 | Off | [Open recipe](qwen3-8b-fp8.md) |
| Qwen3.5-27B | BF16 | MTP1 | [Open recipe](qwen3.5-27b-bf16.md) |
| Qwen3.5-27B | FP8 | Off | [Open recipe](qwen3.5-27b-fp8.md) |
| Qwen3.6-27B | BF16 | MTP3 | [Open recipe](qwen3.6-27b-bf16.md) |
| Qwen3.6-27B | FP8 | MTP3 | [Open recipe](qwen3.6-27b-fp8.md) |

## MoE models

| Model | Precision | Speculative decoding | Recipe |
|---|---|---|---|
| Qwen3.5-35B-A3B | BF16 | MTP3 | [Open recipe](qwen3.5-35b-a3b-bf16.md) |
| Qwen3.5-35B-A3B | FP8 | MTP3 | [Open recipe](qwen3.5-35b-a3b-fp8.md) |
| Qwen3.6-35B-A3B | BF16 | MTP2 | [Open recipe](qwen3.6-35b-a3b-bf16.md) |
| Qwen3.6-35B-A3B | FP8 | MTP2 | [Open recipe](qwen3.6-35b-a3b-fp8.md) |

## Common deployment choices

| Setting | Recommended value |
|---|---|
| Hardware | Recipe-specific (1x, 2x, or 4x S5000) |
| Tensor parallelism | Recipe-specific (TP1, TP2, or TP4) |
| Speculative decoding | Use the profile listed in the recipe (Off, MTP1, MTP2, or MTP3) |
| API | OpenAI-compatible server on port 8000 |
| Model path | `/models/<checkpoint>` |

The `qwen3_5_mtp` value used by some validated commands is the backend method
name; keep it exactly as written. The path, tensor parallelism, and MTP mode
are profile-specific and should be copied together from one recipe page.

See the [cookbook index](../README.md) for client usage and the DeepSeek
recipe.
