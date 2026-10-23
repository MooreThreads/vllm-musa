# DiffusionGemma-26B-A4B-it

## Overview

Block-diffusion text model (`DiffusionGemmaForBlockDiffusion`, 26B total / 4B active). A request
does not decode token by token: it denoises a canvas of `canvas_length` tokens in
`max_denoising_steps` steps and commits the canvas at once.

`/v1/systemone` answers typed questions about a state and returns a probability for every allowed
answer, so a caller reads a decision distribution instead of parsing text. It ships with vLLM v0.32,
so one `vllm serve` process exposes both `/v1/chat/completions` and `/v1/systemone`.

## At a glance

| Setting | Value |
|---|---|
| Image | `registry.mthreads.com/mcconline/inference/vllm/vllm-openai:jev` (rebuilt from this branch after merge; the tag currently carries the v0.28 port) |
| Checkpoint | `google/diffusiongemma-26B-A4B-it` (BF16) |
| Hardware | 1x S5000 (TP1), MUSA 5.2.0, driver `5.2.0-server` |
| Speculative decoding | Off |
| Diffusion canvas | 256 tokens, 48 denoising steps (values from the checkpoint) |
| Maximum context | 8,192 tokens (recipe value; the canvas is 256) |
| Attention backend | `TRITON_ATTN`, selected automatically on MUSA for diffusion models |
| Weights / KV cache | 48.5 GiB / 11.5 GiB at `--gpu-memory-utilization 0.86` |
| Logprob mode | `raw_logprobs` (default) or `processed_logprobs`; other modes are rejected |

`TRITON_ATTN` is pinned by `vllm_musa/platform.py` for diffusion models: a diffusion batch carries a
per-request `causal` tensor, which the MUSA FlashAttention path cannot express. An explicit
`--attention-backend FLASH_ATTN` is overridden.

## Launching the server

```bash
# The :jev tag carries the v0.28 port today; rebuild it from this branch before using this block.
docker run -d --name diffusiongemma \
  --runtime=mthreads -e MTHREADS_VISIBLE_DEVICES=0 \
  -v /models:/models:ro -p 8000:8000 \
  registry.mthreads.com/mcconline/inference/vllm/vllm-openai:jev \
  vllm serve /models/diffusiongemma-26B-A4B-it \
    --served-model-name diffusiongemma-26b \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.86 \
    --logprobs-mode raw_logprobs \
    --max-logprobs 32
```

With an installed vLLM-MUSA of this line, run the same `vllm serve …` line on the host. On a
shared host, port 8000 is often taken: remap it (`-p 18091:8000`) and use that port below.

Notes:

- `--logprobs-mode raw_logprobs` is the default; the flag is shown for explicitness.
- `--max-logprobs 32` does not bound a read: reads send `logprob_token_ids`, which is capped at 128
  ids independently.
- No `--diffusion-config` is needed; pass it only to override the checkpoint's canvas, e.g.
  `--diffusion-config '{"canvas_length": 256, "max_denoising_steps": 48}'`.
- `--enforce-eager` is not part of this recipe; the validated run is the default compiled path.

Cold start on one S5000: weights `94.9 s`, engine init `236.3 s` (compilation `79.3 s`), ready in
about six minutes.

## Generating text

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "diffusiongemma-26b",
  "messages": [{"role": "user", "content": "Is the sky blue during a clear day?"}],
  "max_tokens": 64
}'
```

Excerpt of the measured output (first line is the model's thinking slot; the completion ran to
`64` tokens):

```
thought
Yes, the sky is blue during a clear day.
```

## Structured reads

A read sends a state and a question set. Every question type returns a probability distribution
and a `confidence`; a `choice` names the winning option.

```bash
curl -s http://localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "diffusiongemma-26b",
  "state": {"alert": "disk_usage on prod-db-1 is 91% and rising 2%/hour, demo in 3 hours"},
  "questions": {
    "urgent": {"type": "noul", "instructions": "Does this need a reply within the hour?",
               "criteria": {"true": "capacity is exhausted or money is moving"}},
    "team": {"type": "choice", "instructions": "Which team should own this?",
             "criteria": {"billing": "payments and refunds",
                          "platform": "databases, disk and capacity",
                          "security": "account access"}},
    "severity": {"type": "score", "instructions": "How severe is this?",
                 "criteria": ["none", "mild", "serious", "critical"]}
  },
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

Measured response, first of three identical reads at `0.17 s` each (warm, values rounded):

```json
{
  "id": "decision-857e55f1e07037ad",
  "object": "structured_decision",
  "created": 1792665080,
  "model": "diffusiongemma-26b",
  "answers": {
    "urgent": {"type": "noul", "noul": 0.9742, "confidence": 0.9725,
               "probabilities": {"yes": 0.9742, "no": 0.0258}},
    "team": {"type": "choice", "choice": "platform", "confidence": 0.9998,
             "probabilities": {"billing": 0.000002, "platform": 0.99982,
                               "security": 0.00018}},
    "severity": {"type": "score", "score": 2.9792, "confidence": 0.9791,
                 "legend": {"0": "none", "1": "mild", "2": "serious", "3": "critical"},
                 "probabilities": {"0": 0.00002, "1": 0.00013, "2": 0.02048, "3": 0.97938}}
  },
  "diagnostics": {
    "urgent": {"argmax_is_label": true, "label_mass": 0.99832},
    "team": {"argmax_is_label": true, "label_mass": 0.99995},
    "severity": {"argmax_is_label": true, "label_mass": 0.99970}
  },
  "usage": {"input_tokens": 242, "output_tokens": 6}
}
```

| field | meaning |
|---|---|
| `noul` | probability of "yes" for a yes/no question |
| `choice` / `probabilities` | argmax option and the distribution over the options |
| `score` | expectation over the ordered levels (0-indexed), with `legend` |
| `confidence` | the winning option's probability times `label_mass` |
| `diagnostics.label_mass` | probability mass that fell on the allowed labels |

## Per-request diffusion knobs

These live in `vllm_xargs` on a normal request, not in the `/v1/systemone` body:

| field | effect |
|---|---|
| `diffusion_seed_canvas` | pre-fill the canvas with fixed token ids |
| `diffusion_pinned` | hold those positions at their seed value across denoise steps |
| `diffusion_read_only` | emit the canvas at convergence without the commit forward |
| `diffusion_max_steps` | cap denoise steps for this request |
| `diffusion_canvas_length` | run this request on a narrower canvas than the served one |

## Validation behind this recipe

One S5000, driver `5.2.0-server`, MUSA 5.2.0; `torch`/`torch_musa`
`2.11.0.post2+musa5.2.0`, `triton` `3.2.0`, `transformers` `5.19.0`, `vllm` `0.32.0`,
`vllm_musa` `0.1.32` from this branch. Checkpoint: BF16 `google/diffusiongemma-26B-A4B-it`,
11 shards, 48.10 GiB.

- `tests/v1/sample/test_diffusion_gemma_reads.py`: **21 passed** (it skips without this branch).
- One `/v1/systemone` request over `noul`, `choice` and `score`: `200`, three identical reads,
  `0.17 s` each warm.
- One `/v1/chat/completions` request: `Yes, the sky is blue during a clear day.`

Not covered: tensor parallelism, concurrent reads, vision inputs, canvases other than 256,
`--enforce-eager`, precision other than BF16.
