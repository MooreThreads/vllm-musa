# Qwen-Image-2.1-Turbo BF16

## Overview

Qwen-Image-2.1-Turbo is a distilled checkpoint that carries its own 8-entry
`sample_sigmas` schedule in `model_index.json`. It reaches an image in 8
denoising steps instead of 20. The saving is the step count: the per-step cost is
unchanged.

> [!TIP]
> Treat Turbo as a different model rather than a faster setting: its output is
> its own, not an approximation of the base model's.

## At a glance

| Setting | Value |
|---|---|
| Image | vLLM-Omni image built by `docker/build_omni_image.sh` |
| Hardware | 1x S5000 |
| Precision | BF16 |
| Tensor parallelism | TP1 |
| Denoising steps | 8, from the checkpoint's own `sample_sigmas` |
| Resolution | 1024x1024 |
| Guidance | `true_cfg_scale` 1.0 |
| Endpoint | `POST /v1/images/generations` |

## Validation

| Item | Value |
|---|---|
| Checkpoint | [`Qwen/Qwen-Image-2.1-Turbo`](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) (BF16, 8-entry `sample_sigmas`) |
| vLLM-MUSA | vLLM 0.32.0, vllm-musa 0.1.32 |
| vLLM-Omni | main @ `31c8986250d` |
| Torch | `torch` / `torch_musa` 2.13.0.post1+musa5.2.0, torchada 0.1.111, Python 3.12 |
| MUSA driver | 5.2.0 |
| Hardware | 1x S5000, TP1 |
| Served request | 1024x1024, batch 1, seed 42, `true_cfg_scale` 1.0, 8 steps |
| Step source | an explicit `num_inference_steps: 8` and omitting the field produced a byte-identical PNG |

## Launching the server

```bash
export TORCHINDUCTOR_COMPILE_THREADS=1
export MUSA_VISIBLE_DEVICES=0

vllm serve /models/Qwen-Image-2.1-Turbo --omni \
  --host 0.0.0.0 \
  --port 8000
```

## Generating an image

```bash
curl -s -X POST http://localhost:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a photorealistic latte in a white ceramic cup on a wooden table, morning light",
    "size": "1024x1024",
    "seed": 42,
    "true_cfg_scale": 1.0
  }' | python3 -c 'import base64,json,sys; open("latte.png","wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

## Configuration notes

- **Leave the step count out.** The schedule comes from the checkpoint. Sending
  `num_inference_steps: 8` is accepted and produced a byte-identical image to
  omitting the field, so the field is redundant here; sending a *different*
  number departs from the grid the checkpoint was distilled for.
- **Turbo is not scored against the base model.** Comparing a Turbo image to a
  base-model image measures the distillation, not quality: dropping the base model
  from 20 to 8 steps already changes its output on its own, and Turbo at 8 steps
  differs from base-at-8-steps by more still. Evaluate Turbo on prompt adherence
  and preference against your own bar.
- The per-step cost is not lower than the base model's; Turbo buys *fewer* steps.
  Fixed per-request work — text encoding and VAE decode — is therefore a larger
  share of the wall here than at 20 steps.
- The checkpoint is released under the Qwen Research License. Read it before
  deploying; it is not one of the permissive licences.
- Everything in [the BF16 recipe](qwen-image-2.1-bf16.md)'s configuration notes
  applies, including the mandatory `TORCHINDUCTOR_COMPILE_THREADS=1`.

Return to the [Qwen-Image recipe index](README.md).
