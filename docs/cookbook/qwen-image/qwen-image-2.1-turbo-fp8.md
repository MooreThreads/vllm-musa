# Qwen-Image-2.1-Turbo FP8

## Overview

The distilled Turbo checkpoint with the DiT's linear layers quantized to FP8 at
load time. The two levers are independent and stack: 8 steps instead of 20, and
FP8 on the DiT linears. This is the fastest of the four profiles in this section.

> [!TIP]
> Fastest, and the furthest from the base model's output. Use it when throughput
> is the goal and the images are judged on their own merit.

## At a glance

| Setting | Value |
|---|---|
| Image | vLLM-Omni image built by `docker/build_omni_image.sh` |
| Hardware | 1x S5000 |
| Precision | FP8 |
| Quantization | Online per-tensor FP8 on the DiT linears, from the BF16 checkpoint |
| Tensor parallelism | TP1 |
| Denoising steps | 8, from the checkpoint's own `sample_sigmas` |
| Resolution | 1024x1024 |
| Guidance | `true_cfg_scale` 1.0 |
| Endpoint | `POST /v1/images/generations` |

## Validation

| Item | Value |
|---|---|
| Checkpoint | [`Qwen/Qwen-Image-2.1-Turbo`](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) (BF16, quantized at load) |
| vLLM-MUSA | vLLM 0.32.0, vllm-musa 0.1.32 |
| vLLM-Omni | main @ `31c8986250d` |
| Torch | `torch` / `torch_musa` 2.13.0.post1+musa5.2.0, torchada 0.1.111, Python 3.12 |
| MUSA driver | 5.2.0 |
| Hardware | 1x S5000, TP1 |
| Served request | 1024x1024, batch 1, seed 42, `true_cfg_scale` 1.0, 8 steps |

## Launching the server

```bash
export TORCHINDUCTOR_COMPILE_THREADS=1
export MUSA_VISIBLE_DEVICES=0

vllm serve /models/Qwen-Image-2.1-Turbo --omni \
  --host 0.0.0.0 \
  --port 8000 \
  --diffusion-quantization-config '{"transformer":{"method":"fp8"}}'
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

- **Two independent changes at once.** Against the base BF16 reference this
  recipe changes both the checkpoint and the numerics, so its output is not
  comparable to that reference image. The FP8 fidelity caveat in
  [the FP8 recipe](qwen-image-2.1-fp8.md) applies to the base checkpoint, and the
  Turbo checkpoint's own quality bar is discussed in
  [the Turbo BF16 recipe](qwen-image-2.1-turbo-bf16.md). Neither covers the
  combination.
- FP8 is worth proportionally less here than at 20 steps: with fewer denoising
  steps, the fixed per-request work that FP8 does not touch — text encoding and
  VAE decode — is a larger share of the wall.
- `fp8_per_block` does not start on the revision this recipe pins; use `fp8` as
  shown. See [the FP8 recipe](qwen-image-2.1-fp8.md) for that and for the
  `ignored_layers` glob caveat.
- The checkpoint is released under the Qwen Research License. Read it before
  deploying.
- Everything in [the BF16 recipe](qwen-image-2.1-bf16.md)'s configuration notes
  applies, including the mandatory `TORCHINDUCTOR_COMPILE_THREADS=1`.

Return to the [Qwen-Image recipe index](README.md).
