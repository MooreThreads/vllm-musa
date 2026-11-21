# Qwen-Image-2.1 FP8

## Overview

The same BF16 checkpoint the BF16 recipe serves, with the DiT's linear layers
quantized to FP8 at load time. No FP8 checkpoint is needed. Latency drops about
18%; the image changes.

> [!TIP]
> Use this profile for throughput. Do not use it where a pinned prompt and seed
> must keep reproducing a known image — see the fidelity note below.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 1x S5000 |
| Precision | FP8 (online, per-tensor; BF16 checkpoint quantized at load) |
| Tensor parallelism | TP1 |
| Denoising steps | 20 (recipe value, not a checkpoint default) |
| Resolution | 1024x1024 |
| Guidance | `true_cfg_scale` 1.0 |
| Endpoint | `POST /v1/images/generations` |
| Median latency | 3.42 s per image, warm — **18.5% below BF16** |
| Cold start | about 137 s on a warm Inductor cache |

## Validation

| Item | Value |
|---|---|
| Checkpoint | [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) (BF16, quantized at load) |
| vLLM-MUSA | vLLM 0.32.0, vllm-musa 0.1.32 |
| vLLM-Omni | main @ `31c8986250d` |
| Torch | `torch` / `torch_musa` 2.13.0.post1+musa5.2.0, torchada 0.1.111, Python 3.12 |
| MUSA driver | 5.2.0 |
| Hardware | 1x S5000, TP1 |
| Served request | 1024x1024, batch 1, seed 42, `true_cfg_scale` 1.0, 20 steps |
| Result | median **3.4200 s** over five warm requests, spread 1.26%, against 4.1956 s BF16 in the same window |
| Fidelity | 14-prompt sweep against the BF16 image: PSNR median 31.34 dB (min 19.08), SSIM median 0.9704 (min 0.7883), 7 of 14 prompts below 30 dB |

## Launching the server

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export MUSA_VISIBLE_DEVICES=0

vllm serve /models/Qwen-Image-2.1 --omni \
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
    "true_cfg_scale": 1.0,
    "num_inference_steps": 20
  }' | python3 -c 'import base64,json,sys; open("latte.png","wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

## Configuration notes

- **What the fidelity numbers mean.** FP8 is not a lossless speed-up here. On
  many prompts the image is visually indistinguishable from BF16; on roughly half
  of them the output moves enough to see, and at the tail the composition itself
  changes. Judge it on your own prompt set before deploying, and keep the BF16
  recipe for any request whose exact output is part of the contract.
- **Use `fp8`, not `fp8_per_block`.** The block-scaled method fails at engine
  start on this platform with
  `AssertionError: <built-in function getitem> is not an OpOverload` raised from
  the Inductor lowering, so there is no working block-128 configuration to serve.
- **`ignored_layers` needs a glob.** Patterns are matched against the full layer
  prefix, so a bare module name such as `img_mlp` matches nothing and is silently
  ignored; the working form is `"*img_mlp*"`. On this model it is a poor trade in
  any case: keeping the image-stream MLPs in BF16 returned the median PSNR to
  34.07 dB but gave back almost all the speed (latency back to 4.05 s, 1.9% below
  BF16) without fixing the worst prompts.
- Everything in the BF16 recipe's configuration notes applies here too, including
  the mandatory `TORCHINDUCTOR_COMPILE_THREADS=1` and the inert
  `enable_cuda_graph_decode` flag.

Return to the [Qwen-Image recipe index](README.md).
