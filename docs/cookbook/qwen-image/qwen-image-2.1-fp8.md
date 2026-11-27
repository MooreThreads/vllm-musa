# Qwen-Image-2.1 FP8

## Overview

The same BF16 checkpoint the BF16 recipe serves, with the DiT's linear layers
quantized to FP8 at load time. No FP8 checkpoint is needed. It serves faster than
the BF16 recipe; the image changes.

> [!TIP]
> Use this profile for throughput. Do not use it where a pinned prompt and seed
> must keep reproducing a known image — see the configuration notes below.

## At a glance

| Setting | Value |
|---|---|
| Image | vLLM-Omni image built by `docker/build_omni_image.sh` |
| Hardware | 1x S5000 |
| Precision | FP8 |
| Quantization | Online per-tensor FP8 on the DiT linears, from the BF16 checkpoint |
| Tensor parallelism | TP1 |
| Denoising steps | 20 (recipe value, not a checkpoint default) |
| Resolution | 1024x1024 |
| Guidance | `true_cfg_scale` 1.0 |
| Endpoint | `POST /v1/images/generations` |

## Baseline validation

The table records the previous recipe on Omni `31c8986250d`. See the
[shared image build instructions](README.md#building-the-omni-image) for the
revision required by the VAE compile flag.

| Item | Value |
|---|---|
| Checkpoint | [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) (BF16, quantized at load) |
| vLLM-MUSA | vLLM 0.32.0, vllm-musa 0.1.32 |
| vLLM-Omni | main @ `31c8986250d` |
| Torch | `torch` / `torch_musa` 2.13.0.post1+musa5.2.0, torchada 0.1.111, Python 3.12 |
| MUSA driver | 5.2.0 |
| Hardware | 1x S5000, TP1 |
| Served request | 1024x1024, batch 1, seed 42, `true_cfg_scale` 1.0, 20 steps |

## Launching the server

```bash
export TORCHINDUCTOR_COMPILE_THREADS=1
export MUSA_VISIBLE_DEVICES=0

vllm serve /models/Qwen-Image-2.1 --omni \
  --host 0.0.0.0 \
  --additional-config '{"qwen_image_21_compile_vae_decoder":true}' \
  --port 8000 \
  --diffusion-quantization-config '{"transformer":{"method":"fp8"}}'
```

## Generating an image

Use this request to [warm up the decoder](README.md#warming-up-the-vae-decoder)
before measuring steady-state latency.

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

- **FP8 is not a lossless speed-up here.** On many prompts the image is visually
  indistinguishable from BF16; on roughly half of them the output moves enough to
  see, and at the tail the composition itself changes. Judge it on your own prompt
  set before deploying, and keep the BF16 recipe for any request whose exact
  output is part of the contract.
- **Use `fp8`, not `fp8_per_block`.** On the vLLM-Omni revision in the baseline validation
  table, the block-scaled method fails at engine start with
  `AssertionError: <built-in function getitem> is not an OpOverload` raised from
  the Inductor lowering, so there is no working block-128 configuration to serve.
- **`ignored_layers` needs a glob.** On that same revision, patterns are matched
  against the full layer prefix, so a bare module name such as `img_mlp` matches
  nothing and is silently ignored; the working form is `"*img_mlp*"`. On this
  model it is a poor trade in any case: keeping the image-stream MLPs in BF16
  improves fidelity but gives back most of the FP8 speed-up, and does not fix the
  worst prompts.
- The checkpoint is released under the Qwen Research License. Read it before
  deploying; it is not one of the permissive licences.
- Everything in the BF16 recipe's configuration notes applies here too, including
  the mandatory `TORCHINDUCTOR_COMPILE_THREADS=1` and the inert
  `enable_cuda_graph_decode` flag.

Return to the [Qwen-Image recipe index](README.md).
