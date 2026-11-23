# Qwen-Image-2.1 BF16

## Overview

Text-to-image recipe for the base Qwen-Image-2.1 checkpoint at its native BF16
precision. This is the reference configuration for the other three recipes in
this section.

> [!TIP]
> Use this profile when the output must match what the checkpoint produces
> elsewhere. The FP8 recipe in this section trades a visible fidelity cost for
> lower latency.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 1x S5000 |
| Precision | BF16 |
| Tensor parallelism | TP1 |
| Denoising steps | 20 (recipe value, not a checkpoint default) |
| Resolution | 1024x1024 |
| Guidance | `true_cfg_scale` 1.0 |
| Endpoint | `POST /v1/images/generations` |

## Validation

| Item | Value |
|---|---|
| Checkpoint | [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) (BF16) |
| vLLM-MUSA | vLLM 0.32.0, vllm-musa 0.1.32 |
| vLLM-Omni | main @ `31c8986250d` |
| Torch | `torch` / `torch_musa` 2.13.0.post1+musa5.2.0, torchada 0.1.111, Python 3.12 |
| MUSA driver | 5.2.0 |
| Hardware | 1x S5000, TP1 |
| Served request | 1024x1024, batch 1, seed 42, `true_cfg_scale` 1.0, 20 steps |

## Launching the server

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export MUSA_VISIBLE_DEVICES=0

vllm serve /models/Qwen-Image-2.1 --omni \
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
    "true_cfg_scale": 1.0,
    "num_inference_steps": 20
  }' | python3 -c 'import base64,json,sys; open("latte.png","wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

## Configuration notes

- `TORCHINDUCTOR_COMPILE_THREADS=1` is required, not a tuning choice. Inductor's
  parallel compile workers fork, and the MUSA runtime cannot be re-initialised in
  a forked subprocess, so the first cold compile dies before any image is
  produced. `VLLM_WORKER_MULTIPROC_METHOD=spawn` does not cover it: the fork is
  inside Inductor's own pool. Serialising compilation changes compile wall time
  only, not the generated code.
- Set `MUSA_VISIBLE_DEVICES` to a concrete index. The literal `all` is rejected
  downstream where the value is parsed as an integer.
- 20 steps is this recipe's choice. The checkpoint's scheduler declares no step
  count, so the number is yours to pick; latency is close to linear in it.
- `true_cfg_scale` 1.0 disables classifier-free guidance. Higher values add a
  second transformer evaluation per step, so the cost per image goes up.
- `enable_cuda_graph_decode` defaults to on and has **no effect on this
  platform**: the capture path is gated to CUDA devices, so the flag is inert.
  Leaving it at the default is correct — forcing the capture path on was measured
  slower per image, because graph replay costs more per step than eager execution
  here and each request pays a capture.
- The first start compiles the pipeline, so the server takes a few minutes to
  accept requests, longer on a cold Inductor cache. The first request after that
  is slower than the steady state.

Return to the [Qwen-Image recipe index](README.md).
