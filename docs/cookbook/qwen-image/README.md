# Qwen-Image Cookbook

Text-to-image recipes. These serve a **diffusion** pipeline through vLLM-Omni, so
they differ from the rest of the cookbook in three ways: the server needs the
`--omni` flag, requests go to `/v1/images/generations` instead of
`/v1/chat/completions`, and the image is the Omni image built by
`docker/build_omni_image.sh` rather than a `vllm-openai` release tag.

Back to the [full serving cookbook](../README.md).

| Model | Precision | Steps | Hardware | Recipe |
|---|---|---|---|---|
| Qwen-Image-2.1 | BF16 | 20 | 1x S5000, TP1 | [Open recipe](qwen-image-2.1-bf16.md) |
| Qwen-Image-2.1 | FP8 | 20 | 1x S5000, TP1 | [Open recipe](qwen-image-2.1-fp8.md) |
| Qwen-Image-2.1-Turbo | BF16 | 8 | 1x S5000, TP1 | [Open recipe](qwen-image-2.1-turbo-bf16.md) |
| Qwen-Image-2.1-Turbo | FP8 | 8 | 1x S5000, TP1 | [Open recipe](qwen-image-2.1-turbo-fp8.md) |

## Choosing a recipe

- **BF16** is the reference. Use it when the output must match what the
  checkpoint produces elsewhere.
- **FP8** quantizes the DiT's linears at load time from the same BF16 checkpoint
  and serves it faster. It changes the image: over a 14-prompt sweep some outputs
  were indistinguishable from BF16 and about half moved enough to see. Use it for
  throughput, not when a pinned prompt and seed must keep reproducing a known
  image.
- **Turbo** is a different, distilled checkpoint that carries its own 8-step
  schedule. It is the largest win available here and its output is its own, not
  an approximation of the base model's.
- The two stack: Turbo plus FP8 is the fastest of the four.

Latency depends on resolution, step count, guidance and host, so these pages
document configuration rather than published numbers. Measure your own prompt
set and request shape before committing to a profile.

Both checkpoints are released under the Qwen Research License. Read it before
deploying either one; neither is under a permissive licence.

## Building the Omni image

All four recipes run in the same image. Build it once from a pinned vLLM-MUSA
v0.32 base:

```bash
export VLLM_MUSA_IMAGE=<your pinned vLLM-MUSA v0.32 image>
bash docker/build_omni_image.sh --omni-ref origin/main
```

The default tag is `vllm-omni-musa:<omni-sha>`. Pin `--omni-ref` to a known
revision for a reproducible image; vLLM-Omni tracks its own main and is usually
newer than the vLLM commit this branch pins. `docs/omni_compat.md` describes the
shim that absorbs that drift and how to check it.

## Verify a server

```bash
curl -s -X POST http://localhost:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a red bicycle leaning on a brick wall","size":"1024x1024","seed":42}' \
  | python3 -c 'import base64,json,sys; open("out.png","wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

A pass is a decoded PNG of the requested size whose content matches the prompt.
A `200` alone is not a pass.
