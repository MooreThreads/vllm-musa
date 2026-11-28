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

- **BF16** is the reference precision for the DiT. To compare pixels against
  a reference generated with eager VAE decoding, omit the VAE compile flag.
- **FP8** quantizes the DiT's linears at load time from the same BF16 checkpoint
  and serves it faster. It changes the image, sometimes visibly. Use it for
  throughput, not when a pinned prompt and seed must keep reproducing a known
  image.
- **Turbo** is a different, distilled checkpoint that carries its own 8-step
  schedule. It is the largest win available here and its output is its own, not
  an approximation of the base model's.
- The two stack: Turbo plus FP8 is the fastest of the four.

Latency depends on resolution, step count, guidance and host; measure your own
prompt set and request shape before committing to a profile.

Both checkpoints are released under the Qwen Research License. Read it before
deploying either one; neither is under a permissive licence.

## Building the Omni image

All four performance recipes enable VAE decoder compilation. They require an
Omni revision containing [vLLM-Omni PR #11061](https://github.com/vllm-project/vllm-omni/pull/11061).
The validation tables record the previous Omni pin, `31c8986250d`, which
predates this setting. The new pin and four-mode measurements are pending.

Set `QWEN_IMAGE_OMNI_REF` to the full commit SHA containing the change and
`VLLM_MUSA_IMAGE` to the pinned vLLM-MUSA v0.32 base image. Build one image
for all four recipes:

```bash
: "${VLLM_MUSA_IMAGE:?Set VLLM_MUSA_IMAGE to the pinned vLLM-MUSA base image}"
: "${QWEN_IMAGE_OMNI_REF:?Set QWEN_IMAGE_OMNI_REF to the pinned Omni revision}"
bash docker/build_omni_image.sh --omni-ref "$QWEN_IMAGE_OMNI_REF" \
  --tag "vllm-omni-musa:$QWEN_IMAGE_OMNI_REF"
```

See [Omni compatibility](../../omni_compat.md) for the vLLM compatibility shim.

## Warming up the VAE decoder

After the server is ready, run the selected recipe's **Generating an image**
command once for every resolution you will benchmark or serve. Match the
measured request's shape and settings. The examples use CFG 1.0, explicitly
set 20 steps for the base checkpoint, and use Turbo's bundled eight-step
schedule. Decode the PNG and confirm its dimensions and prompt content.

The first decode of a new shape can spend significant time compiling. Check
the server log for `Qwen-Image-2.1: enabled VAE decoder compilation` and check
that the warmup did not log a compiler failure or eager fallback. Start timing
after warmup requests complete. For an eager VAE comparison on the same
image, omit only `--additional-config '{"qwen_image_21_compile_vae_decoder":true}'`.

## Verify a server

```bash
curl -s -X POST http://localhost:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a red bicycle leaning on a brick wall","size":"1024x1024","seed":42}' \
  | python3 -c 'import base64,json,sys; open("out.png","wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

A pass is a decoded PNG of the requested size whose content matches the prompt.
A `200` alone is not a pass.
