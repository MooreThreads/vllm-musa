# DiffusionGemma-26B-A4B-it

## Overview

Block-diffusion text model (`DiffusionGemmaForBlockDiffusion`, ≈25.8B parameters) served by
vLLM-MUSA. A request does not decode token by token: it denoises a whole canvas of
`canvas_length` tokens in `max_denoising_steps` steps and commits the canvas at once.

The published image also carries the **structured-read interposer** from upstream
[vllm#57250](https://github.com/vllm-project/vllm/pull/57250) on port 18011, so a caller can ask
questions with fixed options and read the answer distribution instead of parsing free text.
`/v1/systemone` is provided by that interposer, not by vLLM itself.

> [!TIP]
> Start from the published image
> `registry.mthreads.com/mcconline/inference/vllm/vllm-openai:jev`. One container serves both
> endpoints and its entrypoint waits for the model before starting the interposer.

## At a glance

| Setting | Value |
|---|---|
| Checkpoint | `google/diffusiongemma-26B-A4B-it` |
| Hardware | 1x S5000 (TP1) |
| Speculative decoding | Off |
| Diffusion canvas | 256 tokens, 48 denoising steps (values from the checkpoint) |
| Maximum context | 4,096 tokens (recipe value) |
| Attention backend | `TRITON_ATTN`, eager execution |
| Generation API | port 8000 (`/v1/chat/completions`) |
| Structured-read API | port 18011 (`/v1/systemone`) |

## Launching the server

### Option A — the published image

```bash
docker run -d --name diffusiongemma \
  --runtime=mthreads -e MTHREADS_VISIBLE_DEVICES=0 \
  -v /models:/models:ro -p 8000:8000 -p 18011:18011 \
  -e MODEL=/models/diffusiongemma-26B-A4B-it \
  registry.mthreads.com/mcconline/inference/vllm/vllm-openai:jev
```

If the checkpoint is mounted elsewhere, update both the mount and `MODEL`. On a shared host, `8000`
is often already taken — remap the host side (`-p 18091:8000`) and use that port in the client
commands below; the interposer port `18011` must stay free for the CLI to work as written.

The entrypoint starts `vllm serve`, waits until `/v1/models` answers (~5 minutes for this
checkpoint on one S5000), then starts the interposer. Overrides: `MODEL`, `SERVED_NAME`, `PORT`,
`API_PORT`, `CANVAS`, `DENOISE_STEPS`, `GMEM` (default 0.75), `MAX_LEN`, `BACKEND`.

### Option B — from an installed vLLM-MUSA

Requires the structured-read port from MooreThreads/vllm-musa#249 on top of `v0.28.0`: this branch
does not carry it, and without it the four `diffusion_*` request knobs have no reader (the request is
accepted and silently read from an unseeded canvas, so `/v1/systemone` would return a well-formed but
wrong distribution). Use the `:jev` image above, or apply #249 before starting the interposer.

```bash
export TORCHDYNAMO_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True

vllm serve /models/diffusiongemma-26B-A4B-it \
  --served-model-name diffusiongemma-26b \
  --host 0.0.0.0 --port 8000 \
  --attention-backend TRITON_ATTN \
  --enforce-eager \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.75 \
  --max-logprobs 32 \
  --diffusion-config '{"canvas_length": 256, "max_denoising_steps": 48}'
```

`--max-logprobs 32` matches upstream's own example for this interposer, and a wide schema needs it:
the interposer asks for `logprobs` **and** for the logprobs of every label token in the schema, and
the pinned request path requires `logprobs == len(logprob_token_ids)` while also bounding `logprobs`
by `max_logprobs` (default 20). A `choice` question with 25 alternatives — the interposer allows up
to 26 — therefore needs a cap of at least 25, which the image's entrypoint sets to 32.

Then start the interposer against it (the file is upstream's `examples/features/structured_diffusion/structured_server.py`,
also at `/opt/jev/structured_server.py` inside the image):

```bash
python structured_server.py --upstream http://127.0.0.1:8000 \
  --model diffusiongemma-26b --tokenizer /models/diffusiongemma-26B-A4B-it \
  --canvas 256 --host 0.0.0.0 --port 18011
```

## Generating text

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "diffusiongemma-26b",
  "messages": [{"role": "user", "content": "Is the sky blue during a clear day?"}],
  "max_tokens": 256
}'
```

## Structured reads

A question set, the allowed labels for each question, and the answer template travel in the
system prompt; the client sends the situation and the questions. Labels must be **single tokens**,
so map long option names to `A/B/C/D` on the client side.

```bash
curl -s http://localhost:18011/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "diffusiongemma-26b",
  "state": "Our nightly billing job double-charged three customers; finance is asking for a fix.",
  "questions": {
    "urgent":   {"type": "noul",   "instructions": "Does the customer need a reply within the hour?"},
    "category": {"type": "choice", "instructions": "Which queue should this ticket go to?",
                 "criteria": {"billing": "invoices and charges", "outage": "service down",
                              "security": "credentials or data exposure", "feature": "a product request"}},
    "severity": {"type": "score",  "instructions": "How severe is this issue?",
                 "criteria": ["none", "mild", "serious", "critical"]}
  }
}'
```

Abbreviated response (values from a validation run; the server averages a few independent reads into
one distribution):

```json
{"answers": {
   "urgent":   {"type": "noul", "noul": 0.956},
   "category": {"type": "choice", "choice": "billing", "confidence": 0.998,
                "probabilities": {"billing": 0.998, "outage": 0.002,
                                  "security": 0.000, "feature": 0.000}},
   "severity": {"type": "score", "score": 2.92, "confidence": 0.921,
                "legend": {"0": "none", "1": "mild", "2": "serious", "3": "critical"},
                "probabilities": {"0": 0.0, "1": 0.0, "2": 0.079, "3": 0.921}}},
 "diagnostics": {"steps": 1, "timing": {"reads": 4, "total_ms": 3657}},
 "model": "diffusiongemma-26b"}
```

For a `score` question, `score` is the expectation over the ordered scale you sent (`Σ i·pᵢ`, so
`2.92` here means "between serious and critical"), and `legend` maps the string index back to your
labels; `diagnostics.timing.reads` reports how many reads were averaged.

**Treat the numbers as one sample, not as constants.** Across repeated runs of the same request the
`choice` and `score` answers were stable (`billing` at 0.998, `serious`/`critical` around 0.92), while
the yes/no margin moved a lot — the same `urgent` question came back at 0.36, 0.79 and 0.96 on three
runs. Use `--samples` (or `--extra auto_max=...`) to average more reads when a yes/no margin has to
carry a decision, and treat a single yes/no read as a weak signal.

All questions of a stage are read in one **joint canvas**, so three questions cost one read instead
of three; the server still averages several draws per question (`samples`, default auto), and
`diagnostics.timing.reads` is how many passes that took (4 in the example above). The image ships a
CLI for this (`/opt/jev/systemone.py`, also usable standalone; stdlib only):

```bash
systemone.py --endpoint http://localhost:18011 \
  --state "Everything is down and we have a demo at noon." \
  --yesno urgent "Does the customer need a reply within the hour?"
#   urgent   yes   p=0.941  p(yes)=0.941  [reads=4]
```

`--quiet` prints `id<TAB>answer<TAB>confidence` for scripts, `--json` prints the raw response.
Two practical notes: set `--no-proxy` when the client host exports `http_proxy`, otherwise the call
goes through the proxy and fails with a 502 (the same applies to the `curl` examples above — add
`--noproxy '*'`); and if the very first `/v1/systemone` call fails right after the interposer comes up,
retry once before debugging.

## Per-request diffusion knobs

These travel in `vllm_xargs` (or `SamplingParams(extra_args=...)` in-process):

| Key | Meaning |
|---|---|
| `diffusion_canvas_length` | canvas width for this request, `<= canvas_length` of the model |
| `diffusion_seed_canvas` | seed tokens for the canvas, one id per position, in vocabulary |
| `diffusion_pinned` | positions that keep their seed token **as input** |
| `diffusion_read_only` | read the canvas without generating further steps |
| `diffusion_max_steps` | denoising steps to run (`1` for a pure read) |

`diffusion_pinned` fixes what the model sees, not what it answers: the emitted tokens are the
model's per-step argmax at each position. Seeding a canvas and reading it back is a protocol for
conditioning a decision, not a way to get tokens echoed verbatim.

## Configuration notes

- Eager execution with `TRITON_ATTN` is what this recipe was validated with; keep it as the
  baseline before trying graph capture.
- Leave headroom on the card: the canvas materialises per-step logits.
  `--gpu-memory-utilization 0.90` OOMs at canvas 256 on one S5000; the recipe values used for
  validation were 0.75 and 0.80.
- `PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True` is set by the image entrypoint.
- The canvas is the unit of work, and `max_tokens` only bounds it: the model can stop early, so the
  same short prompt returned 247 tokens at `max_tokens: 256` and 15 tokens at `max_tokens: 64` with
  `finish_reason: stop`. Set `diffusion_canvas_length` when you actually want a narrower canvas.

## Validation and known limits

Validated on one S5000 (MUSA 5.2.0, `torch`/`torch_musa` 2.11.0.post1+musa5.2.0, vLLM-MUSA
v0.28.0) with image digest `sha256:f5ff4913…`:

- upstream's read tests pass inside the image: **64 passed, 0 failed**, with no `xfail` — the four
  `test_read_emits_at_convergence_while_generation_waits_for_commit` cases run unmodified, because the
  ported sampling step emits a converged read in place instead of the host loop re-implementing it;
- both endpoints were exercised end to end, including a three-question read in one canvas and a
  `choice` question with 25 alternatives (the interposer's limit is 26) through `/v1/systemone`;
- the entrypoint's `--max-logprobs 32` is what makes that wide schema legal: `logprobs` is bounded by
  `max_logprobs` and must equal the label-id union, so 25 alternatives exceed the pinned default of 20
  (`--max-logprobs` is a request-level cap; `logprob_token_ids` has its own limit of 128);
- canvas narrowing, seeding, pinning, `read_only` and `max_steps` were exercised, and out-of-vocabulary
  seeds are rejected at admission.

A yes/no margin is per-run: the same `urgent` question has come back at 0.36, 0.79, 0.722 and 0.96.

Not covered here: TP>1, concurrent requests, image/vision inputs, and canvases other than the
model's own 256. The image is built by layering the structured-read port (PR
[MooreThreads/vllm-musa#249](https://github.com/MooreThreads/vllm-musa/pull/249)) on the v0.28.0
release image, so rebuild once that PR is merged rather than keeping the layered artifact.

Return to the [cookbook index](../README.md).
