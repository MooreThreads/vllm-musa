# GLM-5.3 FP8

## Overview

GLM-5.3 FP8 is served across two eight-GPU nodes with tensor parallelism,
pipeline parallelism, and expert parallelism. The recommended profile keeps
asynchronous scheduling enabled and captures decode-only CUDA graphs.

GLM-5.2 FP8 uses the same serving topology and compilation profile. The
checkpoint path, served model name, and no-thinking template handling are
different; see the GLM-5.2 section below.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 2 nodes, 8x S5000 per node |
| Tensor parallelism | TP8 |
| Pipeline parallelism | PP2 |
| Expert parallelism | Enabled |
| KV cache dtype | Auto |
| Maximum context | 16,384 tokens |
| Compilation mode | `NONE` |
| CUDAGraph mode | `FULL_DECODE_ONLY` |
| Capture sizes | 1, 2, 4, 8 |

## Prerequisites

- vLLM-MUSA v0.28.0-dev.
- The checkpoint mounted at `/models/zai-org/GLM-5.3` on both nodes.
- A locally created no-thinking template uploaded as
  `/tmp/glm53-template.nothink.jinja` on both nodes. GLM-5.2 uses its
  checkpoint-provided template and does not use this override.
- The same network interface and checkpoint contents on both nodes.

## GLM-5.3 no-thinking chat template

The GLM-5.3 no-thinking template is derived from the workaround documented in
the vLLM issue comment
[`vllm-project/vllm#54744`](https://github.com/vllm-project/vllm/issues/54744#issuecomment-5506461377).
Create a local file named `glm53-template.nothink.jinja` before starting the
servers. Start from the model's original GLM-5.3 chat template and apply the
exact reasoning blocks shown below; this cookbook does not depend on an
internal workspace or ticket path. Upload the file to both nodes, for example:

```bash
scp glm53-template.nothink.jinja <head-node>:/tmp/glm53-template.nothink.jinja
scp glm53-template.nothink.jinja <worker-node>:/tmp/glm53-template.nothink.jinja
```

The resulting `/tmp/glm53-template.nothink.jinja` is then passed with
`--chat-template` in both launch commands.

The behaviorally relevant parts are kept below so the `enable_thinking=false`
behavior is auditable:

```jinja
[gMASK]<sop>
{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' -%}
{%- if effective_reasoning_effort is not none -%}<|system|>Reasoning Effort: {{ effective_reasoning_effort | capitalize }}{%- endif -%}
{%- set clear_thinking = clear_thinking if clear_thinking is defined else false -%}
```

For prior assistant messages, the template separates visible text from
reasoning and emits an explicit empty thinking block when it must be hidden:

```jinja
{%- set content = visible_text(m.content) %}
{%- if m.reasoning_content is string %}
    {%- set reasoning_content = m.reasoning_content %}
{%- elif '</think>' in content %}
    {%- set reasoning_content = content.split('</think>')[0].split('<think>')[-1] %}
    {%- set content = content.split('</think>')[-1] %}
{%- endif %}
{%- if (not clear_thinking or loop.index0 > ns.last_user_index) and reasoning_content is defined -%}
{{ '<think>' + reasoning_content +  '</think>'}}
{%- else -%}
{{ '<think></think>' }}
{%- endif -%}
{%- if content.strip() -%}
{{ content.strip() }}
{%- endif -%}
```

For a new assistant turn, `enable_thinking=false` selects the same explicit
empty block instead of opening a reasoning block:

```jinja
{%- if add_generation_prompt -%}
    <|assistant|>{{- '<think></think>' if (enable_thinking is defined and not enable_thinking) else '<think>' -}}
{%- endif -%}
```

Use the template together with `enable_thinking=false` in GLM-5.3 requests.
This workaround is not required for GLM-5.2: its checkpoint template honors
`enable_thinking=false` directly.

## Environment

Run on both nodes:

```bash
export IFACE_NAME=bond0
export HEAD_HOST=<head-node-address>
export MASTER_PORT=29560
export GLOO_SOCKET_IFNAME="$IFACE_NAME"
export NCCL_SOCKET_IFNAME="$IFACE_NAME"
export MCCL_SOCKET_IFNAME="$IFACE_NAME"
export TORCH_MUSA_ARCH_LIST=31
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
```

The validated profile leaves `VLLM_MUSA_SPARSE_MLA_TILELANG`,
`VLLM_MUSA_SPARSE_INDEXER_GRAPH_EXACT_DECODE`, `MCCL_LAZY_INIT`, and
`VLLM_MOE_USE_DEEP_GEMM` unset. The exact-decode variable selects an optional
indexer provider; it is not a capture-state signal and is not required by this
profile.

## Recommended graph profile

Run on the head node:

```bash
vllm serve /models/zai-org/GLM-5.3 \
  --enable-expert-parallel \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr "$HEAD_HOST" \
  --master-port "$MASTER_PORT" \
  --trust-remote-code \
  --served-model-name glm53 \
  --max-model-len 16384 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --async-scheduling \
  --chat-template /tmp/glm53-template.nothink.jinja \
  --compilation-config \
  '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8}' \
  --host 0.0.0.0 \
  --port 8000
```

Run on the worker node with the same environment:

```bash
vllm serve /models/zai-org/GLM-5.3 \
  --enable-expert-parallel \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr "$HEAD_HOST" \
  --master-port "$MASTER_PORT" \
  --trust-remote-code \
  --served-model-name glm53 \
  --max-model-len 16384 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --async-scheduling \
  --chat-template /tmp/glm53-template.nothink.jinja \
  --compilation-config \
  '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8}' \
  --headless
```

## Eager diagnostic profile

Add `--enforce-eager` and remove `--compilation-config` from both commands.
Use this profile to isolate graph-capture failures; it is not the recommended
performance configuration.

## GLM-5.2 FP8

Use the same two-node TP8 + PP2 + EP commands and the same compilation
configuration, with these substitutions:

```text
checkpoint: /models/zai-org/GLM-5.2-FP8
served name: glm52
```

Do not pass `--chat-template /tmp/glm53-template.nothink.jinja` for GLM-5.2.
For accuracy requests, use the checkpoint template and send:

```json
{"chat_template_kwargs":{"enable_thinking":false}}
```

The recorded GLM-5.2 graph smoke used this native-template path and returned
clean answer content with zero reasoning tokens. The GLM-5.3 workaround is
needed because its original template does not read `enable_thinking`, as
documented in [vLLM issue #54744](https://github.com/vllm-project/vllm/issues/54744#issuecomment-5506461377).

## Verification

The startup log should include:

```text
Using GLM DSA MATE sparse-MLA adapter.
```

For GLM-5.2 accuracy requests, pass `enable_thinking=false` to the native
checkpoint template. GLM-5.3 deployments using the override template may pass
`clear_thinking=true`.
Keep the full response when scoring natural-language math answers; a
boxed-answer-only parser undercounts valid responses.

Return to the [cookbook index](../README.md).
