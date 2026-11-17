# Configuration

## Runtime environment

```bash
export MUSA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PLUGINS=musa,musa_custom_ops
```

`MUSA_VISIBLE_DEVICES` is authoritative. Keep it aligned with the devices
assigned to the process; do not infer the MUSA list from a CUDA-only setting.
Its syntax is similar to `CUDA_VISIBLE_DEVICES`, but MUSA remains the source of
truth on this platform.

Optional runtime variables:

| Variable | Purpose |
|---|---|
| `VLLM_MUSA_CUSTOM_OP_USE_NATIVE` | Select the native custom-op path (default: false; vLLM-MUSA may force it on for quantized PIECEWISE graphs) |
| `VLLM_USE_V2_MODEL_RUNNER` | Force Model Runner V2 (`1`) or V1 (`0`) |
| `VLLM_MUSA_WORKER_TERMINATION_TIMEOUT_S` | Set the V1 worker shutdown timeout (default: 4 seconds) |
| `VLLM_USE_DEEP_GEMM` | Enable DeepGEMM where supported |
| `VLLM_USE_DEEP_GEMM_E8M0` | Select E8M0 scale handling |
| `VLLM_DEEP_GEMM_WARMUP` | Control DeepGEMM warmup behavior |

Use the model recipe as the source of truth before overriding a variable.

For quantized models captured with `PIECEWISE` graphs, vLLM-MUSA can force the
native custom-op path to preserve FP8 buffer lifetime and output correctness.
Do not force `VLLM_MUSA_CUSTOM_OP_USE_NATIVE=0` for that combination unless you
have separately verified semantic output; eager, `FULL_DECODE_ONLY`, and BF16
paths are not subject to this specific override.

V0 is not supported by this release line. V1 is the default engine; supported
architectures such as Qwen3, DeepSeek-V2, and Llama can select Model Runner V2
automatically. Set
`VLLM_USE_V2_MODEL_RUNNER=0` to force V1 or `1` to force V2 when diagnosing a
model-specific issue.

## MATE Mega MoE (opt-in block-FP8 experts)

Select `--moe-backend mate_mega_moe` to use MATE's self-dispatching FP8
experts. The backend requires expert parallelism and binds its own
passthrough prepare/finalize; it is not a DeepEP backend.

```bash
vllm serve /models/Qwen3-30B-A3B-FP8 \
  --served-model-name Qwen3-30B-A3B-FP8 \
  --data-parallel-size 4 --enable-expert-parallel \
  --moe-backend mate_mega_moe \
  --max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs 64 \
  --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY"}'
```

Use a v0.32 build with this branch's `third_party/PINS` and
`requirements/musa_private.txt`. In particular, MATE must expose the split
decode/prefill entry points provided by `0.2.8.post1`; an image containing
MATE `0.2.7` is not sufficient. Building on an older image does not update
its runtime dependencies unless they are installed explicitly.

For the measured BF16 paged-FA3 geometry (head dimension 128, GQA ratio 8,
page size 64, no DCP/local window/sinks), startup loads the reachable forward,
metadata and combine variants before serving. This avoids request-time lazy
compilation when mixed batches first encounter a new query tile or split
count. It does not change attention numerics or increase the KV cache.

A fresh compilation cache can add several minutes to startup; the observed
four-rank cold startup exceeded the upstream 600-second engine-ready timeout.
Allow that one-time startup explicitly when launching the service:

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=1800
```

Keep the normal request/SLA timeouts unchanged. Reuse the candidate's compiled
MATE cache only within a compatible device/compiler/dependency fingerprint;
a startup timeout is not evidence of KV-cache exhaustion.

The supported configuration is deliberately narrow:

- Single-node EP, at most eight ranks, linear expert placement, and BF16
  activations with block-128 FP8 expert weights.
- Hidden and per-rank intermediate dimensions divisible by 256, intermediate
  dimension at most 8192, top-k at most 32, and at most 65536 tokens per rank
  after rounding the buffer capacity to a multiple of 256.
- Standard SwiGLU only: no clamp limit, custom alpha/beta, bias, LoRA,
  redundant experts, EPLB, DBO, runtime weight transfer, pipeline parallelism
  or speculative decoding.

Unsupported configurations are rejected, not silently routed to a different
experts backend. The activation clamp is rejected because the MATE decode
and prefill entry points do not implement the same clamp contract.

All ranks use the largest DP token count to choose decode (`<=128`) or
prefill; an idle rank still participates. Symmetric workspaces are shared
by layers of the same shape within one EP process group and are explicitly
destroyed before that group's transport is torn down. Do not serve two
independent in-flight forwards concurrently on the same workspace.

For functional validation, run the checked-in multi-rank adapter test on
the exact installed candidate:

```bash
torchrun --standalone --nproc_per_node=4 \
  benchmarks/fused_moe/validate_mate_mega_moe.py \
  --hidden 2048 --intermediate 768 --experts 128 --topk 8 \
  --cycles 3 --replays 8 --output /tmp/mate-validated.json
```

This checks numerics, empty/uneven ranks, the 128/129 boundary, graph
replay with changed inputs, and collective buffer teardown/recreation.
It does not replace full-model accuracy, request-lifecycle testing or
same-build isolated-node performance qualification. Keep the backend
explicitly selected until those gates are complete for the target workload.

## Native builds and ccache

The following variables are available for native extension builds:

| Variable | Purpose |
|---|---|
| `VLLM_MUSA_USE_CCACHE` | Enable or disable ccache (default: `1` when available) |
| `VLLM_MUSA_CCACHE` | Choose the ccache executable (default: first in `PATH`) |
| `VLLM_MUSA_CCACHE_DIR` | Set the cache directory (default: `<repo>/.ccache`) |
| `VLLM_MUSA_CCACHE_MAXSIZE` | Set the cache size |
| `VLLM_MUSA_REAL_MCC` | Choose the underlying MUSA compiler (default: detected `mcc`) |

When ccache is available, source installs route both the host C++ compiler and
MUSA `mcc` through it. The generated wrapper normalizes MUSA-only inputs such
as `.mu` sources for caching while preserving the flags passed to `mcc`.

Inspect cache behavior with:

```bash
ccache --zero-stats
pip install -e . --no-build-isolation -v
ccache --show-stats
```

## Third-party source clones

`setup.py` materializes the pinned upstream vLLM and FlashInfer revisions from
github.com before it builds the MUSA extensions. It fetches only the pinned
commit (`git fetch --depth 1`), not the repository history, and routes that
fetch through a GitHub mirror:

| Variable | Purpose |
|---|---|
| `VLLM_MUSA_GIT_PROXY` | Mirror serving the github.com fetch (default: `https://gh-proxy.org/`) |
| `VLLM_MUSA_GIT_PROXY_STALL_S` | Fail a mirrored attempt after this many seconds without transfer progress (default: `60`; `0` disables the guard) |
| `VLLM_MUSA_GIT_RETRY_ATTEMPTS` | Attempts per candidate (default: `8`); the mirror gets half of them |
| `VLLM_MUSA_GIT_RETRY_DELAY_S` | Base retry delay (default: `10`), multiplied by the attempt number |

The default mirror is used unless the variable says otherwise:

```bash
# default: fetch through https://gh-proxy.org/
pip install . --no-build-isolation -v

# fetch from github.com directly
VLLM_MUSA_GIT_PROXY=0 pip install . --no-build-isolation -v

# use a different mirror
VLLM_MUSA_GIT_PROXY=https://ghproxy.net/ pip install . --no-build-isolation -v
```

The value must be a prefix that serves the full repository URL, that is
`<mirror>https://github.com/<owner>/<repo>.git`. `0`, `false`, `off`, `no`,
`none`, and `disabled` disable it; an unset or empty value keeps the default. The
rewrite is passed to git for the duration of one command, so the materialized
tree keeps the canonical github.com remote and no persistent git configuration
is written. `https://gh-proxy.com/` is the second entry point of the same
service: the git endpoints are answered directly by both hosts, so the choice is
about which one the local network reaches, not about a redirect on every
request.

A mirror that fails is abandoned after half of `VLLM_MUSA_GIT_RETRY_ATTEMPTS`
(four attempts by default) and the fetch then retries github.com directly, so a
flaky link costs time rather than failing the build. An attempt that fails after
the whole stall window without receiving anything is not retried: an unreachable
host answers nothing on the second try either, so the next candidate runs
instead. A mirrored attempt also carries git's low-speed guard: when the transfer
makes no progress for `VLLM_MUSA_GIT_PROXY_STALL_S` seconds the attempt fails and
the direct fetch runs, while a steady slow transfer is left alone — a total
timeout cannot tell those apart. The direct fetch keeps no guard, like any plain
`git fetch`.

Because only the pinned commit is transferred, the build no longer downloads the
upstream history. That transfer, not the mirror, is the dominant cost of a
source install on a restricted network.

## Scheduler and graph settings

Model pages in [the cookbook](cookbook/README.md) pin the scheduler, memory,
attention, and graph settings that belong together. Change one setting at a
time and re-check startup logs before using a different profile in production.
