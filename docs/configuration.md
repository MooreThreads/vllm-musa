# Configuration

## Runtime environment

```bash
export MUSA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
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
