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

## Scheduler and graph settings

Model pages in [the cookbook](cookbook/README.md) pin the scheduler, memory,
attention, and graph settings that belong together. Change one setting at a
time and re-check startup logs before using a different profile in production.
