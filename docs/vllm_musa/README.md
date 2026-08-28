# vllm_musa

Python package that plugs into vLLM's platform and general plugin system to enable inference on Moore Threads MUSA GPUs (MTGPU).

## Package Entry Points

| Entry Point | Function | Purpose |
|---|---|---|
| `vllm.platform_plugins` | `musa_platform_plugin` | Registers `MUSAPlatform` when MUSA hardware is detected |
| `vllm.general_plugins` | `register_custom_ops` | Applies source patches, registers OOT ops, distributed connectors, and attention backends |
| Console script | `collect_env` | Prints MUSA environment info (`vllm_collect_env`) |

## Module Overview

### `runtime_plan/` – Unified optimization planning

Model, topology, compile, graph-capture, and request-path choices are resolved
through one immutable runtime plan. See [Runtime plans](runtime_plan.md) for the
decision catalog, lifecycle rules, built-in plan projection, and extension
workflow. See [Model-specific engine plans](engine_plan.md) for the baseline →
profile → bounded-search workflow, or start with the
[AutoTuner quickstart](autotuner_quickstart.md) for domain discovery, timing
artifacts, plan construction, and clean replay.

See [Declarative RuntimePlan profiles](runtime_profiles.md) for versioned
per-model defaults, the closed condition language, tunability ownership, and
the evidence-backed config-only update workflow.

### `musa.py` – Platform Definition

Implements `MUSAPlatform` (out-of-tree `Platform` subclass) with two variants auto-selected at import:

- **`MtmlMUSAPlatform`** – Uses MTML (pymtml) for device queries without initializing a MUSA context.
- **`NonMtmlMUSAPlatform`** – Falls back to `torch.cuda` APIs when MTML is unavailable.

Key platform capabilities: device capability/memory queries, MtLink topology detection, attention backend selection (Triton-based by default; FlashMLA for MLA models), custom all-reduce support, and CUDA-graph/static-graph support.

### `worker.py` – MTGPU Worker

`MTGPUWorker` extends vLLM's v1 `Worker` for MTGPU execution. Automatically assigned when `worker_cls == "auto"`.

### `patches/` – Runtime Source Patches

Applies targeted string replacements to upstream vLLM source files at runtime for MUSA/Triton compatibility. Each `*.patch.py` file defines a `PATCHES` list of `(old_str, new_str)` tuples targeting a specific vLLM module.

Current patches cover:
- Distributed communicators (all2all, custom all-reduce)
- FP8 quantization and utilities
- Triton unified attention, FlashMLA ops
- Top-k/top-p sampling (Triton variant)
- DeepGEMM utilities, profiler wrapper, GPU worker

### `model_executor/` – OOT Layer Implementations

Custom `forward_oot` implementations registered for MUSA dispatch:

| Module | Functionality |
|---|---|
| `layers/activation.py` | Activation functions (SiluAndMul, GeluAndMul, etc.) |
| `layers/layernorm.py` | RMSNorm / LayerNorm |
| `layers/fused_moe/` | Fused MoE dispatch and unquantized MoE method |
| `layers/quantization/fp8.py` | FP8 linear layer quantization |
| `layers/quantization/utils/fp8_utils.py` | FP8 utility functions |
| `layers/attention/mla_attention.py` | Multi-head Latent Attention (MLA) |
| `warmup/deep_gemm_warmup.py` | DeepGEMM JIT warmup |

#### FP8 MoE backend dispatch

On S5000, block-128 E4M3 MoE layers use a shape-keyed `auto` policy. Only
offline-calibrated per-rank shapes select native GEMV for small token batches.
For large eligible prefill invocations, `auto` preserves the default-on
contiguous DeepGEMM path; intermediate and unsupported shapes use
the established upstream fused-MoE implementation. The experimental grouped
backend added by this dispatcher remains disabled unless both its operator and
serving gates produce a validated threshold. The policy is also keyed by graph
mode and active MP count, so thresholds are not reused across incompatible
devices.

`VLLM_MUSA_FUSED_MOE_DISPATCH` is a diagnostic force/rollback control with the
values `auto` (default), `upstream`, `gemv`, or `grouped_gemm`. Forced modes do
not bypass dtype, layout, expert-parallel, device, or graph-capture safety
checks. In particular, forced GEMV during graph capture is limited to a
calibrated capture shape and token range; outside that range it falls back to
upstream. Production deployments should leave the override unset and use
`auto`. An explicit `upstream` value bypasses both native GEMV and the
default-on DeepGEMM prefill path, which makes it suitable as a diagnostic
rollback rather than a production default.

### `distributed/` – KV-Transfer Connector

Uses the Mooncake connector from the pinned upstream vLLM checkout for
disaggregated prefill/decode serving. The MUSA image installs the compatible
`mooncake-transfer-engine-musa` package; no MUSA-specific constructor rebind is
required. Use Mooncake's official `MC_TE_FILTERS` HCA allow-list for RDMA
selection. The old `MOONCAKE_RDMA_DEVICES` variable is only a deprecated
compatibility alias, and never overrides an explicitly set `MC_TE_FILTERS`.

### `utils/` – Utilities

- `deep_gemm.py` – DeepGEMM integration helpers
- `environ.py` – MUSA-specific environment variable handling

### `v1/` – Attention Backends & Ops

- `v1/attention/backends/mla/` – FlashMLA attention backend (`MUSAFlashMLABackend`) with MLA-specific metadata and common utilities.
- `v1/attention/backends/flash_attn.py` - FlashAttn attention backend (`MUSAFlashAttentionBackend`) with FLASH_ATTN functions.
- `v1/attention/backends/turboquant.py` – TurboQuant KV-cache backend wrapper. The `turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, and `turboquant_3bit_nc` presets are supported on MUSA.
- `v1/attention/ops/flashmla.py` – Low-level FlashMLA forward ops and capability detection.

### `_custom_ops.py` – Python Wrappers for C++ Ops

Thin Python wrappers around the native operators registered in `csrc/` (e.g., `musa_fused_gemv_moe`, fused gemv moe variants).
