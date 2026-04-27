# vllm_musa

Python package that plugs into vLLM's platform and general plugin system to enable inference on Moore Threads MUSA GPUs (MTGPU).

## Package Entry Points

| Entry Point | Function | Purpose |
|---|---|---|
| `vllm.platform_plugins` | `musa_platform_plugin` | Registers `MUSAPlatform` when MUSA hardware is detected |
| `vllm.general_plugins` | `register_custom_ops` | Applies source patches, registers OOT ops, distributed connectors, and attention backends |
| Console script | `collect_env` | Prints MUSA environment info (`vllm_collect_env`) |

## Module Overview

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

### `distributed/` – KV-Transfer Connector

Registers a Mooncake-based KV connector (`mooncake_connector.py`) for disaggregated prefill/decode serving (conditionally loaded when the `mooncake` package is available).

### `utils/` – Utilities

- `deep_gemm.py` – DeepGEMM integration helpers
- `environ.py` – MUSA-specific environment variable handling

### `v1/` – Attention Backends & Ops

- `v1/attention/backends/mla/` – FlashMLA attention backend (`MUSAFlashMLABackend`) with MLA-specific metadata and common utilities.
- `v1/attention/backends/flash_attn.py` - FlashAttn attention backend (`MUSAFlashAttentionBackend`) with FLASH_ATTN functions.
- `v1/attention/ops/flashmla.py` – Low-level FlashMLA forward ops and capability detection.

### `_custom_ops.py` – Python Wrappers for C++ Ops

Thin Python wrappers around the native operators registered in `csrc/` (e.g., `musa_fused_gemv_moe`, fused gemv moe variants).
