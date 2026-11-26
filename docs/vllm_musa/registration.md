# Registering vLLM Extensions on MUSA

`vllm-musa` integrates with vLLM through vLLM's native plugin and extension
contracts. Do not add new global registries, manifest scripts, or import-time
monkey-patches.

## Plugin entry points (`pyproject.toml`)

| Group | Entry point | Target | Responsibility |
| --- | --- | --- | --- |
| `vllm.platform_plugins` | `musa` | `vllm_musa:musa_platform_plugin` | Returns `vllm_musa.platform.MUSAPlatform` without side effects. |
| `vllm.general_plugins` | `musa_out_of_tree` | `vllm_musa:register_out_of_tree` | Idempotent registration of cat-6 runtime patches, Inductor heuristics, OOT layers, model-executor/v1 adapters, and AOT-cache custom-op preloads. |
| `vllm.general_plugins` | `musa_attn_backends` | `vllm_musa:register_attn_backends` | Idempotent registration of MUSA attention backends via `vllm.v1.attention.backends.registry`. |

vLLM automatically discovers all installed `vllm.platform_plugins` and
`vllm.general_plugins` entry points when `VLLM_PLUGINS` is unset — users do not
need to export `VLLM_PLUGINS`.

## Where to add a new extension

| Extension type | vLLM contract | Files to edit |
| --- | --- | --- |
| **vLLM IR op provider** (preferred for ops that exist in `vllm.ir.ops`) | `@ir.ops.<op>.register_impl("musa", ...)` + `Platform.import_ir_kernels()` + `Platform.get_default_ir_op_priority()` | 1. Add the provider in `vllm_musa/ir/` and export it in `vllm_musa/ir/__init__.py`.<br>2. Update `MUSAPlatformBase.get_default_ir_op_priority()` in `vllm_musa/platform.py` if it should be enabled by default.<br>3. If the provider calls a `direct_register_custom_op` module, add that module to `vllm_musa/custom_op_modules.py`. |
| **OOT Layer / CustomOp** | `@<UpstreamLayer>.register_oot` | 1. Subclass the upstream layer under `vllm_musa/model_executor/layers/` with `@<Layer>.register_oot`.<br>2. Import it in Group 1 of `vllm_musa/model_executor/__init__.py`. |
| **AOT / JIT Torch Custom Op** | `csrc/musa/torch_bindings.cpp` or `direct_register_custom_op(..., fake_impl=...)` | 1. AOT: add `.mu` source + binding in `csrc/musa/torch_bindings.cpp` + `setup.py` + wrapper in `vllm_musa/_custom_ops.py` (`vllm_musa._C` is loaded by `MUSAPlatformBase.import_kernels()`).<br>2. JIT: register with `direct_register_custom_op` and add the owning module to `vllm_musa/custom_op_modules.py` (enforced by `tests/test_custom_op_preload.py`). |
| **Attention Backend** (including MATE FA / FlashInfer / FlashMLA adapters) | `register_backend(AttentionBackendEnum.<NAME>, class_path=...)` | 1. Implement the backend adapter under `vllm_musa/v1/attention/backends/`. Keep `mate` / `flash_attn_3` imports inside the adapter.<br>2. Register it in `register_attention_backends()` and `_get_backend_priorities()` in `vllm_musa/platform.py`. |
| **Compatibility adapter** (only when upstream lacks an OOT hook) | Explicit idempotent `install_hooks()` | 1. Define an idempotent `install_hooks()` (no top-level attribute mutation at import time).<br>2. Call it from `vllm_musa/model_executor/__init__.py::register_model_executor_adapters()` or `vllm_musa/v1/__init__.py::register_v1_adapters()`. |

## Directory boundaries

- `vllm_musa/ir/`: **only** vLLM IR op providers (`@ir.ops.*.register_impl`), loaded by `MUSAPlatformBase.import_ir_kernels()`.
- `vllm_musa/kernels/`: standalone MUSA / Triton kernel helpers imported on demand (no IR registration on package import).
- `vllm_musa/jit_kernel/`: TVM-FFI / TileLang JIT kernel sources and `direct_register_custom_op` wrappers.
- `vllm_musa/patches/`: build-time upstream diff series (`series/`) and explicit cat-6 runtime object patches (`*.patch.py::apply()`).
