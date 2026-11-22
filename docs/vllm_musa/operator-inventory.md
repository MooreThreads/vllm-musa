# vLLM-MUSA v0.32 operator and provider inventory

This is the initial AC-1 inventory for commit `d67f45c928`. It records the
runtime entry points and representative operator families. It is intentionally
source-based. Native build, device correctness, model serving, and performance
status must be filled by their separate gates.

## Registration entry points

| Entry | Location | Current role | Owner category |
| --- | --- | --- | --- |
| OOT plugin entry point | `pyproject.toml:musa_custom_ops`, `vllm_musa/__init__.py:register_custom_ops` | Applies object patches, preloads custom-op modules, registers modules/backends | vLLM-MUSA integration |
| AOT/JIT preload list | `vllm_musa/custom_op_modules.py:CUSTOM_OP_MODULES` | Imports modules before vLLM compile-cache use | registration compatibility layer |
| Python vLLM custom op | `vllm.utils.torch_utils.direct_register_custom_op` | Registers `torch.ops.vllm.*` wrappers and fake/mutability metadata | vLLM contract, MUSA provider |
| MUSA Torch library | `csrc/musa/torch_bindings.cpp` | Defines `torch.ops._C_musa_ops.*` schemas and MUSA implementations | vLLM-MUSA native provider |
| vLLM/MoE extension | `vllm_musa/_custom_ops.py`, `vllm._C`, `vllm._moe_C` | Loads compiled extension namespaces and exposes Python wrappers | vLLM contract/native provider |
| MUSA JIT extension | `vllm_musa/jit_kernel/csrc/jit.py:load_musa_jit` | Builds `.mu`/`.cu` sources with a per-arch cache | vLLM-MUSA native provider |
| TileLang custom op | `vllm_musa/jit_kernel/tilelang/*` | Registers selected TileLang functions through vLLM or compatibility wrappers | vLLM-MUSA JIT provider |
| vLLM source patch | `vllm_musa/patches/series/*.patch` | Modifies upstream contracts, layers, model call sites, and ABI/build wiring | vLLM compatibility integration |

## Provider and operator families

| Family | Contract | Provider/adapter | Registration/module | Source/build | Tests or checks | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| Attention, FA, FlashInfer, FlashMLA | vLLM attention backend/ops | MATE through vLLM-MUSA backend adapters | `vllm_musa/v1/attention/backends/*`, `vllm_musa/v1/attention/ops/*` | module-drift and series patches, MATE package pins | `tests/test_attention_backend_capabilities.py`, `tests/jit_kernel/test_fa3_metadata.py`, attention source tests | MATE provider, vLLM-MUSA adapter |
| Mamba2/SSD | vLLM Mamba backend | MATE/FlashInfer adapter | patch `0200`, Mamba backend modules | MATE/FlashInfer dependency and patched vLLM path | Mamba source and backend tests | MATE provider, vLLM-MUSA adapter |
| vLLM fused activation | `torch.ops.vllm.*` | vLLM-MUSA JIT/native | `vllm_musa/model_executor/layers/activation.py`, `jit_kernel/csrc/moe.py` | `.mu` sources `vllm_musa/jit_kernel/csrc/moe/act_and_mul.mu`, `vllm_musa/jit_kernel/csrc/moe/moe_sum_reduce.mu`; JIT loader | `tests/jit_kernel/test_csrc_jit_kernels.py`, MoE source tests | vLLM-MUSA |
| Norm/RMSNorm | vLLM custom op or layer contract | MUSA JIT/native, with vLLM fallback | `jit_kernel/csrc/norm.py`, `model_executor/layers/layernorm.py`, `fused_allreduce_rmsnorm_ops.py` | `csrc/musa/fused_add_rmsnorm.mu`, norm `.mu` sources, native bindings | `tests/test_deepseek_v4_fused_add_rmsnorm.py`, `tests/test_fused_add_rms_norm_compile.py` | vLLM-MUSA |
| RoPE and QKV preprocessing | vLLM layer/custom-op contract | MUSA JIT/native | `jit_kernel/csrc/rope.py`, `kernels/qwen3_qk_rope_kv.py` | `csrc/musa/attention/*`, `csrc/musa/torch_bindings.cpp` | `tests/kernels/test_qwen2_rope_kv.py`, `tests/test_qwen3_qk_rope_kv_presplit.py` | vLLM-MUSA |
| Sampler/top-k/top-p/min-p | vLLM sampling contract | MUSA native and FlashInfer-derived helpers | `vllm_musa/_custom_ops.py`, `jit_kernel/csrc/topk.py`, `vllm_musa/v1/sample/*` | native `csrc/musa/sampler.mu`, `top_p_mask_logits.mu`, bindings | `tests/test_chunked_min_p_sampler.py`, `tests/test_top_k_mask_logits.py`, `tests/test_top_p_mask_logits.py` | vLLM-MUSA; FlashInfer code is a dependency/input |
| Quantization | vLLM quantization/layer contract | MUSA native/JIT, MATE for GEMM paths | `jit_kernel/csrc/quant.py`, `model_executor/layers/quantization/*`, `kernels/fp8_block_requant.py` | native quant sources plus `setup.py` stable/native source lists | `tests/test_fp8_block_gemv_native.py`, quant source and numeric tests | vLLM-MUSA, MATE provider where selected |
| Dense/FP8 GEMV | vLLM linear contract | MUSA native/JIT | `vllm_musa/_custom_ops.py`, `jit_kernel/csrc/fp8_block_gemv.py`, `model_executor/kernels/linear/scaled_mm/block_gemv.py` | `_C_musa_ops`, `gemv.mu`, JIT sources | `tests/test_deepseek_v4_gemv_contract_source.py`, `tests/test_fp8_block_gemv_native.py` | vLLM-MUSA |
| DeepGEMM and MoE GEMM | vLLM linear/MoE contract | MATE adapter | `model_executor/kernels/linear/scaled_mm/deep_gemm.py`, `model_executor/layers/fused_moe/*` | MATE package APIs, vLLM patches `0009`, `0057`, `0136`, etc. | MoE source and model tests | MATE provider, vLLM-MUSA layer adapter |
| MoE routing | vLLM router contract | MATE fused gate plus vLLM-MUSA router helper | `model_executor/layers/fused_moe/router/*`, `jit_kernel/csrc/sqrtsoftplus_topk.py` | MATE APIs, JIT top-k sources | `tests/test_moe_fused_shared_topk_router.py`, router source tests | vLLM-MUSA adapter/provider |
| Distributed all-reduce | vLLM communicator contract | MUSA JIT/native | `distributed/device_communicators/musa_jit_custom_all_reduce.py`, `jit_kernel/csrc/allreduce.py` | `custom_all_reduce.mu`, extension/JIT loader | `tests/test_musa_jit_custom_all_reduce_graph.py` | vLLM-MUSA |
| DeepSeek V4 indexer/cache | model-private contract | MUSA native/JIT plus MATE GEMM paths | `kernels/deepseek_v4_*`, patch `0013`, `0014`, `0184`-`0216` | `csrc/musa/attention/*`, bindings, JIT sources | DeepSeek source/numeric/model tests | vLLM-MUSA model integration |
| Model-private MHC/GDN/TileLang | model/layer-private contract | vLLM-MUSA TileLang/JIT, MATE GEMM where used | `deepseek_v4_mhc.py`, `jit_kernel/tilelang/*`, model executor layers | TileLang/JIT source modules and patch series | `tests/test_deepseek_v4_mhc.py`, GDN/TileLang tests | vLLM-MUSA model/layer integration |
| Multimodal encoder attention | vLLM multimodal layer contract | MUSA custom op | `model_executor/layers/attention/mm_encoder_attention.py` | direct registered custom op and Python implementation | multimodal/backend source tests | vLLM-MUSA |

## Source/build ownership map

The compiled source lists are centralized in `setup.py` but mix upstream vLLM
sources, MUSA bindings, FlashInfer headers, and MoE extensions:

- `VLLM_CSRC_SOURCES` builds the vLLM regular-ABI view bindings.
- `VLLM_STABLE_CSRC_SOURCES` builds upstream stable-library sources adapted by
  the patch series.
- `VLLM_MUSA_CSRC_SOURCES` builds `csrc/musa/torch_bindings.cpp`, MUSA `.mu`,
  selected `.cu`, and FlashInfer-linked sources.
- `VLLM_MOE_CSRC_SOURCES` builds the regular-ABI MoE extension.
- `vllm_musa/jit_kernel/csrc/jit.py` independently compiles JIT source tuples
  and caches them by MUSA architecture.

The same operator can therefore have a Python registration, a Torch library
schema, a setup.py source entry, and a JIT source tuple. The future manifest
must record these as separate fields and validate their relationships.

## Current gaps against AC-1

1. `CUSTOM_OP_MODULES` is a hand-maintained preload tuple, not generated from
   operator records.
2. `direct_register_custom_op`, Torch library schemas, and JIT symbols have no
   shared machine-readable name or owner.
3. MATE-backed paths and vLLM-MUSA-owned native paths are interleaved in model
   and layer code, especially in MoE and attention integration.
4. `setup.py` source lists do not provide a reverse mapping from schema to
   extension/source/test.
5. Existing tests are substantial but are organized by feature/model and do not
   yet enforce complete registration metadata.
6. The patch-series metadata tracks upstream source divergence, not runtime op
   ownership. It should remain a separate concern.

## Inventory migration target

The first manifest should classify each record as one of:

- `vllm_aot_jit`: vLLM custom-op contract with a MUSA implementation
- `vllm_ir`: vLLM IR provider/lowering
- `musa_native`: vLLM-MUSA-owned Torch library or MUSA JIT op
- `mate_adapter`: vLLM backend/layer adapter around MATE
- `model_private`: model-specific assembly or fused path

No record should be marked `mate_adapter` unless the implementation is provided
by MATE and the vLLM-MUSA code only adapts the contract, metadata, capability,
or fallback behavior.
