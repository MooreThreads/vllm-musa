# vLLM-MUSA operator ownership and integration model

## Scope

`vllm-musa` is the vLLM out-of-tree platform plugin. It assembles vLLM
contracts, MUSA providers, model and layer integrations, and runtime
capability selection. It is not a second general-purpose MUSA kernel library.

MATE owns reusable MUSA kernel providers, including attention, FlashAttention,
FlashInfer, Marlin, and DeepGEMM families. vLLM-MUSA owns the adapters that
translate vLLM contracts and metadata to those providers.

## Ownership boundaries

| Layer | Owner | Examples |
| --- | --- | --- |
| vLLM contract | upstream vLLM | custom AOT/JIT schemas, generic IR, layer/backend interfaces |
| Reusable MUSA provider | MATE | attention, FA/FlashInfer, Marlin, DeepGEMM |
| vLLM-MUSA provider | vLLM-MUSA | non-attention MUSA AOT/JIT ops, MUSA IR lowerings, native sampler/norm/cache ops |
| vLLM adapter | vLLM-MUSA | provider selection, layout conversion, capability checks, fallback, compile/cudagraph integration |
| Model/layer assembly | vLLM-MUSA | model-specific layers, DeepSeek/Qwen/GLM assembly, model private fused paths |

An implementation must have one primary owner. An adapter is not a second
implementation of the provider it calls.

## Required implementation record

Every new integrated operator or backend must have a record containing:

- vLLM contract and public call site
- provider and adapter module
- extension or JIT module
- source and build entry
- supported dtype, shape, layout, eager, compile, and CUDA graph modes
- fallback and unsupported-case behavior
- minimum package versions
- owning layer/model and correctness tests

The record is the source for preload, capability reporting, and consistency
checks. Module import must not be the only registration contract.

## Runtime layers

The intended call path is:

```text
model/layer
  -> vLLM contract or backend interface
  -> vLLM-MUSA provider selector
  -> MATE adapter or vLLM-MUSA provider
  -> implementation
```

Model code should not select MATE by importing it directly and should not use
an unclassified `_custom_ops` entry for a new integration.

`custom_op_modules.py` remains a compatibility preload list while entries are
migrated to an explicit registration manifest. Preloading is required for AOT
compile-cache stability, but preload itself does not define ownership.

## IR policy

Device-independent IR schemas belong upstream in `vllm.ir`. vLLM-MUSA supplies
MUSA providers, lowerings, capability predicates, and tolerance overrides.
Model-private or MUSA-semantic operations may remain under a clearly scoped
`vllm_musa` namespace.

## Repository policy

Do not create a second all-purpose `musa-ops` repository. MATE is the reusable
MUSA provider layer. A future repository is justified only for a non-attention
provider that has stable ABI/schema, independent build and correctness CI, and
multiple consumers without vLLM private dependencies.

## Migration sequence

1. Inventory current vLLM contracts, providers, adapters, registrations, and call sites.
2. Add an explicit integration manifest and generate preload and diagnostics from it.
3. Move attention calls behind MATE adapters without copying MATE kernels.
4. Separate vLLM custom AOT/JIT providers, IR providers, native providers, and model-private paths.
5. Add contract, provider, compile/capture, and serving gates before changing defaults.
6. Re-evaluate repository extraction only after a provider has independent consumers.
