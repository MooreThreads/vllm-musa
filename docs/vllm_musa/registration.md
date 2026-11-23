# Registering vLLM extensions on MUSA

Use the vLLM extension point that owns the contract.

| Extension | Registration entry |
| --- | --- |
| Replacement for a vLLM layer | Use the upstream layer's `register_oot` API and load the implementation through `model_executor/registration.py`. |
| Torch custom op needed by a cached graph | Register its schema, implementation and fake implementation in the owning module. Add that module to `custom_op_modules.py` so fresh workers can load the schema before reusing a graph. |
| Implementation of a vLLM IR op | Use the upstream IR op's `register_impl` API. Load it through the provider package reached by `MUSAPlatformBase.import_ir_kernels()`. Check platform priority and support predicates. |

RMSNorm illustrates the two vLLM contracts. The OOT layer is loaded through
model-executor registration. The IR provider is loaded through the platform
hook before vLLM resolves kernel priorities. A backend library such as MATE
can supply the implementation underneath either integration.

Keep device and provider checks in the implementation or adapter that owns
them. When reusing CUDA-shaped interfaces, verify the MUSA dtype, layout,
mutations, fake implementation and graph-capture requirements against the
actual provider. Add a test that exercises the vLLM dispatch path.
