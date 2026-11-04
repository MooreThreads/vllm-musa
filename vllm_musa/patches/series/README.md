# vLLM v0.32.0 build-time patch series

These patches apply, in filename order, to the immutable vLLM commit named
by `VLLM_COMMIT` in `third_party/PINS` (`5df63a7203cdb9328d824a27b682e77c7b9928be`, release `v0.32.0`).
Build-time source patches are separate from the runtime registrations in
`vllm_musa/patches/`.

The versioned sidecars in `../metadata/` are advisory indexes, not application
instructions: `legacy-lineage.json` records the 24 v0.28 numbers without a
separate current entry and their remaining validation; `patch-groups.json`
assigns each current patch one primary functional owner and lists the checks
required by that group. Cross-group `depends_on` references do not change
filename replay order. The README table below explains each legacy disposition;
`musa_sync check-series` validates its IDs against the JSON and validates every
group member against the current filenames. `third_party/PINS` alone selects
the base and `manifest.py` remains the divergence census. Run
`python3.10 tools/musa_sync.py report --metadata` for a concise index.

`.github/workflows/patch-series.yml` runs Python 3.10 offline unit/source
contracts, canonical format/metadata checks, strict build-path and `git am -3`
replay plus `verify` on a pristine pinned checkout, and regeneration round-trip
against a separate commit-bearing checkout. These are source-level gates;
required MUSA native builds, device tests, Mooncake integration and model
smokes are **not** marked passed by CI. Record hardware evidence separately,
scoped to the exact pin, series and environment.

The first 162 patches follow the v0.28 source-patch ancestry, including
follow-up changes carried in independent v0.32 patches. This is not a
one-to-one legacy series: 24 of the 192 old patch numbers have no separate
entry. Five independently maintainable changes that were folded into earlier
entries now follow their prerequisites at the old relative positions: old
`0084`, `0111`, `0112`, `0122`, and `0146`. Other same-file changes stay
together when they implement one coupled adaptation; for example, the fixed
MUSA SSD autotune configurations and the Nemotron shape-specific tile
selection in `0144` share the autotune configuration contract. Missing old
numbers may be upstreamed, superseded, canceled by later patches, folded into
a coherent adaptation, or unresolved; the dispositions below distinguish
these cases instead of treating a missing number as proof of missing code.
The original base patch was divided by file and hunk ancestry. This pass keeps
that ancestry order but combines only changes that implement one independently
maintainable feature: custom all-reduce (`0037`), dedicated Mamba pools
(`0078`), rejection-sampler scalarization (`0111`), and DeepSeek-V4 FP8 expert
dtype/scale resolution (`0134`). Eight v0.32 native/compatibility adaptations
follow (`0163`–`0170`), then 41 new or updated v0.32 features
(`0171`–`0211`). The original cross-version
DeepSeek-V4/V4.1 aux-stream patch stays in the new group because its v0.28
segment depends on later v0.32 attention changes; old `0110` remains in the
legacy group. Old `0137` (Mooncake P/D dedicated Mamba pools) and `0192`
(hybrid attention page-size fast path) have no independent continuation here;
their v0.32 behavior still requires separate review. The upstream follow-ups
`0206` and `0207` now change Mamba accounting and hybrid attention page sizing;
the historical observations below describe the earlier baseline, not a measured
failure of these follow-ups. Hardware validation remains required.

A patch may update several files when they form one atomic feature. The grouped
entries are intentionally multi-file: `0037` contains
`csrc/custom_all_reduce.cu` and `.cuh`; `0078` contains the Mamba interface,
cache utility, and coordinator changes; `0111` contains the coupled rejection
sampler Triton changes; and `0134` contains the DeepSeek-V4 MTP, model, and
DSpark dtype/scale changes. The four runtime/ABI changes at `0058`–`0061`
remain separate because each has an independent build and rollback boundary.
This avoids both a monolithic base patch and artificial one-file fragments.

v0.28 0065: superseded by the compiler fix (`0172`), which applies the
Triton grouping directly. v0.28 0177: superseded by the staged indexer
dispatcher (`0192`), which replaces the original top-k call directly. Neither
intermediate patch has its own entry. This records the final v0.32 source
result, not equivalence across all v0.28 runtime paths.

The 24 old numbers without their own entry are accounted for below. The
comparison is against the pinned vLLM tree after all 210 patches, not a
model-level regression. “Folded” identifies the owner of the final source
change, not proof of equivalent runtime behavior. A blocked/unverified row is
not a canceled feature and must not be restored verbatim without an adapted
contract and regression.

| v0.28 | Disposition / v0.32 source evidence | Validation still needed |
| --- | --- | --- |
| `0065` | Intermediate Triton grouping is directly provided by compiler fix `0172`. | No separate intermediate patch needed. |
| `0069` | Sparse MLA zero-offset allocation is absent: `kv_cache_utils.py` now packs per-spec `offset=byte_offset` into shared tensors, exposed by `kv_cache_interface.py` as strided views. | MATE sparse decode with multiple blocks and a nonzero spec offset; do not force zero offset on the shared allocation. |
| `0088` | Installed pinned-vLLM planner APIs (CPU-only, MUSA predicate forced) expose dedicated-pool accounting inconsistent with shared-pool precheck, override and auto-fit. A probe derived from a local 64-layer Qwen3.5 dense config (48 linear, 16 full-attention layers; TP=8, BF16; no prefix cache; model-length cap 2048; `max_num_seqs=4`) computes GDN state shapes and dtypes with vLLM calculators: Mamba page 204,288 bytes and attention page 65,536 bytes before LBNHC unification. Grouping produces three bare 16-layer Mamba groups and one 16-layer attention group (pages unified to 204,288 bytes), so separate pools are eligible. At 120,000,000 bytes the shared-pool precheck passes (114,401,280 bytes plus a 3,268,608-byte null block), but planning yields just 21 attention blocks (20 usable); default full-ISL admission fails for a 2048-token prompt needing 32 attention blocks. At 156,000,000 bytes the planner yields 32 (31 usable) and admission still fails; at 160,000,000 bytes it yields 33 (32 usable) and admission succeeds. The earlier 16-token Mamba-block-size fixture's first-chunk refusal was an unsupported `none`-mode artifact. Two ranks use the minimum budget consistently; a 129-block override can be rejected by shared-pool accounting despite fitting dedicated-pool geometry. Connectors exclude separate pooling. | Align dedicated-pool memory/override/auto-fit checks with actual per-pool allocations and null blocks; run a fully instantiated Qwen model config and MUSA scheduler/model regression. Config-derived CPU specs with offline mocks are not model-serving proof. |
| `0089` | The 288-thread small-batch MoE alignment scan was not carried into the relocated `csrc/moe/moe_align_sum_kernels.cu` (1024-thread path). | Performance opportunity, not an established correctness failure; profile small-batch Qwen MoE first. |
| `0090` | Folded into DeepSeek-V4 attention `0014` (small-M FP32 score GEMM dispatch). | Check the model's score path on MUSA. |
| `0093` | Folded into sparse-compressor `0138` (C4 indexer compression). | Check the DeepSeek-V4 C4 path on MUSA. |
| `0114` | Old regular-ABI `csrc/cuda_view.cu` is replaced by `0163`'s `csrc/musa_cuda_view.cu` and `_C` registration; `setup.py` builds both regular-ABI sources. | Native build/import and pinned-memory CPU-offload view smoke; not a source-level no-op. |
| `0115` | Relocated MoE helper includes belong to the regular-ABI MoE build adaptation `0061`. | Native extension build. |
| `0116` | Pinned `csrc/torch_bindings.cpp` registers `_C_cuda_utils` schemas and implementations; `setup.py` links `cuda_utils_kernels.cu` into the stable extension. | Native build and op invocation. |
| `0118` | Folded into Triton/Gluon import compatibility `0105`. | Import with the installed Triton distribution. |
| `0119` | The v0.32 runner already passes `num_scheduled_tokens_np` and `batch_req_state.is_prefilling_np` to `_musa_select_uniform_decode_model_inputs`; the Qwen hook checks both arrays' shapes, values and prefill state. | Source call contract matches; run Qwen uniform decode including fallback/priming. |
| `0126` | Chained rejection predicate is folded into scalarized rejection patch `0111`. | Speculative rejection test on MUSA. |
| `0129` | Scalarized stochastic rejection predicate is folded into scalarized rejection patch `0111`. | Speculative rejection test on MUSA. |
| `0137` | Dedicated Mamba pools with Mooncake were **not** restored: `_musa_separate_pool_eligible` excludes all KV connectors. The Mooncake store scheduler still owns the coordinator's shared `block_pool`. | Connector-to-pool ownership routing and hybrid P/D validation before admission. |
| `0139` | Pinned `ModelConfig._get_transformers_backend_cls` guards empty `self.architectures` before indexing, covering the old nested-config failure. | Nested Transformers model-config test. |
| `0140` | Coupled SSD shape-key/pruning and tile adaptation lives in `0137`; splitting the fixed autotune table would leave a temporary single-tile contract. | SSD shape and model test. |
| `0148` | MTP metadata ownership is folded into proposer `0030`. | Async MTP lifecycle test. |
| `0158` | Recent-indexer metadata removal is an intermediate change canceled by the later attention-layer adaptation. | Check final learned-indexer dispatch; do not replay the temporary deletion. |
| `0161` | Learned decode graph guard belongs to MUSA attention-layer adaptation `0174`. | Captured DeepSeek-V4 learned-indexer decode. |
| `0162` | Learned paged-MQA decode belongs to sparse indexer `0013`/`0080`. | Captured DeepSeek-V4 paged-MQA decode. |
| `0175` | Pinned `gather_batch_req_state` now sets `prefill_runs_as_decode_np` for a one-token extension over existing context (excluding PCP), passes `decode_graph_eligible` to graph selection, and keeps `has_prefill` for input preparation. | Existing final-prompt graph test needs installed vLLM; confirm padded-draft and PCP cases on MUSA. |
| `0177` | Intermediate top-k selector is replaced directly by staged indexer dispatch `0192`. | No separate intermediate patch needed. |
| `0186` | Pinned `triton_w4a16_gemm` calls `torch.ops.vllm.triton_w4a16_gemm`; its registered implementation chooses M-dependent tiles and split-K at invocation. | Compile/capture with distinct runtime M values; source registration alone does not prove tile correctness. |
| `0192` | Old hybrid attention page-64/alignment fast path has no proven equivalent across the new grouping and Mamba layouts. | Supported hybrid configuration, group pages/addresses and MUSA model run. |

The mapping below records **patch ancestry, not semantic equivalence**. A
byte-identical final tree confirms that the reorganization did not change the
original v0.32 output, not that every v0.28 behavior survives upstream changes.

The reorganized series is intended to reproduce the updated upstream 58-patch v0.32
output at the pinned vLLM commit. The separate DeepSeek-V4 native indexer
capacity guard is not part of this patch-management change; it requires its own
source-level and MUSA long-context graph/model regression before inclusion.

| New patch | v0.28 ancestry | Responsibility |
| --- | --- | --- |
| `0001-MUSA-vllm.compilation.backends.patch` | v0.28 0001 | vllm.compilation.backends |
| `0002-MUSA-vllm.compilation.caching.patch` | v0.28 0002 | vllm.compilation.caching |
| `0003-MUSA-vllm.compilation.passes.pass_manager.patch` | v0.28 0003 | vllm.compilation.passes.pass_manager |
| `0004-MUSA-vllm.compilation.piecewise_backend.patch` | v0.28 0004 | vllm.compilation.piecewise_backend |
| `0005-MUSA-vllm.distributed.device_communicators.cuda_comm.patch` | v0.28 0005 | vllm.distributed.device_communicators.cuda_communicator |
| `0006-MUSA-vllm.distributed.device_communicators.custom_al.patch` | v0.28 0006 | vllm.distributed.device_communicators.custom_all_reduce |
| `0007-MUSA-vllm.model_executor.layers.attention.attention.patch` | v0.28 0007 | vllm.model_executor.layers.attention.attention |
| `0008-MUSA-vllm.model_executor.layers.fused_moe.activation.patch` | v0.28 0008 | vllm.model_executor.layers.fused_moe.activation |
| `0009-MUSA-vllm.model_executor.layers.fused_moe.experts.de.patch` | v0.28 0009 | vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe |
| `0010-MUSA-vllm.model_executor.layers.mhc.patch` | v0.28 0010 | vllm.model_executor.layers.mhc |
| `0011-MUSA-vllm.model_executor.layers.quantization.fp8.patch` | v0.28 0011 | vllm.model_executor.layers.quantization.fp8 |
| `0012-MUSA-vllm.model_executor.layers.quantization.utils.f.patch` | v0.28 0012 | vllm.model_executor.layers.quantization.utils.fp8_utils |
| `0013-MUSA-vllm.model_executor.layers.sparse_attn_indexer.patch` | v0.28 0013 | vllm.model_executor.layers.sparse_attn_indexer |
| `0014-MUSA-vllm.models.deepseek_v4.attention.patch` | v0.28 0014 | vllm.models.deepseek_v4.attention |
| `0015-MUSA-vllm.models.deepseek_v4.common.ops.fused_inv_ro.patch` | v0.28 0015 | vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant |
| `0016-MUSA-vllm.models.deepseek_v4.nvidia.flashmla.patch` | v0.28 0016 | vllm.models.deepseek_v4.nvidia.flashmla |
| `0017-MUSA-vllm.models.deepseek_v4.nvidia.model.patch` | v0.28 0017 | vllm.models.deepseek_v4.nvidia.model |
| `0018-MUSA-vllm.profiler.wrapper.patch` | v0.28 0018 | vllm.profiler.wrapper |
| `0019-MUSA-vllm.triton_utils.jit_monitor.patch` | v0.28 0019 | vllm.triton_utils.jit_monitor |
| `0020-MUSA-vllm.utils.deep_gemm.patch` | v0.28 0020 | vllm.utils.deep_gemm |
| `0021-MUSA-vllm.v1.attention.backends.mla.flashmla.patch` | v0.28 0021 | vllm.v1.attention.backends.mla.flashmla |
| `0022-MUSA-vllm.v1.attention.backends.mla.sparse_swa.patch` | v0.28 0022 | vllm.v1.attention.backends.mla.sparse_swa |
| `0023-MUSA-vllm.v1.attention.ops.flashmla.patch` | v0.28 0023 | vllm.v1.attention.ops.flashmla |
| `0024-MUSA-vllm.v1.attention.ops.triton_attention_helpers.patch` | v0.28 0024 | vllm.v1.attention.ops.triton_attention_helpers |
| `0025-MUSA-vllm.v1.attention.ops.triton_turboquant_decode.patch` | v0.28 0025 | vllm.v1.attention.ops.triton_turboquant_decode |
| `0026-MUSA-vllm.v1.cudagraph_dispatcher.patch` | v0.28 0026 | vllm.v1.cudagraph_dispatcher |
| `0027-MUSA-vllm.v1.sample.ops.topk_topp_triton.patch` | v0.28 0027 | vllm.v1.sample.ops.topk_topp_triton |
| `0028-MUSA-apply-rejection-sampling-constraints.patch` | v0.28 0028 | vllm.v1.sample.rejection_sampler (v0.28 reanchor) |
| `0029-MUSA-vllm.v1.spec_decode.dflash.patch` | v0.28 0029 | vllm.v1.spec_decode.dflash |
| `0030-MUSA-vllm.v1.spec_decode.llm_base_proposer.patch` | v0.28 0030 | vllm.v1.spec_decode.llm_base_proposer |
| `0031-MUSA-vllm.v1.spec_decode.utils.patch` | v0.28 0031 | vllm.v1.spec_decode.utils |
| `0032-MUSA-vllm.v1.worker.gpu.block_table.patch` | v0.28 0032 | vllm.v1.worker.gpu.block_table |
| `0033-MUSA-vllm.v1.worker.gpu.sample.penalties.patch` | v0.28 0033 | vllm.v1.worker.gpu.sample.penalties |
| `0034-MUSA-vllm.v1.worker.gpu_input_batch.patch` | v0.28 0034 | vllm.v1.worker.gpu_input_batch |
| `0035-MUSA-support-full-graph-drafts-and-pinned-metadata.patch` | v0.28 0035 | vllm.v1.worker.gpu_model_runner (v0.28 reanchor) |
| `0036-MUSA-vllm.v1.worker.gpu_worker.patch` | v0.28 0036 | vllm.v1.worker.gpu_worker |
| `0037-MUSA-build-the-custom-all-reduce-implementation-as-o.patch` | v0.28 0037 + v0.28 0038 | MUSA: build the custom all-reduce implementation as one feature |
| `0038-MUSA-MUSA-csrc-file-csrc-mamba-mamba_ssm-selective_s.patch` | v0.28 0039 | MUSA(csrc-file): csrc/mamba/mamba_ssm/selective_scan_fwd.cu |
| `0039-MUSA-MUSA-csrc-file-csrc-quantization-activation_ker.patch` | v0.28 0040 | MUSA(csrc-file): csrc/quantization/activation_kernels.cu |
| `0040-MUSA-MUSA-csrc-csrc-topk.cu.patch` | v0.28 0041 | MUSA(csrc): csrc/topk.cu |
| `0041-MUSA-MUSA-csrc-csrc-moe-torch_bindings.cpp.patch` | v0.28 0042 | MUSA(csrc): csrc/moe/torch_bindings.cpp |
| `0042-MUSA-MUSA-csrc-csrc-torch_bindings.cpp.patch` | v0.28 0043 | MUSA(csrc): csrc/torch_bindings.cpp |
| `0043-MUSA-MUSA-csrc-libtorch_stable-torch_bindings-Privat.patch` | v0.28 0044 | MUSA(csrc): libtorch_stable torch_bindings PrivateUse1 impls |
| `0044-MUSA-MUSA-csrc-csrc-quantization-w8a8-fp8-nvidia-qua.patch` | v0.28 0045 | MUSA(csrc): csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh |
| `0045-MUSA-MUSA-csrc-csrc-attention-merge_attn_states.cu.patch` | v0.28 0046 | MUSA(csrc): csrc/attention/merge_attn_states.cu |
| `0046-MUSA-MUSA-csrc-csrc-cuda_compat.h.patch` | v0.28 0047 | MUSA(csrc): csrc/cuda_compat.h |
| `0047-MUSA-MUSA-csrc-use-portable-int8-rounding-in-stable-.patch` | v0.28 0048 | MUSA(csrc): use portable int8 rounding in stable quant conversions |
| `0048-MUSA-MUSA-csrc-csrc-moe-moe_align_sum_kernels.cu.patch` | v0.28 0049 | MUSA(csrc): csrc/moe/moe_align_sum_kernels.cu |
| `0049-MUSA-MUSA-csrc-csrc-type_convert.cuh.patch` | v0.28 0050 | MUSA(csrc): csrc/type_convert.cuh |
| `0050-MUSA-vllm.ir.tolerances.patch` | v0.28 0051 | vllm.ir.tolerances |
| `0051-MUSA-vllm.model_executor.kernels.linear.patch` | v0.28 0052 | vllm.model_executor.kernels.linear |
| `0052-MUSA-vllm.distributed.device_communicators.all2all.patch` | v0.28 0053 | vllm.distributed.device_communicators.all2all |
| `0053-MUSA-vllm.models.deepseek_v4.common.ops.fused_compre.patch` | v0.28 0054 | vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache |
| `0054-MUSA-vllm.models.deepseek_v4.common.ops.save_partial.patch` | v0.28 0055 | vllm.models.deepseek_v4.common.ops.save_partial_states |
| `0055-MUSA-vllm.model_executor.models.qwen3_moe.patch` | v0.28 0056 | vllm.model_executor.models.qwen3_moe |
| `0056-MUSA-vllm.v1.attention.backends.gdn_attn.patch` | v0.28 0057 | vllm.v1.attention.backends.gdn_attn |
| `0057-MUSA-vllm.model_executor.layers.fused_moe.fused_moe.patch` | v0.28 0058 | vllm.model_executor.layers.fused_moe.fused_moe |
| `0058-MUSA-MUSA-csrc-regular-ABI-_C_cache_ops-registration.patch` | v0.28 0059 | MUSA(csrc): regular-ABI _C_cache_ops registration for std::string KV-cache ops |
| `0059-MUSA-MUSA-csrc-drop-unboxable-stable-ops-guard-non-c.patch` | v0.28 0060 | MUSA(csrc): drop unboxable stable ops, guard non-core activations, register MUSA cache/runtime ops |
| `0060-MUSA-MUSA-csrc-build-the-relocated-MoE-kernels-as-a-.patch` | v0.28 0061 | MUSA(csrc): build the relocated MoE kernels as a regular at::Tensor extension |
| `0061-MUSA-MUSA-csrc-port-libtorch_stable-torch_utils.h-de.patch` | v0.28 0062 | MUSA(csrc): port libtorch_stable torch_utils.h device APIs |
| `0062-MUSA-vllm.model_executor.layers.minimax_rms_norm.lam.patch` | v0.28 0063 | vllm.model_executor.layers.minimax_rms_norm.lamport_workspace |
| `0063-MUSA-vllm.v1.attention.ops.triton_unified_attention.patch` | v0.28 0064 | vllm.v1.attention.ops.triton_unified_attention |
| `0064-MUSA-vllm.model_executor.layers.fused_moe.routed_exp.patch` | v0.28 0066 | vllm.model_executor.layers.fused_moe.routed_experts |
| `0065-MUSA-vllm.model_executor.kernels.mhc.tilelang.patch` | v0.28 0067 | vllm.model_executor.kernels.mhc.tilelang |
| `0066-MUSA-vllm.models.deepseek_v4.nvidia.ops.o_proj.patch` | v0.28 0068 | vllm.models.deepseek_v4.nvidia.ops.o_proj |
| `0067-MUSA-keep-DeepSeek-V4-MTP-RMSNorm-branch-dtypes-stab.patch` | v0.28 0070 | keep DeepSeek-V4 MTP RMSNorm branch dtypes stable |
| `0068-MUSA-skip-DeepSeek-V4-sparse-MLA-mixed-warmup-for-MT.patch` | v0.28 0071 | skip DeepSeek-V4 sparse MLA mixed warmup for MTP |
| `0069-MUSA-avoid-logits-gather-collectives-on-TP.patch` | v0.28 0072 | avoid logits gather collectives on TP |
| `0070-MUSA-load-weak-tensor-ref-kernels-lazily.patch` | v0.28 0073 | load weak tensor ref kernels lazily |
| `0071-MUSA-MUSA-csrc-use-boxable-strides-in-stable-layerno.patch` | v0.28 0074 | MUSA(csrc): use boxable strides in stable layernorm-quant kernels |
| `0072-MUSA-MUSA-csrc-use-portable-rounding-in-stable-int8-.patch` | v0.28 0075 | MUSA(csrc): use portable rounding in stable int8 quant kernels |
| `0073-MUSA-MUSA-model-fold-the-Qwen3.5-shared-expert-into-.patch` | v0.28 0076 | MUSA(model): fold the Qwen3.5 shared expert into fused_moe |
| `0074-MUSA-MUSA-vllm-fix-GLM-sparse-MLA-uninitialized-quer.patch` | v0.28 0077 | MUSA(vllm): fix GLM sparse MLA uninitialized query buffers |
| `0075-MUSA-allow-Gemma-RMSNorm-residual-donation.patch` | v0.28 0078 | allow Gemma RMSNorm residual donation |
| `0076-MUSA-add-gated-QKV-RMSNorm-MRoPE-IR.patch` | v0.28 0079 | add gated QKV RMSNorm MRoPE IR |
| `0077-MUSA-vllm.v1.attention.backends.gdn_attn-conv1d-cpu-.patch` | v0.28 0080 | vllm.v1.attention.backends.gdn_attn conv1d cpu seq_lens |
| `0078-MUSA-allocate-Mamba-state-in-dedicated-KV-cache-pool.patch` | v0.28 0081 + v0.28 0082 + v0.28 0083 | MUSA: allocate Mamba state in dedicated KV-cache pools |
| `0079-MUSA-route-group-FP8-quant-to-the-register-resident-.patch` | v0.28 0084 | route group FP8 quant to the register-resident kernel |
| `0080-MUSA-MUSA-vllm.v1.attention.backends.mla.indexer.patch` | v0.28 0085 | MUSA vllm.v1.attention.backends.mla.indexer |
| `0081-MUSA-MUSA-model-fuse-QK-RMSNorm-and-MRoPE-for-interl.patch` | v0.28 0086 | MUSA(model): fuse QK-RMSNorm and MRoPE for interleaved MRoPE |
| `0082-MUSA-fused_moe_kernel-even_Ks-scalar-b_scale-fast-pa.patch` | v0.28 0087 | fused_moe_kernel even_Ks + scalar b_scale fast paths |
| `0083-MUSA-perf-musa-unify-Qwen-runtime-fast-paths.patch` | v0.28 0091 | perf(musa): unify Qwen runtime fast paths |
| `0084-MUSA-skip-unused-DeepSeek-V4-recent-indexer-Q-work.patch` | v0.28 0092 | skip unused DeepSeek-V4 recent indexer Q work |
| `0085-MUSA-DeepSeek-V4-fuse-clamp-SwiGLU-FP8-down-projecti.patch` | v0.28 0094 | DeepSeek-V4 fuse clamp SwiGLU FP8 down projection |
| `0086-perf-view-Qwen-identity-logits-hidden-state.patch` | v0.28 0095 | perf: view Qwen identity logits hidden state |
| `0087-perf-cache-Qwen-uniform-sample-counts.patch` | v0.28 0096 | perf: cache Qwen uniform sample counts |
| `0088-perf-fuse-Qwen-next-input-IDs.patch` | v0.28 0097 | perf: fuse Qwen next input IDs |
| `0089-perf-reuse-identity-Qwen-sampling-input-views.patch` | v0.28 0098 | perf: reuse identity Qwen sampling input views |
| `0090-perf-reuse-proven-Qwen-sample-counts-on-host.patch` | v0.28 0099 | perf: reuse proven Qwen sample counts on host |
| `0091-MUSA-allow-ModelOpt-FP8-on-MUSA-capability.patch` | v0.28 0100 | allow ModelOpt FP8 on MUSA capability |
| `0092-perf-reuse-common-GDN-full-decode-metadata.patch` | v0.28 0101 | perf: reuse common GDN full decode metadata |
| `0093-perf-add-gated-sharded-Qwen-Gumbel-logits-path.patch` | v0.28 0102 | perf: add gated sharded Qwen Gumbel logits path |
| `0094-perf-skip-SSM-slot-mapping-in-uniform-decode.patch` | v0.28 0103 | perf: skip SSM slot mapping in uniform decode |
| `0095-MUSA-bind-DeepSeek-V4-optimization-contract.patch` | v0.28 0104 | bind DeepSeek-V4 optimization contract |
| `0096-perf-specialize-Qwen3.6-BF16-W1-tile-at-C25.patch` | v0.28 0105 | perf: specialize Qwen3.6 BF16 W1 tile at C25 |
| `0097-MUSA-preserve-DeepSeek-V4-MTP-sparse-prefill-headroo.patch` | v0.28 0106 | preserve DeepSeek-V4 MTP sparse-prefill headroom |
| `0098-MUSA-fence-DeepSeek-V4-MTP-mixed-prefill-queues.patch` | v0.28 0107 | fence DeepSeek-V4 MTP mixed prefill queues |
| `0099-perf-capture-Qwen3-VL-vision-blocks-on-MUSA.patch` | v0.28 0108 | perf: capture Qwen3-VL vision blocks on MUSA |
| `0100-fix-avoid-CUDA-FA2-import-on-MUSA.patch` | v0.28 0109 | fix: avoid CUDA FA2 import on MUSA |
| `0101-MUSA-keep-DeepSeek-V4-aux-stream-overlap-in-FULL-dec.patch` | v0.28 0110 | keep DeepSeek-V4 aux-stream overlap in FULL decode graphs |
| `0102-MUSA-enable-Qwen3.5-shared-fold-for-TP-only-MoE.patch` | v0.28 0111 | enable Qwen3.5 shared fold for TP-only MoE |
| `0103-MUSA-bind-Qwen3.5-folded-shared-gate-to-router.patch` | v0.28 0112 | bind Qwen3.5 folded shared gate to router |
| `0104-MUSA-fuse-Qwen3.5-QK-mRoPE-cache-update.patch` | v0.28 0113 | fuse Qwen3.5 QK mRoPE cache update |
| `0105-MUSA-tolerate-Triton-without-experimental-Gluon.patch` | v0.28 0117 | tolerate Triton without experimental Gluon |
| `0106-MUSA-hash-fused-MoE-on-Triton-3.2.patch` | v0.28 0120 | hash fused MoE on Triton 3.2 |
| `0107-MUSA-skip-CUDA-only-FA4-warmup.patch` | v0.28 0121 | skip CUDA-only FA4 warmup |
| `0108-MUSA-support-Mamba-pools-during-graph-profiling.patch` | v0.28 0122 | support Mamba pools during graph profiling |
| `0109-MUSA-restore-component-memory-accounting-for-profili.patch` | v0.28 0123 | restore component memory accounting for profiling |
| `0110-MUSA-route-DSV4-prenorm-through-MATE-DeepGEMM.patch` | v0.28 0124 | route DSV4 prenorm through MATE DeepGEMM |
| `0111-MUSA-make-rejection-sampling-predicates-scalar-on-Tr.patch` | v0.28 0125 + v0.28 0127 + v0.28 0128 | MUSA: make rejection sampling predicates scalar on Triton |
| `0112-MUSA-restore-contiguous-segregated-Mamba-cache-pools.patch` | v0.28 0130 | restore contiguous segregated Mamba cache pools |
| `0113-MUSA-keep-DeepSeek-V4-on-MRV1-by-default.patch` | v0.28 0131 | keep DeepSeek-V4 on MRV1 by default |
| `0114-MUSA-keep-DeepSeek-V4-hc_head-on-fused-TileLang.patch` | v0.28 0132 | keep DeepSeek-V4 hc_head on fused TileLang |
| `0115-MUSA-make-DeepEP-Buffer-shutdown-cleanup-compatible.patch` | v0.28 0133 | make DeepEP Buffer shutdown cleanup compatible |
| `0116-MUSA-bypass-grouped-DeepGEMM-UE8M0-cast-on-MATE-0.2..patch` | v0.28 0134 | bypass grouped DeepGEMM UE8M0 cast on MATE 0.2.6 |
| `0117-MUSA-fix-DeepSeek-OCR-tokenizer-and-vision-precision.patch` | v0.28 0135 | fix DeepSeek-OCR tokenizer and vision precision |
| `0118-MUSA-support-KV-first-FlashAttention-in-Mooncake.patch` | v0.28 0136 | support KV-first FlashAttention in Mooncake |
| `0119-MUSA-complete-Mamba-prefix-cache-pool-ownership.patch` | v0.28 0138 | complete Mamba prefix-cache pool ownership |
| `0120-MUSA-preserve-dedicated-Mamba-prefix-cache.patch` | v0.28 0138 continued | preserve dedicated Mamba prefix cache |
| `0121-MUSA-share-Triton-version-check.patch` | v0.28 0141 | share-Triton-version-check |
| `0122-MUSA-mamba2-ssd-concrete-optional-pointer.patch` | v0.28 0142 | mamba2-ssd-concrete-optional-pointer |
| `0123-MUSA-mamba2-ssd-gated-fp32-dot.patch` | v0.28 0143 | mamba2-ssd-gated-fp32-dot |
| `0124-MUSA-pass-Mamba2-graph-metadata.patch` | v0.28 0144 | pass-Mamba2-graph-metadata |
| `0125-MUSA-v1-avoid-prefill-uniform-decode-collision.patch` | v0.28 0145 | v1-avoid-prefill-uniform-decode-collision |
| `0126-MUSA-give-Mamba-SSD-initial-state-concrete-strides.patch` | v0.28 0146 | give Mamba SSD initial state concrete strides |
| `0127-MUSA-disable-naive-moe-on-triton32.patch` | v0.28 0147 | disable-naive-moe-on-triton32 |
| `0128-MUSA-fix-Triton-chained-boolean-in-Mamba-postprocess.patch` | v0.28 0149 | fix-Triton-chained-boolean-in-Mamba-postprocess |
| `0129-MUSA-use-MUSA-event-for-async-MTP-state-readiness.patch` | v0.28 0150 | use-MUSA-event-for-async-MTP-state-readiness |
| `0130-MUSA-keep-DSV4-learned-indexer-on-graph-capture.patch` | v0.28 0151 | keep DSV4 learned indexer on graph capture |
| `0131-MUSA-use-stream-waits-for-DSV4-overlap-hand-off.patch` | v0.28 0152 | use stream waits for DSV4 overlap hand-off |
| `0132-MUSA-dispatch-DeepSeek-V4-DSpark-context-KV-insert.patch` | v0.28 0153 | dispatch DeepSeek-V4 DSpark context KV insert |
| `0133-MUSA-provide-typed-optional-pointers-for-DSpark-reje.patch` | v0.28 0154 | provide typed optional pointers for DSpark rejection sampler |
| `0134-MUSA-resolve-DeepSeek-V4-FP8-expert-dtype-and-scales.patch` | v0.28 0155 + v0.28 0156 + v0.28 0157 | MUSA: resolve DeepSeek-V4 FP8 expert dtype and scales consistently |
| `0135-MUSA-keep-DSV4-learned-indexer-dispatch.patch` | v0.28 0159 | keep DSV4 learned indexer dispatch |
| `0136-MUSA-keep-DSV4-learned-indexer-on-graph-decode.patch` | v0.28 0160 | keep DSV4 learned indexer on graph decode |
| `0137-MUSA-mamba2-ssd-nemotron-prefill-tile.patch` | v0.28 0163 | mamba2 ssd nemotron prefill tile |
| `0138-MUSA-dispatch-DeepSeek-V4-sparse-compressor-to-nativ.patch` | v0.28 0164 | dispatch DeepSeek-V4 sparse compressor to native op |
| `0139-MUSA-fuse-DeepSeek-V4-save_partial-into-native-compr.patch` | v0.28 0165 | fuse DeepSeek-V4 save_partial into native compressor |
| `0140-MUSA-mrope-interleave-via-strided-copy.patch` | v0.28 0166 | mrope-interleave-via-strided-copy |
| `0141-MUSA-vllm.model_executor.layers.fused_moe.router.gat.patch` | v0.28 0167 | vllm.model_executor.layers.fused_moe.router.gate_linear |
| `0142-MUSA-vllm.model_executor.layers.fused_moe.fused_moe.patch` | v0.28 0168 | vllm.model_executor.layers.fused_moe.fused_moe |
| `0143-MUSA-vllm.v1.attention.ops.triton_unified_attention.patch` | v0.28 0169 | vllm.v1.attention.ops.triton_unified_attention |
| `0144-MUSA-bound-DSV4-long-prefill-indexer-logits.patch` | v0.28 0170 | bound DSV4 long-prefill indexer logits |
| `0145-MUSA-enable-GLM-5-models-with-eager-MCCL-init-and-GL.patch` | v0.28 0171 | enable GLM-5 models with eager MCCL init and GLM sparse dispatch |
| `0146-MUSA-rotate-V2-output-copy-streams.patch` | v0.28 0172 continued | rotate V2 output copy streams |
| `0147-MUSA-skip-Prometheus-observe-for-connectors-without-.patch` | v0.28 0173 continued | skip Prometheus observe for connectors without metrics |
| `0148-MUSA-perf-musa-extend-DSV4-score-DeepGEMM-to-decode-.patch` | v0.28 0174 continued | perf(musa): extend DSV4 score DeepGEMM to decode batches |
| `0149-MUSA-pass-the-SwiGLU-clamp-through-functional-fused_.patch` | v0.28 0176 continued | pass the SwiGLU clamp through functional fused_experts |
| `0150-MUSA-shorten-the-DFlash-input-prep-padding-tail.patch` | v0.28 0178 continued | shorten the DFlash input-prep padding tail |
| `0151-MUSA-route-DeepSeek-V4-prefill-score-GEMM-to-DeepGEM.patch` | v0.28 0179 continued | route DeepSeek-V4 prefill score GEMM to DeepGEMM |
| `0152-MUSA-split-multi-request-indexer-prefill-chunks-per-.patch` | v0.28 0180 continued | split multi-request indexer prefill chunks per request |
| `0153-MUSA-enable-AWQ-GPTQ-W4A16-via-the-Triton-kernel.patch` | v0.28 0181 continued | enable AWQ/GPTQ W4A16 via the Triton kernel |
| `0154-MUSA-constrain-the-Triton-W4A16-wide-tile.patch` | v0.28 0182 continued | constrain the Triton W4A16 wide tile |
| `0155-MUSA-run-the-Triton-W4A16-kernel-tests-on-MUSA.patch` | v0.28 0183 continued | run the Triton W4A16 kernel tests on MUSA |
| `0156-MUSA-use-a-block-size-the-MUSA-backend-supports.patch` | v0.28 0184 continued | use a block size the MUSA backend supports |
| `0157-MUSA-split-K-on-the-W4A16-kernel-for-decode-speed.patch` | v0.28 0185 continued | split K on the W4A16 kernel for decode speed |
| `0158-MUSA-route-small-M-W4A16-GEMMs-to-the-cheap-unpack.patch` | v0.28 0187 continued | route small-M W4A16 GEMMs to the cheap unpack |
| `0159-MUSA-decode-tile-for-the-widest-W4A16-projection.patch` | v0.28 0188 continued | decode tile for the widest W4A16 projection |
| `0160-MUSA-hook-PaddleOCR-VL-rotary-fast-path.patch` | v0.28 0189 continued | hook PaddleOCR-VL rotary fast path |
| `0161-MUSA-add-minimal-MinerU-Qwen2-VL-rotary-dispatch.patch` | v0.28 0190 continued | add minimal MinerU Qwen2-VL rotary dispatch |
| `0162-MUSA-OvisOCR2-model-local-rotary-route.patch` | v0.28 0191 continued | OvisOCR2 model-local rotary route |
| `0163-MUSA-register-regular-ABI-native-view-op.patch` | v0.32 native adaptation | register regular-ABI native view op |
| `0164-MUSA-select-MUSA-owned-graph-capture-streams.patch` | v0.32 native adaptation | select MUSA-owned graph capture streams |
| `0165-MUSA-accept-MUSA-profiler-activity.patch` | v0.32 native adaptation | accept MUSA profiler activity |
| `0166-MUSA-reuse-FP8-input-scale-during-graph-capture.patch` | v0.32 native adaptation | reuse FP8 input scale during graph capture |
| `0167-MUSA-accept-MUSA-online-quantization-capability.patch` | v0.32 native adaptation | accept MUSA online quantization capability |
| `0168-MUSA-skip-unavailable-MUSA-Triton-warmup-hook.patch` | v0.32 native adaptation | skip unavailable MUSA Triton warmup hook |
| `0169-MUSA-export-attention-compatibility-helpers.patch` | v0.32 native adaptation | export attention compatibility helpers |
| `0170-MUSA-forward-attention-capture-metadata-and-allocate.patch` | v0.32 native adaptation | forward attention capture metadata and allocate separate KV pools |
| `0171-MUSA-skip-unsafe-v2-warmup-for-Qwen3.5-family.patch` | v0.32 new/updated | skip unsafe v2 warmup for Qwen3.5 family |
| `0172-MUSA-split-chained-boolean-operators-in-Triton-kerne.patch` | v0.32 new/updated | split chained boolean operators in Triton kernels |
| `0173-MUSA-zero-padded-heads-in-the-BF16-sparse-MLA-prefil.patch` | v0.32 new/updated | zero padded heads in the BF16 sparse MLA prefill |
| `0174-MUSA-carry-the-DeepSeek-V4-MUSA-paths-onto-the-atten.patch` | v0.32 new/updated | carry the DeepSeek-V4 MUSA paths onto the attention layer |
| `0175-MUSA-fix-warmup-and-runner-helper-references.patch` | v0.32 new/updated | fix warmup and runner helper references |
| `0176-MUSA-keep-MoE-shared-experts-on-the-compute-stream.patch` | v0.32 new/updated | keep MoE shared experts on the compute stream |
| `0177-MUSA-keep-V2-only-configurations-on-Model-Runner-V2.patch` | v0.32 new/updated | keep V2-only configurations on Model Runner V2 |
| `0178-MUSA-compile-the-watermark-Philox-helper-on-Triton-3.patch` | v0.32 new/updated | compile the watermark Philox helper on Triton 3.2 |
| `0179-MUSA-enable-DeepSeek-V4.1-Engram-on-MUSA.patch` | v0.32 new/updated | enable DeepSeek-V4.1 Engram on MUSA |
| `0180-MUSA-route-DeepSeek-V4.1-attention-through-the-MUSA-.patch` | v0.32 new/updated | route DeepSeek-V4.1 attention through the MUSA paths |
| `0181-MUSA-route-the-delayed-mHC-pre-blocks-to-the-MUSA-pr.patch` | v0.32 new/updated | route the delayed mHC pre blocks to the MUSA provider |
| `0182-MUSA-load-the-DeepSeek-V4.1-ratio-2-compressor-rows-.patch` | v0.32 new/updated | load the DeepSeek-V4.1 ratio-2 compressor rows separately |
| `0183-MUSA-load-DeepSeek-V4-FP8-experts-with-their-own-blo.patch` | v0.32 new/updated | load DeepSeek-V4 FP8 experts with their own block size |
| `0184-MUSA-bound-the-exact-indexer-cache-rows-to-each-bloc.patch` | v0.32 new/updated | bound the exact indexer cache rows to each block |
| `0185-MUSA-claim-DeepSeek-V4.1-checkpoints-for-the-DeepSee.patch` | v0.32 new/updated | claim DeepSeek-V4.1 checkpoints for the DeepSeek-V4 FP8 config |
| `0186-MUSA-load-the-MoE-top-k-gating-rows-element-by-eleme.patch` | v0.32 new/updated | load the MoE top-k gating rows element by element |
| `0187-MUSA-score-decode-indexer-keys-of-any-cache-block-si.patch` | v0.32 new/updated | score decode indexer keys of any cache block size in graphs |
| `0188-MUSA-score-prefill-indexer-rows-in-batches.patch` | v0.32 new/updated | score prefill indexer rows in batches |
| `0189-MUSA-run-fine-block-FP8-linears-through-the-MUSA-blo.patch` | v0.32 new/updated | run fine-block FP8 linears through the MUSA block GEMV |
| `0190-MUSA-score-prefill-indexer-rows-with-the-FP8-MQA-log.patch` | v0.32 new/updated | score prefill indexer rows with the FP8 MQA logits kernel |
| `0191-MUSA-score-decode-indexer-keys-with-FP8-dot-products.patch` | v0.32 new/updated | score decode indexer keys with FP8 dot products |
| `0192-MUSA-select-indexer-top-k-with-the-staged-radix-kern.patch` | v0.32 new/updated | select indexer top-k with the staged radix kernel |
| `0193-MUSA-insert-the-DeepSeek-V4.1-DSpark-context-KV-with.patch` | v0.32 new/updated | insert the DeepSeek-V4.1 DSpark context KV with the MUSA op |
| `0194-MUSA-fold-the-DeepSeek-V4-shared-expert-into-the-rou.patch` | v0.32 new/updated | fold the DeepSeek-V4 shared expert into the routed FP8 experts |
| `0195-MUSA-run-the-DeepSeek-V4.1-Engram-wkv-projection-row.patch` | v0.32 new/updated | run the DeepSeek-V4.1 Engram wkv projection row-parallel |
| `0196-MUSA-share-the-DeepSeek-V4.1-decode-plan-across-KV-c.patch` | v0.32 new/updated | share the DeepSeek-V4.1 decode plan across KV-cache groups |
| `0197-MUSA-let-the-DeepSeek-V4-router-read-the-BF16-gate-o.patch` | v0.32 new/updated | let the DeepSeek-V4 router read the BF16 gate output |
| `0198-MUSA-map-the-DeepSeek-V4.1-decode-top-k-once-per-ind.patch` | v0.32 new/updated | map the DeepSeek-V4.1 decode top-k once per index source |
| `0199-MUSA-let-adaptive-verification-trim-the-flattened-DS.patch` | v0.32 new/updated | let adaptive verification trim the flattened DSA indexer decode |
| `0200-MUSA-profile-adaptive-verification-costs-in-the-real.patch` | v0.32 new/updated | profile adaptive verification costs in the real graph capture |
| `0201-MUSA-keep-DeepSeek-V4-aux-stream-overlap-in-FULL-dec.patch` | v0.32 new/updated | keep DeepSeek-V4 aux-stream overlap in FULL decode graphs |
| `0202-MUSA-run-on-Python-3.10.patch` | v0.32 new/updated | run on Python 3.10 |
| `0203-MUSA-keep-the-allocator-settings-while-loading-weigh.patch` | v0.32 new/updated | keep the allocator settings while loading weights |
| `0204-MUSA-number-warmup-blocks-within-each-dedicated-Mamb.patch` | v0.32 new/updated | number warmup blocks within each dedicated Mamba pool |

| `0205-MUSA-run-the-DiffusionGemma-read-tests-on-MUSA.patch` | v0.32 upstream update | run the DiffusionGemma read tests on MUSA |
| `0206-MUSA-size-the-dedicated-Mamba-pool-per-align-mode-sn.patch` | v0.32 upstream update | size the dedicated Mamba pool per align-mode snapshot |
| `0207-MUSA-optimize-hybrid-attention-page-size-fastpath.patch` | v0.32 upstream update | optimize hybrid attention page-size fastpath |
| `0208-MUSA-handle-expandable-segments-around-graph-capture.patch` | v0.32 upstream update | handle expandable segments around graph capture |
| `0209-MUSA-route-Mamba2-SSD-to-FlashInfer.patch` | v0.32 new | route the Mamba2 SSD/`--mamba-backend flashinfer` prefill to MATE's `flashinfer.mamba` |
| `0210-MUSA-initialize-the-Mamba2-profiling-output.patch` | v0.32 new | initialize the supplied Mamba2 output during the profiling run |
| `0211-MUSA-one-contiguous-block-pool-per-Mamba-state.patch` | v0.32 new | give each Mamba state its own contiguous block pool |

Currently **211 patches** and **0 upstream entries** in `upstream/`.

The candidate is rebased onto vllm-musa `a1153f9f` with the same immutable
vLLM pin. Independent application of upstream's original 58-patch chain and
the 210-patch rebase described above gives tree
`a0f85ed14689123a7ee7963d1a4c867cf0bb3dcd`.
This is source equivalence, not a MUSA runtime-readiness claim. Offline replay,
round-trip, metadata and source-contract validation must be repeated after edits.

The round-trip holder is produced by applying this exact series to a clean
checkout at `VLLM_COMMIT`; an already-patched v0.32 checkout with a different
commit grouping is not a valid holder for that check. To verify the stack, use a
**pristine** checkout or archive of the pinned vLLM commit with
`python3.10 tools/musa_sync.py verify --repo <pristine-vllm>`. The verifier checks
every patch in order; an already-patched tree is not a valid input for that
command. A strict replay is `git apply --check` followed by `git apply` for each
patch in filename order. Compare the resulting source hashes with the runtime
import tree when reporting hardware evidence.

## Upstream commits and pin bumps

Step-by-step commands, conflict handling and the checks to run are in
`docs/mdm-developer-guide.md`: §4a backports an upstream commit, §5 bumps the pin.

`upstream/` holds upstream vLLM commits that the pin predates. The build and
`rebase` apply them after the MUSA entries (`0001…`, then `upstream/0001…`),
in the order they were generated, so adding or dropping one changes no MUSA
entry. Each directory is numbered from `0001` on its own. vLLM v0.32.0
contains every upstream commit the series needs, so `upstream/` is empty.

An upstream entry is a `git cherry-pick -x` of its upstream commit: the
message keeps the upstream subject, whose `(#N)` names the pull request, and
git's `(cherry picked from commit <sha>)` line. An `upstream/` entry without
that line is a MUSA fix for the upstream commit before it. `check-series`
fails a MUSA entry with the line, and an `upstream/` that does not start with
one. `report` lists the upstream entries.

Add one with `musa_sync backport <sha> [--from <vLLM clone>]` while
`third_party/vllm` holds the series (after `musa_sync rebase`). It
cherry-picks the commit on top of the series and regenerates it. After
resolving a cherry-pick conflict, or amending the commit after `--no-regen`,
run `musa_sync regen`. A MUSA commit made on top of the clone stays a MUSA
entry unless it changes a file that an upstream entry changes; such a commit
stays in `upstream/` after them, as a fix. Backport a commit's upstream
prerequisites first, oldest first. When entries are added or removed, update
the counts above; the tests check them.

A pin bump:

1. Run `musa_sync upgrade-plan <new ref> --history <full vLLM clone>`. For each
   entry it prints `keep`, `drop` or `review` with the evidence, and fails
   while anything needs review. An upstream entry counts as present when the
   new ref contains:
   - its commit;
   - a commit whose subject ends in its `(#N)`, which is how release branches
     carry it;
   - a `(cherry picked from commit …)` of it;
   - a commit with the same `git patch-id --stable`.

   It is dropped when it changes what its upstream commit changes, and
   reviewed when it carries local changes or the change was reverted. A MUSA
   fix in `upstream/` is reviewed unless every upstream entry before it is
   kept. The series is also
   applied in order to the pristine new ref. A MUSA entry whose change is
   already there is flagged for review, and the result settles any upstream
   entry the history cannot place.
2. While `third_party/PINS` still names the old pin, run `musa_sync rebase
   <new ref> --drop-upstreamed --history <clone>`. It skips the `drop`
   entries and replays the rest with `git am -3`. To drop a `review` entry
   too, delete its file first. A conflict stops at that entry; the guide's §5
   shows how to resolve it and apply the rest.
3. Set `VLLM_COMMIT` and `VLLM_TAG` in `third_party/PINS` to the new ref, then
   run `musa_sync regen`. `regen` exports the commits after the `PINS` commit,
   so it must name the ref the series now sits on.

Without a target, `upgrade-plan` checks the current pin. It lists anything
the series still carries that the pin already contains.
