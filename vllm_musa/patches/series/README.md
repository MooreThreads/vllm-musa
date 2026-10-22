# vLLM v0.32.0 build-time patch series

The files in this directory apply to the immutable vLLM commit named by
`VLLM_COMMIT` in `third_party/PINS` (`5df63a7203cdb9328d824a27b682e77c7b9928be`,
release tag `v0.32.0`, the head of `releases/v0.32.0`). `setup.py` applies
them in filename order before the editable or wheel install. They are source patches; runtime registrations in
`vllm_musa/patches/` are separate.

| Patch | Purpose |
| --- | --- |
| `0001-MUSA-base-platform-port.patch` | Base MUSA adaptation: the native build (the regular-ABI `vllm._C` view op, the regular-ABI `_moe_C` MoE kernels, and the `_C_cache_ops` and `merge_attn_states` registrations in `_C_stable_libtorch`), attention, model runner, and KV cache paths. |
| `0002-MUSA-skip-unsafe-v2-warmup-for-Qwen3.5-family.patch` | Skip the V2 dummy-decode warmup on MUSA Qwen3.5-family models; preserve CUDAGraph capture and semantic output. |
| `0003-MUSA-preserve-dedicated-Mamba-prefix-cache.patch` | Port the applicable v0.28 `0138` dedicated Mamba pool prefix-cache changes: cache Mamba groups first and keep hashed and null states across frees; upstream already routes block frees by `block.pool`. |
| `0004-MUSA-rotate-V2-output-copy-streams.patch` | Port the v0.28 `0172` per-step V2 output-copy streams. |
| `0005-MUSA-skip-Prometheus-observe-for-connectors-without-.patch` | Port the v0.28 `0173` `MultiConnector` metrics fix: skip the Prometheus observe for a connector that reports transfer stats without registering metrics (e.g. `MooncakeConnector`). |
| `0006-perf-musa-extend-DSV4-score-DeepGEMM-to-decode-batch.patch` | Port the v0.28 `0174` DeepSeek-V4 FP32-output score projections to DeepGEMM up to the multi-stream token threshold. |
| `0007-MUSA-pass-the-SwiGLU-clamp-through-functional-fused_.patch` | Port the v0.28 `0176` `gemm1_clamp_limit` forwarding from the functional `fused_experts` entry point to the SiLU-and-mul activation. |
| `0008-MUSA-route-DSV4-decode-indexer-top-k-to-its-own-op.patch` | Port the v0.28 `0177` DeepSeek-V4 decode indexer top-k op; GLM-5.2 rows keep the shared op. |
| `0009-MUSA-shorten-the-DFlash-input-prep-padding-tail.patch` | Port the v0.28 `0178` fixed 1024-wide DFlash input-prep padding blocks. |
| `0010-MUSA-route-DeepSeek-V4-prefill-score-GEMM-to-DeepGEM.patch` | Port the v0.28 `0179` contract-gated DeepGEMM route for DeepSeek-V4 score GEMMs above the multi-stream token threshold. |
| `0011-MUSA-split-multi-request-indexer-prefill-chunks-per-.patch` | Port the v0.28 `0180` split of multi-request sparse-indexer prefill chunks into single-request chunks before the per-row fallback. |
| `0012-MUSA-split-chained-boolean-operators-in-Triton-kerne.patch` | Nest the three-operand conditions in `compute_tile_loop_bounds` and the DeepSeek-V3.2 fused norm-RoPE kernel, which MUSA Triton 3.2 rejects as chained boolean operators. |
| `0013-MUSA-zero-padded-heads-in-the-BF16-sparse-MLA-prefil.patch` | Zero the padded query heads of the BF16 sparse MLA prefill so stale values cannot reach the MUSA kernel. |
| `0014-MUSA-carry-the-DeepSeek-V4-MUSA-paths-onto-the-atten.patch` | Apply the MUSA DeepSeek-V4 attention and sparse-indexer paths on the upstream structure: the q-lora split, wq_b projection, `compress_ratio` and `skip_compressor` are kept; graph decode selects the learned indexer top-k with the native kernel; C4 indexer compression strips `launch_pdl`. |
| `0015-MUSA-fix-warmup-and-runner-helper-references.patch` | Model Runner V1 kernel warmup, the V1 KV-cache allocation context and the MUSA sparse-indexer schedule metadata (`num_states`) follow the upstream definitions. |
| `0016-MUSA-keep-MoE-shared-experts-on-the-compute-stream.patch` | Keep MoE shared experts on the compute stream on MUSA: graph capture runs on a torch_musa-owned stream that the cached `current_stream()` does not track, so the aux-stream fork would escape the capture. |
| `0017-MUSA-keep-V2-only-configurations-on-Model-Runner-V2.patch` | DeepSeek-V4 and the other MRV1-default architectures stay on Model Runner V1 on MUSA unless the configuration needs a V2-only feature (DSpark, prefill context parallel, ...), which keeps the V2 default. |
| `0018-MUSA-compile-the-watermark-Philox-helper-on-Triton-3.patch` | The V2 rejection sampler's resample kernel references the watermark Philox helper; MUSA Triton 3.2 has no `tl.mul`, so the helper uses the wrapping `uint32` product. |
| `0019-MUSA-enable-DeepSeek-V4.1-Engram-on-MUSA.patch` | Create and accept the Engram configuration on MUSA, and load power-of-two float32 Engram scales as their ue8m0 exponent bytes. |
| `0020-MUSA-route-DeepSeek-V4.1-attention-through-the-MUSA-.patch` | DeepSeek-V4.1 attention takes the MUSA FlashMLA ops, stream-wait overlaps, serialized long prefill, the fp32-output score helper, and the MUSA fused RoPE/KV insert without the per-head q norm. |
| `0021-MUSA-route-the-delayed-mHC-pre-blocks-to-the-MUSA-pr.patch` | The delayed mHC pre block and the fused post + delayed pre pair go through the MUSA MHC provider. |
| `0022-MUSA-load-the-DeepSeek-V4.1-ratio-2-compressor-rows-.patch` | The ratio-2 compressor loads its two pooled rows as separate tiles on MUSA instead of joining their pointers. |
| `0023-MUSA-load-DeepSeek-V4-FP8-experts-with-their-own-blo.patch` | FP8 routed experts use the block size of the checkpoint (or `expert_weight_block_size`), so 128x128 experts load beside 32x32 linears; DeepSeek-V4.1 resolves the expert dtype through the quant config. |
| `0024-MUSA-bound-the-exact-indexer-cache-rows-to-each-bloc.patch` | The MUSA exact indexer views its fp8 cache one row per block over that block's own bytes; a shared-page block stride (DeepSeek-V4.1) no longer runs the last row past the cache. |
| `0025-MUSA-claim-DeepSeek-V4.1-checkpoints-for-the-DeepSee.patch` | The DeepSeek-V4 FP8 config also claims DeepSeek-V4.1 model types, so their FP8 checkpoints get the DeepSeek expert dispatch instead of the generic FP8 config. |
| `0026-MUSA-load-the-MoE-top-k-gating-rows-element-by-eleme.patch` | The softmax and softplus-sqrt top-k gating kernels load each row element by element on MUSA; the vectorized loads ignore the scores for 320, 384 and 448 experts (DeepSeek-V4.1). |
| `0027-MUSA-score-decode-indexer-keys-of-any-cache-block-si.patch` | Decode indexer top-k scores cache blocks the materialized scorer does not take (DeepSeek-V4.1's 128-entry ratio-1 blocks) with a graph-safe Triton paged-logits kernel plus the radix top-k; a capture with no graph-safe path fails instead of selecting no keys. |
| `0028-MUSA-score-prefill-indexer-rows-in-batches.patch` | The exact MUSA prefill indexer fallback scores row chunks against all keys with each row's window masked and one top-k per chunk, instead of one einsum, top-k and two host syncs per query row. |
| `0029-MUSA-run-fine-block-FP8-linears-through-the-MUSA-blo.patch` | The MUSA block-FP8 linear list tries the MUSA block-GEMV kernel before Triton; it takes scale blocks finer than DeepGEMM's 128 (DeepSeek-V4.1's 32x32) and runs a GEMV for small M. |
| `0030-MUSA-score-prefill-indexer-rows-with-the-FP8-MQA-log.patch` | The MUSA prefill indexer fallback gathers the FP8 keys and scales without dequantizing, scores them with the FP8 MQA logits kernel and selects each row's window with the windowed top-k kernel; the FP32 path remains the fallback. |
| `0031-MUSA-score-decode-indexer-keys-with-FP8-dot-products.patch` | The graph-safe MUSA decode paged-logits kernel scores each key tile against all heads with one dot product on the FP8 operands (any cache block size); fewer than 16 heads keep the per-head reduction. |
| `0032-MUSA-select-indexer-top-k-with-the-staged-radix-kern.patch` | MUSA indexer top-k up to 512 uses the staged radix select for decode rows and eager prefill windows that start at zero; it returns the same index set and order as the generic kernel, which the other windows keep. |
| `0033-MUSA-insert-the-DeepSeek-V4.1-DSpark-context-KV-with.patch` | DSpark inserts the target's context KV into the draft layers' sliding-window cache, and the decoder replay writes the last KV-source layer's KV, with the MUSA fused q-norm/RoPE/KV insert; the fused CUDA insert op is not built on MUSA. |
| `0034-MUSA-fold-the-DeepSeek-V4-shared-expert-into-the-rou.patch` | A TP-only DeepSeek-V4 MoE with sqrt-softplus routing passes its shared expert to vllm-musa instead of the MoE runner: the shared projections join the routed FP8 expert stack after loading and the router gives every token one extra slot on them at weight 1.0. |
| `0035-MUSA-run-the-DeepSeek-V4.1-Engram-wkv-projection-row.patch` | Each tensor-parallel rank already looks up a contiguous slice of the Engram hash heads, so on MUSA the Engram `wkv` projection runs row-parallel on that slice and one all-reduce sums the partial key and value rows; elsewhere it stays column-parallel. |
| `0036-MUSA-share-the-DeepSeek-V4.1-decode-plan-across-KV-c.patch` | Every KV-cache group's sparse-SWA metadata builder makes its own FlashMLA plan objects; the DeepSeek-V4.1 decode attention reuses the forward pass's first plan for a layer type and cache layout, so the tile-scheduler planner runs once per layer type instead of once per group. |
| `0037-MUSA-let-the-DeepSeek-V4-router-read-the-BF16-gate-o.patch` | On MUSA the non-hash DeepSeek-V4 MoE gates keep their BF16 output; the MUSA biased router widens it exactly, so routing runs no FP32 cast kernel. Hash layers and the upstream fallback route on FP32 logits. |
| `0038-MUSA-map-the-DeepSeek-V4.1-decode-top-k-once-per-ind.patch` | The DeepSeek-V4.1 layers after an index source read its top-k indices through the same compressed cache until the next source; the MUSA decode attention maps them to global KV slots once per index source and cache in a forward pass instead of once per layer. |
| `0039-MUSA-let-adaptive-verification-trim-the-flattened-DS.patch` | On MUSA the flattened DSA indexer decode accepts device-decided verify lengths: each verify token is its own indexer row built from device metadata, so adaptive verification can trim DSpark verify batches. |
| `0040-MUSA-profile-adaptive-verification-costs-in-the-real.patch` | The CUDA graph memory profile skips the adaptive verification step timings: as eager dummy runs inside that pass, their cached activations would count as graph memory and shrink the KV cache. The real capture profiles them. |
| `0041-MUSA-enable-AWQ-GPTQ-W4A16-via-the-Triton-kernel.patch` | AWQ and GPTQ 4-bit checkpoints run through the Triton W4A16 kernel on MUSA: the WNA16 kernel selector offers it to the out-of-tree MUSA platform, and the AWQ/GPTQ configs accept the MUSA capability and skip the Marlin check. |
| `0042-MUSA-constrain-the-Triton-W4A16-wide-tile.patch` | Above 64 rows the MUSA W4A16 GEMM uses a 128x64x32 tile; MUSA Triton 3.2 computes the `BLOCK_N=128` tile incorrectly. |
| `0043-MUSA-run-the-Triton-W4A16-kernel-tests-on-MUSA.patch` | The Triton W4A16 kernel tests run on MUSA. |
| `0044-MUSA-use-a-block-size-the-MUSA-backend-supports.patch` | The test runner's default KV block size is 64 on MUSA, whose attention backend needs a multiple of 64. |
| `0045-MUSA-split-K-on-the-W4A16-kernel-for-decode-speed.patch` | Up to 64 rows the MUSA W4A16 GEMM splits K into up to eight group-aligned pieces, chosen from the static shape, and sums them in FP32, so the narrow decode projections expose enough programs. |
| `0046-MUSA-route-small-M-W4A16-GEMMs-to-the-cheap-unpack.patch` | Every MUSA W4A16 GEMM up to 64 rows takes the split kernel and its reshape unpack; with one piece it writes the output directly. |
| `0047-MUSA-decode-tile-for-the-widest-W4A16-projection.patch` | MUSA W4A16 GEMMs up to 16 rows with `N >= 8192` use a 16x64 tile with `BLOCK_K=64`. |
| `0048-MUSA-hook-PaddleOCR-VL-rotary-fast-path.patch` | PaddleOCR-VL asks the optimization contract for the MUSA rotary path at model construction, for its exact vision and language geometry. |
| `0049-MUSA-add-minimal-MinerU-Qwen2-VL-rotary-dispatch.patch` | MinerU Qwen2-VL selects the shared MUSA rotary path for its exact vision and language geometry. |
| `0050-MUSA-OvisOCR2-model-local-rotary-route.patch` | OvisOCR2 installs the MUSA rotary path on the selected Qwen3.5 vision blocks with the exact BF16 geometry. |
| `0051-MUSA-keep-DeepSeek-V4-aux-stream-overlap-in-FULL-dec.patch` | DeepSeek-V4 and DeepSeek-V4.1 attention overlap their aux streams only in FULL CUDA graphs, which hold decode and verify batches; prefill and mixed steps run serially, because overlapping them can stall the compute queue until the GPU run timeout fires. |
| `0052-MUSA-run-on-Python-3.10.patch` | The MUSA image runs Python 3.10: the package metadata accepts it, and `vllm/__init__.py` backfills the Python 3.11 `typing` names and `datetime.UTC` from `typing_extensions` before any other vLLM module loads. |
| `0053-MUSA-keep-the-allocator-settings-while-loading-weigh.patch` | The worker's allocator split-size cap during weight loading uses the CUDA/ROCm accelerator allocator-settings API; MUSA keeps its allocator settings. |
| `0054-MUSA-number-warmup-blocks-within-each-dedicated-Mamb.patch` | The V2 kernel warmups number the blocks of each dedicated MUSA Mamba pool from 1 inside that pool, so hybrid Mamba models warm up without writing state past it; groups in the shared KV pool draw from one counter. |
| `0055-MUSA-size-a-multi-group-attention-pool-from-the-layo.patch` | The separated-Mamba-pool allocator keeps the per-slot attention tensors when a model has a single attention group (so MTP and non-speculation hybrids are untouched) and builds them from the resolved layout strides when a speculative-decoding drafter owns its own attention group, so DFlash2's draft page size no longer makes a KV view run past its tensor. |

Currently **55 patches** and **0 upstream entries** in `upstream/`.

Keep each patch as a `git format-patch` artifact with its `index` lines. To
verify the stack, use a **pristine** checkout or archive of the pinned vLLM
commit with `python3.10 tools/musa_sync.py verify --repo <pristine-vllm>`.
The verifier checks every patch in order; an already-patched tree is not a
valid input for that command. A strict replay is `git apply --check` followed
by `git apply` for each patch in filename order. Compare the resulting source
hashes with the runtime import tree when reporting hardware evidence.

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
