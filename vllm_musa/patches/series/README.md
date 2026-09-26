# `vllm_musa/patches/series/` — build-time patch series

**THE** vLLM-MUSA source-patch mechanism (no runtime fallback). A `git format-patch`
series of MUSA's source modifications against the immutable upstream revision in
`third_party/PINS` (`VLLM_COMMIT`, with `VLLM_TAG` as its release label), applied
at build to the cloned `third_party/vllm` *before* install so the installed vLLM
is pre-patched.

- **Applied at build** by `setup.py::_apply_musa_patch_series` → `build_apply.py`
  (`git apply`, idempotent `--reverse --check`).
- **Generated/regenerated** by `make -f Makefile.sync format-patches`
  (`git format-patch --no-signature --no-numbered --zero-commit`, keeping `index`
  blob lines so `git am -3` 3-way works across version bumps). Regeneration stages
  a complete replacement so patches removed from the commit stack cannot leave
  stale files. Numeric prefixes are regenerated as one contiguous sequence;
  count the `.patch` files rather than relying on historical patch numbers.
  Author headers are normalized to the synthetic
  `musa <musa@local>` identity.

Currently **186 patches**. This branch includes the Qwen3.6 patches for common
GDN decode metadata reuse, uniform-decode SSM slot-mapping removal, and the
BF16 W1 tile specialization, plus the contract-bound DeepSeek-V4 MTP
sparse-prefill headroom and mixed-prefill queue-fence patches. It additionally
adds Qwen3.5-122B/Qwen3-VL MM encoder FlashAttention routing, TP-only shared
expert folding and shared-gate binding, QK/mRoPE cache-out fusion, and opt-in
vision-block graph capture. It also serializes DeepSeek-V4 long-prefill
attention branches on MUSA while preserving decode/MTP auxiliary-stream
overlap, bounds materialized indexer logits past 4k compressed keys,
restores MUSA component-based memory profiling, and routes the v0.28
DeepSeek-V4 MHC paths through MUSA providers. DeepSeek-V4 graph capture keeps
the learned indexer. The metadata-only recent window, its Q/weight skip, and
the CUDAGraph recent-fill fallback are removed so capture, eager, and native
decode share the same learned indices. Eager decode kernel selection uses request length; CUDAGraph capture
never host-syncs. Flattened DSV4 decode scores through mate paged-MQA
when schedule metadata is available, otherwise the learned native kernel. Auxiliary overlap still uses stream
waits instead of CUDA events. The DSpark additions route context-KV insertion through the MUSA
custom operator and provide typed optional pointers for greedy rejection sampling.
They also honor the resolved FP8 expert dtype when converted checkpoints omit
the HF metadata field. The final five patches adapt the v0.28 Model Runner V2 rejection kernels to MUSA Triton scalar-predicate and
Gumbel-helper contracts without changing the upstream acceptance or resampling
algorithm. DeepSeek-V4 remains on Model Runner V1 by default on MUSA for its
faster FULL_DECODE_ONLY serving path; users and V2-only speculative paths can
still opt into Model Runner V2 explicitly. DeepSeek-V4 512-d sparse C4/C128 compression on decode rows
1..128 is dispatched to a native MUSA kernel, with Triton kept as the
shape fallback. On that same native path the compressor also writes
packed kv/score+ape into the state cache so Triton `save_partial_states`
is skipped. Interleaved MRoPE rebuilds the T/H/W frequency layout with a
strided copy rather than per-channel index arithmetic, which keeps the
per-layer launch count off the eager prefill path. The fused TileLang `hc_head` is
enabled on MUSA by importing TileLang before the eager JIT decorators capture
their module globals. DeepEP shutdown now drops cached handles before native
teardown and supports both explicit `destroy()` and legacy destructor-only
MUSA Buffer implementations.
Mooncake now also accepts MUSA FlashAttention's K/V-first cache layout, using
separate dense K/V regions or a padded-page-aware blocks-first hybrid FA/GDN
view as appropriate without changing the kernel-facing cache format. Hybrid
topology setup skips GDN-style backends without a KV-cache shape when probing
the physical FlashAttention layout. The upstream Mooncake Mamba-pool patch
keeps the dedicated-pool eligibility scoped to MooncakeConnector, and the
following MUSA adaptation preserves Mamba pool ownership through prefix-cache
lookup/store, pin, CoW/deferred-free, reset, and eviction paths.
The unified attention kernel takes its per-token-head scale strides as
`tl.int64` zeros rather than `int | None`, which Triton 3.2 rejects as a
kernel-signature annotation.
The series contains
MUSA source edits against the immutable vLLM commit recorded as `VLLM_COMMIT`
in `third_party/PINS` (release label `v0.28.0`), applied at build. Runtime
object/registration patches (which patch live objects at import) are kept
separately in `vllm_musa/patches/`, not in this build-time series. Run
`python3 tools/musa_sync.py verify` to replay and verify the complete manifest
against that exact pinned commit.

The DiffusionGemma structured-read port (upstream vllm#57250) adds the per-request canvas
fields — `diffusion_seed_canvas`, `diffusion_pinned`, `diffusion_read_only`,
`diffusion_max_steps` and `diffusion_canvas_length` — checked at admission by the ported
`vllm/utils/diffusion.py` helper, plus the async diffusion scheduler tier and the
sampler/input-processor plumbing that carries a seeded canvas and its logprob token ids, so
one denoise step can be read as a distribution.

Upstream's own tests come with it: the state-level read suite in
`tests/v1/sample/test_diffusion_gemma_reads.py` (entry 0181) and the diffusion cases in
`tests/test_sampling_params.py` (entry 0182), gated on the platform's GPU device type rather
than CUDA so they run on MUSA instead of skipping; entry 0182 also adds
`MockModelConfig.return_sampling_mask` because this base's `InputProcessor._validate_params`
reads it. Entries 0183/0184 carry upstream's one-line
`MODEL_ARCH_CONFIG_CONVERTORS["diffusion_gemma"]` registration and its head-dim case — not
cosmetic here, because the served checkpoint has `head_dim=256` against
`global_head_dim=512`. Entries 0185/0186 add the `create_scheduler` diffusion hook and the
canvas-width, selection and narrowing cases for the async tier; three further upstream
scheduler cases are not carried: the two deferral cases are what drive entry 0174's
`or 1` throttle (an uncapped read loses a scheduling slot on alternate calls here, where
upstream leaves it unthrottled), and they drive the tier through a step loop that does not
terminate at this pin without a model; the trimmed-worker-draft case exercises
`update_draft_token_ids_in_output` and needs `_RecordingGrammar`, which this pin lacks; and
`test_diffusion_scheduler_is_selected_by_default` did not complete in an hour-long budget on a
leased S5000 container, where the same run against the unpatched image parks 195 of 196 threads in
`futex_wait_queue_me` at 0% CPU, so it is carried as upstream wrote it rather than claimed as
passing. This port
therefore claims the tier is selected and width-aware, not that it is exercised end to end.

Read-only emission stays where upstream put it — inside the compiled step: it grows the same
`read_only` argument, emits at convergence instead of being scheduled for the commit forward,
clears `is_encoder_phase` for that slot, and stashes and hands out logprobs off
`num_sampled`, so upstream's four
`test_read_emits_at_convergence_while_generation_waits_for_commit` cases run unmodified. The
step also carries upstream's `@torch._dynamo.config.patch(recompile_limit=64)` guard, which
this fork needs because the per-width tile loop adds specializations on top of upstream's set.

Fork-local glue that upstream does not carry is confined to entries 0174, 0177 and 0178: the
per-request draft-token width hook (`ModelState.num_draft_tokens_per_req`,
`DraftTokensHandler.set_draft_tokens`), which a canvas narrower than the served one needs
under spec decode. Diffusion models are pinned to `TRITON_ATTN`. Three upstream changes are
deliberately not carried: the example interposer under
`examples/features/structured_diffusion/` (this fork ships it as an image layer instead, since
`/v1/systemone` is a front end for the API rather than an engine feature), the
comment-and-log rewording in `vllm/model_executor/models/config.py` (no behaviour change), and
the two mock fields in `tests/v1/engine/test_input_processor_trace_replay.py` (that file does
not exist at this pin).

