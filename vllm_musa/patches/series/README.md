# `vllm_musa/patches/series/` — build-time patch series

**THE** vLLM-MUSA source-patch mechanism (no runtime fallback). A `git format-patch`
series of MUSA's source modifications against the immutable upstream revision in
`third_party/PINS` (`VLLM_COMMIT`, with `VLLM_TAG` as its release label), applied
at build to the cloned `third_party/vllm` *before* install so the installed vLLM
is pre-patched.

- **Applied at build** by `setup.py::_apply_musa_patch_series` → `build_apply.py`
  (`git apply`, idempotent `--reverse --check`).
- **Generated/regenerated** by `make -f Makefile.sync format-patches`
  (`git format-patch --no-signature --no-numbered --zero-commit` with
  `core.abbrev=10`, keeping `index` blob lines so `git am -3` 3-way works across
  version bumps). Regeneration stages
  a complete replacement so patches removed from the commit stack cannot leave
  stale files. Numeric prefixes are regenerated as one contiguous sequence per
  directory; count the `.patch` files rather than relying on historical patch
  numbers.
  Author headers are normalized to the synthetic
  `musa <musa@local>` identity.

Currently **188 patches**, plus **3 upstream entries** in `upstream/`.
This branch includes the Qwen3.6 patches for common
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
algorithm. DeepSeek-V4 score FP32 projections keep the MUSA DeepGEMM path
up to the multi-stream token threshold, which covers every DSpark-4 decode
batch. On contract-matched DeepSeek-V4 DSpark
deployments, Model Runner V2 replays the uniform decode graph for a remote-
prefilled request's final prompt token padded to the verify shape. The functional
fused_experts entry point forwards a model's SwiGLU clamp limit to the routed
expert activation. DeepSeek-V4 decode rows select their
indexer top-k with a dedicated MUSA op; GLM-5.2 keeps the shared one. The DFlash/DSpark draft input preparation pads its
CUDA-graph buffers in fixed 1024-wide blocks. DeepSeek-V4 remains on Model Runner V1 by default on MUSA for its
faster FULL_DECODE_ONLY serving path; users and V2-only speculative paths can
still opt into Model Runner V2 explicitly. DeepSeek-V4 512-d sparse C4/C128 compression on decode rows
1..320 is dispatched to a native MUSA kernel, with Triton kept as the
shape fallback. On that same native path the compressor also writes
packed kv/score+ape into the state cache so Triton `save_partial_states`
is skipped. Interleaved MRoPE rebuilds the T/H/W frequency layout with a
strided copy rather than per-channel index arithmetic, which keeps the
per-layer launch count off the eager prefill path. Model Runner V2 gives each
in-flight step its own output copy stream so a step's sampled tokens are not
held back by a later step's copy-stream wait. The fused TileLang `hc_head` is
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
`MultiConnector` skips Prometheus observation for a sub-connector that
reports transfer stats without registering metrics, so `MooncakeConnector`
can run alongside `MooncakeStoreConnector`.
The unified attention kernel takes its per-token-head scale strides as
`tl.int64` zeros rather than `int | None`, which Triton 3.2 rejects as a
kernel-signature annotation.
DeepSeek-V4 score GEMMs above the multi-stream token threshold run through
DeepGEMM instead of an FP32 SIMT sgemm.
A sparse-indexer prefill chunk that holds several long requests is split
into single-request chunks, which the MUSA top-k kernels accept, before the
per-row PyTorch fallback.
A KV load failure on a hybrid KV cache, whose groups disagree on block
size, recomputes or fails the whole request instead of raising in the
scheduler. The Mooncake connector reports a failed remote KV load (a
P-side timeout, transfer error or unknown remote engine) to the scheduler
as finished receiving with load-error blocks, so the request fails or
recomputes and its KV blocks are freed. These two entries are the
unmodified upstream vLLM commits `c351fd3c6495` and `2c7ee87223f0` in
`upstream/`, which the pinned `VLLM_COMMIT` predates; the Mooncake entry
needs the scheduler one. The MUSA fix after them only renames the
backported DeepSeek-V4 test spec field to the pin's `compress_ratio`.
The series contains
MUSA source edits against the immutable vLLM commit recorded as `VLLM_COMMIT`
in `third_party/PINS` (release label `v0.28.0`), applied at build. Runtime
object/registration patches (which patch live objects at import) are kept
separately in `vllm_musa/patches/`, not in this build-time series. Run
`python3 tools/musa_sync.py verify` to replay and verify the complete manifest
against that exact pinned commit.

## Upstream commits and pin bumps

Step-by-step commands, conflict handling and the checks to run are in
`docs/mdm-developer-guide.md`: §4a backports an upstream commit, §5 bumps the pin.

`upstream/` holds upstream vLLM commits that the pin predates. The build and
`rebase` apply them after the MUSA entries (`0001…`, then `upstream/0001…`),
in the order they were generated, so adding or dropping one changes no MUSA
entry. Each directory is numbered from `0001` on its own.

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
prerequisites first, oldest first. When entries are added or removed, update the
counts at the top of this file; the tests check them.

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
