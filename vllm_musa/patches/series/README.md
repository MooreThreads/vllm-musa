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
  stale files. Historical duplicate numeric prefixes remain until the next full
  regeneration; count the `.patch` files rather than inferring the count from
  the highest prefix. Author headers are normalized to the synthetic
  `musa <musa@local>` identity.

Currently **114 patches**. This branch includes the Qwen3.6 patches for common
GDN decode metadata reuse, uniform-decode SSM slot-mapping removal, and the
BF16 W1 tile specialization, plus the contract-bound DeepSeek-V4 MTP
sparse-prefill headroom and mixed-prefill queue-fence patches. It additionally
adds Qwen3.5-122B/Qwen3-VL MM encoder FlashAttention routing, TP-only shared
expert folding and shared-gate binding, QK/mRoPE cache-out fusion, and opt-in
vision-block graph capture. The series contains
MUSA source edits against the immutable vLLM commit recorded as `VLLM_COMMIT`
in `third_party/PINS` (release label `v0.24.0`), applied at build. Runtime
object/registration patches (which patch live objects at import) are kept
separately in `vllm_musa/patches/`, not in this build-time series. Run
`python3 tools/musa_sync.py verify` to replay and verify the complete manifest
against that exact pinned commit.
