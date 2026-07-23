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
  a complete replacement, so filenames always form one contiguous `0001`–`NNNN`
  sequence and patches removed from the commit stack cannot leave stale files.
  Author headers are normalized to the synthetic `musa <musa@local>` identity.

Currently **91 patches** — the MUSA source edits against the immutable vLLM commit
recorded as `VLLM_COMMIT` in `third_party/PINS` (release label `v0.24.0`), applied
at build. Runtime object/registration patches (which patch live objects at import)
are kept separately in `vllm_musa/patches/`, not in this build-time series. Run
`python3 tools/musa_sync.py verify` to replay and verify the complete manifest
against that exact pinned commit.
