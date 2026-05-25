# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MUSA-0090: Dispatch EagleProposer.propose to the MUSA full-loop graph runner.

This patch installs a wrapper on `vllm.v1.spec_decode.eagle.EagleProposer.propose`
that:
  1. On MUSA: try the EagleFullLoopRunner; fallback to vLLM's iterative path
     on any error.
  2. On CUDA: behave identically to upstream (no-op wrapper).
  3. Lazy-init the runner on the FIRST propose() call (after vLLM has set
     up everything else). Capture happens then; subsequent calls just
     replay.

Architecture:
  - Patches the EagleProposer class via attribute assignment (MONKEY-PATCH).
  - Uses an empty `PATCHES = []` so the parent `patches/__init__.py`'s
    file-mutation framework skips the find/replace pass. The side effects
    of importing this file (the monkey-patch) are what install the
    dispatcher.
  - Idempotent via the `_musa_eagle_patched` class attribute. Re-importing
    is safe — does not stack wrappers.

Rationale for MONKEY-PATCH over PATCHES = [(old, new)]:
  - MUSA-0089 confirmed the file-mutation framework is non-idempotent
    against new-block INSERTIONS (vs string replacements). The
    cuda_communicator.py patch accumulated 10 duplicate blocks across
    re-imports. INSERT semantics need either a marker-aware find pattern
    (fragile) or a different mechanism — monkey-patching is the
    different mechanism.
  - Class-level monkey-patch + marker attribute is exactly-once per
    process; the patches/__init__.py exec_module call from
    `_load_patch_config` runs side effects once, the marker prevents
    re-application within the same process if exec is re-triggered.
"""

PATCHES: list = []  # No file mutation; the dispatcher install is the side effect.

# --- monkey-patch installation (runs at module load via _load_patch_config) ---

import logging

_log = logging.getLogger(__name__)

# CRITICAL ORDER: import vllm_musa.v1.spec_decode.utils BEFORE eagle.
# vllm-musa's utils module installs a MUSA-Triton-adapted version of
# `eagle_prepare_next_token_padded_kernel` on vllm.v1.spec_decode.utils.
# If we import eagle here without first ensuring that monkey-patch has
# fired, eagle.py's `from vllm.v1.spec_decode.utils import ...` will
# bind to the UPSTREAM unpatched kernel, which then fails at MUSA
# Triton compile time with 'mismatched type for valid_count' inside
# eagle_prepare_next_token_padded_kernel. (Diagnosed via isolation test
# 21:41 confirming: with this patch.py disabled, MUSA-0089 v7 config
# boots fine + smoke PASSes; with this patch.py active but without the
# prime import, smoke FAILS in the Triton kernel.)
import vllm_musa.v1.spec_decode.utils  # noqa: F401 — prime the kernel monkey-patch

try:
    from vllm.v1.spec_decode.eagle import EagleProposer
except ImportError as exc:  # pragma: no cover
    _log.debug(
        "MUSA-0090 eagle patch: cannot import vllm.v1.spec_decode.eagle "
        "(probably an unsupported vllm version). Skipping: %s",
        exc,
    )
    EagleProposer = None  # type: ignore


def _maybe_init_runner(proposer, args, kwargs) -> None:
    """Build + capture the EagleFullLoopRunner on the first MUSA propose() call.

    Sets `proposer._musa_full_loop_runner` on success. On any failure,
    re-raises (caller is expected to swallow + mark None to disable
    retries).
    """
    from vllm_musa.v1.spec_decode.eagle_full_loop_runner import EagleFullLoopRunner

    num_speculative_tokens = int(getattr(proposer, "num_speculative_tokens", 0))
    if num_speculative_tokens < 1:
        raise RuntimeError(
            f"num_speculative_tokens={num_speculative_tokens} is invalid; "
            "spec-decode appears not properly configured."
        )

    # MUSA-0090 step 5l.1: capture ONLY for the actual batch_size of the
    # current call. The base_metadata is for `num_reqs=batch_size`; trying
    # to capture for a different size requires synthetic metadata
    # (query_start_loc shape needs to match num_reqs+1, etc.). Capturing
    # for the observed size guarantees the metadata is consistent.
    # Later calls at different batch_sizes will fall back to vllm iterative
    # (cheap — just one cudagraph dispatch overhead) until we re-capture
    # for that size (future work).
    next_token_ids_arg = (
        kwargs.get("next_token_ids")
        if "next_token_ids" in kwargs
        else (args[3] if len(args) > 3 else None)
    )
    if next_token_ids_arg is not None:
        cudagraph_capture_sizes = [int(next_token_ids_arg.shape[0])]
    else:
        cudagraph_capture_sizes = [1]
    _log.info(
        "MUSA-0090: capturing for observed batch_size: %s",
        cudagraph_capture_sizes,
    )

    # Hidden size: probe multiple paths since vllm config layouts vary.
    hidden_size = 0
    mc = getattr(proposer, "vllm_config", None)
    if mc is not None:
        model_config = getattr(mc, "model_config", None)
        if model_config is not None:
            try:
                hidden_size = int(model_config.get_hidden_size())
            except AttributeError:
                hidden_size = int(
                    getattr(getattr(model_config, "hf_config", None), "hidden_size", 0)
                )
    if hidden_size <= 0:
        try:
            hidden_size = int(proposer.model.config.hidden_size)
        except Exception:
            hidden_size = 4096  # M2.5 default; will surface in capture if wrong.
        _log.warning("MUSA-0090: hidden_size probe fell back to %d", hidden_size)

    import torch

    device = getattr(proposer, "device", None) or torch.device("cuda")

    runner = EagleFullLoopRunner(
        proposer=proposer,
        num_speculative_tokens=num_speculative_tokens,
        cudagraph_capture_sizes=cudagraph_capture_sizes,
        hidden_size=hidden_size,
        device=device,
    )

    # Capture (uses the first propose() call's common_attn_metadata as the
    # template for per-step metadata). On capture failure, propagate; the
    # caller marks the runner as None to disable future attempts.
    base_metadata = kwargs.get("common_attn_metadata")
    if base_metadata is None and len(args) >= 5:
        base_metadata = args[4]
    if base_metadata is None:
        raise RuntimeError(
            "MUSA-0090: cannot find common_attn_metadata in propose() args"
        )
    runner.capture(base_metadata)

    proposer._musa_full_loop_runner = runner
    _log.info(
        "MUSA-0090: EagleFullLoopRunner installed (N=%d, capture_sizes=%s, "
        "hidden_size=%d, buffer_footprint=%.2f MiB)",
        num_speculative_tokens,
        cudagraph_capture_sizes,
        hidden_size,
        runner.memory_footprint_bytes() / 1024 / 1024,
    )


def _make_dispatched_propose(_original_propose):
    """Closure-build the patched propose() so the original is captured.

    MUSA-0090 step 5l.4 (2026-05-16): runner architecture works for
    capture but has deeper replay-time correctness issues (KV cache
    page state, MUSA 'unknown error' on graph replay). Pending a full
    SGLang-style per-step-backend refactor, disable the runner
    dispatcher and always fall through to vllm's iterative path.
    Set VLLM_MUSA_EAGLE_RUNNER=1 to re-enable for development.
    """

    import os

    _RUNNER_ENABLED = os.environ.get("VLLM_MUSA_EAGLE_RUNNER", "0") == "1"

    def _musa_dispatched_propose(self, *args, **kwargs):
        if not _RUNNER_ENABLED:
            return _original_propose(self, *args, **kwargs)
        # Only dispatch on MUSA. On CUDA, fall through to the original.
        try:
            from vllm.platforms import current_platform

            if not current_platform.is_musa():
                return _original_propose(self, *args, **kwargs)
        except Exception:
            return _original_propose(self, *args, **kwargs)

        # Lazy-init on first MUSA call.
        if not hasattr(self, "_musa_full_loop_runner"):
            try:
                _maybe_init_runner(self, args, kwargs)
            except Exception as exc:
                _log.warning(
                    "MUSA-0090: full-loop runner init failed (%s); falling "
                    "back to vLLM iterative path for the rest of this run",
                    exc,
                )
                self._musa_full_loop_runner = None

        runner = getattr(self, "_musa_full_loop_runner", None)
        if runner is None:
            return _original_propose(self, *args, **kwargs)

        # Extract args. EagleProposer.propose signature (vllm v0.20.1.dev0):
        #   def propose(self, target_token_ids, target_positions,
        #               target_hidden_states, next_token_ids,
        #               token_indices_to_sample, common_attn_metadata,
        #               sampling_metadata, ...)
        try:
            target_positions = (
                kwargs.get("target_positions")
                if "target_positions" in kwargs
                else args[1]
            )
            target_hidden_states = (
                kwargs.get("target_hidden_states")
                if "target_hidden_states" in kwargs
                else args[2]
            )
            next_token_ids = (
                kwargs.get("next_token_ids") if "next_token_ids" in kwargs else args[3]
            )
            # token_indices_to_sample may be None (proposer computes from
            # query_start_loc) — that's fine; runner.replay() handles it.
            token_indices_to_sample = (
                kwargs.get("token_indices_to_sample")
                if "token_indices_to_sample" in kwargs
                else (args[4] if len(args) > 4 else None)
            )
            common_attn_metadata = (
                kwargs.get("common_attn_metadata")
                if "common_attn_metadata" in kwargs
                else args[5]
            )
        except (IndexError, KeyError):
            _log.debug("MUSA-0090: propose() signature mismatch; fallback to original")
            return _original_propose(self, *args, **kwargs)

        batch_size = int(next_token_ids.shape[0])
        if not runner.can_run(batch_size):
            return _original_propose(self, *args, **kwargs)

        try:
            result = runner.replay(
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                common_attn_metadata=common_attn_metadata,
                batch_size=batch_size,
                target_positions=target_positions,
                token_indices_to_sample=token_indices_to_sample,
            )
            return result.draft_token_ids
        except Exception as exc:
            _log.warning(
                "MUSA-0090: runner.replay failed (%s); falling back to "
                "vLLM iterative path for this call",
                exc,
            )
            return _original_propose(self, *args, **kwargs)

    return _musa_dispatched_propose


# Idempotent install.
if EagleProposer is not None and not getattr(
    EagleProposer, "_musa_eagle_patched", False
):
    _original_propose = EagleProposer.propose
    EagleProposer.propose = _make_dispatched_propose(_original_propose)
    EagleProposer._musa_eagle_patched = True
    _log.info(
        "MUSA-0090: EagleProposer.propose monkey-patched "
        "(dispatcher installed; runner lazy-inits on first MUSA call)"
    )
