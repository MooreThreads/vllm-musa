# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
run the Eagle3 draft model at TP=1 (replicated) while the
target model runs at the configured TP (e.g. 8).

Why
---
Profiling (workspace generated/musa0124/PROFILE_draft_target_split.md)
measured the Eagle3 draft forward at ~23.5% of decode-step time on the
M2.5 + Eagle3 narrow-d5 SOTA config — almost entirely cross-rank
AR/barrier overhead. Each draft pass is a 1-layer model (tiny compute)
but, sharded over TP=8, pays full barrier cost 5× per step (one pass
per tree depth). Running the draft at TP=1 — full weights replicated on
every rank, every rank drafts independently — eliminates every draft
all-reduce. The draft's un-sharded compute (~1 ms) is far cheaper than
the ~7-8 ms of barrier overhead removed. Projected: M2.5+Eagle3
99.5 -> ~123 tok/s.

Mechanism
---------
vLLM ships `patch_tensor_parallel_group()` in
`vllm/distributed/parallel_state.py` — a context manager whose docstring
states it is "for draft workers of speculative decoding to run draft
model with different tp degree" — but nothing in the V1 engine calls it.
This patch wires it for MUSA:

  1. Build a per-rank TP=1 `GroupCoordinator` (each rank is its own
     1-process group).
  2. Wrap `SpecDecodeBaseProposer.load_model` so the draft model is
     CONSTRUCTED while `_TP` is the TP=1 group — its linear / attention
     / embedding layers read `get_tensor_model_parallel_world_size()==1`
     at `__init__` and build unsharded.
  3. Wrap `EagleProposer.propose` so the draft FORWARD also executes
     under the TP=1 group.

The torch.compile cache-corruption that the upstream
`draft_model.py` `draft_tp == target_tp` assertion guards against does
NOT apply here: the M2.5 SOTA config already sets
`VLLM_DISABLE_COMPILE_CACHE=1`, so no rank reads or writes the
per-rank compile-cache directory and the write collision cannot occur.

Gating
------
Off by default during bring-up. Enable with `VLLM_MUSA_DRAFT_TP1=1`.
The default flips to on once the perf gate passes.

Patch style
-----------
`PATCHES = []` — no file mutation; the monkey-patch is the import side
effect (same mechanism as the eagle patch). Idempotent via
the `_musa_draft_tp1_patched` class marker.

Filename note: `patches/__init__.py` `apply_object_patches()` derives a
module name from the filename (`__` -> `.`) for the patch report. This file
is therefore named after `vllm.distributed.parallel_state` — a real
module, and the one that provides the `patch_tensor_parallel_group`
context manager this patch wires. It does NOT mutate parallel_state.py
(`PATCHES = []`); the monkey-patch targets the spec-decode proposers.
"""

PATCHES: list = []  # No file mutation; the monkey-patch install is the side effect.

import logging
import os

_log = logging.getLogger(__name__)

# Off by default during bring-up; flip to "1" default once 's
# perf gate passes.
_ENABLED = os.environ.get("VLLM_MUSA_DRAFT_TP1", "0") == "1"

# Lazily-built per-rank TP=1 GroupCoordinator (one 1-process group per rank).
_DRAFT_TP1_GROUP = None


def _get_draft_tp1_group():
    """Build (once) and return this rank's TP=1 GroupCoordinator.

    `group_ranks=[[0],[1],...,[N-1]]` makes every rank its own singleton
    group, so each process's GroupCoordinator has `world_size == 1`.
    """
    global _DRAFT_TP1_GROUP
    if _DRAFT_TP1_GROUP is not None:
        return _DRAFT_TP1_GROUP

    import torch
    from vllm.distributed.parallel_state import (
        get_world_group,
        init_model_parallel_group,
    )

    world_size = torch.distributed.get_world_size()
    backend = torch.distributed.get_backend(get_world_group().device_group)
    group_ranks = [[r] for r in range(world_size)]
    # `init_model_parallel_group(group_ranks, local_rank, ...)` — the
    # second arg is the **device-local rank** used by GroupCoordinator to
    # assign the device (e.g. `cuda:{local_rank}`). It is NOT used to
    # select which entry of `group_ranks` this process belongs to —
    # GroupCoordinator does that internally via
    # `torch.distributed.get_rank()` (the global rank).
    #
    # On a multi-node job (e.g. 2 nodes × 8 GPUs = 16 ranks),
    # `local_rank=0` is passed from both node 0 rank 0 and node 1 rank 8;
    # each picks its own singleton group (`[0]` and `[8]` respectively)
    # via global-rank lookup and assigns device 0 on its own node. See
    # `vllm/distributed/parallel_state.py::GroupCoordinator` docstring
    # for the `local_rank` vs `rank_in_group` distinction. vLLM's own
    # `initialize_model_parallel` uses the same `get_world_group().local_rank`
    # idiom here.
    _DRAFT_TP1_GROUP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=False,
        group_name="musa_draft_tp1",
    )
    _log.info(
        "built per-rank TP=1 draft group (world_size=%d)",
        world_size,
    )
    return _DRAFT_TP1_GROUP


def _is_musa() -> bool:
    try:
        from vllm.platforms import current_platform

        return bool(current_platform.is_musa())
    except Exception:  # pragma: no cover
        return False


def apply() -> None:
    """explicit object-patch entry (was an import-time side effect).

    wire the per-rank TP=1 draft group onto
    ``SpecDecodeBaseProposer.load_model`` + ``EagleProposer.propose``, gated by
    ``VLLM_MUSA_DRAFT_TP1`` (default off; idempotent via the
    ``_musa_draft_tp1_patched`` class marker). Called by
    ``vllm_musa.patches.apply_object_patches()`` at plugin load.
    """
    if not _ENABLED:
        _log.debug("draft-TP=1 patch dormant (set VLLM_MUSA_DRAFT_TP1=1 to enable)")
        return

    # CRITICAL ORDER (same hazard as the eagle patch): importing
    # `vllm.v1.spec_decode.eagle` / `llm_base_proposer` triggers their
    # `from vllm.v1.spec_decode.utils import eagle_prepare_next_token_padded_kernel`.
    # vllm-musa replaces that kernel with a MUSA-Triton-adapted version via
    # `vllm_musa.v1.spec_decode.utils`. If we import the proposer modules
    # before that monkey-patch has fired, they bind the UPSTREAM kernel,
    # which fails at MUSA Triton compile time with
    # "mismatched type for valid_count". This patch file sorts
    # alphabetically before `vllm__v1__spec_decode__eagle.patch.py`, so we
    # must prime the kernel patch ourselves before importing the proposers.
    import vllm_musa.v1.spec_decode.utils as _spec_decode_utils

    _spec_decode_utils.install_hooks()

    try:
        from vllm.v1.spec_decode.eagle import EagleProposer
        from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
    except ImportError as exc:  # pragma: no cover
        _log.debug(
            "draft-TP=1 patch: cannot import proposer classes "
            "(probably an unsupported vllm version). Skipping: %s",
            exc,
        )
        EagleProposer = None  # type: ignore
        SpecDecodeBaseProposer = None  # type: ignore

    if SpecDecodeBaseProposer is not None and not getattr(
        SpecDecodeBaseProposer, "_musa_draft_tp1_patched", False
    ):
        from vllm.distributed.parallel_state import patch_tensor_parallel_group

        # --- wrap draft CONSTRUCTION ---
        _orig_load_model = SpecDecodeBaseProposer.load_model

        def _musa_tp1_load_model(self, target_model):
            if not _is_musa():
                return _orig_load_model(self, target_model)
            tp1 = _get_draft_tp1_group()
            _log.info("constructing draft model under TP=1 context")
            with patch_tensor_parallel_group(tp1):
                return _orig_load_model(self, target_model)

        SpecDecodeBaseProposer.load_model = _musa_tp1_load_model

        # --- wrap draft FORWARD ---
        # EagleProposer.propose may already be wrapped by the
        # patch; we wrap whatever it currently is. Both wrappers only
        # delegate, so nesting order does not matter for correctness.
        if EagleProposer is not None:
            _orig_propose = EagleProposer.propose

            def _musa_tp1_propose(self, *args, **kwargs):
                if not _is_musa():
                    return _orig_propose(self, *args, **kwargs)
                tp1 = _get_draft_tp1_group()
                with patch_tensor_parallel_group(tp1):
                    return _orig_propose(self, *args, **kwargs)

            EagleProposer.propose = _musa_tp1_propose

        SpecDecodeBaseProposer._musa_draft_tp1_patched = True
        _log.info(
            "draft-TP=1 patch installed "
            "(load_model + propose wrapped; VLLM_MUSA_DRAFT_TP1=1)"
        )
