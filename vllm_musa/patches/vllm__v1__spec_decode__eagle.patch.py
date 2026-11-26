# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA spec-decode kernel monkey-patch shim.

Loaded by ``vllm_musa.patches`` as ``vllm.v1.spec_decode.eagle``; the
filename's underscore-encoded module path is the patch system's lookup key.

This file used to install ``EagleFullLoopRunner``, a MUSA-only custom runner
that captured the full N-step Eagle3 draft loop as one CUDAGraph. That runner
was deleted in favor of the upstream pattern from vllm-project/vllm#34880:
wrap the draft model with the standard ``CUDAGraphWrapper`` plus per-step
``CudagraphDispatcher``.

This file is now the minimal shim that ensures the MUSA-Triton-adapted
``eagle_prepare_next_token_padded_kernel`` from
``vllm_musa.v1.spec_decode.utils`` is imported before upstream
``vllm.v1.spec_decode.llm_base_proposer`` binds its own copy of the kernel.
Without this prime the proposer's Triton compile fails with
``mismatched type for valid_count``.
"""

PATCHES: list = []


def apply() -> None:
    """Explicit object-patch entry (idempotent).

    Installs the MUSA-Triton-adapted ``eagle_prepare_next_token_padded_kernel``
    from ``vllm_musa.v1.spec_decode.utils`` before upstream
    ``vllm.v1.spec_decode.llm_base_proposer`` binds its own copy (otherwise the
    proposer's Triton compile fails with "mismatched type for valid_count").

    Called explicitly by ``vllm_musa.patches.apply_object_patches()`` at plugin
    load — importing this file does not trigger the hook installation.
    """
    import vllm_musa.v1.spec_decode.utils as _spec_decode_utils

    _spec_decode_utils.install_hooks()
