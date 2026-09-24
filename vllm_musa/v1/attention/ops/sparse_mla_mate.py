# SPDX-License-Identifier: Apache-2.0
"""GLM DSA sparse-MLA prefill adapter backed by MATE TileLang."""

from __future__ import annotations

from mate.sparse_mla.tilelang.sparse_mla_prefill import sparse_mla_prefill_fwd


def sparse_mla_fwd_bf16(q, kv, indices, sm_scale, d_v=512, topk_length=None):
    """Run the MATE v32 prefill kernel for the GLM DSA layout.

    The caller gates this adapter on the GLM DSA architecture and on the
    validated 64-head/576-wide/2048-index contract.  Keeping it in a separate
    module prevents the MATE dependency and its layout from changing the
    generic sparse-MLA TileLang implementation used by other models.
    """
    return sparse_mla_prefill_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=d_v,
        topk_length=topk_length,
    )[0]
