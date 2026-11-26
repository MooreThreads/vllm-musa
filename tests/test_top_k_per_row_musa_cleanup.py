# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA registration contract for the top_k_per_row CUDA ops.

The MUSA indexer uses the native ``deepseek_v4_indexer_topk_*`` /
``glm52_indexer_topk_*`` / ``sparse_indexer_topk*`` kernels.  The upstream
``top_k_per_row_prefill/decode`` schemas are intentionally not registered on
MUSA, so callers must not see a schema that cannot dispatch.
"""

import pytest


def _require_musa():
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    pytest.importorskip("vllm_musa")
    from vllm.platforms import current_platform

    if not current_platform.is_musa():
        pytest.skip("MUSA-only registration contract")
    return torch


def test_top_k_per_row_schemas_are_not_registered_on_musa():
    torch = _require_musa()
    import vllm._custom_ops  # noqa: F401  (loads the _C stable fragment)

    assert not hasattr(torch.ops._C, "top_k_per_row_prefill")
    assert not hasattr(torch.ops._C, "top_k_per_row_decode")


def test_native_indexer_top_k_remains_registered_on_musa():
    torch = _require_musa()
    import vllm_musa._custom_ops  # noqa: F401  (loads _C_musa_ops)

    assert hasattr(torch.ops, "_C_musa_ops")
    assert hasattr(torch.ops._C_musa_ops, "sparse_indexer_topk_decode")
