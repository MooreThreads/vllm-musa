# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: I001

import torchada  # noqa: F401

from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
    TritonExperts as UpstreamTritonExperts,
)

from vllm_musa.model_executor.layers.fused_moe import (
    unquantized_fused_moe_method,
)


def test_triton_experts_uses_canonical_upstream_module() -> None:
    assert unquantized_fused_moe_method.TritonExperts is UpstreamTritonExperts
