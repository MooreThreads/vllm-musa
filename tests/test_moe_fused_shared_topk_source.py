# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_unquantized_moe_uses_fused_shared_topk_extension() -> None:
    source = (
        REPO_ROOT
        / "vllm_musa/model_executor/layers/fused_moe/"
        "unquantized_fused_moe_method.py"
    ).read_text()

    assert "from vllm_musa.jit_kernel.extend_topk_shared import" in source
    assert "topk_weights, topk_ids = extend_topk_with_shared(" in source
    assert "VLLM_MUSA_MOE_FUSED_SHARED_TOPK" not in source
    assert "torch.cat([topk_weights, shared_weight]" not in source


def test_fused_shared_topk_preserves_bf16_sigmoid_rounding() -> None:
    source = (
        REPO_ROOT / "vllm_musa/jit_kernel/extend_topk_shared.py"
    ).read_text()

    assert "shared_logits_ptr.dtype.element_ty" in source


def test_fused_shared_topk_has_no_runtime_gate() -> None:
    source = (REPO_ROOT / "vllm_musa/utils/environ.py").read_text()

    assert "VLLM_MUSA_MOE_FUSED_SHARED_TOPK" not in source
