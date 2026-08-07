# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_unquantized_moe_keeps_shared_topk_extension_fallback() -> None:
    source = (
        REPO_ROOT
        / "vllm_musa/model_executor/layers/fused_moe/"
        "unquantized_fused_moe_method.py"
    ).read_text()

    assert "from vllm_musa.jit_kernel.extend_topk_shared import" in source
    assert "topk_weights, topk_ids = extend_topk_with_shared(" in source
    assert "VLLM_MUSA_MOE_FUSED_SHARED_TOPK" not in source
    assert "torch.cat([topk_weights, shared_weight]" not in source


def test_plain_router_moves_shared_gate_before_jit_topk() -> None:
    source = (
        REPO_ROOT
        / "vllm_musa/model_executor/layers/fused_moe/router/"
        "fused_topk_router.py"
    ).read_text()

    assert 'shared_gate = getattr(self, "_musa_shared_gate", None)' in source
    assert "shared_logits, _ = shared_gate(hidden_states)" in source
    assert "shared_expert_gate_output=shared_logits" in source
    assert "num_fused_shared_experts=1 if shared_logits is not None else 0" in source


def test_jit_topk_writes_shared_column_with_input_dtype_rounding() -> None:
    source = (
        REPO_ROOT / "vllm_musa/jit_kernel/csrc/topk/topk_gating.mu"
    ).read_text()

    assert "topk_softmax_no_bias_warp_shared1_kernel_fixed_k" in source
    assert "const SharedT shared_weight" in source
    assert "from_float<SharedT>(stable_sigmoid" in source
    assert "topk_ids[shared_out_idx] = NumExperts" in source
    assert "if (renormalize)" in source
    assert "TVM_FFI_ICHECK(!renormalize)" not in source


def test_fused_shared_topk_preserves_bf16_sigmoid_rounding() -> None:
    source = (
        REPO_ROOT / "vllm_musa/jit_kernel/extend_topk_shared.py"
    ).read_text()

    assert "shared_logits_ptr.dtype.element_ty" in source


def test_fused_shared_topk_has_no_runtime_gate() -> None:
    source = (REPO_ROOT / "vllm_musa/utils/environ.py").read_text()

    assert "VLLM_MUSA_MOE_FUSED_SHARED_TOPK" not in source


def test_qwen35_tp_only_fold_uses_explicit_ep_setting() -> None:
    patch = (
        REPO_ROOT
        / "vllm_musa/patches/series/"
        "0104-MUSA-enable-qwen35-shared-fold-tp-only.patch"
    ).read_text()

    assert "+            and not parallel_config.enable_expert_parallel" in patch
    assert "-            and self.ep_size <= 1" in patch


def test_qwen35_fold_stashes_owning_router() -> None:
    patch = (
        REPO_ROOT
        / "vllm_musa/patches/series/"
        "0105-MUSA-bind-qwen35-shared-gate-to-router.patch"
    ).read_text()

    assert "+            routed._musa_shared_router = self.experts.router" in patch
