# SPDX-License-Identifier: Apache-2.0
"""Offline contracts for the split build-time patch series."""

from pathlib import Path

SERIES = Path(__file__).resolve().parents[1] / "vllm_musa/patches/series"


def _patch_section(text: str, path: str) -> str:
    start = text.find("diff --git a/" + path)
    assert start >= 0, f"{path} not found in patch"
    end = text.find("\ndiff --git ", start + 10)
    return text[start:] if end < 0 else text[start:end]


def _read_patch(path: Path) -> str:
    return path.read_bytes().decode("utf-8", errors="surrogateescape")


def test_modelopt_fp8_uses_musa_capability_floor() -> None:
    patch_path = (
        Path(__file__).parent.parent
        / "vllm_musa/patches/series/0090-MUSA-allow-ModelOpt-FP8-on-MUSA-capability.patch"
    )
    section = _patch_section(
        patch_path.read_text(),
        "vllm/model_executor/layers/quantization/modelopt.py",
    )

    assert "-        return 80" in section
    assert "+        return 31" in section


def test_gemma_norm_series_patch_uses_generic_inplace_ir_contract() -> None:
    series = _patch_section(
        _read_patch(SERIES / "0075-MUSA-allow-Gemma-RMSNorm-residual-donation.patch"),
        "vllm/model_executor/layers/layernorm.py",
    )

    assert "fused_add_rms_norm.maybe_inplace" in series
    assert "allow_inplace" not in series


def test_gated_qkv_series_patch_uses_generic_ir_contract() -> None:
    base = _read_patch(SERIES / "0076-MUSA-add-gated-QKV-RMSNorm-MRoPE-IR.patch")
    op = _patch_section(base, "vllm/ir/ops/gated_qkv.py")
    caller = _read_patch(SERIES / "0103-MUSA-fuse-Qwen3.5-QK-mRoPE-cache-update.patch")
    section = _patch_section(caller, "vllm/model_executor/models/qwen3_next.py")

    assert '@register_op(activations=["packed_qkv"])' in op
    assert "def gated_qkv_rms_norm_rope(" in op
    assert "ir.ops.gated_qkv_rms_norm_rope(" in section
    assert "q, k = ir.ops.gated_qkv_rms_norm_rope(" in section
    assert "q, k, gate = ir.ops.gated_qkv_rms_norm_rope(" not in section
    assert "attn_output.view_as(gate) * torch.sigmoid(gate)" in section
    assert "VLLM_MUSA_EXPERIMENTAL_GATED_QKV_MROPE" not in base
    assert "VLLM_MUSA_EXPERIMENTAL_GATED_QKV_MROPE" not in caller


def test_deepseek_v4_spec_metadata_upload_series_patch_present() -> None:
    series = _patch_section(
        _read_patch(
            SERIES / "0035-MUSA-support-full-graph-drafts-and-pinned-metadata.patch"
        ),
        "vllm/v1/worker/gpu_model_runner.py",
    )

    assert "_spec_cu_num_draft_tokens" in series
    assert "_spec_cu_num_sampled_tokens" in series
    assert "_spec_logits_indices" in series
    assert "_spec_target_logits_indices" in series
    assert "_spec_bonus_logits_indices" in series
    assert (
        "self._spec_target_logits_indices = self._make_buffer(\n"
        "+            self.max_num_tokens, dtype=torch.int64\n"
        "+        )" in series
    )
    assert "current_platform.is_musa()" in series
    assert "copy_to_gpu(num_sampled_total)" in series
    musa_branch = series.split("MUSA-3406: copy through", 1)[1].split("else:", 1)[0]
    assert "torch.from_numpy(cu_num_draft_tokens).to" not in musa_branch
    assert "torch.from_numpy(cu_num_sampled_tokens).to" not in musa_branch
    assert "torch.from_numpy(logits_indices).to" not in musa_branch
    assert "torch.from_numpy(target_logits_indices).to" not in musa_branch
    assert "torch.from_numpy(bonus_logits_indices).to" not in musa_branch


def test_deepseek_v4_materialized_indexer_series_patch_present() -> None:
    series = _patch_section(
        _read_patch(
            SERIES / "0013-MUSA-vllm.model_executor.layers.sparse_attn_indexer.patch"
        ),
        "vllm/model_executor/layers/sparse_attn_indexer.py",
    )

    assert "VLLM_MUSA_DEEPSEEK_V4_INDEXER_TOPK_PREFILL" not in series
    assert "return True" in series
    assert "width = 640" in series
    assert "chunk_rows = 512" in series
    assert "sorted=materialized_topk_sorted" in series
    assert "materialized_direct_topk" in series
    assert "direct_local" in series
    assert "deepseek_v4_indexer_rerank_prefill" in series
    assert "Paged-MQA logits use absolute prefix coordinates" in series
    assert "context_lens = row_ends.reshape(rows, 1).contiguous()" in series
    assert "_musa_try_fill_prefill_topk_from_materialized_logits" in series
