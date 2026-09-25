from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "vllm_musa" / "patches" / "series"


def _patch(number: str) -> str:
    # Entries are addressed by number: `musa_sync regen` owns the slug part of
    # the file name, so a rename must not break a source assertion.
    matches = sorted(SERIES.glob(f"{number}-*.patch"))
    assert len(matches) == 1, f"series entry {number} resolves to {matches}"
    return matches[0].read_text()


def test_dspark_context_kv_uses_musa_custom_op() -> None:
    patch = _patch("0153")
    assert "from vllm.platforms import current_platform" in patch
    assert "if current_platform.is_musa():" in patch
    assert "_musa_custom_ops.deepseek_v4_qnorm_rope_kv_insert(" in patch
    assert "torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope" not in patch


def test_dspark_rejection_sampler_preserves_optional_feature_flags() -> None:
    patch = _patch("0154")
    assert "synthetic_mode = synthetic_conditional_rates is not None" in patch
    assert 'target_logits.device.type == "musa"' in patch
    assert "draft_logits = target_logits.new_empty((1, 1, 1))" in patch
    assert "SYNTHETIC_MODE=synthetic_mode" in patch


def test_dsv4_loaders_fall_back_to_resolved_quant_expert_dtype() -> None:
    patch = _patch("0155")
    assert "vllm_config.quant_config, \"expert_dtype\"" in patch
    assert "self.quant_config, \"expert_dtype\"" in patch
    assert "FP8 block scale names" in patch


def test_dsv4_loaders_prefer_resolved_fp8_over_config_default() -> None:
    patch = _patch("0156")
    assert 'resolved_quant_dtype in ("fp4", "fp8")' in patch
    assert 'resolved_quant_dtype == "fp8"' in patch



def test_dspark_loader_prefers_resolved_fp8_expert_scales() -> None:
    patch = _patch("0157")
    assert "resolved_quant_dtype in (\"fp4\", \"fp8\")" in patch
    assert "expert_scale_suffix" in patch
