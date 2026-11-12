from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "vllm_musa" / "patches" / "series"
MODEL = (
    ROOT
    / "third_party"
    / "vllm"
    / "vllm"
    / "models"
    / "deepseek_v4"
    / "nvidia"
    / "model.py"
)
DSPARK = (
    ROOT
    / "third_party"
    / "vllm"
    / "vllm"
    / "models"
    / "deepseek_v4"
    / "nvidia"
    / "dspark.py"
)


def _patch(number: str) -> str:
    # Entries are addressed by number: `musa_sync regen` owns the slug part of
    # the file name, so a rename must not break a source assertion.
    matches = sorted(SERIES.glob(f"{number}-*.patch"))
    assert len(matches) == 1, f"series entry {number} resolves to {matches}"
    return matches[0].read_text()


def test_dspark_context_kv_uses_musa_custom_op() -> None:
    patch = _patch("0187")
    assert "from vllm.platforms import current_platform" in patch
    assert "if current_platform.is_musa():" in patch
    assert "_musa_custom_ops.deepseek_v4_qnorm_rope_kv_insert(" in patch
    assert "torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope" not in patch


def test_dspark_rejection_sampler_preserves_optional_feature_flags() -> None:
    patch = _patch("0128")
    assert "synthetic_mode = synthetic_conditional_rates is not None" in patch
    assert 'target_logits.device.type == "musa"' in patch
    assert "draft_logits = target_logits.new_empty((1, 1, 1))" in patch
    assert "SYNTHETIC_MODE=synthetic_mode" in patch


def test_dsv4_loaders_fall_back_to_resolved_quant_expert_dtype() -> None:
    patch = _patch("0129")
    assert 'getattr(vllm_config.quant_config, "expert_dtype", None)' in patch
    assert 'getattr(self.quant_config, "expert_dtype", None)' in patch
    assert 'expert_dtype == "fp4" and resolved_quant_dtype == "fp8"' in patch
    assert '".weight_scale" if expert_dtype == "fp4" else ".weight_scale_inv"' in patch


def test_dsv4_loaders_prefer_resolved_fp8_over_config_default() -> None:
    patch = _patch("0129")
    assert 'resolved_quant_dtype in ("fp4", "fp8")' in patch
    assert 'resolved_quant_dtype == "fp8"' in patch


def test_final_model_and_dspark_loaders_resolve_expert_dtype_independently() -> None:
    model = MODEL.read_text()
    dspark = DSPARK.read_text()

    model_start = model.index(
        "    def __init__(", model.index("class DeepseekV4ForCausalLM")
    )
    model_end = model.index("    def ", model_start + 1)
    model_init = model[model_start:model_end]
    assert 'getattr(vllm_config.quant_config, "expert_dtype", None)' in model_init
    assert 'resolved_quant_dtype in ("fp4", "fp8")' in model_init
    assert 'expert_dtype == "fp4" and resolved_quant_dtype == "fp8"' in model_init
    assert (
        "self.hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper(expert_dtype)"
        in model_init
    )

    dspark_start = dspark.index("class DSparkDeepseekV4ForCausalLM")
    dspark_end = len(dspark)
    dspark_class = dspark[dspark_start:dspark_end]
    assert 'getattr(self.quant_config, "expert_dtype", None)' in dspark_class
    assert 'resolved_quant_dtype in ("fp4", "fp8")' in dspark_class
    assert 'expert_dtype == "fp4" and resolved_quant_dtype == "fp8"' in dspark_class
    assert (
        '".weight_scale" if expert_dtype == "fp4" else ".weight_scale_inv"'
        in dspark_class
    )
