from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MUSA_ROOT = ROOT / "vllm_musa"


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _entry(slug: str) -> str:
    """Read one series entry by its slug, independent of its series number."""
    matches = sorted((MUSA_ROOT / "patches" / "series").glob(f"*-{slug}"))
    assert len(matches) == 1, matches
    return matches[0].read_text(encoding="utf-8")


def test_qwen_model_specific_env_gates_are_removed() -> None:
    forbidden = {
        "VLLM_MUSA_MOE_SHARED_EXPERT_FUSION",
        "VLLM_MUSA_FUSED_QK_MROPE",
        "VLLM_MUSA_MAMBA_SEPARATE_POOL",
    }
    production_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in MUSA_ROOT.rglob("*")
        if path.suffix in {".py", ".patch"}
    )
    for gate in forbidden:
        assert gate not in production_sources


def test_qwen_patch_consumers_bind_contract_features() -> None:
    shared_expert = _entry("MUSA-MUSA-model-fold-the-Qwen3.5-shared-expert-into-.patch")
    fused_mrope = _entry("MUSA-MUSA-model-fuse-QK-RMSNorm-and-MRoPE-for-interl.patch")

    assert "QWEN35_SHARED_EXPERT_FOLD" in shared_expert
    assert "QWEN35_INTERLEAVED_MROPE_QK" in fused_mrope


def test_deepseek_provider_is_registered_explicitly() -> None:
    providers = _source("vllm_musa/optimization_contract/providers.py")
    assert "resolve_qwen_contract" in providers
    assert "resolve_deepseek_v4_contract" in providers
    assert "resolve_deepseek_v4_contract," in providers
