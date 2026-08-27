from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MUSA_ROOT = ROOT / "vllm_musa"


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


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
    shared_expert = _source(
        "vllm_musa/patches/series/"
        "0076-MUSA-model-fold-the-Qwen3.5-shared-expert-into-fused.patch"
    )
    fused_mrope = _source(
        "vllm_musa/patches/series/"
        "0086-MUSA-model-fuse-QK-RMSNorm-and-MRoPE-for-interleaved.patch"
    )
    mamba_pool = _source(
        "vllm_musa/patches/series/0082-MUSA-separate-mamba-state-BlockPools.patch"
    )

    assert "QWEN35_SHARED_EXPERT_FOLD" in shared_expert
    assert "QWEN35_INTERLEAVED_MROPE_QK" in fused_mrope
    assert "def musa_mamba_separate_pool_enabled" in mamba_pool
    assert "+    return True" in mamba_pool


def test_deepseek_provider_is_registered_explicitly() -> None:
    providers = _source("vllm_musa/optimization_contract/providers.py")
    assert "resolve_qwen_contract" in providers
    assert "resolve_deepseek_v4_contract" in providers
    assert "resolve_deepseek_v4_contract," in providers
