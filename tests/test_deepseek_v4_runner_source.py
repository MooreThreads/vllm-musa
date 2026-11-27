from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_musa_keeps_dsv4_on_mrv1_by_default() -> None:
    source = _read("third_party/vllm/vllm/config/vllm.py")

    assert '"DeepseekV4ForCausalLM"' in source
    assert "MUSA_DEFAULT_MRV1_ARCHITECTURES" in source
    assert 'is_musa = getattr(current_platform, "is_musa", None)' in source
    assert "on_musa = callable(is_musa) and is_musa()" in source
    assert (
        "needs_v2 = on_musa and bool(self._get_v1_model_runner_unsupported_features())"
        in source
    )
    assert "if model_config is not None and on_musa and not needs_v2:" in source
    assert (
        "if any(arch in MUSA_DEFAULT_MRV1_ARCHITECTURES for arch in architectures):"
        in source
    )
    branch_start = source.index(
        "if any(arch in MUSA_DEFAULT_MRV1_ARCHITECTURES for arch in architectures):"
    )
    branch_end = source.index("        if not HAS_TRITON:", branch_start)
    assert "                return False" in source[branch_start:branch_end]


def test_dsv4_runner_override_and_dspark_force_remain_upstream() -> None:
    source = _read("third_party/vllm/vllm/config/vllm.py")

    assert "if use_v2_model_runner is not None:" in source
    assert "return use_v2_model_runner" in source
    assert 'self.speculative_config.method == "dspark"' in source
    unsupported = source.index('self.speculative_config.method == "dspark"')
    assert 'unsupported.append("dspark speculative decoding")' in source[unsupported:]
    assert "if model_config is not None and on_musa and not needs_v2:" in source
