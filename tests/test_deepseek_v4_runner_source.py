from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_musa_keeps_dsv4_on_mrv1_by_default() -> None:
    source = _read("third_party/vllm/vllm/config/vllm.py")

    assert 'is_musa = getattr(current_platform, "is_musa", None)' in source
    assert "callable(is_musa) and is_musa()" in source
    assert (
        'DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES - {"DeepseekV4ForCausalLM"}' in source
    )


def test_dsv4_runner_override_and_dspark_force_remain_upstream() -> None:
    source = _read("third_party/vllm/vllm/config/vllm.py")

    assert "if use_v2_model_runner is not None:" in source
    assert 'self.speculative_config.method == "dspark"' in source
    dspark = source.index('self.speculative_config.method == "dspark"')
    assert "return True" in source[dspark:]
