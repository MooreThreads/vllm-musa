import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY_BOOLEAN_GATE = re.compile(r"\bVLLM_MUSA_FUSED_ADD_RMSNORM\b")


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_fused_add_rmsnorm_dispatch_has_no_legacy_boolean_env_gate() -> None:
    runtime_paths = sorted((ROOT / "vllm_musa").rglob("*.py"))

    for path in runtime_paths:
        source = path.read_text(encoding="utf-8")
        assert LEGACY_BOOLEAN_GATE.search(source) is None, path.relative_to(ROOT)


def test_block_size_env_remains_an_explicit_kernel_tuning_override() -> None:
    kernel = _source("csrc/musa/fused_add_rmsnorm.mu")
    platform = _source("vllm_musa/platform.py")

    assert 'std::getenv("VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X")' in kernel
    assert "VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X" not in platform
