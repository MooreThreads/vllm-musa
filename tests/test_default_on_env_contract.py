import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY_BOOLEAN_GATES = tuple(
    re.compile(rf"\b{name}\b")
    for name in (
        "VLLM_MUSA_FUSED_ADD_RMSNORM",
        "VLLM_MUSA_FUSED_AR_RMSNORM",
        "VLLM_MUSA_FUSED_AR_RMSNORM_GRAPH_REGISTERED_INPUT",
        "VLLM_MUSA_ENABLE_JIT_TOPK",
        "VLLM_MUSA_SEEDED_MULTINOMIAL",
        "VLLM_MUSA_RESHAPE_CACHE_FLASH",
    )
)


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_default_on_process_gates_are_absent_from_runtime() -> None:
    runtime_paths = list((ROOT / "vllm_musa").rglob("*.py"))
    runtime_paths.extend(
        path
        for path in (ROOT / "csrc").rglob("*")
        if path.suffix in {".cpp", ".cu", ".cuh", ".h", ".mu"}
    )

    for path in sorted(runtime_paths):
        source = path.read_text(encoding="utf-8")
        for legacy_gate in LEGACY_BOOLEAN_GATES:
            assert legacy_gate.search(source) is None, path.relative_to(ROOT)


def test_block_size_env_remains_an_explicit_kernel_tuning_override() -> None:
    kernel = _source("csrc/musa/fused_add_rmsnorm.mu")
    platform = _source("vllm_musa/platform.py")

    assert 'std::getenv("VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X")' in kernel
    assert "VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X" not in platform
