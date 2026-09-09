import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOVED_BOOLEAN_GATES = tuple(
    re.compile(rf"\b{name}\b")
    for name in (
        "VLLM_MUSA_FUSED_ADD_RMSNORM",
        "VLLM_MUSA_ENABLE_JIT_TOPK",
        "VLLM_MUSA_SEEDED_MULTINOMIAL",
        "VLLM_MUSA_RESHAPE_CACHE_FLASH",
        "VLLM_MUSA_FUSED_AR_RMSNORM",
        "VLLM_MUSA_FUSED_AR_RMSNORM_GRAPH_REGISTERED_INPUT",
    )
)


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_removed_default_on_process_gates_are_absent_from_runtime() -> None:
    # Scan source extensions rather than just Python and the hand-written
    # csrc suffix subset.  In particular, MUSA headers (`.muh`) can contain
    # provider dispatch and must not silently retain a process gate.
    source_suffixes = {
        ".c",
        ".cc",
        ".cpp",
        ".cu",
        ".cuh",
        ".h",
        ".mu",
        ".muh",
        ".py",
    }
    runtime_paths = [
        path
        for root in (ROOT / "vllm_musa", ROOT / "csrc")
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in source_suffixes
        and "patches/series" not in path.as_posix()
    ]

    for path in sorted(runtime_paths):
        source = path.read_text(encoding="utf-8")
        for removed_gate in REMOVED_BOOLEAN_GATES:
            assert removed_gate.search(source) is None, path.relative_to(ROOT)

    # Patch payloads are also shipped runtime input.  Keep this a separate
    # scan so a deleted upstream hunk cannot hide a live gate in a patch file.
    for path in sorted((ROOT / "vllm_musa/patches/series").glob("*.patch")):
        source = path.read_text(encoding="utf-8")
        for removed_gate in REMOVED_BOOLEAN_GATES:
            assert removed_gate.search(source) is None, path.relative_to(ROOT)


def test_car_rmsnorm_uses_standard_pass_config_without_a_process_switch() -> None:
    platform = _source("vllm_musa/platform.py")
    fusion = _source("vllm_musa/_inductor/musa_allreduce_rms_fusion.py")
    layernorm = _source("vllm_musa/model_executor/layers/layernorm.py")
    communicator = _source(
        "vllm_musa/distributed/device_communicators/"
        "musa_jit_custom_all_reduce.py"
    )

    assert 'getattr(pass_config, "fuse_allreduce_rms", None) is None' in platform
    assert 'getattr(pass_config, "fuse_allreduce_rms", None) is not True' in platform
    assert 'getattr(self.pass_config, "fuse_allreduce_rms", None) is not True' in fusion
    assert 'pass_value = getattr(pass_config, "fuse_allreduce_rms", None)' in layernorm
    assert "_car_rmsnorm_pass_enabled_for_current_model()" in communicator
    assert 'getattr(pass_config, "fuse_allreduce_rms", None) is True' in communicator
    assert "CAR-RMSNorm disabled by compilation pass config" in communicator


def test_registered_input_transport_has_no_feature_specific_switch() -> None:
    communicator = _source(
        "vllm_musa/distributed/device_communicators/"
        "musa_jit_custom_all_reduce.py"
    )

    assert "self._graph_registered_input_enabled" in communicator
    assert "self._use_graph_registered_inputs" in communicator


def test_block_size_env_remains_an_explicit_kernel_tuning_override() -> None:
    kernel = _source("csrc/musa/fused_add_rmsnorm.mu")
    platform = _source("vllm_musa/platform.py")

    assert 'std::getenv("VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X")' in kernel
    assert "VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X" not in platform
