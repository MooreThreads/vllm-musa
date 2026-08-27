from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_regular_cuda_view_source_is_present_for_musa_binding() -> None:
    setup_source = (ROOT / "setup.py").read_text()
    cuda_view = ROOT / "third_party" / "vllm" / "csrc" / "cuda_view.cu"
    bindings = (
        ROOT / "third_party" / "vllm" / "csrc" / "torch_bindings.cpp"
    ).read_text()

    assert '"csrc/cuda_view.cu"' in setup_source
    assert cuda_view.is_file()
    assert "get_cuda_view_from_cpu_tensor" in cuda_view.read_text()
    assert "get_cuda_view_from_cpu_tensor" in bindings


def test_cuda_utils_kernel_is_linked_into_stable_extension() -> None:
    setup_source = (ROOT / "setup.py").read_text()
    regular_sources = setup_source.split("VLLM_CSRC_SOURCES = [", 1)[1].split(
        "VLLM_STABLE_CSRC_SOURCES = [", 1
    )[0]
    stable_sources = setup_source.split("VLLM_STABLE_CSRC_SOURCES = [", 1)[
        1
    ].split("VLLM_MUSA_CSRC_SOURCES = [", 1)[0]
    cuda_utils_source = (
        '"csrc/libtorch_stable/cuda_utils_kernels.cu"'
    )
    stable_bindings = (
        ROOT
        / "third_party"
        / "vllm"
        / "csrc"
        / "libtorch_stable"
        / "torch_bindings.cpp"
    ).read_text()

    assert cuda_utils_source not in regular_sources
    assert cuda_utils_source in stable_sources
    assert "get_device_attribute" in stable_bindings

    preprocessor_guards: list[str] = []
    for line in stable_bindings.splitlines():
        stripped = line.strip()
        if stripped.startswith(("#if ", "#ifdef ", "#ifndef ")):
            preprocessor_guards.append(stripped)
        elif stripped.startswith("#endif"):
            preprocessor_guards.pop()
        elif stripped.startswith(
            "STABLE_TORCH_LIBRARY_FRAGMENT(_C_cuda_utils"
        ):
            assert "#if !defined(USE_MUSA)" not in preprocessor_guards


def test_regular_moe_sources_use_resolvable_stable_helper_includes() -> None:
    moe_dir = ROOT / "third_party" / "vllm" / "csrc" / "libtorch_stable" / "moe"
    for source_name in (
        "topk_softmax_kernels.cu",
        "topk_softplus_sqrt_kernels.cu",
    ):
        source = (moe_dir / source_name).read_text()
        assert '#include "../cub_helpers.h"' in source
        assert '#include "../torch_utils.h"' in source


def test_vllm_editable_install_uses_pinned_release_version() -> None:
    setup_source = (ROOT / "setup.py").read_text()
    override = 'env.setdefault("VLLM_VERSION_OVERRIDE", "0.28.0")'

    assert override in setup_source
    assert setup_source.index(override) < setup_source.index("vllm_install_cmd = [")
