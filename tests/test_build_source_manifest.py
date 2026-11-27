import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VLLM_CSRC = ROOT / "third_party" / "vllm" / "csrc"


def test_regular_cuda_view_source_is_present_for_musa_binding() -> None:
    setup_source = (ROOT / "setup.py").read_text()
    cuda_view = VLLM_CSRC / "musa_cuda_view.cu"
    bindings = (VLLM_CSRC / "musa_torch_bindings.cpp").read_text()

    assert '"csrc/musa_cuda_view.cu"' in setup_source
    assert '"csrc/musa_torch_bindings.cpp"' in setup_source
    assert cuda_view.is_file()
    assert "get_cuda_view_from_cpu_tensor" in cuda_view.read_text()
    assert "get_cuda_view_from_cpu_tensor" in bindings


def test_cuda_utils_kernel_is_linked_into_stable_extension() -> None:
    setup_source = (ROOT / "setup.py").read_text()
    regular_sources = setup_source.split("VLLM_CSRC_SOURCES = [", 1)[1].split(
        "VLLM_STABLE_CSRC_SOURCES = [", 1
    )[0]
    stable_sources = setup_source.split("VLLM_STABLE_CSRC_SOURCES = [", 1)[1].split(
        "VLLM_MUSA_CSRC_SOURCES = [", 1
    )[0]
    cuda_utils_source = '"csrc/cuda_utils_kernels.cu"'
    stable_bindings = (VLLM_CSRC / "torch_bindings.cpp").read_text()

    assert cuda_utils_source not in regular_sources
    assert cuda_utils_source in stable_sources
    assert "get_device_attribute" in stable_bindings

    musa_excluded = ("#ifndef USE_MUSA", "#if !defined(USE_MUSA)")
    preprocessor_guards: list[str] = []
    found = False
    for line in stable_bindings.splitlines():
        stripped = line.strip()
        if stripped.startswith(("#if ", "#ifdef ", "#ifndef ")):
            preprocessor_guards.append(stripped)
        elif stripped.startswith("#endif"):
            preprocessor_guards.pop()
        elif stripped.startswith("STABLE_TORCH_LIBRARY_FRAGMENT(_C_cuda_utils"):
            found = True
            assert not any(g in musa_excluded for g in preprocessor_guards)
    assert found


def test_regular_moe_sources_use_resolvable_helper_includes() -> None:
    moe_dir = VLLM_CSRC / "moe"
    for source_name in (
        "topk_softmax_kernels.cu",
        "topk_softplus_sqrt_kernels.cu",
    ):
        source = (moe_dir / source_name).read_text()
        headers = re.findall(
            r'^\s*#include "([^"]*(?:cub_helpers|torch_utils)\.h)"', source, re.M
        )
        names = sorted(Path(h).name for h in headers)
        assert names == ["cub_helpers.h", "torch_utils.h"]
        for header in headers:
            assert (moe_dir / header).is_file() or (VLLM_CSRC / header).is_file(), (
                f"{source_name}: unresolvable include {header}"
            )


def test_vllm_editable_install_uses_pinned_release_version() -> None:
    setup_source = (ROOT / "setup.py").read_text()
    override = 'env.setdefault("VLLM_VERSION_OVERRIDE", _PINS["VLLM_TAG"].lstrip("v"))'

    assert override in setup_source
    assert setup_source.index(override) < setup_source.index("vllm_install_cmd = [")
