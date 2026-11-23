# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard DeepSelect registration and indexer dispatch without GPU hardware."""

from __future__ import annotations

import ast
import math
import os
import shutil
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "vllm_musa" / "patches" / "series"
BUILD_PATCH = SERIES / "0218-MUSA-build-DeepSelect-through-vLLM-extension-target.patch"
PLUGIN = ROOT / "vllm_musa" / "__init__.py"
PINS = ROOT / "third_party" / "PINS"
INDEXER_PATCH = (
    SERIES / "0217-MUSA-route-explicit-DeepSelect-through-DSV4-indexer.patch"
)


def _deep_select_patches() -> list[Path]:
    patches = sorted(SERIES.glob("*-MUSA-*DeepSelect*.patch"))
    assert patches, "no DeepSelect series patch found"
    return patches


def _added_lines(text: str) -> str:
    return "\n".join(
        line[1:]
        for line in text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def test_series_patch_retains_native_extension_import_detection() -> None:
    source = (
        SERIES / "0216-MUSA-register-an-out-of-tree-DeepSelect-top-k-backen.patch"
    ).read_text(encoding="utf-8")
    assert "elif not _HAS_DEEP_SELECT_C:" in source
    assert "vllm._deepselect_C could not be imported" in source
    assert 'hasattr(torch.ops.deep_select, "topk")' not in source
    assert "_deep_select_op_available" not in source


def test_series_patch_keeps_the_cuda_capability_check() -> None:
    added = "\n".join(
        _added_lines(p.read_text(encoding="utf-8")) for p in _deep_select_patches()
    )
    # SM100-only stays the rule for CUDA; only the "op exists" half became
    # platform-neutral.
    assert "current_platform.is_device_capability_family(100)" in added
    assert (
        "self._is_cuda and not current_platform.is_device_capability_family(100)"
        in added
    )
    assert "self._is_musa = current_platform.is_musa()" in added
    assert "current_platform.is_out_of_tree()" not in added
    assert "if not self._is_cuda and not self._is_musa:" in added


def _deep_select_validation_helpers(family: int, extension_available: bool) -> dict:
    patch = SERIES / "0216-MUSA-register-an-out-of-tree-DeepSelect-top-k-backen.patch"
    source_path = ROOT / "third_party/vllm/vllm/model_executor/layers/indexer_topk.py"
    if not source_path.exists():
        pytest.skip("Applied vLLM source is required")
    source = source_path.read_text(encoding="utf-8")
    branch = source[source.index('        elif self._backend == "deep_select":') :]
    branch = branch.split('        elif self._backend == "aiter":', 1)[0]
    validation = (
        "def validate(self, logits, topk_tokens):\n    failures = []\n"
        + textwrap.indent(textwrap.dedent(branch).replace("elif ", "if ", 1), "    ")
        + "\n    return failures\n"
    )
    calls = []

    def is_family(requested: int) -> bool:
        calls.append("platform")
        return family // 10 == requested // 10

    namespace = {
        "_HAS_DEEP_SELECT_C": extension_available,
        "current_platform": SimpleNamespace(is_device_capability_family=is_family),
        "is_deep_select_supported": lambda *_: True,
    }
    exec(compile(validation, str(patch), "exec"), namespace)
    namespace["calls"] = calls
    return namespace


@pytest.mark.parametrize(
    ("platform", "family", "extension_available", "error"),
    [
        ("musa", 0, True, None),
        ("cuda", 100, True, None),
        ("cuda", 103, True, None),
        ("cuda", 90, True, "SM100a/SM103a"),
        ("cpu", 0, True, "CUDA or MUSA"),
        ("rocm", 0, True, "CUDA or MUSA"),
        ("musa", 0, False, "could not be imported"),
    ],
)
def test_deep_select_validation_separates_platform_and_extension(
    platform: str,
    family: int,
    extension_available: bool,
    error: str | None,
) -> None:
    namespace = _deep_select_validation_helpers(family, extension_available)
    selector = SimpleNamespace(
        _is_cuda=platform == "cuda",
        _is_musa=platform == "musa",
        _backend="deep_select",
    )
    logits = SimpleNamespace(dtype="float32", stride=lambda: (1024, 1))
    failures = namespace["validate"](selector, logits, 512)
    if error is None:
        assert failures == []
    else:
        assert len(failures) == 1 and error in failures[0]
    assert ("platform" in namespace["calls"]) == (platform == "cuda")


def test_series_patch_routes_supported_windows_to_explicit_backend() -> None:
    source = INDEXER_PATCH.read_text(encoding="utf-8")
    added = _added_lines(source)
    assert added.count('topk_backend == "deep_select"') >= 3
    assert "_musa_try_fill_decode_topk_from_paged_logits" in added
    assert "deep_select_topk(" in added
    assert "requires zero-based indexer windows" not in added


@pytest.mark.parametrize(
    "shape, topk, supported",
    [
        ((2, 4096), 512, True),
        ((2, 4096), 4096, True),
        ((2, 4096), 0, False),
        ((2, 4096), -1, False),
        ((2, 4096), 4097, False),
        ((2, 0), 512, False),
        ((2, 2**23), 512, False),
        ((4096,), 512, False),
        ((2, 4, 4096), 512, False),
    ],
)
@pytest.mark.parametrize("offset, aligned", [(0, True), (1, False), (8, True)])
def test_deep_select_support_checks_native_shape_and_topk_bounds(
    shape: tuple[int, ...], topk: int, supported: bool, offset: int, aligned: bool
) -> None:
    source_path = ROOT / "third_party/vllm/vllm/model_executor/layers/indexer_topk.py"
    if not source_path.is_file():
        pytest.skip("Applied vLLM source is required")
    function = next(
        node
        for node in ast.parse(source_path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "is_deep_select_supported"
    )
    namespace = {
        "torch": SimpleNamespace(Tensor=object, float32="float32", bfloat16="bfloat16"),
        "get_deep_select_stride_requirement": lambda: (32, 32),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    logits = SimpleNamespace(
        shape=shape,
        dim=lambda: len(shape),
        dtype="float32",
        stride=lambda axis: (4096, 1)[axis],
        storage_offset=lambda: offset,
        element_size=lambda: 4,
    )
    assert namespace["is_deep_select_supported"](logits, topk) is (
        supported and aligned
    )


def _indexer_selection_helpers(
    capturing: bool = False, supported: bool = True
) -> tuple[dict, list]:
    staged_patch = next(
        SERIES.glob("*-MUSA-select-indexer-top-k-with-the-staged-radix-kern.patch")
    )
    source = _added_lines(staged_patch.read_text(encoding="utf-8").split("@@")[2])
    source = _apply_indexer_patch(source, INDEXER_PATCH)
    calls = []

    def record(name: str):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))

        return call

    namespace = {
        "torch": SimpleNamespace(int32="int32"),
        "get_deep_select_stride_requirement": lambda: (32, 32),
        "deep_select_topk": record("deep_select"),
        "is_deep_select_supported": lambda logits, topk: supported,
        "_musa_sparse_indexer_is_current_stream_capturing": lambda: capturing,
        "_musa_custom_ops": SimpleNamespace(
            deepseek_v4_sparse_indexer_topk_decode=record("staged"),
            sparse_indexer_topk_decode=record("decode"),
            sparse_indexer_topk=record("windows"),
        ),
    }
    exec(
        compile(
            "from __future__ import annotations\n" + source, str(INDEXER_PATCH), "exec"
        ),
        namespace,
    )
    return namespace, calls


def _apply_indexer_patch(source: str, patch: Path) -> str:
    for hunk in patch.read_text(encoding="utf-8").split("@@")[2::2]:
        lines = hunk.splitlines()[1:]
        before = "\n".join(line[1:] for line in lines if line.startswith((" ", "-")))
        after = "\n".join(line[1:] for line in lines if line.startswith((" ", "+")))
        if before and before in source:
            source = source.replace(before, after, 1)
    return source


class _IndexerTensorMetadata:
    def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = "musa"

    def numel(self) -> int:
        return math.prod(self.shape)

    def stride(self, axis: int) -> int:
        return math.prod(self.shape[axis + 1 :])

    def storage_offset(self) -> int:
        return 0

    def element_size(self) -> int:
        return {"int32": 4, "int64": 8}[self.dtype]

    def is_contiguous(self) -> bool:
        return True

    def reshape(self, *shape):
        assert shape == (-1,)
        return self

    def __getitem__(self, key):
        return self


@pytest.mark.parametrize(
    ("backend", "block_size", "supported", "dtypes", "selected"),
    [
        ("deep_select", 64, True, ["bfloat16"], "deep_select"),
        ("deep_select", 128, True, ["bfloat16"], "deep_select"),
        ("auto", 128, True, ["float32"], "staged"),
        ("per_row", 64, True, ["float32"], "staged"),
        ("deep_select", 128, False, ["bfloat16", "float32"], "staged"),
        ("deep_select", 24, True, ["bfloat16", "float32"], "deep_select"),
    ],
)
def test_paged_triton_writes_consumer_supported_logits_dtype(
    backend: str,
    block_size: int,
    supported: bool,
    dtypes: list[str],
    selected: str,
) -> None:
    source_patch = next(
        SERIES.glob("*-MUSA-score-decode-indexer-keys-of-any-cache-block-si.patch")
    )
    source = _added_lines(source_patch.read_text(encoding="utf-8").split("@@")[2])
    staged_patch = next(
        SERIES.glob("*-MUSA-select-indexer-top-k-with-the-staged-radix-kern.patch")
    )
    for patch in (staged_patch, INDEXER_PATCH):
        source = _apply_indexer_patch(source, patch)
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_musa_try_fill_decode_topk_from_paged_logits"
    )
    namespace, calls = _indexer_selection_helpers(capturing=True)
    allocations = []
    launches = []

    def empty(shape, *, dtype, device):
        tensor = _IndexerTensorMetadata(shape, dtype)
        assert device == tensor.device
        allocations.append(tensor)
        return tensor

    def supports(logits, topk):
        itemsize = 2 if logits.dtype == "bfloat16" else 4
        return supported and logits.stride(0) * itemsize % 32 == 0

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, args, kwargs))

            return launch

    namespace.update(
        torch=SimpleNamespace(
            int32="int32",
            float8_e4m3fn="float8_e4m3fn",
            uint8="uint8",
            float32="float32",
            bfloat16="bfloat16",
            empty=empty,
        ),
        triton=SimpleNamespace(cdiv=lambda value, tile: (value + tile - 1) // tile),
        is_deep_select_supported=supports,
        _musa_decode_block_table_for_token_rows=lambda table, rows: table,
        _musa_paged_mqa_logits_kernel=Kernel(),
    )
    exec(
        compile(
            "from __future__ import annotations\n" + ast.unparse(function),
            str(INDEXER_PATCH),
            "exec",
        ),
        namespace,
    )
    tensor = _IndexerTensorMetadata
    assert namespace[function.name](
        tensor((2, 32, 128), "float8_e4m3fn"),
        tensor((8, block_size, 132), "uint8"),
        tensor((2, 32), "float32"),
        SimpleNamespace(
            seq_lens=tensor((2,), "int32"),
            block_table=tensor((2, 3), "int32"),
        ),
        tensor((2, 512), "int32"),
        512,
        128,
        backend,
    )
    assert [allocation.dtype for allocation in allocations] == dtypes
    assert len(launches) == 1
    assert launches[0][1][13] is allocations[-1]
    assert launches[0][2]["BLOCK_SIZE"] == block_size
    assert allocations[-1].shape == (2, 3 * block_size)
    assert len(calls) == 1 and calls[0][0] == selected
    assert calls[0][1][0] is allocations[-1]


@pytest.mark.parametrize(
    ("backend", "starts_at_zero", "topk", "expected"),
    [
        ("deep_select", True, 512, "deep_select"),
        ("deep_select", False, 512, "windows"),
        ("deep_select", True, 1024, "deep_select"),
        ("deep_select", False, 1024, "windows"),
        ("auto", True, 512, "staged"),
        ("auto", True, 1024, "decode"),
        ("auto", False, 512, "windows"),
        ("per_row", False, 512, "windows"),
    ],
)
def test_window_dispatch_preserves_legacy_fallback(
    backend: str, starts_at_zero: bool, topk: int, expected: str
) -> None:
    namespace, calls = _indexer_selection_helpers()
    logits = _IndexerTensorMetadata((2, 4096), "bfloat16")
    starts = ends = _IndexerTensorMetadata((2,), "int32")
    output = _IndexerTensorMetadata((2, topk), "int64")
    namespace["_musa_select_indexer_topk_windows"](
        logits, starts, ends, output, topk, starts_at_zero, backend
    )
    if expected == "deep_select":
        assert calls == [
            (
                expected,
                (logits, topk),
                {"end": ends, "output_idx": output, "indices_dtype": output.dtype},
            )
        ]
    elif expected == "windows":
        assert calls == [(expected, (logits, starts, ends, output, topk), {})]
    else:
        assert calls == [(expected, (logits, ends, output, topk), {})]


@pytest.mark.parametrize("input_supported, output_padding", [(False, 0), (True, 1)])
@pytest.mark.parametrize("topk, expected", [(512, "staged"), (1024, "decode")])
def test_unsupported_deep_select_layout_uses_native_topk(
    topk: int, expected: str, input_supported: bool, output_padding: int
) -> None:
    namespace, calls = _indexer_selection_helpers(
        capturing=True, supported=input_supported
    )
    logits = _IndexerTensorMetadata((2, 4096), "bfloat16")
    starts = ends = _IndexerTensorMetadata((2,), "int32")
    output = _IndexerTensorMetadata((2, topk + output_padding), "int32")
    namespace["_musa_select_indexer_topk_windows"](
        logits, starts, ends, output, topk, True, "deep_select"
    )
    assert calls == [(expected, (logits, ends, output, topk), {})]


@pytest.mark.parametrize(
    ("backend", "topk", "nonzero", "capturing", "expected"),
    [
        ("deep_select", 512, False, False, True),
        ("deep_select", 1024, False, False, True),
        ("deep_select", 512, True, False, False),
        ("deep_select", 512, False, True, False),
        ("auto", 512, False, False, True),
        ("auto", 1024, False, False, False),
        ("auto", 512, True, False, False),
        ("auto", 512, False, True, False),
    ],
)
def test_window_support_keeps_legacy_gates_without_capture_sync(
    backend: str, topk: int, nonzero: bool, capturing: bool, expected: bool
) -> None:
    namespace, _ = _indexer_selection_helpers(capturing)

    def any_start():
        assert not capturing, "capture must not read a GPU scalar"
        return nonzero

    starts = SimpleNamespace(any=any_start)
    assert (
        namespace["_musa_indexer_windows_start_at_zero"](starts, topk, backend)
        is expected
    )


def test_prefill_deep_select_guards_leave_nonzero_windows_on_legacy_paths() -> None:
    added = _added_lines(INDEXER_PATCH.read_text(encoding="utf-8"))
    guards = [
        "_musa_indexer_windows_start_at_zero(\n            starts, topk, topk_backend\n",
        "_musa_indexer_windows_start_at_zero(\n            lo.squeeze(1), width, topk_backend\n",
        "_musa_indexer_windows_start_at_zero(\n        chunk.cu_seqlen_ks, topk_tokens, topk_backend\n",
    ]
    for guard in guards:
        assert guard in added
    assert "if bool(starts.any()):" not in added
    assert "if bool(lo.any()):" not in added


def test_plugin_does_not_register_deep_select() -> None:
    source = PLUGIN.read_text(encoding="utf-8")
    assert "_register_deep_select" not in source
    assert "from .deep_select" not in source
    assert not (ROOT / "vllm_musa/deep_select/registry.py").exists()


def test_microbenchmark_uses_vllm_extension_without_standalone_package() -> None:
    assert not (ROOT / "tools/bench_indexer_topk_arms.py").exists()
    source = (ROOT / "benchmarks/benchmark_deepselect_topk.py").read_text()
    assert "import vllm._deepselect_C as deepselect_extension" in source
    assert "vllm_musa.deep_select" not in source
    assert "deep_select.deep_select_musa_cpp" not in source
    assert '"deep-select"' not in source


def test_build_provides_vllm_extension_from_deepselect_source() -> None:
    source = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert 'name="vllm._deepselect_C"' not in source
    assert "_DEEPSELECT_REPO" not in source
    assert "_DEEPSELECT_SRC_DIR" not in source
    assert "deep_select.deep_select_musa_cpp" not in source
    build = BUILD_PATCH.read_text(encoding="utf-8")
    assert 'CMakeExtension(name="vllm._deepselect_C")' in build
    assert "deepselect_extension.py_limited_api = False" in build
    assert "include(cmake/external_projects/deepselect.cmake)" in build
    assert 'if(VLLM_TARGET_DEVICE STREQUAL "empty")' in build
    added = _added_lines(build)
    assert "CMAKE_CXX_COMPILER" not in added
    assert "csrc/musa_api.mu" in added
    assert "define_extension_target(" in added
    assert "musa_api.cpp" not in added
    assert "vllm_musa_toolchain.cmake" not in added
    assert "deepselect_source.cmake" not in added
    assert "TORCH_LIBRARY" not in added
    assert "musa_deepselect_adapter" not in added
    assert "configure_file(" not in added
    assert not (ROOT / "csrc/deep_select/torch_bindings.mu").exists()


@pytest.mark.parametrize("skip", [False, True])
def test_native_build_delegates_deepselect_to_vllm(skip: bool) -> None:
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    build_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_CustomBuildExt"
    )
    build_class.body = [
        node
        for node in build_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    ]
    module = ast.Module(body=[build_class], type_ignores=[])
    events = []
    clones = []

    class BuildExtension:
        def run(self) -> None:
            events.append("build")

    environment = {"SKIP_THIRD_PARTY": "1" if skip else "0"}
    namespace = {
        "BuildExtension": BuildExtension,
        "os": SimpleNamespace(environ=environment),
        "resolve_proxy": lambda: None,
    }
    for name in ("VLLM", "FLASHINFER"):
        namespace[f"_{name}_REPO"] = SimpleNamespace(
            source_dir=name,
            git_repository="repository",
            git_tag="pinned-commit",
            git_shallow=False,
        )
    exec(compile(module, "setup.py", "exec"), namespace)
    build = namespace["_CustomBuildExt"]()
    build._clone_and_checkout = lambda source, *arguments: clones.append(source)
    build._apply_musa_patch_series = lambda source: events.append("patch")
    build._install_vllm = lambda source: events.append("install")
    build.run()
    assert events == ["patch", "install", "build"]
    expected_clones = [] if skip else ["VLLM", "FLASHINFER"]
    assert clones == expected_clones


def test_deepselect_source_selection_uses_public_musa_ops_repository() -> None:
    assert "DEEPSELECT_" not in PINS.read_text(encoding="utf-8")
    source_patch = (
        BUILD_PATCH.read_text()
        .split("diff --git a/cmake/external_projects/deepselect.cmake", 1)[1]
        .split("diff --git", 1)[0]
    )
    added = _added_lines(source_patch)
    assert "https://github.com/MooreThreads/DeepSelect.git" in added
    assert "2464c80eade254ef04d19182e93249010e4957d7" in added
    assert "musa_api.mu" in added



@pytest.mark.parametrize(
    "configured",
    [{}, {"VLLM_TARGET_DEVICE": "empty", "MTGPU_TARGET": "mp_22"}],
)
def test_install_vllm_keeps_empty_mode_without_injecting_hardware(
    configured: dict[str, str],
) -> None:
    tree = ast.parse((ROOT / "setup.py").read_text())
    build_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_CustomBuildExt"
    )
    method = next(
        node
        for node in build_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_install_vllm"
    )
    prefix = method.body[
        : next(
            index
            for index, node in enumerate(method.body)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(name, ast.Name) and name.id == "vllm_install_cmd"
                for name in node.targets
            )
        )
    ]
    function = ast.FunctionDef(
        name="environment",
        args=method.args,
        body=[*prefix, ast.Return(value=ast.Name(id="env", ctx=ast.Load()))],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    inherited = {"DEEPSELECT_SRC_DIR": "/prepared/DeepSelect", **configured}
    namespace = {
        "Path": Path,
        "os": SimpleNamespace(environ=inherited),
        "_PINS": {"VLLM_TAG": "v0.32.0"},
    }
    exec(compile(module, "setup.py", "exec"), namespace)
    environment = namespace["environment"]("/prepared/vllm")
    assert environment == {
        **inherited,
        "VLLM_TARGET_DEVICE": "empty",
        "VLLM_VERSION_OVERRIDE": "0.32.0",
    }


def test_vllm_build_does_not_add_musa_platform_or_compiler_detection() -> None:
    added = _added_lines(BUILD_PATCH.read_text())
    for forbidden in (
        "torch.version.musa",
        'getattr(torch.version, "musa"',
        "_is_musa()",
        'VLLM_TARGET_DEVICE = "musa"',
        'VLLM_TARGET_DEVICE STREQUAL "musa"',
        "CMAKE_CXX_COMPILER=",
        "MTGPU_TARGET",
    ):
        assert forbidden not in added


@pytest.mark.parametrize("empty", [False, True])
def test_vllm_empty_mode_always_builds_vllm_deepselect_extension(empty: bool) -> None:
    setup_patch = BUILD_PATCH.read_text().split("diff --git a/setup.py", 1)[1]
    source = "if _no_device():\n    ext_modules = []\n" + _added_lines(setup_patch)
    extensions = [SimpleNamespace(name="existing")]
    namespace = {
        "_no_device": lambda: empty,
        "CMakeExtension": lambda **kwargs: SimpleNamespace(**kwargs),
        "ext_modules": extensions,
    }
    exec(compile(source, str(BUILD_PATCH), "exec"), namespace)
    result = namespace["ext_modules"]
    if empty:
        assert [extension.name for extension in result] == ["vllm._deepselect_C"]
        assert result[0].py_limited_api is False
    else:
        assert result is extensions


@pytest.mark.parametrize("stale_only", [False, True])
def test_editable_copy_does_not_overwrite_vllm_owned_deepselect(
    tmp_path: Path, stale_only: bool
) -> None:
    tree = ast.parse((ROOT / "setup.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "develop_dynamic_library"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"Path": Path, "shutil": shutil, "PackageNotFoundError": RuntimeError}
    exec(compile(module, "setup.py", "exec"), namespace)
    source = tmp_path / "plugin" / "vllm"
    target = tmp_path / "upstream" / "vllm"
    source.mkdir(parents=True)
    target.mkdir(parents=True)
    (source / "_deepselect_C.so").write_text("stale plugin binary")
    (target / "_deepselect_C.so").write_text("new vllm binary")
    binary_source = source
    if stale_only:
        binary_source = source.parent / "build/lib.test/vllm"
        binary_source.mkdir(parents=True)
    (binary_source / "_C.so").write_text("current musa binary")
    if stale_only:
        os.utime(binary_source, (1, 1))
    namespace["develop_dynamic_library"]("vllm", source.parent, target)
    assert (target / "_deepselect_C.so").read_text() == "new vllm binary"
    assert (target / "_C.so").read_text() == "current musa binary"
