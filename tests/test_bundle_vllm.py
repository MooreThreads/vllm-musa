# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import os
from pathlib import Path
from runpy import run_path
from sysconfig import get_config_var
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from build_utils.bundle_vllm import (
    BUNDLED_DISTRIBUTION,
    EXPECTED_METADATA_REWRITES,
    bundle_vllm_package,
    find_prebuilt_extensions,
)

VLLM_TAG = "v0.32.0"
VLLM_REVISION = "5df63a7203cdb9328d824a27b682e77c7b9928be"
EXT_SUFFIX = get_config_var("EXT_SUFFIX")


@pytest.fixture
def fake_vllm(tmp_path: Path) -> Path:
    package = tmp_path / "source" / "vllm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("from ._version import __version__\n")
    (package / "_version.py").write_text('__version__ = "0.24.0"\n')
    (package / "metadata_lookups.py").write_text(
        "\n".join(
            (
                'importlib.metadata.version("vllm")',
                "importlib.metadata.metadata('vllm')",
                'version("vllm")',
                "metadata('vllm')",
                'distribution("vllm")',
            )
        )
    )
    (package / "payload.json").write_text("{}\n")
    for native_module in ("_C", "_C_stable_libtorch", "_moe_C", "_deepselect_C"):
        (package / f"{native_module}{EXT_SUFFIX}").write_bytes(b"prebuilt")
        (package / f"{native_module}.so").write_bytes(b"untagged")
    (package / "_rust_tool_parser.abi3.so").write_bytes(b"rust")
    (package / "vllm-rs").write_bytes(b"binary")
    return package


def test_bundle_vllm_copies_payload_and_retargets_metadata(
    fake_vllm: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "wheel" / "vllm"

    result = bundle_vllm_package(
        fake_vllm, destination, tag=VLLM_TAG, revision=VLLM_REVISION
    )

    assert result.metadata_rewrites == 5
    assert (destination / "payload.json").read_text() == "{}\n"
    assert sorted(path.name for path in destination.glob("*.so")) == [
        "_rust_tool_parser.abi3.so"
    ]
    assert (destination / "_rust_tool_parser.abi3.so").read_bytes() == b"rust"
    assert (destination / "vllm-rs").read_bytes() == b"binary"
    assert run_path(str(destination / "_version.py"))["__version__"] == "0.32.0"
    assert run_path(str(fake_vllm / "_version.py"))["__version__"] == "0.24.0"
    lookups = (destination / "metadata_lookups.py").read_text()
    assert lookups.count(BUNDLED_DISTRIBUTION) == 5
    assert 'version("vllm")' not in lookups


def test_patched_vllm_metadata_rewrite_count(tmp_path: Path) -> None:
    patched_repo = os.environ.get("PATCHED_VLLM_REPO")
    if patched_repo is None:
        pytest.skip(
            "Set PATCHED_VLLM_REPO to the vLLM checkout with the series applied."
        )

    result = bundle_vllm_package(
        Path(patched_repo) / "vllm",
        tmp_path / "vllm",
        tag=VLLM_TAG,
        revision=VLLM_REVISION,
    )
    assert result.metadata_rewrites == EXPECTED_METADATA_REWRITES


def test_bundle_vllm_writes_pinned_version_from_archive(
    fake_vllm: Path, tmp_path: Path
) -> None:
    (fake_vllm / "_version.py").unlink()
    destination = tmp_path / "wheel" / "vllm"

    bundle_vllm_package(fake_vllm, destination, tag=VLLM_TAG, revision=VLLM_REVISION)

    version = run_path(str(destination / "_version.py"))
    assert version["__version__"] == "0.32.0"
    assert version["__version_tuple__"] == (0, 32, 0)
    assert version["__commit_id__"] == VLLM_REVISION


def test_runtime_metadata_constraint_is_numpy_one_series():
    requirements = (
        Path(__file__).parents[1] / "requirements" / "common.txt"
    ).read_text()

    assert "numpy<2" in requirements
    assert "numpy==1.26.4" not in requirements


def test_hardware_private_requirements_are_an_explicit_extra():
    setup = (Path(__file__).parents[1] / "setup.py").read_text()

    assert '"common.txt"' in setup
    assert (
        'musa = {file = ["requirements/musa_private.txt"]}'
        in (Path(__file__).parents[1] / "pyproject.toml").read_text()
    )


def test_prebuilt_extensions_match_exact_abi_and_prefer_staged_package(
    tmp_path: Path,
) -> None:
    staged = tmp_path / "staged"
    checkout = tmp_path / "checkout"
    vllm_extension = Path("vllm") / f"_C{EXT_SUFFIX}"
    musa_extension = Path("vllm_musa") / f"_C{EXT_SUFFIX}"
    for path in (
        staged / vllm_extension,
        checkout / vllm_extension,
        checkout / musa_extension,
    ):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"prebuilt")

    assert find_prebuilt_extensions(
        (vllm_extension, musa_extension), (staged, checkout)
    ) == {
        vllm_extension: staged / vllm_extension,
        musa_extension: checkout / musa_extension,
    }


def test_prebuilt_extensions_reject_wrong_abi_and_untagged_files(
    tmp_path: Path,
) -> None:
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "_C.cpython-39-x86_64-linux-gnu.so").write_bytes(b"stale")
    (package / "_C.so").write_bytes(b"untagged")
    (package / f"_C_stable_libtorch{EXT_SUFFIX}").write_bytes(b"different module")

    with pytest.raises(RuntimeError, match="missing: vllm/_C"):
        find_prebuilt_extensions((Path("vllm") / f"_C{EXT_SUFFIX}",), (tmp_path,))


def test_reusing_prepared_extensions_does_not_replay_source_patches(
    tmp_path, monkeypatch
):
    setup = Path(__file__).parents[1] / "setup.py"
    tree = ast.parse(setup.read_text())
    build_ext = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_CustomBuildExt"
    )
    prepare = next(
        node
        for node in build_ext.body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_third_party"
    )
    repos = []
    for name in ("vllm", "flashinfer"):
        source_dir = tmp_path / name
        source_dir.mkdir()
        repos.append(SimpleNamespace(source_dir=source_dir))
    namespace = {
        "os": os,
        "_VLLM_REPO": repos[0],
        "_FLASHINFER_REPO": repos[1],
    }
    exec(compile(ast.Module([prepare], []), str(setup), "exec"), namespace)
    builder = SimpleNamespace(
        _third_party_prepared=False, _apply_musa_patch_series=Mock()
    )
    monkeypatch.setenv("VLLM_MUSA_REUSE_PREBUILT", "1")
    namespace["prepare_third_party"].__func__(builder)
    assert builder._third_party_prepared
    builder._apply_musa_patch_series.assert_not_called()
