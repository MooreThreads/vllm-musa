# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

from build_utils.bundle_vllm import BUNDLED_DISTRIBUTION, bundle_vllm_package


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
    (package / "_C.cpython-310-x86_64-linux-gnu.so").write_bytes(b"stale")
    (package / "_rust_tool_parser.abi3.so").write_bytes(b"rust")
    (package / "vllm-rs").write_bytes(b"binary")
    return package


def test_bundle_vllm_copies_payload_and_retargets_metadata(
    fake_vllm: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "wheel" / "vllm"

    result = bundle_vllm_package(fake_vllm, destination)

    assert result.metadata_rewrites == 5
    assert (destination / "payload.json").read_text() == "{}\n"
    assert not (destination / "_C.cpython-310-x86_64-linux-gnu.so").exists()
    assert (destination / "_rust_tool_parser.abi3.so").read_bytes() == b"rust"
    assert (destination / "vllm-rs").read_bytes() == b"binary"
    lookups = (destination / "metadata_lookups.py").read_text()
    assert lookups.count(BUNDLED_DISTRIBUTION) == 5
    assert 'version("vllm")' not in lookups


def test_bundle_vllm_rejects_missing_generated_version(
    fake_vllm: Path, tmp_path: Path
) -> None:
    (fake_vllm / "_version.py").unlink()

    with pytest.raises(RuntimeError, match="version module is missing"):
        bundle_vllm_package(fake_vllm, tmp_path / "wheel" / "vllm")
