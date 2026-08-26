#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify bundled-wheel or editable-install package ownership."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path


def _direct_url(distribution_name: str) -> dict:
    distribution = importlib.metadata.distribution(distribution_name)
    path = Path(distribution._path) / "direct_url.json"
    return json.loads(path.read_text())


def verify_bundled() -> None:
    import vllm

    import vllm_musa

    assert importlib.metadata.version("vllm-musa")
    try:
        importlib.metadata.distribution("vllm")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise AssertionError("the bundled install must not provide a vllm distribution")

    owners = importlib.metadata.packages_distributions()
    assert "vllm-musa" in owners["vllm"]
    assert "vllm-musa" in owners["vllm_musa"]
    assert vllm.__version__ != "dev"
    assert "/vllm-workspace" not in str(Path(vllm.__file__).resolve())
    assert "/vllm-workspace" not in str(Path(vllm_musa.__file__).resolve())

    for module_name in (
        "vllm._C",
        "vllm._C_stable_libtorch",
        "vllm._moe_C",
        "vllm_musa._C",
    ):
        importlib.import_module(module_name)
    rust_frontend = Path(vllm.__file__).with_name("vllm-rs")
    if rust_frontend.exists():
        assert rust_frontend.stat().st_mode & 0o111
        importlib.import_module("vllm._rust_tool_parser")

    from vllm.utils.import_utils import get_vllm_optional_dependencies

    optional_dependencies = get_vllm_optional_dependencies()
    assert "audio" in optional_dependencies
    assert "bench" in optional_dependencies

    entry_points = importlib.metadata.entry_points()
    for group, names in {
        "vllm.platform_plugins": {"musa"},
        "vllm.general_plugins": {
            "lora_filesystem_resolver",
            "lora_hf_hub_resolver",
            "musa_custom_ops",
        },
    }.items():
        selected = {entry.name: entry for entry in entry_points.select(group=group)}
        for name in names:
            selected[name].load()

    print(
        "PASS bundled_install distribution=vllm-musa "
        f"vllm_version={vllm.__version__}"
    )


def verify_editable() -> None:
    import vllm

    import vllm_musa

    expected = {
        "vllm": "/vllm-workspace/third_party/vllm",
        "vllm-musa": "/vllm-workspace",
    }
    for name, url_suffix in expected.items():
        direct_url = _direct_url(name)
        assert direct_url["dir_info"]["editable"] is True
        assert direct_url["url"].endswith(url_suffix)

    assert str(Path(vllm.__file__).resolve()).startswith(expected["vllm"])
    assert str(Path(vllm_musa.__file__).resolve()).startswith(expected["vllm-musa"])
    importlib.import_module("vllm._C")
    importlib.import_module("vllm_musa._C")
    print("PASS editable_install distributions=vllm,vllm-musa")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("bundled", "editable"))
    args = parser.parse_args()
    if args.mode == "bundled":
        verify_bundled()
    else:
        verify_editable()


if __name__ == "__main__":
    main()
