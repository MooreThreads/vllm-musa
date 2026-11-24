#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify bundled-wheel or editable-install package ownership."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import sys
import sysconfig
from pathlib import Path
from urllib.parse import unquote, urlsplit

from packaging.specifiers import SpecifierSet
from packaging.version import Version

NATIVE_MODULES = ("vllm._C", "vllm._C_stable_libtorch", "vllm._moe_C", "vllm_musa._C")


def _module_files() -> dict[str, str]:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    assert suffix, "Python extension suffix is unavailable"
    return {
        "vllm_musa": "vllm_musa/__init__.py",
        "vllm": "vllm/__init__.py",
        **{name: f"{name.replace('.', '/')}{suffix}" for name in NATIVE_MODULES},
    }


def _import_verified_modules(paths: dict[str, Path]) -> None:
    for name, expected in paths.items():
        spec = importlib.util.find_spec(name)
        assert spec is not None and spec.origin is not None, f"cannot locate {name}"
        actual = Path(spec.origin).resolve()
        assert (
            actual == expected.resolve()
        ), f"{name}: expected {expected}, found {actual}"
        importlib.import_module(name)


def _editable_source(distribution_name: str) -> Path:
    dist = importlib.metadata.distribution(distribution_name)
    raw = dist.read_text("direct_url.json")
    assert raw, f"{distribution_name} has no direct_url.json"
    direct_url = json.loads(raw)
    assert direct_url.get("dir_info", {}).get("editable") is True, distribution_name
    url = urlsplit(direct_url["url"])
    assert url.scheme == "file" and url.netloc in ("", "localhost"), url.geturl()
    return Path(unquote(url.path)).resolve()


def verify_bundled() -> None:
    dist = importlib.metadata.distribution("vllm-musa")
    try:
        importlib.metadata.distribution("vllm")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise AssertionError("the bundled install must not provide a vllm distribution")

    assert dist.read_text("RECORD") is not None, "vllm-musa wheel RECORD is missing"
    recorded = {file.as_posix() for file in dist.files or ()}
    module_files = _module_files()
    jit_header = "vllm_musa/jit_kernel/tilelang/_atomic_helper.h"
    missing = (set(module_files.values()) | {jit_header}) - recorded
    assert not missing, f"vllm-musa wheel RECORD is missing {sorted(missing)}"
    assert Path(dist.locate_file(jit_header)).is_file(), jit_header
    _import_verified_modules(
        {name: Path(dist.locate_file(file)) for name, file in module_files.items()}
    )

    import numpy
    import vllm

    assert numpy.__version__ in SpecifierSet("<2"), numpy.__version__
    assert not Version(vllm.__version__).is_devrelease, vllm.__version__
    rust_frontend = Path(dist.locate_file("vllm/vllm-rs"))
    if rust_frontend.exists():
        assert "vllm/vllm-rs" in recorded
        assert rust_frontend.stat().st_mode & 0o111

    for group, names in {
        "vllm.platform_plugins": {"musa"},
        "vllm.general_plugins": {
            "lora_filesystem_resolver",
            "lora_hf_hub_resolver",
            "musa_custom_ops",
        },
    }.items():
        selected = {
            entry.name: entry for entry in dist.entry_points.select(group=group)
        }
        for name in names:
            selected[name].load()

    print(
        f"PASS bundled_install distribution=vllm-musa vllm_version={vllm.__version__}"
    )


def verify_editable() -> None:
    musa_source = _editable_source("vllm-musa")
    vllm_source = _editable_source("vllm")
    expected_vllm_source = (musa_source / "third_party/vllm").resolve()
    assert vllm_source == expected_vllm_source, vllm_source
    sources = {"vllm_musa": musa_source, "vllm": vllm_source}
    _import_verified_modules(
        {
            name: sources[name.partition(".")[0]] / file
            for name, file in _module_files().items()
        }
    )
    print("PASS editable_install distributions=vllm,vllm-musa")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("bundled", "editable"))
    args = parser.parse_args()
    if args.mode == "bundled":
        # Check module resolution from the runtime working directory.
        sys.path.insert(0, str(Path.cwd()))
        verify_bundled()
    else:
        verify_editable()


if __name__ == "__main__":
    main()
