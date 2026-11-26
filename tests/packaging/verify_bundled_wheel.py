#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the contents and ownership metadata of a bundled vllm-musa wheel."""

from __future__ import annotations

import argparse
import csv
import re
from configparser import ConfigParser
from email.parser import Parser
from pathlib import Path
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

REQUIRED_FILES = {
    "vllm/__init__.py",
    "vllm/_version.py",
    "vllm/py.typed",
    "vllm/transformers_utils/chat_templates/template_chatml.jinja",
    "vllm/entrypoints/serve/instrumentator/static/swagger-ui.css",
    "vllm/entrypoints/serve/instrumentator/static/swagger-ui-bundle.js",
    "vllm/utils/numa_wrapper.sh",
    "vllm_musa/__init__.py",
    "vllm_musa/jit_kernel/tilelang/_atomic_helper.h",
    "vllm_musa_omni_compat_pth.py",
}

NATIVE_MODULES = ("vllm/_C", "vllm/_C_stable_libtorch", "vllm/_moe_C", "vllm_musa/_C")
EXTRA_REQUIREMENTS = {
    "dev": {"pytest", "ruff"},
    "musa": {"torch", "torch-musa", "mate", "flashinfer-python"},
}

EXPECTED_ENTRY_POINTS = {
    "console_scripts": {
        "vllm": "vllm.entrypoints.cli.main:main",
        "vllm_collect_env": "vllm_musa:collect_env",
    },
    "vllm.platform_plugins": {"musa": "vllm_musa:musa_platform_plugin"},
    "vllm.general_plugins": {
        "lora_filesystem_resolver": "vllm.plugins.lora_resolvers.filesystem_resolver:register_filesystem_resolver",
        "lora_hf_hub_resolver": "vllm.plugins.lora_resolvers.hf_hub_resolver:register_hf_hub_resolver",
        "musa_out_of_tree": "vllm_musa:register_out_of_tree",
        "musa_attn_backends": "vllm_musa:register_attn_backends",
    },
}


def _verify_native_modules(wheel: Path, names: set[str]) -> None:
    # The wheel tag identifies the ABI even when this verifier runs on a different Python.
    _, _, _, tags = parse_wheel_filename(wheel.name)
    assert len(tags) == 1, tags
    (tag,) = tags
    abi = re.fullmatch(r"cp(\d+)", tag.abi)
    assert abi and tag.interpreter == tag.abi, tag
    assert tag.platform.startswith("linux_"), tag
    arch = tag.platform.removeprefix("linux_")
    suffix = f".cpython-{abi[1]}-{arch}-linux-gnu.so"
    for module in NATIVE_MODULES:
        expected = f"{module}{suffix}"
        actual = {
            name
            for name in names
            if name.startswith(f"{module}.") and name.endswith(".so")
        }
        assert actual == {
            expected
        }, f"{module}: expected {expected}, found {sorted(actual)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--require-rust", action="store_true")
    return parser.parse_args()


def _verify_extra_requirements(
    requirements: list[Requirement], extras: set[str]
) -> None:
    assert EXTRA_REQUIREMENTS.keys() <= extras, extras
    for extra, expected in EXTRA_REQUIREMENTS.items():
        matching = [
            req for req in requirements if canonicalize_name(req.name) in expected
        ]
        assert {canonicalize_name(req.name) for req in matching} == expected, extra
        for req in matching:
            assert req.marker is not None and req.marker.evaluate({"extra": extra}), req
            assert not any(
                req.marker.evaluate({"extra": other})
                for other in (extras - {extra}) | {""}
            ), req


def main() -> None:
    args = parse_args()
    with ZipFile(args.wheel) as archive:
        names = set(archive.namelist())
        missing = REQUIRED_FILES - names
        assert not missing, f"missing required wheel files: {sorted(missing)}"

        _verify_native_modules(args.wheel, names)

        if args.require_rust:
            assert "vllm/vllm-rs" in names

        dist_info = {name.split("/", 1)[0] for name in names if ".dist-info/" in name}
        assert len(dist_info) == 1, dist_info
        (dist_info_dir,) = dist_info
        assert dist_info_dir.startswith("vllm_musa-"), dist_info_dir
        record = archive.read(f"{dist_info_dir}/RECORD").decode()
        recorded = {path for path, _, _ in csv.reader(record.splitlines())}
        assert REQUIRED_FILES <= recorded, sorted(REQUIRED_FILES - recorded)

        metadata = Parser().parsestr(archive.read(f"{dist_info_dir}/METADATA").decode())
        assert canonicalize_name(metadata["Name"]) == "vllm-musa"
        extras = set(metadata.get_all("Provides-Extra", []))
        requirements = [
            Requirement(requirement)
            for requirement in metadata.get_all("Requires-Dist", [])
        ]
        assert all(canonicalize_name(req.name) != "vllm" for req in requirements)
        numpy_requirements = [
            req for req in requirements if canonicalize_name(req.name) == "numpy"
        ]
        assert numpy_requirements == [Requirement("numpy<2")], numpy_requirements
        _verify_extra_requirements(requirements, extras)

        top_level = archive.read(f"{dist_info_dir}/top_level.txt").decode()
        assert set(top_level.split()) == {
            "vllm",
            "vllm_musa",
            "vllm_musa_omni_compat_pth",
        }
        entry_points = ConfigParser(interpolation=None)
        entry_points.read_string(
            archive.read(f"{dist_info_dir}/entry_points.txt").decode()
        )
        for group, expected in EXPECTED_ENTRY_POINTS.items():
            for name, target in expected.items():
                assert entry_points.get(group, name, fallback=None) == target, (
                    group,
                    name,
                )

        version_module = archive.read("vllm/_version.py").decode()
        version = re.search(
            r"^__version__ = version = ['\"]([^'\"]+)['\"]", version_module, re.M
        )
        assert (
            version and not Version(version[1]).is_devrelease
        ), "missing vLLM release version"

        old_metadata_lookups = []
        for name in names:
            if name.startswith("vllm/") and name.endswith(".py"):
                source = archive.read(name).decode()
                if re.search(
                    r"(?:version|metadata|distribution)\(\s*['\"]vllm['\"]",
                    source,
                ):
                    old_metadata_lookups.append(name)
        assert not old_metadata_lookups, old_metadata_lookups

    print(
        f"PASS bundled_wheel wheel={args.wheel.name} "
        f"files={len(names)} native={len(NATIVE_MODULES)} rust_required={args.require_rust}"
    )


if __name__ == "__main__":
    main()
