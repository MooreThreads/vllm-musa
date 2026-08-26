#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the contents and ownership metadata of a bundled vllm-musa wheel."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from zipfile import ZipFile

REQUIRED_FILES = {
    "vllm/__init__.py",
    "vllm/_version.py",
    "vllm/py.typed",
    "vllm/transformers_utils/chat_templates/template_chatml.jinja",
    "vllm/entrypoints/serve/instrumentator/static/swagger-ui.css",
    "vllm/entrypoints/serve/instrumentator/static/swagger-ui-bundle.js",
    "vllm/utils/numa_wrapper.sh",
    "vllm_musa/__init__.py",
}

EXPECTED_ENTRY_POINTS = {
    "vllm = vllm.entrypoints.cli.main:main",
    "lora_filesystem_resolver = vllm.plugins.lora_resolvers.filesystem_resolver:register_filesystem_resolver",
    "lora_hf_hub_resolver = vllm.plugins.lora_resolvers.hf_hub_resolver:register_hf_hub_resolver",
    "musa = vllm_musa:musa_platform_plugin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--require-rust", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with ZipFile(args.wheel) as archive:
        names = set(archive.namelist())
        missing = REQUIRED_FILES - names
        assert not missing, f"missing required wheel files: {sorted(missing)}"

        native = {
            name
            for name in names
            if name.endswith(".so")
            and (name.startswith("vllm/") or name.startswith("vllm_musa/"))
        }
        for prefix in (
            "vllm/_C.",
            "vllm/_C_stable_libtorch.",
            "vllm/_moe_C.",
            "vllm_musa/_C.",
        ):
            assert any(name.startswith(prefix) for name in native), prefix

        if args.require_rust:
            assert "vllm/vllm-rs" in names
            assert "vllm/_rust_tool_parser.abi3.so" in names

        dist_info = {
            name.split("/", 1)[0]
            for name in names
            if ".dist-info/" in name
        }
        assert len(dist_info) == 1, dist_info
        (dist_info_dir,) = dist_info
        assert dist_info_dir.startswith("vllm_musa-"), dist_info_dir

        top_level = archive.read(f"{dist_info_dir}/top_level.txt").decode()
        assert set(top_level.split()) == {"vllm", "vllm_musa"}
        entry_points = archive.read(f"{dist_info_dir}/entry_points.txt").decode()
        for expected in EXPECTED_ENTRY_POINTS:
            assert expected in entry_points

        version_module = archive.read("vllm/_version.py").decode()
        assert re.search(r"__version__ = version = ['\"](?!dev)", version_module)

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
        f"files={len(names)} native={len(native)} rust_required={args.require_rust}"
    )


if __name__ == "__main__":
    main()
