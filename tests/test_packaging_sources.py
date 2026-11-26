# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
from pathlib import Path
from typing import Iterator

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from setuptools.command.egg_info import FileList

ROOT = Path(__file__).resolve().parents[1]

# Upstream and MUSA specifiers for deliberate overrides. Review both on upgrades.
RUNTIME_OVERRIDES = {
    # The MUSA stack requires NumPy 1.x, which is supported by OpenCV 4.11.
    "numpy": ("", "<2"),
    "opencv-python-headless": (">=4.13.0", ">=4.11,<4.12"),
    # Keep the validated tokenizer and Transformers stack.
    "transformers": (">=5.16.1,<5.20.0", "==5.19.0"),
    "huggingface-hub": (">=1.31.0", ">=1.31.0,<2.0"),
    "tokenizers": (">=0.23.2", "==0.23.3"),
}


def test_sdist_manifest_includes_native_sources_and_patch_metadata(monkeypatch):
    # Check the MANIFEST.in fallback independently of setuptools-scm's Git inventory.
    monkeypatch.chdir(ROOT)
    expected = {
        path.as_posix()
        for folder in ("csrc", "vllm_musa/patches/metadata")
        for path in Path(folder).rglob("*")
        if path.is_file()
        and path.suffix in {".cpp", ".cu", ".mu", ".h", ".cuh", ".muh", ".json"}
    }
    assert expected
    manifest = FileList()
    manifest.set_allfiles(sorted(expected))
    for line in Path("MANIFEST.in").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            manifest.process_template_line(line)
    assert expected <= set(manifest.files), sorted(expected - set(manifest.files))


def _active_requirements(
    path: Path, environment: dict[str, str]
) -> Iterator[tuple[str, Requirement]]:
    for raw_line in path.read_text().splitlines():
        line = raw_line.split(" #", 1)[0].strip()
        if not line or line.startswith(("#", "--")):
            continue
        if line.startswith("-r "):
            yield from _active_requirements(
                path.parent / line.split(maxsplit=1)[1], environment
            )
            continue
        requirement = Requirement(line)
        if requirement.marker is None or requirement.marker.evaluate(environment):
            yield canonicalize_name(requirement.name), requirement


def test_pinned_upstream_requirements():
    pin_repo = os.environ.get("PIN_REPO")
    if pin_repo is None:
        pytest.skip("Set PIN_REPO to a checkout of the vLLM pin.")
    pins = dict(
        line.split("=", 1)
        for line in (ROOT / "third_party/PINS").read_text().splitlines()
        if line and not line.startswith("#")
    )
    assert (
        subprocess.check_output(
            ["git", "-C", pin_repo, "rev-parse", "HEAD"], text=True
        ).strip()
        == pins["VLLM_COMMIT"]
    )
    # Match the Linux x86_64 / Python 3.12 production wheel.
    environment = {
        "os_name": "posix",
        "platform_machine": "x86_64",
        "platform_system": "Linux",
        "sys_platform": "linux",
        "python_full_version": "3.12.0",
        "python_version": "3.12",
    }
    local = {}
    for filename in ("common.txt", "vllm_runtime_transitive.txt", "musa_private.txt"):
        for name, requirement in _active_requirements(
            ROOT / "requirements" / filename, environment
        ):
            assert name not in local, f"Duplicate MUSA runtime requirement for {name}"
            local[name] = requirement

    upstream = _active_requirements(
        Path(pin_repo) / "requirements/common.txt", environment
    )
    errors = []
    for name, requirement in upstream:
        ours = local.get(name)
        override = RUNTIME_OVERRIDES.get(name)
        if (
            ours is None
            or not requirement.extras <= ours.extras
            or requirement.url != ours.url
            or (
                requirement.specifier != ours.specifier
                and (
                    override is None
                    or requirement.specifier != SpecifierSet(override[0])
                    or ours.specifier != SpecifierSet(override[1])
                )
            )
        ):
            errors.append(f"{requirement} (MUSA: {ours or 'missing'})")
    assert not errors, "Review changed vLLM runtime requirements:\n" + "\n".join(errors)
