# SPDX-License-Identifier: Apache-2.0
import os
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

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


def test_sdist_contains_build_inputs_without_git(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT, text=True
    ).split("\0")
    for path in filter(None, tracked):
        original = ROOT / path
        if original.is_file():
            target = source / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
    assert not (source / ".git").exists()
    build_inputs = (
        "build_utils/**/*.py",
        "csrc/**/*",
        "requirements/**/*.txt",
        "vllm_musa/patches/metadata/**/*",
        "vllm_musa/patches/series/**/*.patch",
    )
    expected = {
        path.relative_to(source).as_posix()
        for pattern in build_inputs
        for path in source.glob(pattern)
        if path.is_file()
    }
    expected.update(
        {
            "PKG-INFO",
            "third_party/PINS",
            "vllm_musa/patches/build_apply.py",
            "vllm_musa/jit_kernel/tilelang/_atomic_helper.h",
            "vllm_musa_omni_compat_pth.py",
        }
    )
    dist = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(dist)],
        cwd=source,
        check=True,
    )
    (sdist,) = dist.glob("*.tar.gz")
    with tarfile.open(sdist) as archive:
        packaged = {
            member.name.split("/", 1)[1]
            for member in archive.getmembers()
            if member.isfile()
        }
    assert expected <= packaged, sorted(expected - packaged)


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
