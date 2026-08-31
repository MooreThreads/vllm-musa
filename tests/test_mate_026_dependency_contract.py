# SPDX-License-Identifier: Apache-2.0
"""Contract checks for the common MATE 0.2.6 dependency cohort."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_REQUIREMENTS = ROOT / "requirements/musa_private.txt"
DOCKERFILE = ROOT / "docker/musa.Dockerfile"

MATE_026_COHORT = {
    "mate": ("0.2.6", "mate"),
    "mate-mubin": ("0.2.6", "mate_mubin"),
    "flash_attn_3": ("0.2.6+musa", "flash_attn_3"),
    "flash_mla": ("0.2.6+musa", "flash_mla"),
    "deep-gemm": ("0.2.6+musa", "deep_gemm"),
    "flashinfer-python": ("0.2.6+musa", "flashinfer"),
    "sageattention": ("0.2.6+musa", "sageattention"),
    "tilelang_musa": ("0.1.12+musa.2", "tilelang"),
    "apache-tvm-ffi": ("0.1.11.post1+musa.1", "tvm_ffi"),
}


def _exact_requirements() -> dict[str, str]:
    requirements: dict[str, str] = {}
    for raw_line in PRIVATE_REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([^=<>!~\s]+)==([^\s]+)", line)
        assert match is not None, f"expected an exact private requirement: {line}"
        distribution, version = match.groups()
        assert distribution not in requirements, f"duplicate pin: {distribution}"
        requirements[distribution] = version
    return requirements


def test_mate_026_cohort_is_exactly_pinned() -> None:
    requirements = _exact_requirements()

    for distribution, (version, _) in MATE_026_COHORT.items():
        assert requirements.get(distribution) == version


def test_docker_verifies_distribution_and_import_from_requirements() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    exact_gate = dockerfile.split("exact_version_dists =", 1)[1].split(
        "expected =", 1
    )[0]

    for distribution, (version, module) in MATE_026_COHORT.items():
        expected = (
            f'("{distribution}", "{module}", '
            f'requirement_prefix("{distribution}"))'
        )
        assert expected in dockerfile
        assert f'"{distribution}"' in exact_gate
        assert f"{distribution}=={version}" not in dockerfile, (
            f"{distribution} version must come from requirements/musa_private.txt"
        )
