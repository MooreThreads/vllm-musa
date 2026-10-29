# SPDX-License-Identifier: Apache-2.0
"""Contract checks for the common MATE 0.2.8 dependency cohort."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_REQUIREMENTS = ROOT / "requirements/musa_private.txt"
DOCKERFILE = ROOT / "docker/musa.Dockerfile"
VERIFIER = ROOT / "docker/verify_installed_dependencies.py"

MATE_028_COHORT = {
    "mate": ("0.2.8", "mate"),
    "mate-mubin": ("0.2.8", "mate_mubin"),
    "flash_attn_3": ("0.2.8+musa", "flash_attn_3"),
    "flash_mla": ("0.2.8+musa", "flash_mla"),
    "deep-gemm": ("0.2.8+musa", "deep_gemm"),
    "flashinfer-python": ("0.2.8+musa", "flashinfer"),
    "sageattention": ("0.2.8+musa", "sageattention"),
    "apache-tvm-ffi": ("0.1.11.post1+musa.1", "tvm_ffi"),
}

# tilelang-musa is not a direct pin: MATE's [tilelang] extra declares it, so the
# requirement file asks for the extra and the Dockerfile reads the expected
# version back from that metadata instead of restating it.
MATE_TILELANG_EXTRA = "mate[tilelang]==0.2.8"
MATE_TILELANG_DIST = "tilelang_musa"
MATE_TILELANG_MODULE = "tilelang"


def _exact_requirements() -> dict[str, str]:
    requirements: dict[str, str] = {}
    for raw_line in PRIVATE_REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([^=<>!~\s]+)==([^\s]+)", line)
        assert match is not None, f"expected an exact private requirement: {line}"
        distribution, version = match.groups()
        distribution = distribution.split("[", 1)[0]
        assert distribution not in requirements, f"duplicate pin: {distribution}"
        requirements[distribution] = version
    return requirements


def test_mate_028_cohort_is_exactly_pinned() -> None:
    requirements = _exact_requirements()

    for distribution, (version, _) in MATE_028_COHORT.items():
        assert requirements.get(distribution) == version


def test_image_verifier_reads_every_version_from_the_requirements() -> None:
    verifier = VERIFIER.read_text(encoding="utf-8")
    exact_gate = verifier.split("EXACT_VERSION_DISTS =", 1)[1].split(
        "#: Distributions imported", 1
    )[0]

    for distribution, (version, module) in MATE_028_COHORT.items():
        assert f'"{distribution}"' in verifier, distribution
        assert f'requirement_prefix("{distribution}")' in verifier, distribution
        assert f'"{distribution}"' in exact_gate, distribution
        assert f"{distribution}=={version}" not in verifier, (
            f"{distribution} version must come from requirements/musa_private.txt"
        )

    # The Dockerfile only runs the verifier, so cohort changes stay in one file.
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "RUN python docker/verify_installed_dependencies.py" in dockerfile
    assert "exact_version_dists" not in dockerfile


def test_tilelang_comes_from_the_mate_extra() -> None:
    requirements_text = PRIVATE_REQUIREMENTS.read_text(encoding="utf-8")
    assert MATE_TILELANG_EXTRA in requirements_text.splitlines()
    assert "tilelang_musa==" not in requirements_text, (
        "tilelang-musa must not be pinned by hand: MATE's [tilelang] extra owns it"
    )

    verifier = VERIFIER.read_text(encoding="utf-8")
    assert 'mate_extra_pin("tilelang", "tilelang-musa")' in verifier, (
        "the image verifier must read the tilelang pin from MATE metadata"
    )
    assert f'("{MATE_TILELANG_DIST}", "{MATE_TILELANG_MODULE}"' in verifier
    assert '"0.1.12+musa.5"' not in verifier, (
        "the tilelang version must come from mate's metadata, not a literal"
    )
