# SPDX-License-Identifier: Apache-2.0
"""third_party/PINS is the single source of truth for upstream pins.

Asserts setup.py and Makefile.sync BOTH prefer the immutable vLLM commit from
`third_party/PINS` (and neither hardcodes it), while retaining the release tag as
a human-readable label. Pure file parsing + `sed` — no MUSA hardware or heavy
setup.py import.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parent.parent
PINS = ROOT / "third_party" / "PINS"


def _parse_pins() -> dict:
    """The same KEY=VALUE parse setup.py::_read_pins uses."""
    pins = {}
    for line in PINS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        pins[key.strip()] = value.split("#", 1)[0].strip()
    return pins


def _normalized_requirements(path: Path) -> set[str]:
    requirements = set()
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if " #" in line:
            line = line.split(" #", 1)[0].strip()
        requirements.add(str(Requirement(line)))
    return requirements


def test_pins_exists_with_required_keys():
    assert PINS.is_file(), "third_party/PINS missing"
    pins = _parse_pins()
    assert pins.get("VLLM_TAG"), pins
    assert pins.get("VLLM_COMMIT"), pins
    assert pins.get("FLASHINFER_COMMIT"), pins


def test_pins_is_tracked_not_ignored():
    # third_party/* is ignored but PINS must be re-included (tracked).
    r = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "third_party/PINS"],
        capture_output=True,
        text=True,
    )
    # check-ignore exits 0 (and echoes the path) when the path IS ignored.
    assert r.returncode != 0, "third_party/PINS is gitignored — it must be tracked"


def test_setup_py_reads_pins_not_literal():
    src = (ROOT / "setup.py").read_text()
    assert (
        "_read_pins(" in src and '_PINS.get("VLLM_COMMIT", _PINS["VLLM_TAG"])' in src
    ), "setup.py must prefer VLLM_COMMIT from PINS"
    assert '_PINS["FLASHINFER_COMMIT"]' in src
    assert 'git_tag="v0.22.0"' not in src, "setup.py still hardcodes the vLLM tag"


def test_makefile_sync_reads_pins_not_literal():
    mk = (ROOT / "Makefile.sync").read_text()
    assert "third_party/PINS" in mk, "Makefile.sync must read the pin from PINS"
    assert "VLLM_REF :=" in mk
    assert "checkout -f --detach FETCH_HEAD" in mk
    assert "checkout -f v0.24.0" not in mk


@pytest.mark.skipif(shutil.which("sed") is None, reason="sed unavailable")
def test_setup_and_makefile_resolve_same_commit():
    pins_commit = _parse_pins()["VLLM_COMMIT"]
    # Replicate Makefile.sync's exact commit extraction.
    sed_commit = subprocess.run(
        ["sed", "-n", "s/^VLLM_COMMIT=//p", str(PINS)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert sed_commit == pins_commit, (sed_commit, pins_commit)
    assert len(pins_commit) == 40


def test_vllm_runtime_requirements_match_prepared_pin():
    upstream = ROOT / "third_party" / "vllm" / "requirements" / "common.txt"
    if not upstream.is_file():
        pytest.skip("pinned upstream vLLM source is not prepared")

    bundled = ROOT / "requirements" / "vllm_common.txt"
    from tools.sync_vllm_requirements import sync_snapshot

    assert sync_snapshot(upstream, bundled, check=True)
    assert _normalized_requirements(bundled) == _normalized_requirements(upstream)
