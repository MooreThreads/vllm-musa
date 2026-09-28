# SPDX-License-Identifier: Apache-2.0
"""The post2/Triton 3.6 stack must use the upstream SSD autotune search."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERIES = ROOT / "vllm_musa" / "patches" / "series"


def test_triton32_ssd_compatibility_patches_are_removed():
    names = {p.name for p in SERIES.glob("*.patch")}
    prefixes = ("0117-", "0118-", "0140-", "0141-", "0142-", "0143-", "0146-", "0147-", "0163-")
    assert not any(name.startswith(prefix) for prefix in prefixes for name in names)
    text = "\n".join(p.read_text() for p in SERIES.glob("*.patch"))
    assert "is_musa_triton_32" not in text
    assert "Triton 3.2" not in text
