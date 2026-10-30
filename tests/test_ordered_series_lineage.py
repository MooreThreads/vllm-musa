# SPDX-License-Identifier: Apache-2.0
"""Contracts for the ordered v0.32 source-patch series."""

import re
from pathlib import Path


SERIES = Path(__file__).resolve().parents[1] / "vllm_musa" / "patches" / "series"


def test_series_numbers_are_contiguous_and_lineage_is_complete() -> None:
    patches = sorted(SERIES.glob("[0-9][0-9][0-9][0-9]-*.patch"))
    numbers = [int(path.name[:4]) for path in patches]
    assert numbers == list(range(1, len(patches) + 1))

    readme = (SERIES / "README.md").read_text(encoding="utf-8")
    mapped = re.findall(r"^\| `(\d{4}-[^`]+\.patch)` \|", readme, re.M)
    assert mapped == [patch.name for patch in patches]
    assert f"Currently **{len(patches)} patches**" in readme
    assert "v0.28 0065: superseded" in readme
    assert "v0.28 0177: superseded" in readme

    grouped = {
        37: ("v0.28 0037", "v0.28 0038"),
        78: ("v0.28 0081", "v0.28 0082", "v0.28 0083"),
        111: ("v0.28 0125", "v0.28 0127", "v0.28 0128"),
        134: ("v0.28 0155", "v0.28 0156", "v0.28 0157"),
    }
    for new, ancestry in grouped.items():
        line = next(
            line for line in readme.splitlines()
            if line.startswith(f"| `{new:04d}-")
        )
        assert all(item in line for item in ancestry)

    for old, new in ((84, 79), (111, 102), (112, 103), (122, 108), (146, 126)):
        assert f"v0.28 {old:04d}" in next(
            line for line in readme.splitlines() if line.startswith(f"| `{new:04d}-")
        )


def test_superseded_changes_are_applied_directly() -> None:
    compiler_fix = next(SERIES.glob("*-split-chained-boolean-operators-*.patch"))
    topk_dispatch = next(SERIES.glob("*-select-indexer-top-k-with-the-staged-*.patch"))
    compiler_source = compiler_fix.read_text(encoding="utf-8")
    topk_source = topk_dispatch.read_text(encoding="utf-8")

    assert "-    if USE_MM_PREFIX or USE_R_SWA or USE_PER_SEQ_CAUSAL" in compiler_source
    assert "+    if USE_MM_PREFIX or (USE_R_SWA or" in compiler_source
    assert "-        _musa_custom_ops.sparse_indexer_topk_decode(" in topk_source
    assert "+        _musa_select_indexer_topk(" in topk_source
    assert "-    select_topk = (" not in topk_source
