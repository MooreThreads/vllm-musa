# SPDX-License-Identifier: Apache-2.0
"""Source contract for the shape-scoped MUSA SSD chunk-scan tile.

The MUSA Triton 3.2 stack pins two tiles: `32x32x32`, one stage, eight warps for the verified
SSD tuple, and `16x16x32`, one stage, two warps for every other shape. These assertions fail if
that selection widens to every shape, if the verified tile's warp count moves, or if the small
tile's configuration drifts.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERIES = ROOT / "vllm_musa" / "patches" / "series"
MEMBER = "0163-MUSA-mamba2-ssd-nemotron-prefill-tile.patch"


def _member() -> str:
    return (SERIES / MEMBER).read_text()


def _body() -> str:
    """The post-image body: context and added lines, with marker and indentation removed.

    In this member the wide tile is an added line and the small tile is a context line of the
    same hunk, so presence has to be read from the applied body rather than from added lines
    alone.
    """
    lines = []
    for line in _member().splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith(("+", " ")):
            lines.append(line[1:].strip())
    return "\n".join(lines)


def _removed() -> str:
    """The pre-image lines the member takes away."""
    return "\n".join(
        line[1:].strip()
        for line in _member().splitlines()
        if line.startswith("-") and not line.startswith("---")
    )


def test_tile_selection_stays_shape_scoped():
    body, removed = _body(), _removed()
    # the platform guard: the pruning and the widened key are for MUSA Triton 3.2 only, and
    # every other stack keeps the upstream autotune key and search
    assert "if not is_musa_triton_32():" in body
    assert '"key": ["chunk_size", "hdim", "dstate", "nheads_ngroups_ratio", "IS_CAUSAL"]' in body
    assert 'return {"key": ["chunk_size", "hdim", "dstate", "IS_CAUSAL"]}' in body
    # the gate itself must stay, and so must the wiring that installs it
    assert "prune_configs_by" in body
    assert "**_ssd_autotune_kwargs()," in body
    assert "_musa_ssd_early_config_prune" in body
    assert "tile = 32 if is_nemotron_shape else 16" in body
    assert 'args.get("chunk_size") == 128' in body
    assert 'args.get("hdim") == 64' in body
    assert 'args.get("dstate") == 128' in body
    assert 'args.get("nheads_ngroups_ratio") == 8' in body
    # and nothing may delete the gate or its autotune key widening
    assert "_musa_ssd_early_config_prune" not in removed
    assert "_ssd_autotune_kwargs" not in removed


def test_verified_shape_uses_the_measured_configuration():
    body = _body()
    assert (
        '{"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},\n'
        "num_stages=1,\n"
        "num_warps=8,"
    ) in body, "the verified shape must select the measured 32x32x32 / 1 / 8 tile"
    assert (
        '{"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 32},\n'
        "num_stages=1,\n"
        "num_warps=2,"
    ) in body, "every other shape must keep the shipped 16x16x32 / 1 / 2 tile"
