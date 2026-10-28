# SPDX-License-Identifier: Apache-2.0
"""Table-driven selection for the split-K FP8 GEMM.

The numbers used to live in two parallel dicts in ``kernels/fp8_block_splitk.py``,
one for the row limit of a captured launch and one for an eager launch, with the
dispatch picking between them by calling ``is_current_stream_capturing()``.  They
are rows of ``tactics/fp8_block_splitk.json`` now: same numbers, same bands, but
the band a row was measured for is written down instead of implied by which dict
it is in.

This module is deliberately free of torch.  Selection is a pure function of
``(N, K, rows, capture)``, so it is tested against a golden capture of the
pre-migration behaviour (``tests/data/fp8_block_splitk_golden.json``) on a machine
with no GPU and no torch installed.  The caller keeps the parts that need a
tensor: the dtype, stride and scale checks, and the launch itself.
"""

from __future__ import annotations

from ..problem import LayerShape, TuningProblem
from ..tables import TacticEntry, TacticNotFound, lookup_entry

#: The table this module reads.
KERNEL = "fp8_block_splitk"

__all__ = ["KERNEL", "config_for"]

#: Group width of the 32x32 block-scaled weight format.
_GROUP = 32


def _entry_for(
    shape_n: int, shape_k: int, rows: int, graph_capture: bool
) -> TacticEntry | None:
    problem = TuningProblem(
        layer=LayerShape(
            shape_n=shape_n,
            shape_k=shape_k,
            dtype="bf16",
            weight_dtype="fp8",
            group_size=_GROUP,
        ),
        m=rows,
        graph_capture=graph_capture,
    )
    try:
        return lookup_entry(KERNEL, problem)
    except TacticNotFound:
        return None


def config_for(
    shape_n: int, shape_k: int, rows: int, *, graph_capture: bool
) -> tuple[int, int, int, int, int, int, int] | None:
    """The launch tuple for these rows, or ``None`` when the GEMM does not apply.

    The tuple is ``(split, block_n, num_warps, unroll, min_rows, max_rows,
    tile_m)``.  ``min_rows``/``max_rows`` are the band the matched row was
    measured for; the kernel ignores them, and callers that log them are saying
    which band answered.
    """
    if rows <= 0 or shape_n <= 0 or shape_k <= 0:
        return None
    entry = _entry_for(shape_n, shape_k, rows, graph_capture)
    if entry is None:
        return None
    tactic = entry.tactic
    low, high = entry.match.m_range
    return (
        tactic.split,
        tactic.block_n,
        tactic.num_warps,
        tactic.unroll,
        low,
        high,
        tactic.tile_m,
    )
