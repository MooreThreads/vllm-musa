# SPDX-License-Identifier: Apache-2.0
"""Data-driven kernel tactics.

A *tactic* is a choice that is correct on every arch but fastest on one: tile
shape, block size, split-K factor, pipeline depth.  The code that applies a
tactic is the same everywhere; only the numbers differ.  Keeping the numbers in
``tactics/*.json`` instead of in ``if`` chains means supporting a new arch is a
data change, and a missing entry is a loud lookup failure rather than a slow
default inherited by accident.

The decisions are a pure function of :class:`~.problem.TuningProblem`: the
layer's shape, the batch size, the compute semantics the caller needs, and the
device profile.  Nothing here reads the environment or the current device, so the
same problem always resolves to the same tactic -- which is what makes the tables
testable without a GPU and enumerable offline for a bin this machine does not
have.

The lookup ladder, in order, is:

1. ``(arch_name, mp_count)`` with an M band covering the problem
2. ``(arch_name, mp_count)`` with no band
3. ``(arch_name, null)`` with or without a band
4. ``(null, null)`` -- the table's own fallback
5. the caller's ``default=``  -- explicit opt-in
6. :class:`TacticNotFound`

A table may carry a ``(null, null)`` row, and then it always answers; without
one, an untuned bin fails closed.  That is the difference between "we decided a
generic tactic is fine here" and "nobody got to this bin yet", and keeping them
distinguishable is the whole point: on S5000 a wrong tiling is a 2x regression
that looks like a hardware limit, and nobody re-reads a tuning table for an arch
they just enabled.

Every row that claims an arch must carry a ``note`` saying how it was measured.
Only the ``(null, null)`` row may be generic, so provenance cannot silently go
missing.
"""

from __future__ import annotations

from .problem import (
    DENSE,
    MOE_CONTIGUOUS,
    MOE_MASKED,
    ComputeSemantics,
    DeviceProfile,
    LayerShape,
    TuningProblem,
)
from .resolve import COMPUTED_FALLBACKS, computed_fallback, resolve_tactic
from .tables import (
    DEFAULT_DIR,
    SCHEMA_VERSION,
    TacticEntry,
    TacticNotFound,
    load,
    lookup,
    table_path,
)
from .values import REGISTRY, MoeTactic, RmsnormTactic, Tactic, tactic_class

__all__ = [
    "COMPUTED_FALLBACKS",
    "DEFAULT_DIR",
    "DENSE",
    "MOE_CONTIGUOUS",
    "MOE_MASKED",
    "MoeTactic",
    "REGISTRY",
    "RmsnormTactic",
    "SCHEMA_VERSION",
    "ComputeSemantics",
    "DeviceProfile",
    "LayerShape",
    "Tactic",
    "TacticEntry",
    "TacticNotFound",
    "TuningProblem",
    "computed_fallback",
    "load",
    "lookup",
    "resolve_tactic",
    "table_path",
    "tactic_class",
]
