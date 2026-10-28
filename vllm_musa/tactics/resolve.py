# SPDX-License-Identifier: Apache-2.0
"""One entry point: a tuning problem in, a tactic out.

The order is deliberate.

1. **The table.**  Measured numbers beat derived ones: a table entry exists
   because someone ran this kernel on this bin, which no amount of arithmetic
   replaces.
2. **A computed answer**, but only where the arithmetic is *sound*: a bound the
   kernel's own resource model decides (shared memory, barrier budget, register
   budget).  A computed tile shape is a guess wearing a formula, so the computed
   path is registered per kernel and most kernels do not have one.
3. **The caller's ``default=``.**  This is the caller stating that a generic
   tactic is acceptable here.  It has to be explicit, because it is what silences
   the failure for a bin nobody has tuned -- and a wrong tiling on S5000 looks
   like a hardware limit, not like a missing table row.
4. :class:`TacticNotFound`.

Steps 2 and 3 are deliberately not merged: a computed answer is evidence, a
default is a waiver, and a log line should be able to tell them apart.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from .problem import TuningProblem
from .tables import TacticNotFound, lookup
from .values import Tactic

__all__ = [
    "COMPUTED_FALLBACKS",
    "computed_fallback",
    "resolve_tactic",
]

logger = logging.getLogger(__name__)

#: ``kernel name -> f(problem) -> tactic or None``.  Register only bounds the
#: kernel's resource model actually decides; see the module docstring.
COMPUTED_FALLBACKS: dict[str, Callable[[TuningProblem], Tactic | None]] = {}


def computed_fallback(kernel: str):
    """Register ``f(problem) -> Tactic | None`` as the computed path for ``kernel``."""

    def register(
        function: Callable[[TuningProblem], Tactic | None]
    ) -> Callable[[TuningProblem], Tactic | None]:
        COMPUTED_FALLBACKS[kernel] = function
        return function

    return register


def resolve_tactic(
    kernel: str,
    problem: TuningProblem,
    *,
    default: Tactic | None = None,
    directory: Path | None = None,
) -> Tactic:
    """The tactic for ``problem``: table, computed, ``default``, then raise."""
    try:
        return lookup(kernel, problem, directory)
    except TacticNotFound as missing:
        computed = COMPUTED_FALLBACKS.get(kernel)
        if computed is not None:
            tactic = computed(problem)
            if tactic is not None:
                logger.debug(
                    "tactics: %s for %s came from the resource model, "
                    "not a table entry",
                    kernel,
                    problem,
                )
                return tactic
        if default is not None:
            logger.debug(
                "tactics: %s for %s fell back to the caller's default", kernel, problem
            )
            return default
        raise missing
