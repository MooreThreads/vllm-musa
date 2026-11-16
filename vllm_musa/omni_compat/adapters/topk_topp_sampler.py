# SPDX-License-Identifier: Apache-2.0
"""``apply_top_k_top_p`` on a pin that predates its ``has_topp_only_rows`` hint.

vLLM grew a fourth parameter after v0.32.0::

    def apply_top_k_top_p(logits, k, p, has_topp_only_rows: bool = True)

documented as "must be True if any row has p < 1 and k >= logits.shape[-1] (no
active top-k)", and forwarded to the Triton kernel so it can skip the top-p-only
pipeline when no row needs it. vLLM-Omni main passes it by keyword from the
Qwen3-Omni and MiniCPM-o 4.5 talker samplers, so on the pinned three-parameter
function every talker sampling step raises TypeError.

Dropping the argument is safe in both directions, which is why this is an
adapter rather than a declared-unsupported path:

* ``True`` is the parameter's default and means "take the top-p-only path",
  which is what the pinned function does unconditionally.
* ``False`` asserts no row is top-p-only. The pinned function still runs its
  top-p masking, and masking a row with ``p == 1`` keeps the whole
  distribution, so the result is identical; the pin merely does work the newer
  version would have skipped.

The same holds for vllm-musa's own replacement, which derives the top-p-only
case from the shapes it was given rather than from a caller's hint.

Delete this adapter when ``third_party/PINS`` reaches a vLLM whose
``apply_top_k_top_p`` takes the parameter itself.
"""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

ADAPTED = "__vllm_musa_drops_has_topp_only_rows__"
NAME = "apply_top_k_top_p"


class _ReapplyingModule(ModuleType):
    """Re-runs the adapter when anything rebinds the adapted name.

    ``vllm_musa.v1.sample.topk_topp_sampler`` installs its own faster sampler
    with ``topk_topp_sampler.apply_top_k_top_p = _apply_top_k_top_p`` at module
    scope. That assignment is necessarily later than this adapter, which runs
    while the vLLM module is still being created, so a plain wrapper is
    discarded without a word and the TypeError it absorbs comes back inside the
    talker. Re-wrapping on assignment makes the adaptation independent of who
    writes last, and keeps whichever implementation won: the wrapper delegates
    to the function that was just bound, so MUSA's kernels stay in the path and
    only the argument list is widened.

    Recursion ends after one extra level: re-applying assigns the wrapper, which
    arrives here already carrying ``ADAPTED``, so :func:`apply` returns without
    assigning again.
    """

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        if name == NAME:
            apply(self)


def is_applied(module: ModuleType) -> list[str]:
    """Which names are adapted **right now**, read-only.

    A verifier must ask this rather than read a marker left on the module at
    patch time: the whole hazard here is that something rebinds the name
    afterwards, and a module-level marker cannot see that.
    """
    live = getattr(module, NAME, None)
    return [NAME] if getattr(live, ADAPTED, False) else []


def apply(module: ModuleType) -> list[str]:
    """Let the pinned ``apply_top_k_top_p`` accept and ignore the newer hint."""
    original = getattr(module, NAME, None)
    if original is None or getattr(original, ADAPTED, False):
        return []

    try:
        parameters = inspect.signature(original).parameters
    except (TypeError, ValueError):  # pragma: no cover - C callables
        parameters = {}
    if "has_topp_only_rows" in parameters:
        return []  # the pin already takes it; nothing to adapt

    @functools.wraps(original)
    def apply_top_k_top_p(*args, has_topp_only_rows=None, **kwargs):
        # Whoever is bound takes the full top-p path, which is what the hint's
        # True value selects and is equivalent to its False value.
        del has_topp_only_rows
        return original(*args, **kwargs)

    setattr(apply_top_k_top_p, ADAPTED, True)
    # Assign before guarding so this first patch does not re-enter __setattr__.
    module.apply_top_k_top_p = apply_top_k_top_p
    if type(module) is not _ReapplyingModule:
        module.__class__ = _ReapplyingModule
    return [NAME]
