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


def _is_the_bridge(obj: object) -> bool:
    """Whether ``obj`` is the wrapper this adapter installed.

    The marker is the function itself, so ``functools.wraps`` copying it out of
    our ``__dict__`` into somebody else's wrapper cannot forge the answer --
    that wrapper's marker names ours, not itself. Identity is the right test
    here because the hazard is *replacement*: vllm-musa binds its own
    ``apply_top_k_top_p``, and one that copied our metadata with
    ``functools.wraps`` would otherwise read as adapted while not taking the
    keyword at all, so the guard below would skip re-wrapping it.

    Local rather than shared: the ``.pth`` bootstrap loads each adapter by file
    path, so an adapter must not import a sibling module.
    """
    return getattr(obj, ADAPTED, None) is obj


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
    arrives here as its own marker, so :func:`apply` returns without assigning
    again.
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
    return [NAME] if _is_the_bridge(getattr(module, NAME, None)) else []


def is_satisfied(module: ModuleType) -> bool:
    """Whether the live callable already accepts the newer hint.

    True in two unrelated ways -- the adapter is in place, or the pin caught up
    and declares the parameter itself -- and the loader has to tell both from
    "the adapter matched nothing", which is the only one worth failing an
    import over.
    """
    live = getattr(module, NAME, None)
    if live is None:
        return False
    if _is_the_bridge(live):
        return True
    try:
        # follow_wrapped=False: the wrapper this adapter installs carries
        # __wrapped__, so a followed signature reports the pin's parameters and
        # answers a different question than the one asked here.
        parameters = inspect.signature(live, follow_wrapped=False).parameters
    except (TypeError, ValueError):  # pragma: no cover - C callables
        return False
    if "has_topp_only_rows" in parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def apply(module: ModuleType) -> list[str]:
    """Let the pinned ``apply_top_k_top_p`` accept and ignore the newer hint."""
    original = getattr(module, NAME, None)
    if original is None or _is_the_bridge(original):
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

    setattr(apply_top_k_top_p, ADAPTED, apply_top_k_top_p)
    # Assign before guarding so this first patch does not re-enter __setattr__.
    module.apply_top_k_top_p = apply_top_k_top_p
    if type(module) is not _ReapplyingModule:
        module.__class__ = _ReapplyingModule
    return [NAME]
