# SPDX-License-Identifier: Apache-2.0
"""Call-time drift: adapt a pinned vLLM function that Omni calls with newer arguments.

The import shim in :mod:`vllm_musa.omni_compat._finder` handles modules and
symbols the pin does not have. A harder class remains: the symbol **is** there,
Omni imports it successfully, and then calls it with an argument the pinned
signature cannot take. ``apply_top_k_top_p(logits, k, p, has_topp_only_rows=...)``
is the worked example -- vLLM 0.33 added that parameter, Omni's Qwen3-Omni talker
sampler passes it by keyword, and the pinned three-parameter function raises
``TypeError`` the first time a talker samples. No import-level check sees it.

Timing is the whole problem. Omni binds the function into its own module
namespace at import time (``from vllm... import apply_top_k_top_p``), so a patch
applied afterwards is invisible to it. The adapter therefore has to be in place
before Omni's modules are imported, which is also too early to import vLLM.

So this finder sits **first** in ``sys.meta_path`` for a short, explicit list of
module names, immediately delegates to the ordinary finder for the real module,
and wraps its loader to run the adapter right after the module executes. Every
other import is untouched: the hot path is one set lookup.
"""

from __future__ import annotations

import importlib
import logging
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder
from types import ModuleType

from vllm_musa.omni_compat.table import ShimEntry, adapter_entries

logger = logging.getLogger(__name__)

ADAPTER_PACKAGE = "vllm_musa.omni_compat.adapters"


class _PatchingLoader(Loader):
    """Runs the real loader, then hands the module to the adapter."""

    def __init__(self, inner: Loader, entry: ShimEntry) -> None:
        self.inner = inner
        self.entry = entry

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        return self.inner.create_module(spec)  # type: ignore[attr-defined]

    def exec_module(self, module: ModuleType) -> None:
        self.inner.exec_module(module)  # type: ignore[attr-defined]
        entry = self.entry
        adapter = importlib.import_module(f"{ADAPTER_PACKAGE}.{entry.adapter}")
        applied = adapter.apply(module)
        # A patch that silently does nothing is the failure mode that matters:
        # the run continues and breaks later, in the model, with no mention of
        # the shim. Fail here instead.
        if not applied:
            raise ImportError(
                f"vllm_musa.omni_compat adapter {entry.adapter} patched nothing in "
                f"{entry.module}. The entry is stale -- {entry.remove_when}"
            )
        module.__vllm_musa_adapted__ = {
            "names": list(applied),
            "reason": entry.reason,
            "remove_when": entry.remove_when,
        }
        logger.debug("omni_compat: adapted %s in %s", applied, entry.module)

    def __getattr__(self, name: str):  # pragma: no cover - loader protocol passthrough
        return getattr(self.inner, name)


class OmniAdapterFinder(MetaPathFinder):
    """First in ``sys.meta_path``, but only for the adapter entries' modules."""

    def __init__(self) -> None:
        from vllm_musa.omni_compat._finder import (
            installed_vllm_version,
            version_matches,
        )

        self._entries = {
            entry.module: entry
            for entry in adapter_entries()
            if version_matches(entry.applies_to, installed_vllm_version())
        }
        self._in_progress: set[str] = set()

    @property
    def modules(self) -> frozenset[str]:
        return frozenset(self._entries)

    def find_spec(self, fullname: str, path=None, target=None) -> ModuleSpec | None:
        entry = self._entries.get(fullname)
        if entry is None or fullname in self._in_progress:
            return None
        # Re-entrancy guard: PathFinder may import the parent package, which can
        # come back through us for the same name.
        self._in_progress.add(fullname)
        try:
            spec = PathFinder.find_spec(fullname, path, target)
        finally:
            self._in_progress.discard(fullname)
        if spec is None or spec.loader is None:
            return None  # let the ordinary machinery report the real failure
        spec.loader = _PatchingLoader(spec.loader, entry)
        return spec

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<OmniAdapterFinder modules={sorted(self._entries)}>"


def install_adapters() -> bool:
    """Put the adapter finder first in ``sys.meta_path``. Idempotent.

    Returns whether an adapter finder is active. A module that was already
    imported cannot be hooked, so this warns rather than pretending.
    """
    for finder in sys.meta_path:
        if isinstance(finder, OmniAdapterFinder):
            return True
    finder = OmniAdapterFinder()
    if not finder.modules:
        return False
    already = sorted(name for name in finder.modules if name in sys.modules)
    if already:
        logger.warning(
            "omni_compat: %s already imported, its call-time adapter will not apply; "
            "install the shim earlier (the .pth does this at interpreter startup)",
            ", ".join(already),
        )
    sys.meta_path.insert(0, finder)
    return True


def uninstall_adapters() -> int:
    before = len(sys.meta_path)
    sys.meta_path[:] = [
        f for f in sys.meta_path if not isinstance(f, OmniAdapterFinder)
    ]
    return before - len(sys.meta_path)
