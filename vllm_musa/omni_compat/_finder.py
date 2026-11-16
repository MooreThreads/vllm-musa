# SPDX-License-Identifier: Apache-2.0
"""The import hook that serves the drift table.

The finder is **appended** to ``sys.meta_path``, never inserted at the front, so
the real vLLM always wins: Python consults it only after every ordinary finder
has failed to locate the module. A pin that already provides the module is
therefore untouched, and the shim needs no "does this exist?" probe that could
recurse into itself.

Resolution is lazy in two ways. Installing imports nothing but the standard
library, so the hook is cheap enough to run from a ``.pth`` at interpreter
startup. The vLLM version is read only when a shimmed name is actually
requested, at which point ``vllm`` is necessarily already in ``sys.modules``
because its parent package had to be imported to get here.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType

from vllm_musa.omni_compat.table import (
    RELOCATION,
    ShimEntry,
    entries_for_module,
)

logger = logging.getLogger(__name__)

ENABLE_ENV = "VLLM_MUSA_OMNI_COMPAT"
"""Set to ``0`` to make the shim inert without uninstalling it."""

VERSION_ENV = "VLLM_VERSION"
"""Override the detected vLLM version, for editable or dev installs.

The name matches the convention other vLLM plugins use for the same purpose.
vLLM itself reads it at build time for setuptools-scm, which does not collide:
that is a different process from the one serving requests.
"""

BACKPORT_PACKAGE = "vllm_musa.omni_compat.backports"


def is_enabled() -> bool:
    return os.environ.get(ENABLE_ENV, "1") not in ("0", "false", "False")


def installed_vllm_version() -> str | None:
    """The installed vLLM version, or ``None`` when it cannot be determined.

    Reads ``sys.modules`` rather than importing: the finder runs *during* a vLLM
    import, and importing vllm again from here would deadlock on the partially
    initialised module.
    """
    override = os.environ.get(VERSION_ENV)
    if override:
        return override
    vllm = sys.modules.get("vllm")
    version = getattr(vllm, "__version__", None)
    return version if isinstance(version, str) else None


def version_matches(specifier: str, version: str | None) -> bool:
    """Whether ``version`` satisfies ``specifier``.

    An undeterminable version matches, so a dev build of vLLM still gets the
    shim; that is the safer default, because the alternative is an ImportError
    on a tree the operator deliberately built.
    """
    if version is None:
        return True
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import InvalidVersion, Version
    except ImportError:  # pragma: no cover - packaging ships with vLLM
        return True
    try:
        parsed = Version(version)
    except InvalidVersion:
        logger.debug(
            "omni_compat: unparsable vllm version %r, applying the shim", version
        )
        return True
    # Compare on the public segment so a local build tag (0.32.0+musa) keeps the
    # version identity of its release.
    return Version(parsed.public) in SpecifierSet(specifier)


class _ShimLoader(Loader):
    """Builds the module an entry describes."""

    def __init__(self, entry: ShimEntry) -> None:
        self.entry = entry

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        return None  # default module creation

    def exec_module(self, module: ModuleType) -> None:
        entry = self.entry
        if entry.kind == RELOCATION:
            self._exec_relocation(module, entry)
        else:
            self._exec_backport(module, entry)
        module.__vllm_musa_shim__ = {
            "kind": entry.kind,
            "provides": entry.names,
            "reason": entry.reason,
            "remove_when": entry.remove_when,
            "source": entry.source or f"{BACKPORT_PACKAGE}.{entry.backport}",
        }
        logger.debug("omni_compat: served %s as %s", entry.module, entry.kind)

    @staticmethod
    def _exec_relocation(module: ModuleType, entry: ShimEntry) -> None:
        source = importlib.import_module(entry.source)  # type: ignore[arg-type]
        missing = [name for name in entry.names if not hasattr(source, name)]
        if missing:
            # A wrong table row must fail loudly here rather than produce a
            # module that is silently missing what the caller asked for.
            raise ImportError(
                f"vllm_musa.omni_compat cannot serve {entry.module}: "
                f"{entry.source} has no {', '.join(missing)}. "
                f"The table entry is stale -- {entry.remove_when}"
            )
        for name in entry.names:
            setattr(module, name, getattr(source, name))
        # Delegate anything the table does not list, so an Omni file that
        # imports a name we did not record still works.
        module.__getattr__ = lambda name, _source=source: getattr(_source, name)  # type: ignore[attr-defined]
        module.__all__ = list(entry.names)

    @staticmethod
    def _exec_backport(module: ModuleType, entry: ShimEntry) -> None:
        backport = importlib.import_module(f"{BACKPORT_PACKAGE}.{entry.backport}")
        missing = [name for name in entry.names if not hasattr(backport, name)]
        if missing:
            raise ImportError(
                f"vllm_musa.omni_compat backport {entry.backport} does not define "
                f"{', '.join(missing)}, which {entry.module} must provide"
            )
        for name in entry.names:
            setattr(module, name, getattr(backport, name))
        module.__getattr__ = lambda name, _source=backport: getattr(_source, name)  # type: ignore[attr-defined]
        module.__all__ = list(entry.names)


class OmniCompatFinder(MetaPathFinder):
    """Serves the drift table's modules, last in ``sys.meta_path``."""

    def find_spec(self, fullname: str, path=None, target=None) -> ModuleSpec | None:
        entry = entries_for_module(fullname)
        if entry is None:
            return None
        if not is_enabled():
            return None
        version = installed_vllm_version()
        if not version_matches(entry.applies_to, version):
            # Outside the entry's version range the pin is expected to provide
            # the module itself; letting the import fail reports the real error.
            logger.debug(
                "omni_compat: %s not shimmed, vllm %s is outside %s",
                fullname,
                version,
                entry.applies_to,
            )
            return None
        return importlib.util.spec_from_loader(fullname, _ShimLoader(entry))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<OmniCompatFinder>"
