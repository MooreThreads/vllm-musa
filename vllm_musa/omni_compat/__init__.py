# SPDX-License-Identifier: Apache-2.0
"""Run a newer vLLM-Omni tree on vLLM-MUSA's pinned vLLM.

vLLM-MUSA pins one immutable upstream vLLM commit per release branch
(``third_party/PINS``) and validates it on MUSA. vLLM-Omni declares no ``vllm``
requirement, tracks vLLM main, and imports vLLM internals directly, so an Omni
tree that is newer than the pin fails at import time. This package absorbs that
gap for the drift recorded in :mod:`vllm_musa.omni_compat.table`.

Not installed by ``import vllm_musa``. A shim that rewrites the meaning of
``import vllm.*`` is scoped to images that actually run Omni, so activation is
explicit:

* ``python -m vllm_musa.omni_compat install-pth`` writes a ``.pth`` into
  site-packages, which installs the hook at interpreter startup. This is what
  ``docker/omni.Dockerfile`` does, and it is the only mechanism early enough:
  ``vllm_omni.entrypoints.cli.main`` imports moved vLLM modules before vLLM gets
  to load any plugin entry point.
* :func:`install` for a process that can call it before importing Omni.

``VLLM_MUSA_OMNI_COMPAT=0`` makes an installed hook inert.
"""

from __future__ import annotations

import sys
import sysconfig
from pathlib import Path

from vllm_musa.omni_compat._finder import (
    ENABLE_ENV,
    VERSION_ENV,
    OmniCompatFinder,
    installed_vllm_version,
    is_enabled,
    version_matches,
)
from vllm_musa.omni_compat._patcher import install_adapters, uninstall_adapters
from vllm_musa.omni_compat.table import ENTRIES, capability_entries, importable_entries

__all__ = [
    "ENABLE_ENV",
    "VERSION_ENV",
    "active_hooks",
    "install",
    "install_adapters",
    "installed_vllm_version",
    "is_enabled",
    "is_installed",
    "pth_path",
    "status",
    "uninstall",
    "unsupported_paths",
    "version_matches",
    "write_pth",
]

PTH_NAME = "zzz-vllm-musa-omni-compat.pth"
"""``.pth`` files are processed in sorted order; the ``zzz`` prefix keeps this
one after anything that might add site directories the shim could need."""

_PTH_LINE = "import vllm_musa_omni_compat_pth\n"
"""The hook goes through the top-level bootstrap, never through this package:
importing :mod:`vllm_musa` runs its ``__init__``, which imports torchada and
torch, and a ``.pth`` runs in every interpreter on the machine."""


def install() -> bool:
    """Install both hooks. Idempotent; returns whether any is active.

    The import hook is **appended**, so the real vLLM always wins and a pin that
    already provides a module is untouched. The adapter hook is **inserted
    first**, because a pinned module that loads normally would otherwise never
    reach its patch. Prefer the ``.pth`` for anything that imports vLLM-Omni:
    Omni binds the pinned functions into its own namespace at import time, and a
    patch applied after that is invisible to it.
    """
    if not is_enabled():
        return False
    if is_installed():
        return True
    install_adapters()
    sys.meta_path.append(OmniCompatFinder())
    return True


def uninstall() -> int:
    """Remove every installed hook. Returns how many were removed."""
    before = len(sys.meta_path)
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, OmniCompatFinder)]
    return (before - len(sys.meta_path)) + uninstall_adapters()


def active_hooks() -> dict[str, bool]:
    """Which hook mechanisms are live in this interpreter.

    Two installers exist on purpose. :func:`install` adds the hooks from this
    package, for a process that can call it before importing Omni. The ``.pth``
    installs ``vllm_musa_omni_compat_pth``'s own hooks at interpreter startup,
    which is the only point early enough for vLLM-Omni's CLI and which must not
    import this package at all. Either is a working shim, so both count as
    installed; ``serve`` and ``adapt`` say which halves are live.
    """
    hooks = {"serve": False, "adapt": False, "startup": False}
    for finder in sys.meta_path:
        name = type(finder).__name__
        if name == "OmniCompatFinder":
            hooks["serve"] = True
        elif name == "OmniAdapterFinder":
            hooks["adapt"] = True
        elif name == "OmniCompatPathFinder":
            # The startup hook from the .pth bootstrap, which carries its own
            # role because it implements both halves without this package.
            hooks["startup"] = True
            role = getattr(finder, "role", None)
            if role in hooks:
                hooks[role] = True
    return hooks


def is_installed() -> bool:
    return any(active_hooks().values())


def pth_path(purelib: str | Path | None = None) -> Path:
    """Where :func:`write_pth` writes, by default this interpreter's site-packages."""
    base = (
        Path(purelib) if purelib is not None else Path(sysconfig.get_paths()["purelib"])
    )
    return base / PTH_NAME


def write_pth(purelib: str | Path | None = None) -> Path:
    """Install the startup hook for this interpreter and return the path written."""
    path = pth_path(purelib)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_PTH_LINE)
    return path


def unsupported_paths() -> tuple[str, ...]:
    """Feature paths the pin cannot serve for the recorded Omni tree.

    These are the ``CAPABILITY`` entries: nothing is missing at import time, but
    a pinned API means something different under the same name, so the Omni code
    path that uses it is wrong rather than absent. An import shim cannot fix
    them; a caller that needs one of these needs a newer pin.
    """
    version = installed_vllm_version()
    paths: list[str] = []
    for entry in capability_entries():
        if version_matches(entry.applies_to, version):
            paths.extend(entry.unsupported)
    return tuple(paths)


def status() -> dict:
    """Machine-readable state, for the build gate and ``vllm_collect_env``."""
    version = installed_vllm_version()
    entries = []
    for entry in ENTRIES:
        entries.append(
            {
                "module": entry.module,
                "kind": entry.kind,
                "names": list(entry.names),
                "applies_to": entry.applies_to,
                "active": version_matches(entry.applies_to, version),
                "served": entry.module in sys.modules,
                "source": entry.source,
                "backport": entry.backport,
                "target": entry.target,
                "unsupported": list(entry.unsupported),
                "reason": entry.reason,
                "remove_when": entry.remove_when,
                "omni_refs": list(entry.omni_refs),
            }
        )
    return {
        "enabled": is_enabled(),
        "installed": is_installed(),
        "hooks": active_hooks(),
        "vllm_version": version,
        "pth": str(pth_path()),
        "pth_present": pth_path().exists(),
        "entries": entries,
        "importable": len(importable_entries()),
        "unsupported_paths": list(unsupported_paths()),
    }
