# SPDX-License-Identifier: Apache-2.0
"""Interpreter-startup entry point for the vLLM-Omni drift shim.

Installed by ``python -m vllm_musa.omni_compat install-pth``, which writes a
``.pth`` whose only content is ``import vllm_musa_omni_compat_pth``. Site
processes that line in **every** interpreter on the machine, so this module is
standard-library only and never imports the shim package itself. Importing
:mod:`vllm_musa` runs its package ``__init__``, which pulls in ``torchada``,
``torch`` and parts of vLLM. Two reasons that is not acceptable here:

* cost -- a health probe or a ``pip`` subprocess would pay seconds of torch
  import for a hook it never uses;
* correctness -- the hook fires *during* a vLLM import, and re-entering vLLM
  through ``vllm_musa``'s initialisation at that moment raises
  ``cannot import name 'rocm_aiter_ops' from partially initialized module
  'vllm._aiter_ops'``, which breaks every import of the adapted modules.

So everything is loaded by file path with :mod:`importlib.util`, from the
installed ``vllm_musa/omni_compat/`` directory located without executing the
package. ``table.py``, the backports and the adapters are all written to be
dependency-free at module level for exactly this reason, and the adapters import
vLLM only inside the functions they install.

A top-level module rather than a submodule for the same reason: there is no way
to reach ``vllm_musa.omni_compat`` without executing ``vllm_musa``. It is
declared in ``pyproject.toml``'s ``py-modules`` so both wheel and editable
installs expose it.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys

_ENABLE_ENV = "VLLM_MUSA_OMNI_COMPAT"
_PACKAGE_RELPATH = ("omni_compat",)


def _disabled() -> bool:
    return os.environ.get(_ENABLE_ENV, "1") in ("0", "false", "False")


def _package_dir() -> str | None:
    """The installed ``vllm_musa/omni_compat`` directory, without importing it."""
    try:
        spec = importlib.util.find_spec("vllm_musa")  # does not execute __init__
    except (ImportError, ValueError):  # pragma: no cover - broken install
        return None
    locations = (
        list(getattr(spec, "submodule_search_locations", None) or []) if spec else []
    )
    if not locations:
        return None
    path = os.path.join(locations[0], *_PACKAGE_RELPATH)
    return path if os.path.isdir(path) else None


def _load_by_path(name: str, path: str):
    """Execute a module from a file without importing its package."""
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    # Registering before exec is required, not optional: table.py uses
    # ``from __future__ import annotations``, and dataclasses resolves those
    # string annotations through ``sys.modules[cls.__module__]``.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # pragma: no cover - a broken file must not break python
        sys.modules.pop(name, None)
        return None
    return module


def _version_matches(specifier: str) -> bool:
    """Whether the installed vLLM satisfies ``specifier``.

    Read from ``sys.modules`` rather than imported: the hook only fires for
    ``vllm.*`` submodules, so ``vllm`` is necessarily imported by then, and
    importing it again from inside its own import would deadlock.
    """
    version = os.environ.get("VLLM_VERSION")
    if not version:
        version = getattr(sys.modules.get("vllm"), "__version__", None)
    if not isinstance(version, str):
        return True  # undeterminable: shim rather than a surprise ImportError
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import InvalidVersion, Version
    except ImportError:  # pragma: no cover - packaging ships with pip
        return True
    try:
        parsed = Version(version)
    except InvalidVersion:
        return True
    # Compare on the public segment so a local build tag (0.32.0+musa) keeps the
    # version identity of its release.
    return Version(parsed.public) in SpecifierSet(specifier)


class _ServingLoader:
    """Provides a module the pin does not have, per the table's entry."""

    def __init__(self, entry, package_dir: str) -> None:
        self.entry = entry
        self.package_dir = package_dir

    def create_module(self, spec):
        return None  # default module creation

    def exec_module(self, module) -> None:
        import importlib

        entry = self.entry
        if entry.source:
            origin = importlib.import_module(entry.source)
            where = entry.source
        else:
            where = f"omni_compat/backports/{entry.backport}.py"
            origin = _load_by_path(
                f"_vllm_musa_omni_backport_{entry.backport}",
                os.path.join(self.package_dir, "backports", f"{entry.backport}.py"),
            )
            if origin is None:
                raise ImportError(
                    f"vllm_musa.omni_compat cannot load {where} for {entry.module}"
                )
        missing = [name for name in entry.names if not hasattr(origin, name)]
        if missing:
            raise ImportError(
                f"vllm_musa.omni_compat cannot serve {entry.module}: {where} has no "
                f"{', '.join(missing)}. The table entry is stale -- {entry.remove_when}"
            )
        for name in entry.names:
            setattr(module, name, getattr(origin, name))
        # Delegate anything the table does not list, so an Omni file that
        # imports a name we did not record still works.
        module.__getattr__ = lambda name, _origin=origin: getattr(_origin, name)
        module.__all__ = list(entry.names)
        module.__vllm_musa_shim__ = {
            "kind": entry.kind,
            "provides": list(entry.names),
            "source": where,
            "reason": entry.reason,
            "remove_when": entry.remove_when,
        }


class _AdaptingLoader:
    """Runs the pin's own loader, then applies the entry's adapter."""

    def __init__(self, inner, entry, package_dir: str) -> None:
        self.inner = inner
        self.entry = entry
        self.package_dir = package_dir

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module) -> None:
        self.inner.exec_module(module)
        entry = self.entry
        adapter = _load_by_path(
            f"_vllm_musa_omni_adapter_{entry.adapter}",
            os.path.join(self.package_dir, "adapters", f"{entry.adapter}.py"),
        )
        if adapter is None:
            raise ImportError(
                f"vllm_musa.omni_compat cannot load adapters/{entry.adapter}.py "
                f"for {entry.module}"
            )
        applied = adapter.apply(module)
        # "Patched nothing" is two different states, and only one of them is a
        # failure. An adapter that matched nothing has left the module unable
        # to serve Omni's call and nothing downstream would mention the shim,
        # so fail the import. An adapter with nothing left to do -- the pin
        # grew the shape itself -- is a stale table entry: a repo bug, not a
        # runtime one. Raising there would take every model on the image down
        # over a shim that is no longer needed, so record it and let the build
        # gate (docker/omni/verify_shim.py) refuse the image.
        satisfied = getattr(adapter, "is_satisfied", None)
        if not applied and not (satisfied is not None and satisfied(module)):
            raise ImportError(
                f"vllm_musa.omni_compat adapter {entry.adapter} patched nothing in "
                f"{entry.module}. The entry is stale -- {entry.remove_when}"
            )
        module.__vllm_musa_adapted__ = {
            "names": list(applied),
            "reason": entry.reason,
            "remove_when": entry.remove_when,
            "stale": not applied,
        }

    def __getattr__(self, name):  # pragma: no cover - loader protocol passthrough
        return getattr(self.inner, name)


class OmniCompatPathFinder:
    """Serves and adapts the table's modules, loading nothing eagerly.

    ``role`` decides both the behaviour and the required position in
    ``sys.meta_path``: ``"serve"`` provides modules the pin lacks and must be
    **last**, so the real vLLM always wins and a newer pin needs no code change;
    ``"adapt"`` patches modules the pin does have and must be **first**, because
    an ordinary finder would otherwise satisfy the import before the hook is
    consulted.
    """

    def __init__(self, entries: dict, role: str, package_dir: str) -> None:
        self._entries = entries
        self._role = role
        self._package_dir = package_dir
        self._in_progress: set = set()

    @property
    def role(self) -> str:
        return self._role

    @property
    def modules(self) -> frozenset:
        return frozenset(self._entries)

    def find_spec(self, fullname: str, path=None, target=None):
        # Hot path for every import in the process: one dict lookup.
        entry = self._entries.get(fullname)
        if entry is None or fullname in self._in_progress or _disabled():
            return None
        if not _version_matches(entry.applies_to):
            # Outside the entry's range the pin is expected to provide the
            # module itself; letting the import proceed reports the real state.
            return None
        if self._role == "serve":
            return importlib.util.spec_from_loader(
                fullname, _ServingLoader(entry, self._package_dir)
            )
        # Re-entrancy guard: PathFinder may import the parent package, which can
        # come back through us for the same name.
        self._in_progress.add(fullname)
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        finally:
            self._in_progress.discard(fullname)
        if spec is None or spec.loader is None:
            return None  # let the ordinary machinery report the real failure
        spec.loader = _AdaptingLoader(spec.loader, entry, self._package_dir)
        return spec

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<OmniCompatPathFinder {self._role} {sorted(self._entries)}>"


def install() -> bool:
    """Install both hooks. Idempotent; returns whether any is active."""
    if _disabled():
        return False
    if any(isinstance(finder, OmniCompatPathFinder) for finder in sys.meta_path):
        return True
    package_dir = _package_dir()
    if package_dir is None:
        return False
    table = _load_by_path(
        "_vllm_musa_omni_table", os.path.join(package_dir, "table.py")
    )
    if table is None:
        return False
    served = {entry.module: entry for entry in table.importable_entries()}
    adapted = {entry.module: entry for entry in table.adapter_entries()}
    if adapted:
        sys.meta_path.insert(0, OmniCompatPathFinder(adapted, "adapt", package_dir))
    if served:
        sys.meta_path.append(OmniCompatPathFinder(served, "serve", package_dir))
    return bool(served or adapted)


install()
