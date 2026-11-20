#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Find where a vLLM-Omni tree disagrees with a pinned vLLM, before a user does.

vLLM-Omni imports vLLM internals and tracks vLLM main; vLLM-MUSA pins one vLLM
commit. This tool answers two questions about a candidate Omni commit:

``imports``
    Which vLLM symbols does Omni import that the pin does not have? Each is
    reported as **shimmed** when :mod:`vllm_musa.omni_compat` covers it or
    **unshimmed** when nothing does. Only unshimmed misses fail.

``attrs``
    Which attributes does Omni read off a vLLM module that the pin's module does
    not have? The import resolves, so an import-only check sees nothing.

``staleness``
    Is every entry in the shim table still needed on this pin, or has the pin
    moved past one?

``semantics``
    Which vLLM functions that Omni calls kept their name and arity but changed
    their **signature** between the pin and the vLLM release Omni targets? These
    import cleanly and then misbehave, so no import-level check can find them.
    Needs ``--target-vllm`` to compare against; reported as review items.

What none of them see yet: constructor keywords, return arity, and names hidden
behind a ``**kwargs`` splat. ``docs/omni_compat.md`` records the live drift that
slips through.

All four passes are pure ``ast`` over source trees: no MUSA driver, no torch, no
install step, so this runs in CI on a candidate Omni commit in seconds.

    # against the installed vLLM, in an image
    python3 tools/check_omni_compat.py --omni-root /vllm-workspace/vllm-omni

    # against extracted sources, with the semantic diff
    python3 tools/check_omni_compat.py \\
        --omni-root ./vllm-omni \\
        --vllm-root ./vllm-0.32/vllm --vllm-version 0.32.0 \\
        --target-vllm ./vllm-0.33/vllm \\
        --report report.json

Exit code 0 means the pin can serve this Omni tree's imports; 1 means it cannot.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 3
SKIP_DIRS = {"__pycache__", "build", "dist", ".git", ".venv", "tests"}
REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Source indexing
# --------------------------------------------------------------------------- #
def iter_python_files(root: Path):
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        yield path


def parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(
            path.read_text(encoding="utf-8", errors="replace"), filename=str(path)
        )
    except SyntaxError:
        return None


@dataclass
class VllmIndex:
    """What a vLLM source tree exports, by module, without importing it.

    Two vLLM idioms decide whether this index is usable or merely noisy, because
    both make a name available without a binding statement in the module:

    * ``from .parallel_state import *`` -- ``vllm/distributed/__init__.py`` is
      nothing but star re-exports, so every ``from vllm.distributed import x``
      looks missing until the stars are resolved.
    * a module-level ``def __getattr__`` -- ``vllm/platforms/__init__.py``
      resolves ``current_platform`` lazily that way, and no static index can
      enumerate what such a module provides.
    """

    root: Path
    version: str | None = None
    modules: dict[str, set[str]] = field(default_factory=dict)
    signatures: dict[str, dict] = field(default_factory=dict)
    star_sources: dict[str, list[str]] = field(default_factory=dict)
    dynamic: set[str] = field(default_factory=set)

    @classmethod
    def build(cls, root: Path, version: str | None = None) -> VllmIndex:
        index = cls(root=root, version=version)
        for path in iter_python_files(root):
            rel = path.relative_to(root)
            parts = list(rel.parts)
            if parts[-1] == "__init__.py":
                parts.pop()
            else:
                parts[-1] = parts[-1][: -len(".py")]
            module = ".".join(["vllm", *parts]) if parts else "vllm"
            tree = parse(path)
            if tree is None:
                index.modules.setdefault(module, set())
                continue
            index.modules[module] = _module_exports(tree)
            index.star_sources[module] = _star_sources(module, tree)
            all_names = _declared_all(tree)
            if all_names is not None:
                index._declared_all[module] = all_names
            if _has_module_getattr(tree):
                index.dynamic.add(module)
            for qualname, signature in _module_signatures(module, tree).items():
                index.signatures[qualname] = signature
        index._resolve_star_imports()
        return index

    def _resolve_star_imports(self) -> None:
        """Pull star-imported names into the importing module, to a fixed point.

        ``__all__`` wins when the source declares it; otherwise every public
        top-level name is re-exported, which is what ``import *`` does.
        """
        for _ in range(10):  # star chains are shallow; bound the iteration
            changed = False
            for module, sources in self.star_sources.items():
                for source in sources:
                    if source == module:
                        continue
                    exported = self._star_exports(source)
                    before = len(self.modules.setdefault(module, set()))
                    self.modules[module] |= exported
                    if len(self.modules[module]) != before:
                        changed = True
                    if source in self.dynamic:
                        self.dynamic.add(module)
            if not changed:
                return

    def _star_exports(self, module: str) -> set[str]:
        names = self.modules.get(module)
        if names is None:
            return set()
        declared = self._declared_all.get(module)
        if declared is not None:
            return set(declared)
        return {name for name in names if not name.startswith("_")}

    _declared_all: dict[str, list[str]] = field(default_factory=dict)

    def has_module(self, module: str) -> bool:
        return module in self.modules

    def has_symbol(self, module: str, name: str) -> bool:
        if name in self.modules.get(module, ()):
            return True
        # A module that resolves attributes lazily cannot be enumerated, so
        # trusting it is the only sound choice; the alternative is 150 false
        # "missing" lines that train a reviewer to ignore the gate.
        return module in self.dynamic


def _module_exports(tree: ast.Module) -> set[str]:
    """Top-level names a module binds.

    Module-level control flow counts: vLLM guards bindings behind
    ``if TYPE_CHECKING:``, version checks and ``try``/``except ImportError``
    fallbacks -- ``vllm/version.py`` binds ``__version__`` from a generated
    module in a ``try`` and falls back to a literal in the ``except`` -- and a
    name bound in any of those branches is importable.
    """
    names: set[str] = set()
    _collect_bindings(tree.body, names)
    return names


_CONTROL_FLOW = (ast.If, ast.Try, ast.With, ast.For, ast.While)


def _collect_bindings(body, names: set[str]) -> None:
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    names.update(e.id for e in target.elts if isinstance(e, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    continue
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, _CONTROL_FLOW):
            for attr in ("body", "orelse", "finalbody"):
                _collect_bindings(getattr(node, attr, []) or [], names)
            for handler in getattr(node, "handlers", []) or []:
                _collect_bindings(handler.body, names)


def _star_sources(module: str, tree: ast.Module) -> list[str]:
    """Absolute module names this module does ``from X import *`` from."""
    sources: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if not any(alias.name == "*" for alias in node.names):
            continue
        if node.level:
            base = module.split(".")
            # level 1 is the current package; __init__ modules are the package.
            trim = node.level - 1
            base = base[: len(base) - trim] if trim else base
            sources.append(
                ".".join([*base, node.module]) if node.module else ".".join(base)
            )
        elif node.module:
            sources.append(node.module)
    return sources


def _has_module_getattr(tree: ast.Module) -> bool:
    """Whether the module defines PEP 562 ``__getattr__`` at the top level."""
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "__getattr__"
        for node in tree.body
    )


def _declared_all(tree: ast.Module) -> list[str] | None:
    """The module's ``__all__``, when it is a plain list/tuple of strings."""
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            continue
        value = node.value
        if isinstance(value, (ast.List, ast.Tuple)):
            names = [
                e.value
                for e in value.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
            if len(names) == len(value.elts):
                return names
    return None


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict:
    args = node.args
    positional = [a.arg for a in (*args.posonlyargs, *args.args)]
    return {
        "positional": positional,
        "annotations": [
            ast.unparse(a.annotation) if a.annotation else None
            for a in (*args.posonlyargs, *args.args)
        ],
        "defaults": len(args.defaults),
        "vararg": args.vararg.arg if args.vararg else None,
        "kwonly": [a.arg for a in args.kwonlyargs],
        "kwarg": args.kwarg.arg if args.kwarg else None,
        "returns": ast.unparse(node.returns) if node.returns else None,
        "lineno": node.lineno,
    }


def _module_signatures(module: str, tree: ast.Module) -> dict[str, dict]:
    """``module.func`` and ``module.Class.method`` signatures of a vLLM module."""
    out: dict[str, dict] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[f"{module}.{node.name}"] = _signature(node)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f"{module}.{node.name}.{sub.name}"] = _signature(sub)
    return out


# --------------------------------------------------------------------------- #
# What Omni asks of vLLM
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OmniImport:
    module: str
    name: str | None  # None for `import vllm.x`
    omni_file: str
    lineno: int
    type_checking: bool


def collect_omni_imports(omni_root: Path) -> list[OmniImport]:
    """Every ``vllm`` import in the Omni tree, with its source location."""
    found: list[OmniImport] = []
    package = omni_root / "vllm_omni"
    scan_root = package if package.is_dir() else omni_root
    for path in iter_python_files(scan_root):
        tree = parse(path)
        if tree is None:
            continue
        rel = (
            str(path.relative_to(omni_root))
            if omni_root in path.parents or path.parent == omni_root
            else str(path)
        )
        guarded = _type_checking_lines(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level or not node.module or not _is_vllm(node.module):
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    found.append(
                        OmniImport(
                            node.module,
                            alias.name,
                            rel,
                            node.lineno,
                            node.lineno in guarded,
                        )
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_vllm(alias.name):
                        found.append(
                            OmniImport(
                                alias.name,
                                None,
                                rel,
                                node.lineno,
                                node.lineno in guarded,
                            )
                        )
    return found


def collect_module_attribute_uses(omni_root: Path) -> list[dict]:
    """Attributes Omni reads off a vLLM **module** it imported as a module.

    The class the import audit alone cannot see. ``vllm_omni/utils/mm_prefix.py``
    does ``from vllm.v1.attention.backends import utils`` -- which resolves on
    any pin, because the module exists -- and then reads
    ``utils.compute_mm_prefix_ranges``, which 0.33 added. The import check
    reports nothing and ``import vllm_omni`` dies with an AttributeError.
    """
    package = omni_root / "vllm_omni"
    scan_root = package if package.is_dir() else omni_root
    uses: list[dict] = []
    for path in iter_python_files(scan_root):
        tree = parse(path)
        if tree is None:
            continue
        try:
            rel = str(path.relative_to(omni_root))
        except ValueError:  # pragma: no cover
            rel = path.name
        # Local name -> vLLM module it is bound to, for this file only.
        bound: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level or not node.module or not _is_vllm(node.module):
                    continue
                for alias in node.names:
                    if alias.name != "*":
                        bound[alias.asname or alias.name] = (
                            f"{node.module}.{alias.name}"
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if not _is_vllm(alias.name):
                        continue
                    if alias.asname:
                        bound[alias.asname] = alias.name
        if not bound:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or not isinstance(
                node.value, ast.Name
            ):
                continue
            target = bound.get(node.value.id)
            if target is None or node.attr.startswith("__"):
                continue
            uses.append(
                {
                    "module": target,
                    "attr": node.attr,
                    "omni_file": rel,
                    "lineno": node.lineno,
                }
            )
    return uses


def audit_module_attributes(uses: list[dict], index: VllmIndex, table) -> dict:
    """Whether each attribute read off a pinned vLLM module exists on this pin."""
    missing: list[dict] = []
    shimmed: list[dict] = []
    checked = 0
    for use in uses:
        module, attr = use["module"], use["attr"]
        # `bound` may name a symbol rather than a module (from X import f, then
        # f.attr); only module-valued bindings are in scope here.
        if not index.has_module(module):
            continue
        checked += 1
        if index.has_symbol(module, attr):
            continue
        cover = shim_covers(table, index, module, attr)
        if cover is None:
            for entry in table.adapter_entries():
                if entry.module == module and attr in entry.names:
                    cover = {"kind": entry.kind, "adapter": entry.adapter}
                    break
        (shimmed if cover else missing).append({**use, "shim": cover})
    return {
        "status": "pass" if not missing else "fail",
        "checked": checked,
        "shimmed": shimmed,
        "missing": missing,
    }


def _is_vllm(module: str) -> bool:
    return module == "vllm" or module.startswith("vllm.")


def _type_checking_lines(tree: ast.Module) -> set[int]:
    """Line numbers inside ``if TYPE_CHECKING:`` blocks."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        is_guard = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )
        if is_guard:
            for sub in ast.walk(node):
                if hasattr(sub, "lineno"):
                    lines.add(sub.lineno)
    return lines


def _typeerror_guarded_lines(tree: ast.Module) -> set[int]:
    """Lines whose ``TypeError`` is already handled by an enclosing ``try``.

    Omni supports several vLLM versions in one source tree by calling the new
    signature and catching the ``TypeError``::

        try:
            return get_rope(**new_kwargs)
        except TypeError:
            return get_rope(rotary_dim=..., base=...)   # older vLLM

    Both calls are deliberate, and the second one not fitting the pin is the
    point of writing it, so neither is a break. Without this the audit reports
    a version-compatibility idiom as a defect.

    The handler has to name ``TypeError``. A broad ``except Exception`` means
    the call's failure becomes an error response, not that an older signature
    is supported, so it does not count.
    """
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        # Only an explicit ``except TypeError``. A broad ``except Exception``
        # is error handling, not a supported fallback -- Omni's streaming chat
        # handler wraps 840 lines in one, which would absolve every call in it.
        handlers = []
        for handler in node.handlers:
            caught = handler.type
            if isinstance(caught, ast.Name):
                names = {caught.id}
            elif isinstance(caught, ast.Tuple):
                names = {e.id for e in caught.elts if isinstance(e, ast.Name)}
            else:
                continue
            if "TypeError" in names:
                handlers.append(handler)
        if not handlers:
            continue
        for statement in [*node.body, *(s for h in handlers for s in h.body)]:
            end = getattr(statement, "end_lineno", statement.lineno)
            guarded.update(range(statement.lineno, end + 1))
    return guarded


def collect_omni_calls(omni_root: Path, interesting: set[str]) -> dict[str, list[dict]]:
    """Where Omni calls any of ``interesting``, by plain name or as an attribute.

    An attribute call cannot be resolved to a type by AST alone, so a hit is a
    locator plus the arguments actually passed, not a proof of the callee. The
    arguments are what matters here: a vLLM function that gained a parameter
    only breaks on the pin if Omni passes it.
    """
    package = omni_root / "vllm_omni"
    scan_root = package if package.is_dir() else omni_root
    hits: dict[str, list[dict]] = {}
    for path in iter_python_files(scan_root):
        tree = parse(path)
        if tree is None:
            continue
        try:
            rel = str(path.relative_to(omni_root))
        except ValueError:  # pragma: no cover - scan_root outside omni_root
            rel = path.name
        guarded = _typeerror_guarded_lines(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
                # `self.f(...)` and `obj.f(...)` bind the receiver, which is not
                # a declared parameter at the call site.
                positional = len(node.args) + 1
            elif isinstance(node.func, ast.Name):
                name = node.func.id
                positional = len(node.args)
            else:
                continue
            if name not in interesting:
                continue
            hits.setdefault(name, []).append(
                {
                    "attribute": isinstance(node.func, ast.Attribute),
                    "guarded": node.lineno in guarded,
                    "omni_file": rel,
                    "lineno": node.lineno,
                    "positional": positional,
                    "keywords": sorted(kw.arg for kw in node.keywords if kw.arg),
                    "starargs": any(isinstance(a, ast.Starred) for a in node.args)
                    or any(kw.arg is None for kw in node.keywords),
                    "call": ast.unparse(node)[:160],
                }
            )
    return hits


# --------------------------------------------------------------------------- #
# The shim's view
# --------------------------------------------------------------------------- #
def table_path() -> Path:
    """The drift table, from this checkout or from an installed vllm-musa.

    Both matter: CI runs the tool from the repo against extracted vLLM sources,
    and the image build runs it inside a container where vllm-musa is installed
    and this repository is not present.
    """
    import importlib.util

    local = REPO_ROOT / "vllm_musa" / "omni_compat" / "table.py"
    if local.exists():
        return local
    spec = importlib.util.find_spec("vllm_musa")  # does not execute __init__
    locations = (
        list(getattr(spec, "submodule_search_locations", None) or []) if spec else []
    )
    for location in locations:
        installed = Path(location) / "omni_compat" / "table.py"
        if installed.exists():
            return installed
    raise RuntimeError(
        "cannot locate vllm_musa/omni_compat/table.py in this checkout or install"
    )


def load_shim_table():
    """The drift table, without importing ``vllm_musa`` (which needs torch)."""
    import importlib.util

    path = table_path()
    spec = importlib.util.spec_from_file_location("_omni_compat_table", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot load the drift table from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses needs it for string annotations
    spec.loader.exec_module(module)
    return module


def shim_covers(table, index: VllmIndex, module: str, name: str | None) -> dict | None:
    """The entry covering ``module.name`` on the indexed vLLM, if any.

    A ``RELOCATION`` module is served by delegating every attribute to its
    source, so a name the table does not list is covered exactly when the
    source module has it. Treating the kind alone as coverage would make this
    pass unable to fail for a relocated module -- and a relocated module's
    names are all absent from the pin by construction, which is the only reason
    the pass consults this function at all.
    """
    for entry in table.importable_entries():
        if entry.module != module:
            continue
        if not _version_ok(entry.applies_to, index.version):
            continue
        covered = name is None or name in entry.names
        if not covered and entry.kind == table.RELOCATION and entry.source:
            covered = index.has_symbol(entry.source, name) or index.has_module(
                f"{entry.source}.{name}"
            )
        if covered:
            return {
                "kind": entry.kind,
                "source": entry.source,
                "backport": entry.backport,
            }
    # An adapter can add a missing attribute to a module that the pin already
    # provides. Keep this in the same coverage helper so import and module-
    # attribute passes agree about a newly introduced vLLM name.
    for entry in table.adapter_entries():
        if entry.module != module or not _version_ok(entry.applies_to, index.version):
            continue
        if name is None or name in entry.names:
            return {"kind": entry.kind, "adapter": entry.adapter}
    return None


def _version_ok(specifier: str, version: str | None) -> bool:
    if version is None:
        return True
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import InvalidVersion, Version
    except ImportError:
        return True
    try:
        return Version(Version(version).public) in SpecifierSet(specifier)
    except InvalidVersion:
        return True


# --------------------------------------------------------------------------- #
# Passes
# --------------------------------------------------------------------------- #
def audit_imports(omni_imports: list[OmniImport], index: VllmIndex, table) -> dict:
    shimmed: list[dict] = []
    unshimmed: list[dict] = []
    checked = 0
    for item in omni_imports:
        checked += 1
        if item.name is None:
            present = index.has_module(item.module)
        else:
            present = index.has_symbol(item.module, item.name) or index.has_module(
                f"{item.module}.{item.name}"
            )
        if present:
            continue
        cover = shim_covers(table, index, item.module, item.name)
        record = {
            "module": item.module,
            "name": item.name,
            "omni_file": item.omni_file,
            "lineno": item.lineno,
            "type_checking": item.type_checking,
            "shim": cover,
        }
        (shimmed if cover else unshimmed).append(record)
    # A TYPE_CHECKING-only import never executes, so it cannot break a run.
    blocking = [r for r in unshimmed if not r["type_checking"]]
    return {
        "status": "pass" if not blocking else "fail",
        "checked": checked,
        "shimmed": shimmed,
        "unshimmed": unshimmed,
        "blocking": len(blocking),
    }


def _risk(pin_sig: dict, target_sig: dict, calls: list[dict]) -> tuple[str, list[dict]]:
    """How a signature difference actually affects a call on the pin.

    ``breaks``  an Omni call passes something the pinned signature cannot take.
    ``meaning`` same arity, renamed or re-annotated parameter: the call type-checks
                and means something else. This is the class no import audit sees.
    ``latent``  the signatures differ only in parts no Omni call site uses.
    """
    pin_params = set(pin_sig["positional"]) | set(pin_sig["kwonly"])
    pin_positional = len(pin_sig["positional"])
    accepts_extra_kw = pin_sig["kwarg"] is not None
    accepts_extra_pos = pin_sig["vararg"] is not None

    offenders: list[dict] = []
    for call in calls:
        if call["starargs"]:
            continue  # cannot tell statically what is passed
        if call.get("guarded"):
            continue  # its TypeError is caught; see _typeerror_guarded_lines
        unknown = [kw for kw in call["keywords"] if kw not in pin_params]
        too_many = call["positional"] > pin_positional
        if (unknown and not accepts_extra_kw) or (too_many and not accepts_extra_pos):
            offenders.append(
                {**call, "unknown_keywords": unknown, "too_many_positional": too_many}
            )
    if offenders:
        return "breaks", offenders
    same_arity = len(pin_sig["positional"]) == len(target_sig["positional"])
    if same_arity and pin_sig["positional"] != target_sig["positional"]:
        return "meaning", []
    if same_arity and pin_sig["annotations"] != target_sig["annotations"]:
        return "meaning", []
    return "latent", []


# Omni call sites that the pin cannot accept and no shim entry can fix: the
# same call breaks on the vLLM that Omni targets, because Omni's call predates
# a vLLM refactor that landed before the pin. Verified by reading the symbol in
# the 0.32.0 pin and in 0.33 -- the signatures are identical, so the pin is not
# the cause. Each is keyed by what the call actually passes, so a *different*
# unaccepted argument to the same symbol still fails the gate.
OMNI_STALE_CALLS: dict[str, dict] = {
    "vllm.model_executor.models.utils._merge_multimodal_embeddings": {
        "unknown_keywords": (),
        "positional": 4,
        "why": (
            "step_audio2's thinker passes the pre-refactor (input_ids, inputs_embeds, "
            "multimodal_embeddings, placeholder_token_id) argument list; both the pin "
            "and 0.33 take (inputs_embeds, multimodal_embeddings, is_multimodal)."
        ),
    },
    "vllm.tool_parsers.streaming.extract_required_tool_call_streaming": {
        "unknown_keywords": ("delta_text", "function_name_returned"),
        "positional": None,
        "why": (
            "serving_chat passes delta_text and function_name_returned, which neither "
            "the pin nor 0.33 declares; it reaches only tool_choice='required' "
            "streaming."
        ),
    },
}


def _offence_key(call: dict) -> tuple[tuple[str, ...], int | None]:
    """What a call passes that the pin cannot take, as a comparable key."""
    return (
        tuple(sorted(call.get("unknown_keywords") or ())),
        call["positional"] if call.get("too_many_positional") else None,
    )


def audit_calls(
    omni_imports: list[OmniImport], pin: VllmIndex, table, omni_root: Path
) -> dict:
    """Omni calls the *pinned* signature cannot accept, from the pin alone.

    This is the fatal half of the call-shape question, and unlike
    :func:`audit_semantics` it needs no second vLLM tree: whether a call site
    passes an argument the pin cannot take is decided by the pin's signature
    and Omni's call. So both build gates can run it, which is the point -- the
    target diff answers the different and softer question of which signatures
    moved between two vLLM trees, and is skipped whenever no ``--target-vllm``
    is given.

    Only a name Omni imports from vLLM and then calls plainly, in the file that
    imported it, is scored. An attribute call's receiver is unknown to the AST
    -- Omni has its own ``apply_top_k_top_p`` and its own ``post_forward``
    protocol -- so scoring those against vLLM's function would invent failures.
    """
    declared = {entry.target for entry in table.capability_entries() if entry.target}
    adapted = {
        f"{entry.module}.{name}"
        for entry in table.adapter_entries()
        for name in entry.names
    }

    # Keyed by qualname, so a call is scored only against the signature its own
    # file imported: two Omni files may import one short name from two modules.
    importers: dict[str, set[str]] = {}
    for item in omni_imports:
        if item.name is None:
            continue
        qualname = f"{item.module}.{item.name}"
        if qualname in pin.signatures:
            importers.setdefault(qualname, set()).add(item.omni_file)

    shorts = {qualname.rsplit(".", 1)[1] for qualname in importers}
    calls = collect_omni_calls(omni_root, shorts) if shorts else {}

    # One file importing the same short name from two vLLM modules binds only
    # the last one, and the AST does not say which: ambiguous, so not scored.
    bindings: dict[tuple[str, str], int] = {}
    for qualname, files in importers.items():
        short = qualname.rsplit(".", 1)[1]
        for omni_file in files:
            bindings[(short, omni_file)] = bindings.get((short, omni_file), 0) + 1

    offences: list[dict] = []
    stale_calls: list[dict] = []
    scored = 0
    for qualname in sorted(importers):
        short = qualname.rsplit(".", 1)[1]
        sites = [
            call
            for call in calls.get(short, [])
            if not call["attribute"]
            and call["omni_file"] in importers[qualname]
            and bindings[(short, call["omni_file"])] == 1
        ]
        if not sites:
            continue
        scored += len(sites)
        pin_sig = pin.signatures[qualname]
        # Both arguments are the pin: only the offender arm can fire, which is
        # the one that needs no target.
        risk, offenders = _risk(pin_sig, pin_sig, sites)
        if risk != "breaks":
            continue
        owner = qualname.rsplit(".", 1)[0].split(".")[-1]
        if qualname in adapted or any(
            d.endswith(f"{owner}.{short}") or d == short for d in declared
        ):
            continue
        known = OMNI_STALE_CALLS.get(qualname)
        expected = (
            (tuple(sorted(known["unknown_keywords"])), known["positional"])
            if known
            else None
        )
        record = {
            "symbol": qualname,
            "pin": {
                key: pin_sig[key] for key in ("positional", "kwonly", "vararg", "kwarg")
            },
            "omni_calls": offenders[:5],
            "omni_call_count": len(sites),
        }
        if expected is not None and all(
            _offence_key(call) == expected for call in offenders
        ):
            stale_calls.append({**record, "why": known["why"]})
            continue
        offences.append(record)
    unfired = sorted(set(OMNI_STALE_CALLS) - {r["symbol"] for r in stale_calls})
    return {
        "status": "fail" if offences else "pass",
        "symbols": len(importers),
        "calls": scored,
        "offences": offences,
        "declared_stale": stale_calls,
        "declared_stale_unfired": unfired,
    }


def audit_semantics(
    omni_imports: list[OmniImport],
    pin: VllmIndex,
    target: VllmIndex,
    table,
    omni_root: Path,
) -> dict:
    """Signature drift for symbols Omni uses that exist in both vLLM trees.

    This is the class an import check cannot see: a function whose name still
    resolves but whose contract moved. The pinned
    ``VllmConfig.adjust_dcp_kv_cache_interleave_size`` is the worked example --
    0.32 takes a ``KVCacheConfig`` and reads ``.kv_cache_groups`` off it, 0.33
    takes the group list itself, so Omni main's call raises AttributeError on
    the pin, but only on the decode-context-parallel path.

    Each difference is classified by what an Omni call site actually passes, so
    a parameter 0.33 merely *added* is not reported as a problem unless Omni
    passes it.
    """
    # Covered means "the table has an answer for this": a CAPABILITY entry
    # declares the path unsupported, an ADAPTER actually fixes it. Either way a
    # reviewer has already decided, so it is not an open risk.
    declared = {entry.target for entry in table.capability_entries() if entry.target}
    adapted = {
        f"{entry.module}.{name}"
        for entry in table.adapter_entries()
        for name in entry.names
    }

    # Every symbol Omni imports by name, plus the methods of the classes it
    # imports, so a drifted method is found even though the call is an attribute.
    interesting_funcs: dict[str, str] = {}
    class_names: set[str] = set()
    for item in omni_imports:
        if item.name is None:
            continue
        qualname = f"{item.module}.{item.name}"
        if qualname in target.signatures or qualname in pin.signatures:
            interesting_funcs[item.name] = qualname
        if item.name[:1].isupper():
            class_names.add(item.name)
    methods: dict[str, str] = {}
    for qualname in target.signatures:
        parts = qualname.split(".")
        if len(parts) >= 3 and parts[-2] in class_names:
            methods.setdefault(parts[-1], qualname)

    # Confidence matters more than coverage here. A name Omni imported from a
    # vLLM module resolves to that function; a bare attribute call does not --
    # vLLM-Omni has its own `post_forward` hook protocol, which name matching
    # alone cannot tell from ActiveKVConnector.post_forward. Attribute hits are
    # therefore review items, never failures.
    lookup = {**methods, **interesting_funcs}
    resolution = {
        name: ("import" if name in interesting_funcs else "attribute")
        for name in lookup
    }
    calls = collect_omni_calls(omni_root, set(lookup)) if lookup else {}

    # A plain-name call only means the vLLM function in files that imported it
    # from vLLM: vllm_omni/platforms/npu/.../qwen3_tts.py has its own
    # apply_top_k_top_p, and attributing its arguments to vLLM's would be wrong.
    importers: dict[str, set[str]] = {}
    for item in omni_imports:
        if item.name is not None:
            importers.setdefault(item.name, set()).add(item.omni_file)
    for name, sites in calls.items():
        files = importers.get(name, set())
        calls[name] = [c for c in sites if c["attribute"] or c["omni_file"] in files]

    changed: list[dict] = []
    for short, qualname in sorted(lookup.items()):
        pin_sig = pin.signatures.get(qualname)
        target_sig = target.signatures.get(qualname)
        if pin_sig is None or target_sig is None:
            continue
        if (
            pin_sig["positional"] == target_sig["positional"]
            and pin_sig["annotations"] == target_sig["annotations"]
        ):
            continue
        risk, offenders = _risk(pin_sig, target_sig, calls.get(short, []))
        how = resolution[short]
        if how == "attribute" and risk == "breaks":
            # The receiver is unknown, so "Omni passes an argument the pin
            # cannot take" is a hypothesis about a possibly unrelated callee.
            risk = "meaning"
        owner = qualname.rsplit(".", 1)[0].split(".")[-1]
        covered_by_adapter = qualname in adapted
        changed.append(
            {
                "symbol": qualname,
                "risk": risk,
                "resolved_by": how,
                "adapted": covered_by_adapter,
                "pin": {
                    k: pin_sig[k] for k in ("positional", "annotations", "returns")
                },
                "target": {
                    k: target_sig[k] for k in ("positional", "annotations", "returns")
                },
                "declared": covered_by_adapter
                or any(d.endswith(f"{owner}.{short}") or d == short for d in declared),
                "omni_calls": (offenders or calls.get(short, []))[:5],
                "omni_call_count": len(calls.get(short, [])),
            }
        )

    blocking = [c for c in changed if c["risk"] == "breaks" and not c["declared"]]
    review = [c for c in changed if c["risk"] == "meaning" and not c["declared"]]
    return {
        "status": "fail" if blocking else ("review" if review else "pass"),
        "compared": len(lookup),
        "changed": changed,
        "breaks": len(blocking),
        "meaning": len(review),
        "latent": sum(1 for c in changed if c["risk"] == "latent"),
        "declared_capabilities": sorted(declared),
    }


def audit_staleness(pin: VllmIndex, table, target: VllmIndex | None = None) -> dict:
    """Whether every entry active on this pin is still needed.

    A shim that outlives its drift is worse than no shim: it is dead code that
    a reader has to disprove. The import hook sits last in ``sys.meta_path``, so
    a stale entry is invisible at runtime -- the pin serves the module and the
    hook is never consulted. This pass is what makes it visible.
    """
    stale: list[dict] = []
    active: list[dict] = []
    for entry in table.ENTRIES:
        if not _version_ok(entry.applies_to, pin.version):
            continue
        reason = None
        if entry.kind in (table.RELOCATION, table.BACKPORT):
            if pin.has_module(entry.module):
                reason = f"the pin provides {entry.module}"
        elif entry.kind == table.ADAPTER:
            # An adapter is stale when the pin already has the shape Omni calls,
            # which needs the vLLM Omni targets to compare against. Two of the
            # three adapter cases are additive -- the pin does not have the
            # symbol at all -- so "absent from the pin" is the adapter working,
            # not a stale entry. At runtime _PatchingLoader catches the rest: an
            # adapter whose apply() patches nothing fails the import.
            if target is not None:
                matched = []
                for name in entry.names:
                    qualname = f"{entry.module}.{name}"
                    pin_sig = pin.signatures.get(qualname)
                    target_sig = target.signatures.get(qualname)
                    if pin_sig is not None and target_sig is not None:
                        matched.append(
                            pin_sig["positional"] == target_sig["positional"]
                        )
                if matched and all(matched):
                    reason = (
                        f"the pin's {entry.module} already matches the targeted vLLM"
                    )
        record = {
            "module": entry.module,
            "kind": entry.kind,
            "remove_when": entry.remove_when,
        }
        if reason:
            stale.append({**record, "why": reason})
        else:
            active.append(record)
    return {
        "status": "fail" if stale else "pass",
        "active": active,
        "stale": stale,
    }


OMNI_ROOT: Path | None = None
"""Set by :func:`main`; the audit functions take the root explicitly."""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _resolve_package_dir(root: Path, flag: str) -> Path | None:
    """Accept either the ``vllm`` package dir or the repo that contains it.

    ``--omni-root`` takes a repo root and ``--vllm-root`` takes a package dir,
    and getting them the wrong way round is silent rather than fatal: a repo root
    indexes no modules, so every symbol Omni imports looks absent and the run
    reports a long list of invented misses with a FAIL verdict. Resolve both
    spellings and name which one was used instead.
    """
    root = root.resolve()
    if (root / "__init__.py").is_file():
        return root
    nested = root / "vllm"
    if (nested / "__init__.py").is_file():
        print(f"[note]      {flag} is a source root; using {nested}")
        return nested
    print(
        f"[error] {flag} {root} is not a vllm package dir: it has no __init__.py "
        "and no vllm/__init__.py under it",
        file=sys.stderr,
    )
    return None


def installed_vllm_root() -> tuple[Path | None, str | None]:
    try:
        import vllm
    except Exception:
        return None, None
    root = Path(vllm.__file__).parent
    return root, getattr(vllm, "__version__", None)


def main(argv: list[str] | None = None) -> int:
    global OMNI_ROOT
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--omni-root",
        required=True,
        type=Path,
        help="vLLM-Omni checkout (the dir holding vllm_omni/)",
    )
    parser.add_argument(
        "--vllm-root",
        type=Path,
        default=None,
        help="pinned vllm package dir (default: the installed one)",
    )
    parser.add_argument(
        "--vllm-version",
        default=None,
        help="version of --vllm-root (default: the installed one)",
    )
    parser.add_argument(
        "--target-vllm",
        type=Path,
        default=None,
        help="vllm package dir Omni targets, enables the semantic pass",
    )
    parser.add_argument("--target-version", default=None)
    parser.add_argument(
        "--report", type=Path, default=None, help="write the full JSON report here"
    )
    parser.add_argument(
        "--fail-on-review",
        action="store_true",
        help="also fail when a pinned symbol changed meaning under the same arity",
    )
    args = parser.parse_args(argv)

    OMNI_ROOT = args.omni_root.resolve()
    if not (OMNI_ROOT / "vllm_omni").is_dir():
        print(f"[error] {OMNI_ROOT} does not contain vllm_omni/", file=sys.stderr)
        return 2

    pin_root, pin_version = args.vllm_root, args.vllm_version
    if pin_root is None:
        pin_root, detected = installed_vllm_root()
        pin_version = pin_version or detected
        if pin_root is None:
            print("[error] no --vllm-root and vllm is not importable", file=sys.stderr)
            return 2
    else:
        pin_root = _resolve_package_dir(pin_root, "--vllm-root")
        if pin_root is None:
            return 2
    if args.target_vllm is not None:
        resolved_target = _resolve_package_dir(args.target_vllm, "--target-vllm")
        if resolved_target is None:
            return 2
        args.target_vllm = resolved_target

    table = load_shim_table()
    omni_imports = collect_omni_imports(OMNI_ROOT)
    pin = VllmIndex.build(Path(pin_root), pin_version)

    imports = audit_imports(omni_imports, pin, table)
    print(
        f"[imports]   {imports['status']}: {imports['checked']} vllm imports, "
        f"{len(imports['shimmed'])} shimmed, {imports['blocking']} unshimmed"
    )
    for record in imports["shimmed"]:
        print(
            f"  SHIMMED  {record['module']}.{record['name']} <- {record['shim']['kind']} ({record['omni_file']})"
        )
    for record in imports["unshimmed"]:
        mark = "TYPING " if record["type_checking"] else "MISSING"
        print(
            f"  {mark}  {record['module']}.{record['name']} ({record['omni_file']}:{record['lineno']})"
        )

    target: VllmIndex | None = None
    if args.target_vllm is not None:
        target = VllmIndex.build(args.target_vllm, args.target_version)

    attribute_uses = collect_module_attribute_uses(OMNI_ROOT)
    attributes = audit_module_attributes(attribute_uses, pin, table)
    print(
        f"[attrs]     {attributes['status']}: {attributes['checked']} module attributes used, "
        f"{len(attributes['shimmed'])} shimmed, {len(attributes['missing'])} missing"
    )
    for record in attributes["shimmed"]:
        kind = record["shim"].get("kind")
        print(
            f"  SHIMMED  {record['module']}.{record['attr']} <- {kind} ({record['omni_file']})"
        )
    for record in attributes["missing"]:
        print(
            f"  MISSING  {record['module']}.{record['attr']} ({record['omni_file']}:{record['lineno']})"
        )

    calls = audit_calls(omni_imports, pin, table, OMNI_ROOT)
    print(
        f"[calls]     {calls['status']}: {calls['calls']} calls to "
        f"{calls['symbols']} pinned symbols, {len(calls['offences'])} the pin cannot accept, "
        f"{len(calls['declared_stale'])} declared stale in Omni"
    )
    for record in calls["declared_stale"]:
        print(f"  OMNI-BUG {record['symbol']}: {record['why']}")
    for name in calls["declared_stale_unfired"]:
        print(f"  DEAD     {name} is declared a stale Omni call but no longer breaks")
    for record in calls["offences"]:
        print(f"  BREAKS   {record['symbol']}")
        print(f"           pin    {record['pin']['positional']}")
        for call in record["omni_calls"][:3]:
            extra = ""
            if call.get("unknown_keywords"):
                extra = f" passes {call['unknown_keywords']}"
            elif call.get("too_many_positional"):
                extra = f" passes {call['positional']} positional"
            print(f"           {call['omni_file']}:{call['lineno']}{extra}")

    staleness = audit_staleness(pin, table, target)
    print(
        f"[staleness] {staleness['status']}: {len(staleness['active'])} entries still needed, "
        f"{len(staleness['stale'])} stale"
    )
    for record in staleness["stale"]:
        print(
            f"  STALE  {record['module']} ({record['kind']}): {record['why']} -- {record['remove_when']}"
        )

    semantics = {"status": "skipped", "reason": "no --target-vllm"}
    if target is not None:
        semantics = audit_semantics(omni_imports, pin, target, table, OMNI_ROOT)
        print(
            f"[semantics] {semantics['status']}: {semantics['compared']} symbols compared, "
            f"{semantics['breaks']} breaking, {semantics['meaning']} changed meaning, "
            f"{semantics['latent']} latent"
        )
        for record in semantics["changed"]:
            if record["risk"] == "latent":
                continue
            if record.get("adapted"):
                flag = "ADAPTED "
            elif record["declared"]:
                flag = "DECLARED"
            else:
                flag = record["risk"].upper().ljust(8)
            how = (
                ""
                if record["resolved_by"] == "import"
                else "  (attribute call, receiver unresolved)"
            )
            print(f"  {flag} {record['symbol']}{how}")
            print(f"           pin    {record['pin']['positional']}")
            print(f"           target {record['target']['positional']}")
            for call in record["omni_calls"][:3]:
                extra = ""
                if call.get("unknown_keywords"):
                    extra = f" passes {call['unknown_keywords']}"
                elif call.get("too_many_positional"):
                    extra = f" passes {call['positional']} positional"
                print(f"           {call['omni_file']}:{call['lineno']}{extra}")
    else:
        print(
            "[semantics] skipped: pass --target-vllm to diff signatures against the vLLM Omni targets"
        )

    unsupported = [
        path
        for entry in table.capability_entries()
        if _version_ok(entry.applies_to, pin.version)
        for path in entry.unsupported
    ]
    if unsupported:
        print("[unsupported] feature paths this pin cannot serve for this Omni tree:")
        for path in unsupported:
            print(f"  - {path}")

    # A breaking call on the pin is as fatal as a missing import; a changed
    # meaning is a question for a human, so it fails only on request.
    failed = (
        imports["status"] == "fail"
        or attributes["status"] == "fail"
        or staleness["status"] == "fail"
        or calls["status"] == "fail"
        or semantics.get("breaks")
        or (args.fail_on_review and semantics.get("meaning"))
    )
    verdict = "FAIL" if failed else "PASS"
    # The verdict has to carry its own coverage. Without this a run with no
    # --target-vllm printed a bare PASS and exited 0 while the semantic diff
    # had not run at all, and automation reads the exit code.
    coverage = ["imports", "attributes", "calls", "staleness"]
    if target is not None:
        coverage.append("semantics")
    note = "" if target is not None else "; semantics not compared, no --target-vllm"
    print(f"[verdict]   {verdict} ({', '.join(coverage)}{note})")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "omni_root": str(OMNI_ROOT),
                    "vllm_root": str(pin_root),
                    "vllm_version": pin.version,
                    "target_vllm": str(args.target_vllm) if args.target_vllm else None,
                    "attributes": attributes,
                    "calls": calls,
                    "coverage": coverage,
                    "imports": imports,
                    "semantics": semantics,
                    "staleness": staleness,
                    "unsupported": unsupported,
                    "verdict": verdict.lower(),
                },
                indent=1,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"[report]    {args.report}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
