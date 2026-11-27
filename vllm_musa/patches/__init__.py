# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime patches for vLLM compatibility with the MUSA platform.

vLLM **source** edits are applied at BUILD time: ``setup.py`` clones the pinned
upstream vLLM into ``third_party/vllm`` and applies the
``vllm_musa/patches/series/`` ``git format-patch`` series via ``build_apply.py``
(``Makefile.sync`` regenerates the series across vLLM versions). The installed
vLLM is therefore already patched; this module does **not** rewrite installed
vLLM source at runtime and there is no runtime fallback.

What remains here is runtime-only by nature:

- :func:`apply_object_patches` — the explicit in-process object/monkey-patch
  phase (live-object edits with no source-diff form: the spec-decode kernel
  prime and the draft-TP=1 wiring). These patch live objects, not files.
- :func:`patch_report` — a read-only audit of those object patches, surfaced by
  ``vllm_collect_env``.
"""

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from vllm.logger import init_logger

logger = init_logger(__name__)

_object_patches_applied = False


@dataclass(frozen=True)
class PatchSpec:
    """Declarative metadata for one vLLM-MUSA patch.

    Fields are *derived* by default; a patch file may override any of them with
    optional module-level constants (``PATCH_ID``, ``PATCH_PHASE``,
    ``PATCH_REQUIRED``, ``PATCH_VERSION_RANGE``, ``PATCH_REMOVAL_CONDITION``,
    ``PATCH_COMPAT_NOTE``). Existing patch files define none of these, so they
    keep working unchanged — the override is opt-in.
    """

    id: str
    module: str
    file: str
    kind: str  # "source-transform" | "side-effect"
    phase: str = "plugin-load"
    process_scope: str = (
        "disk-persistent"  # source-transform; "process-local" for side-effect
    )
    required: bool = True
    version_range: str | None = None
    removal_condition: str | None = None
    compat_note: str | None = None


def _load_patch_spec(
    module_name: str,
    patch_file: Path,
    patch_module: ModuleType | None = None,
) -> PatchSpec:
    """Build the:class:`PatchSpec` for a patch file.

    ``kind``/``process_scope`` are derived from the loaded module; the rest
    default sensibly and may be overridden by optional ``PATCH_*`` constants.
    """
    pid = patch_file.name
    if pid.endswith(".patch.py"):
        pid = pid[: -len(".patch.py")]
    if patch_module is None:
        patch_module = _load_patch_module(patch_file)
    patches = getattr(patch_module, "PATCHES", []) if patch_module else []
    has_norm = (
        callable(getattr(patch_module, "normalize_source", None))
        if patch_module
        else False
    )
    kind = "source-transform" if (patches or has_norm) else "side-effect"
    scope = "process-local" if kind == "side-effect" else "disk-persistent"

    def _override(name: str, default):
        return getattr(patch_module, name, default) if patch_module else default

    return PatchSpec(
        id=_override("PATCH_ID", pid),
        module=module_name,
        file=patch_file.name,
        kind=kind,
        phase=_override("PATCH_PHASE", "plugin-load"),
        process_scope=scope,
        required=bool(_override("PATCH_REQUIRED", True)),
        version_range=_override("PATCH_VERSION_RANGE", None),
        removal_condition=_override("PATCH_REMOVAL_CONDITION", None),
        compat_note=_override("PATCH_COMPAT_NOTE", None),
    )


def _get_patch_files():
    """Get all patch files in the patches directory, in deterministic order.

    sort by filename so discovery/application order is reproducible
    across machines and runs (``glob`` order is filesystem-dependent). Patches
    target independent modules so order does not affect correctness, but a
    stable order makes the patch report and any import-side-effect ordering
    deterministic.
    """
    patches_dir = Path(__file__).parent
    patch_files = []

    for patch_file in sorted(patches_dir.glob("*.patch.py")):
        # Extract module name from filename
        # Format: module.name.patch.py -> module.name
        module_name = patch_file.stem.rsplit(".patch", 1)[0]
        # Convert filename format to module format
        # vllm__attention__ops__triton_unified_attention ->
        # vllm.attention.ops.triton_unified_attention
        module_name = module_name.replace("__", ".")
        patch_files.append((module_name, patch_file))

    return patch_files


def _load_patch_module(patch_file: Path) -> ModuleType | None:
    """Load a patch module.

    Patch files may define a PATCHES list of (old_str, new_str) tuples and an
    optional normalize_source(source: str) -> str hook for idempotent cleanup.
    """
    spec = importlib.util.spec_from_file_location("patch_config", patch_file)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.warning("Failed to load patch config from %s: %s", patch_file, e)
        return None


def _resolve_module_origin(module_name: str) -> str | None:
    """Resolve a module source file without importing the target module."""
    parts = module_name.split(".")
    if parts and parts[0] == "vllm":
        roots: list[Path] = []
        try:
            root_spec = importlib.util.find_spec("vllm")
        except (ModuleNotFoundError, ImportError):
            root_spec = None
        if root_spec is not None:
            if root_spec.submodule_search_locations:
                roots.extend(Path(p) for p in root_spec.submodule_search_locations)
            elif root_spec.origin is not None:
                roots.append(Path(root_spec.origin).parent)
        for sys_path in sys.path:
            if not sys_path:
                sys_path = os.getcwd()
            candidate_root = Path(sys_path) / "vllm"
            if candidate_root.is_dir():
                roots.append(candidate_root)

        seen: set[Path] = set()
        for root in roots:
            root = root.resolve()
            if root in seen:
                continue
            seen.add(root)
            rel = Path(*parts[1:]) if len(parts) > 1 else Path()
            module_file = (root / rel).with_suffix(".py")
            if module_file.is_file():
                return str(module_file)
            package_file = root / rel / "__init__.py"
            if package_file.is_file():
                return str(package_file)

    try:
        spec = importlib.util.find_spec(module_name)
    except (ModuleNotFoundError, ImportError) as e:
        logger.debug(
            "Module %s not found or has import issues: %s, "
            "skipping patch (this is expected for version-specific patches "
            "or when modules are not yet fully initialized)",
            module_name,
            e,
        )
        return None
    if spec is None or spec.origin is None:
        logger.debug("Module %s not found, skipping patch", module_name)
        return None
    return spec.origin


def patch_report() -> list[dict]:
    """Read-only status of the runtime object patches.

    vLLM **source** edits are applied at BUILD time (the ``series/`` diff series
    via ``build_apply.py``), so the installed vLLM is already patched and is not
    audited here. This reports the in-process **object** patches — each a
    ``*.patch.py`` with an empty ``PATCHES`` list and a module-level
    ``apply()`` — and is surfaced by ``vllm_collect_env``. It never modifies
    anything and is safe to call at any time.

    Each entry carries ``module``, ``file``, ``id``, ``kind``, ``object_patch``
    (whether it exposes ``apply()``), ``status`` (``side-effect`` /
    ``load-failed`` / ``misplaced-source-patch`` / ``error``), ``phase``,
    ``process_scope``, ``required``, ``version_range``, ``removal_condition`` and
    ``is_failure``. Returned in deterministic ``_get_patch_files()`` order.
    """
    report: list[dict] = []
    for module_name, patch_file in _get_patch_files():
        entry: dict = {
            "module": module_name,
            "file": patch_file.name,
            "id": patch_file.name,
            "kind": "unknown",
            "target_resolved": False,
            "status": "unknown",
            "phase": "plugin-load",
            "process_scope": "process-local",
            "required": True,
            "version_range": None,
            "removal_condition": None,
        }
        try:
            entry["target_resolved"] = _resolve_module_origin(module_name) is not None
            patch_module = _load_patch_module(patch_file)
            spec = _load_patch_spec(module_name, patch_file, patch_module)
            entry.update(
                id=spec.id,
                kind=spec.kind,
                phase=spec.phase,
                process_scope=spec.process_scope,
                required=spec.required,
                version_range=spec.version_range,
                removal_condition=spec.removal_condition,
            )
            if patch_module is None:
                entry.update(kind="load-failed", status="load-failed")
            elif spec.kind == "side-effect":
                entry["object_patch"] = callable(getattr(patch_module, "apply", None))
                entry["status"] = "side-effect"
            else:
                # A non-empty-PATCHES (source-transform) .patch.py is misplaced
                # now that source edits live in series/ and apply at build time.
                # Flag it loudly rather than silently treating it as runtime.
                entry["status"] = "misplaced-source-patch"
        except Exception as e:  # a report must never raise
            entry.update(status="error", error=str(e))
        entry["is_failure"] = bool(entry.get("required", True)) and entry["status"] in {
            "load-failed",
            "misplaced-source-patch",
            "error",
        }
        report.append(entry)
    return report


def apply_object_patches(force: bool = False) -> list[dict]:
    """Apply explicit in-process object/monkey patches.

    Source-transform edits mutate vLLM's *files* and are applied at BUILD time
    (the ``series/`` diff series). A few MUSA patches instead install in-process
    object monkey-patches or prime a Triton kernel — live-object edits with no
    source-diff form. These historically ran as an **import-time side effect**
    of the legacy disk patcher — implicit, order-fragile, and invisible to any
    report. Making them explicit, such a patch file keeps
    ``PATCHES = []`` and defines a module-level idempotent ``def apply() -> None``;
    this phase loads each patch module in deterministic :func:`_get_patch_files`
    order and calls ``apply()`` where present.

    Ordering / correctness notes:

    - Files sort by name, so ``vllm.distributed.parallel_state`` (draft-TP=1)
      is visited before ``vllm.v1.spec_decode.eagle`` (kernel prime). This
      matches the previous import-time order exactly. Both orders
      are safe: parallel_state's ``apply()`` is dormant unless
      ``VLLM_MUSA_DRAFT_TP1=1`` and, when active, primes the kernel itself
      before importing the proposer classes; eagle's ``apply()`` is an
      idempotent prime import. The proposer modules are not bound by vLLM until
      model load, which is after this phase.
    - Each ``apply()`` is individually idempotent (import is a no-op after the
      first; the draft-TP=1 wiring guards on a class marker), so this phase is
      safe to call more than once. ``force`` re-runs it anyway.

    Returns a per-patch list of ``{module, file, status, [error]}`` dicts for
    logging/diagnostics. Never raises — a failing object-patch is logged and
    recorded (best-effort contract).
    """
    global _object_patches_applied
    if _object_patches_applied and not force:
        return []

    results: list[dict] = []
    for module_name, patch_file in _get_patch_files():
        patch_module = _load_patch_module(patch_file)
        if patch_module is None:
            continue
        apply_fn = getattr(patch_module, "apply", None)
        if not callable(apply_fn):
            continue
        entry: dict = {
            "module": module_name,
            "file": patch_file.name,
            "status": "applied",
        }
        try:
            apply_fn()
            logger.info("Applied object-patch %s", module_name)
        except Exception as e:
            entry.update(status="error", error=str(e))
            logger.warning("Failed to apply object-patch %s: %s", module_name, e)
        results.append(entry)

    _object_patches_applied = True
    return results
