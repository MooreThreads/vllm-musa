"""Compatibility preload for modules that register vLLM-MUSA operators."""

from dataclasses import dataclass
from importlib import import_module

from .operator_manifest import preload_modules

CUSTOM_OP_MODULES = preload_modules()


@dataclass(frozen=True)
class PreloadReport:
    loaded: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()


def preload_custom_op_modules() -> PreloadReport:
    loaded = []
    failed = []
    for module_name in CUSTOM_OP_MODULES:
        try:
            import_module(module_name)
        except Exception:
            failed.append(module_name)
            raise
        loaded.append(module_name)
    return PreloadReport(tuple(loaded), tuple(failed))
