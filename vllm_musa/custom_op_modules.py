"""Compatibility preload for modules that register vLLM-MUSA operators."""

from importlib import import_module

from .operator_manifest import preload_modules

CUSTOM_OP_MODULES = preload_modules()


def preload_custom_op_modules() -> None:
    for module_name in CUSTOM_OP_MODULES:
        import_module(module_name)
