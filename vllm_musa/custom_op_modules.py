"""Compatibility preload for modules that register vLLM-MUSA operators."""

import logging
from dataclasses import dataclass
from importlib import import_module

from .operator_manifest import preload_modules

logger = logging.getLogger(__name__)

CUSTOM_OP_MODULES = preload_modules()


@dataclass(frozen=True)
class PreloadReport:
    loaded: tuple[str, ...] = ()


def preload_custom_op_modules() -> PreloadReport:
    loaded = []
    for module_name in CUSTOM_OP_MODULES:
        try:
            import_module(module_name)
        except Exception:
            logger.error(
                "vLLM-MUSA operator preload failed while importing %s", module_name
            )
            raise
        loaded.append(module_name)
    return PreloadReport(loaded=tuple(loaded))
