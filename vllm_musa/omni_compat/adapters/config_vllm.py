# SPDX-License-Identifier: Apache-2.0
"""Bridge the v0.34 VllmConfig compile-factor entry point."""

from __future__ import annotations

from types import ModuleType
from typing import Any


def _compile_factors(self: Any, include_version: bool = True) -> dict[str, object]:
    """Keep the pin's complete legacy config hash as one factor."""
    return {"legacy_compute_hash": self.compute_hash(include_version=include_version)}


def is_applied(module: ModuleType) -> list[str]:
    vllm_config = getattr(module, "VllmConfig", None)
    return ["VllmConfig"] if getattr(vllm_config, "compile_factors", None) is _compile_factors else []


def is_satisfied(module: ModuleType) -> bool:
    vllm_config = getattr(module, "VllmConfig", None)
    return callable(getattr(vllm_config, "compile_factors", None))


def apply(module: ModuleType) -> list[str]:
    vllm_config = getattr(module, "VllmConfig", None)
    if vllm_config is None:
        return []
    if hasattr(vllm_config, "compile_factors"):
        return []
    vllm_config.compile_factors = _compile_factors
    return ["VllmConfig"]
