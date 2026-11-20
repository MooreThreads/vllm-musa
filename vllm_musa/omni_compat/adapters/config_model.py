# SPDX-License-Identifier: Apache-2.0
"""Bridge Omni's ``compile_factors`` override to the v0.32 model config."""

from __future__ import annotations

from types import ModuleType
from typing import Any

ADAPTED = "__vllm_musa_compile_factors_model_bridge__"


def _compile_factors(self: Any) -> dict[str, object]:
    """Represent the legacy model hash as one stable compile factor.

    vLLM 0.32 computes one complete model-config hash. vLLM 0.34 stores the
    equivalent information in a factor mapping. Keeping the old hash as one
    factor preserves invalidation semantics without reimplementing the broad
    0.34 config-field inventory on the pinned branch.
    """
    return {"legacy_compute_hash": self.compute_hash()}


def is_applied(module: ModuleType) -> list[str]:
    model_config = getattr(module, "ModelConfig", None)
    return ["ModelConfig"] if getattr(model_config, "compile_factors", None) is _compile_factors else []


def is_satisfied(module: ModuleType) -> bool:
    model_config = getattr(module, "ModelConfig", None)
    return callable(getattr(model_config, "compile_factors", None))


def apply(module: ModuleType) -> list[str]:
    model_config = getattr(module, "ModelConfig", None)
    if model_config is None:
        return []
    if hasattr(model_config, "compile_factors"):
        return []
    model_config.compile_factors = _compile_factors
    return ["ModelConfig"]
