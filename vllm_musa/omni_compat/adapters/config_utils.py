# SPDX-License-Identifier: Apache-2.0
"""Expose the v0.34 compile-factor type alias on the v0.32 pin.

The alias is deliberately kept independent of the rest of vLLM's compile-cache
refactor.  The method adapters for ``ModelConfig`` and ``VllmConfig`` provide
the small runtime bridge needed by Omni, while this module only fills the
imported type name.
"""

from __future__ import annotations

from types import ModuleType

ADAPTED = "__vllm_musa_compile_factors_alias__"

CompileFactors = dict[str, object]


def is_applied(module: ModuleType) -> list[str]:
    alias = getattr(module, ADAPTED, None)
    return (
        ["CompileFactors"] if alias is getattr(module, "CompileFactors", None) else []
    )


def is_satisfied(module: ModuleType) -> bool:
    return hasattr(module, "CompileFactors")


def apply(module: ModuleType) -> list[str]:
    if hasattr(module, "CompileFactors"):
        return []
    alias = CompileFactors
    module.CompileFactors = alias
    setattr(module, ADAPTED, alias)
    return ["CompileFactors"]
