from __future__ import annotations

from collections.abc import Callable

from .declarative import resolve_declarative_runtime_plan
from .types import ExecutionSignature, ModelSignature, RuntimePlan

BuiltinRuntimePlanProvider = Callable[
    [ModelSignature, ExecutionSignature],
    RuntimePlan | None,
]

# Built-in providers are conservative plans used when no external artifact is
# active. They share the same RuntimePlanApplication/consumer API as external
# plans; there is no second contract resolver.
BUILTIN_RUNTIME_PLAN_PROVIDERS: tuple[BuiltinRuntimePlanProvider, ...] = (
    resolve_declarative_runtime_plan,
)
