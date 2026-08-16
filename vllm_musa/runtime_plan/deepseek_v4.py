from __future__ import annotations

from .types import ExecutionSignature, ModelSignature, RuntimePlan


def resolve_deepseek_v4_plan(
    model: ModelSignature,
    execution: ExecutionSignature,
) -> RuntimePlan | None:
    """Resolve DeepSeek-V4 defaults from the versioned declarative profile."""

    from .declarative import resolve_declarative_runtime_plan

    return resolve_declarative_runtime_plan(
        model,
        execution,
        identifier="deepseek_v4",
    )
