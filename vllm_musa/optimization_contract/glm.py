from __future__ import annotations

from dataclasses import replace

from .types import (
    ExecutionSignature,
    ModelFamily,
    ModelRole,
    ModelSignature,
    MusaOptimizationContract,
    OptimizationFeature,
)


def _is_glm5_dsa(model: ModelSignature) -> bool:
    architectures = set(model.outer_architectures or model.architectures)
    return (
        "GlmMoeDsaForCausalLM" in architectures
        or model.model_type == "glm_moe_dsa"
    )


def resolve_glm_contract(
    model: ModelSignature,
    execution: ExecutionSignature,
) -> MusaOptimizationContract | None:
    if not _is_glm5_dsa(model):
        return None

    model = replace(model, family=ModelFamily.GLM5, role=ModelRole.TEXT)
    preferred = {
        OptimizationFeature.GLM5_EAGER_MCCL_INIT,
    }
    if (
        model.dtype == "bfloat16"
        and model.quantization == "fp8"
        and model.uses_mla is True
        and model.index_topk == 2048
    ):
        preferred.add(OptimizationFeature.GLM5_SPARSE_MLA_MATE_PREFILL)

    return MusaOptimizationContract(
        model=model,
        execution=execution,
        profile="glm5",
        supported_features=frozenset(preferred),
        preferred_features=frozenset(preferred),
    )
