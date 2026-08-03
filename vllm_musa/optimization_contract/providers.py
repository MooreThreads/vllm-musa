from __future__ import annotations

from collections.abc import Callable

from .qwen import resolve_qwen_contract
from .types import ExecutionSignature, ModelSignature, MusaOptimizationContract

ContractProvider = Callable[
    [ModelSignature, ExecutionSignature],
    MusaOptimizationContract | None,
]

# Keep provider registration explicit. DeepSeek-V4 will be added here only
# after its merged policy has an isolated evidence and migration pass.
CONTRACT_PROVIDERS: tuple[ContractProvider, ...] = (resolve_qwen_contract,)
