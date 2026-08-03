from __future__ import annotations

from collections.abc import Callable

from .deepseek_v4 import resolve_deepseek_v4_contract
from .qwen import resolve_qwen_contract
from .types import ExecutionSignature, ModelSignature, MusaOptimizationContract

ContractProvider = Callable[
    [ModelSignature, ExecutionSignature],
    MusaOptimizationContract | None,
]

# Keep provider registration explicit. Providers must fail closed when their
# exact family metadata is incomplete so one family cannot enable another's
# fast paths.
CONTRACT_PROVIDERS: tuple[ContractProvider, ...] = (
    resolve_deepseek_v4_contract,
    resolve_qwen_contract,
)
