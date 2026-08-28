# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host-owned compatibility boundary for the live vLLM IR registry."""

from __future__ import annotations

from typing import Any

from .api import EngineIrProviderMetadata


def _load_ir_registry() -> dict[str, Any]:
    from vllm.ir.op import IrOp
    from vllm.platforms import current_platform

    current_platform.import_ir_kernels()
    return IrOp.registry


def _provider_metadata(
    operation: str,
    provider: str,
    implementation: Any,
) -> EngineIrProviderMetadata | None:
    if not getattr(implementation, "supported", False):
        return None
    return EngineIrProviderMetadata(
        operation=operation,
        provider=provider,
        implementation_fingerprint=str(implementation.uuid()),
    )


def list_engine_ir_providers() -> tuple[EngineIrProviderMetadata, ...]:
    """Return a deterministic snapshot of supported live IR providers."""

    providers: list[EngineIrProviderMetadata] = []
    for operation, ir_op in sorted(_load_ir_registry().items()):
        for provider, implementation in sorted(ir_op.impls.items()):
            metadata = _provider_metadata(operation, provider, implementation)
            if metadata is not None:
                providers.append(metadata)
    return tuple(providers)


def find_engine_ir_provider(
    operation: str,
    provider: str,
) -> EngineIrProviderMetadata | None:
    """Resolve one supported provider through the host compatibility layer."""

    ir_op = _load_ir_registry().get(operation)
    if ir_op is None:
        return None
    implementation = ir_op.impls.get(provider)
    if implementation is None:
        return None
    return _provider_metadata(operation, provider, implementation)
