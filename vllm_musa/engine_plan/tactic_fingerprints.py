# SPDX-License-Identifier: Apache-2.0

"""Host-owned implementation identities for runtime-decision tactics."""

from __future__ import annotations

from dataclasses import dataclass

from .artifacts import compute_artifact_fingerprint


@dataclass(frozen=True, slots=True)
class RuntimeDecisionImplementation:
    """Live provider dependency and selector ABI for one runtime decision."""

    provider_operation: str | None
    provider: str | None
    selector_abi: str
    choice_abis: tuple[tuple[str, str], ...] = ()

    @property
    def provider_key(self) -> tuple[str, str] | None:
        if self.provider_operation is None or self.provider is None:
            return None
        return self.provider_operation, self.provider

    def fingerprint(
        self,
        provider_fingerprint: str | None = None,
        *,
        choice: str | None = None,
    ) -> str:
        """Bind a selector ABI to its provider or atomic backend identity."""

        if self.provider_key is not None:
            if not isinstance(provider_fingerprint, str) or not provider_fingerprint:
                raise ValueError("provider-backed tactic requires a fingerprint")
            # Preserve the schema used by existing RMSNorm artifacts exactly.
            return compute_artifact_fingerprint(
                {
                    "provider": provider_fingerprint,
                    "selector_abi": self.selector_abi,
                }
            )
        choice_abis = dict(self.choice_abis)
        if choice not in choice_abis:
            raise ValueError(f"runtime tactic choice {choice!r} is not registered")
        return compute_artifact_fingerprint(
            {
                "selector_abi": self.selector_abi,
                "choice": choice,
                "choice_abi": choice_abis[choice],
            }
        )


_RUNTIME_DECISION_IMPLEMENTATIONS = {
    "musa.fused_add_rms_norm.min_rows": RuntimeDecisionImplementation(
        provider_operation="fused_add_rms_norm",
        provider="musa",
        selector_abi="musa.fused_add_rms_norm.threshold.v1",
    ),
    "musa.fused_moe.dispatch_policy": RuntimeDecisionImplementation(
        provider_operation=None,
        provider=None,
        selector_abi="musa.fused_moe.dispatch_policy.v1",
        choice_abis=(
            ("upstream", "vllm.fused_moe.established_dispatch.v1"),
            ("gemv", "vllm_musa.fused_moe.native_gemv.v1"),
            ("grouped_gemm", "vllm_musa.fused_moe.grouped_gemm.v1"),
        ),
    ),
}


def runtime_decision_implementation(
    operation: str,
) -> RuntimeDecisionImplementation | None:
    """Return the live identity recipe, or ``None`` to fail closed."""

    return _RUNTIME_DECISION_IMPLEMENTATIONS.get(operation)
