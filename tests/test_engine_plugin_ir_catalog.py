# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

from vllm_musa.engine_plugins import (
    EngineIrProviderMetadata,
    find_engine_ir_provider,
    list_engine_ir_providers,
)

ROOT = Path(__file__).parents[1]


class _FakeImplementation:
    def __init__(self, fingerprint: str, *, supported: bool = True):
        self.supported = supported
        self._fingerprint = fingerprint

    def uuid(self) -> str:
        return self._fingerprint


def _fake_registry():
    return {
        "fused_add_rms_norm": SimpleNamespace(
            impls={
                "native": _FakeImplementation("sha256:native"),
                "musa": _FakeImplementation("sha256:musa"),
                "disabled": _FakeImplementation(
                    "sha256:disabled",
                    supported=False,
                ),
            }
        )
    }


def test_host_ir_catalog_exposes_supported_provider_metadata(monkeypatch):
    from vllm_musa.engine_plugins import ir_catalog

    monkeypatch.setattr(ir_catalog, "_load_ir_registry", _fake_registry)

    providers = list_engine_ir_providers()

    assert providers == (
        EngineIrProviderMetadata(
            operation="fused_add_rms_norm",
            provider="musa",
            implementation_fingerprint="sha256:musa",
        ),
        EngineIrProviderMetadata(
            operation="fused_add_rms_norm",
            provider="native",
            implementation_fingerprint="sha256:native",
        ),
    )


def test_host_ir_catalog_resolves_one_provider_without_exposing_registry(
    monkeypatch,
):
    from vllm_musa.engine_plugins import ir_catalog

    monkeypatch.setattr(ir_catalog, "_load_ir_registry", _fake_registry)

    assert find_engine_ir_provider(
        "fused_add_rms_norm",
        "musa",
    ) == EngineIrProviderMetadata(
        operation="fused_add_rms_norm",
        provider="musa",
        implementation_fingerprint="sha256:musa",
    )
    assert find_engine_ir_provider("fused_add_rms_norm", "disabled") is None
    assert find_engine_ir_provider("unknown", "native") is None


def test_example_catalog_projects_only_operations_with_native_fallback(monkeypatch):
    import vllm_musa.engine_plugins as host_api
    from vllm_musa.engine_plan.cli import _runtime_catalog

    monkeypatch.setattr(
        host_api,
        "list_engine_ir_providers",
        lambda: (
            EngineIrProviderMetadata(
                operation="fused_add_rms_norm",
                provider="musa",
                implementation_fingerprint="sha256:musa",
            ),
            EngineIrProviderMetadata(
                operation="fused_add_rms_norm",
                provider="native",
                implementation_fingerprint="sha256:native",
            ),
            EngineIrProviderMetadata(
                operation="no_native",
                provider="musa",
                implementation_fingerprint="sha256:other",
            ),
        ),
    )

    catalog = _runtime_catalog()

    assert {item["id"] for item in catalog} == {
        "vllm.ir.fused_add_rms_norm:musa",
        "vllm.ir.fused_add_rms_norm:native",
    }


def test_external_plan_package_does_not_reach_into_vllm_ir_internals():
    source_root = ROOT / "vllm_musa" / "engine_plan"
    source = "\n".join(
        (source_root / filename).read_text(encoding="utf-8")
        for filename in ("cli.py", "runtime.py")
    )

    assert "from vllm.ir.op import IrOp" not in source
    assert "IrOp.registry" not in source
    assert ".impls" not in source
