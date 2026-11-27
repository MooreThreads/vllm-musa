# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_musa.optimization_contract import (
    bind_quantized_piecewise_native_route,
    should_route_quantized_piecewise_ops_native,
    should_use_native_quantized_piecewise_op,
)
from vllm_musa.utils.environ import envs


def _cg_mode(piecewise: bool) -> SimpleNamespace:
    return SimpleNamespace(has_piecewise_cudagraphs=lambda: piecewise)


def test_should_route_quantized_piecewise_ops_native_filters_non_fp8() -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False, quantization=None),
        compilation_config=SimpleNamespace(
            mode=SimpleNamespace(name="VLLM_COMPILE"),
            cudagraph_mode=_cg_mode(True),
        ),
        quant_config=None,
    )
    assert not should_route_quantized_piecewise_ops_native(config)

    # Opaque quant_config fails closed (True)
    config.quant_config = object()
    assert should_route_quantized_piecewise_ops_native(config)

    # Known non-FP8 schemes keep MUSA custom ops
    for non_fp8 in (
        "awq",
        "awq_marlin",
        "gptq",
        "gptq_marlin",
        "marlin",
        "w8a8_int8",
        "bitsandbytes",
        "gguf",
    ):
        config.quant_config = SimpleNamespace(get_name=lambda n=non_fp8: n)
        assert not should_route_quantized_piecewise_ops_native(config)

    # FP8 / multi-format quant configs keep piecewise native safety route
    for fp8_like in ("fp8", "compressed-tensors", "modelopt", "quark", "ptpc_fp8"):
        config.quant_config = SimpleNamespace(get_name=lambda n=fp8_like: n)
        assert should_route_quantized_piecewise_ops_native(config)

    # Non-piecewise CUDAGraph mode, CompilationMode.NONE, or eager mode keeps MUSA ops
    config.compilation_config.cudagraph_mode = _cg_mode(False)
    assert not should_route_quantized_piecewise_ops_native(config)
    config.compilation_config.cudagraph_mode = _cg_mode(True)
    config.compilation_config.mode = SimpleNamespace(name="NONE")
    assert not should_route_quantized_piecewise_ops_native(config)


def test_config_scoped_binding_preserves_env_and_override_restores_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.name, raising=False)
    fp8_cfg = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=SimpleNamespace(
            mode=SimpleNamespace(name="VLLM_COMPILE"),
            cudagraph_mode=_cg_mode(True),
        ),
        quant_config=SimpleNamespace(get_name=lambda: "fp8"),
    )
    bf16_cfg = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=SimpleNamespace(
            mode=SimpleNamespace(name="VLLM_COMPILE"),
            cudagraph_mode=_cg_mode(True),
        ),
        quant_config=None,
    )

    fp8_layer = SimpleNamespace()
    bf16_layer = SimpleNamespace()
    assert bind_quantized_piecewise_native_route(fp8_layer, fp8_cfg) is True
    assert bind_quantized_piecewise_native_route(bf16_layer, bf16_cfg) is False

    assert not envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.is_set()
    assert not envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get()
    assert should_use_native_quantized_piecewise_op(fp8_layer) is True
    assert should_use_native_quantized_piecewise_op(bf16_layer) is False

    with envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.override(False):
        assert should_use_native_quantized_piecewise_op(fp8_layer) is False
    with pytest.raises(ValueError, match="expected failure"):
        with envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.override(True):
            assert should_use_native_quantized_piecewise_op(bf16_layer) is True
            raise ValueError("expected failure")
    assert not envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.is_set()
    assert should_use_native_quantized_piecewise_op(bf16_layer) is False
