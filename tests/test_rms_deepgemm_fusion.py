# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S5000 validation for residual RMSNorm plus FP8 DeepGEMM fusion."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import pytest

pytest.importorskip("torchada")
torch = pytest.importorskip("torch")
pytest.importorskip("torch_musa")

HIDDEN = 4096
GROUP_SIZE = 128
EPSILON = 1e-6


@pytest.fixture(scope="module", autouse=True)
def _musa_device() -> Iterator[None]:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")
    torch.musa.set_device(0)

    import vllm_musa
    from vllm_musa.model_executor.kernels.linear.scaled_mm import deep_gemm

    if tuple(torch.musa.get_device_capability(0)) != (3, 1):
        pytest.skip("the fused RMSNorm group-quant kernel requires S5000/mp31")
    vllm_musa.register_custom_ops()
    assert deep_gemm is not None
    yield


def _inputs(m: int = 4, n: int = 128):
    torch.manual_seed(760400 + m + n)
    x = torch.randn((m, HIDDEN), device="musa", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    norm_weight = torch.randn(HIDDEN, device="musa", dtype=torch.bfloat16)
    weight = torch.randn((n, HIDDEN), device="musa", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    weight_scale = torch.full(
        (n // GROUP_SIZE, HIDDEN // GROUP_SIZE),
        0.02,
        device="musa",
        dtype=torch.float32,
    )
    return x, residual, norm_weight, weight, weight_scale


def _reference_quant(x, residual, norm_weight):
    summed = x.float() + residual.float()
    residual_out = summed.to(torch.bfloat16)
    inv_rms = torch.rsqrt(summed.square().mean(dim=-1, keepdim=True) + EPSILON)
    # This is the exact captured-Inductor lowering, not eager IR execution.
    normalized = (summed * inv_rms * norm_weight.float()).to(x.dtype)
    q = torch.empty_like(normalized, dtype=torch.float8_e4m3fn)
    scale = torch.empty(
        (x.shape[0], HIDDEN // GROUP_SIZE), device="musa", dtype=torch.float32
    )
    torch.ops._C_musa_ops.per_token_group_quant_8bit_vec(
        normalized, q, scale, GROUP_SIZE, 1e-10, -448.0, 448.0
    )
    return q, scale, residual_out


@pytest.mark.parametrize("m", [1, 4, 16])
def test_native_fused_quant_is_bit_exact(m: int) -> None:
    x, residual, norm_weight, _, _ = _inputs(m=m)
    reference_q, reference_scale, reference_residual = _reference_quant(
        x, residual, norm_weight
    )
    actual_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    actual_scale = torch.empty_like(reference_scale)
    actual_residual = torch.empty_like(x)

    torch.ops._C_musa_ops.fused_add_rms_norm_per_token_group_fp8_quant(
        x,
        residual,
        norm_weight,
        actual_residual,
        actual_q,
        actual_scale,
        EPSILON,
    )
    torch.musa.synchronize()

    assert torch.equal(
        reference_q.view(torch.uint8).cpu(), actual_q.view(torch.uint8).cpu()
    )
    assert torch.equal(
        reference_scale.view(torch.int32).cpu(), actual_scale.view(torch.int32).cpu()
    )
    assert torch.equal(
        reference_residual.view(torch.uint16).cpu(),
        actual_residual.view(torch.uint16).cpu(),
    )


def test_custom_op_schema_fake_and_aot_dynamic() -> None:
    x, residual, norm_weight, weight, weight_scale = _inputs()
    op = torch.ops.vllm.musa_fused_add_rms_deepgemm_fp8_op.default

    schema = str(op._schema)
    assert "!" not in schema
    assert schema.endswith("-> (Tensor, Tensor)")
    output, residual_out = op(
        x, residual, norm_weight, weight, weight_scale, GROUP_SIZE, False, EPSILON
    )
    assert output.shape == (x.shape[0], weight.shape[0])
    assert output.dtype == torch.bfloat16
    assert residual_out.shape == x.shape
    assert residual_out.dtype == torch.bfloat16
    assert output.untyped_storage().data_ptr() not in {
        tensor.untyped_storage().data_ptr()
        for tensor in (x, residual, norm_weight, weight, weight_scale)
    }
    assert residual_out.untyped_storage().data_ptr() not in {
        tensor.untyped_storage().data_ptr()
        for tensor in (x, residual, norm_weight, weight, weight_scale)
    }

    torch.library.opcheck(
        op,
        (
            x,
            residual,
            norm_weight,
            weight,
            weight_scale,
            GROUP_SIZE,
            False,
            EPSILON,
        ),
        test_utils=("test_faketensor", "test_aot_dispatch_dynamic"),
    )


def test_default_auto_scope_is_evaluated_per_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_musa.compilation.passes.pass_manager import (
        _rms_deepgemm_fusion_requested,
    )

    validated = object()
    other = object()
    monkeypatch.setattr(
        "vllm_musa.optimization_contract.policy.prefers_feature",
        lambda config, _feature: config is validated,
    )

    assert _rms_deepgemm_fusion_requested(validated)
    assert not _rms_deepgemm_fusion_requested(other)


def test_rms_fusion_has_no_kernel_environment_control() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    pass_manager = (
        root / "vllm_musa" / "compilation" / "passes" / "pass_manager.py"
    ).read_text()
    environ = (root / "vllm_musa" / "utils" / "environ.py").read_text()

    assert "VLLM_MUSA_RMS_DEEPGEMM_FUSION" not in pass_manager
    assert "VLLM_MUSA_RMS_DEEPGEMM_FUSION" not in environ
    assert "VLLM_MUSA_CUSTOM_OP_USE_NATIVE" not in pass_manager


@pytest.mark.parametrize("misaligned_input", [False, True])
def test_unsupported_inputs_preserve_ir_fallback(misaligned_input: bool) -> None:
    hidden = 5120
    n = 128
    m = 4
    torch.manual_seed(7605120 + int(misaligned_input))
    if misaligned_input:
        storage = torch.randn(m * hidden + 1, device="musa", dtype=torch.bfloat16)
        x = storage[1:].view(m, hidden)
        assert x.is_contiguous() and x.data_ptr() % 16 != 0
    else:
        x = torch.randn((m, hidden), device="musa", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    # Gemma/Qwen3.5 passes an effective FP32 gain (learned weight + 1).
    norm_weight = torch.randn(hidden, device="musa", dtype=torch.float32)
    weight = torch.randn((n, hidden), device="musa", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    weight_scale = torch.full(
        (n // GROUP_SIZE, hidden // GROUP_SIZE),
        0.02,
        device="musa",
        dtype=torch.float32,
    )

    summed = x.float() + residual.float()
    normalized = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + EPSILON)
    normalized = (normalized * norm_weight.float()).to(x.dtype)
    expected_residual = summed.to(x.dtype)
    expected_output = torch.ops.vllm.musa_deepgemm_fp8_op(
        normalized, weight, weight_scale, GROUP_SIZE, False
    )
    actual_output, actual_residual = torch.ops.vllm.musa_fused_add_rms_deepgemm_fp8_op(
        x,
        residual,
        norm_weight,
        weight,
        weight_scale,
        GROUP_SIZE,
        False,
        EPSILON,
    )
    torch.musa.synchronize()

    assert actual_output.dtype == x.dtype
    assert actual_residual.dtype == x.dtype
    assert torch.equal(
        actual_output.view(torch.uint16).cpu(),
        expected_output.view(torch.uint16).cpu(),
    )
    assert torch.equal(
        actual_residual.view(torch.uint16).cpu(),
        expected_residual.view(torch.uint16).cpu(),
    )


def test_rewrites_the_native_production_graph() -> None:
    from torch._inductor.compile_fx import compile_fx
    from vllm.config import (
        CompilationConfig,
        CompilationMode,
        PassConfig,
        VllmConfig,
        set_current_vllm_config,
    )

    from vllm_musa.compilation.passes.rms_deepgemm_fusion import (
        MusaRMSDeepGemmFusionPass,
    )

    x, residual, norm_weight, weight, weight_scale = _inputs()

    def model(x, residual, norm_weight, weight, weight_scale):
        normalized, residual_out = torch.ops.vllm_ir.fused_add_rms_norm(
            x, residual, norm_weight, EPSILON, None
        )
        output = torch.ops.vllm.musa_deepgemm_fp8_op(
            normalized, weight, weight_scale, GROUP_SIZE, False
        )
        return output, residual_out

    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            backend="inductor",
            custom_ops=["all"],
            pass_config=PassConfig(fuse_norm_quant=True, eliminate_noops=True),
        )
    )

    class FusionBackend:
        def __init__(self, fusion=None):
            self.fusion = fusion
            self.targets_before = set()
            self.targets_after = set()
            self.inductor_config = deepcopy(
                config.compilation_config.inductor_compile_config
            )
            self.inductor_config["force_disable_caches"] = True
            self.inductor_config["post_grad_custom_post_pass"] = self.post_pass

        def post_pass(self, graph):
            self.targets_before = {node.target for node in graph.nodes}
            if self.fusion is not None:
                self.fusion(graph)
            self.targets_after = {node.target for node in graph.nodes}

        def __call__(self, graph_module, example_inputs):
            return compile_fx(
                graph_module,
                example_inputs,
                config_patches=self.inductor_config,
            )

    with set_current_vllm_config(config, check_compile=False):
        control_backend = FusionBackend()
        compiled_control = torch.compile(model, backend=control_backend, fullgraph=True)
        expected = compiled_control(x, residual, norm_weight, weight, weight_scale)

        torch._dynamo.reset()
        fusion = MusaRMSDeepGemmFusionPass(config)
        backend = FusionBackend(fusion)
        compiled = torch.compile(model, backend=backend, fullgraph=True)
        actual = compiled(x, residual, norm_weight, weight, weight_scale)

    control_op = torch.ops.vllm.musa_deepgemm_fp8_op.default
    fused_op = torch.ops.vllm.musa_fused_add_rms_deepgemm_fp8_op.default
    assert control_op in control_backend.targets_after
    assert fused_op not in control_backend.targets_after
    assert control_op in backend.targets_before
    assert fused_op not in backend.targets_before
    assert control_op not in backend.targets_after
    assert fused_op in backend.targets_after
    assert fusion.matched_count == 1

    # DeepGEMM may choose a different but numerically equivalent launch when
    # called behind the opaque fused op. The fused quantizer itself is checked
    # bit-for-bit above; use the FP8 GEMM's bounded BF16 tolerance here.
    torch.testing.assert_close(actual[0], expected[0], atol=0.125, rtol=0.1)
    assert torch.equal(
        actual[1].view(torch.uint16).cpu(), expected[1].view(torch.uint16).cpu()
    )
