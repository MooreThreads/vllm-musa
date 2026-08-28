# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S5000 validation for dense SwiGLU plus FP8 DeepGEMM fusion."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest

pytest.importorskip("torchada")
torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")
pytest.importorskip("torch_musa")
auto_functionalized = pytest.importorskip(
    "torch._higher_order_ops.auto_functionalize"
).auto_functionalized


def _logical_graph_targets(graph) -> set[object]:
    targets = {node.target for node in graph.nodes}
    for node in graph.nodes:
        if node.target is auto_functionalized and node.args:
            targets.add(node.args[0])
    return targets


@pytest.fixture(scope="module", autouse=True)
def _musa_device() -> Iterator[None]:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")
    torch.musa.set_device(0)

    from vllm.platforms import current_platform

    import vllm_musa
    from vllm_musa.model_executor.kernels.linear.scaled_mm import deep_gemm

    if not current_platform.is_device_capability((3, 1)):
        pytest.skip("the fused SiLU group-quant kernel requires S5000/mp31")
    vllm_musa.register_custom_ops()
    assert deep_gemm is not None  # Import registers the two DeepGEMM custom ops.
    yield


def _quant_outputs(m: int, hidden: int):
    q = torch.empty((m, hidden), device="musa", dtype=torch.float8_e4m3fn)
    s = torch.empty((m, hidden // 128), device="musa", dtype=torch.float32)
    return q, s


@pytest.mark.parametrize("m", [1, 4, 4104])
def test_fused_quant_is_bit_exact_for_qwen_shapes(m: int) -> None:
    hidden = 12288
    torch.manual_seed(759000 + m)
    x = torch.randn((m, hidden * 2), device="musa", dtype=torch.bfloat16)
    reference_q, reference_s = _quant_outputs(m, hidden)
    fused_q, fused_s = _quant_outputs(m, hidden)

    activated = F.swish_glu(x)
    torch.ops._C_musa_ops.per_token_group_quant_8bit_vec(
        activated,
        reference_q,
        reference_s,
        128,
        1e-10,
        -448.0,
        448.0,
    )
    torch.ops._C_musa_ops.silu_and_mul_per_token_group_fp8_quant(
        x,
        fused_q,
        fused_s,
        128,
        1e-10,
        -448.0,
        448.0,
    )
    torch.musa.synchronize()

    assert torch.equal(
        reference_q.view(torch.uint8).cpu(), fused_q.view(torch.uint8).cpu()
    )
    assert torch.equal(
        reference_s.view(torch.int32).cpu(), fused_s.view(torch.int32).cpu()
    )


def _small_deepgemm_inputs():
    torch.manual_seed(759100)
    x = torch.randn((4, 512), device="musa", dtype=torch.bfloat16)
    weight = torch.randn((256, 256), device="musa", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    weight_scale = torch.ones((2, 2), device="musa", dtype=torch.float32)
    return x, weight, weight_scale


def test_pass_manager_qualname_resolves() -> None:
    from vllm.utils.import_utils import resolve_obj_by_qualname

    from vllm_musa.compilation.passes import MusaPostGradPassManager
    from vllm_musa.platform import MUSAPlatformBase

    resolved = resolve_obj_by_qualname(MUSAPlatformBase.get_pass_manager_cls())
    assert resolved is MusaPostGradPassManager


@pytest.mark.parametrize("is_moe, expected", [(False, True), (True, False)])
def test_dense_model_gate_uses_model_config_api(is_moe: bool, expected: bool) -> None:
    from vllm_musa.compilation.passes.pass_manager import _is_dense_model

    config = SimpleNamespace(model_config=SimpleNamespace(is_model_moe=lambda: is_moe))
    assert _is_dense_model(config) is expected


def _validated_qwen3_config(**overrides):
    hf_config = SimpleNamespace(
        architectures=["Qwen3ForCausalLM"],
        model_type="qwen3",
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
    )
    model_config = SimpleNamespace(
        architectures=["Qwen3ForCausalLM"],
        hf_text_config=hf_config,
        quantization="fp8",
        dtype=torch.bfloat16,
    )
    parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
    )
    config = SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
        speculative_config=None,
    )
    for dotted_name, value in overrides.items():
        owner_name, attribute = dotted_name.split("__", maxsplit=1)
        setattr(getattr(config, owner_name), attribute, value)
    return config


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_config__architectures": ["Qwen3MoeForCausalLM"]},
        {"model_config__quantization": None},
        {"model_config__dtype": torch.float16},
        {"parallel_config__tensor_parallel_size": 2},
        {"parallel_config__pipeline_parallel_size": 2},
    ],
)
def test_validated_qwen3_8b_auto_scope(overrides) -> None:
    from vllm_musa.runtime_plan import policy
    from vllm_musa.runtime_plan.types import RuntimeDecision

    feature = RuntimeDecision.QWEN3_DENSE_FP8_POST_GRAD_FUSIONS
    assert policy.runtime_plan_enabled(_validated_qwen3_config(), feature)
    assert not policy.runtime_plan_enabled(
        _validated_qwen3_config(**overrides), feature
    )


def test_default_auto_scope_is_evaluated_per_config() -> None:
    from vllm_musa.compilation.passes.pass_manager import (
        _silu_deepgemm_fusion_requested,
    )

    validated = _validated_qwen3_config()
    other = _validated_qwen3_config(
        model_config__architectures=["Qwen3ForSequenceClassification"]
    )

    assert _silu_deepgemm_fusion_requested(validated)
    assert not _silu_deepgemm_fusion_requested(other)


def test_custom_op_schema_fake_and_aot_dynamic() -> None:
    x, weight, weight_scale = _small_deepgemm_inputs()
    op = torch.ops.vllm.musa_silu_deepgemm_fp8_op.default

    input_bits = x.view(torch.uint16).cpu().clone()
    weight_bits = weight.view(torch.uint8).cpu().clone()
    weight_scale_bits = weight_scale.view(torch.int32).cpu().clone()

    schema = str(op._schema)
    assert "!" not in schema
    assert schema.endswith("-> Tensor")
    output = op(x, weight, weight_scale, 128, False)
    assert output.shape == (x.shape[0], weight.shape[0])
    assert output.dtype == torch.bfloat16
    assert output.device == x.device
    assert output.untyped_storage().data_ptr() not in {
        tensor.untyped_storage().data_ptr() for tensor in (x, weight, weight_scale)
    }
    assert torch.equal(input_bits, x.view(torch.uint16).cpu())
    assert torch.equal(weight_bits, weight.view(torch.uint8).cpu())
    assert torch.equal(weight_scale_bits, weight_scale.view(torch.int32).cpu())

    # MUSA's muDNN cannot compare FP8 tensors, so opcheck's schema checker
    # fails while trying to prove that the FP8 weight was not mutated. The
    # raw-byte checks above cover that contract without invoking unsupported
    # FP8 EQ; keep the FakeTensor and AOT-dynamic checks.
    torch.library.opcheck(
        op,
        (x, weight, weight_scale, 128, False),
        test_utils=("test_faketensor", "test_aot_dispatch_dynamic"),
    )


def test_rewrites_the_native_production_graph() -> None:
    from copy import deepcopy

    from torch._inductor.compile_fx import compile_fx
    from vllm.config import (
        CompilationConfig,
        CompilationMode,
        PassConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.model_executor.layers.activation import SiluAndMul

    from vllm_musa.compilation.passes.silu_deepgemm_fusion import (
        MusaSiluDeepGemmFusionPass,
    )

    x, weight, weight_scale = _small_deepgemm_inputs()

    def model(x, weight, weight_scale):
        activated = SiluAndMul.forward_native(x)
        return torch.ops.vllm.musa_deepgemm_fp8_op(
            activated, weight, weight_scale, 128, False
        )

    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            backend="inductor",
            custom_ops=["all"],
            pass_config=PassConfig(fuse_act_quant=True, eliminate_noops=True),
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
            self.targets_before = _logical_graph_targets(graph)
            if self.fusion is not None:
                self.fusion(graph)
            self.targets_after = _logical_graph_targets(graph)

        def __call__(self, graph_module, example_inputs):
            return compile_fx(
                graph_module,
                example_inputs,
                config_patches=self.inductor_config,
            )

    with set_current_vllm_config(config, check_compile=False):
        control_backend = FusionBackend()
        compiled_control = torch.compile(model, backend=control_backend, fullgraph=True)
        expected = compiled_control(x, weight, weight_scale)

        torch._dynamo.reset()
        fusion = MusaSiluDeepGemmFusionPass(config)
        backend = FusionBackend(fusion)
        compiled = torch.compile(model, backend=backend, fullgraph=True)
        actual = compiled(x, weight, weight_scale)

    assert torch.ops.vllm.musa_deepgemm_fp8_op.default in control_backend.targets_after
    assert (
        torch.ops.vllm.musa_silu_deepgemm_fp8_op.default
        not in control_backend.targets_after
    )
    assert torch.ops.vllm.musa_deepgemm_fp8_op.default in backend.targets_before
    assert (
        torch.ops.vllm.musa_silu_deepgemm_fp8_op.default not in backend.targets_before
    )
    assert torch.ops.vllm.musa_deepgemm_fp8_op.default not in backend.targets_after
    assert torch.ops.vllm.musa_silu_deepgemm_fp8_op.default in backend.targets_after
    assert fusion.matched_count == 1

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
