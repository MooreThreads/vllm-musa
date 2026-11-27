from __future__ import annotations

from importlib import import_module

import pytest

pytest.importorskip("torchada")
torch = pytest.importorskip("torch")

if getattr(torch.version, "musa", None) is None:
    pytest.skip("Requires MUSA", allow_module_level=True)

# Imported only after the module-level skips above: the skip must run before
# the helper module pulls in vLLM.
from tests.test_deepseek_v41_optimization_contract import (  # noqa: E402
    _v41_flash_config,
    _v41_w4afp8_config,
)

requant_module = import_module("vllm_musa.kernels.fp8_block_requant")
requantize_fp8_block32_to_block128 = requant_module.requantize_fp8_block32_to_block128
block_gemv = import_module(
    "vllm_musa.model_executor.kernels.linear.scaled_mm.block_gemv"
)


def _reference(weight: torch.Tensor, scales: torch.Tensor):
    rows, cols = weight.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_cols = (cols + 127) // 128 * 128
    padded_scales = torch.zeros((padded_rows // 32, padded_cols // 32))
    padded_scales[: scales.shape[0], : scales.shape[1]] = scales
    scales128 = padded_scales.view(padded_rows // 128, 4, padded_cols // 128, 4)
    scales128 = scales128.amax(dim=(1, 3))
    scales128 = torch.where(scales128 > 0, scales128, 1.0)
    original_scales = scales.repeat_interleave(32, 0).repeat_interleave(32, 1)
    target_scales = scales128.repeat_interleave(128, 0).repeat_interleave(128, 1)
    output = torch.zeros((rows, padded_cols))
    output[:, :cols] = weight.float() * (
        original_scales[:rows, :cols] / target_scales[:rows, :cols]
    )
    return output.to(weight.dtype), scales128


@pytest.mark.parametrize(
    "shape", [(64, 32), (65, 129), (128, 128), (576, 5120), (5120, 288)]
)
@pytest.mark.parametrize("zero_scales", [False, True])
def test_requant_matches_reference_and_initializes_padding(shape, zero_scales) -> None:
    generator = torch.Generator().manual_seed(3720)
    weight = (torch.randn(shape, generator=generator) * 16).to(torch.float8_e4m3fn)
    scales = (
        2.0
        ** torch.randint(
            -6, 3, ((shape[0] + 31) // 32, (shape[1] + 31) // 32), generator=generator
        ).float()
    )
    if zero_scales:
        scales.zero_()
    expected_weight, expected_scales = _reference(weight, scales)
    actual_weight, actual_scales = requantize_fp8_block32_to_block128(
        weight.to("musa"), scales.to("musa")
    )
    assert torch.equal(
        actual_weight.cpu().view(torch.uint8), expected_weight.view(torch.uint8)
    )
    torch.testing.assert_close(actual_scales.cpu(), expected_scales, rtol=0, atol=0)
    assert torch.isfinite(actual_weight.float()).all()


def test_requant_accepts_strided_scales_and_weights() -> None:
    weight = torch.randn((256, 256), device="musa").to(torch.float8_e4m3fn).t()
    scales = torch.rand((8, 8), device="musa").t() + 0.1
    expected_weight, expected_scales = _reference(weight.cpu(), scales.cpu())
    actual_weight, actual_scales = requantize_fp8_block32_to_block128(weight, scales)
    assert torch.equal(
        actual_weight.cpu().view(torch.uint8), expected_weight.view(torch.uint8)
    )
    torch.testing.assert_close(actual_scales.cpu(), expected_scales, rtol=0, atol=0)


def test_requant_rejects_expert_weights_and_wrong_scale_layout() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        requantize_fp8_block32_to_block128(
            torch.empty((2, 128, 128), dtype=torch.float8_e4m3fn, device="musa"),
            torch.ones((2, 4, 4), device="musa"),
        )
    with pytest.raises(ValueError, match="32x32"):
        requantize_fp8_block32_to_block128(
            torch.empty((128, 128), dtype=torch.float8_e4m3fn, device="musa"),
            torch.ones((1, 1), device="musa"),
        )


@pytest.mark.parametrize("shape", [(576, 5120), (4096, 1280), (5120, 288)])
@pytest.mark.parametrize("rows", [1, 4, 16, 512])
def test_mudnn_gemm_with_requantized_weight_matches_dense_reference(
    shape, rows
) -> None:
    from mate.deep_gemm import fp8_gemm_nt

    from vllm_musa.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    torch.manual_seed(3720)
    weight = torch.randn(shape, device="musa").to(torch.float8_e4m3fn)
    scales = torch.ones(((shape[0] + 31) // 32, (shape[1] + 31) // 32), device="musa")
    weight128, scales128 = requantize_fp8_block32_to_block128(weight, scales)
    inputs = torch.randn((rows, shape[1]), dtype=torch.bfloat16, device="musa")
    padded = torch.nn.functional.pad(inputs, (0, weight128.shape[1] - shape[1]))
    quantized, input_scales = per_token_group_quant_fp8(
        padded, 128, use_ue8m0=False, column_major_scales=False
    )
    output = torch.empty((rows, shape[0]), dtype=torch.bfloat16, device="musa")
    fp8_gemm_nt(
        (quantized, input_scales), (weight128, scales128), output, backend="mudnn"
    )
    reference = torch.nn.functional.linear(inputs, weight.to(torch.bfloat16))
    relative_error = (
        output.float() - reference.float()
    ).norm() / reference.float().norm()
    assert torch.isfinite(output).all()
    assert relative_error < 0.05, relative_error.item()


def test_requantized_mudnn_gemm_can_be_captured_and_replayed() -> None:
    from mate.deep_gemm import fp8_gemm_nt

    from vllm_musa.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    weight = torch.randn((128, 128), device="musa").to(torch.float8_e4m3fn)
    weight128, scales128 = requantize_fp8_block32_to_block128(
        weight, torch.ones((4, 4), device="musa")
    )
    inputs = torch.randn((4, 128), dtype=torch.bfloat16, device="musa")
    output = torch.empty((4, 128), dtype=torch.bfloat16, device="musa")

    def forward() -> None:
        quantized, input_scales = per_token_group_quant_fp8(
            inputs, 128, use_ue8m0=False
        )
        fp8_gemm_nt(
            (quantized, input_scales), (weight128, scales128), output, backend="mudnn"
        )

    for iteration in range(3):
        forward()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        forward()
    expected = output.clone()
    inputs.mul_(2)
    graph.replay()
    torch.testing.assert_close(output, expected * 2, rtol=0, atol=0)


def _make_layer_kernel(
    monkeypatch,
    shape,
    prefix="",
    fold_shared=False,
    block128=True,
    is_bmm=False,
    weight_block_size=(32, 32),
    vllm_config=None,
):
    from types import SimpleNamespace

    monkeypatch.setattr(
        block_gemv,
        "get_current_vllm_config",
        lambda: vllm_config if vllm_config is not None else SimpleNamespace(),
    )
    if vllm_config is None:
        monkeypatch.setattr(
            block_gemv,
            "prefers_feature",
            lambda config, feature: (
                block128
                if feature
                == block_gemv.OptimizationFeature.DEEPSEEK_V41_DENSE_FP8_BLOCK128
                else (
                    fold_shared
                    if feature
                    == block_gemv.OptimizationFeature.DEEPSEEK_V41_SHARED_EXPERT_FOLD
                    else True
                )
            ),
        )
    layer = torch.nn.Module()
    layer.weight_block_size = list(weight_block_size)
    layer.prefix = prefix
    if is_bmm:
        layer.is_bmm = True
    original = torch.randn(shape, device="musa").to(torch.float8_e4m3fn)
    scales = (
        torch.rand(((shape[0] + 31) // 32, (shape[1] + 31) // 32), device="musa") + 0.1
    )
    layer.register_parameter(
        "weight", torch.nn.Parameter(original, requires_grad=False)
    )
    layer.register_parameter(
        "weight_scale_inv", torch.nn.Parameter(scales, requires_grad=False)
    )
    kernel = block_gemv.MUSAFp8BlockGemvScaledMMKernel.__new__(
        block_gemv.MUSAFp8BlockGemvScaledMMKernel
    )
    kernel.config = SimpleNamespace(
        out_dtype=torch.bfloat16,
        activation_quant_key=SimpleNamespace(
            scale=SimpleNamespace(group_shape=block_gemv.GroupShape(1, 32))
        ),
    )
    kernel.weight_group_shape = (32, 32)
    kernel.process_weights_after_loading(layer)
    return layer, kernel, original, scales


def test_weight_block_size_defaults_to_block32() -> None:
    kernel = block_gemv.MUSAFp8BlockGemvScaledMMKernel.__new__(
        block_gemv.MUSAFp8BlockGemvScaledMMKernel
    )
    assert kernel._weight_block_size is block_gemv._Fp8WeightBlockSize.BLOCK32


@pytest.mark.parametrize("factory", [_v41_flash_config, _v41_w4afp8_config])
def test_real_model_contract_reaches_load_time_conversion(monkeypatch, factory) -> None:
    layer, kernel, original, scales = _make_layer_kernel(
        monkeypatch, (576, 5120), vllm_config=factory()
    )

    assert layer.weight.data_ptr() != original.data_ptr()
    assert layer.weight_block_size == [128, 128]
    assert kernel._weight_block_size is block_gemv._Fp8WeightBlockSize.BLOCK128


@pytest.mark.parametrize(
    "shape,kwargs",
    [
        ((576, 5120), {"block128": False}),
        ((576, 5120), {"is_bmm": True}),
        ((576, 5120), {"weight_block_size": (64, 32)}),
        ((65, 288), {}),
    ],
)
def test_block128_loader_guards_preserve_ineligible_layers(
    monkeypatch, shape, kwargs
) -> None:
    layer, kernel, original, scales = _make_layer_kernel(monkeypatch, shape, **kwargs)

    assert layer.weight.data_ptr() == original.data_ptr()
    assert layer.weight_scale_inv.data_ptr() == scales.data_ptr()
    assert layer.weight_block_size == list(kwargs.get("weight_block_size", (32, 32)))
    assert kernel._weight_block_size is block_gemv._Fp8WeightBlockSize.BLOCK32


@pytest.mark.parametrize("shape", [(576, 5120), (5120, 288)])
def test_load_time_conversion_keeps_only_block128_weights(monkeypatch, shape) -> None:
    layer, kernel, original, scales = _make_layer_kernel(monkeypatch, shape)
    expected_weight, expected_scales = requantize_fp8_block32_to_block128(
        original, scales
    )
    assert layer.weight.shape == (shape[0], (shape[1] + 127) // 128 * 128)
    assert layer.weight.data_ptr() != original.data_ptr()
    assert layer.weight_block_size == [128, 128]
    assert kernel.weight_group_shape.col == 128
    assert kernel._weight_block_size is block_gemv._Fp8WeightBlockSize.BLOCK128
    assert set(layer.state_dict()) == {"weight", "weight_scale_inv"}
    assert not list(layer.named_buffers())
    assert torch.equal(
        layer.weight.view(torch.uint8), expected_weight.view(torch.uint8)
    )
    torch.testing.assert_close(layer.weight_scale_inv, expected_scales, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape,projection",
    [((576, 5120), "gate_up_proj"), ((5120, 288), "down_proj")],
)
@pytest.mark.parametrize("fold_shared", [False, True])
def test_shared_expert_preserves_original_block32_weights(
    monkeypatch, shape, projection, fold_shared
) -> None:
    prefix = f"model.layers.0.mlp.shared_experts.{projection}"
    layer, kernel, original, scales = _make_layer_kernel(
        monkeypatch, shape, prefix, fold_shared=fold_shared
    )
    assert layer.weight.data_ptr() == original.data_ptr()
    torch.testing.assert_close(layer.weight_scale_inv, scales, rtol=0, atol=0)
    assert layer.weight_block_size == [32, 32]
    assert kernel._weight_block_size is block_gemv._Fp8WeightBlockSize.BLOCK32


def test_dense_projection_with_shared_expert_shape_remains_eligible(
    monkeypatch,
) -> None:
    layer, kernel, original, scales = _make_layer_kernel(
        monkeypatch, (576, 5120), "model.layers.0.self_attn.dense_proj"
    )
    assert layer.weight.data_ptr() != original.data_ptr()
    assert kernel._weight_block_size is block_gemv._Fp8WeightBlockSize.BLOCK128


@pytest.mark.parametrize("shape", [(64, 256)])
@pytest.mark.parametrize("rows", [1, 512])
def test_converted_layer_rejects_legacy_fused_swiglu(monkeypatch, shape, rows) -> None:
    layer, kernel, original, scales = _make_layer_kernel(monkeypatch, shape)
    gate_up = torch.randn((rows, 2 * shape[1]), dtype=torch.bfloat16, device="musa")
    assert kernel._weight_block_size is block_gemv._Fp8WeightBlockSize.BLOCK128

    def legacy_path(*args, **kwargs):
        pytest.fail("Converted dense layer must not use a block32 SwiGLU path")

    monkeypatch.setattr(block_gemv, "fp8_block_gemv_quant_supported", legacy_path)
    assert kernel.apply_swiglu_clamp_weights(layer, gate_up, 7.0) is None


@pytest.mark.parametrize("rows", [0, 1, 4, 16, 512])
def test_converted_layer_uses_mudnn_for_all_rows(monkeypatch, rows) -> None:
    layer, kernel, original, scales = _make_layer_kernel(monkeypatch, (64, 288))
    inputs = torch.randn((2, rows, 288), device="musa", dtype=torch.bfloat16)
    bias = torch.ones((64,), device="musa", dtype=torch.bfloat16)
    calls = []

    def mudnn_forward(activations, weight, weight_scales, out_dtype):
        calls.append((activations.shape, weight, weight_scales))
        return torch.zeros(
            (activations.shape[0], weight.shape[0]),
            dtype=out_dtype,
            device=activations.device,
        )

    def legacy_path(*args, **kwargs):
        pytest.fail("Converted dense layer must not use a block32 fallback")

    monkeypatch.setattr(block_gemv, "fp8_block128_bf16_linear", mudnn_forward)
    monkeypatch.setattr(block_gemv, "_small_m_path", legacy_path)
    output = kernel.apply_weights(layer, inputs, bias=bias)
    assert len(calls) == 1
    assert calls[0][0] == (2 * rows, 288)
    assert calls[0][1] is layer.weight
    assert calls[0][2] is layer.weight_scale_inv
    torch.testing.assert_close(output, torch.ones_like(output))
    assert output.shape == (2, rows, 64)


@pytest.mark.parametrize("shape", [(576, 5120), (5120, 288)])
@pytest.mark.parametrize("rows", [1, 4, 16, 512])
def test_converted_bf16_linear_matches_original_weight_reference(
    monkeypatch, shape, rows
) -> None:
    layer, kernel, original, scales = _make_layer_kernel(monkeypatch, shape)
    inputs = torch.randn((rows, shape[1]), dtype=torch.bfloat16, device="musa")
    output = kernel.apply_weights(layer, inputs)
    restored_weight = original.float() * scales.repeat_interleave(
        32, 0
    ).repeat_interleave(32, 1)
    reference = inputs.float() @ restored_weight[: shape[0], : shape[1]].t()
    assert torch.isfinite(output).all()
    assert (output.float() - reference).norm() / reference.norm() < 0.05


def test_block128_bf16_helper_explicitly_selects_mudnn(monkeypatch) -> None:
    import mate.deep_gemm as deep_gemm

    inputs = torch.randn((1, 288), dtype=torch.bfloat16, device="musa")
    weight = torch.zeros((64, 384), dtype=torch.float8_e4m3fn, device="musa")
    scales = torch.ones((1, 3), device="musa")
    backends = []

    def gemm(activations, weights, output, *, backend):
        backends.append(backend)
        assert activations[0].shape == (1, 384)
        assert weights[0] is weight
        output.zero_()

    monkeypatch.setattr(deep_gemm, "fp8_gemm_nt", gemm)
    result = requant_module.fp8_block128_bf16_linear(
        inputs, weight, scales, torch.bfloat16
    )
    assert backends == ["mudnn"]
    assert result.shape == (1, 64)
