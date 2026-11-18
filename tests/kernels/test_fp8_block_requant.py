from __future__ import annotations

from importlib import import_module

import pytest

pytest.importorskip("torchada")
torch = pytest.importorskip("torch")

if getattr(torch.version, "musa", None) is None:
    pytest.skip("Requires MUSA", allow_module_level=True)

requant_module = import_module("vllm_musa.kernels.fp8_block_requant")
requantize_fp8_block32_to_block128 = requant_module.requantize_fp8_block32_to_block128


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


def _make_layer_kernel(monkeypatch, shape, prefix="", fold_shared=False):
    from types import SimpleNamespace

    from vllm_musa.model_executor.kernels.linear.scaled_mm import block_gemv

    monkeypatch.setattr(
        block_gemv, "get_current_vllm_config", lambda: SimpleNamespace()
    )
    monkeypatch.setattr(
        block_gemv,
        "prefers_feature",
        lambda config, feature: (
            feature != block_gemv.OptimizationFeature.DEEPSEEK_V41_SHARED_EXPERT_FOLD
            or fold_shared
        ),
    )
    layer = torch.nn.Module()
    layer.weight_block_size = [32, 32]
    layer.prefix = prefix
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
    kernel.config = SimpleNamespace(out_dtype=torch.bfloat16)
    kernel.weight_group_shape = (32, 32)
    kernel.process_weights_after_loading(layer)
    return layer, kernel, original, scales


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
    assert kernel._musa_block128 is True
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
    assert not hasattr(layer, "musa_weight_block128")
    assert not hasattr(kernel, "_musa_block128")


def test_dense_projection_with_shared_expert_shape_remains_eligible(
    monkeypatch,
) -> None:
    layer, kernel, original, scales = _make_layer_kernel(
        monkeypatch, (576, 5120), "model.layers.0.self_attn.dense_proj"
    )
    assert layer.weight.data_ptr() != original.data_ptr()
    assert kernel._musa_block128 is True


@pytest.mark.parametrize("rows", [0, 1, 4, 16, 512])
def test_converted_layer_uses_mudnn_for_all_rows(monkeypatch, rows) -> None:
    from vllm_musa.model_executor.kernels.linear.scaled_mm import block_gemv

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


@pytest.mark.parametrize("prequantized", [False, True])
def test_block128_helpers_explicitly_select_mudnn(monkeypatch, prequantized) -> None:
    import mate.deep_gemm as deep_gemm

    from vllm_musa.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

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
    if prequantized:
        quantized, input_scales = per_token_group_quant_fp8(
            inputs, 32, use_ue8m0=False, column_major_scales=False
        )
        result = requant_module.fp8_block128_prequantized_linear(
            quantized, input_scales, weight, scales, torch.bfloat16
        )
    else:
        result = requant_module.fp8_block128_bf16_linear(
            inputs, weight, scales, torch.bfloat16
        )
    assert backends == ["mudnn"]
    assert result.shape == (1, 64)


@pytest.mark.parametrize("shape", [(1, 32), (4, 128), (16, 288), (512, 1280)])
@pytest.mark.parametrize("zero_scales", [False, True])
def test_activation_requant_has_exact_power_of_two_reference(
    shape, zero_scales
) -> None:
    rows, cols = shape
    torch.manual_seed(3720)
    inputs = torch.randn(shape).to(torch.float8_e4m3fn)
    scales = 2.0 ** torch.randint(-6, 3, (rows, (cols + 31) // 32)).float()
    if zero_scales:
        scales.zero_()
    groups128 = (cols + 127) // 128
    padded_scales = torch.zeros((rows, groups128 * 4))
    padded_scales[:, : scales.shape[1]] = scales
    expected_scales = padded_scales.view(rows, groups128, 4).amax(-1)
    expected_scales = torch.where(expected_scales > 0, expected_scales, 1.0)
    expanded_target = expected_scales.repeat_interleave(128, 1)
    expected = torch.zeros((rows, groups128 * 128))
    expected[:, :cols] = inputs.float() * (
        scales.repeat_interleave(32, 1)[:, :cols] / expanded_target[:, :cols]
    )
    actual, actual_scales = requant_module.requantize_fp8_activation32_to128(
        inputs.to("musa"), scales.to("musa")
    )
    assert torch.equal(
        actual.cpu().view(torch.uint8),
        expected.to(torch.float8_e4m3fn).view(torch.uint8),
    )
    torch.testing.assert_close(actual_scales.cpu(), expected_scales, rtol=0, atol=0)


def test_activation_requant_accepts_strides_and_preserves_group128_input() -> None:
    inputs = torch.randn((8, 256), device="musa").to(torch.float8_e4m3fn)[:, ::2]
    scales = torch.rand((4, 8), device="musa").t() + 0.1
    actual, actual_scales = requant_module.requantize_fp8_activation32_to128(
        inputs, scales
    )
    assert actual.shape == (8, 128)
    assert torch.isfinite(actual.float()).all()
    expected = inputs.float() * scales.repeat_interleave(32, 1)
    restored = actual.float() * actual_scales.repeat_interleave(128, 1)
    assert (restored - expected).norm() / expected.norm() < 0.04
    same, same_scales = requant_module.requantize_fp8_activation32_to128(
        actual, actual_scales
    )
    assert same is actual and same_scales is actual_scales


@pytest.mark.parametrize("rows", [4, 512])
@pytest.mark.parametrize("shape", [(4096, 1280), (5120, 288)])
def test_prequantized_linear_dispatch_matches_block32_reference(
    monkeypatch, shape, rows
) -> None:
    from vllm_musa.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    layer, kernel, original, scales = _make_layer_kernel(monkeypatch, shape)
    inputs = torch.randn((rows, shape[1]), dtype=torch.bfloat16, device="musa")
    quantized, input_scales = per_token_group_quant_fp8(
        inputs, 32, use_ue8m0=False, column_major_scales=False
    )
    output = kernel.apply_block_scaled_mm(
        quantized, layer.weight, input_scales, layer.weight_scale_inv
    )
    restored_inputs = quantized.float() * input_scales.repeat_interleave(32, 1)
    restored_weight = original.float() * scales.repeat_interleave(
        32, 0
    ).repeat_interleave(32, 1)
    reference = restored_inputs @ restored_weight[: shape[0], : shape[1]].t()
    assert torch.isfinite(output).all()
    assert (output.float() - reference).norm() / reference.norm() < 0.05
    if shape == (4096, 1280) and rows == 512:
        alternate = (-layer.weight.float()).to(layer.weight.dtype)
        alternate_output = kernel.apply_block_scaled_mm(
            quantized, alternate, input_scales, layer.weight_scale_inv
        )
        assert (alternate_output.float() + reference).norm() / reference.norm() < 0.05


def test_prequantized_consumer_can_be_captured_and_replayed(monkeypatch) -> None:
    from vllm_musa.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    layer, kernel, original, scales = _make_layer_kernel(monkeypatch, (576, 5120))
    inputs = torch.randn((512, 5120), dtype=torch.bfloat16, device="musa")
    quantized, input_scales = per_token_group_quant_fp8(
        inputs, 32, use_ue8m0=False, column_major_scales=False
    )

    def forward():
        return kernel.apply_block_scaled_mm(
            quantized, layer.weight, input_scales, layer.weight_scale_inv
        )

    for iteration in range(3):
        forward()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        output = forward()
    graph.replay()
    expected = output.clone()
    input_scales.mul_(2)
    graph.replay()
    torch.testing.assert_close(output, expected * 2, rtol=0, atol=0)
