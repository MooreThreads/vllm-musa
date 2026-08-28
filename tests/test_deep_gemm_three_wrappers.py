# SPDX-License-Identifier: Apache-2.0
"""Provider-boundary contracts for the three grouped DeepGEMM wrappers."""

from __future__ import annotations

import inspect
from typing import Any

import pytest
import torchada  # noqa: F401
import torch

from vllm.utils import deep_gemm


GROUPED_WRAPPERS = (
    ("m_grouped_fp8_gemm_nt_contiguous", "_grouped_impl"),
    ("m_grouped_fp8_fp4_gemm_nt_contiguous", "_grouped_fp4_impl"),
    ("fp8_m_grouped_gemm_nt_masked", "_grouped_masked_impl"),
)


@pytest.mark.parametrize(("name", "impl_name"), GROUPED_WRAPPERS)
def test_grouped_wrappers_omit_ue8m0_on_musa(
    monkeypatch: pytest.MonkeyPatch, name: str, impl_name: str
) -> None:
    received: dict[str, Any] = {}

    def implementation(*args: Any, **kwargs: Any) -> str:
        received["args"] = args
        received["kwargs"] = kwargs
        return "provider-result"

    monkeypatch.setattr(deep_gemm, "_lazy_init", lambda: None)
    monkeypatch.setattr(deep_gemm, impl_name, implementation)
    monkeypatch.setattr(deep_gemm.current_platform, "is_musa", lambda: True)

    assert getattr(deep_gemm, name)("operand", caller_keyword="value") == "provider-result"
    assert received == {
        "args": ("operand",),
        "kwargs": {"caller_keyword": "value"},
    }


@pytest.mark.parametrize(("name", "impl_name"), GROUPED_WRAPPERS)
def test_grouped_wrappers_preserve_ue8m0_off_musa(
    monkeypatch: pytest.MonkeyPatch, name: str, impl_name: str
) -> None:
    received: dict[str, Any] = {}

    def implementation(*args: Any, **kwargs: Any) -> str:
        received["kwargs"] = kwargs
        return "provider-result"

    monkeypatch.setattr(deep_gemm, "_lazy_init", lambda: None)
    monkeypatch.setattr(deep_gemm, impl_name, implementation)
    monkeypatch.setattr(deep_gemm.current_platform, "is_musa", lambda: False)
    monkeypatch.setattr(deep_gemm, "is_deep_gemm_e8m0_used", lambda: False)

    assert getattr(deep_gemm, name)("operand") == "provider-result"
    assert received["kwargs"]["disable_ue8m0_cast"] is True


def test_non_grouped_fp8_gemm_keeps_ue8m0_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, Any] = {}

    def implementation(*args: Any, **kwargs: Any) -> str:
        received["kwargs"] = kwargs
        return "provider-result"

    monkeypatch.setattr(deep_gemm, "_lazy_init", lambda: None)
    monkeypatch.setattr(deep_gemm, "_fp8_gemm_nt_impl", implementation)
    monkeypatch.setattr(deep_gemm, "is_deep_gemm_e8m0_used", lambda: False)

    assert deep_gemm.fp8_gemm_nt("operand") == "provider-result"
    assert received["kwargs"]["disable_ue8m0_cast"] is True


@pytest.mark.parametrize(("name", "impl_name"), GROUPED_WRAPPERS)
def test_missing_grouped_provider_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch, name: str, impl_name: str
) -> None:
    monkeypatch.setattr(deep_gemm, "_lazy_init", lambda: None)
    monkeypatch.setattr(deep_gemm, impl_name, None)

    with pytest.raises(RuntimeError, match="backend is unavailable"):
        getattr(deep_gemm, name)("operand")


def test_provider_identity_and_grouped_surface() -> None:
    provider = pytest.importorskip("deep_gemm")
    for name, _ in GROUPED_WRAPPERS:
        assert hasattr(provider, name), f"missing provider wrapper: {name}"
        assert str(inspect.signature(getattr(provider, name)))
    assert provider.__file__
    assert getattr(provider, "__version__", None)


def _musa_available() -> bool:
    return bool(getattr(torch.version, "musa", None)) and hasattr(torch, "musa")


requires_musa = pytest.mark.skipif(
    not _musa_available(), reason="requires a real MUSA runtime"
)


def _random_fp8_operands(experts: int, tokens: int) -> tuple[Any, ...]:
    device = "musa"
    k = n = 128
    activations = torch.randn(experts * tokens, k, device=device).to(torch.float8_e4m3fn)
    activation_scales = torch.ones(experts * tokens, 1, device=device)
    weights = torch.randn(experts, n, k, device=device).to(torch.float8_e4m3fn)
    weight_scales = torch.ones(experts, 1, 1, device=device)
    layout = torch.repeat_interleave(
        torch.arange(experts, device=device, dtype=torch.int32), tokens
    )
    return activations, activation_scales, weights, weight_scales, layout


def _production_groupwise_operands(experts: int, tokens: int) -> tuple[Any, ...]:
    """Build the same `(1,128,128)` FP8/dequant contract used by MATE tests."""
    helpers = pytest.importorskip("mate.testing.utils")
    k = n = 128
    m = experts * tokens
    activations = torch.randn((m, k), device="musa", dtype=torch.float32)
    weights = torch.randn((experts, n, k), device="musa", dtype=torch.float32)
    fp8_a, scale_a = helpers.group_quantize_fp8(
        activations, (m, 1), (1, 128), torch.float8_e4m3fn, "K"
    )
    fp8_b, scale_b = helpers.group_quantize_fp8(
        weights, (experts, 1, 1), (1, 128, 128), torch.float8_e4m3fn, "K"
    )
    dequant_a = helpers.group_dequantize_fp8(fp8_a, scale_a, "K")
    dequant_b = helpers.group_dequantize_fp8(fp8_b, scale_b, "K")
    return fp8_a, scale_a, fp8_b, scale_b, dequant_a, dequant_b


def _assert_production_reference(actual: torch.Tensor, reference: torch.Tensor) -> None:
    actual_f = actual.float()
    reference_f = reference.float()
    assert torch.isfinite(actual_f).all()
    relative_l2 = torch.linalg.vector_norm(actual_f - reference_f) / torch.linalg.vector_norm(reference_f)
    cosine = torch.nn.functional.cosine_similarity(actual_f.flatten(), reference_f.flatten(), dim=0)
    assert relative_l2 < 0.005
    assert cosine > 0.999


@requires_musa
@pytest.mark.parametrize("name", GROUPED_WRAPPERS[:2], ids=lambda item: item[0])
def test_grouped_wrapper_matches_production_dequant_reference(name: tuple[str, str]) -> None:
    """Numerical reference uses actual groupwise scales, not unit-scale FP8."""
    provider = pytest.importorskip("deep_gemm")
    torch.manual_seed(100004)
    experts, tokens = 2, 128
    fp8_a, scale_a, fp8_b, scale_b, dequant_a, dequant_b = _production_groupwise_operands(experts, tokens)
    indices = torch.repeat_interleave(torch.arange(experts, device="musa", dtype=torch.int32), tokens)
    expected = torch.empty((experts * tokens, 128), device="musa", dtype=torch.float32)
    for expert in range(experts):
        rows = indices == expert
        expected[rows] = dequant_a[rows] @ dequant_b[expert].t()
    actual = torch.zeros_like(expected, dtype=torch.bfloat16)
    getattr(provider, name[0])((fp8_a, scale_a), (fp8_b, scale_b), actual, indices, (1, 128, 128))
    torch.musa.synchronize()
    _assert_production_reference(actual, expected)


@requires_musa
def test_masked_wrapper_matches_production_dequant_reference() -> None:
    """Masked active rows match groupwise reference and inactive tails stay zero."""
    provider = pytest.importorskip("deep_gemm")
    torch.manual_seed(100005)
    experts, tokens = 2, 128
    fp8_a, scale_a, fp8_b, scale_b, dequant_a, dequant_b = _production_groupwise_operands(experts, tokens)
    counts = torch.tensor([128, 64], device="musa", dtype=torch.int32)
    actual = torch.zeros((experts, tokens, 128), device="musa", dtype=torch.bfloat16)
    provider.fp8_m_grouped_gemm_nt_masked(
        (fp8_a.reshape(experts, tokens, 128), scale_a.reshape(experts, tokens, 1)),
        (fp8_b, scale_b), actual, counts, tokens, (1, 128, 128)
    )
    expected = torch.cat((
        dequant_a[:128] @ dequant_b[0].t(),
        dequant_a[128:192] @ dequant_b[1].t(),
    ))
    observed = torch.cat((actual[0, :128], actual[1, :64]))
    torch.musa.synchronize()
    _assert_production_reference(observed, expected)
    assert torch.count_nonzero(actual[1, 64:]) == 0


@requires_musa
@pytest.mark.parametrize("name", GROUPED_WRAPPERS[:2], ids=lambda item: item[0])
@pytest.mark.parametrize(
    ("experts", "tokens"), ((2, 1), (8, 129), (8, 512), (256, 1))
)
def test_grouped_wrapper_random_fp8_matrix_is_finite_and_deterministic(
    name: tuple[str, str], experts: int, tokens: int
) -> None:
    """Hardware matrix: random operands, edge token counts, deterministic output."""
    wrapper, _ = name
    torch.manual_seed(100004)
    a, sa, b, sb, layout = _random_fp8_operands(experts, tokens)
    first = torch.zeros(experts * tokens, 128, device="musa", dtype=torch.bfloat16)
    second = torch.zeros_like(first)
    getattr(deep_gemm, wrapper)((a, sa), (b, sb), first, layout)
    torch.musa.synchronize()
    getattr(deep_gemm, wrapper)((a, sa), (b, sb), second, layout)
    torch.musa.synchronize()
    assert torch.isfinite(first.float()).all()
    assert torch.equal(first, second)


@requires_musa
@pytest.mark.parametrize("name", GROUPED_WRAPPERS[:2], ids=lambda item: item[0])
@pytest.mark.parametrize("experts", (2, 8, 128, 256))
@pytest.mark.parametrize("tokens", (1, 2, 8, 32, 64, 127, 128, 129, 256, 512, 2048))
@pytest.mark.parametrize("seed", (100004, 100005, 100006))
def test_grouped_wrapper_full_ticket_boundary_matrix(
    name: tuple[str, str], experts: int, tokens: int, seed: int
) -> None:
    """Required random-FP8 token/expert matrix; two equal calls prove determinism."""
    wrapper, _ = name
    torch.manual_seed(seed)
    a, sa, b, sb, layout = _random_fp8_operands(experts, tokens)
    first = torch.zeros(experts * tokens, 128, device="musa", dtype=torch.bfloat16)
    second = torch.zeros_like(first)
    getattr(deep_gemm, wrapper)((a, sa), (b, sb), first, layout)
    torch.musa.synchronize()
    getattr(deep_gemm, wrapper)((a, sa), (b, sb), second, layout)
    torch.musa.synchronize()
    assert torch.isfinite(first.float()).all()
    assert torch.equal(first, second)


@requires_musa
@pytest.mark.parametrize(
    ("counts", "tokens"),
    (
        ([0] * 8, 129),
        ([129] + [0] * 7, 129),
        ([129, 64, 1, 0, 0, 0, 0, 0], 129),
    ),
    ids=("all-zero", "one-active", "mixed-with-zero-experts"),
)
def test_masked_wrapper_keeps_inactive_tails_zero(
    counts: list[int], tokens: int
) -> None:
    """Hardware boundary matrix for zero-token and padded grouped experts."""
    experts = len(counts)
    torch.manual_seed(100005)
    a, sa, b, sb, _ = _random_fp8_operands(experts, tokens)
    output = torch.zeros(experts, tokens, 128, device="musa", dtype=torch.bfloat16)
    deep_gemm.fp8_m_grouped_gemm_nt_masked(
        (a.reshape(experts, tokens, 128), sa.reshape(experts, tokens, 1)),
        (b, sb),
        output,
        torch.tensor(counts, device="musa", dtype=torch.int32),
        tokens,
    )
    torch.musa.synchronize()
    assert torch.isfinite(output.float()).all()
    for expert, active in enumerate(counts):
        assert torch.count_nonzero(output[expert, active:]) == 0


@requires_musa
def test_masked_wrapper_e256_one_active_boundary() -> None:
    """Keep the E=256 production-expert boundary in the single test file."""
    experts, tokens = 256, 1
    torch.manual_seed(100006)
    a, sa, b, sb, _ = _random_fp8_operands(experts, tokens)
    output = torch.zeros(experts, tokens, 128, device="musa", dtype=torch.bfloat16)
    counts = torch.tensor([1] + [0] * (experts - 1), device="musa", dtype=torch.int32)
    deep_gemm.fp8_m_grouped_gemm_nt_masked(
        (a.reshape(experts, tokens, 128), sa.reshape(experts, tokens, 1)),
        (b, sb),
        output,
        counts,
        tokens,
    )
    torch.musa.synchronize()
    assert torch.isfinite(output.float()).all()
    assert torch.count_nonzero(output[1:]) == 0
