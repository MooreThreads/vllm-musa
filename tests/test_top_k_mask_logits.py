# ruff: noqa: I001

import pytest
import torchada  # noqa: F401
import torch

import vllm_musa._custom_ops  # noqa: F401
from vllm_musa._custom_ops import top_k_mask_logits

VOCAB = 248320


def _musa_available() -> bool:
    return hasattr(torch, "musa") and torch.musa.is_available()


requires_musa = pytest.mark.skipif(
    not _musa_available(), reason="requires a MUSA device and rebuilt extension"
)


def _reference(logits: torch.Tensor, k: int) -> torch.Tensor:
    threshold = logits.topk(k, dim=-1).values[:, -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


@requires_musa
@pytest.mark.parametrize("rows", [1, 4, 16, 64, 128, 256])
@pytest.mark.parametrize("k", [1, 20, 50, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_native_mask_matches_topk(rows: int, k: int, dtype: torch.dtype) -> None:
    torch.manual_seed(7000 + rows + k)
    logits = (torch.randn((rows, VOCAB), device="musa", dtype=torch.float32) * 2).to(
        dtype
    )
    expected = _reference(logits, k)

    actual = top_k_mask_logits(logits.clone(), k)
    assert actual is not None
    torch.musa.synchronize()

    assert torch.equal(torch.isinf(actual), torch.isinf(expected))
    keep = ~torch.isinf(expected)
    assert torch.equal(actual[keep], expected[keep])
    kept = keep.sum(dim=-1)
    assert bool((kept >= k).all())


@requires_musa
def test_tied_rows_stay_eligible() -> None:
    tied = torch.zeros((2, VOCAB), device="musa", dtype=torch.bfloat16)
    expected = _reference(tied, 50)
    actual = top_k_mask_logits(tied.clone(), 50)
    torch.musa.synchronize()
    assert actual is not None
    assert torch.equal(torch.isinf(actual), torch.isinf(expected))


@requires_musa
def test_nan_and_negative_infinity_are_safe() -> None:
    logits = torch.randn((2, VOCAB), device="musa", dtype=torch.float32).to(
        torch.bfloat16
    )
    logits[0, :8] = float("nan")
    logits[1, :8] = float("-inf")
    actual = top_k_mask_logits(logits.clone(), 50)
    torch.musa.synchronize()
    assert actual is not None
    assert not bool(torch.isnan(actual).any())
    assert bool((~torch.isinf(actual[0])).sum() >= 50)


@requires_musa
def test_mask_matches_probability_domain_mask_under_gumbel() -> None:
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    from vllm_musa._custom_ops import top_k_mask_logits_radix

    rows = 64
    torch.manual_seed(31)
    logits = (torch.randn((rows, VOCAB), device="musa", dtype=torch.float32) * 2).to(
        torch.bfloat16
    )
    native = top_k_mask_logits(logits.clone(), 50)
    hybrid = top_k_mask_logits_radix(logits.clone(), 50)
    assert native is not None and hybrid is not None

    mapping = torch.arange(rows, dtype=torch.int32, device="musa")
    temperature = torch.ones(rows, dtype=torch.float32, device="musa")
    pos = torch.arange(rows, dtype=torch.int64, device="musa")
    identical = 0
    for seed in range(10):
        seeds = torch.full((rows,), seed, dtype=torch.int64, device="musa")
        a = gumbel_sample(
            native, mapping, temperature, seeds, pos,
            apply_temperature=False, is_drafting=False, use_fp64=False,
        )
        b = gumbel_sample(
            hybrid, mapping, temperature, seeds, pos,
            apply_temperature=False, is_drafting=False, use_fp64=False,
        )
        identical += int((a == b).sum())
    assert identical == 10 * rows


@requires_musa
def test_wrapper_declines_unsupported_shapes() -> None:
    too_many_rows = torch.zeros((4097, 8), device="musa", dtype=torch.bfloat16)
    assert top_k_mask_logits(too_many_rows, 4) is None

    small = torch.zeros((4, VOCAB), device="musa", dtype=torch.bfloat16)
    assert top_k_mask_logits(small, 1025) is None
    assert top_k_mask_logits(small, VOCAB) is None
    assert top_k_mask_logits(small.to(torch.float16), 50) is None

    other_vocab = torch.zeros((4, 4096), device="musa", dtype=torch.bfloat16)
    assert top_k_mask_logits(other_vocab, 50) is None


@requires_musa
def test_per_row_top_k_tensor_matches_per_row_topk() -> None:
    torch.manual_seed(1009)
    rows = 16
    logits = (torch.randn((rows, VOCAB), device="musa", dtype=torch.float32) * 2).to(
        torch.bfloat16
    )
    k_row = torch.tensor([20, 40, 50, 100, 1024, 1] * 3, device="musa", dtype=torch.int32)[:rows]

    actual = top_k_mask_logits(logits.clone(), k_row)
    assert actual is not None
    torch.musa.synchronize()
    keep = torch.isfinite(actual)
    for row in range(rows):
        expected = _reference(logits[row : row + 1], int(k_row[row]))
        assert torch.equal(keep[row : row + 1], torch.isfinite(expected)), f"row {row}"
