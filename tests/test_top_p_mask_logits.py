# ruff: noqa: I001

import pytest
import torch
import torchada  # noqa: F401

import vllm_musa._custom_ops  # noqa: F401
from vllm_musa._custom_ops import top_k_mask_logits, top_p_mask_logits

VOCAB = 248320


def _musa_available() -> bool:
    return hasattr(torch, "musa") and torch.musa.is_available()


requires_musa = pytest.mark.skipif(
    not _musa_available(), reason="requires a MUSA device and rebuilt extension"
)


def _reference_mask_probs(logits: torch.Tensor, k, p):
    """Sort based top-k then top-p, mirroring apply_top_k_top_p_pytorch.

    Returns the keep mask and the fp32 probabilities the reference compares
    against, so a test can check the resolved mass rather than only the mask.
    """
    x = logits.to(torch.float32)
    if k is not None:
        threshold = x.topk(int(k), dim=-1).values[:, -1:]
        x = x.masked_fill(x < threshold, float("-inf"))
    probs = x.softmax(dim=-1, dtype=torch.float32)
    keep = torch.ones_like(probs, dtype=torch.bool)
    if p is not None:
        sorted_probs = probs.sort(dim=-1, descending=True).values
        cumsum = sorted_probs.cumsum(dim=-1)
        keep_count = ((cumsum - sorted_probs) < p).sum(dim=-1).clamp_min(1)
        index = (keep_count - 1).unsqueeze(1)
        threshold_p = sorted_probs.gather(1, index)
        keep = probs >= threshold_p
    return keep, probs


def _reference(logits: torch.Tensor, k, p) -> torch.Tensor:
    keep, _ = _reference_mask_probs(logits, k, p)
    return logits.to(torch.float32).masked_fill(~keep, float("-inf"))


# The reference walks a sequential fp32 cumsum over the sorted row while the
# kernel accumulates per-bin masses, so the two agree only to ~1e-5 of the row
# total. A row is therefore only allowed to differ when the reference's own
# cumulative lands within that window of `p`, and then only inside the boundary
# band (bounded by the band's mass, never below `p`).
CUMULATIVE_TOL = 1e-4
BAND_TOL = 1e-6


def _assert_parity(keep_ref, probs, actual, p, label="") -> None:
    keep_nat = _keep(actual)
    assert bool(keep_nat.any(dim=-1).all()), f"{label}: native masked a whole row"
    for name, keep in (("reference", keep_ref), ("native", keep_nat)):
        lowest = torch.where(keep, probs, probs.new_full(probs.shape, float("inf")))
        highest = torch.where(~keep, probs, probs.new_full(probs.shape, -float("inf")))
        assert bool(
            (lowest.min(dim=-1).values >= highest.max(dim=-1).values).all()
        ), f"{label}: {name} mask is not a probability threshold"
    mass_ref = (probs * keep_ref).sum(dim=-1)
    mass_nat = (probs * keep_nat).sum(dim=-1)
    differs = (keep_ref != keep_nat).any(dim=-1)
    if p is not None and p < 1.0:
        assert bool(
            (mass_nat >= p - 1e-9).all()
        ), f"{label}: native kept mass {mass_nat.min().item()} below p={p}"
    ambiguous = (mass_ref - p).abs() <= CUMULATIVE_TOL if p is not None else None
    if ambiguous is not None:
        assert bool((~differs | ambiguous).all()), (
            f"{label}: mask differs where the reference cumulative is "
            f"{(mass_ref - p).abs().max().item()} away from p"
        )
    band = torch.where(keep_ref != keep_nat, probs, torch.zeros_like(probs)).sum(dim=-1)
    assert bool((mass_nat - mass_ref <= band + BAND_TOL).all()), (
        f"{label}: native overshot by {(mass_nat - mass_ref).max().item()} "
        f"with a boundary band of {band.max().item()}"
    )
    assert bool(
        (mass_nat >= mass_ref - BAND_TOL).all()
    ), f"{label}: native kept less than the reference"


def _keep(logits: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(logits.to(torch.float32))


@requires_musa
@pytest.mark.parametrize("rows", [1, 16, 64, 256])
@pytest.mark.parametrize("p", [0.9, 0.95, 1.0])
@pytest.mark.parametrize("k", [None, 50])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_top_p_mask_matches_reference(rows, p, k, dtype) -> None:
    torch.manual_seed(9100 + rows + int(p * 1000) + (k or 0))
    logits = (torch.randn((rows, VOCAB), device="musa", dtype=torch.float32) * 2).to(
        dtype
    )
    expected, probs = _reference_mask_probs(logits, k, p)

    actual = top_p_mask_logits(logits.clone(), p, top_k=k)
    assert actual is not None
    torch.musa.synchronize()
    _assert_parity(expected, probs, actual, p, label=f"rows={rows} k={k} p={p}")


@requires_musa
def test_top_p_only_matches_reference() -> None:
    torch.manual_seed(4242)
    rows = 64
    logits = (torch.randn((rows, VOCAB), device="musa", dtype=torch.float32)).to(
        torch.bfloat16
    )
    expected, probs = _reference_mask_probs(logits, None, 0.9)
    actual = top_p_mask_logits(logits.clone(), 0.9, top_k=None)
    assert actual is not None
    torch.musa.synchronize()
    _assert_parity(expected, probs, actual, 0.9, label="p-only")
    kept = int(_keep(actual).sum())
    assert 0 < kept < rows * VOCAB


@requires_musa
def test_per_row_top_p_and_top_k_tensors() -> None:
    torch.manual_seed(777)
    rows = 32
    logits = (torch.randn((rows, VOCAB), device="musa", dtype=torch.float32) * 2).to(
        torch.bfloat16
    )
    k_row = torch.tensor(
        [20, 40, 50, 100, 200, 1024] * 8, device="musa", dtype=torch.int32
    )[:rows]
    p_row = torch.full((rows,), 0.95, device="musa", dtype=torch.float32)
    p_row[::4] = 0.8

    actual = top_p_mask_logits(logits.clone(), p_row, top_k=k_row)
    assert actual is not None
    torch.musa.synchronize()
    keep = _keep(actual)
    for row in range(rows):
        expected, probs = _reference_mask_probs(
            logits[row : row + 1], int(k_row[row]), float(p_row[row])
        )
        _assert_parity(
            expected,
            probs,
            actual[row : row + 1],
            float(p_row[row]),
            label=f"row {row}",
        )
        assert torch.equal(keep[row : row + 1], _keep(actual[row : row + 1]))


@requires_musa
def test_per_row_top_k_matches_scalar_path() -> None:
    torch.manual_seed(555)
    rows = 16
    logits = (torch.randn((rows, VOCAB), device="musa", dtype=torch.float32)).to(
        torch.bfloat16
    )
    k_row = torch.tensor([20, 40, 50, 100] * 4, device="musa", dtype=torch.int64)

    tensor_mask = top_k_mask_logits(logits.clone(), k_row)
    assert tensor_mask is not None
    torch.musa.synchronize()
    keep = _keep(tensor_mask)
    for row in range(rows):
        scalar = top_k_mask_logits(logits[row : row + 1].clone(), int(k_row[row]))
        assert scalar is not None
        assert torch.equal(keep[row : row + 1], _keep(scalar)), f"row {row}"


@requires_musa
def test_nan_and_negative_infinity_are_safe() -> None:
    logits = torch.randn((2, VOCAB), device="musa", dtype=torch.float32).to(
        torch.bfloat16
    )
    logits[0, :8] = float("nan")
    logits[1, :8] = float("-inf")
    actual = top_p_mask_logits(logits.clone(), 0.9, top_k=50)
    assert actual is not None
    torch.musa.synchronize()
    assert not bool(torch.isnan(actual).any())
    assert bool(_keep(actual[1]).sum() > 0)


@requires_musa
def test_uniform_row_keeps_every_token() -> None:
    """A row of identical logits is one tie group, so top-p keeps all of it."""
    logits = torch.full((2, VOCAB), 1.5, device="musa", dtype=torch.bfloat16)
    head = torch.linspace(2.0, 3.0, 64, device="musa", dtype=torch.bfloat16)
    logits[1, :64] = head
    actual = top_p_mask_logits(logits.clone(), 0.9, top_k=None)
    assert actual is not None
    torch.musa.synchronize()
    keep = _keep(actual)
    assert bool(keep[0].all()), "uniform row lost tokens"
    assert int(keep[1].sum()) >= 64


@requires_musa
def test_gumbel_tokens_match_reference_mask() -> None:
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    rows = 64
    torch.manual_seed(31337)
    logits = (torch.randn((rows, VOCAB), device="musa", dtype=torch.float32) * 2).to(
        torch.bfloat16
    )
    reference = _reference(logits, 50, 0.95).to(torch.bfloat16)
    native = top_p_mask_logits(logits.clone(), 0.95, top_k=50)
    assert native is not None
    torch.musa.synchronize()
    assert torch.equal(_keep(native), _keep(reference))

    mapping = torch.arange(rows, dtype=torch.int32, device="musa")
    temperature = torch.ones(rows, dtype=torch.float32, device="musa")
    pos = torch.arange(rows, dtype=torch.int64, device="musa")
    identical = 0
    for seed in range(10):
        seeds = torch.full((rows,), seed, dtype=torch.int64, device="musa")
        a = gumbel_sample(
            native,
            mapping,
            temperature,
            seeds,
            pos,
            apply_temperature=False,
            is_drafting=False,
            use_fp64=False,
        )
        b = gumbel_sample(
            reference,
            mapping,
            temperature,
            seeds,
            pos,
            apply_temperature=False,
            is_drafting=False,
            use_fp64=False,
        )
        identical += int((a == b).sum())
    assert identical == 10 * rows


@requires_musa
def test_wrapper_declines_unsupported_inputs() -> None:
    big = torch.zeros((4097, 8), device="musa", dtype=torch.bfloat16)
    assert top_p_mask_logits(big, 0.9, top_k=4) is None

    small = torch.zeros((4, VOCAB), device="musa", dtype=torch.bfloat16)
    assert top_p_mask_logits(small, 0.0, top_k=50) is None
    assert top_p_mask_logits(small, 1.5, top_k=50) is None
    assert top_p_mask_logits(small, 0.9, top_k=1025) is None
    assert top_p_mask_logits(small.to(torch.float16), 0.9, top_k=50) is None

    other_vocab = torch.zeros((4, 4096), device="musa", dtype=torch.bfloat16)
    assert top_p_mask_logits(other_vocab, 0.9, top_k=50) is None

    bad_p = torch.full((3,), 0.9, device="musa", dtype=torch.float32)
    assert top_p_mask_logits(small, bad_p, top_k=50) is None
