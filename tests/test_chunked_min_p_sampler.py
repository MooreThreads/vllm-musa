# SPDX-License-Identifier: Apache-2.0
"""S5000 contract checks for the chunked Qwen min-p sampler."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

pytest.importorskip("torchada")
torch = pytest.importorskip("torch")
pytest.importorskip("torch_musa")

QWEN3_VOCAB = 151936
QWEN35_VOCAB = 248320
SUPPORTED_VOCABS = (QWEN3_VOCAB, QWEN35_VOCAB)


@pytest.fixture(scope="module", autouse=True)
def _musa_device() -> Iterator[None]:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")
    torch.musa.set_device(0)

    from vllm.platforms import current_platform

    import vllm_musa

    if not current_platform.is_device_capability((3, 1)):
        pytest.skip("the chunked min-p sampler requires S5000/mp31")
    vllm_musa.register_all()
    yield


def _probs(batch: int, vocab: int = QWEN3_VOCAB) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(763000 + batch + vocab)
    logits = torch.randn((batch, vocab), generator=generator, dtype=torch.float32)
    return torch.softmax(logits.mul_(1.5), dim=-1).contiguous().to("musa")


def _candidate(
    probs: torch.Tensor,
    min_p: float | torch.Tensor,
    *,
    indices: torch.Tensor | None = None,
    seed: int = 763,
) -> torch.Tensor:
    output = torch.empty(probs.size(0), dtype=torch.int32, device="musa")
    generator = torch.Generator(device="musa").manual_seed(seed)
    min_p_arr = min_p if isinstance(min_p, torch.Tensor) else None
    min_p_val = 0.0 if min_p_arr is not None else min_p
    torch.ops._C_musa_ops.musa_chunked_min_p_sampling_from_probs(
        probs,
        output,
        indices,
        min_p_arr,
        min_p_val,
        True,
        generator,
    )
    return output


def _assert_support(
    probs: torch.Tensor,
    samples: torch.Tensor,
    min_p: float | torch.Tensor,
    indices: torch.Tensor | None = None,
) -> None:
    rows = (
        torch.arange(probs.size(0), dtype=torch.int32, device="musa")
        if indices is None
        else indices
    )
    row_probs = probs[rows.long()]
    chosen = row_probs.gather(1, samples.long()[:, None]).view(-1)
    threshold = row_probs.amax(dim=1) * min_p
    assert bool((chosen + 1.0e-12 >= threshold).all().item())


@pytest.mark.parametrize("vocab", SUPPORTED_VOCABS)
@pytest.mark.parametrize("batch", [1, 4, 16, 64])
def test_seeded_candidate_is_repeatable_and_in_support(batch: int, vocab: int) -> None:
    probs = _probs(batch, vocab)
    first = _candidate(probs, 0.05, seed=1000 + batch)
    second = _candidate(probs, 0.05, seed=1000 + batch)
    torch.musa.synchronize()

    assert first.dtype == torch.int32
    assert first.shape == (batch,)
    assert torch.equal(first, second)
    assert bool(((first >= 0) & (first < vocab)).all().item())
    _assert_support(probs, first, 0.05)


def test_per_row_threshold_and_row_indices_contract() -> None:
    batch = 16
    probs = _probs(batch)
    indices = torch.arange(batch - 1, -1, -1, dtype=torch.int32, device="musa")
    min_p = torch.linspace(0.01, 0.20, batch, dtype=torch.float32, device="musa")
    first = _candidate(probs, min_p, indices=indices, seed=9001)
    second = _candidate(probs, min_p, indices=indices, seed=9001)
    torch.musa.synchronize()

    assert torch.equal(first, second)
    _assert_support(probs, first, min_p, indices)


@pytest.mark.parametrize("vocab", SUPPORTED_VOCABS)
@pytest.mark.parametrize("batch", [1, 4, 16, 64])
def test_candidate_matches_production_philox_contract(batch: int, vocab: int) -> None:
    probs = _probs(batch, vocab)
    current_output = torch.empty(batch, dtype=torch.int32, device="musa")
    candidate_output = torch.empty(batch, dtype=torch.int32, device="musa")
    current_generator = torch.Generator(device="musa").manual_seed(763100 + batch)
    candidate_generator = torch.Generator(device="musa").manual_seed(763100 + batch)

    torch.ops._C_musa_ops.min_p_sampling_from_probs(
        probs, current_output, None, None, 0.05, True, current_generator
    )
    torch.ops._C_musa_ops.musa_chunked_min_p_sampling_from_probs(
        probs, candidate_output, None, None, 0.05, True, candidate_generator
    )
    assert torch.equal(current_generator.get_state(), candidate_generator.get_state())
    current_next = torch.rand(8, generator=current_generator, device="musa")
    candidate_next = torch.rand(8, generator=candidate_generator, device="musa")
    torch.musa.synchronize()

    assert torch.equal(current_output, candidate_output)
    assert torch.equal(current_next, candidate_next)


@pytest.mark.parametrize("min_p", [0.0, 1.0])
def test_min_p_edge_values(min_p: float) -> None:
    probs = _probs(4)
    samples = _candidate(probs, min_p, seed=1234)
    torch.musa.synchronize()
    _assert_support(probs, samples, min_p)
    if min_p == 1.0:
        selected = probs.gather(1, samples.long()[:, None]).view(-1)
        torch.testing.assert_close(selected, probs.amax(dim=1), rtol=0, atol=0)


def test_custom_op_wrapper_falls_back_outside_qwen_shape() -> None:
    from vllm_musa import _custom_ops

    probs = torch.softmax(torch.randn((4, 1024), device="musa"), dim=-1)
    samples = _custom_ops.min_p_sampling_from_probs(probs, 0.05)
    torch.musa.synchronize()

    assert samples.dtype == torch.int32
    assert samples.shape == (4,)
    _assert_support(probs, samples, 0.05)


def test_default_dispatch_contract() -> None:
    from vllm_musa import _custom_ops

    probs = _probs(1)
    assert _custom_ops._can_use_chunked_min_p_sampler(probs, None, None, True, None)


def test_default_wrapper_preserves_philox_contract() -> None:
    from vllm_musa import _custom_ops

    probs = _probs(1)
    expected = torch.empty(1, dtype=torch.int32, device="musa")
    expected_generator = torch.Generator(device="musa").manual_seed(763200)
    torch.ops._C_musa_ops.min_p_sampling_from_probs(
        probs, expected, None, None, 0.05, True, expected_generator
    )

    actual_generator = torch.Generator(device="musa").manual_seed(763200)
    actual = _custom_ops.min_p_sampling_from_probs(
        probs, 0.05, generator=actual_generator
    )
    torch.musa.synchronize()

    assert torch.equal(actual, expected)
    assert torch.equal(actual_generator.get_state(), expected_generator.get_state())


def test_unsupported_contracts_and_seeded_cpu_fallback() -> None:
    from vllm_musa import _custom_ops

    probs = _probs(1)
    musa_generator = torch.Generator(device="musa")
    cpu_generator = torch.Generator(device="cpu")

    assert _custom_ops._can_use_chunked_min_p_sampler(
        probs, None, None, True, musa_generator
    )
    assert not _custom_ops._can_use_chunked_min_p_sampler(
        probs, None, None, True, cpu_generator
    )
    assert not _custom_ops._can_use_chunked_min_p_sampler(
        probs, None, None, False, None
    )
    assert not _custom_ops._can_use_chunked_min_p_sampler(
        probs.half(), None, None, True, None
    )
    assert not _custom_ops._can_use_chunked_min_p_sampler(
        _probs(1, 1024), None, None, True, None
    )
    assert not _custom_ops._can_use_chunked_min_p_sampler(
        probs.expand(65, -1), None, None, True, None
    )

    non_contiguous = torch.empty(
        (1, 2 * probs.shape[1]), dtype=torch.float32, device="musa"
    )[:, ::2]
    assert not non_contiguous.is_contiguous()
    assert not _custom_ops._can_use_chunked_min_p_sampler(
        non_contiguous, None, None, True, None
    )

    bad_indices = torch.arange(1, dtype=torch.int64, device="musa")
    assert not _custom_ops._can_use_chunked_min_p_sampler(
        probs, bad_indices, None, True, None
    )
    bad_min_p = torch.full((1,), 0.05, dtype=torch.float16, device="musa")
    assert not _custom_ops._can_use_chunked_min_p_sampler(
        probs, None, bad_min_p, True, None
    )


@pytest.mark.parametrize("vocab", SUPPORTED_VOCABS)
def test_candidate_captures_and_replays(vocab: int) -> None:
    batch = 64
    probs = torch.full((batch, vocab), 1.0 / vocab, dtype=torch.float32, device="musa")
    output = torch.empty(batch, dtype=torch.int32, device="musa")
    torch.musa.manual_seed(763300 + vocab)
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        torch.ops._C_musa_ops.musa_chunked_min_p_sampling_from_probs(
            probs, output, None, None, 0.05, True, None
        )
    replayed = []
    for _ in range(3):
        graph.replay()
        replayed.append(output.clone())
    torch.musa.synchronize()

    for samples in replayed:
        assert bool(((samples >= 0) & (samples < probs.shape[1])).all().item())
        _assert_support(probs, samples, 0.05)
    assert not torch.equal(replayed[0], replayed[1])
    assert not torch.equal(replayed[1], replayed[2])
