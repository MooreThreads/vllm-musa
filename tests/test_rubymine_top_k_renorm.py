# ruff: noqa: I001

import numpy as np
import pytest
import torch
import torchada  # noqa: F401

import vllm_musa._custom_ops  # noqa: F401
from vllm_musa.v1.sample.topk_topp_sampler import _is_uniform_top_k_50


def _musa_available() -> bool:
    return hasattr(torch, "musa") and torch.musa.is_available()


requires_musa = pytest.mark.skipif(
    not _musa_available(), reason="requires a MUSA device and rebuilt extension"
)


def test_uniform_top_k_50_uses_existing_cpu_state() -> None:
    assert _is_uniform_top_k_50(np.array([50], dtype=np.int32))
    assert _is_uniform_top_k_50(np.array([50, 50, 50], dtype=np.int32))
    assert not _is_uniform_top_k_50(np.array([], dtype=np.int32))
    assert not _is_uniform_top_k_50(np.array([50, 49], dtype=np.int32))


def _native(probs: torch.Tensor, top_k: int | torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(probs)
    if isinstance(top_k, torch.Tensor):
        maybe_top_k_arr = top_k.int()
        top_k_val = 0
    else:
        maybe_top_k_arr = None
        top_k_val = top_k
    torch.ops._C_musa_ops.top_k_renorm_probs.default(
        probs, output, maybe_top_k_arr, top_k_val
    )
    return output


def _candidate(probs: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(probs)
    torch.ops._C_musa_ops.musa_rubymine_top_k_renorm_probs.default(probs, output, 50)
    return output


def _assert_native_parity(probs: torch.Tensor) -> None:
    expected = _native(probs, 50)
    actual = _candidate(probs)
    torch.musa.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
    assert torch.equal(actual != 0, expected != 0)
    torch.testing.assert_close(
        actual.sum(dim=-1),
        torch.ones(probs.shape[0], device=probs.device),
        rtol=2e-5,
        atol=2e-6,
    )


@requires_musa
@pytest.mark.parametrize("rows", [1, 4, 16, 64, 128, 256])
def test_qwen_vocab_native_parity(rows: int) -> None:
    torch.manual_seed(1234 + rows)
    probs = torch.softmax(
        torch.randn((rows, 151936), device="musa", dtype=torch.float32), dim=-1
    )
    _assert_native_parity(probs)


@requires_musa
@pytest.mark.parametrize("rows", [16, 64, 128, 256])
def test_radix_top_k_mask_matches_topk(rows: int) -> None:
    from vllm_musa._custom_ops import top_k_mask_logits_radix

    torch.manual_seed(9000 + rows)
    logits = (torch.randn((rows, 248320), device="musa", dtype=torch.float32) * 2).to(
        torch.bfloat16
    )
    reference = logits.clone()
    threshold = reference.topk(50, dim=-1).values[:, -1:]
    expected_masked = reference < threshold

    actual = top_k_mask_logits_radix(logits, 50)
    assert actual is not None
    torch.musa.synchronize()

    assert torch.equal(torch.isinf(actual), expected_masked)
    assert torch.equal(actual[~expected_masked], reference[~expected_masked])


@requires_musa
def test_radix_top_k_mask_declines_unsupported_rows() -> None:
    from vllm_musa._custom_ops import top_k_mask_logits_radix

    logits = torch.zeros((300, 248320), device="musa", dtype=torch.bfloat16)
    assert top_k_mask_logits_radix(logits, 50) is None
    assert top_k_mask_logits_radix(logits[:4].contiguous(), 49) is None


@requires_musa
def test_tied_and_sparse_rows_native_parity() -> None:
    tied = torch.full((1, 1024), 1.0 / 1024, device="musa")
    sparse = torch.zeros((1, 1024), device="musa")
    sparse[:, :600] = 1.0 / 600
    one_hot = torch.zeros((1, 1024), device="musa")
    one_hot[:, 17] = 1.0
    for probs in (tied, sparse, one_hot):
        _assert_native_parity(probs)


@requires_musa
def test_invalid_probabilities_are_sanitized_without_illegal_access() -> None:
    probs = torch.softmax(torch.randn((1, 1024), device="musa"), dim=-1)
    probs[0, 0] = torch.nan
    probs[0, 1] = torch.inf
    probs[0, 2] = -1.0
    probs[0, 3] = 2.0

    sanitized = probs.clone()
    sanitized[~torch.isfinite(sanitized)] = 0.0
    sanitized[(sanitized < 0.0) | (sanitized > 1.0)] = 0.0
    expected = _native(sanitized, 50)
    actual = _candidate(probs)
    torch.musa.synchronize()

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
    assert torch.equal(actual != 0, expected != 0)


@requires_musa
@pytest.mark.parametrize("vocab", [151936, 248320])
def test_valid_qwen_vocab_wrapper_uses_default_candidate(vocab: int) -> None:
    from vllm_musa import _custom_ops

    probs = torch.softmax(
        torch.randn((1, vocab), device="musa", dtype=torch.float32), dim=-1
    )
    actual = _custom_ops.top_k_renorm_probs(probs, 50)
    expected = _candidate(probs)
    torch.musa.synchronize()

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)

    expected_k49 = _native(probs, 49)
    actual_k49 = _custom_ops.top_k_renorm_probs(probs, 49)
    per_row_k = torch.tensor([50], device="musa", dtype=torch.int32)
    expected_per_row = _native(probs, per_row_k)
    actual_per_row = _custom_ops.top_k_renorm_probs(probs, per_row_k)
    torch.testing.assert_close(actual_k49, expected_k49, rtol=0, atol=0)
    torch.testing.assert_close(actual_per_row, expected_per_row, rtol=0, atol=0)


@requires_musa
def test_heterogeneous_and_unsupported_k_keep_native_fallback() -> None:
    from vllm_musa import _custom_ops

    probs = torch.softmax(torch.randn((2, 1024), device="musa"), dim=-1)
    per_row_k = torch.tensor([50, 49], device="musa", dtype=torch.int32)

    expected_per_row = _native(probs, per_row_k)
    expected_k49 = _native(probs, 49)
    actual_per_row = _custom_ops.top_k_renorm_probs(probs, per_row_k)
    actual_k49 = _custom_ops.top_k_renorm_probs(probs, 49)

    torch.testing.assert_close(actual_per_row, expected_per_row, rtol=0, atol=0)
    torch.testing.assert_close(actual_k49, expected_k49, rtol=0, atol=0)


@requires_musa
def test_repeated_musa_graph_replay() -> None:
    static_probs = torch.softmax(
        torch.randn((4, 151936), device="musa", dtype=torch.float32), dim=-1
    )
    static_output = torch.empty_like(static_probs)

    warmup_stream = torch.musa.Stream()
    warmup_stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(warmup_stream):
        for _ in range(3):
            torch.ops._C_musa_ops.musa_rubymine_top_k_renorm_probs.default(
                static_probs, static_output, 50
            )
    torch.musa.current_stream().wait_stream(warmup_stream)

    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        torch.ops._C_musa_ops.musa_rubymine_top_k_renorm_probs.default(
            static_probs, static_output, 50
        )

    for seed in (7, 11, 19):
        torch.manual_seed(seed)
        static_probs.copy_(torch.softmax(torch.randn_like(static_probs), dim=-1))
        graph.replay()
        expected = _native(static_probs, 50)
        torch.musa.synchronize()
        torch.testing.assert_close(static_output, expected, rtol=2e-5, atol=2e-7)
        assert torch.equal(static_output != 0, expected != 0)


@requires_musa
def test_two_concurrent_streams_are_independent() -> None:
    probs_a = torch.softmax(torch.randn((1, 151936), device="musa"), dim=-1)
    probs_b = torch.softmax(torch.randn((4, 151936), device="musa"), dim=-1)
    output_a = torch.empty_like(probs_a)
    output_b = torch.empty_like(probs_b)
    stream_a = torch.musa.Stream()
    stream_b = torch.musa.Stream()

    with torch.musa.stream(stream_a):
        torch.ops._C_musa_ops.musa_rubymine_top_k_renorm_probs.default(
            probs_a, output_a, 50
        )
    with torch.musa.stream(stream_b):
        torch.ops._C_musa_ops.musa_rubymine_top_k_renorm_probs.default(
            probs_b, output_b, 50
        )
    torch.musa.synchronize()

    torch.testing.assert_close(output_a, _native(probs_a, 50), rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(output_b, _native(probs_b, 50), rtol=2e-5, atol=2e-7)
