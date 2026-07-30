# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: I001

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torchada  # noqa: F401
import torch

from vllm_musa.v1.sample import uniform_sample_counts as sample_counts


def make_batch(batch_size: int = 4):
    return SimpleNamespace(
        num_reqs=batch_size,
        num_scheduled_tokens=np.ones(batch_size, dtype=np.int32),
        is_prefilling_np=np.zeros(batch_size, dtype=np.bool_),
        num_draft_tokens=0,
        seq_lens=torch.full((batch_size,), 128, dtype=torch.int32),
    )


def install_test_gates(monkeypatch) -> None:
    monkeypatch.setattr(sample_counts.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(sample_counts, "is_musa_tensor", lambda _tensor: True)
    monkeypatch.setattr(sample_counts, "_is_qwen_sampler_vocab", lambda _tensor: True)


def test_qwen_uniform_sample_counts_accepts_and_reuses(monkeypatch) -> None:
    install_test_gates(monkeypatch)
    sampler = SimpleNamespace(_musa_qwen_family=True)
    logits = torch.empty((4, 151936), dtype=torch.bfloat16)
    first = sample_counts.select_qwen_uniform_sample_counts(
        sampler, logits, make_batch()
    )
    second = sample_counts.select_qwen_uniform_sample_counts(
        sampler, logits, make_batch()
    )

    assert first is not None and second is not None
    assert first[0].tolist() == [1, 1, 1, 1]
    assert first[1].tolist() == [0, 0, 0, 0]
    assert first[0].data_ptr() == second[0].data_ptr()
    assert first[1].data_ptr() == second[1].data_ptr()
    first_host = getattr(first[0], sample_counts.UNIFORM_NUM_SAMPLED_TOKENS_HOST_ATTR)
    second_host = getattr(second[0], sample_counts.UNIFORM_NUM_SAMPLED_TOKENS_HOST_ATTR)
    assert first_host.dtype == np.int32
    assert first_host.tolist() == [1, 1, 1, 1]
    assert first_host.ctypes.data == second_host.ctypes.data


def test_qwen_uniform_sample_counts_accepts_supported_qwen_vocab_sizes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sample_counts.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(sample_counts, "is_musa_tensor", lambda _tensor: True)
    for vocab_size in (151936, 248320):
        output = sample_counts.select_qwen_uniform_sample_counts(
            SimpleNamespace(_musa_qwen_family=True),
            torch.empty((2, vocab_size), dtype=torch.bfloat16),
            make_batch(2),
        )
        assert output is not None
        assert output[0].tolist() == [1, 1]

    assert (
        sample_counts.select_qwen_uniform_sample_counts(
            SimpleNamespace(_musa_qwen_family=True),
            torch.empty((2, 32000), dtype=torch.bfloat16),
            make_batch(2),
        )
        is None
    )


def test_qwen_uniform_sample_counts_grows_and_changes_dtype(monkeypatch) -> None:
    install_test_gates(monkeypatch)
    sampler = SimpleNamespace(_musa_qwen_family=True)
    logits = torch.empty((4, 151936), dtype=torch.bfloat16)
    first = sample_counts.select_qwen_uniform_sample_counts(
        sampler, logits, make_batch()
    )

    larger_batch = make_batch(8)
    larger = sample_counts.select_qwen_uniform_sample_counts(
        sampler,
        torch.empty((8, 151936), dtype=torch.bfloat16),
        larger_batch,
    )
    int64_batch = make_batch(8)
    int64_batch.seq_lens = int64_batch.seq_lens.to(torch.int64)
    int64_counts = sample_counts.select_qwen_uniform_sample_counts(
        sampler,
        torch.empty((8, 151936), dtype=torch.bfloat16),
        int64_batch,
    )

    assert first is not None and larger is not None and int64_counts is not None
    assert larger[0].shape == (8,)
    assert larger[0].data_ptr() != first[0].data_ptr()
    assert int64_counts[0].dtype == torch.int64
    assert int64_counts[0].data_ptr() != larger[0].data_ptr()
    int64_host = getattr(
        int64_counts[0], sample_counts.UNIFORM_NUM_SAMPLED_TOKENS_HOST_ATTR
    )
    assert int64_host.dtype == np.int64
    assert int64_host.tolist() == [1] * 8


def test_qwen_uniform_sample_counts_bound_hook(monkeypatch) -> None:
    install_test_gates(monkeypatch)
    monkeypatch.setattr(
        sample_counts.Sampler,
        "_musa_select_num_sampled_and_rejected",
        sample_counts.select_qwen_uniform_sample_counts,
        raising=False,
    )
    sampler = sample_counts.Sampler.__new__(sample_counts.Sampler)
    sampler._musa_qwen_family = True
    logits = torch.empty((4, 151936), dtype=torch.bfloat16)

    counts = sampler._musa_select_num_sampled_and_rejected(logits, make_batch())

    assert counts is not None
    assert counts[0].tolist() == [1, 1, 1, 1]
    assert counts[1].tolist() == [0, 0, 0, 0]


def test_qwen_uniform_sample_counts_fails_closed(monkeypatch) -> None:
    install_test_gates(monkeypatch)
    logits = torch.empty((4, 151936), dtype=torch.bfloat16)
    cases = []
    batch = make_batch()
    batch.num_scheduled_tokens[0] = 2
    cases.append((logits, batch))
    batch = make_batch()
    batch.is_prefilling_np[0] = True
    cases.append((logits, batch))
    batch = make_batch()
    batch.num_draft_tokens = 1
    cases.append((logits, batch))
    cases.append((logits[:3], make_batch()))
    cases.append((logits.float(), make_batch()))
    batch = make_batch()
    batch.seq_lens = batch.seq_lens[:3]
    cases.append((logits, batch))
    batch = make_batch()
    del batch.is_prefilling_np
    cases.append((logits, batch))

    for candidate_logits, batch in cases:
        assert (
            sample_counts.select_qwen_uniform_sample_counts(
                SimpleNamespace(_musa_qwen_family=True), candidate_logits, batch
            )
            is None
        )


def test_qwen_uniform_sample_counts_rejects_non_qwen_trait(monkeypatch) -> None:
    logits = torch.empty((4, 151936), dtype=torch.bfloat16)
    assert (
        sample_counts.select_qwen_uniform_sample_counts(
            SimpleNamespace(_musa_qwen_family=False), logits, make_batch()
        )
        is None
    )


def test_qwen_uniform_sample_counts_rejects_unsupported_count_dtype(
    monkeypatch,
) -> None:
    install_test_gates(monkeypatch)
    batch = make_batch(2)
    batch.seq_lens = batch.seq_lens.to(torch.float32)
    logits = torch.empty((2, 151936), dtype=torch.bfloat16)
    assert (
        sample_counts.select_qwen_uniform_sample_counts(
            SimpleNamespace(_musa_qwen_family=True), logits, batch
        )
        is None
    )


def test_qwen_async_output_patch_is_tag_gated() -> None:
    patch = (
        Path(__file__).parents[1]
        / "vllm_musa"
        / "patches"
        / "series"
        / "0098-perf-elide-qwen-uniform-count-dtoh.patch"
    ).read_text()
    assert sample_counts.UNIFORM_NUM_SAMPLED_TOKENS_HOST_ATTR in patch
    assert "if host_num_sampled_tokens is None" in patch
    assert "async_copy_to_np(num_sampled_tokens)" in patch
    assert "import vllm_musa" not in patch


def test_qwen_uniform_count_host_tag_is_automatic(monkeypatch) -> None:
    install_test_gates(monkeypatch)
    output = sample_counts.select_qwen_uniform_sample_counts(
        SimpleNamespace(_musa_qwen_family=True),
        torch.empty((2, 151936), dtype=torch.bfloat16),
        make_batch(2),
    )
    assert output is not None
    assert hasattr(output[0], sample_counts.UNIFORM_NUM_SAMPLED_TOKENS_HOST_ATTR)
