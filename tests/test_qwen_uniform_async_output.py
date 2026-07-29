# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
from vllm.v1.worker.gpu import async_utils

from vllm_musa.v1.sample.uniform_sample_counts import (
    UNIFORM_NUM_SAMPLED_TOKENS_HOST_ATTR,
)


class FakeEvent:
    def __init__(self) -> None:
        self.recorded = False
        self.synchronized = False

    def record(self, stream) -> None:
        del stream
        self.recorded = True

    def synchronize(self) -> None:
        self.synchronized = True


@contextmanager
def fake_stream(to_stream, from_stream):
    del to_stream, from_stream
    yield


class FakeCopyStream:
    def __init__(self) -> None:
        self.waited = False

    def wait_stream(self, main_stream) -> None:
        del main_stream
        self.waited = True


def make_output(counts) -> tuple[SimpleNamespace, SimpleNamespace]:
    return (
        SimpleNamespace(prompt_logprobs_dict={}),
        SimpleNamespace(
            sampled_token_ids=object(),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled_tokens=counts,
        ),
    )


def test_async_output_skips_only_tagged_uniform_count_copy(monkeypatch) -> None:
    sampled_host = np.asarray([[7], [11]], dtype=np.int64)
    fallback_host = np.ones(2, dtype=np.int32)
    calls = []

    def fake_copy(value):
        calls.append(value)
        return sampled_host if len(calls) % 2 == 1 else fallback_host

    monkeypatch.setattr(async_utils, "stream", fake_stream)
    monkeypatch.setattr(async_utils.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(async_utils, "async_copy_to_np", fake_copy)

    tagged_counts = SimpleNamespace()
    cached_host = np.ones(2, dtype=np.int32)
    setattr(tagged_counts, UNIFORM_NUM_SAMPLED_TOKENS_HOST_ATTR, cached_host)
    model_output, sampler_output = make_output(tagged_counts)
    tagged_stream = FakeCopyStream()
    tagged = async_utils.AsyncOutput(
        model_output, sampler_output, tagged_counts, object(), tagged_stream
    )
    assert len(calls) == 1
    assert tagged.num_sampled_tokens_np is cached_host
    assert tagged_stream.waited
    assert tagged.copy_event.recorded
    assert tagged.get_output().sampled_token_ids == [[7], [11]]
    assert tagged.copy_event.synchronized

    calls.clear()
    fallback_counts = SimpleNamespace()
    model_output, sampler_output = make_output(fallback_counts)
    fallback_stream = FakeCopyStream()
    fallback = async_utils.AsyncOutput(
        model_output, sampler_output, fallback_counts, object(), fallback_stream
    )
    assert len(calls) == 2
    assert fallback.num_sampled_tokens_np is fallback_host
    assert fallback_stream.waited
    assert fallback.get_output().sampled_token_ids == [[7], [11]]
