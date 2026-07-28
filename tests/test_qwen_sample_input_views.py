from types import SimpleNamespace

import numpy as np
import torch

from vllm_musa.v1.sample import qwen_sample_input_views as sample_views


def make_batch(batch_size: int = 4, padded: int = 8):
    return SimpleNamespace(
        num_reqs=batch_size,
        num_reqs_after_padding=padded,
        num_tokens=batch_size,
        num_tokens_after_padding=padded,
        num_draft_tokens=0,
        num_scheduled_tokens=np.ones(batch_size, dtype=np.int32),
        is_prefilling_np=np.zeros(batch_size, dtype=np.bool_),
        positions=torch.arange(padded, dtype=torch.int64),
        input_ids=torch.arange(padded, dtype=torch.int32),
    )


def select(monkeypatch, batch=None, **kwargs):
    monkeypatch.setattr(
        sample_views.envs.VLLM_MUSA_QWEN_SAMPLE_INPUT_VIEWS,
        "get",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm_musa.v1.sample.topk_topp_sampler.can_use_qwen_v2_unfiltered_gumbel",
        lambda *args, **kw: True,
    )
    batch = batch or make_batch()
    logits = torch.zeros(batch.num_reqs, 16, dtype=torch.bfloat16)
    return sample_views._select_qwen_sample_input_views(
        SimpleNamespace(),
        logits,
        batch,
        torch.arange(batch.num_reqs, dtype=torch.int32),
        np.arange(batch.num_reqs, dtype=np.int32),
        kwargs.pop("return_logprobs", False),
    )


def test_uniform_decode_uses_identity_views(monkeypatch):
    batch = make_batch()
    selected = select(monkeypatch, batch)
    assert selected is not None
    assert selected[0].data_ptr() == batch.positions.data_ptr()
    assert selected[1].data_ptr() == batch.input_ids.data_ptr()
    assert selected[0].shape == selected[1].shape == (batch.num_reqs,)


def test_non_uniform_or_logprobs_falls_back(monkeypatch):
    for overrides in (
        {"num_scheduled_tokens": np.array([1, 1, 2, 1], dtype=np.int32)},
        {"is_prefilling_np": np.array([False, True, False, False])},
        {"num_draft_tokens": 1},
        {"return_logprobs": True},
    ):
        batch = make_batch()
        for key, value in overrides.items():
            if key != "return_logprobs":
                setattr(batch, key, value)
        assert select(monkeypatch, batch, **overrides) is None


def test_mismatched_padding_falls_back(monkeypatch):
    batch = make_batch()
    batch.num_reqs_after_padding = 4
    batch.num_tokens_after_padding = 8
    assert select(monkeypatch, batch) is None
