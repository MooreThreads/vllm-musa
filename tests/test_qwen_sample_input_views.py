from types import SimpleNamespace

import numpy as np
import torch
from qwen_contract_test_utils import qwen_sampler

from vllm_musa.optimization_contract import OptimizationFeature
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


def select(monkeypatch, batch=None, *, is_qwen_family=True, **kwargs):
    monkeypatch.setattr(
        "vllm_musa.v1.sample.topk_topp_sampler.can_use_qwen_v2_unfiltered_gumbel",
        lambda sampler, *args, **kw: sampler._musa_optimization_contract.prefers(
            OptimizationFeature.QWEN_V2_SAMPLING
        ),
    )
    batch = batch or make_batch()
    logits = torch.zeros(batch.num_reqs, 16, dtype=torch.bfloat16)
    return sample_views._select_qwen_sample_input_views(
        qwen_sampler(enabled=is_qwen_family),
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


def test_uniform_decode_views_match_upstream_gather_with_graph_padding(monkeypatch):
    batch = make_batch(batch_size=4, padded=8)
    batch.positions = torch.tensor([19, 7, 31, 3, -1, -1, -1, -1], dtype=torch.int64)
    batch.input_ids = torch.tensor(
        [101, 303, 202, 404, -1, -1, -1, -1], dtype=torch.int32
    )

    # For one scheduled token per actual request, upstream's query-end minus
    # one producer yields the identity logits indices even if request rows are
    # reordered and graph padding follows the actual rows.
    query_start_loc = torch.arange(batch.num_reqs + 1, dtype=torch.int32)
    logits_indices = query_start_loc[1:] - 1
    assert torch.equal(logits_indices, torch.arange(batch.num_reqs, dtype=torch.int32))

    selected = select(monkeypatch, batch)
    assert selected is not None
    assert torch.equal(selected[0], batch.positions[logits_indices])
    assert torch.equal(selected[1], batch.input_ids[logits_indices])
    assert selected[0].data_ptr() == batch.positions.data_ptr()
    assert selected[1].data_ptr() == batch.input_ids.data_ptr()


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


def test_non_qwen_trait_falls_back(monkeypatch):
    assert select(monkeypatch, is_qwen_family=False) is None
