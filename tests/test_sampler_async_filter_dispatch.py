# SPDX-License-Identifier: Apache-2.0

import torchada  # noqa: F401
import torch

from vllm_musa.utils.environ import envs
from vllm_musa.v1.sample import topk_topp_sampler as sampler


def test_async_filter_dispatch_has_no_runtime_gate() -> None:
    assert not hasattr(envs, "VLLM_MUSA_SAMPLER_ASYNC_FILTER_DISPATCH")


def test_async_filter_dispatch_uses_tensor_presence(monkeypatch) -> None:
    probs = torch.full((2, 8), 1.0 / 8, dtype=torch.float32)
    top_k = torch.full((2,), 8, dtype=torch.int32)
    top_p = torch.ones(2, dtype=torch.float32)
    expected = torch.tensor([1, 3], dtype=torch.int32)
    call = {}

    def fake_joint(probs_arg, top_k_arg, top_p_arg, *, filter_apply_order):
        call.update(
            probs=probs_arg,
            top_k=top_k_arg,
            top_p=top_p_arg,
            filter_apply_order=filter_apply_order,
        )
        return expected

    monkeypatch.setattr(
        sampler._ops, "top_k_top_p_sampling_from_probs", fake_joint
    )
    actual = sampler.sample_from_probs(probs, top_k, top_p)

    assert torch.equal(actual, expected.long())
    assert call["probs"] is probs
    assert call["top_k"] is top_k
    assert call["top_p"] is top_p
    assert call["filter_apply_order"] == "joint"
