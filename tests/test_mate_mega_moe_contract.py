# SPDX-License-Identifier: Apache-2.0
"""Dependency-free execution tests of the actual Mega MoE policy and lifetime."""

from types import SimpleNamespace

import pytest

from tests.moe_contract_utils import load_mega_contract


@pytest.fixture
def mega():
    return load_mega_contract(SimpleNamespace(bfloat16="bf16"))


def _moe(**overrides):
    values = dict(
        hidden_dim=2048,
        intermediate_size_per_partition=768,
        num_experts=128,
        num_logical_experts=128,
        experts_per_token=8,
        max_num_tokens=4096,
        in_dtype="bf16",
        has_bias=False,
        swiglu_alpha=None,
        swiglu_beta=None,
        swiglu_limit=None,
        skip_final_all_reduce=False,
        is_lora_enabled=False,
        moe_parallel_config=SimpleNamespace(ep_size=8, enable_eplb=False),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _vllm_config(mega):
    mega.get_current_vllm_config = lambda: SimpleNamespace(
        weight_transfer_config=None,
        speculative_config=None,
        parallel_config=SimpleNamespace(
            enable_dbo=False, nnodes=1, pipeline_parallel_size=1
        ),
    )


@pytest.mark.parametrize(
    "parallel,field,fragment",
    [
        (True, "nnodes", "single node"),
        (True, "pipeline_parallel_size", "pipeline parallel"),
        (False, "speculative_config", "speculative"),
    ],
)
def test_unvalidated_execution_modes_fail_closed(mega, parallel, field, fragment):
    _vllm_config(mega)
    config = mega.get_current_vllm_config()
    target = config.parallel_config if parallel else config
    setattr(target, field, 2 if parallel else object())
    mega.get_current_vllm_config = lambda: config
    reasons = mega.mate_mega_moe_unsupported_reasons(_moe())
    assert any(fragment in reason for reason in reasons)


@pytest.mark.parametrize("missing", ["_run_decode_mega_moe", "_run_prefill_mega_moe"])
def test_legacy_mate_api_is_rejected_before_weight_repack(mega, missing):
    module = SimpleNamespace(
        _run_decode_mega_moe=lambda: None,
        _run_prefill_mega_moe=lambda: None,
    )
    delattr(module, missing)
    with pytest.raises(RuntimeError, match=missing):
        mega.require_mate_mega_moe_api(module)


def test_default_swiglu_is_supported(mega):
    _vllm_config(mega)
    assert mega.mate_mega_moe_unsupported_reasons(_moe()) == ()


@pytest.mark.parametrize("limit", [0.0, 7.0, 7.1])
def test_nonempty_swiglu_limit_fails_closed(mega, limit):
    _vllm_config(mega)
    reasons = mega.mate_mega_moe_unsupported_reasons(_moe(swiglu_limit=limit))
    assert any(
        "limit" in reason.lower() or "clamp" in reason.lower() for reason in reasons
    )


@pytest.mark.parametrize(
    "counts,decode",
    [
        ([0, 0, 0, 0], True),
        ([0, 1, 127, 128], True),
        ([0, 1, 128, 129], False),
        ([256, 0, 0, 1], False),
    ],
)
def test_every_rank_uses_the_same_path_across_boundary(mega, counts, decode):
    mega.is_forward_context_available = lambda: True
    mega.get_forward_context = lambda: SimpleNamespace(
        dp_metadata=SimpleNamespace(
            num_tokens_across_dp_cpu=SimpleNamespace(max=lambda: max(counts))
        )
    )
    for local in counts:
        assert mega.use_decode_kernels(local) is decode


def _key(mega, group, tokens=256):
    return mega._BufferKey(group, 128, tokens, 8, 2048, 768)


def test_explicit_cleanup_is_group_scoped_and_idempotent(mega):
    own, other = object(), object()
    events = []
    first, second = _key(mega, own), _key(mega, own, 512)
    unrelated = _key(mega, other)
    mega._SYMM_BUFFERS.update(
        {
            second: SimpleNamespace(destroy=lambda: events.append("second")),
            first: SimpleNamespace(destroy=lambda: events.append("first")),
            unrelated: SimpleNamespace(destroy=lambda: events.append("unrelated")),
        }
    )
    mega.destroy_mate_mega_moe_buffers(own)
    assert events == ["first", "second"]
    assert set(mega._SYMM_BUFFERS) == {unrelated}
    mega.destroy_mate_mega_moe_buffers(own)
    assert events == ["first", "second"]


def test_cleanup_failure_keeps_the_failed_buffer_for_retry(mega):
    group = object()
    key = _key(mega, group)

    def fail():
        raise RuntimeError("injected cleanup failure")

    buffer = SimpleNamespace(destroy=fail)
    mega._SYMM_BUFFERS[key] = buffer
    with pytest.raises(RuntimeError, match="injected cleanup failure"):
        mega.destroy_mate_mega_moe_buffers(group)
    assert mega._SYMM_BUFFERS[key] is buffer
