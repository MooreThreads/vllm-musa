"""Decision-parity tests for the removed default-on process gates.

These tests deliberately set each legacy variable to unset/0/1 around the
same provider decision.  The variables are no longer read by production code;
the loop makes that invariant executable while the assertions exercise the
actual capability and semantic guards that remain responsible for fallback.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

LEGACY_ENV_NAMES = (
    "VLLM_MUSA_FUSED_ADD_RMSNORM",
    "VLLM_MUSA_FUSED_AR_RMSNORM",
    "VLLM_MUSA_FUSED_AR_RMSNORM_GRAPH_REGISTERED_INPUT",
    "VLLM_MUSA_ENABLE_JIT_TOPK",
    "VLLM_MUSA_SEEDED_MULTINOMIAL",
    "VLLM_MUSA_RESHAPE_CACHE_FLASH",
)
LEGACY_ENV_VALUES = (None, "0", "1")


def _with_legacy_env(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    for name in LEGACY_ENV_NAMES:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


class _FakeDevice:
    type = "musa"
    index = 0

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _FakeDevice)
            and other.type == self.type
            and other.index == self.index
        )


class _FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        contiguous: bool = True,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = _FakeDevice()
        self._contiguous = contiguous

    def dim(self) -> int:
        return len(self.shape)

    def is_contiguous(self) -> bool:
        return self._contiguous


def test_jit_topk_capability_and_missing_provider_are_env_invariant(monkeypatch):
    from vllm_musa.model_executor.layers.fused_moe.router import (
        grouped_topk_router as router,
    )

    monkeypatch.setattr(router.current_platform, "is_musa", lambda: True)
    hidden = _FakeTensor((4, 128), torch.bfloat16)
    logits = _FakeTensor((4, 256), torch.bfloat16)
    expected = router._can_use_musa_jit_topk(hidden, logits, 8, None)
    assert expected is True

    # A missing JIT provider must still use the upstream fused_topk fallback;
    # this is a provider capability decision, not an environment decision.
    monkeypatch.setattr(router, "_maybe_import_musa_jit_topk", lambda: None)
    for value in LEGACY_ENV_VALUES:
        _with_legacy_env(monkeypatch, value)
        assert router._can_use_musa_jit_topk(hidden, logits, 8, None) is expected
        assert (
            router._musa_jit_fused_topk(
                hidden,
                logits,
                8,
                True,
                torch.int32,
            )
            is None
        )

    # Existing shape/scoring guards continue to reject unsupported inputs.
    assert not router._can_use_musa_jit_topk(
        hidden, _FakeTensor((4, 2048), torch.bfloat16), 8, None
    )
    assert router._can_use_musa_jit_topk(
        hidden, _FakeTensor((4, 768), torch.bfloat16), 8, None
    )
    assert router._can_use_musa_jit_topk(
        hidden, _FakeTensor((4, 1000), torch.bfloat16), 8, None
    )
    assert router._can_use_musa_jit_topk(
        hidden, _FakeTensor((4, 1024), torch.bfloat16), 8, None
    )
    assert not router._can_use_musa_jit_topk(
        hidden, _FakeTensor((4, 1025), torch.bfloat16), 8, None
    )
    assert not router._can_use_musa_jit_topk(hidden, logits, 0, None)


def test_jit_topk_unsupported_scoring_stays_on_upstream_fallback(monkeypatch):
    from vllm_musa.model_executor.layers.fused_moe.router import (
        grouped_topk_router as router,
    )

    monkeypatch.setattr(router.current_platform, "is_musa", lambda: True)
    hidden = _FakeTensor((1, 128), torch.bfloat16)
    logits = _FakeTensor((1, 256), torch.bfloat16)

    class _FakeOutput:
        dtype = torch.int32

        def to(self, _dtype):
            return self

    class _FakeProvider:
        def topk_softmax(self, *_args, **_kwargs):
            return None

        def topk_sigmoid(self, *_args, **_kwargs):
            return None

    # Avoid allocating a real MUSA tensor while still exercising the provider
    # selection branch.
    monkeypatch.setattr(
        router.torch,
        "empty",
        lambda *_args, **_kwargs: _FakeOutput(),
    )
    monkeypatch.setattr(router, "_maybe_import_musa_jit_topk", lambda: _FakeProvider())
    correction_bias = _FakeTensor((256,), torch.float32)
    for value in LEGACY_ENV_VALUES:
        _with_legacy_env(monkeypatch, value)
        assert (
            router._musa_jit_fused_topk(
                hidden,
                logits,
                8,
                True,
                None,
                scoring_func="unsupported",
            )
            is None
        )
        assert (
            router._musa_jit_fused_topk(
                hidden,
                logits,
                8,
                True,
                None,
                correction_bias=correction_bias,
                scoring_func="softmax",
            )
            is None
        )


def test_biased_topk_caller_passes_upstream_fallback_contract(monkeypatch):
    from vllm_musa.model_executor.layers.fused_moe.router import (
        grouped_topk_router as router,
    )

    captured = {}
    expected_weights = torch.tensor([[0.4, 0.6]], dtype=torch.float32)
    expected_ids = torch.tensor([[1, 2]], dtype=torch.int64)

    def fallback(**kwargs):
        captured.update(kwargs)
        return expected_weights, expected_ids

    monkeypatch.setattr(router, "_musa_jit_fused_topk", lambda **_kwargs: None)
    monkeypatch.setattr(router, "fused_topk_bias", fallback)
    router_state = SimpleNamespace(
        num_expert_group=8,
        e_score_correction_bias=SimpleNamespace(data=torch.zeros(8)),
        top_k=2,
        renormalize=True,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
    )
    weights, ids = router._compute_routing(
        router_state,
        torch.zeros((1, 4)),
        torch.zeros((1, 8)),
        torch.int64,
    )

    assert weights is expected_weights
    assert ids is expected_ids
    assert captured["scoring_func"] == "softmax"
    assert captured["indices_type"] == torch.int64


def _worker_sampling_states(*, seeded: bool) -> SimpleNamespace:
    return SimpleNamespace(
        has_user_seed=np.asarray([seeded, seeded], dtype=np.bool_),
        temperature=SimpleNamespace(np=np.asarray([1.0, 1.0], dtype=np.float32)),
        musa_generators={0: object(), 1: object()} if seeded else None,
    )


def test_worker_seeded_sampler_and_fallback_guards_are_env_invariant(monkeypatch):
    from vllm_musa.v1.sample import topk_topp_sampler as sampler

    monkeypatch.setattr(sampler.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(sampler, "is_musa_tensor", lambda _tensor: True)
    logits = torch.empty((2, 151936), dtype=torch.bfloat16)
    indices = np.asarray([0, 1], dtype=np.int64)

    for value in LEGACY_ENV_VALUES:
        _with_legacy_env(monkeypatch, value)
        seeded = _worker_sampling_states(seeded=True)
        unseeded = _worker_sampling_states(seeded=False)
        assert sampler.can_use_worker_seeded_multinomial(
            logits, "raw_logprobs", seeded, indices
        )
        assert not sampler.can_use_worker_seeded_multinomial(
            logits, "processed_logprobs", seeded, indices
        )
        assert not sampler.can_use_worker_seeded_multinomial(
            logits, "raw_logprobs", unseeded, indices
        )
        assert sampler.can_use_worker_sampler(logits, "raw_logprobs", unseeded, indices)
        assert not sampler.can_use_worker_sampler(
            logits, "raw_logprobs", seeded, indices
        )


def test_seeded_multinomial_mixed_rows_replay_and_advance_are_env_invariant(
    monkeypatch,
):
    from vllm_musa.v1.sample import topk_topp_sampler as sampler

    probs = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.25] * 4],
        dtype=torch.float32,
    )

    def run_once():
        generators = {
            0: torch.Generator().manual_seed(60086),
            2: torch.Generator().manual_seed(60088),
        }
        before = {row: generator.get_state() for row, generator in generators.items()}
        torch.manual_seed(60087)
        sampled = sampler.sample_probs_seeded_multinomial(probs, generators)
        advanced = {
            row: not torch.equal(before[row], generator.get_state())
            for row, generator in generators.items()
        }
        return sampled, advanced

    for value in LEGACY_ENV_VALUES:
        _with_legacy_env(monkeypatch, value)
        first, first_advanced = run_once()
        second, second_advanced = run_once()
        assert torch.equal(first, second)
        assert all(first_advanced.values())
        assert all(second_advanced.values())


@pytest.mark.skipif(
    not hasattr(torch, "musa")
    or not torch.musa.is_available()
    or not getattr(torch.musa, "is_available", lambda: False)(),
    reason="requires a MUSA device",
)
def test_native_reshape_cache_and_fallback_guards_are_env_invariant(monkeypatch):
    from vllm.platforms import current_platform

    from vllm_musa.v1.attention.backends import fa_utils

    if not current_platform.is_musa():
        pytest.skip("MUSA platform plugin is not active")
    if not hasattr(fa_utils, "_can_use_musa_reshape_and_cache_flash_nhd"):
        pytest.skip("MUSA FlashAttention provider is not loaded")
    monkeypatch.setattr(fa_utils, "_HAS_NATIVE_RESHAPE_CACHE_FLASH", True)
    key = torch.empty((2, 4, 64), dtype=torch.bfloat16, device="musa")
    value = torch.empty_like(key)
    key_cache = torch.empty((1, 16, 4, 64), dtype=torch.bfloat16, device="musa")
    value_cache = torch.empty_like(key_cache)
    padded_value_storage = torch.empty(
        (2, 16, 4, 72), dtype=torch.bfloat16, device="musa"
    )
    asymmetric_value_cache = padded_value_storage.as_strided(
        (1, 16, 4, 64),
        (padded_value_storage.stride(0), 4 * 64, 64, 1),
    )
    slots = torch.arange(2, dtype=torch.long, device="musa")
    scale = torch.ones(1, dtype=torch.float32, device="musa")

    for value_env in LEGACY_ENV_VALUES:
        _with_legacy_env(monkeypatch, value_env)
        assert fa_utils._can_use_musa_reshape_and_cache_flash_nhd(
            key,
            value,
            key_cache,
            value_cache,
            slots,
            "auto",
            scale,
            scale,
        )
        assert not fa_utils._can_use_musa_reshape_and_cache_flash_nhd(
            key,
            value.to(torch.float16),
            key_cache,
            value_cache,
            slots,
            "auto",
            scale,
            scale,
        )
        assert not fa_utils._can_use_musa_reshape_and_cache_flash_nhd(
            key,
            value,
            key_cache,
            asymmetric_value_cache,
            slots,
            "auto",
            scale,
            scale,
        )

        native_calls = []
        fallback_calls = []
        monkeypatch.setattr(
            fa_utils.musa_ops,
            "musa_reshape_and_cache_flash_nhd",
            lambda *args: native_calls.append(args),
        )
        monkeypatch.setattr(
            fa_utils.ops,
            "reshape_and_cache_flash",
            lambda *args: fallback_calls.append(args),
        )
        fa_utils.reshape_and_cache_flash(
            key, value, key_cache, value_cache, slots, "auto", scale, scale
        )
        assert len(native_calls) == 1
        assert not fallback_calls
        fa_utils.reshape_and_cache_flash(
            key,
            value.to(torch.float16),
            key_cache,
            value_cache,
            slots,
            "auto",
            scale,
            scale,
        )
        assert len(fallback_calls) == 1


def test_ar_unavailable_comm_is_fail_closed_and_pass_config_is_preserved(
    monkeypatch,
):
    from vllm_musa.distributed.device_communicators.musa_jit_custom_all_reduce import (
        MusaJitCustomAllreduce,
    )

    comm = MusaJitCustomAllreduce.__new__(MusaJitCustomAllreduce)
    comm._jit_comm = None
    input_tensor = object()
    weight = object()
    for value in LEGACY_ENV_VALUES:
        _with_legacy_env(monkeypatch, value)
        assert comm.disabled
        assert not comm.should_fused_allreduce_rmsnorm(input_tensor, weight)
        assert comm.fused_allreduce_rmsnorm(input_tensor, weight, 1e-5) is None

    # The pass manager remains the only pass-level off switch; no process gate
    # is allowed to bypass the standard vLLM pass configuration.
    source = (
        __import__("pathlib").Path(__file__).parents[1]
        / "vllm_musa/patches/series/0003-MUSA-vllm.compilation.passes.pass_manager.patch"
    ).read_text()
    assert (
        "if current_platform.is_musa() and self.pass_config.fuse_allreduce_rms"
        in source
    )


def test_ar_available_comm_dispatch_is_env_invariant(monkeypatch):
    from vllm_musa.distributed.device_communicators.musa_jit_custom_all_reduce import (
        MusaJitCustomAllreduce,
    )

    sentinel = (object(), object())
    jit_comm = SimpleNamespace(
        disabled=False,
        should_fused_allreduce_rmsnorm=lambda *_args: True,
        fused_allreduce_rmsnorm=lambda *_args: sentinel,
    )
    comm = MusaJitCustomAllreduce.__new__(MusaJitCustomAllreduce)
    comm._jit_comm = jit_comm
    for value in LEGACY_ENV_VALUES:
        _with_legacy_env(monkeypatch, value)
        assert comm.fused_allreduce_rmsnorm(object(), object(), 1e-5) is sentinel


@pytest.mark.parametrize("registered", [False, True])
def test_ar_registered_and_staging_launches_are_env_invariant(
    monkeypatch,
    registered,
):
    import vllm_musa.distributed.device_communicators.musa_jit_custom_all_reduce as ar

    impl = ar._MusaJitCustomAllreduceImpl.__new__(ar._MusaJitCustomAllreduceImpl)
    staging_rank_data = object()
    registered_rank_data = object()
    impl.buffer_rank_data = staging_rank_data
    impl.signal_ptrs_cpu = object()
    impl.world_size = 2
    impl.rank = 0
    impl.max_size = 4096
    impl.meta_ptrs = [101, 102]
    impl.buffer_ptrs = [201, 202]
    impl._use_registered_graph_input = lambda _input: registered
    impl._graph_rank_data_for_input = lambda _input: registered_rank_data

    launches = []
    monkeypatch.setattr(ar.jit_ar, "preferred_shot", lambda *_args: 1)
    monkeypatch.setattr(
        ar.jit_ar,
        "launch_fused_allreduce_rmsnorm_registered",
        lambda *args: launches.append(("registered", args[0])),
    )
    monkeypatch.setattr(
        ar.jit_ar,
        "launch_fused_allreduce_rmsnorm_unregistered",
        lambda *args: launches.append(("staging", args[0])),
    )
    input_tensor = torch.ones((1, 8), dtype=torch.bfloat16)
    weight = torch.ones((8,), dtype=torch.bfloat16)

    for value in LEGACY_ENV_VALUES:
        _with_legacy_env(monkeypatch, value)
        launches.clear()
        norm_out, reduced = impl._fused_allreduce_rmsnorm_impl(
            input_tensor, weight, 1e-5
        )
        assert norm_out.shape == input_tensor.shape
        assert reduced.shape == input_tensor.shape
        expected_name = "registered" if registered else "staging"
        expected_rank_data = registered_rank_data if registered else staging_rank_data
        assert launches == [(expected_name, expected_rank_data)]
