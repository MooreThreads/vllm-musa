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
        return isinstance(other, _FakeDevice) and other.type == self.type


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
    assert not router._can_use_musa_jit_topk(hidden, logits, 0, None)


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


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="requires a MUSA device",
)
def test_native_reshape_cache_and_fallback_guards_are_env_invariant(monkeypatch):
    from vllm_musa.v1.attention.backends import fa_utils

    monkeypatch.setattr(fa_utils, "_HAS_NATIVE_RESHAPE_CACHE_FLASH", True)
    key = torch.empty((2, 4, 64), dtype=torch.bfloat16, device="musa")
    value = torch.empty_like(key)
    key_cache = torch.empty((1, 16, 4, 64), dtype=torch.bfloat16, device="musa")
    value_cache = torch.empty_like(key_cache)
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


def test_ar_unavailable_comm_and_pass_config_are_fail_closed(monkeypatch):
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
