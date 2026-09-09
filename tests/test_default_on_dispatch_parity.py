"""Decision-parity tests for default-on MUSA features.

CAR-RMSNorm uses vLLM's standard pass config as its only feature switch. The
registered-input transport and remaining feature decisions are environment-invariant.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

LEGACY_ENV_NAMES = (
    "VLLM_MUSA_FUSED_ADD_RMSNORM",
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


def _ar_platform_config(
    *,
    optimization_level: int = 2,
    tp_size: int = 2,
    hidden_size: int = 5120,
    pp_size: int = 1,
    pass_value: bool | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        optimization_level=optimization_level,
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            get_hidden_size=lambda: hidden_size,
            enforce_eager=False,
            hf_config=SimpleNamespace(architectures=[]),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
        ),
        compilation_config=SimpleNamespace(
            custom_ops=[],
            pass_config=SimpleNamespace(fuse_allreduce_rms=pass_value),
            inductor_compile_config={},
            compile_ranges_endpoints=[],
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
    )


def _patch_car_model_family(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_musa.optimization_contract.car_rmsnorm import (
        FUSED_ALLREDUCE_RMSNORM_MODEL_FAMILY,
    )
    from vllm_musa import platform as musa_platform

    monkeypatch.setattr(
        musa_platform,
        "infer_car_rmsnorm_model_family",
        lambda _config: FUSED_ALLREDUCE_RMSNORM_MODEL_FAMILY,
    )


@pytest.mark.parametrize(
    ("tp_size", "hidden_size"),
    [(2, 5120), (4, 2048)],
)
@pytest.mark.parametrize("optimization_level", [2, 3])
def test_ar_pass_defaults_on_for_supported_signatures(
    monkeypatch,
    tp_size: int,
    hidden_size: int,
    optimization_level: int,
) -> None:
    from vllm_musa.platform import MUSAPlatformBase

    _patch_car_model_family(monkeypatch)
    config = _ar_platform_config(
        optimization_level=optimization_level,
        tp_size=tp_size,
        hidden_size=hidden_size,
    )

    MUSAPlatformBase.apply_config_platform_defaults(config)

    assert config.compilation_config.pass_config.fuse_allreduce_rms is True


def test_ar_explicit_false_blocks_default_pass_injection(monkeypatch) -> None:
    from vllm_musa.platform import MUSAPlatformBase

    _patch_car_model_family(monkeypatch)
    config = _ar_platform_config(pass_value=False)

    MUSAPlatformBase.apply_config_platform_defaults(config)

    assert config.compilation_config.pass_config.fuse_allreduce_rms is False


def test_ar_explicit_false_skips_compile_range_partition(monkeypatch) -> None:
    from vllm_musa.platform import _configure_fused_allreduce_rmsnorm_compile_range

    _patch_car_model_family(monkeypatch)
    config = _ar_platform_config(pass_value=False)

    changed = _configure_fused_allreduce_rmsnorm_compile_range(
        config,
        native_custom_ops=False,
    )

    assert changed is False
    assert config.compilation_config.compile_ranges_endpoints == []


@pytest.mark.parametrize(
    ("optimization_level", "tp_size", "hidden_size", "pp_size"),
    [
        (0, 2, 5120, 1),
        (1, 2, 5120, 1),
        (2, 1, 5120, 1),
        (2, 6, 2048, 1),
        (2, 2, 4096, 1),
        (2, 2, 5120, 2),
    ],
)
def test_ar_pass_default_is_fail_closed_outside_supported_scope(
    monkeypatch,
    optimization_level: int,
    tp_size: int,
    hidden_size: int,
    pp_size: int,
) -> None:
    from vllm_musa.platform import MUSAPlatformBase

    _patch_car_model_family(monkeypatch)
    config = _ar_platform_config(
        optimization_level=optimization_level,
        tp_size=tp_size,
        hidden_size=hidden_size,
        pp_size=pp_size,
    )

    MUSAPlatformBase.apply_config_platform_defaults(config)

    assert config.compilation_config.pass_config.fuse_allreduce_rms is None


@pytest.mark.parametrize("pass_value", [False, True])
def test_ar_pass_explicit_standard_config_is_preserved(
    monkeypatch, pass_value: bool
) -> None:
    from vllm_musa.platform import MUSAPlatformBase

    _patch_car_model_family(monkeypatch)
    config = _ar_platform_config(pass_value=pass_value)

    MUSAPlatformBase.apply_config_platform_defaults(config)

    assert config.compilation_config.pass_config.fuse_allreduce_rms is pass_value


@pytest.mark.parametrize(
    ("pass_value", "optimization_level", "tp_size", "hidden_size", "expected"),
    [
        (True, 0, 1, 4096, True),
        (False, 3, 2, 5120, False),
        (None, 2, 2, 5120, True),
        (None, 3, 4, 2048, True),
        (None, 1, 2, 5120, False),
        (None, 2, 2, 4096, False),
    ],
)
def test_gemma_ir_path_predicts_the_same_default_as_platform(
    monkeypatch,
    pass_value: bool | None,
    optimization_level: int,
    tp_size: int,
    hidden_size: int,
    expected: bool,
) -> None:
    from vllm_musa.model_executor.layers import layernorm
    from vllm_musa.optimization_contract.car_rmsnorm import (
        FUSED_ALLREDUCE_RMSNORM_MODEL_FAMILY,
    )

    config = _ar_platform_config(
        optimization_level=optimization_level,
        tp_size=tp_size,
        hidden_size=hidden_size,
        pass_value=pass_value,
    )
    monkeypatch.setattr(layernorm, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(
        layernorm,
        "infer_car_rmsnorm_model_family",
        lambda _config: FUSED_ALLREDUCE_RMSNORM_MODEL_FAMILY,
    )

    assert layernorm._car_rmsnorm_ir_fusion_enabled() is expected


def test_gemma_ir_path_honors_explicit_pass_false(monkeypatch) -> None:
    from vllm_musa.model_executor.layers import layernorm

    monkeypatch.setattr(
        layernorm,
        "get_current_vllm_config_or_none",
        lambda: _ar_platform_config(pass_value=False),
    )

    assert not layernorm._car_rmsnorm_ir_fusion_enabled()


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

    # The pass manager remains the standard upper-level switch.
    source = (
        __import__("pathlib").Path(__file__).parents[1]
        / "vllm_musa/patches/series/0003-MUSA-vllm.compilation.passes.pass_manager.patch"
    ).read_text()
    assert (
        "if current_platform.is_musa() and self.pass_config.fuse_allreduce_rms"
        in source
    )


def test_ar_available_comm_dispatch_is_unrelated_env_invariant(monkeypatch):
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


def test_ar_explicit_pass_false_disables_fused_communicator():
    import vllm_musa.distributed.device_communicators.musa_jit_custom_all_reduce as ar

    impl = ar._MusaJitCustomAllreduceImpl.__new__(ar._MusaJitCustomAllreduceImpl)
    impl.disabled = False
    impl._fused_allreduce_rmsnorm_enabled = False
    comm = ar.MusaJitCustomAllreduce.__new__(ar.MusaJitCustomAllreduce)
    comm._jit_comm = impl
    input_tensor = object()
    weight = object()

    assert not comm.disabled
    assert not comm.should_fused_allreduce_rmsnorm(input_tensor, weight)
    assert comm.fused_allreduce_rmsnorm(input_tensor, weight, 1e-5) is None
    reason = impl._reject_fused_allreduce_rmsnorm_reason(input_tensor, weight)
    assert reason is not None
    assert "compilation pass config" in reason


@pytest.mark.parametrize(
    "pass_value, expected", [(True, True), (False, False), (None, False)]
)
def test_ar_communicator_reads_pass_config_once(
    monkeypatch, pass_value: bool | None, expected: bool
) -> None:
    import vllm_musa.distributed.device_communicators.musa_jit_custom_all_reduce as ar

    config = _ar_platform_config(pass_value=pass_value)
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config_or_none", lambda: config
    )

    assert ar._car_rmsnorm_pass_enabled_for_current_model() is expected


def test_ar_fused_registered_transport_uses_configured_transport(monkeypatch):
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
    impl._use_graph_registered_inputs = True
    impl._IS_CAPTURING = True
    impl._is_current_stream_capturing = lambda: True
    impl._use_graph_collective_fallback = False
    impl._use_graph_staging_arena = False
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

    impl._graph_registered_input_enabled = impl._use_graph_registered_inputs

    norm_out, reduced = impl._fused_allreduce_rmsnorm_impl(
        input_tensor, weight, 1e-5
    )
    assert norm_out.shape == input_tensor.shape
    assert reduced.shape == input_tensor.shape
    assert launches == [("registered", registered_rank_data)]


@pytest.mark.parametrize(
    ("tp_size", "hidden_size", "quantized", "expected_name"),
    [
        (4, 2048, False, "staging"),
        (4, 2048, True, "registered"),
        (2, 5120, False, "registered"),
        (2, 5120, True, "registered"),
    ],
)
def test_ar_generic_registered_transport_uses_contract_policy(
    monkeypatch,
    tp_size,
    hidden_size,
    quantized,
    expected_name,
):
    import vllm_musa.distributed.device_communicators.musa_jit_custom_all_reduce as ar
    from vllm_musa.optimization_contract.car_rmsnorm import (
        FUSED_ALLREDUCE_RMSNORM_MODEL_FAMILY,
        can_use_registered_graph_input_for_generic_car,
    )

    impl = ar._MusaJitCustomAllreduceImpl.__new__(ar._MusaJitCustomAllreduceImpl)
    staging_rank_data = object()
    registered_rank_data = object()
    impl.buffer_rank_data = staging_rank_data
    impl.signal_ptrs_cpu = object()
    impl.world_size = tp_size
    impl.rank = 0
    impl.max_size = 64 * 1024
    impl.meta_ptrs = list(range(101, 101 + tp_size))
    impl.buffer_ptrs = list(range(201, 201 + tp_size))
    impl._graph_registered_input_enabled = True
    impl._generic_graph_registered_input_enabled = (
        can_use_registered_graph_input_for_generic_car(
            tp_size=tp_size,
            hidden_size=hidden_size,
            model_family=FUSED_ALLREDUCE_RMSNORM_MODEL_FAMILY,
            quantized=quantized,
        )
    )
    impl._use_graph_registered_inputs = True
    impl._IS_CAPTURING = True
    impl._is_current_stream_capturing = lambda: True
    impl._use_graph_staging_arena = False
    impl._require_custom_ar_graph_path = lambda: None
    impl._graph_rank_data_for_input = lambda _input: registered_rank_data

    launches = []
    monkeypatch.setattr(ar.jit_ar, "preferred_shot", lambda *_args: 1)
    monkeypatch.setattr(
        ar.jit_ar,
        "launch_graph_registered",
        lambda *args: launches.append(("registered", args[0])),
    )
    monkeypatch.setattr(
        ar.jit_ar,
        "launch_unregistered",
        lambda *args: launches.append(("staging", args[0])),
    )

    output = impl._custom_all_reduce_impl(
        torch.ones((1, hidden_size), dtype=torch.bfloat16)
    )

    assert output.shape == (1, hidden_size)
    expected_rank_data = (
        registered_rank_data if expected_name == "registered" else staging_rank_data
    )
    assert launches == [(expected_name, expected_rank_data)]
