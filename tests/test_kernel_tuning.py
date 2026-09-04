# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vllm_musa import tuning

ROOT = Path(__file__).parents[1]


def test_kernel_hardware_cache_key_keeps_mp_count_exact() -> None:
    mp56 = tuning.MusaKernelHardware((3, 1), 56)
    mp60 = tuning.MusaKernelHardware((3, 1), 60)

    assert mp56.cache_key == "mp31-mps56"
    assert mp60.cache_key == "mp31-mps60"
    assert mp56.cache_key != mp60.cache_key
    assert tuning.UNKNOWN_MUSA_KERNEL_HARDWARE.cache_key == "unknown"


def test_query_kernel_hardware_reads_device_properties(monkeypatch) -> None:
    monkeypatch.setattr(
        tuning,
        "_get_musa_device_properties",
        lambda _index: SimpleNamespace(
            major=3,
            minor=1,
            multi_processor_count=56,
        ),
    )

    assert tuning.query_musa_kernel_hardware(0) == tuning.MusaKernelHardware((3, 1), 56)


def test_query_kernel_hardware_fails_closed(monkeypatch) -> None:
    def fail(_index):
        raise RuntimeError("device query unavailable")

    monkeypatch.setattr(
        tuning,
        "_get_musa_device_properties",
        fail,
    )

    hardware = tuning.query_musa_kernel_hardware(0)
    assert hardware == tuning.UNKNOWN_MUSA_KERNEL_HARDWARE
    assert not hardware.is_known


def test_query_kernel_hardware_recovers_after_early_unknown(monkeypatch) -> None:
    results = iter(
        (
            RuntimeError("runtime not initialized"),
            SimpleNamespace(major=3, minor=1, multi_processor_count=60),
        )
    )

    def query(_index):
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(tuning, "_get_musa_device_properties", query)

    assert tuning.query_musa_kernel_hardware(0).cache_key == "unknown"
    assert tuning.query_musa_kernel_hardware(0).cache_key == "mp31-mps60"


def test_cached_kernel_hardware_does_not_cache_early_unknown(monkeypatch) -> None:
    results = iter(
        (
            tuning.UNKNOWN_MUSA_KERNEL_HARDWARE,
            tuning.MusaKernelHardware((3, 1), 60),
        )
    )
    calls = 0

    def query(_index):
        nonlocal calls
        calls += 1
        return next(results)

    monkeypatch.setattr(tuning, "_MUSA_KERNEL_HARDWARE_CACHE", {})
    monkeypatch.setattr(tuning, "query_musa_kernel_hardware", query)

    assert tuning.query_cached_musa_kernel_hardware(0).cache_key == "unknown"
    assert tuning.query_cached_musa_kernel_hardware(0).cache_key == "mp31-mps60"
    assert tuning.query_cached_musa_kernel_hardware(0).cache_key == "mp31-mps60"
    assert calls == 2


def test_primed_kernel_hardware_freezes_unknown(monkeypatch) -> None:
    results = iter(
        (
            tuning.UNKNOWN_MUSA_KERNEL_HARDWARE,
            tuning.MusaKernelHardware((3, 1), 60),
        )
    )
    calls = 0

    def query(_index):
        nonlocal calls
        calls += 1
        return next(results)

    monkeypatch.setattr(tuning, "_FROZEN_MUSA_KERNEL_HARDWARE", {})
    monkeypatch.setattr(tuning, "query_musa_kernel_hardware", query)

    assert tuning.prime_musa_kernel_hardware(0).cache_key == "unknown"
    assert tuning.prime_musa_kernel_hardware(0).cache_key == "unknown"
    assert tuning.get_primed_musa_kernel_hardware(0).cache_key == "unknown"
    assert calls == 1


def test_primed_kernel_hardware_is_exact_per_device(monkeypatch) -> None:
    monkeypatch.setattr(tuning, "_FROZEN_MUSA_KERNEL_HARDWARE", {})
    monkeypatch.setattr(
        tuning,
        "query_musa_kernel_hardware",
        lambda index: tuning.MusaKernelHardware((3, 1), 56 + index * 4),
    )

    assert tuning.prime_musa_kernel_hardware(0).cache_key == "mp31-mps56"
    assert tuning.prime_musa_kernel_hardware(1).cache_key == "mp31-mps60"
    assert tuning.get_primed_musa_kernel_hardware(2).cache_key == "unknown"


def test_worker_primes_hardware_after_device_init() -> None:
    worker_source = (ROOT / "vllm_musa/worker.py").read_text()
    init_source = worker_source.split("def init_device(self) -> None:", 1)[1].split(
        "def ", 1
    )[0]
    assert init_source.index("super().init_device()") < init_source.index(
        "prime_musa_kernel_hardware"
    )


def test_query_kernel_hardware_fails_closed_on_unexpected_probe_error(
    monkeypatch,
) -> None:
    def raise_unexpected(_device_index):
        raise OSError("transient MUSA property query failure")

    monkeypatch.setattr(tuning, "_get_musa_device_properties", raise_unexpected)
    assert tuning.query_musa_kernel_hardware(0) == tuning.UNKNOWN_MUSA_KERNEL_HARDWARE


def test_engine_scheduler_profile_is_fail_closed(monkeypatch) -> None:
    monkeypatch.delenv(tuning._ENGINE_MAX_NUM_SEQS_ENV, raising=False)
    assert tuning.query_musa_engine_max_num_seqs() is None

    tuning.configure_musa_engine_scheduler_profile(4)
    assert tuning.query_musa_engine_max_num_seqs() == 4

    monkeypatch.setenv(tuning._ENGINE_MAX_NUM_SEQS_ENV, "invalid")
    assert tuning.query_musa_engine_max_num_seqs() is None

    tuning.configure_musa_engine_scheduler_profile(None)
    assert tuning.query_musa_engine_max_num_seqs() is None


def test_platform_publishes_resolved_scheduler_profile(monkeypatch) -> None:
    monkeypatch.delenv(tuning._ENGINE_MAX_NUM_SEQS_ENV, raising=False)

    tuning.configure_musa_engine_scheduler_profile_from_config(
        SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=64))
    )
    assert tuning.query_musa_engine_max_num_seqs() == 64

    tuning.configure_musa_engine_scheduler_profile_from_config(SimpleNamespace())
    assert tuning.query_musa_engine_max_num_seqs() is None


def test_forward_graph_bucket_query_is_validated_and_fail_closed(monkeypatch) -> None:
    vllm_module = ModuleType("vllm")
    vllm_module.__path__ = []
    forward_context = ModuleType("vllm.forward_context")
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", forward_context)

    forward_context.is_forward_context_available = lambda: False
    forward_context.get_forward_context = lambda: SimpleNamespace(
        batch_descriptor=SimpleNamespace(
            num_tokens=1,
            num_reqs=1,
            uniform=True,
            has_lora=False,
            num_active_loras=0,
        ),
        cudagraph_runtime_mode=SimpleNamespace(name="FULL"),
    )
    assert tuning.query_musa_forward_graph_bucket() is None

    forward_context.is_forward_context_available = lambda: True
    forward_context.get_forward_context = lambda: SimpleNamespace(
        batch_descriptor=SimpleNamespace(
            num_tokens=1,
            num_reqs=1,
            uniform=True,
            has_lora=False,
            num_active_loras=0,
        ),
        cudagraph_runtime_mode=SimpleNamespace(name="FULL"),
    )
    bucket = tuning.query_musa_forward_graph_bucket()
    assert bucket is not None
    assert bucket.num_tokens == 1
    assert bucket.runtime_mode == "FULL"
    assert bucket.has_lora is False
    assert bucket.num_active_loras == 0

    for descriptor in (
        None,
        SimpleNamespace(
            num_tokens=0,
            num_reqs=1,
            uniform=True,
            has_lora=False,
            num_active_loras=0,
        ),
        SimpleNamespace(
            num_tokens=4,
            num_reqs=0,
            uniform=True,
            has_lora=False,
            num_active_loras=0,
        ),
        SimpleNamespace(
            num_tokens=4,
            num_reqs=4,
            uniform=1,
            has_lora=False,
            num_active_loras=0,
        ),
    ):
        forward_context.get_forward_context = lambda descriptor=descriptor: (
            SimpleNamespace(
                batch_descriptor=descriptor,
                cudagraph_runtime_mode=SimpleNamespace(name="FULL"),
            )
        )
        assert tuning.query_musa_forward_graph_bucket().present

    for mode, has_lora, active_loras in (("PIECEWISE", False, 0), ("FULL", True, 1)):
        forward_context.get_forward_context = (
            lambda mode=mode,
            has_lora=has_lora,
            active_loras=active_loras: SimpleNamespace(
                batch_descriptor=SimpleNamespace(
                    num_tokens=1,
                    num_reqs=1,
                    uniform=True,
                    has_lora=has_lora,
                    num_active_loras=active_loras,
                ),
                cudagraph_runtime_mode=SimpleNamespace(name=mode),
            )
        )
        bucket = tuning.query_musa_forward_graph_bucket()
        assert bucket.present
        assert bucket.runtime_mode == mode

    forward_context.get_forward_context = lambda: SimpleNamespace(
        batch_descriptor=SimpleNamespace(
            num_tokens=1,
            num_reqs=1,
            uniform=True,
            has_lora=False,
            num_active_loras=0,
        ),
        cudagraph_runtime_mode=SimpleNamespace(name="FULL_DECODE_ONLY"),
    )
    assert tuning.query_musa_forward_graph_bucket() == (
        tuning.MusaForwardGraphBucket.invalid()
    )


def test_forward_graph_bucket_api_drift_is_present_invalid(monkeypatch) -> None:
    vllm_module = ModuleType("vllm")
    vllm_module.__path__ = []
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.delitem(sys.modules, "vllm.forward_context", raising=False)

    bucket = tuning.query_musa_forward_graph_bucket()
    assert bucket is not None
    assert bucket.present
    assert bucket.num_tokens is None
    assert bucket.runtime_mode is None


def test_forward_graph_bucket_matches_pinned_vllm_context_objects() -> None:
    forward_context = pytest.importorskip("vllm.forward_context")
    vllm_config = pytest.importorskip("vllm.config")

    cases = (
        (
            vllm_config.CUDAGraphMode.FULL,
            forward_context.BatchDescriptor(1, 1, True, False, 0),
            (1, 1, True, "FULL", False, 0),
        ),
        (
            vllm_config.CUDAGraphMode.PIECEWISE,
            forward_context.BatchDescriptor(4, None, False, False, 0),
            (4, None, False, "PIECEWISE", False, 0),
        ),
        (
            vllm_config.CUDAGraphMode.NONE,
            forward_context.BatchDescriptor(4),
            (4, None, False, "NONE", False, 0),
        ),
        (
            vllm_config.CUDAGraphMode.FULL,
            forward_context.BatchDescriptor(1, 1, True, True, 1),
            (1, 1, True, "FULL", True, 1),
        ),
    )
    for mode, descriptor, expected in cases:
        context = forward_context.ForwardContext(
            {},
            {},
            {},
            cudagraph_runtime_mode=mode,
            batch_descriptor=descriptor,
        )
        with forward_context.override_forward_context(context):
            bucket = tuning.query_musa_forward_graph_bucket()
        assert bucket is not None
        assert (
            bucket.num_tokens,
            bucket.num_reqs,
            bucket.uniform,
            bucket.runtime_mode,
            bucket.has_lora,
            bucket.num_active_loras,
        ) == expected


@pytest.mark.parametrize(
    ("multiprocessor_count", "expected_waves", "expected_tail", "expected_util"),
    [
        (52, 3, 16, 120 / 156),
        (56, 3, 8, 120 / 168),
        (60, 2, 60, 1.0),
        (64, 2, 56, 120 / 128),
    ],
)
def test_wave_quantization_exposes_core_bin_discontinuity(
    multiprocessor_count: int,
    expected_waves: int,
    expected_tail: int,
    expected_util: float,
) -> None:
    estimate = tuning.estimate_wave_quantization(120, multiprocessor_count)

    assert estimate.waves == expected_waves
    assert estimate.tail_tiles == expected_tail
    assert estimate.overall_slot_utilization == pytest.approx(expected_util)


def test_wave_quantization_validates_inputs() -> None:
    with pytest.raises(ValueError, match="total_tiles"):
        tuning.estimate_wave_quantization(-1, 60)
    with pytest.raises(ValueError, match="multiprocessor_count"):
        tuning.estimate_wave_quantization(1, 0)
    with pytest.raises(ValueError, match="blocks_per_multiprocessor"):
        tuning.estimate_wave_quantization(1, 60, blocks_per_multiprocessor=0)


def _dsv4_m5_weighted_rmsnorm_tactic(
    hardware: tuning.MusaKernelHardware,
    **overrides,
):
    contract = {
        "hardware": hardware,
        "rows": 5,
        "hidden_size": 4096,
        "input_dtype": "torch.bfloat16",
        "weight_dtype": "torch.bfloat16",
        "contiguous": True,
    }
    contract.update(overrides)
    return tuning.select_mhc_weighted_rmsnorm_tactic(**contract)


def test_mp48_dsv4_m5_weighted_rmsnorm_uses_exact_jit_tactic():
    assert _dsv4_m5_weighted_rmsnorm_tactic(
        tuning.MusaKernelHardware((3, 1), 48)
    ) == tuning.MusaMhcWeightedRmsnormTactic(
        threads=256,
        source="mp48-dsv4-mtp4-m5-wrms-256t-v1",
    )


@pytest.mark.parametrize(
    ("hardware", "overrides"),
    [
        (tuning.UNKNOWN_MUSA_KERNEL_HARDWARE, {}),
        (tuning.MusaKernelHardware((3, 0), 48), {}),
        (tuning.MusaKernelHardware((3, 1), 56), {}),
        (tuning.MusaKernelHardware((3, 1), 60), {}),
        (tuning.MusaKernelHardware((3, 1), 48), {"rows": 1}),
        (tuning.MusaKernelHardware((3, 1), 48), {"rows": 20}),
        (tuning.MusaKernelHardware((3, 1), 48), {"hidden_size": 5120}),
        (
            tuning.MusaKernelHardware((3, 1), 48),
            {"input_dtype": "torch.float16"},
        ),
        (
            tuning.MusaKernelHardware((3, 1), 48),
            {"weight_dtype": "torch.float32"},
        ),
        (tuning.MusaKernelHardware((3, 1), 48), {"contiguous": False}),
    ],
)
def test_weighted_rmsnorm_tactic_mismatch_falls_back(hardware, overrides):
    assert _dsv4_m5_weighted_rmsnorm_tactic(hardware, **overrides) is None


@pytest.mark.parametrize(
    ("mode", "rows"),
    [
        ("plain", 512),
        ("gemma", 512),
        ("gemma", 4096),
        ("fused_gemma", 512),
    ],
)
def test_mp60_h5120_uses_exact_jit_rmsnorm_tactic(mode, rows):
    assert tuning.select_jit_rmsnorm_tactic(
        hardware=tuning.MusaKernelHardware((3, 1), 60),
        mode=mode,
        rows=rows,
        hidden_size=5120,
        input_dtype="torch.bfloat16",
        weight_dtype="torch.bfloat16",
        contiguous=True,
    ) == tuning.MusaJitRMSNormTactic(
        block_threads=320,
        source=("mp60-jit-rmsnorm-h5120-320t-" f"{mode}-m{rows}-v1"),
    )


@pytest.mark.parametrize("mode", ["plain", "gemma", "fused", "fused_gemma"])
def test_mp56_h5120_uses_exact_jit_rmsnorm_tactic(mode):
    assert tuning.select_jit_rmsnorm_tactic(
        hardware=tuning.MusaKernelHardware((3, 1), 56),
        mode=mode,
        rows=512,
        hidden_size=5120,
        input_dtype="torch.bfloat16",
        weight_dtype="torch.bfloat16",
        contiguous=True,
    ) == tuning.MusaJitRMSNormTactic(
        block_threads=320,
        source=("mp56-jit-rmsnorm-h5120-320t-" f"{mode}-m512-v1"),
    )


def test_mp48_plain_h5120_uses_exact_jit_rmsnorm_tactic():
    assert tuning.select_jit_rmsnorm_tactic(
        hardware=tuning.MusaKernelHardware((3, 1), 48),
        mode="plain",
        rows=512,
        hidden_size=5120,
        input_dtype="torch.bfloat16",
        weight_dtype="torch.bfloat16",
        contiguous=True,
    ) == tuning.MusaJitRMSNormTactic(
        block_threads=256,
        source="mp48-jit-rmsnorm-h5120-256t-plain-m512-v1",
    )


@pytest.mark.parametrize(
    ("hardware", "overrides"),
    [
        (tuning.UNKNOWN_MUSA_KERNEL_HARDWARE, {}),
        (tuning.MusaKernelHardware((3, 0), 60), {}),
        (tuning.MusaKernelHardware((3, 1), 48), {"mode": "gemma"}),
        (tuning.MusaKernelHardware((3, 1), 48), {"mode": "fused"}),
        (tuning.MusaKernelHardware((3, 1), 48), {"mode": "fused_gemma"}),
        (tuning.MusaKernelHardware((3, 1), 60), {"mode": "fused"}),
        (tuning.MusaKernelHardware((3, 1), 60), {"rows": 128}),
        (tuning.MusaKernelHardware((3, 1), 60), {"hidden_size": 4096}),
        (
            tuning.MusaKernelHardware((3, 1), 60),
            {"input_dtype": "torch.float16"},
        ),
        (tuning.MusaKernelHardware((3, 1), 60), {"contiguous": False}),
    ],
)
def test_jit_rmsnorm_tactic_mismatch_falls_back(hardware, overrides):
    contract = {
        "hardware": hardware,
        "mode": "plain",
        "rows": 512,
        "hidden_size": 5120,
        "input_dtype": "torch.bfloat16",
        "weight_dtype": "torch.bfloat16",
        "contiguous": True,
    }
    contract.update(overrides)
    assert tuning.select_jit_rmsnorm_tactic(**contract) is None
