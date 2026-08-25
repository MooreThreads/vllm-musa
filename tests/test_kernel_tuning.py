# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_musa import tuning

PLATFORM_SOURCE = Path(__file__).resolve().parents[1] / "vllm_musa" / "platform.py"


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


def test_query_kernel_hardware_fails_closed_on_unexpected_probe_error(monkeypatch) -> None:
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


def test_platform_publishes_resolved_scheduler_profile() -> None:
    source = PLATFORM_SOURCE.read_text(encoding="utf-8")
    assert "configure_musa_engine_scheduler_profile(" in source
    assert '"max_num_seqs"' in source


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
