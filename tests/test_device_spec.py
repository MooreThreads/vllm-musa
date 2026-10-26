# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the per-architecture MUSA hardware spec.

These run without a MUSA device: the dataclass is the contract, and the
device-probing helper is exercised through a fake properties object.
"""

from __future__ import annotations

import sys
import types

import pytest

from torchada.musa import build as ta_build
from torchada.musa import device as ta_device

from vllm_musa import device_spec
from vllm_musa.device_spec import (
    NATIVE_FP8_ARCHS,
    VALIDATED_MUSA_ARCHS,
    MusaDeviceSpec,
    MusaFeature,
    get_device_spec,
)


def make_spec(**overrides) -> MusaDeviceSpec:
    base = dict(
        device_capability=(3, 1),
        arch_code=310,
        arch_name="mp_31",
        mp_count=56,
        warp_size=32,
        total_memory=85804969984,
        l2_cache_size=60 * 1024 * 1024,
        shared_memory_per_block=196608,
        shared_memory_per_multiprocessor=196608,
        regs_per_multiprocessor=131072,
        max_threads_per_block=1024,
        max_threads_per_multiprocessor=3072,
        device_name="MTT S5000",
        uuid=None,
    )
    base.update(overrides)
    return MusaDeviceSpec(**base)


def test_s5000_spec_is_validated_and_has_native_fp8() -> None:
    spec = make_spec()

    assert spec.is_validated
    assert spec.supports_native_fp8
    assert spec.supports(MusaFeature.VALIDATED_TARGET)
    assert spec.supports(MusaFeature.NATIVE_FP8)


@pytest.mark.parametrize(
    "major,minor,arch_code,arch_name",
    [
        (2, 2, 220, "mp_22"),
        (3, 2, 320, "mp_32"),
        (4, 2, 420, "mp_42"),
    ],
)
def test_unvalidated_arch_fails_closed(
    major: int, minor: int, arch_code: int, arch_name: str
) -> None:
    """An arch nobody has run the S5000 kernels on gets the generic path.

    This is the property the old ``== (3, 1)`` comparisons only had by
    accident: any tuple other than the literal was rejected, including arches
    newer than S5000. Here it is explicit and per-capability.
    """
    spec = make_spec(
        device_capability=(major, minor),
        arch_code=arch_code,
        arch_name=arch_name,
        warp_size=128 if arch_code != 310 else 32,
    )

    assert not spec.is_validated
    assert not spec.supports_native_fp8
    assert not spec.supports(MusaFeature.VALIDATED_TARGET)


def test_arch_code_and_name_derivation() -> None:
    """Both spellings of an arch come from the shared module, not a local copy."""
    assert device_spec.arch_code(3, 1) == 310
    assert device_spec.arch_code(4, 2) == 420
    assert device_spec.arch_name(2, 2) == "mp_22"


def test_the_compile_key_ignores_the_multiprocessor_count() -> None:
    """Codegen input is arch plus compiler; SM count only moves the tiling.

    Two bins of one arch compile the same source to the same instructions, so
    folding mp_count into the JIT directory would fragment the cache without
    changing a single object file.  Tuning that does depend on the bin lives in
    the tactics table, which is keyed separately.
    """
    a = make_spec(mp_count=48)
    b = make_spec(mp_count=56)

    assert a.arch_name == b.arch_name
    assert not hasattr(a, "cache_tag")


def test_explicit_sets_are_the_source_of_truth() -> None:
    """Guard against someone re-deriving capability from a numeric order."""
    assert 310 in VALIDATED_MUSA_ARCHS
    assert 310 in NATIVE_FP8_ARCHS
    # 320/420 are newer *numbers* and still not validated targets.
    assert 320 not in VALIDATED_MUSA_ARCHS
    assert 420 not in VALIDATED_MUSA_ARCHS


def test_unknown_feature_is_rejected() -> None:
    class NotAFeature:
        pass

    with pytest.raises(ValueError):
        make_spec().supports(NotAFeature())  # type: ignore[arg-type]


def test_get_device_spec_builds_from_device_properties(monkeypatch) -> None:
    class FakeProps:
        major = 3
        minor = 1
        multi_processor_count = 56
        warp_size = 32
        total_memory = 85804969984
        L2_cache_size = 60 * 1024 * 1024
        shared_memory_per_block = 196608
        shared_memory_per_multiprocessor = 196608
        regs_per_multiprocessor = 131072
        max_threads_per_block = 1024
        max_threads_per_multi_processor = 3072
        name = "MTT S5000"
        uuid = None

    monkeypatch.setattr(ta_device, "_get_properties", lambda device_id: FakeProps())
    device_spec.clear_device_spec_cache()

    spec = get_device_spec(0)

    assert spec is not None
    assert spec.device_capability == (3, 1)
    assert spec.arch_code == 310
    assert spec.arch_name == "mp_31"
    assert spec.mp_count == 56
    assert spec.max_threads_per_multiprocessor == 3072
    device_spec.clear_device_spec_cache()


def test_get_device_spec_returns_none_when_probe_fails(monkeypatch) -> None:
    """A failed probe must be "unknown", never a guessed arch."""

    def boom(device_id: int):
        raise RuntimeError("device not ready")

    monkeypatch.setattr(ta_device, "_get_properties", boom)
    device_spec.clear_device_spec_cache()

    assert get_device_spec(0) is None
    device_spec.clear_device_spec_cache()


# --- regression guard: no capability literals outside device_spec -----------

# Files that legitimately mention a capability literal. Each entry must be
# either the identity check inside device_spec, a per-arch data default in the
# MoE dispatch policy, or upstream code already overridden for MUSA by an
# explicit `is_musa()` branch (documented next to each site).
_LITERAL_ALLOWLIST = {
    "vllm_musa/device_spec.py",
    "vllm_musa/model_executor/layers/fused_moe/dispatch_policy.py",
    "vllm_musa/v1/attention/backends/flash_attn.py",  # MUSA early-returns first
    "vllm_musa/v1/attention/backends/mla/common.py",  # _pad_v forced off on MUSA
}


def test_no_new_capability_literal_comparisons() -> None:
    """Arch selection must go through device_spec, not a tuple literal.

    ``== (3, 1)`` happens to be correct today, but it silently means "the one
    arch we validated" and it fails open for nobody: a new arch falls off the
    S5000 path *and* off the capability path at the same time. Routing through
    ``MusaDeviceSpec`` keeps the decision in one place and forces unvalidated
    arches onto the generic path with a warning.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for path in (root / "vllm_musa").rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel in _LITERAL_ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            if any(
                token in line
                for token in (
                    "== (3, 1)",
                    "== (3,1)",
                    "has_device_capability((",
                    "is_device_capability((",
                )
            ):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, (
        "capability literals must be replaced by vllm_musa.device_spec:\n"
        + "\n".join(offenders)
    )


def test_a_failed_probe_is_not_cached(monkeypatch) -> None:
    """A probe that fails before the device is ready must be retried.

    Caching the failure would pin every arch gate to the generic path for the
    rest of the process -- FP8, gated QKV, MoE tactics, min-p -- with only a
    debug line to show for it.
    """
    calls = {"n": 0}

    class Properties:
        major = 3
        minor = 1
        name = "MTT S5000"
        multi_processor_count = 56
        warp_size = 32
        total_memory = 80 << 30
        L2_cache_size = 8 << 20
        shared_memory_per_block = 48 << 10
        shared_memory_per_multiprocessor = 256 << 10
        regs_per_multiprocessor = 65536
        max_threads_per_block = 1024
        max_threads_per_multiprocessor = 2048
        uuid = None

    def flaky(device_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("device not ready")
        return Properties()

    ta_device._probe_device_spec.cache_clear()
    monkeypatch.setattr(ta_device, "_get_properties", flaky)
    monkeypatch.setattr(device_spec, "read_build_stamp", lambda: None)

    assert device_spec.get_device_spec(0) is None
    recovered = device_spec.get_device_spec(0)
    assert calls["n"] == 2
    assert recovered is not None
    assert recovered.arch_name == "mp_31"


def test_a_build_target_mismatch_is_not_swallowed(monkeypatch) -> None:
    """A wrong-arch wheel must not be reported as "no validated device".

    The check lives in ``vllm_musa.device_spec`` (the shared module only compares
    when asked), so this drives the real path: a stamp for another arch plus a
    probe that succeeds.  Folding the mismatch into ``False`` would leave the
    wrong binary running the generic kernels of the same wrong binary.
    """

    class Props:
        major = 3
        minor = 1
        name = "MTT S5000"
        multi_processor_count = 56
        warp_size = 32
        total_memory = 80 << 30
        L2_cache_size = 8 << 20
        shared_memory_per_block = 48 << 10
        shared_memory_per_multiprocessor = 256 << 10
        regs_per_multiprocessor = 65536
        max_threads_per_block = 1024
        max_threads_per_multiprocessor = 2048
        uuid = None

    ta_device._probe_device_spec.cache_clear()
    monkeypatch.setattr(ta_device, "_get_properties", lambda device_id: Props())
    stamp = types.ModuleType("vllm_musa._build_stamp")
    stamp.MTGPU_TARGETS = ("mp_22",)
    stamp.ENABLE_FP8 = False
    monkeypatch.setitem(sys.modules, "vllm_musa._build_stamp", stamp)

    with pytest.raises(device_spec.MusaBuildTargetMismatch, match="mp_22"):
        device_spec.get_device_spec(0)
    with pytest.raises(device_spec.MusaBuildTargetMismatch, match="mp_22"):
        device_spec.is_validated_device(0)
