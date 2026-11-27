# SPDX-License-Identifier: Apache-2.0
"""Tests for the default DeepSeek-V4 TileLang MUSA compilation path."""

from vllm_musa.deepseek_v4_jit import kernel_common


class _PassConfigKey:
    TL_DEVICE_COMPILE_FLAGS = "compile_flags"
    TL_DISABLE_HOST_ASSERTS = "disable_host_asserts"
    TL_ENABLE_MUSA_BURST = "burst"
    TL_ENABLE_REDUCE_BURST = "reduce_burst"
    TL_DISABLE_THREAD_STORAGE_SYNC = "disable_thread_storage_sync"
    TL_DISABLE_SAFE_MEMORY_ACCESS = "disable_safe_memory"
    TL_ENABLE_LOWER_LDGSTG = "lower_ldgstg"
    TL_ENABLE_LOWER_LDGSTG_PREDICATED = "lower_ldgstg_predicated"
    TL_DISABLE_INDEX_TYPE_PROMOTION = "disable_index_promotion"


class _TileLangStub:
    PassConfigKey = _PassConfigKey


def test_default_compile_profile_is_dsa_full():
    flags = kernel_common._tilelang_musa_compile_profile_flags()
    assert flags is not None
    assert "-mtgpu-combine-instr-with-burst=1" in flags


def test_default_pass_config_is_optimized():
    configs = kernel_common._tilelang_musa_pass_configs(_TileLangStub())
    assert configs["disable_host_asserts"] is True
    assert "-mtgpu-combine-instr-with-burst=1" in configs["compile_flags"]


def test_burst_and_aggressive_helpers_keep_validated_flags():
    burst = kernel_common._tilelang_musa_burst_reduce_pass_configs(_TileLangStub())
    assert burst["burst"] is True
    assert burst["reduce_burst"] is True
    assert burst["disable_host_asserts"] is True

    aggressive = kernel_common._tilelang_musa_aggressive_pass_configs(_TileLangStub())
    assert aggressive["disable_thread_storage_sync"] is True
    assert aggressive["disable_safe_memory"] is True
    assert aggressive["lower_ldgstg"] is True
    assert aggressive["lower_ldgstg_predicated"] is True
    assert aggressive["disable_index_promotion"] is True
    assert aggressive["disable_host_asserts"] is True


def test_tilelang_jit_uses_default_pass_config():
    class Stub(_TileLangStub):
        def jit(self, **kwargs):
            return kwargs

    result = kernel_common._tilelang_jit(Stub(), "kernel_name")
    assert result["name"] == "kernel_name"
    assert result["target"] == "musa"
    assert result["pass_configs"]["disable_host_asserts"] is True
