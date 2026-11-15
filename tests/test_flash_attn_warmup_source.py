# SPDX-License-Identifier: Apache-2.0
"""Bounded FA3 prewarm covers actual paged runtime variants, not model names."""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vllm_musa/model_executor/warmup/flash_attn_warmup.py"


@pytest.fixture
def warmup(monkeypatch):
    logger = ModuleType("vllm.logger")
    logger.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "vllm.logger", logger)
    spec = importlib.util.spec_from_file_location("fa3_warmup_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kernel_config(query, ratio, head, head_v, element_size, *args):
    pack = query <= 256
    effective = query * ratio if pack else query
    tile = next(
        (tile for tile in (16, 32, 64, 128, 192, 256) if effective <= tile), 256
    )
    consumers = (tile + 63) // 64
    return tile, 64, 3, 3, head, head_v, consumers, consumers, 0, pack


def signature(config):
    return tuple(sorted(config.items()))


def test_runtime_tiles_and_mixed_decode_contracts_are_covered(warmup):
    forward, metadata = warmup._fa3_warmup_configs(kernel_config, 32, 4, 128, 4096, 64)
    assert len(forward) == len({signature(row) for row in forward})
    variants = {
        (row["tile_m"], row["is_packgqa"], row["has_metadata"], row["has_q_descale"])
        for row in forward
    }
    # Mixed decode omits scheduler metadata and Q descale.
    assert (16, True, False, False) in variants
    assert (16, True, False, True) in variants
    assert (16, True, True, True) in variants
    for tile in (16, 32, 64, 128, 192, 256):
        assert (tile, True, True, True) in variants
    assert (256, False, True, True) in variants
    assert len(forward) <= 10
    assert {(row["is_packgqa"], row["num_warps"]) for row in metadata} == {
        (pack, warps) for pack in (False, True) for warps in (1, 2, 3)
    }
    assert all(row["has_k_descale"] and row["has_v_descale"] for row in forward)
    assert all(row["element"] == "mutlass::bfloat16_t" for row in forward)


def test_scheduler_limits_bound_reachable_variants(warmup):
    forward, metadata = warmup._fa3_warmup_configs(kernel_config, 32, 4, 128, 8, 4)
    assert {row["tile_m"] for row in forward} == {16, 32, 64}
    assert all(row["is_packgqa"] for row in forward)
    assert all(row["num_warps"] == 1 for row in metadata)


def test_query_representatives_cover_every_runtime_dispatch(warmup):
    for limit in (1, 8, 64, 256, 257, 4096):
        points = warmup._query_points(limit, 8)
        assert all(1 <= point <= limit for point in points)
        actual = {kernel_config(query, 8, 128, 128, 2) for query in range(1, limit + 1)}
        selected = {kernel_config(query, 8, 128, 128, 2) for query in points}
        assert actual == selected
        assert len(points) <= 22


def test_loads_real_dispatch_modules_without_tensor_launches(warmup):
    calls = []
    forward = ModuleType("fake_fwd")
    forward._get_fwd_kernel_config = kernel_config
    forward._fmha_fwd_module = lambda config: calls.append(
        ("forward", signature(config))
    )
    metadata = ModuleType("fake_metadata")
    metadata._fmha_metadata_module = lambda config: calls.append(
        ("metadata", signature(config))
    )
    warmup._load_fa3_modules(32, 4, 128, 4096, 64, forward, metadata)
    assert len(calls) <= 16
    assert len(calls) == len(set(calls))
    assert {row[0] for row in calls} == {"forward", "metadata"}


def impl(**overrides):
    cls = type("FlashAttentionImpl", (), {})
    cls.__module__ = "vllm_musa.v1.attention.backends.flash_attn"
    value = cls()
    fields = dict(
        num_heads=32,
        num_kv_heads=4,
        head_size=128,
        kv_cache_dtype="auto",
        vllm_flash_attn_version=3,
        attn_type="decoder",
        sinks=None,
        alibi_slopes=None,
        logits_soft_cap=0,
        sliding_window=(-1, -1),
    )
    fields.update(overrides)
    for key, item in fields.items():
        setattr(value, key, item)
    return value


def test_geometry_gate_is_narrow_and_deduplicates_layers(warmup):
    layer = lambda implementation: SimpleNamespace(impl=implementation, head_size_v=128)
    model = SimpleNamespace(
        modules=lambda: iter(
            [
                layer(impl()),
                layer(impl()),
                layer(impl(kv_cache_dtype="fp8")),
                layer(impl(head_size=64)),
                layer(impl(num_heads=16)),
                layer(impl(attn_type="encoder")),
                layer(impl(sliding_window=(1023, 0))),
                layer(impl(sinks=object())),
                layer(impl(logits_soft_cap=50)),
                object(),
            ]
        )
    )
    assert warmup._model_fa3_geometries(model) == {(32, 4, 128)}


def test_retained_config_works_outside_global_context(warmup, monkeypatch):
    model = SimpleNamespace(modules=lambda: iter([SimpleNamespace(impl=impl())]))
    context = SimpleNamespace(
        model_config=SimpleNamespace(dtype="bf16"),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=64),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=64),
    )
    monkeypatch.setattr(warmup, "_runtime_context", lambda: (None, "bf16", 0))
    calls = []

    def loaded(*args):
        calls.append(args)
        return 9, 6

    monkeypatch.setattr(warmup, "_load_fa3_modules", loaded)
    warmup.warmup_flash_attention(model, 4096, config=context)
    assert calls == [(32, 4, 128, 4096, 64)]


def test_failed_build_does_not_poison_success_cache(warmup, monkeypatch):
    calls = []
    model = SimpleNamespace(
        modules=lambda: iter([SimpleNamespace(impl=impl(), head_size_v=128)])
    )
    context = SimpleNamespace(
        model_config=SimpleNamespace(dtype="bf16"),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=64),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=64),
    )
    warmup._runtime_context = lambda: (context, "bf16", "device0")

    def fail(*args):
        calls.append(args)
        raise RuntimeError("injected compile failure")

    monkeypatch.setattr(warmup, "_load_fa3_modules", fail)
    with pytest.raises(RuntimeError, match="injected compile failure"):
        warmup.warmup_flash_attention(model, 4096)
    assert not warmup._WARMED

    def loaded(*args):
        calls.append(args)
        return 9, 6

    monkeypatch.setattr(warmup, "_load_fa3_modules", loaded)
    warmup.warmup_flash_attention(model, 4096)
    warmup.warmup_flash_attention(model, 4096)
    assert len(calls) == 2
    assert len(warmup._WARMED) == 1
