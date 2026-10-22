# SPDX-License-Identifier: Apache-2.0
"""Source contracts for the default-on chunked min-p dispatch."""

import ast
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _load_device_guard(musa):
    source_path = ROOT / "vllm_musa/_custom_ops.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), source_path)
    helper_names = {
        "_musa_device_index",
        "_is_validated_musa_device",
    }
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    # The guard asks vllm_musa.device_spec for the device facts; provide a fake
    # that resolves them through the injected `musa` namespace so these tests
    # keep asserting *which* device id gets queried.
    @cache
    def get_device_spec(device_id: int):
        major, minor = musa.get_device_capability(device_id)
        return SimpleNamespace(
            device_capability=(major, minor),
            is_validated=(major, minor) == (3, 1),
        )

    class MusaBuildTargetMismatch(RuntimeError):
        """Stand-in for the real class: this fake never raises it."""

    namespace = {
        "Optional": Optional,
        "MusaBuildTargetMismatch": MusaBuildTargetMismatch,
        "cache": cache,
        "torch": SimpleNamespace(device=object, musa=musa),
        "get_device_spec": get_device_spec,
    }
    exec(
        compile(ast.Module(body=helpers, type_ignores=[]), source_path, "exec"),
        namespace,
    )
    return namespace["_is_validated_musa_device"]


def test_min_p_dispatch_has_no_environment_gate() -> None:
    source = _read("vllm_musa/_custom_ops.py")
    sampler = _read("vllm_musa/v1/sample/topk_topp_sampler.py")
    environ = _read("vllm_musa/utils/environ.py")

    assert "envs." not in source
    assert "CHUNKED_MIN_P_SAMPLER" not in environ
    assert "VLLM_MUSA_SAMPLER_FAST_PATH" not in sampler
    assert "VLLM_MUSA_SAMPLER_FAST_PATH" not in environ
    assert "sampler_fast_path_enabled" not in sampler


def test_min_p_dispatch_keeps_the_validated_contract_guard() -> None:
    source = _read("vllm_musa/_custom_ops.py")

    assert "def _can_use_chunked_min_p_sampler(" in source
    assert "deterministic" in source
    assert "and probs.dtype == torch.float32" in source
    assert "and probs.shape[1] in _QWEN_MIN_P_VOCABS" in source
    assert "and 0 < probs.shape[0] <= _QWEN_MIN_P_MAX_BATCH" in source
    assert "and probs.is_contiguous()" in source
    assert "and _is_validated_musa_device(probs.device)" in source
    # The arch predicate now lives in vllm_musa/device_spec.py; the guard must
    # ask the spec rather than compare a capability tuple against a literal.
    assert "spec.is_validated" in source
    assert "== (3, 1)" not in source
    assert "and _is_supported_musa_generator(generator, probs.device)" in source
    assert "musa_chunked_min_p_sampling_from_probs.default" in source
    assert "min_p_sampling_from_probs.default" in source


def test_device_guard_queries_tensor_logical_device_id() -> None:
    queried_device_ids = []
    musa = SimpleNamespace(
        current_device=lambda: 0,
        device_count=lambda: 2,
        get_device_capability=lambda device_id: (
            queried_device_ids.append(device_id) or (3, 1)
        ),
    )
    is_validated = _load_device_guard(musa)

    assert is_validated(SimpleNamespace(type="musa", index=1))
    assert is_validated(SimpleNamespace(type="musa", index=1))
    assert queried_device_ids == [1]


def test_device_guard_does_not_cache_runtime_query_failures() -> None:
    queried_device_ids = []

    def fail_capability_query(device_id: int) -> tuple[int, int]:
        queried_device_ids.append(device_id)
        if len(queried_device_ids) == 1:
            raise RuntimeError("capability query failed")
        return (3, 1)

    musa = SimpleNamespace(
        current_device=lambda: 0,
        device_count=lambda: 2,
        get_device_capability=fail_capability_query,
    )
    is_validated = _load_device_guard(musa)

    assert not is_validated(SimpleNamespace(type="musa", index=1))
    assert is_validated(SimpleNamespace(type="musa", index=1))
    assert queried_device_ids == [1, 1]


def test_min_p_dispatch_preserves_unsupported_input_fallbacks() -> None:
    source = _read("vllm_musa/_custom_ops.py")

    assert "indices.dtype == torch.int32" in source
    assert "indices.is_contiguous()" in source
    assert "maybe_min_p_arr.dtype == torch.float32" in source
    assert "maybe_min_p_arr.is_contiguous()" in source
    assert "input_probs = probs" in source
    assert "input_min_p_arr = maybe_min_p_arr" in source


def test_min_p_kernel_uses_graph_safe_philox_state() -> None:
    source = _read("csrc/musa/min_p_sampler.mu")

    assert "MUSAGraphsUtils.muh" in source
    assert "at::PhiloxMusaState philox_state" in source
    assert "at::musa::philox::unpack(philox_state)" in source
    assert "philox_state.seed_.val" not in source
    assert "philox_state.offset_.val" not in source
    assert source.index("OptionalMUSAGuard device_guard") < source.index(
        "getDefaultMUSAGenerator"
    )
    assert "getDefaultMUSAGenerator(probs.get_device())" in source
