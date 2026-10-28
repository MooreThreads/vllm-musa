# SPDX-License-Identifier: Apache-2.0
"""The tuning problem: frozen, validated, and free of any device I/O."""

from __future__ import annotations

import pytest

from vllm_musa.tactics import (
    ComputeSemantics,
    DeviceProfile,
    LayerShape,
    TuningProblem,
)


class Spec:
    def __init__(self, **overrides) -> None:
        self.arch_name = "mp_31"
        self.mp_count = 56
        self.warp_size = 32
        self.shared_memory_per_block = 96 * 1024
        self.shared_memory_per_multiprocessor = 192 * 1024
        self.regs_per_multiprocessor = 64 * 1024
        self.max_threads_per_block = 1024
        self.max_threads_per_multiprocessor = 2048
        for name, value in overrides.items():
            setattr(self, name, value)


def dense_layer(**overrides) -> LayerShape:
    fields = dict(shape_n=4096, shape_k=4096, dtype="bf16", weight_dtype="fp8")
    fields.update(overrides)
    return LayerShape(**fields)


def test_a_layer_shape_needs_a_positive_projection() -> None:
    with pytest.raises(ValueError, match="positive"):
        LayerShape(shape_n=0, shape_k=4096, dtype="bf16", weight_dtype="fp8")


def test_a_moe_layer_cannot_route_wider_than_its_experts() -> None:
    with pytest.raises(ValueError, match="top_k"):
        LayerShape(
            shape_n=4096,
            shape_k=4096,
            dtype="bf16",
            weight_dtype="fp8",
            num_experts=8,
            top_k=9,
        )


def test_rows_per_expert_divides_the_routed_tokens() -> None:
    layer = dense_layer(num_experts=8, top_k=2)
    assert layer.is_moe
    # 64 tokens, top-2, 8 experts -> 128 routed rows -> 16 per expert.
    assert layer.expert_rows(64) == 16
    # Fewer routed rows than experts still leaves one row per expert.
    assert layer.expert_rows(1) == 1


def test_a_dense_layer_keeps_m_unchanged() -> None:
    assert dense_layer().expert_rows(64) == 64


def test_unknown_semantics_are_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="gemm_type"):
        ComputeSemantics(gemm_type="moe_sideways")
    with pytest.raises(ValueError, match="tp_shard_k"):
        ComputeSemantics(tp_shard_k=0)


def test_the_device_profile_mirrors_the_spec_it_came_from() -> None:
    spec = Spec()
    profile = DeviceProfile.from_spec(spec)
    assert profile.arch_name == "mp_31"
    assert profile.mp_count == 56
    assert profile.shared_memory_per_multiprocessor == 192 * 1024
    # The two shared-memory limits stay distinct: which one binds depends on
    # whether the kernel opts into dynamic shared memory.
    assert (
        profile.shared_memory_per_block != profile.shared_memory_per_multiprocessor
    )


def test_an_object_that_is_not_a_spec_says_which_attribute_is_missing() -> None:
    with pytest.raises(TypeError, match="missing"):
        DeviceProfile.from_spec(object())


def test_a_problem_without_a_device_matches_no_arch() -> None:
    problem = TuningProblem(layer=dense_layer(), m=8)
    assert problem.arch_name == ""
    assert problem.mp_count is None


def test_a_problem_is_frozen() -> None:
    problem = TuningProblem(
        layer=dense_layer(), m=8, device=DeviceProfile.from_spec(Spec())
    )
    with pytest.raises(Exception):
        problem.m = 9  # type: ignore[misc]


def test_a_problem_carries_the_tuning_m_for_moe() -> None:
    problem = TuningProblem(
        layer=dense_layer(num_experts=8, top_k=2),
        m=64,
        device=DeviceProfile.from_spec(Spec()),
    )
    assert problem.tuning_m == 16


def test_two_problems_with_the_same_fields_are_equal_and_hash_alike() -> None:
    def build():
        return TuningProblem(
            layer=dense_layer(), m=8, device=DeviceProfile.from_spec(Spec())
        )

    assert build() == build()
    assert hash(build()) == hash(build())
