import ast
import dataclasses
import importlib.util
import sys
from pathlib import Path

POLICY_PATH = (
    Path(__file__).parents[1]
    / "vllm_musa/model_executor/layers/fused_moe/dispatch_policy.py"
)
SPEC = importlib.util.spec_from_file_location(
    "musa_fused_moe_dispatch_policy", POLICY_PATH
)
assert SPEC is not None and SPEC.loader is not None
POLICY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = POLICY
SPEC.loader.exec_module(POLICY)

MUSA_FUSED_MOE_DISPATCH_ENV = POLICY.MUSA_FUSED_MOE_DISPATCH_ENV
MusaFusedMoeBackend = POLICY.MusaFusedMoeBackend
MusaFusedMoeShape = POLICY.MusaFusedMoeShape
MusaFusedMoeThresholds = POLICY.MusaFusedMoeThresholds
MusaForwardGraphBucket = POLICY.MusaForwardGraphBucket
parse_dispatch_backend = POLICY.parse_dispatch_backend
resolve_fused_moe_graph_mode = POLICY.resolve_fused_moe_graph_mode
select_fused_moe_backend = POLICY.select_fused_moe_backend
thresholds_for_shape = POLICY.thresholds_for_shape
FUSED_MOE_PATH = (
    Path(__file__).parents[1] / "vllm_musa/model_executor/layers/fused_moe/fused_moe.py"
)
PLATFORM_PATH = Path(__file__).parents[1] / "vllm_musa/platform.py"


def _shape(**overrides):
    values = {
        "device_capability": (3, 1),
        "multiprocessor_count": 64,
        "local_experts": 128,
        "w1_output_size": 768,
        "w2_input_size": 384,
        "hidden_size": 4096,
        "top_k": 8,
        "block_n": 128,
        "block_k": 128,
        "activation": "silu",
        "expert_parallel": False,
        "hidden_dtype": "torch.bfloat16",
        "weight_dtype": "torch.float8_e4m3fn",
        "scale_dtype": "torch.float32",
        "w1_scale_shape": (128, 6, 32),
        "w2_scale_shape": (128, 32, 3),
        "gemv_block": "auto",
        "graph_mode": "eager",
    }
    values.update(overrides)
    return MusaFusedMoeShape(**values)


def test_unknown_shape_stays_on_upstream_path():
    shape = _shape()

    assert thresholds_for_shape(shape).source == "uncalibrated-shape"
    assert (
        select_fused_moe_backend(
            shape=shape,
            num_tokens=4,
            can_use_gemv=True,
            can_use_grouped_gemm=True,
            stream_is_capturing=False,
        )
        == MusaFusedMoeBackend.UPSTREAM
    )
    assert (
        select_fused_moe_backend(
            shape=shape,
            num_tokens=16,
            can_use_gemv=True,
            can_use_grouped_gemm=True,
            stream_is_capturing=False,
        )
        == MusaFusedMoeBackend.UPSTREAM
    )


def test_grouped_gemm_is_never_selected_during_capture():
    shape = _shape()

    assert (
        select_fused_moe_backend(
            shape=shape,
            num_tokens=4096,
            can_use_gemv=True,
            can_use_grouped_gemm=True,
            stream_is_capturing=True,
            requested=MusaFusedMoeBackend.GROUPED_GEMM,
        )
        == MusaFusedMoeBackend.UPSTREAM
    )


def test_compile_mode_is_always_fail_closed():
    shape = _shape(graph_mode="compile")

    for requested in MusaFusedMoeBackend:
        assert (
            select_fused_moe_backend(
                shape=shape,
                num_tokens=1,
                can_use_gemv=True,
                can_use_grouped_gemm=True,
                stream_is_capturing=False,
                requested=requested,
                thresholds=MusaFusedMoeThresholds(
                    gemv_max_tokens=4,
                    grouped_gemm_min_tokens=1,
                    source="must-not-specialize-during-compile",
                ),
            )
            == MusaFusedMoeBackend.UPSTREAM
        )


def test_graph_mode_resolution_prioritizes_symbolic_compile():
    assert (
        resolve_fused_moe_graph_mode(
            is_compiling=True,
            stream_is_capturing=True,
        )
        == "compile"
    )
    assert (
        resolve_fused_moe_graph_mode(
            is_compiling=False,
            stream_is_capturing=True,
        )
        == "capture"
    )
    assert (
        resolve_fused_moe_graph_mode(
            is_compiling=False,
            stream_is_capturing=False,
        )
        == "eager"
    )


def test_calibrated_threshold_boundaries_and_device_identity(monkeypatch):
    shape = _shape()
    thresholds = MusaFusedMoeThresholds(
        gemv_max_tokens=4,
        grouped_gemm_min_tokens=16,
        source="test-calibration",
    )
    monkeypatch.setattr(POLICY, "_CALIBRATED_THRESHOLDS", {shape: thresholds})

    assert (
        select_fused_moe_backend(
            shape=shape,
            num_tokens=4,
            can_use_gemv=True,
            can_use_grouped_gemm=True,
            stream_is_capturing=False,
        )
        == MusaFusedMoeBackend.GEMV
    )
    assert (
        select_fused_moe_backend(
            shape=shape,
            num_tokens=5,
            can_use_gemv=True,
            can_use_grouped_gemm=True,
            stream_is_capturing=False,
        )
        == MusaFusedMoeBackend.UPSTREAM
    )
    assert (
        select_fused_moe_backend(
            shape=shape,
            num_tokens=16,
            can_use_gemv=True,
            can_use_grouped_gemm=True,
            stream_is_capturing=False,
        )
        == MusaFusedMoeBackend.GROUPED_GEMM
    )
    assert thresholds_for_shape(_shape(device_capability=(4, 0))).source == (
        "uncalibrated-shape"
    )
    assert thresholds_for_shape(_shape(multiprocessor_count=56)).source == (
        "uncalibrated-shape"
    )


def test_calibrated_dimension_prefilter(monkeypatch):
    dimensions = frozenset({(128, 512, 256, 4096, 6)})
    monkeypatch.setattr(POLICY, "_CALIBRATED_DIMENSIONS", dimensions)
    assert POLICY.has_calibrated_dimensions(
        local_experts=128,
        w1_output_size=512,
        w2_input_size=256,
        hidden_size=4096,
        top_k=6,
    )
    assert not POLICY.has_calibrated_dimensions(
        local_experts=64,
        w1_output_size=512,
        w2_input_size=256,
        hidden_size=4096,
        top_k=6,
    )


def test_s5000_calibrated_shapes_use_route_worst_boundaries():
    qwen = _shape(
        multiprocessor_count=60,
        local_experts=256,
        w1_output_size=256,
        w2_input_size=128,
        hidden_size=2048,
        top_k=8,
        w1_scale_shape=(256, 2, 16),
        w2_scale_shape=(256, 16, 1),
    )
    dsv4 = _shape(
        multiprocessor_count=60,
        local_experts=256,
        w1_output_size=512,
        w2_input_size=256,
        hidden_size=4096,
        top_k=6,
        w1_scale_shape=(256, 4, 32),
        w2_scale_shape=(256, 32, 2),
        gemv_block="32x8",
    )
    dsv2 = _shape(
        multiprocessor_count=60,
        local_experts=64,
        w1_output_size=2816,
        w2_input_size=1408,
        hidden_size=2048,
        top_k=6,
        w1_scale_shape=(64, 22, 16),
        w2_scale_shape=(64, 16, 11),
    )

    assert thresholds_for_shape(qwen).gemv_max_tokens == 13
    assert thresholds_for_shape(qwen).grouped_gemm_min_tokens is None
    assert thresholds_for_shape(dsv4).gemv_max_tokens == 5
    dsv4_block16 = _shape(
        multiprocessor_count=60,
        local_experts=256,
        w1_output_size=512,
        w2_input_size=256,
        hidden_size=4096,
        top_k=6,
        w1_scale_shape=(256, 4, 32),
        w2_scale_shape=(256, 32, 2),
        gemv_block="16x8",
    )
    assert thresholds_for_shape(dsv4_block16).gemv_max_tokens == 5
    assert thresholds_for_shape(dsv4).grouped_gemm_min_tokens is None
    assert thresholds_for_shape(dsv2).gemv_max_tokens == 3
    assert thresholds_for_shape(dsv2).grouped_gemm_min_tokens is None

    dsv2_capture = _shape(**{**dsv2.__dict__, "graph_mode": "capture"})
    assert thresholds_for_shape(dsv2_capture).gemv_max_tokens == 1
    assert thresholds_for_shape(dsv2_capture).grouped_gemm_min_tokens is None


def test_s5000_calibration_remains_exact_for_layout_and_mp_count():
    shape = _shape(
        multiprocessor_count=60,
        local_experts=64,
        w1_output_size=2816,
        w2_input_size=1408,
        hidden_size=2048,
        top_k=6,
        w1_scale_shape=(64, 22, 16),
        w2_scale_shape=(64, 16, 11),
    )

    assert thresholds_for_shape(
        _shape(**{**shape.__dict__, "multiprocessor_count": 64})
    ).source == ("uncalibrated-shape")
    assert thresholds_for_shape(
        _shape(**{**shape.__dict__, "w1_scale_shape": (64, 16, 22)})
    ).source == ("uncalibrated-shape")


def test_pr113_folded_qwen_shape_requires_fresh_calibration():
    folded_qwen = _shape(
        multiprocessor_count=60,
        local_experts=257,
        w1_output_size=256,
        w2_input_size=128,
        hidden_size=2048,
        top_k=9,
        w1_scale_shape=(257, 2, 16),
        w2_scale_shape=(257, 16, 1),
    )

    # Shared-expert folding appends one expert and one route column.  The old
    # E=256/topk=8 threshold must not be reused without a new sweep.
    assert thresholds_for_shape(folded_qwen).source == "uncalibrated-shape"


def test_qwen35_bf16_decode_gemv_uses_mp60_route_worst_crossover():
    shape = _shape(
        multiprocessor_count=60,
        local_experts=257,
        w1_output_size=256,
        w2_input_size=128,
        hidden_size=2048,
        top_k=9,
        block_n=0,
        block_k=0,
        weight_dtype="torch.bfloat16",
        scale_dtype="none",
        w1_scale_shape=(),
        w2_scale_shape=(),
    )

    assert thresholds_for_shape(shape).gemv_max_tokens == 4
    for token_count in (1, 4):
        assert (
            select_fused_moe_backend(
                shape=shape,
                num_tokens=token_count,
                can_use_gemv=True,
                can_use_grouped_gemm=False,
                stream_is_capturing=False,
            )
            == MusaFusedMoeBackend.GEMV
        )
    for token_count in (8, 16):
        assert (
            select_fused_moe_backend(
                shape=shape,
                num_tokens=token_count,
                can_use_gemv=True,
                can_use_grouped_gemm=False,
                stream_is_capturing=False,
            )
            == MusaFusedMoeBackend.UPSTREAM
        )
    capture_shape = _shape(**{**shape.__dict__, "graph_mode": "capture"})
    assert thresholds_for_shape(capture_shape).gemv_max_tokens == 4

    unfolded = _shape(
        multiprocessor_count=60,
        local_experts=256,
        w1_output_size=256,
        w2_input_size=128,
        hidden_size=2048,
        top_k=8,
        block_n=0,
        block_k=0,
        weight_dtype="torch.bfloat16",
        scale_dtype="none",
        w1_scale_shape=(),
        w2_scale_shape=(),
    )
    assert thresholds_for_shape(unfolded).gemv_max_tokens == 12
    assert (
        thresholds_for_shape(
            dataclasses.replace(unfolded, max_num_seqs=16)
        ).gemv_max_tokens
        == 12
    )


def test_graph_bucket_guard_preserves_existing_mp60_runtime_modes():
    qwen = _shape(
        multiprocessor_count=60,
        local_experts=257,
        w1_output_size=256,
        w2_input_size=128,
        hidden_size=2048,
        top_k=9,
        block_n=0,
        block_k=0,
        weight_dtype="torch.bfloat16",
        scale_dtype="none",
        w1_scale_shape=(),
        w2_scale_shape=(),
    )
    dsv4 = _shape(
        multiprocessor_count=60,
        local_experts=256,
        w1_output_size=512,
        w2_input_size=256,
        hidden_size=4096,
        top_k=6,
        w1_scale_shape=(256, 4, 32),
        w2_scale_shape=(256, 32, 2),
        gemv_block="32x8",
    )
    runtime_cases = (
        (
            "eager",
            MusaForwardGraphBucket(
                num_tokens=4,
                num_reqs=None,
                uniform=False,
                runtime_mode="NONE",
                has_lora=False,
                num_active_loras=0,
                present=True,
            ),
        ),
        (
            "capture",
            MusaForwardGraphBucket(
                num_tokens=4,
                num_reqs=None,
                uniform=False,
                runtime_mode="PIECEWISE",
                has_lora=False,
                num_active_loras=0,
                present=True,
            ),
        ),
        (
            "capture",
            MusaForwardGraphBucket(
                num_tokens=4,
                num_reqs=4,
                uniform=True,
                runtime_mode="FULL",
                has_lora=False,
                num_active_loras=0,
                present=True,
            ),
        ),
    )

    for base_shape in (qwen, dsv4):
        for graph_mode, descriptor in runtime_cases:
            shape = dataclasses.replace(
                base_shape,
                graph_mode=graph_mode,
                graph_bucket=descriptor,
            )
            thresholds = thresholds_for_shape(shape)
            assert thresholds.gemv_max_tokens is not None
            assert (
                select_fused_moe_backend(
                    shape=shape,
                    num_tokens=4,
                    can_use_gemv=True,
                    can_use_grouped_gemm=False,
                    stream_is_capturing=graph_mode == "capture",
                    thresholds=thresholds,
                )
                == MusaFusedMoeBackend.GEMV
            )

    for descriptor in (
        MusaForwardGraphBucket.invalid(),
        MusaForwardGraphBucket(
            num_tokens=1,
            num_reqs=1,
            uniform=True,
            runtime_mode="NONE",
            has_lora=True,
            num_active_loras=1,
            present=True,
        ),
        MusaForwardGraphBucket(
            num_tokens=4,
            num_reqs=2,
            uniform=True,
            runtime_mode="FULL",
            has_lora=False,
            num_active_loras=0,
            present=True,
        ),
    ):
        shape = dataclasses.replace(qwen, graph_bucket=descriptor)
        assert (
            select_fused_moe_backend(
                shape=shape,
                num_tokens=1,
                can_use_gemv=True,
                can_use_grouped_gemm=False,
                stream_is_capturing=False,
            )
            == MusaFusedMoeBackend.UPSTREAM
        )


def test_qwen35_bf16_decode_gemv_uses_mp56_route_worst_crossover():
    base_shape = _shape(
        multiprocessor_count=56,
        local_experts=257,
        w1_output_size=256,
        w2_input_size=128,
        hidden_size=2048,
        top_k=9,
        block_n=0,
        block_k=0,
        weight_dtype="torch.bfloat16",
        scale_dtype="none",
        w1_scale_shape=(),
        w2_scale_shape=(),
    )
    assert thresholds_for_shape(base_shape).gemv_max_tokens is None
    assert (
        thresholds_for_shape(
            dataclasses.replace(base_shape, max_num_seqs=16)
        ).gemv_max_tokens
        is None
    )
    # The engine-static profile is part of the key: a larger profile must not
    # accidentally inherit the small-batch GEMV entry during symbolic compile.
    assert all(
        key.max_num_seqs in (None, 1, 2, 4)
        for key in POLICY._CALIBRATED_THRESHOLDS
        if key.multiprocessor_count == 56
    )
    assert all(
        key.graph_mode != "compile"
        for key in POLICY._CALIBRATED_THRESHOLDS
        if key.multiprocessor_count == 56
    )
    for graph_mode in ("eager", "capture"):
        for max_num_seqs in (1, 2, 4):
            shape = dataclasses.replace(
                base_shape,
                graph_mode=graph_mode,
                max_num_seqs=max_num_seqs,
            )
            thresholds = thresholds_for_shape(shape)
            assert thresholds.gemv_max_tokens == 4
            assert "mp56" in thresholds.source
            assert "e257" in thresholds.source
            assert f"maxseq{max_num_seqs}" in thresholds.source
            assert "route-worst" in thresholds.source
            assert select_fused_moe_backend(
                shape=shape,
                num_tokens=4,
                can_use_gemv=True,
                can_use_grouped_gemm=False,
                stream_is_capturing=graph_mode == "capture",
            ) == (
                MusaFusedMoeBackend.GEMV
                if graph_mode == "eager"
                else MusaFusedMoeBackend.UPSTREAM
            )
            assert (
                select_fused_moe_backend(
                    shape=shape,
                    num_tokens=5,
                    can_use_gemv=True,
                    can_use_grouped_gemm=False,
                    stream_is_capturing=graph_mode == "capture",
                )
                == MusaFusedMoeBackend.UPSTREAM
            )


def test_qwen_mp56_uses_only_exact_full_decode_graph_buckets():
    base_shape = _shape(
        multiprocessor_count=56,
        local_experts=257,
        w1_output_size=256,
        w2_input_size=128,
        hidden_size=2048,
        top_k=9,
        block_n=0,
        block_k=0,
        weight_dtype="torch.bfloat16",
        scale_dtype="none",
        w1_scale_shape=(),
        w2_scale_shape=(),
        graph_mode="capture",
        max_num_seqs=64,
    )

    for graph_bucket in ((1, 1, True), (2, 2, True), (4, 4, True)):
        descriptor = MusaForwardGraphBucket(
            num_tokens=graph_bucket[0],
            num_reqs=graph_bucket[1],
            uniform=graph_bucket[2],
            runtime_mode="FULL",
            has_lora=False,
            num_active_loras=0,
            present=True,
        )
        thresholds = thresholds_for_shape(
            dataclasses.replace(base_shape, graph_bucket=descriptor)
        )
        assert thresholds.gemv_max_tokens == 4
        assert f"maxseq{graph_bucket[0]}" in thresholds.source
        assert (
            select_fused_moe_backend(
                shape=dataclasses.replace(base_shape, graph_bucket=descriptor),
                num_tokens=graph_bucket[0],
                can_use_gemv=True,
                can_use_grouped_gemm=False,
                stream_is_capturing=True,
                requested=MusaFusedMoeBackend.AUTO,
                thresholds=thresholds,
            )
            == MusaFusedMoeBackend.GEMV
        )

    bucket_one_shape = dataclasses.replace(
        base_shape,
        graph_bucket=MusaForwardGraphBucket(
            num_tokens=1,
            num_reqs=1,
            uniform=True,
            runtime_mode="FULL",
            has_lora=False,
            num_active_loras=0,
            present=True,
        ),
    )
    assert (
        select_fused_moe_backend(
            shape=bucket_one_shape,
            num_tokens=2,
            can_use_gemv=True,
            can_use_grouped_gemm=False,
            stream_is_capturing=True,
            requested=MusaFusedMoeBackend.AUTO,
            thresholds=thresholds_for_shape(bucket_one_shape),
        )
        == MusaFusedMoeBackend.UPSTREAM
    )

    eager_none = MusaForwardGraphBucket(
        num_tokens=4,
        num_reqs=None,
        uniform=False,
        runtime_mode="NONE",
        has_lora=False,
        num_active_loras=0,
        present=True,
    )
    eager_shape = dataclasses.replace(
        base_shape,
        graph_mode="eager",
        max_num_seqs=4,
        graph_bucket=eager_none,
    )
    eager_thresholds = thresholds_for_shape(eager_shape)
    assert eager_thresholds.gemv_max_tokens == 4
    assert (
        select_fused_moe_backend(
            shape=eager_shape,
            num_tokens=4,
            can_use_gemv=True,
            can_use_grouped_gemm=False,
            stream_is_capturing=False,
            thresholds=eager_thresholds,
        )
        == MusaFusedMoeBackend.GEMV
    )

    # Missing ForwardContext is tolerated only for eager direct-operator
    # callers. It cannot specialize a graph capture.
    assert (
        select_fused_moe_backend(
            shape=base_shape,
            num_tokens=1,
            can_use_gemv=True,
            can_use_grouped_gemm=False,
            stream_is_capturing=True,
        )
        == MusaFusedMoeBackend.UPSTREAM
    )

    for graph_bucket in (
        (16, 16, True),
        (4, 2, True),
        (4, None, False),
        (1, None, False),
    ):
        descriptor = MusaForwardGraphBucket(
            num_tokens=graph_bucket[0],
            num_reqs=graph_bucket[1],
            uniform=graph_bucket[2],
            runtime_mode="FULL" if graph_bucket[2] else "PIECEWISE",
            has_lora=False,
            num_active_loras=0,
            present=True,
        )
        invalid_shape = dataclasses.replace(base_shape, graph_bucket=descriptor)
        thresholds = thresholds_for_shape(invalid_shape)
        assert thresholds.source == "uncalibrated-shape"
        assert (
            select_fused_moe_backend(
                shape=invalid_shape,
                num_tokens=graph_bucket[0],
                can_use_gemv=True,
                can_use_grouped_gemm=False,
                stream_is_capturing=True,
                requested=MusaFusedMoeBackend.AUTO,
                thresholds=thresholds,
            )
            == MusaFusedMoeBackend.UPSTREAM
        )

    for descriptor in (
        MusaForwardGraphBucket(
            num_tokens=1,
            num_reqs=1,
            uniform=True,
            runtime_mode="FULL",
            has_lora=True,
            num_active_loras=1,
            present=True,
        ),
        MusaForwardGraphBucket.invalid(),
    ):
        shape = dataclasses.replace(base_shape, graph_bucket=descriptor)
        assert thresholds_for_shape(shape).source == "uncalibrated-shape"
        assert (
            select_fused_moe_backend(
                shape=shape,
                num_tokens=1,
                can_use_gemv=True,
                can_use_grouped_gemm=False,
                stream_is_capturing=True,
                requested=MusaFusedMoeBackend.AUTO,
                thresholds=thresholds_for_shape(shape),
            )
            == MusaFusedMoeBackend.UPSTREAM
        )
        assert (
            select_fused_moe_backend(
                shape=shape,
                num_tokens=1,
                can_use_gemv=True,
                can_use_grouped_gemm=False,
                stream_is_capturing=False,
                requested=MusaFusedMoeBackend.GEMV,
                thresholds=thresholds_for_shape(shape),
            )
            == MusaFusedMoeBackend.UPSTREAM
        )

    assert thresholds_for_shape(base_shape).source == "uncalibrated-shape"
    assert (
        thresholds_for_shape(
            dataclasses.replace(base_shape, multiprocessor_count=48)
        ).source
        == "uncalibrated-shape"
    )


def test_force_modes_preserve_eligibility_checks():
    shape = _shape()

    assert (
        select_fused_moe_backend(
            shape=shape,
            num_tokens=64,
            can_use_gemv=True,
            can_use_grouped_gemm=False,
            stream_is_capturing=False,
            requested=MusaFusedMoeBackend.GROUPED_GEMM,
        )
        == MusaFusedMoeBackend.UPSTREAM
    )
    assert (
        select_fused_moe_backend(
            shape=shape,
            num_tokens=64,
            can_use_gemv=False,
            can_use_grouped_gemm=False,
            stream_is_capturing=False,
            requested=MusaFusedMoeBackend.GEMV,
        )
        == MusaFusedMoeBackend.UPSTREAM
    )


def test_forced_gemv_preserves_capture_calibration_boundary(monkeypatch):
    eager_shape = _shape(graph_mode="eager")
    capture_shape = _shape(graph_mode="capture")
    capture_thresholds = MusaFusedMoeThresholds(
        gemv_max_tokens=4,
        grouped_gemm_min_tokens=None,
        source="capture-test",
    )
    monkeypatch.setattr(
        POLICY,
        "_CALIBRATED_THRESHOLDS",
        {capture_shape: capture_thresholds},
    )

    # Eager force remains a diagnostic override outside calibrated shapes.
    assert (
        select_fused_moe_backend(
            shape=eager_shape,
            num_tokens=64,
            can_use_gemv=True,
            can_use_grouped_gemm=False,
            stream_is_capturing=False,
            requested=MusaFusedMoeBackend.GEMV,
        )
        == MusaFusedMoeBackend.GEMV
    )
    assert (
        select_fused_moe_backend(
            shape=capture_shape,
            num_tokens=4,
            can_use_gemv=True,
            can_use_grouped_gemm=False,
            stream_is_capturing=True,
            requested=MusaFusedMoeBackend.GEMV,
        )
        == MusaFusedMoeBackend.UPSTREAM
    )
    capture_shape_with_context = dataclasses.replace(
        capture_shape,
        graph_bucket=MusaForwardGraphBucket(
            num_tokens=4,
            num_reqs=4,
            uniform=True,
            runtime_mode="FULL",
            has_lora=False,
            num_active_loras=0,
            present=True,
        ),
    )
    assert (
        select_fused_moe_backend(
            shape=capture_shape_with_context,
            num_tokens=4,
            can_use_gemv=True,
            can_use_grouped_gemm=False,
            stream_is_capturing=True,
            requested=MusaFusedMoeBackend.GEMV,
        )
        == MusaFusedMoeBackend.GEMV
    )
    assert (
        select_fused_moe_backend(
            shape=capture_shape,
            num_tokens=5,
            can_use_gemv=True,
            can_use_grouped_gemm=False,
            stream_is_capturing=True,
            requested=MusaFusedMoeBackend.GEMV,
        )
        == MusaFusedMoeBackend.UPSTREAM
    )
    assert (
        select_fused_moe_backend(
            shape=_shape(graph_mode="capture", local_experts=127),
            num_tokens=1,
            can_use_gemv=True,
            can_use_grouped_gemm=False,
            stream_is_capturing=True,
            requested=MusaFusedMoeBackend.GEMV,
        )
        == MusaFusedMoeBackend.UPSTREAM
    )


def test_generic_override_parser(monkeypatch):
    monkeypatch.setenv(MUSA_FUSED_MOE_DISPATCH_ENV, "grouped")
    assert parse_dispatch_backend() == MusaFusedMoeBackend.GROUPED_GEMM

    monkeypatch.setenv(MUSA_FUSED_MOE_DISPATCH_ENV, "invalid")
    try:
        parse_dispatch_backend()
    except ValueError as exc:
        assert MUSA_FUSED_MOE_DISPATCH_ENV in str(exc)
    else:
        raise AssertionError("invalid override must fail closed")


def test_upstream_fallback_does_not_forward_removed_inplace_keyword():
    tree = ast.parse(FUSED_MOE_PATH.read_text())
    dispatch = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_musa_fused_experts_impl_dispatch"
    )
    upstream_calls = [
        node
        for node in ast.walk(dispatch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_musa_original_fused_experts_impl"
    ]

    assert len(upstream_calls) == 1
    assert "inplace" not in {keyword.arg for keyword in upstream_calls[0].keywords}


def test_model_specific_dispatch_environment_is_removed_from_production_code():
    legacy_names = (
        "VLLM_MUSA_DEEPSEEK_V4_FUSED_MOE_GEMV",
        "VLLM_MUSA_DEEPSEEK_V4_MOE_DEEPGEMM_PREFILL",
    )
    source = FUSED_MOE_PATH.read_text() + PLATFORM_PATH.read_text()

    assert all(name not in source for name in legacy_names)
