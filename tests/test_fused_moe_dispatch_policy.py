import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

POLICY_PATH = (
    Path(__file__).parents[1]
    / "vllm_musa/model_executor/layers/fused_moe/dispatch_policy.py"
)
PACKAGE_NAME = "_musa_fused_moe_policy_test"
PACKAGE = types.ModuleType(PACKAGE_NAME)
PACKAGE.__path__ = [str(POLICY_PATH.parent)]
sys.modules[PACKAGE_NAME] = PACKAGE
TYPES_SPEC = importlib.util.spec_from_file_location(
    f"{PACKAGE_NAME}.dispatch_types", POLICY_PATH.with_name("dispatch_types.py")
)
assert TYPES_SPEC is not None and TYPES_SPEC.loader is not None
TYPES_MODULE = importlib.util.module_from_spec(TYPES_SPEC)
sys.modules[TYPES_SPEC.name] = TYPES_MODULE
TYPES_SPEC.loader.exec_module(TYPES_MODULE)
SPEC = importlib.util.spec_from_file_location(
    f"{PACKAGE_NAME}.dispatch_policy", POLICY_PATH
)
assert SPEC is not None and SPEC.loader is not None
POLICY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = POLICY
SPEC.loader.exec_module(POLICY)

MUSA_FUSED_MOE_DISPATCH_ENV = POLICY.MUSA_FUSED_MOE_DISPATCH_ENV
MusaFusedMoeBackend = POLICY.MusaFusedMoeBackend
MusaFusedMoeShape = POLICY.MusaFusedMoeShape
MusaFusedMoeThresholds = POLICY.MusaFusedMoeThresholds
active_fused_moe_runtime_policy_receipt = POLICY.active_fused_moe_runtime_policy_receipt
configure_fused_moe_runtime_policy = POLICY.configure_fused_moe_runtime_policy
fused_moe_runtime_policy_token_boundaries = (
    POLICY.fused_moe_runtime_policy_token_boundaries
)
parse_dispatch_backend = POLICY.parse_dispatch_backend
record_modular_fused_moe_runtime_plan_binding = (
    POLICY.record_modular_fused_moe_runtime_plan_binding
)
resolve_fused_moe_backend = POLICY.resolve_fused_moe_backend
resolve_runtime_plan_for_dimensions = POLICY.resolve_runtime_plan_for_dimensions
runtime_plan_bindings_for_dimensions = POLICY.runtime_plan_bindings_for_dimensions
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


def _runtime_policy(shape, ranges):
    entry = (
        (
            "ranges",
            tuple(
                tuple(
                    sorted(
                        {
                            "min_tokens": min_tokens,
                            "max_tokens": max_tokens,
                            "backend": backend,
                        }.items()
                    )
                )
                for min_tokens, max_tokens, backend in ranges
            ),
        ),
        ("shape", tuple(sorted(shape.__dict__.items()))),
    )
    return (
        ("entries", (entry,)),
        ("schema", "musa.fused_moe.dispatch_policy.v1"),
    )


@pytest.fixture(autouse=True)
def _reset_runtime_policy():
    configure_fused_moe_runtime_policy(())
    yield
    configure_fused_moe_runtime_policy(())


def test_runtime_plan_entry_overrides_legacy_and_carries_plan_receipt():
    shape = _shape(
        multiprocessor_count=60,
        local_experts=256,
        w1_output_size=256,
        w2_input_size=128,
        hidden_size=2048,
        top_k=8,
        w1_scale_shape=(256, 2, 16),
        w2_scale_shape=(256, 16, 1),
    )
    assert thresholds_for_shape(shape).gemv_max_tokens == 13
    configure_fused_moe_runtime_policy(
        _runtime_policy(shape, ((1, 4, "upstream"), (5, 8, "gemv"))),
        plan_id="qwen-control",
        plan_fingerprint="sha256:qwen-control",
        profile="qwen3.text_generation",
    )

    selection = resolve_fused_moe_backend(
        shape=shape,
        num_tokens=1,
        can_use_gemv=True,
        can_use_grouped_gemm=False,
        stream_is_capturing=False,
    )

    assert selection.backend is MusaFusedMoeBackend.UPSTREAM
    assert selection.source == "runtime_plan"
    assert selection.plan_id == "qwen-control"
    assert selection.plan_fingerprint == "sha256:qwen-control"
    assert (selection.min_tokens, selection.max_tokens) == (1, 4)
    receipt = active_fused_moe_runtime_policy_receipt()
    assert receipt.plan_fingerprint == "sha256:qwen-control"
    assert receipt.entry_count == 1
    assert fused_moe_runtime_policy_token_boundaries() == (4, 8)


def test_runtime_plan_resolves_modular_backend_from_static_dimensions():
    shape = _shape(
        multiprocessor_count=60,
        local_experts=256,
        w1_output_size=256,
        w2_input_size=128,
        hidden_size=2048,
        top_k=8,
        w1_scale_shape=(256, 2, 16),
        w2_scale_shape=(256, 16, 1),
    )
    configure_fused_moe_runtime_policy(
        _runtime_policy(shape, ((1, 4, "upstream"), (5, 8, "gemv"))),
        plan_id="modular-control",
        plan_fingerprint="sha256:modular-control",
        profile="qwen3.text_generation",
    )

    selection = resolve_runtime_plan_for_dimensions(
        local_experts=256,
        w1_output_size=256,
        w2_input_size=128,
        hidden_size=2048,
        top_k=8,
        num_tokens=2,
    )

    assert selection is not None
    assert selection.backend is MusaFusedMoeBackend.UPSTREAM
    assert selection.source == "runtime_plan"
    assert selection.plan_id == "modular-control"


def test_runtime_plan_resolves_modular_capture_entry_only_during_capture():
    capture_shape = _shape(
        multiprocessor_count=60,
        local_experts=256,
        w1_output_size=256,
        w2_input_size=128,
        hidden_size=2048,
        top_k=8,
        w1_scale_shape=(256, 2, 16),
        w2_scale_shape=(256, 16, 1),
        graph_mode="capture",
    )
    eager_shape = _shape(
        multiprocessor_count=60,
        local_experts=256,
        w1_output_size=256,
        w2_input_size=128,
        hidden_size=2048,
        top_k=8,
        w1_scale_shape=(256, 2, 16),
        w2_scale_shape=(256, 16, 1),
        graph_mode="eager",
    )
    capture_entry = _runtime_policy(capture_shape, ((1, 4, "upstream"),))[0][1][0]
    eager_entry = _runtime_policy(eager_shape, ((1, 4, "upstream"),))[0][1][0]
    configure_fused_moe_runtime_policy(
        (
            ("entries", (capture_entry, eager_entry)),
            ("schema", "musa.fused_moe.dispatch_policy.v1"),
        ),
        plan_id="modular-capture-control",
        plan_fingerprint="sha256:modular-capture-control",
        profile="qwen3.text_generation",
    )

    eager_selection = resolve_runtime_plan_for_dimensions(
        local_experts=256,
        w1_output_size=256,
        w2_input_size=128,
        hidden_size=2048,
        top_k=8,
        num_tokens=2,
        stream_is_capturing=False,
    )
    capture_selection = resolve_runtime_plan_for_dimensions(
        local_experts=256,
        w1_output_size=256,
        w2_input_size=128,
        hidden_size=2048,
        top_k=8,
        num_tokens=2,
        stream_is_capturing=True,
    )

    assert eager_selection is not None
    assert eager_selection.source == "runtime_plan"
    assert capture_selection is not None
    assert capture_selection.source == "runtime_plan"
    assert capture_selection.plan_id == "modular-capture-control"


def test_modular_binding_receipt_is_static_and_idempotent(caplog, monkeypatch):
    monkeypatch.setenv("RANK", "3")
    shape = _shape(
        local_experts=256,
        w1_output_size=256,
        w2_input_size=128,
        hidden_size=2048,
        top_k=8,
        w1_scale_shape=(256, 2, 16),
        w2_scale_shape=(256, 16, 1),
    )
    configure_fused_moe_runtime_policy(
        _runtime_policy(shape, ((1, 4, "upstream"), (5, 8, "gemv"))),
        plan_id="modular-binding-control",
        plan_fingerprint="sha256:modular-binding-control",
        profile="qwen3.text_generation",
    )

    class Weight:
        def __init__(self, dimensions):
            self.shape = dimensions

    class Config:
        experts_per_token = 8
        tp_rank = 3

    class RoutedExperts:
        w13_weight = Weight((256, 256, 2048))
        w2_weight = Weight((256, 2048, 128))
        moe_config = Config()

    class Kernel:
        fused_experts = object()

    with caplog.at_level("INFO"):
        record_modular_fused_moe_runtime_plan_binding(
            routed_experts=RoutedExperts(),
            moe_kernel=Kernel(),
            layer_name="layer.0",
        )
        record_modular_fused_moe_runtime_plan_binding(
            routed_experts=RoutedExperts(),
            moe_kernel=Kernel(),
            layer_name="layer.1",
        )

    receipts = [
        record
        for record in caplog.records
        if "MUSA fused-MoE plan binding receipt" in record.message
    ]
    assert len(receipts) == 2
    assert all("phase=bind" in record.message for record in receipts)
    assert all("planned_backend=" in record.message for record in receipts)
    assert all("execution_backend=uncontrolled" in record.message for record in receipts)
    assert all("rank=3" in record.message for record in receipts)
    assert all("plan_fingerprint=sha256:modular-binding-control" in record.message
               for record in receipts)

    bindings = runtime_plan_bindings_for_dimensions(
        local_experts=256,
        w1_output_size=256,
        w2_input_size=128,
        hidden_size=2048,
        top_k=8,
    )
    assert len(bindings) == 1
    assert [r.backend for r in bindings[0][1]] == [
        MusaFusedMoeBackend.UPSTREAM,
        MusaFusedMoeBackend.GEMV,
    ]


def test_modular_binding_does_not_emit_for_a_plan_without_moe_entries(caplog):
    configure_fused_moe_runtime_policy(
        (),
        plan_id="other-domain",
        plan_fingerprint="sha256:other-domain",
        profile="rms_norm",
    )

    class Weight:
        shape = (2, 4, 8)

    class Config:
        experts_per_token = 1

    class RoutedExperts:
        w13_weight = Weight()
        w2_weight = Weight()
        moe_config = Config()

    with caplog.at_level("INFO"):
        record_modular_fused_moe_runtime_plan_binding(
            routed_experts=RoutedExperts(),
            moe_kernel=object(),
        )

    assert not any("plan binding receipt" in record.message for record in caplog.records)


def test_modular_binding_rejects_non_rank_three_weights(caplog):
    shape = _shape(
        local_experts=2,
        w1_output_size=4,
        w2_input_size=2,
        hidden_size=8,
        top_k=1,
    )
    configure_fused_moe_runtime_policy(
        _runtime_policy(shape, ((1, 4, "upstream"),)),
        plan_id="strict-shape",
        plan_fingerprint="sha256:strict-shape",
        profile="qwen3.text_generation",
    )

    class Weight:
        def __init__(self, dimensions):
            self.shape = dimensions

    class Config:
        experts_per_token = 1
        is_act_and_mul = True

    class RoutedExperts:
        w13_weight = Weight((2, 4, 8, 1))
        w2_weight = Weight((2, 8, 2))
        moe_config = Config()

    with caplog.at_level("WARNING"):
        record_modular_fused_moe_runtime_plan_binding(
            routed_experts=RoutedExperts(),
            moe_kernel=object(),
        )

    assert any("invalid weight metadata" in record.message for record in caplog.records)


def test_runtime_plan_miss_uses_legacy_and_ineligible_hit_fails_upstream():
    shape = _shape(
        multiprocessor_count=60,
        local_experts=256,
        w1_output_size=256,
        w2_input_size=128,
        hidden_size=2048,
        top_k=8,
        w1_scale_shape=(256, 2, 16),
        w2_scale_shape=(256, 16, 1),
    )
    configure_fused_moe_runtime_policy(
        _runtime_policy(shape, ((1, 4, "gemv"),)),
        plan_id="qwen-gemv",
        plan_fingerprint="sha256:qwen-gemv",
        profile="qwen3.text_generation",
    )

    ineligible = resolve_fused_moe_backend(
        shape=shape,
        num_tokens=1,
        can_use_gemv=False,
        can_use_grouped_gemm=False,
        stream_is_capturing=False,
    )
    legacy = resolve_fused_moe_backend(
        shape=shape,
        num_tokens=8,
        can_use_gemv=True,
        can_use_grouped_gemm=False,
        stream_is_capturing=False,
    )

    assert ineligible.backend is MusaFusedMoeBackend.UPSTREAM
    assert ineligible.source == "runtime_plan_ineligible"
    assert legacy.backend is MusaFusedMoeBackend.GEMV
    assert legacy.source == "legacy"


def test_diagnostic_override_has_priority_over_runtime_plan():
    shape = _shape()
    configure_fused_moe_runtime_policy(
        _runtime_policy(shape, ((1, 8, "upstream"),)),
        plan_id="control",
        plan_fingerprint="sha256:control",
        profile="qwen3.text_generation",
    )

    selection = resolve_fused_moe_backend(
        shape=shape,
        num_tokens=2,
        can_use_gemv=True,
        can_use_grouped_gemm=False,
        stream_is_capturing=False,
        requested=MusaFusedMoeBackend.GEMV,
    )

    assert selection.backend is MusaFusedMoeBackend.GEMV
    assert selection.source == "diagnostic_override"


def test_active_runtime_plan_shape_participates_in_cheap_prefilter():
    shape = _shape()
    assert not POLICY.has_calibrated_dimensions(
        local_experts=shape.local_experts,
        w1_output_size=shape.w1_output_size,
        w2_input_size=shape.w2_input_size,
        hidden_size=shape.hidden_size,
        top_k=shape.top_k,
    )
    assert not POLICY.has_runtime_policy_dimensions(
        local_experts=shape.local_experts,
        w1_output_size=shape.w1_output_size,
        w2_input_size=shape.w2_input_size,
        hidden_size=shape.hidden_size,
        top_k=shape.top_k,
    )

    configure_fused_moe_runtime_policy(
        _runtime_policy(shape, ((1, 8, "gemv"),)),
        plan_id="new-shape",
        plan_fingerprint="sha256:new-shape",
        profile="qwen3.text_generation",
    )

    assert POLICY.has_calibrated_dimensions(
        local_experts=shape.local_experts,
        w1_output_size=shape.w1_output_size,
        w2_input_size=shape.w2_input_size,
        hidden_size=shape.hidden_size,
        top_k=shape.top_k,
    )
    assert POLICY.has_runtime_policy_dimensions(
        local_experts=shape.local_experts,
        w1_output_size=shape.w1_output_size,
        w2_input_size=shape.w2_input_size,
        hidden_size=shape.hidden_size,
        top_k=shape.top_k,
    )


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


def test_qwen35_bf16_decode_gemv_uses_tp4_local_crossover():
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

    assert thresholds_for_shape(shape).gemv_max_tokens == 12
    for token_count in (1, 4, 8, 12):
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
    assert (
        select_fused_moe_backend(
            shape=shape,
            num_tokens=16,
            can_use_gemv=True,
            can_use_grouped_gemm=False,
            stream_is_capturing=False,
        )
        == MusaFusedMoeBackend.UPSTREAM
    )
    capture_shape = _shape(**{**shape.__dict__, "graph_mode": "capture"})
    assert thresholds_for_shape(capture_shape).gemv_max_tokens == 12

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
