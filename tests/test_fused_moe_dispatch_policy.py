import ast
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
parse_dispatch_backend = POLICY.parse_dispatch_backend
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


def test_mp48_dsv4_native_gemv_boundary_and_shape_isolation():
    for graph_mode in ("eager", "capture"):
        shape = _shape(
            multiprocessor_count=48,
            local_experts=256,
            w1_output_size=512,
            w2_input_size=256,
            top_k=6,
            w1_scale_shape=(256, 4, 32),
            w2_scale_shape=(256, 32, 2),
            gemv_block="16x8",
            graph_mode=graph_mode,
        )
        for num_tokens in (1, 2, 8, 11, 12, 13, 16):
            backend = select_fused_moe_backend(
                shape=shape,
                num_tokens=num_tokens,
                can_use_gemv=True,
                can_use_grouped_gemm=True,
                stream_is_capturing=graph_mode == "capture",
            )
            expected = (
                MusaFusedMoeBackend.GEMV
                if num_tokens <= 12
                else MusaFusedMoeBackend.UPSTREAM
            )
            assert backend == expected
        for override in ({"top_k": 8}, {"local_experts": 128}, {"expert_parallel": True}):
            unrelated = _shape(**{**shape.__dict__, **override})
            assert thresholds_for_shape(unrelated).source == "uncalibrated-shape"


def test_mp56_dsv4_native_gemv_boundary():
    for graph_mode in ("eager", "capture"):
        shape = _shape(
            multiprocessor_count=56,
            local_experts=256,
            w1_output_size=512,
            w2_input_size=256,
            top_k=6,
            w1_scale_shape=(256, 4, 32),
            w2_scale_shape=(256, 32, 2),
            gemv_block="16x8",
            graph_mode=graph_mode,
        )
        assert thresholds_for_shape(shape).gemv_max_tokens == 8
        for num_tokens in (1, 2, 5, 8, 9, 10, 12, 13, 16):
            backend = select_fused_moe_backend(
                shape=shape,
                num_tokens=num_tokens,
                can_use_gemv=True,
                can_use_grouped_gemm=True,
                stream_is_capturing=graph_mode == "capture",
            )
            expected = (
                MusaFusedMoeBackend.GEMV
                if num_tokens <= 8
                else MusaFusedMoeBackend.UPSTREAM
            )
            assert backend == expected


def _dsv4_tp8_shape(multiprocessor_count, graph_mode):
    return _shape(
        multiprocessor_count=multiprocessor_count,
        local_experts=256,
        w1_output_size=512,
        w2_input_size=256,
        top_k=6,
        w1_scale_shape=(256, 4, 32),
        w2_scale_shape=(256, 32, 2),
        gemv_block="16x8",
        graph_mode=graph_mode,
    )


def test_mp56_dsv4_triton_configs_cover_the_upstream_decode_ladder():
    for graph_mode in ("eager", "capture"):
        shape = _dsv4_tp8_shape(56, graph_mode)
        # Every calibrated target (5R) and draft (4R) decode shape.
        for num_tokens in sorted(POLICY._DSV4_TP8_MP56_TRITON_CONFIGS):
            config = POLICY.triton_config_for_shape(shape, num_tokens)
            assert config is not None, num_tokens
            assert config["BLOCK_SIZE_M"] == 16
            assert config["BLOCK_SIZE_K"] == 128
            assert config["GROUP_SIZE_M"] == 1
            assert config["num_stages"] == 1
        # Outside the calibrated range the tuned-folder lookup is kept.
        for num_tokens in (1, 8, 9, 81, 128, 4096):
            assert POLICY.triton_config_for_shape(shape, num_tokens) is None


def test_triton_config_lookup_uses_nearest_entry_and_returns_a_copy():
    shape = _dsv4_tp8_shape(56, "capture")
    table = POLICY._CALIBRATED_TRITON_CONFIGS[shape]
    assert POLICY.triton_config_for_shape(shape, 30) == table[28]
    assert POLICY.triton_config_for_shape(shape, 70) == table[65]
    config = POLICY.triton_config_for_shape(shape, 25)
    config["BLOCK_SIZE_M"] = 64
    assert table[25]["BLOCK_SIZE_M"] == 16


def test_triton_configs_are_isolated_to_the_calibrated_mp56_shape():
    for multiprocessor_count in (48, 60):
        for graph_mode in ("eager", "capture"):
            shape = _dsv4_tp8_shape(multiprocessor_count, graph_mode)
            assert POLICY.triton_config_for_shape(shape, 25) is None
    unrelated = _shape(**{**_dsv4_tp8_shape(56, "eager").__dict__, "top_k": 8})
    assert POLICY.triton_config_for_shape(unrelated, 25) is None


def test_upstream_fallback_applies_calibrated_triton_config_only():
    source = FUSED_MOE_PATH.read_text()
    assert "triton_config_for_shape(shape, hidden_states.shape[0])" in source
    assert "if backend == MusaFusedMoeBackend.UPSTREAM and shape is not None" in source
    assert "with _upstream_triton_config_scope(triton_config):" in source
    tree = ast.parse(source)
    scope = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_upstream_triton_config_scope"
    )
    # The previous config is restored even when the fused-experts call raises.
    tries = [node for node in ast.walk(scope) if isinstance(node, ast.Try)]
    assert tries and any(
        "_config = previous" in ast.unparse(stmt)
        for node in tries
        for stmt in node.finalbody
    )


def test_fused_experts_dispatch_rejects_non_positive_clamp_limit():
    source = FUSED_MOE_PATH.read_text()
    assert "if gemm1_clamp_limit is not None and gemm1_clamp_limit <= 0:" in source


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
    assert thresholds_for_shape(dsv4_block16).gemv_max_tokens == 12
    dsv4_block16_capture = _shape(
        multiprocessor_count=60,
        local_experts=256,
        w1_output_size=512,
        w2_input_size=256,
        hidden_size=4096,
        top_k=6,
        w1_scale_shape=(256, 4, 32),
        w2_scale_shape=(256, 32, 2),
        gemv_block="16x8",
        graph_mode="capture",
    )
    assert thresholds_for_shape(dsv4_block16_capture).gemv_max_tokens == 12
    dsv4_block16_mp56 = _shape(
        multiprocessor_count=56,
        local_experts=256,
        w1_output_size=512,
        w2_input_size=256,
        hidden_size=4096,
        top_k=6,
        w1_scale_shape=(256, 4, 32),
        w2_scale_shape=(256, 32, 2),
        gemv_block="16x8",
    )
    assert thresholds_for_shape(dsv4_block16_mp56).gemv_max_tokens == 8
    dsv4_block16_mp56_capture = _shape(
        multiprocessor_count=56,
        local_experts=256,
        w1_output_size=512,
        w2_input_size=256,
        hidden_size=4096,
        top_k=6,
        w1_scale_shape=(256, 4, 32),
        w2_scale_shape=(256, 32, 2),
        gemv_block="16x8",
        graph_mode="capture",
    )
    assert thresholds_for_shape(dsv4_block16_mp56_capture).gemv_max_tokens == 8
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
