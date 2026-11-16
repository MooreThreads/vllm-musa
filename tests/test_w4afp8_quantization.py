from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vllm_musa/model_executor/layers/quantization/w4afp8.py"


@pytest.fixture
def implementation():
    class BaseConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.ignored_layers = kwargs.get("ignored_layers") or []
            self.packed_modules_mapping = kwargs.get("packed_modules_mapping", {})

        def get_quant_method(self, layer, prefix):
            return "dense_fp8", self.is_scale_e8m0

    parser_namespace = {"QuantizationConfig": BaseConfig}
    quantization_root = (
        ROOT / "third_party/vllm/vllm/model_executor/layers/quantization"
    )
    for filename, class_name, method_names in (
        ("base_config.py", "QuantizationConfig", {"get_from_keys", "get_from_keys_or"}),
        ("fp8.py", "Fp8Config", {"from_config"}),
    ):
        path = quantization_root / filename
        if not path.is_file():
            pytest.skip(f"requires pinned vLLM config source: {path}")
        tree = ast.parse(path.read_text())
        definition = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        tree.body = [
            node
            for node in definition.body
            if isinstance(node, ast.FunctionDef) and node.name in method_names
        ]
        exec(
            compile(tree, str(path), "exec", flags=__future_annotations_flag()),
            parser_namespace,
        )
        for method_name in method_names:
            setattr(BaseConfig, method_name, parser_namespace[method_name])

    class BaseMoE:
        def __init__(self, config):
            self.moe = config

    def set_attrs(parameter, attrs):
        for name, value in attrs.items():
            setattr(parameter, name, value)

    tree = ast.parse(SOURCE.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    namespace = dict(
        torch=torch,
        Parameter=torch.nn.Parameter,
        DeepseekV4FP8Config=BaseConfig,
        FusedMoEMethodBase=BaseMoE,
        RoutedExperts=torch.nn.Linear,
        FusedMoeWeightScaleSupported=SimpleNamespace(
            GROUP=SimpleNamespace(value="group")
        ),
        set_weight_attrs=set_attrs,
        is_layer_skipped=lambda prefix, ignored_layers, fused_mapping: (
            prefix in ignored_layers
        ),
        logger=SimpleNamespace(info_once=lambda *args: None),
    )
    exec(
        compile(tree, str(SOURCE), "exec", flags=__future_annotations_flag()), namespace
    )
    return namespace["MusaW4AFp8Config"], namespace["MusaW4AFp8MoEMethod"]


def __future_annotations_flag():
    import __future__

    return __future__.annotations.compiler_flag


def test_config_keeps_dense_scale_format_separate_from_int4_experts(implementation):
    config_type, _ = implementation
    config = config_type.from_config(
        dict(
            quant_method="w4afp8",
            activation_scheme="dynamic",
            moe_group_size=128,
            weight_block_size=[32, 32],
            scale_fmt="ue8m0",
        )
    )
    assert config.get_name() == "w4afp8"
    assert config.expert_dtype == "int4"
    assert config.is_scale_e8m0
    assert config.weight_block_size == [32, 32]
    assert config.get_quant_method(torch.nn.Identity(), "shared_expert") == (
        "dense_fp8",
        True,
    )


def test_routed_experts_use_w4afp8_method(implementation):
    config_type, method_type = implementation
    config = config_type()
    layer = torch.nn.Linear(128, 128)
    layer.moe_config = SimpleNamespace(swiglu_limit=10.0)

    method = config.get_quant_method(layer, "model.layers.0.mlp.experts")

    assert isinstance(method, method_type)
    assert method.quant_config is config
    assert method.moe is layer.moe_config


def test_ignored_experts_keep_parent_dispatch(implementation):
    config_type, _ = implementation
    prefix = "model.layers.0.mlp.experts"
    config = config_type(ignored_layers=[prefix])

    assert config.get_quant_method(torch.nn.Linear(128, 128), prefix) == (
        "dense_fp8",
        True,
    )


@pytest.mark.parametrize("scale_fmt,expected", [("ue8m0", True), ("float32", False)])
def test_dense_scale_format_does_not_change_routed_expert_dtype(
    implementation, scale_fmt, expected
):
    config_type, _ = implementation
    config = config_type.from_config(
        dict(
            quant_method="w4afp8",
            activation_scheme="dynamic",
            scale_fmt=scale_fmt,
        )
    )

    assert config.is_scale_e8m0 is expected
    assert config.expert_dtype == "int4"
    assert config.get_quant_method(torch.nn.Identity(), "shared_expert") == (
        "dense_fp8",
        expected,
    )


def test_config_preserves_required_parent_activation_scheme(implementation):
    config_type, _ = implementation
    with pytest.raises(ValueError, match="activation_scheme"):
        config_type.from_config(dict(quant_method="w4afp8", scale_fmt="ue8m0"))


@pytest.mark.parametrize(
    "overrides", [dict(moe_group_size=32), dict(activation_scheme="static")]
)
def test_unsupported_checkpoint_contract_is_rejected(implementation, overrides):
    config_type, _ = implementation
    with pytest.raises(ValueError):
        config_type.from_config(dict(quant_method="w4afp8", **overrides))


def test_tp8_unaligned_groups_fail_instead_of_silently_slicing_scales(implementation):
    config_type, method_type = implementation
    method = method_type(config_type(), SimpleNamespace())
    assert method.maybe_roundup_sizes(5120, 2304, torch.bfloat16, None) == (5120, 2304)
    with pytest.raises(ValueError, match="enable-expert-parallel"):
        method.maybe_roundup_sizes(5120, 288, torch.bfloat16, None)


def test_packed_weight_allocation_and_scale_conversion(implementation):
    config_type, method_type = implementation
    method = method_type(config_type(), SimpleNamespace())
    layer = torch.nn.Module()
    loader = object()
    method.create_weights(layer, 2, 256, 384, torch.bfloat16, weight_loader=loader)
    assert layer.w13_weight.shape == (2, 768, 128)
    assert layer.w2_weight.shape == (2, 256, 192)
    assert layer.w13_weight_scale_inv.shape == (2, 768, 2)
    assert layer.w2_weight_scale_inv.shape == (2, 256, 3)
    assert layer.w13_weight.dtype == torch.int8
    assert layer.w13_weight.weight_loader is loader
    assert layer.w2_weight_scale_inv.quant_method == "group"
    scales = torch.linspace(
        0.001, 0.013, layer.w13_weight_scale_inv.numel()
    ).reshape_as(layer.w13_weight_scale_inv)
    layer.w13_weight_scale_inv.data.copy_(scales)
    layer.w2_weight_scale_inv.data.fill_(0.007)
    method.process_weights_after_loading(layer)
    assert layer.w13_weight.dtype == torch.int8
    assert layer.w13_weight_scale_inv.dtype == torch.bfloat16
    torch.testing.assert_close(
        layer.w13_weight_scale_inv, scales.to(torch.bfloat16), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "invalid", [float("nan"), float("inf"), -0.1, 0.0, 1.0e-42, 3.4e38]
)
@pytest.mark.parametrize("scale_name", ["w13_weight_scale_inv", "w2_weight_scale_inv"])
def test_invalid_loaded_scales_fail_before_kernel_launch(
    implementation, invalid, scale_name
):
    config_type, method_type = implementation
    method = method_type(config_type(), SimpleNamespace())
    layer = torch.nn.Module()
    method.create_weights(layer, 1, 128, 128, torch.bfloat16, weight_loader=object())
    layer.w13_weight_scale_inv.data.fill_(0.01)
    layer.w2_weight_scale_inv.data.fill_(0.01)
    getattr(layer, scale_name).data.fill_(invalid)
    with pytest.raises(
        ValueError, match=f"Invalid W4AFP8 weight scale in {scale_name}"
    ):
        method.process_weights_after_loading(layer)


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1)])
def test_packed_weights_and_group128_scales_use_matching_loader_shards(
    implementation, tp_size, tp_rank
):
    path = (
        ROOT / "third_party/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py"
    )
    if not path.exists():
        pytest.skip("requires the pinned patched vLLM source tree")
    tree = ast.parse(path.read_text())
    loader_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RoutedExperts"
    )
    methods = {
        "_load_model_weight_or_group_weight_scale",
        "_load_w13",
        "_load_w2",
        "_tp_shard_indices",
        "_get_hidden_dim",
        "_narrow_expert_data_for_padding",
    }
    loader_class.bases = []
    loader_class.decorator_list = []
    loader_class.body = [
        node
        for node in loader_class.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    tree.body = [loader_class]
    namespace = {"torch": torch}
    exec(compile(tree, str(path), "exec", flags=__future_annotations_flag()), namespace)
    loader = namespace["RoutedExperts"]()
    loader.moe_config = SimpleNamespace(
        tp_shard_with_padding=False,
        is_act_and_mul=True,
        moe_parallel_config=SimpleNamespace(tp_size=tp_size),
    )
    config_type, method_type = implementation
    method = method_type(config_type(), SimpleNamespace())
    layer = torch.nn.Module()
    intermediate, hidden = 256, 128
    partition = intermediate // tp_size
    method.create_weights(layer, 1, hidden, partition, torch.bfloat16)
    for shard_id, parameter_name, shape, shard_dim, dtype in (
        ("w1", "w13_weight", (intermediate, hidden // 2), 0, torch.int8),
        ("w3", "w13_weight", (intermediate, hidden // 2), 0, torch.int8),
        ("w2", "w2_weight", (hidden, intermediate // 2), 1, torch.int8),
        ("w1", "w13_weight_scale_inv", (intermediate, hidden // 128), 0, torch.float32),
        ("w3", "w13_weight_scale_inv", (intermediate, hidden // 128), 0, torch.float32),
        ("w2", "w2_weight_scale_inv", (hidden, intermediate // 128), 1, torch.float32),
    ):
        checkpoint = torch.arange(shape[0] * shape[1]).reshape(shape).to(dtype)
        destination = getattr(layer, parameter_name).data[0]
        loader._load_model_weight_or_group_weight_scale(
            shard_dim,
            destination,
            shard_id,
            checkpoint,
            tp_rank,
            is_scale="scale" in parameter_name,
        )
        expected = checkpoint.chunk(tp_size, dim=shard_dim)[tp_rank]
        if shard_id in ("w1", "w3"):
            destination = destination.chunk(2, dim=0)[shard_id == "w3"]
        torch.testing.assert_close(destination, expected, rtol=0, atol=0)


def test_model_resolves_int4_despite_checkpoint_fp4_placeholder(implementation):
    path = ROOT / "third_party/vllm/vllm/models/deepseek_v41/nvidia/model.py"
    if not path.exists():
        pytest.skip("requires the pinned patched vLLM source tree")
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_expert_dtype"
    ]
    namespace = {}
    exec(compile(tree, str(path), "exec", flags=__future_annotations_flag()), namespace)
    config_type, _ = implementation
    resolved = namespace["_resolve_expert_dtype"](
        SimpleNamespace(expert_dtype="fp4"), config_type()
    )
    assert resolved == "int4"


@pytest.fixture
def kernel_entrypoint():
    path = ROOT / "vllm_musa/kernels/w4afp8_moe.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "w4afp8_moe"
    ]
    namespace = {"torch": torch}
    exec(compile(tree, str(path), "exec", flags=__future_annotations_flag()), namespace)
    return namespace["w4afp8_moe"]


def test_empty_kernel_input_does_not_require_mate_or_launch(kernel_entrypoint):
    output = kernel_entrypoint(
        torch.empty(0, 128, dtype=torch.bfloat16),
        torch.empty(1, 256, 64, dtype=torch.int8),
        torch.empty(1, 128, 64, dtype=torch.int8),
        torch.ones(1, 256, 1, dtype=torch.bfloat16),
        torch.ones(1, 128, 1, dtype=torch.bfloat16),
        torch.empty(0, 1),
        torch.empty(0, 1, dtype=torch.int32),
    )
    assert output.shape == (0, 128)
    assert output.dtype == torch.bfloat16


@pytest.fixture
def stubbed_kernel_entrypoint(kernel_entrypoint, monkeypatch):
    class NoOpKernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: None

    gemm = ModuleType("mate.gemm")
    calls = []
    gemm.GemmMixedDType = SimpleNamespace(S4FP8=object())
    gemm.masked_moe_gemm_mixed_dtype = lambda *args, **kwargs: calls.append(
        (args, kwargs)
    )
    gemm.ragged_moe_gemm_mixed_dtype = lambda *args, **kwargs: calls.append(
        (args, kwargs)
    )
    mate = ModuleType("mate")
    mate.gemm = gemm
    monkeypatch.setitem(sys.modules, "mate", mate)
    monkeypatch.setitem(sys.modules, "mate.gemm", gemm)
    monkeypatch.setitem(
        kernel_entrypoint.__globals__,
        "triton",
        SimpleNamespace(
            cdiv=lambda numerator, denominator: (
                (numerator + denominator - 1) // denominator
            ),
            next_power_of_2=lambda value: 1 << (value - 1).bit_length(),
        ),
    )
    for name in (
        "_initialize_ragged_indices",
        "_assign_expert_rows",
        "_scatter_quantized_rows",
        "_swiglu_quantized_rows",
        "_combine_expert_rows",
    ):
        monkeypatch.setitem(kernel_entrypoint.__globals__, name, NoOpKernel())
    return kernel_entrypoint, calls


def test_output_allocation_is_contiguous_for_transposed_input(
    stubbed_kernel_entrypoint,
):
    kernel_entrypoint, _ = stubbed_kernel_entrypoint
    states = torch.empty(128, 2, dtype=torch.bfloat16).T
    assert not states.is_contiguous()

    output = kernel_entrypoint(
        states,
        torch.empty(1, 256, 64, dtype=torch.int8),
        torch.empty(1, 128, 64, dtype=torch.int8),
        torch.ones(1, 256, 1, dtype=torch.bfloat16),
        torch.ones(1, 128, 1, dtype=torch.bfloat16),
        torch.ones(2, 1),
        torch.zeros(2, 1, dtype=torch.int32),
    )

    assert output.is_contiguous()
    assert output.shape == states.shape
    assert output.dtype == states.dtype


@pytest.mark.parametrize(
    "tokens,topk,backend,capacity",
    [
        (1, 6, "auto", 8),
        (16, 2, "auto", 32),
        (33, 1, "auto", 128),
        (17, 2, "auto", 128),
        (64, 2, "auto", 128),
        (65, 2, "mubin", None),
    ],
)
def test_small_masked_capacity_and_backend_preserve_large_paths(
    stubbed_kernel_entrypoint, tokens, topk, backend, capacity
):
    kernel_entrypoint, calls = stubbed_kernel_entrypoint
    kernel_entrypoint(
        torch.empty(tokens, 128, dtype=torch.bfloat16),
        torch.empty(1, 256, 64, dtype=torch.int8),
        torch.empty(1, 128, 64, dtype=torch.int8),
        torch.ones(1, 256, 1, dtype=torch.bfloat16),
        torch.ones(1, 128, 1, dtype=torch.bfloat16),
        torch.full((tokens, topk), 1.0 / topk),
        torch.zeros(tokens, topk, dtype=torch.int32),
    )

    assert len(calls) == 2
    for arguments, options in calls:
        assert options["backend"] == backend
        if capacity is None:
            assert arguments[0][0].ndim == 2
            assert options["alignment_m"] == 128
        else:
            assert arguments[0][0].shape[:2] == (1, capacity)
            assert options["expect_tokens"] == tokens * topk


@pytest.mark.parametrize(
    "experts,routes", [(256, 129), (512, 130), (2, 129), (48, 384)]
)
def test_ragged_capacity_bounds_only_potentially_active_experts(
    stubbed_kernel_entrypoint, experts, routes
):
    kernel_entrypoint, calls = stubbed_kernel_entrypoint
    kernel_entrypoint(
        torch.empty(routes, 128, dtype=torch.bfloat16),
        torch.empty(experts, 256, 64, dtype=torch.int8),
        torch.empty(experts, 128, 64, dtype=torch.int8),
        torch.ones(experts, 256, 1, dtype=torch.bfloat16),
        torch.ones(experts, 128, 1, dtype=torch.bfloat16),
        torch.ones(routes, 1),
        torch.zeros(routes, 1, dtype=torch.int32),
    )

    active_experts = min(experts, routes)
    expected_rows = ((routes + active_experts * 127 + 127) // 128) * 128
    counts = [routes - active_experts + 1] + [1] * (active_experts - 1)
    for distribution in (counts, [routes], [0] * experts):
        needed_rows = sum(((count + 127) // 128) * 128 for count in distribution)
        assert needed_rows <= expected_rows

    assert len(calls) == 2
    for arguments, options in calls:
        assert options["backend"] == "mubin"
        assert options["alignment_m"] == 128
        assert arguments[0][0].shape[0] == expected_rows
        assert arguments[0][1].shape == (expected_rows, 1)
        assert arguments[2].shape == (expected_rows,)
        assert arguments[3].shape[0] == expected_rows


@pytest.mark.parametrize(
    "kernel_name,index_name",
    [
        ("_scatter_quantized_rows", "route"),
        ("_scatter_quantized_rows", "destination"),
        ("_swiglu_quantized_rows", "row"),
        ("_combine_expert_rows", "token"),
        ("_combine_expert_rows", "source"),
    ],
)
def test_large_row_pointer_offsets_promote_indices_before_multiplication(
    kernel_name, index_name
):
    tree = ast.parse((ROOT / "vllm_musa/kernels/w4afp8_moe.py").read_text())
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == kernel_name
    )
    casts = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == index_name
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "to"
        and any(
            isinstance(argument, ast.Attribute)
            and isinstance(argument.value, ast.Name)
            and argument.value.id == "tl"
            and argument.attr == "int64"
            for argument in node.value.args
        )
    ]
    assert casts, f"{kernel_name}.{index_name} must use 64-bit pointer arithmetic"


def test_kernel_forward_has_no_host_scalar_reads_or_weight_dequantization():
    source = (ROOT / "vllm_musa/kernels/w4afp8_moe.py").read_text()
    tree = ast.parse(source)
    names = [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)]
    assert "item" not in names
    assert "cpu" not in names
    assert "ragged_moe_gemm_mixed_dtype" in names
    assert "masked_moe_gemm_mixed_dtype" in names
    assert "GemmMixedDType" in names
