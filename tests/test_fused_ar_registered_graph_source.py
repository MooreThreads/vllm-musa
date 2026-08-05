"""Source-level guards for fused CAR-RMSNorm and its Graph input path."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parents[1]
COMM = ROOT / (
    "vllm_musa/distributed/device_communicators/"
    "musa_jit_custom_all_reduce.py"
)
LAUNCHER = ROOT / "vllm_musa/jit_kernel/csrc/allreduce.py"
KERNEL = ROOT / (
    "vllm_musa/jit_kernel/csrc/distributed/custom_all_reduce.mu"
)
ENVIRON = ROOT / "vllm_musa/utils/environ.py"
FUSED_OPS = ROOT / "vllm_musa/fused_allreduce_rmsnorm_ops.py"
FUSION_PASS = ROOT / "vllm_musa/_inductor/musa_allreduce_rms_fusion.py"
PASS_MANAGER_PATCH = ROOT / (
    "vllm_musa/patches/series/"
    "0003-MUSA-vllm.compilation.passes.pass_manager.patch"
)


def _python_function_source(
    source: str, function_name: str, class_name: str | None = None
) -> str:
    tree = ast.parse(source)
    scope: ast.AST = tree
    if class_name is not None:
        classes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        assert len(classes) == 1
        scope = classes[0]
    functions = [
        node
        for node in ast.walk(scope)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    assert len(functions) == 1
    function = functions[0]
    assert function.end_lineno is not None
    return "\n".join(source.splitlines()[function.lineno - 1 : function.end_lineno])


def _isolated_fusion_helpers(torch_namespace: object, fx_namespace: object) -> type:
    """Load the actual fail-closed helpers without importing vLLM or MUSA."""
    tree = ast.parse(FUSION_PASS.read_text())
    fusion_classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "MusaAllReduceRMSNormFusionPass"
    ]
    assert len(fusion_classes) == 1
    fusion_class = fusion_classes[0]
    helper_names = {
        "_is_add_node",
        "_max_fusable_tokens",
        "is_applicable_for_range",
        "_node_tensor_meta",
        "_manual_residual_inputs_supported",
    }
    methods = [
        node
        for node in fusion_class.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    assert {method.name for method in methods} == helper_names
    for method in methods:
        method.returns = None
        arguments = (
            method.args.posonlyargs + method.args.args + method.args.kwonlyargs
        )
        for argument in arguments:
            argument.annotation = None
        if method.args.vararg is not None:
            method.args.vararg.annotation = None
        if method.args.kwarg is not None:
            method.args.kwarg.annotation = None

    helper_class = ast.ClassDef(
        name="_IsolatedFusionHelpers",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    ast.copy_location(helper_class, fusion_class)
    helper_class.end_lineno = fusion_class.end_lineno
    helper_class.end_col_offset = fusion_class.end_col_offset
    module = ast.fix_missing_locations(ast.Module(body=[helper_class], type_ignores=[]))
    namespace = {"torch": torch_namespace, "fx": fx_namespace}
    exec(compile(module, str(FUSION_PASS), "exec"), namespace)
    return namespace["_IsolatedFusionHelpers"]


class _FakeDevice:
    def __init__(self, device_type: str, index: int = 0) -> None:
        self.type = device_type
        self.index = index

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _FakeDevice)
            and self.type == other.type
            and self.index == other.index
        )


class _FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: object,
        device: _FakeDevice,
        contiguous: bool = True,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self._contiguous = contiguous

    def dim(self) -> int:
        return len(self.shape)

    def numel(self) -> int:
        result = 1
        for size in self.shape:
            result *= size
        return result

    def is_contiguous(self) -> bool:
        return self._contiguous


_MISSING = object()


class _FakeNode:
    def __init__(
        self,
        *,
        op: str = "call_function",
        target: object | None = None,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
        value: object = _MISSING,
    ) -> None:
        self.op = op
        self.target = target
        self.args = args
        self.kwargs = {} if kwargs is None else kwargs
        self.meta = {} if value is _MISSING else {"val": value}


def _fake_fusion_namespaces() -> tuple[
    object, object, object, object, object, object
]:
    tensor_overload = object()
    scalar_overload = object()
    addmm_overload = object()
    add_packet = type(
        "_FakeAddPacket",
        (),
        {"Tensor": tensor_overload, "Scalar": scalar_overload},
    )
    torch_namespace = type(
        "_FakeTorch",
        (),
        {
            "Tensor": _FakeTensor,
            "float16": "float16",
            "bfloat16": "bfloat16",
            "float32": "float32",
            "ops": type(
                "_FakeOps",
                (),
                {
                    "aten": type(
                        "_FakeAten",
                        (),
                        {
                            "add": add_packet,
                            "addmm": type(
                                "_FakeAddmmPacket",
                                (),
                                {"default": addmm_overload},
                            ),
                        },
                    )
                },
            ),
        },
    )
    fx_namespace = type("_FakeFx", (), {"Node": _FakeNode})
    return (
        torch_namespace,
        fx_namespace,
        tensor_overload,
        scalar_overload,
        addmm_overload,
        torch_namespace.bfloat16,
    )


def _ffi_function_source(source: str, function_name: str) -> str:
    start = source.index(f"void {function_name}(")
    opening_brace = source.index("{", start)
    depth = 0
    for index in range(opening_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated C++ function: {function_name}")


def _cpp_braced_block(source: str, marker: str, start: int = 0) -> tuple[str, int]:
    marker_start = source.index(marker, start)
    opening_brace = source.index("{", marker_start)
    depth = 0
    for index in range(opening_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[marker_start : index + 1], index + 1
    raise AssertionError(f"unterminated C++ block after: {marker}")


def test_fused_path_has_no_process_environment_gate() -> None:
    sources = (ENVIRON.read_text(), COMM.read_text(), FUSION_PASS.read_text())
    assert all("VLLM_MUSA_FUSED_AR_RMSNORM" not in source for source in sources)
    assert all(
        "VLLM_MUSA_FUSED_AR_RMSNORM_GRAPH_REGISTERED_INPUT" not in source
        for source in sources
    )


def test_fusion_uses_standard_pass_and_runtime_capability_gates() -> None:
    comm_source = COMM.read_text()
    fusion_source = FUSION_PASS.read_text()
    pass_manager_source = PASS_MANAGER_PATCH.read_text()

    assert "self.pass_config.fuse_allreduce_rms" in pass_manager_source
    assert "if self.disabled:" in comm_source
    assert "if self.tp_size <= 1:" in fusion_source
    assert "_graph_registered_input_eligible" in comm_source


def test_python_registered_path_uses_shared_graph_lifecycle() -> None:
    comm_source = COMM.read_text()
    launcher_source = LAUNCHER.read_text()
    comm_tree = ast.parse(comm_source)
    ast.parse(launcher_source)
    comm_class = "_MusaJitCustomAllreduceImpl"

    # Registration belongs to the communicator's existing capture lifecycle;
    # a second object would duplicate buffers and slot accounting.
    assert "_GraphInputRegistration" not in comm_source
    assert not any(
        isinstance(node, ast.ClassDef) and node.name == "_GraphInputRegistration"
        for node in ast.walk(comm_tree)
    )
    assert "self.graph_rank_data: torch.Tensor | None = None" in comm_source
    assert "self._pending_graph_inputs: list[torch.Tensor] = []" in comm_source

    capture_source = _python_function_source(comm_source, "capture", comm_class)
    first_reset = capture_source.index("self._pending_graph_inputs = []")
    registration = capture_source.index("self._register_graph_buffers()")
    final_reset = capture_source.rindex("self._pending_graph_inputs = []")
    assert first_reset < capture_source.index("yield") < registration < final_reset
    assert "if capture_succeeded and self._pending_graph_inputs:" in capture_source

    registration_source = _python_function_source(
        comm_source, "_register_graph_buffers", comm_class
    )
    for expected in (
        "count = len(self._pending_graph_inputs)",
        "dist.broadcast_object_list(",
        "self.graph_rank_data[start:end].copy_(rank_data_cpu.to(self.device))",
        "torch.musa.synchronize(self.device)",
        "self._graph_input_refs.extend(self._pending_graph_inputs)",
        "self._pending_graph_inputs = []",
        "self._next_graph_slot = end",
    ):
        assert expected in registration_source

    slot_source = _python_function_source(
        comm_source, "_graph_rank_data_for_input", comm_class
    )
    assert "slot = self._next_graph_slot + len(self._pending_graph_inputs)" in slot_source
    assert "self._pending_graph_inputs.append(tensor)" in slot_source
    assert "return self.graph_rank_data[slot]" in slot_source

    eligibility_source = _python_function_source(
        comm_source, "_use_registered_graph_input", comm_class
    )
    assert "self._graph_registered_input_eligible(tensor)" in eligibility_source
    assert "self._IS_CAPTURING" in eligibility_source
    assert "self._is_current_stream_capturing()" in eligibility_source

    graph_ar_source = _python_function_source(
        comm_source, "_graph_custom_all_reduce_impl", comm_class
    )
    assert graph_ar_source.count("self._graph_rank_data_for_input(input)") == 1
    assert "jit_ar.launch_graph_registered(" in graph_ar_source

    fused_impls = {
        "_fused_allreduce_rmsnorm_impl": (
            "launch_fused_allreduce_rmsnorm_registered",
            "launch_fused_allreduce_rmsnorm_unregistered",
        ),
        "_fused_allreduce_residual_rmsnorm_impl": (
            "launch_fused_allreduce_residual_rmsnorm_registered",
            "launch_fused_allreduce_residual_rmsnorm_unregistered",
        ),
        "_fused_allreduce_residual_rmsnorm_no_raw_impl": (
            "launch_fused_allreduce_residual_rmsnorm_no_raw_registered",
            "launch_fused_allreduce_residual_rmsnorm_no_raw_unregistered",
        ),
    }
    for function_name, launchers in fused_impls.items():
        function_source = _python_function_source(
            comm_source, function_name, comm_class
        )
        assert function_source.count("self._use_registered_graph_input(input)") == 1
        assert function_source.count("self._graph_rank_data_for_input(input)") == 1
        for launcher in launchers:
            assert f"jit_ar.{launcher}" in function_source

    registered_launchers = {
        "launch_fused_allreduce_rmsnorm_registered": (
            "launch_fused_allreduce_rmsnorm_unregistered"
        ),
        "launch_fused_allreduce_residual_rmsnorm_registered": (
            "launch_fused_allreduce_residual_rmsnorm_unregistered"
        ),
        "launch_fused_allreduce_residual_rmsnorm_no_raw_registered": (
            "launch_fused_allreduce_residual_rmsnorm_no_raw_unregistered"
        ),
    }
    for function_name, staging_launcher in registered_launchers.items():
        function_source = _python_function_source(launcher_source, function_name)
        assert "_check_registered_rank_data(rank_data)" in function_source
        assert f"{staging_launcher}(" in function_source


def test_fused_kernel_copy_is_cpu_rank_data_only() -> None:
    source = KERNEL.read_text()
    fused_apis = (
        "vllm_musa_fused_ar_rmsnorm_launch_unregistered",
        "vllm_musa_fused_ar_residual_rmsnorm_launch_unregistered",
        "vllm_musa_fused_ar_residual_rmsnorm_no_raw_launch_unregistered",
    )
    for function_name in fused_apis:
        function_source = _ffi_function_source(source, function_name)
        assert "const bool rank_data_on_cpu" in function_source
        assert (
            "rank_data.device().device_type == inp.device().device_type"
            in function_source
        )
        assert "const RankData* device_data = nullptr;" in function_source

        rank_data_block, first_block_end = _cpp_braced_block(
            function_source, "if (rank_data_on_cpu)"
        )
        second_guard = function_source.index(
            "if (rank_data_on_cpu)", first_block_end
        )
        copy_block, _ = _cpp_braced_block(
            function_source, "if (rank_data_on_cpu)", first_block_end
        )
        device_branch = function_source[first_block_end:second_guard]
        assert "musaMemcpyAsync(" not in rank_data_block
        assert "device_data = reinterpret_cast<const RankData*>" in device_branch
        assert copy_block.count("musaMemcpyAsync(") == 1
        assert "self_buffer_ptr" in copy_block
        assert "inp.data_ptr()" in copy_block
        assert "data, device_data, sg" in function_source
        assert function_source.count("musaMemcpyAsync(") == 1


def test_tp2_specialized_fast_paths_preserve_vllm_abis() -> None:
    source = KERNEL.read_text()
    assert "fused_ar_rmsnorm_tp2_specialized_kernel" in source
    assert source.count("launch_fused_ar_rmsnorm_tp2_specialized<T, WT,") == 3
    assert "hidden == 5120 && rows <= 128" in source
    assert "hidden == 2048 && rows <= 128" in source
    assert "load_weight_scalar<WT>" in source
    assert "if constexpr (WriteReduced)" in source
    assert source.count(
        'TVM_FFI_ICHECK(shot == 1 || shot == 2) << "shot must be 1 or 2";'
    ) == 3
    assert source.count(
        "if (shot == 1) {\n    if constexpr (nranks == 2)"
    ) == 3


def test_fused_ops_reuse_the_lifecycle_managed_jit_registry() -> None:
    comm_source = COMM.read_text()
    fused_source = FUSED_OPS.read_text()
    fusion_source = FUSION_PASS.read_text()
    ast.parse(fused_source)
    assert "def get_musa_jit_custom_allreduce_comm" in comm_source
    assert "get_musa_jit_custom_allreduce_comm" in fused_source
    assert "self.comm_id = self.jit_comm_id" in fusion_source


def test_manual_rewrite_is_scoped_to_comm_and_full_variance() -> None:
    source = FUSION_PASS.read_text()
    assert "def _is_target_musa_car_node" in source
    assert "self._car_comm_id(node) == self.jit_comm_id" in source
    assert "if not self._is_target_musa_car_node(car):" in source
    assert source.count("variance_size is not None") == 3
    rewrite_source = _python_function_source(
        source,
        "_manual_rewrite_residual_musa_jit_car_rmsnorm",
        "MusaAllReduceRMSNormFusionPass",
    )
    assert rewrite_source.count("self._manual_residual_inputs_supported(") == 2


def test_manual_add_rewrite_accepts_only_tensor_overload_with_unit_alpha() -> None:
    (
        torch_namespace,
        fx_namespace,
        tensor_overload,
        scalar_overload,
        addmm_overload,
        _,
    ) = _fake_fusion_namespaces()
    helpers = _isolated_fusion_helpers(torch_namespace, fx_namespace)
    lhs = _FakeNode(op="placeholder")
    rhs = _FakeNode(op="placeholder")

    assert helpers._is_add_node(
        _FakeNode(target=tensor_overload, args=(lhs, rhs))
    )
    assert helpers._is_add_node(
        _FakeNode(target=tensor_overload, args=(lhs, rhs), kwargs={"alpha": 1.0})
    )
    assert not helpers._is_add_node(
        _FakeNode(target=tensor_overload, args=(lhs, rhs), kwargs={"alpha": 2})
    )
    assert not helpers._is_add_node(
        _FakeNode(target=scalar_overload, args=(lhs, 1))
    )
    assert not helpers._is_add_node(
        _FakeNode(target=addmm_overload, args=(lhs, rhs, lhs))
    )
    assert not helpers._is_add_node(
        _FakeNode(target=tensor_overload, args=(lhs, rhs, 1))
    )
    assert not helpers._is_add_node(
        _FakeNode(
            target=tensor_overload,
            args=(lhs, rhs),
            kwargs={"alpha": 1, "out": _FakeNode(op="placeholder")},
        )
    )


def test_manual_residual_metadata_gate_is_positive_and_fail_closed() -> None:
    (
        torch_namespace,
        fx_namespace,
        _,
        _,
        _,
        input_dtype,
    ) = _fake_fusion_namespaces()
    helpers = _isolated_fusion_helpers(torch_namespace, fx_namespace)()
    helpers.hidden_dim = 16
    device = _FakeDevice("musa", 0)

    def tensor_node(
        shape: tuple[int, ...],
        dtype: object = input_dtype,
        tensor_device: _FakeDevice = device,
        contiguous: bool = True,
    ) -> _FakeNode:
        return _FakeNode(
            op="placeholder",
            value=_FakeTensor(shape, dtype, tensor_device, contiguous),
        )

    input_node = tensor_node((4, 16))
    car = _FakeNode(args=(input_node,), value=_FakeTensor((4, 16), input_dtype, device))
    residual = tensor_node((4, 16))
    weight = tensor_node((16,), torch_namespace.float32)

    assert helpers._manual_residual_inputs_supported(car, residual, weight)
    assert not helpers._manual_residual_inputs_supported(car, 1, weight)
    assert not helpers._manual_residual_inputs_supported(
        car, _FakeNode(op="placeholder", value=1), weight
    )
    assert not helpers._manual_residual_inputs_supported(
        car, tensor_node((2, 16)), weight
    )
    assert not helpers._manual_residual_inputs_supported(
        car, tensor_node((4, 16), "float32"), weight
    )
    assert not helpers._manual_residual_inputs_supported(
        car, tensor_node((4, 16), tensor_device=_FakeDevice("musa", 1)), weight
    )
    assert not helpers._manual_residual_inputs_supported(
        car, tensor_node((4, 16), contiguous=False), weight
    )
    assert not helpers._manual_residual_inputs_supported(
        car, residual, tensor_node((8,), torch_namespace.float32)
    )
    helpers.hidden_dim = 8
    assert not helpers._manual_residual_inputs_supported(car, residual, weight)


def test_fusion_runtime_state_participates_in_cache_uuid() -> None:
    source = FUSION_PASS.read_text()
    assert "def uuid(self) -> str:" in source
    assert '"enabled": not self.disabled' in source
    assert '"comm_id": self.comm_id' in source
    assert '"jit_comm_id": self.jit_comm_id' in source
    assert '"group_name": self.group_name' in source
    assert '"max_tokens_by_comm": self.max_tokens_by_comm' in source
    assert '"jit_comm_max_size": self.jit_comm_max_size' in source


def test_fusion_compile_range_respects_jit_comm_byte_limit() -> None:
    (
        torch_namespace,
        fx_namespace,
        _,
        _,
        _,
        _,
    ) = _fake_fusion_namespaces()
    helpers = _isolated_fusion_helpers(torch_namespace, fx_namespace)()
    max_size = 512 * 1024 * 1024
    helpers.max_token_num = helpers._max_fusable_tokens(
        20_000, max_size, 16_384, 2
    )
    assert helpers.max_token_num == 16_384
    helpers.disabled = False
    compile_range = type("_FakeRange", (), {"end": 16_384})()
    assert helpers.is_applicable_for_range(compile_range)
    compile_range.end = 16_385
    assert not helpers.is_applicable_for_range(compile_range)


def test_musa_fusion_respects_vllm_standard_disable_switch() -> None:
    source = PASS_MANAGER_PATCH.read_text()
    assert (
        "if current_platform.is_musa() and self.pass_config.fuse_allreduce_rms:"
        in source
    )


def test_fused_kernel_validates_world_size_before_fixed_registry_arrays() -> None:
    source = KERNEL.read_text()
    for function_name in (
        "vllm_musa_fused_ar_rmsnorm_launch_unregistered",
        "vllm_musa_fused_ar_residual_rmsnorm_launch_unregistered",
        "vllm_musa_fused_ar_residual_rmsnorm_no_raw_launch_unregistered",
    ):
        function_source = _ffi_function_source(source, function_name)
        validate = function_source.index("validate_world_size(world_size);")
        rank_signals = function_source.index("RankSignals sg{};")
        assert validate < rank_signals, function_name


def test_base_all_gather_and_indirect_graph_abis_are_retained() -> None:
    kernel_source = KERNEL.read_text()
    launcher_source = LAUNCHER.read_text()
    all_gather = _ffi_function_source(
        kernel_source, "vllm_musa_custom_ar_launch_all_gather"
    )
    graph_registered = _ffi_function_source(
        kernel_source, "vllm_musa_custom_ar_launch_graph_registered"
    )
    assert "dispatch_all_gather_world_size" in all_gather
    assert "dispatch_world_size_indirect" in graph_registered
    assert (
        ".vllm_musa_custom_ar_launch_all_gather("
        in _python_function_source(launcher_source, "launch_all_gather")
    )
    assert (
        ".vllm_musa_custom_ar_launch_graph_registered("
        in _python_function_source(launcher_source, "launch_graph_registered")
    )
    for function_name in (
        "vllm_musa_custom_ar_launch_all_gather",
        "vllm_musa_custom_ar_launch_graph_registered",
    ):
        assert f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({function_name}," in kernel_source


def test_no_raw_one_shot_matches_raw_dtype_rounding_order() -> None:
    source = KERNEL.read_text()
    assert "const T reduced_value = from_float<T>(acc[i]);" in source
    assert (
        "to_float(reduced_value) + to_float(residual_pack.data[i])"
        in source
    )


def test_tp4_rows64_two_shot_uses_bounded_block_count() -> None:
    kernel_source = KERNEL.read_text()
    assert "kTp4Rows64TwoShotBlockLimit = 32" in kernel_source
    assert "if constexpr (nranks == 4)" in kernel_source
    assert "if (rows == 64 && hidden == 2048)" in kernel_source
    assert (
        kernel_source.count(
            "fused_ar_rmsnorm_2shot_blocks<nranks>(rows, hidden)"
        )
        == 3
    )


def test_registered_graph_input_has_a_fixed_staging_boundary() -> None:
    comm_source = COMM.read_text()
    assert "_GRAPH_REGISTERED_INPUT_MAX_BYTES = 512 * 1024" in comm_source
    assert (
        "tensor.numel() * tensor.element_size()\n"
        "            <= self._GRAPH_REGISTERED_INPUT_MAX_BYTES"
    ) in comm_source
