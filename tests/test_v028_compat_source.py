import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_triton_gluon_is_optional_for_musa_triton_32() -> None:
    source = (
        ROOT / "third_party" / "vllm" / "vllm" / "triton_utils" / "__init__.py"
    ).read_text()
    tree = ast.parse(source)

    gluon_import_guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and any(
            isinstance(child, ast.ImportFrom)
            and child.module == "triton.experimental"
            for child in node.body
        )
        and any(
            isinstance(child, ast.ImportFrom)
            and child.module == "triton.language.core"
            and any(alias.name == "_aggregate" for alias in child.names)
            for child in node.body
        )
        and any(
            isinstance(handler.type, ast.Name)
            and handler.type.id == "ImportError"
            for handler in node.handlers
        )
    ]

    assert len(gluon_import_guards) == 1
    assert source.count("aggregate = TritonLanguagePlaceholder()") == 2


def test_moe_overrides_use_v028_routed_experts_api() -> None:
    fp8_source = (
        ROOT
        / "vllm_musa"
        / "model_executor"
        / "layers"
        / "quantization"
        / "fp8.py"
    ).read_text()
    unquantized_source = (
        ROOT
        / "vllm_musa"
        / "model_executor"
        / "layers"
        / "fused_moe"
        / "unquantized_fused_moe_method.py"
    ).read_text()

    assert "from vllm.model_executor.layers.fused_moe import RoutedExperts" in (
        fp8_source
    )
    assert "layer: FusedMoE" not in fp8_source
    for source in (fp8_source, unquantized_source):
        assert (
            "from vllm.model_executor.layers.fused_moe.fused_moe import "
            "fused_experts"
        ) in source


def test_qwen_uniform_decode_selector_uses_v028_scheduled_tokens_array() -> None:
    source = (
        ROOT
        / "third_party"
        / "vllm"
        / "vllm"
        / "v1"
        / "worker"
        / "gpu"
        / "model_runner.py"
    ).read_text()

    assert "req_ids,\n                num_scheduled_tokens_np," in source
    assert "req_ids,\n                num_scheduled_tokens," not in source
    assert "num_scheduled_tokens_np,\n                batch_req_state.is_prefilling_np," in (
        source
    )


def test_fused_moe_tensor_descriptor_is_hashable_on_musa_triton_32() -> None:
    source = (
        ROOT
        / "third_party"
        / "vllm"
        / "vllm"
        / "model_executor"
        / "layers"
        / "fused_moe"
        / "fused_moe.py"
    ).read_text()

    assert 'hasattr(tl, "make_tensor_descriptor")' in source
    assert 'hasattr(tl, "_experimental_make_tensor_descriptor")' in source
    assert "def make_tensor_descriptor(" in source
    assert "tl.make_tensor_descriptor(" not in source


def test_musa_mla_prefill_backend_uses_v028_clone_contract() -> None:
    source = (
        ROOT
        / "vllm_musa"
        / "v1"
        / "attention"
        / "backends"
        / "mla"
        / "common.py"
    ).read_text()

    assert "class MUSAMLAPrefillBackend(MLAPrefillBackend):" in source
    assert "super().__init__(" in source


def test_musa_mla_uses_v028_decode_context_parallel_size() -> None:
    source = (
        ROOT
        / "vllm_musa"
        / "v1"
        / "attention"
        / "backends"
        / "mla"
        / "common.py"
    ).read_text()

    assert (
        "self.dcp_world_size: int = "
        "parallel_config.decode_context_parallel_size"
    ) in source
    assert "self.dcp_world_size: int = -1" not in source


def test_musa_qwen_gdn_forwards_v028_reduce_results() -> None:
    source = (
        ROOT
        / "vllm_musa"
        / "model_executor"
        / "layers"
        / "mamba"
        / "gdn"
        / "qwen_gdn_linear_attn.py"
    ).read_text()

    assert "reduce_results: bool = True," in source
    assert "reduce_results=reduce_results," in source
    assert source.count("vllm.third_party.flash_linear_attention.ops") == 2
    assert "vllm.model_executor.layers.fla.ops" not in source

    tree = ast.parse(source)
    musa_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "MusaQwenGatedDeltaNetAttention"
    )
    forward_cuda = next(
        node
        for node in musa_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward_cuda"
    )
    assert [arg.arg for arg in forward_cuda.args.args] == ["self", "hidden_states"]
    assert "return self._output_projection(core_attn_out, z)" in source


def test_cuda_only_fa4_warmup_is_skipped_on_musa() -> None:
    source = (
        ROOT
        / "third_party"
        / "vllm"
        / "vllm"
        / "model_executor"
        / "warmup"
        / "kernel_warmup.py"
    ).read_text()

    assert "if not current_platform.is_musa():" in source
    assert 'Skipping CUDA-only FA4 CuTeDSL warmup on MUSA.' in source


def test_musa_mamba_pools_accept_v028_graph_profiling_override() -> None:
    source = (
        ROOT
        / "third_party"
        / "vllm"
        / "vllm"
        / "v1"
        / "core"
        / "kv_cache_utils.py"
    ).read_text()

    assert "profiling_num_blocks = (" in source
    assert "if available_memory == 0" in source
    assert "attn_num_blocks = profiling_num_blocks" in source
