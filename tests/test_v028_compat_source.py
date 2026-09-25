import ast
from pathlib import Path

import pytest

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


UPSTREAM_MLA_ATTENTION = (
    ROOT
    / "third_party"
    / "vllm"
    / "vllm"
    / "model_executor"
    / "layers"
    / "attention"
    / "mla_attention.py"
)
MUSA_MLA_COMMON = (
    ROOT / "vllm_musa" / "v1" / "attention" / "backends" / "mla" / "common.py"
)

# Objects whose attribute reads must exist in the pinned upstream metadata.
_METADATA_RECEIVERS = frozenset(
    {"prefill", "prefill_metadata", "_prefill_metadata", "chunked_context", "chunk"}
)
# Reads past these attributes stay inside a different object, not metadata fields.
_OPAQUE_ATTRIBUTES = frozenset({"dcp_manager"})


def _declared_metadata_attributes(source: str) -> set[str]:
    """Every field, property and method of the MLA prefill metadata tree."""
    wanted = {"MLACommonPrefillMetadata", "ContextChunk", "ChunkedContextMetadata"}
    declared: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef) or node.name not in wanted:
            continue
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target, ast.Name
            ):
                declared.add(statement.target.id)
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                declared.add(statement.name)
    return declared


def _metadata_reads(source: str) -> list[tuple[int, str, tuple[str, ...]]]:
    """Attribute chains rooted at a metadata receiver, as (line, receiver, path)."""
    reads = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Attribute) or not isinstance(node.ctx, ast.Load):
            continue
        path: list[str] = []
        root: ast.expr = node
        while isinstance(root, ast.Attribute):
            path.append(root.attr)
            root = root.value
        if isinstance(root, ast.Name) and root.id in _METADATA_RECEIVERS:
            reads.append((node.lineno, root.id, tuple(reversed(path))))
    return reads


def test_musa_mla_chunked_context_reads_only_pinned_upstream_fields() -> None:
    """Guard the MUSA MLA prefill/context metadata contract against pin drift.

    vLLM v0.28 replaced the flat chunked-context lists (`seq_tot`,
    `cu_seq_lens`, `token_to_seq`, `chunk_total_token`, `starts`,
    `max_seq_lens`, ...) with per-request `chunks: list[ContextChunk]`. This
    module vendors its own MLA prefill execution, so a stale read of a removed
    field does not fail at import or construction: it only raises on the
    chunked-prefill branch, which no smoke without a cached prefix reaches
    (MUSA-100055). Check every metadata read against the pinned upstream
    declarations instead of waiting for that branch to run.
    """
    if not UPSTREAM_MLA_ATTENTION.exists():
        pytest.skip(
            "pinned upstream vLLM source is not present at "
            f"{UPSTREAM_MLA_ATTENTION}; third_party/vllm is cloned during the "
            "image build, so this field-contract guard did NOT run"
        )

    declared = _declared_metadata_attributes(UPSTREAM_MLA_ATTENTION.read_text())
    assert "seq_tot" not in declared, "the pin still carries the pre-v0.28 layout"
    assert "chunks" in declared
    assert "num_context_tokens" in declared

    unknown: list[str] = []
    for lineno, receiver, path in _metadata_reads(MUSA_MLA_COMMON.read_text()):
        for depth, attribute in enumerate(path):
            if attribute in _OPAQUE_ATTRIBUTES:
                break
            if attribute not in declared:
                unknown.append(
                    f"common.py:{lineno}: {receiver}.{'.'.join(path[: depth + 1])}"
                )
                break

    assert not unknown, (
        "MUSA MLA reads metadata fields the pinned upstream vLLM no longer "
        "declares; migrate them to the v0.28 `chunks`/`ContextChunk` layout:\n  "
        + "\n  ".join(sorted(unknown))
    )

    # Both the non-DCP and the DCP context path must iterate the v0.28 layout.
    source = MUSA_MLA_COMMON.read_text()
    assert source.count("for chunk in chunked_context.chunks:") == 2
    assert "chunk.token_slice" in source
    assert "empty_token_slices" in source


def test_musa_mla_helpers_match_pinned_upstream_signatures() -> None:
    """Guard helper call sites whose upstream signature moved with the pin.

    The same v0.28 rework that reshaped the chunked-context metadata also
    changed `reorg_kvcache` (`local_starts` in, `chunk_size`/`chunk_idx` out),
    which is the DCP path's second, independent break: a stale keyword is a
    TypeError that mid-batch prefix-cache traffic alone reveals (MUSA-100055).
    """
    if not UPSTREAM_MLA_ATTENTION.exists():
        pytest.skip(
            "pinned upstream vLLM source is not present at "
            f"{UPSTREAM_MLA_ATTENTION}; third_party/vllm is cloned during the "
            "image build, so this signature guard did NOT run"
        )

    upstream = ast.parse(UPSTREAM_MLA_ATTENTION.read_text())
    fork = ast.parse(MUSA_MLA_COMMON.read_text())

    checked = 0
    for node in ast.walk(upstream):
        if not isinstance(node, ast.FunctionDef) or node.name != "reorg_kvcache":
            continue
        accepted = {arg.arg for arg in node.args.args}
        accepted |= {arg.arg for arg in node.args.kwonlyargs}
        assert "local_starts" in accepted

        for call in ast.walk(fork):
            if not isinstance(call, ast.Call):
                continue
            callee = call.func
            if not (isinstance(callee, ast.Name) and callee.id == "reorg_kvcache"):
                continue
            checked += 1
            stale = sorted(kw.arg for kw in call.keywords if kw.arg not in accepted)
            assert not stale, (
                f"common.py:{call.lineno}: reorg_kvcache no longer accepts {stale}; "
                f"the pin declares {sorted(accepted)}"
            )
        break

    assert checked == 1, f"expected one reorg_kvcache call site, found {checked}"
