import ast
import importlib.util
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
# The pinned class each receiver name is bound to (`chunk` is an element of
# `chunked_context.chunks`). Binding a type per receiver is what makes this guard
# strict: `query_start_loc` and `max_query_len` are declared by BOTH
# `ContextChunk` and `MLACommonPrefillMetadata`, so testing membership in the
# union of the three classes lets a relocation of one of them pass silently.
_RECEIVER_TYPES = {
    "prefill": "MLACommonPrefillMetadata",
    "prefill_metadata": "MLACommonPrefillMetadata",
    "_prefill_metadata": "MLACommonPrefillMetadata",
    "chunked_context": "ChunkedContextMetadata",
    "chunk": "ContextChunk",
}
# A hop that descends into a different metadata object re-targets the expected
# class, so a nested read is checked against its own owner: `chunked_context` is
# declared by `MLACommonPrefillMetadata` while its fields belong to
# `ChunkedContextMetadata`, and `chunks` holds `ContextChunk` elements.
_NESTED_RECEIVERS = {
    "chunked_context": "ChunkedContextMetadata",
    "chunks": "ContextChunk",
}
# Reads past these attributes stay inside a different object, not metadata fields.
_OPAQUE_ATTRIBUTES = frozenset({"dcp_manager"})


def _pinned_metadata_fields(source: str) -> dict[str, set[str]]:
    """Field, property and method names declared by each metadata class."""
    declared: dict[str, set[str]] = {name: set() for name in _RECEIVER_TYPES.values()}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef) or node.name not in declared:
            continue
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target, ast.Name
            ):
                declared[node.name].add(statement.target.id)
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                declared[node.name].add(statement.name)
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
    chunked-prefill branch, which no smoke without a cached prefix reaches.
    Check every metadata read against the pinned upstream declarations instead
    of waiting for that branch to run.

    Rebase-time guard: it needs the upstream source that `make sync` / the image
    build places in `third_party/vllm`, so run it after a sync rather than in a
    bare checkout. The repository has no CI workflow, so nothing runs it
    automatically yet.
    """
    assert UPSTREAM_MLA_ATTENTION.exists(), (
        f"pinned upstream vLLM source is missing at {UPSTREAM_MLA_ATTENTION}; run "
        "`make sync` (or the image build) first, otherwise this field-contract "
        "guard cannot check anything"
    )

    declared = _pinned_metadata_fields(UPSTREAM_MLA_ATTENTION.read_text())
    every_field = set().union(*declared.values())
    assert "seq_tot" not in every_field, "the pin still carries the pre-v0.28 layout"
    assert "chunks" in declared["ChunkedContextMetadata"]
    assert "num_context_tokens" in declared["ContextChunk"]

    unknown: list[str] = []
    for lineno, receiver, path in _metadata_reads(MUSA_MLA_COMMON.read_text()):
        expected = _RECEIVER_TYPES[receiver]
        for depth, attribute in enumerate(path):
            if attribute in _OPAQUE_ATTRIBUTES:
                break
            if attribute not in declared[expected]:
                unknown.append(
                    f"common.py:{lineno}: {receiver}.{'.'.join(path[: depth + 1])} is "
                    f"not declared by {expected} at the pin"
                )
                break
            if attribute in _NESTED_RECEIVERS:
                expected = _NESTED_RECEIVERS[attribute]

    assert not unknown, (
        "MUSA MLA reads metadata fields the pinned upstream vLLM no longer "
        "declares; migrate them to the v0.28 `chunks`/`ContextChunk` layout:\n  "
        + "\n  ".join(sorted(unknown))
    )

    # Both the non-DCP and the DCP context path must iterate the v0.28 layout.
    # Counted over the AST so a renamed loop variable or a reformat cannot break
    # the guard on its own.
    source = MUSA_MLA_COMMON.read_text()
    chunk_loops = [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.For, ast.AsyncFor))
        and isinstance(node.iter, ast.Attribute)
        and node.iter.attr == "chunks"
        and isinstance(node.iter.value, ast.Name)
        and node.iter.value.id == "chunked_context"
    ]
    assert len(chunk_loops) == 2, (
        "expected the non-DCP and DCP context paths to iterate "
        f"`chunked_context.chunks`, found {len(chunk_loops)} at lines {chunk_loops}"
    )
    assert "chunk.token_slice" in source
    assert "empty_token_slices" in source


def test_musa_mla_helpers_match_pinned_upstream_signatures() -> None:
    """Guard helper call sites whose upstream signature moved with the pin.

    The same v0.28 rework that reshaped the chunked-context metadata also
    changed `reorg_kvcache` (`local_starts` in, `chunk_size`/`chunk_idx` out),
    which is the DCP path's second, independent break: a stale keyword is a
    TypeError that mid-batch prefix-cache traffic alone reveals.
    """
    assert UPSTREAM_MLA_ATTENTION.exists(), (
        f"pinned upstream vLLM source is missing at {UPSTREAM_MLA_ATTENTION}; run "
        "`make sync` (or the image build) first, otherwise this signature guard "
        "cannot check anything"
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


MUSA_SYNC_TOOL = ROOT / "tools" / "musa_sync.py"
MODULE_DRIFT_DIR = ROOT / "vllm_musa" / "patches" / "module-drift"


def test_mla_drift_tripwire_matches_its_two_sources() -> None:
    """The stored cat-4a tripwire must equal the diff of its two pinned inputs.

    `tools/musa_sync.py verify` marks the row `drifted-copy` whenever the stored
    file differs from `difflib.unified_diff(upstream, shadow)`, and that status
    fails the command. A stale artifact therefore reports the opposite of what
    happened ("upstream changed under the copy") and hides a real relocation of
    the fields this module reads. Recomputed through the tool's own function so
    the guard cannot drift from the tool itself.
    """
    assert UPSTREAM_MLA_ATTENTION.exists(), (
        f"pinned upstream vLLM source is missing at {UPSTREAM_MLA_ATTENTION}; run "
        "`make sync` (or the image build) first, otherwise this artifact guard "
        "cannot check anything"
    )

    spec = importlib.util.spec_from_file_location("_musa_sync_tool", MUSA_SYNC_TOOL)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    shadow = MUSA_MLA_COMMON.relative_to(ROOT).as_posix()
    entries = [entry for entry in tool.manifest.ENTRIES if entry.path == shadow]
    assert len(entries) == 1, f"expected one manifest row for {shadow}, got {len(entries)}"
    entry = entries[0]

    stored = (MODULE_DRIFT_DIR / f"{entry.id}.diff").read_text()
    recomputed = tool._module_tripwire(ROOT / "third_party" / "vllm", entry)
    assert recomputed is not None, f"cannot recompute the {entry.id} tripwire"
    assert stored == recomputed, (
        f"{MODULE_DRIFT_DIR.name}/{entry.id}.diff is stale: it is not the diff of "
        f"{shadow} against its pinned upstream file. Regenerate it after the last "
        "edit to either file."
    )
