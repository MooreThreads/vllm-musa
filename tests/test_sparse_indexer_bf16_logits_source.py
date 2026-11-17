import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _indexer_source(filename: str) -> ast.Module:
    source = ROOT / "third_party/vllm/vllm/model_executor/layers" / filename
    if not source.is_file():
        pytest.skip("requires the patched third_party/vllm source tree")
    return ast.parse(source.read_text())


@pytest.mark.parametrize("platform", ["musa", "cuda", "rocm", "cpu"])
def test_topk_initialization_does_not_require_musa_api_on_in_tree_platforms(
    platform: str,
) -> None:
    tree = _indexer_source("indexer_topk.py")
    selector = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SparseIndexerTopk"
    )
    initializer = next(
        node
        for node in selector.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assignment = next(
        node
        for node in initializer.body
        if isinstance(node, ast.Assign)
        and any(getattr(target, "attr", None) == "_is_musa" for target in node.targets)
    )
    is_musa = platform == "musa"
    current_platform = SimpleNamespace(is_out_of_tree=lambda: is_musa)
    if is_musa:
        current_platform.is_musa = lambda: True
    result = eval(
        compile(ast.Expression(assignment.value), "indexer_topk.py", "eval"),
        {"current_platform": current_platform},
    )
    assert result is is_musa


def test_decode_reference_logits_keep_the_deepselect_layout_fallback() -> None:
    function = next(
        node
        for node in _indexer_source("sparse_attn_indexer.py").body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_musa_fill_decode_topk_from_indexer_cache_capture"
    )
    branch = next(
        node
        for node in function.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "topk_backend == 'deep_select'"
    )
    selector = Mock()
    lengths = Mock()
    lengths.__getitem__ = Mock(return_value=lengths)
    lengths.to.return_value = lengths
    output = Mock()
    output.__getitem__ = Mock(return_value=output)
    logits = object()
    namespace = {
        "torch": SimpleNamespace(int32="int32"),
        "logits": logits,
        "seq_lens": lengths,
        "topk_indices_buffer": output,
        "rows": 2,
        "topk": 512,
        "topk_backend": "deep_select",
        "_musa_select_indexer_topk": selector,
        "deep_select_topk": Mock(side_effect=AssertionError("unguarded DeepSelect")),
    }
    body = [node for node in branch.body if not isinstance(node, ast.Return)]
    exec(
        compile(ast.Module(body=body, type_ignores=[]), "indexer.py", "exec"), namespace
    )
    selector.assert_called_once_with(logits, lengths, output, 512, "deep_select")


@pytest.mark.parametrize(
    "relative",
    [
        "csrc/musa/attention/deepseek_v4_sparse_indexer_topk.mu",
        "csrc/musa/attention/glm52_indexer_topk.mu",
    ],
)
def test_native_topk_supports_bf16_without_changing_fp32_dispatch(relative):
    source = (ROOT / relative).read_text()

    assert "#include <musa_bf16.h>" in source
    assert "template <typename LogitT, typename OutT>" in source
    assert "const LogitT *row_logits" in source
    assert "torch::kFloat32" in source
    assert "torch::kBFloat16" in source
    for input_type in ("float", "__mt_bfloat16"):
        for output_type in ("int32_t", "int64_t"):
            assert f"<{input_type}, {output_type}>" in source


def test_mate_mqa_bf16_patch_preserves_legacy_provider_and_avoids_fp32_cast():
    patches = list(
        (ROOT / "vllm_musa/patches/series").glob("*-Return-BF16-MQA-logits-*.patch")
    )
    assert len(patches) == 1
    added = "\n".join(
        line[1:]
        for line in patches[0].read_text().splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )

    assert 'provider_name == "mate.deep_gemm"' in added
    assert "_MUSA_INDEXER_LOGITS_DTYPE = torch.bfloat16" in added
    assert '"logits_dtype": _MUSA_INDEXER_LOGITS_DTYPE' in added
    assert "logits_dtype=_MUSA_INDEXER_LOGITS_DTYPE" in added
    assert "_MUSA_INDEXER_LOGITS_DTYPE.itemsize" in added
    assert "else {}" in added
    assert "torch.float32" not in added
