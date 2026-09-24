# SPDX-License-Identifier: Apache-2.0
"""Source contract for GLM MATE sparse-prefill model detection."""

import ast
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "vllm_musa/v1/attention/backends/mla/flashmla_sparse.py"
)


def test_glm_detection_is_folded_into_mate_predicate() -> None:
    source = SOURCE.read_text()
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert "_is_glm_dsa_model" not in functions
    assert "_can_use_glm_mate_sparse_prefill" in functions


def test_glm_detection_checks_nested_hf_configs() -> None:
    source = SOURCE.read_text()
    assert "def _can_use_glm_mate_sparse_prefill(" in source
    assert "is_glm_dsa: bool" in source


def test_mate_dispatch_precedes_generic_tilelang_dispatch() -> None:
    source = SOURCE.read_text()
    mate = source.index("if _can_use_glm_mate_sparse_prefill(")
    generic = source.index("elif _can_use_tilelang_sparse_prefill(", mate)
    assert mate < generic


def test_musa_impl_caches_config_before_runtime_dispatch() -> None:
    source = SOURCE.read_text()
    assert "def _musa_backend_sparse_fwd(" in source
    assert "is_glm_dsa: bool = False" in source
