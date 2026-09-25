# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-contract checks for the MUSA fused allreduce + RMSNorm rewrite.

``fx.Graph.call_function`` leaves ``node.meta`` empty. A node inserted by the
rewrite that later reaches a ``VllmIRLoweringPass`` match raises
``KeyError: 'val'``, because that pass indexes ``arg.meta["val"]`` unguarded.
The rewrite therefore has to publish the metadata, and both of its arms have to
go through the same helper so the invariant cannot drift between them.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "vllm_musa" / "_inductor" / "musa_allreduce_rms_fusion.py"

REWRITER = "_manual_rewrite_residual_musa_jit_car_rmsnorm"
INSERT_HELPER = "_insert_fused_ar_rmsnorm"


def _tree() -> ast.Module:
    return ast.parse(MODULE.read_text())


def _find(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is missing from {MODULE.name}")


def _enclosing_function_owners(tree: ast.Module) -> dict[int, str]:
    """Map each line number to the name of its innermost enclosing function."""
    owners: dict[int, str] = {}
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        for node in ast.walk(func):
            lineno = getattr(node, "lineno", None)
            if lineno is not None:
                owners[lineno] = func.name
    return owners


def _call_function_sites(tree: ast.Module) -> list[tuple[int, str]]:
    owners = _enclosing_function_owners(tree)
    sites: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "call_function":
            sites.append((node.lineno, owners.get(node.lineno, "<module>")))
    return sites


def test_every_inserted_node_comes_from_the_meta_publishing_helper():
    sites = _call_function_sites(_tree())
    assert sites, "expected the rewrite to insert nodes with graph.call_function"

    outside = sorted({owner for _, owner in sites if owner != INSERT_HELPER})
    assert not outside, (
        "graph.call_function outside "
        f"{INSERT_HELPER} would insert nodes with empty meta, which a later "
        f"VllmIRLoweringPass turns into KeyError: 'val'; found in: {outside}"
    )


def test_both_rewrite_arms_go_through_the_helper():
    rewriter = _find(_tree(), REWRITER)

    calls = [
        node
        for node in ast.walk(rewriter)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == INSERT_HELPER
    ]
    assert len(calls) == 2, (
        f"expected one {INSERT_HELPER} call per rewrite arm (fused_add_rms_norm "
        f"and the decomposed add -> rms_norm), found {len(calls)}"
    )


def test_helper_publishes_meta_for_every_node_it_inserts():
    source = MODULE.read_text()
    body = ast.get_source_segment(source, _find(_tree(), INSERT_HELPER))
    assert body is not None

    for node_name in ("fused", "fused_rms", "fused_residual"):
        assert f'{node_name}.meta["val"] =' in body, (
            f"{node_name} is returned to the graph without meta['val']"
        )
    assert 'fused_raw.meta["val"] =' in body, (
        "the use_raw getitem is handed to raw users without meta['val']"
    )


def test_helper_derives_meta_from_fakes_and_never_from_real_tensors():
    source = MODULE.read_text()
    body = ast.get_source_segment(source, _find(_tree(), INSERT_HELPER))
    assert body is not None

    # Deriving metadata from a real tensor would execute an actual fused
    # all-reduce during compilation, so the helper must require fakes.
    assert "isinstance(fake, FakeTensor)" in body
    assert "raise AssertionError" in body
    # A silent skip would leave exactly the empty-meta state this guards.
    assert "cannot derive metadata; leave the graph as it was" not in body
    # Inputs are fakes owned by the ambient mode; nesting a mode raises
    # "Mixing fake modes NYI".
    assert "FakeTensorMode()" not in body


def test_the_silent_module_level_helper_is_gone():
    source = MODULE.read_text()
    assert "_fill_fused_meta" not in source, (
        "the module-level helper skipped publishing on missing inputs while "
        "its caller rewrote the graph anyway; the precondition now lives in "
        f"{INSERT_HELPER}"
    )
