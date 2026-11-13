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


def _function(path, name):
    return next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def test_mate_dispatch_precedes_generic_tilelang_dispatch() -> None:
    source = ast.unparse(_function(SOURCE, "_musa_backend_sparse_fwd"))
    mate = source.index("if _can_use_glm_mate_sparse_prefill(")
    generic = source.index("if _can_use_tilelang_sparse_prefill(", mate)
    assert mate < generic
    assert "_musa_flash_mla_sparse_fwd" in source


def test_mate_provider_tuple_and_backend_aux_contract(monkeypatch) -> None:
    import sys
    from types import ModuleType, SimpleNamespace
    from unittest.mock import Mock

    import torch

    adapter = SOURCE.parents[2] / "ops/sparse_mla_mate.py"
    node = _function(adapter, "sparse_mla_fwd_bf16")
    output = torch.arange(12).reshape(2, 3, 2)
    max_logits = torch.zeros(2, 3)
    lse = torch.arange(6).reshape(2, 3)
    result = (output, max_logits, lse)
    provider = Mock(return_value=result)
    namespace = {"sparse_mla_prefill_fwd": provider}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(adapter), "exec"), namespace)
    wrapper = namespace["sparse_mla_fwd_bf16"]
    q, kv, indices, topk = (object() for _ in range(4))
    assert wrapper(q, kv, indices, 0.5, d_v=2, topk_length=topk) is result
    provider.assert_called_once_with(
        q=q, kv=kv, indices=indices, sm_scale=0.5, d_v=2, topk_length=topk
    )

    module_name = "vllm_musa.v1.attention.ops.sparse_mla_mate"
    module = ModuleType(module_name)
    module.sparse_mla_fwd_bf16 = wrapper
    monkeypatch.setitem(sys.modules, module_name, module)
    node = _function(SOURCE, "_musa_backend_sparse_fwd")
    namespace = {
        "torch": torch,
        "logger": SimpleNamespace(info_once=lambda *_: None),
        "_can_use_glm_mate_sparse_prefill": lambda *_: True,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    backend = namespace["_musa_backend_sparse_fwd"]
    for out in (None, torch.empty_like(output)):
        actual, actual_max, actual_lse = backend(
            q, kv, indices, 0.5, d_v=2, topk_length=topk, out=out, is_glm_dsa=True
        )
        assert actual is (output if out is None else out)
        assert torch.equal(actual, output)
        assert actual_max is max_logits
        assert actual_lse is lse
        assert actual_lse.ndim == 2
        assert torch.equal(actual_lse[:, :2], lse[:, :2])

    namespace["_can_use_glm_mate_sparse_prefill"] = lambda *_: False
    namespace["_can_use_tilelang_sparse_prefill"] = lambda *_: True
    module_name = "vllm_musa.v1.attention.ops.sparse_mla_tilelang"
    module = ModuleType(module_name)
    module.sparse_mla_fwd_bf16 = Mock(return_value=output)
    monkeypatch.setitem(sys.modules, module_name, module)
    for out in (None, torch.empty_like(output)):
        actual, actual_max, actual_lse = backend(
            output, kv, indices, 0.5, d_v=2, topk_length=topk, out=out
        )
        assert actual is (output if out is None else out)
        assert torch.equal(actual, output)
        assert actual_max is actual_lse
        assert actual_max.shape == (0,)
        assert actual_max.dtype == output.dtype
        assert actual_max.device == output.device


def test_musa_impl_caches_config_before_runtime_dispatch() -> None:
    source = SOURCE.read_text()
    assert "def _musa_backend_sparse_fwd(" in source
    assert "is_glm_dsa: bool = False" in source
