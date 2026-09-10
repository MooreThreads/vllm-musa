"""Exercise patched capture entrypoints without importing the GPU runtime.

Set VLLM_SOURCE_DIR to a pinned checkout with the complete series applied.
The actual method bodies are extracted with AST; only their build consumers
and tensor operations are mocked. Hardware graph replay is a separate gate.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

BUILDERS = [
    ("mamba_attn.py", "BaseMambaAttentionMetadataBuilder"),
    ("gdn_attn.py", "GDNAttentionMetadataBuilder"),
    ("linear_attn.py", "BailingLinearAttentionMetadataBuilder"),
]


@pytest.fixture(params=BUILDERS, ids=[entry[1] for entry in BUILDERS])
def capture(request):
    root = Path(os.environ.get("VLLM_SOURCE_DIR", "third_party/vllm"))
    filename, classname = request.param
    path = root / "vllm/v1/attention/backends" / filename
    if not path.exists():
        pytest.skip("Requires a patched pinned vLLM tree via VLLM_SOURCE_DIR")
    module = ast.parse(path.read_text())
    cls = next(
        n for n in module.body if isinstance(n, ast.ClassDef) and n.name == classname
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "build_for_cudagraph_capture"
    )
    # Postponed annotations avoid importing vLLM and Torch just for type names.
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    torch = Mock()
    torch.diff.return_value = Mock(name="reconstructed_accepted")
    torch.diff.return_value.__sub__ = Mock(return_value=Mock())
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
    builder = SimpleNamespace(
        num_spec_tokens=4,
        use_spec_decode=True,
        decode_cudagraph_max_bs=64,
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(mamba_cache_mode="all")
        ),
        build=Mock(return_value="built"),
    )
    metadata = SimpleNamespace(
        max_query_len=5,
        num_reqs=8,
        num_actual_tokens=40,
        query_start_loc=Mock(name="query_start_loc"),
    )
    return classname, namespace[method.name], builder, metadata, torch


def test_preserves_live_metadata_and_accepts_runner_kwargs(capture):
    name, method, builder, metadata, torch = capture
    accepted, drafted, previous = Mock(), Mock(), Mock()
    kwargs = dict(num_accepted_tokens=accepted, num_decode_draft_tokens_cpu=drafted)
    if name.startswith("BaseMamba"):
        kwargs["prev_last_scheduled_idx"] = previous
    assert method(builder, metadata, **kwargs) == "built"
    call = builder.build.call_args
    forwarded_accepted = call.kwargs.get(
        "num_accepted_tokens", call.args[2] if len(call.args) > 2 else None
    )
    forwarded_drafted = call.kwargs.get(
        "num_decode_draft_tokens_cpu", call.args[3] if len(call.args) > 3 else None
    )
    assert forwarded_accepted is accepted
    assert forwarded_drafted is drafted
    if name.startswith("BaseMamba"):
        assert call.kwargs["prev_last_scheduled_idx"] is previous
    torch.diff.assert_not_called()
    torch.zeros.assert_not_called()


def test_capture_only_call_retains_fallback(capture):
    name, method, builder, metadata, torch = capture
    assert method(builder, metadata) == "built"
    torch.diff.assert_called_once_with(metadata.query_start_loc)
    call = builder.build.call_args
    actual = call.kwargs.get(
        "num_accepted_tokens", call.args[2] if len(call.args) > 2 else None
    )
    assert actual is torch.diff.return_value
    if name.startswith("BaseMamba"):
        torch.zeros.assert_called_once()
        assert call.kwargs["prev_last_scheduled_idx"] is torch.zeros.return_value


def test_bailing_validates_query_shape_with_live_counts(capture):
    name, method, builder, metadata, torch = capture
    if name != "BailingLinearAttentionMetadataBuilder":
        pytest.skip("Bailing shape guard")
    metadata.max_query_len = 6
    with pytest.raises(AssertionError, match="query length"):
        method(
            builder,
            metadata,
            num_accepted_tokens=Mock(),
            num_decode_draft_tokens_cpu=Mock(),
        )
    builder.build.assert_not_called()
