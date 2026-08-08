# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_musa.v1.attention.ops import flashmla


class _FakeFlashMLA:
    def __init__(self) -> None:
        self.public_calls = 0

    def flash_mla_sparse_fwd(self, **kwargs):
        self.public_calls += 1
        q = kwargs["q"]
        result = torch.full_like(q, 3)
        aux_shape = (q.shape[0], q.shape[1])
        max_logits = torch.full(aux_shape, 4, dtype=torch.float32)
        lse = torch.full(aux_shape, 5, dtype=torch.float32)
        return result, max_logits, lse


class _FakeAdapter:
    def __init__(
        self,
        *,
        result_idx: tuple[int, ...] = (8, 9, 10),
        launch_error: Exception | None = None,
    ) -> None:
        self.result_idx = list(result_idx)
        self.launch_error = launch_error
        self.calls = 0

    def executable(self, *args):
        self.calls += 1
        if self.launch_error is not None:
            raise self.launch_error
        out, max_logits, lse = args[-3:]
        out.fill_(6)
        max_logits.fill_(7)
        lse.fill_(8)


def _fake_kernel(
    adapter: _FakeAdapter,
    *,
    backend: str = "tvm_ffi",
    param_names: tuple[str, ...] = flashmla._MATE_SPARSE_DIRECT_OUT_PARAM_NAMES,
):
    return SimpleNamespace(
        execution_backend=backend,
        adapter=adapter,
        prim_func=SimpleNamespace(
            params=[SimpleNamespace(name=name) for name in param_names]
        ),
    )


def _inputs(heads: int = 64):
    q = torch.zeros((8, heads, 512), dtype=torch.bfloat16)
    kv = torch.zeros((8, 1, 512), dtype=torch.bfloat16)
    indices = torch.zeros((8, 1, 64), dtype=torch.int32)
    topk_length = torch.full((8,), 64, dtype=torch.int32)
    attn_sink = torch.zeros((heads,), dtype=torch.float32)
    out = torch.empty_like(q)
    return q, kv, indices, topk_length, attn_sink, out


def _direct_out_patches(
    adapter: _FakeAdapter,
    *,
    kernel_backend: str = "tvm_ffi",
    param_names: tuple[str, ...] = flashmla._MATE_SPARSE_DIRECT_OUT_PARAM_NAMES,
    raise_complete_if_dry_run=lambda: None,
    is_musa_tensor=lambda tensor: True,
    is_capturing=lambda: False,
):
    kernel = _fake_kernel(
        adapter,
        backend=kernel_backend,
        param_names=param_names,
    )
    return patch.multiple(
        flashmla,
        _MATE_SPARSE_DIRECT_OUT_IMPORT_ERROR=None,
        _MATE_SPARSE_DIRECT_OUT_ABI_SUPPORTED=True,
        _mate_raise_complete_if_dry_run=raise_complete_if_dry_run,
        _mate_resolve_num_mps=lambda device, requested: 1,
        _mate_sparse_model1_kernel=lambda *args, **kwargs: kernel,
        _mate_optional_prefill_attn_sink=lambda sink, heads, device: (
            torch.empty((0,), dtype=torch.float32),
            sink is not None,
        ),
        _mate_require_token_lengths=(
            lambda lengths, seq_len, topk, device, name: lengths
        ),
        _is_musa_tensor=is_musa_tensor,
        _is_current_stream_capturing=is_capturing,
    )


def _call(
    fake: _FakeFlashMLA,
    *,
    allow_direct_out: bool,
    heads: int = 64,
    out_alias_q: bool = False,
):
    q, kv, indices, topk_length, attn_sink, out = _inputs(heads)
    if out_alias_q:
        out = q
    with patch.object(flashmla, "_flash_mla", fake):
        result = flashmla.flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            1.0,
            attn_sink=attn_sink,
            topk_length=topk_length,
            out=out,
            allow_dsv4_tp8_mtp_direct_out=allow_direct_out,
        )
    return result, out


def test_dsv4_tp8_sparse_prefill_writes_directly_to_out() -> None:
    fake = _FakeFlashMLA()
    adapter = _FakeAdapter()

    with _direct_out_patches(adapter):
        (result, max_logits, lse), out = _call(fake, allow_direct_out=True)

    assert fake.public_calls == 0
    assert adapter.calls == 1
    assert result is out
    assert torch.all(out == 6)
    assert torch.all(max_logits == 7)
    assert torch.all(lse == 8)


@pytest.mark.parametrize(
    ("allow_direct_out", "heads"),
    [(False, 64), (True, 8)],
)
def test_sparse_prefill_preserves_public_fallback_without_exact_owner_and_shape(
    allow_direct_out: bool,
    heads: int,
) -> None:
    fake = _FakeFlashMLA()
    adapter = _FakeAdapter()

    with _direct_out_patches(adapter):
        (result, max_logits, lse), out = _call(
            fake,
            allow_direct_out=allow_direct_out,
            heads=heads,
        )

    assert fake.public_calls == 1
    assert adapter.calls == 0
    assert result is out
    assert torch.all(out == 3)
    assert torch.all(max_logits == 4)
    assert torch.all(lse == 5)


def test_sparse_prefill_preserves_public_fallback_on_non_musa_device() -> None:
    fake = _FakeFlashMLA()
    adapter = _FakeAdapter()

    with _direct_out_patches(
        adapter,
        is_musa_tensor=flashmla._is_musa_tensor,
    ):
        _call(fake, allow_direct_out=True)

    assert fake.public_calls == 1
    assert adapter.calls == 0


def test_sparse_prefill_preserves_public_fallback_during_graph_capture() -> None:
    fake = _FakeFlashMLA()
    adapter = _FakeAdapter()

    with _direct_out_patches(adapter, is_capturing=lambda: True):
        _call(fake, allow_direct_out=True)

    assert fake.public_calls == 1
    assert adapter.calls == 0


def test_sparse_prefill_preserves_public_fallback_when_out_aliases_input() -> None:
    fake = _FakeFlashMLA()
    adapter = _FakeAdapter()

    with _direct_out_patches(adapter):
        _call(fake, allow_direct_out=True, out_alias_q=True)

    assert fake.public_calls == 1
    assert adapter.calls == 0


@pytest.mark.parametrize(
    ("kernel_backend", "param_names", "result_idx"),
    [
        ("python", flashmla._MATE_SPARSE_DIRECT_OUT_PARAM_NAMES, (8, 9, 10)),
        ("tvm_ffi", ("wrong",), (8, 9, 10)),
        ("tvm_ffi", flashmla._MATE_SPARSE_DIRECT_OUT_PARAM_NAMES, (3, 4, 5)),
    ],
)
def test_sparse_prefill_preserves_public_fallback_on_private_abi_mismatch(
    kernel_backend: str,
    param_names: tuple[str, ...],
    result_idx: tuple[int, ...],
) -> None:
    fake = _FakeFlashMLA()
    adapter = _FakeAdapter(result_idx=result_idx)

    with _direct_out_patches(
        adapter,
        kernel_backend=kernel_backend,
        param_names=param_names,
    ):
        _call(fake, allow_direct_out=True)

    assert fake.public_calls == 1
    assert adapter.calls == 0


def test_sparse_prefill_dry_run_completes_before_private_launch() -> None:
    class _DryRunComplete(RuntimeError):
        pass

    fake = _FakeFlashMLA()
    adapter = _FakeAdapter()

    def _raise_dry_run() -> None:
        raise _DryRunComplete

    with _direct_out_patches(
        adapter,
        raise_complete_if_dry_run=_raise_dry_run,
    ):
        with pytest.raises(_DryRunComplete):
            _call(fake, allow_direct_out=True)

    assert fake.public_calls == 0
    assert adapter.calls == 0


def test_sparse_prefill_does_not_retry_after_private_launch_failure() -> None:
    fake = _FakeFlashMLA()
    adapter = _FakeAdapter(launch_error=RuntimeError("launch failed"))

    with _direct_out_patches(adapter):
        with pytest.raises(RuntimeError, match="launch failed"):
            _call(fake, allow_direct_out=True)

    assert fake.public_calls == 0
    assert adapter.calls == 1
