from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
logits_processor = pytest.importorskip("vllm.model_executor.layers.logits_processor")
custom_ar = pytest.importorskip(
    "vllm_musa.distributed.device_communicators.musa_jit_custom_all_reduce"
)


def _processor(org_vocab_size: int):
    processor = object.__new__(logits_processor.LogitsProcessor)
    processor.org_vocab_size = org_vocab_size
    processor.scale = 1.0
    processor.soft_cap = None
    processor.use_all_gather = True
    return processor


def _patch_tp(monkeypatch, world_size: int = 2, rank: int = 0):
    monkeypatch.setattr(
        logits_processor,
        "get_tensor_model_parallel_world_size",
        lambda: world_size,
    )
    monkeypatch.setattr(
        logits_processor, "get_tensor_model_parallel_rank", lambda: rank
    )


def test_tp1_logits_skip_collectives(monkeypatch):
    _patch_tp(monkeypatch, world_size=1)
    local_logits = torch.arange(32, dtype=torch.int32).to(torch.bfloat16).view(1, 32)
    monkeypatch.setattr(
        custom_ar,
        "maybe_musa_jit_logits_all_gather",
        lambda input_tensor, dim: pytest.fail("TP1 must skip IPC gather"),
    )

    output = _processor(151936)._musa_all_reduce_logits(local_logits, SimpleNamespace())

    assert output is local_logits


@pytest.mark.parametrize(
    "org_vocab_size,shape,dtype",
    [
        (151936, (1, 75968), torch.bfloat16),
        (248320, (1, 124160), torch.bfloat16),
        (152064, (8, 76032), torch.bfloat16),
        (129280, (16, 64640), torch.bfloat16),
        (102400, (4, 51200), torch.float16),
        (32000, (4, 16000), torch.float32),
        (248320, (128, 124160), torch.bfloat16),
    ],
)
def test_supported_tp_logits_use_ipc_gather(monkeypatch, org_vocab_size, shape, dtype):
    _patch_tp(monkeypatch)
    local_logits = torch.zeros(shape, dtype=dtype)
    gathered = torch.empty((shape[0], shape[1] * 2), dtype=dtype)
    calls = []

    def gather(input_tensor, dim, output_dtype=None):
        calls.append((input_tensor, dim, output_dtype))
        return gathered

    monkeypatch.setattr(custom_ar, "maybe_musa_jit_logits_all_gather", gather)
    monkeypatch.setattr(
        logits_processor,
        "tensor_model_parallel_all_reduce",
        lambda input_tensor: pytest.fail("all-reduce fallback must not run"),
    )

    output = _processor(org_vocab_size)._musa_all_reduce_logits(
        local_logits, SimpleNamespace()
    )
    assert output is gathered
    assert len(calls) == 1
    assert calls[0][0] is local_logits
    assert calls[0][1] == -1
    assert calls[0][2] is None


@pytest.mark.parametrize(
    "world_size,shard_size",
    [(4, 37984), (8, 18992)],
)
def test_standard_qwen_logits_request_float32_ipc_output(
    monkeypatch, world_size, shard_size
):
    _patch_tp(monkeypatch, world_size=world_size)
    local_logits = torch.empty((1, shard_size), dtype=torch.bfloat16)
    gathered = torch.empty((1, 151936), dtype=torch.float32)
    calls = []

    def gather(input_tensor, dim, output_dtype=None):
        calls.append((input_tensor, dim, output_dtype))
        return gathered

    monkeypatch.setattr(custom_ar, "maybe_musa_jit_logits_all_gather", gather)
    processor = _processor(151936)
    monkeypatch.setattr(processor, "_use_musa_logits_all_reduce", lambda: True)

    output = processor._gather_logits(local_logits, SimpleNamespace())

    assert output is gathered
    assert calls == [(local_logits, -1, torch.float32)]


def test_tp2_standard_qwen_logits_keep_same_dtype_ipc_output(monkeypatch):
    _patch_tp(monkeypatch, world_size=2)
    local_logits = torch.empty((1, 75968), dtype=torch.bfloat16)
    gathered = torch.empty((1, 151936), dtype=torch.bfloat16)
    output_dtypes = []

    def gather(input_tensor, dim, output_dtype=None):
        output_dtypes.append(output_dtype)
        return gathered

    monkeypatch.setattr(custom_ar, "maybe_musa_jit_logits_all_gather", gather)
    processor = _processor(151936)
    monkeypatch.setattr(processor, "_use_musa_logits_all_reduce", lambda: True)

    output = processor._gather_logits(local_logits, SimpleNamespace())

    assert output is gathered
    assert output_dtypes == [None]


@pytest.mark.parametrize("scale,soft_cap", [(2.0, None), (1.0, 30.0)])
def test_transformed_qwen_logits_keep_same_dtype_ipc_output(
    monkeypatch, scale, soft_cap
):
    _patch_tp(monkeypatch, world_size=4)
    local_logits = torch.empty((1, 37984), dtype=torch.bfloat16)
    gathered = torch.empty((1, 151936), dtype=torch.bfloat16)
    output_dtypes = []

    def gather(input_tensor, dim, output_dtype=None):
        output_dtypes.append(output_dtype)
        return gathered

    monkeypatch.setattr(custom_ar, "maybe_musa_jit_logits_all_gather", gather)
    processor = _processor(151936)
    processor.scale = scale
    processor.soft_cap = soft_cap
    monkeypatch.setattr(processor, "_use_musa_logits_all_reduce", lambda: True)

    output = processor._gather_logits(local_logits, SimpleNamespace())

    assert output is gathered
    assert output_dtypes == [None]


@pytest.mark.parametrize(
    "org_vocab_size,rows,shard_size,dtype",
    [
        (152064, 4, 38016, torch.bfloat16),
        (129280, 4, 32320, torch.bfloat16),
        (248320, 65, 62080, torch.bfloat16),
        (151936, 4, 37984, torch.float16),
        (151936, 4, 37984, torch.float32),
    ],
)
def test_non_small_qwen_bf16_logits_keep_same_dtype_ipc_output(
    monkeypatch, org_vocab_size, rows, shard_size, dtype
):
    _patch_tp(monkeypatch, world_size=4)
    local_logits = torch.empty((rows, shard_size), dtype=dtype)
    gathered = torch.empty((rows, shard_size * 4), dtype=dtype)
    output_dtypes = []

    def gather(input_tensor, dim, output_dtype=None):
        output_dtypes.append(output_dtype)
        return gathered

    monkeypatch.setattr(custom_ar, "maybe_musa_jit_logits_all_gather", gather)
    processor = _processor(org_vocab_size)
    monkeypatch.setattr(processor, "_use_musa_logits_all_reduce", lambda: True)

    output = processor._gather_logits(local_logits, SimpleNamespace())

    assert output is gathered
    assert output_dtypes == [None]


@pytest.mark.parametrize(
    "org_vocab_size,shape,dtype",
    [
        (32000, (4, 15996), torch.bfloat16),
        (32000, (4, 15998), torch.float32),
        (248320, (4, 124160), torch.float64),
        (248320, (2, 2, 124160), torch.bfloat16),
    ],
)
def test_unsupported_logits_preserve_all_reduce_fallback(
    monkeypatch, org_vocab_size, shape, dtype
):
    _patch_tp(monkeypatch)
    numel = 1
    for dim_size in shape:
        numel *= dim_size
    local_logits = torch.arange(numel, dtype=torch.int32).to(dtype).view(shape)
    fallback_inputs = []

    monkeypatch.setattr(
        custom_ar,
        "maybe_musa_jit_logits_all_gather",
        lambda input_tensor, dim, output_dtype=None: pytest.fail(
            "IPC gather gate must not run"
        ),
    )

    def all_reduce(input_tensor):
        fallback_inputs.append(input_tensor.clone())
        return input_tensor

    monkeypatch.setattr(
        logits_processor, "tensor_model_parallel_all_reduce", all_reduce
    )
    output = _processor(org_vocab_size)._musa_all_reduce_logits(
        local_logits, SimpleNamespace()
    )

    assert output.shape == (*shape[:-1], shape[-1] * 2)
    assert torch.equal(output[..., : shape[-1]], local_logits)
    assert torch.count_nonzero(output[..., shape[-1] :]).item() == 0
    assert len(fallback_inputs) == 1


def test_ipc_gather_declining_falls_back_to_all_reduce(monkeypatch):
    _patch_tp(monkeypatch, world_size=2, rank=1)
    local_logits = torch.arange(64, dtype=torch.int32).to(torch.bfloat16).view(2, 32)
    monkeypatch.setattr(
        custom_ar,
        "maybe_musa_jit_logits_all_gather",
        lambda input_tensor, dim, output_dtype=None: None,
    )
    monkeypatch.setattr(
        logits_processor,
        "tensor_model_parallel_all_reduce",
        lambda input_tensor: input_tensor,
    )

    output = _processor(64)._musa_all_reduce_logits(local_logits, SimpleNamespace())

    assert output.shape == (2, 64)
    assert torch.equal(output[:, 32:64], local_logits)


def test_tp6_logits_preserve_all_reduce_fallback(monkeypatch):
    _patch_tp(monkeypatch, world_size=6, rank=3)
    local_logits = torch.arange(32, dtype=torch.int32).to(torch.bfloat16).view(1, 32)
    monkeypatch.setattr(
        custom_ar,
        "maybe_musa_jit_logits_all_gather",
        lambda input_tensor, dim: pytest.fail("TP6 IPC gather must stay disabled"),
    )
    monkeypatch.setattr(
        logits_processor,
        "tensor_model_parallel_all_reduce",
        lambda input_tensor: input_tensor,
    )

    output = _processor(248320)._musa_all_reduce_logits(local_logits, SimpleNamespace())

    assert output.shape == (1, 32 * 6)
    assert torch.equal(output[:, 32 * 3 : 32 * 4], local_logits)
