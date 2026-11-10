# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "vllm_musa" / "patches" / "series"
PATCH = SERIES / "0193-MUSA-let-adaptive-verification-trim-the-flattened-DS.patch"
CAPTURE_PATCH = (
    SERIES / "0194-MUSA-profile-adaptive-verification-costs-in-the-real.patch"
)

HEADS, HEAD_DIM, BLOCK_SIZE, TOPK = 32, 128, 64, 512
BLOCKS_PER_REQ = 32
MAX_TOKENS = 64


def _has_musa_device() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return hasattr(torch, "musa") and torch.musa.is_available()


def test_series_declares_the_flattened_indexer_trimmable_on_musa() -> None:
    text = PATCH.read_text()
    hook = text[text.index("def _supports_flattened_device_query_lens") :]
    hook = hook[: hook.index("current_platform.is_cuda()")]
    assert "+    if current_platform.is_musa():" in hook
    assert "+        return True" in hook


def test_series_profiles_adaptive_costs_only_in_the_real_capture() -> None:
    text = CAPTURE_PATCH.read_text()
    assert "-                    if self.adaptive_verification is not None:\n" in text
    assert (
        "+                    if self.adaptive_verification is not None and not profile_only:\n"
        in text
    )


@pytest.mark.parametrize("profile_only", [True, False])
def test_capture_model_profiles_adaptive_costs_only_in_the_real_capture(
    monkeypatch: pytest.MonkeyPatch, profile_only: bool
) -> None:
    from contextlib import contextmanager, nullcontext

    torch = pytest.importorskip("torch")
    # The MUSA plugin loads before the V2 runner, as in a server process.
    pytest.importorskip("vllm_musa.v1")
    model_runner = pytest.importorskip("vllm.v1.worker.gpu.model_runner")

    monkeypatch.setattr(model_runner, "freeze_gc_for_cudagraph_capture", nullcontext)
    monkeypatch.setattr(model_runner, "lock_workspace", lambda: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda *_: (0, 0))

    calls: list[tuple[str, object]] = []

    class StepTiming:
        @contextmanager
        def collect(self):
            yield ["sample"]

    class Manager:
        def batches_to_profile(self, capture_sizes):
            calls.append(("batches", list(capture_sizes)))
            yield {"num_tokens": 2}

        def set_initial_cost_curves(self, samples):
            calls.append(("curves", samples))

    runner = object.__new__(model_runner.GPUModelRunner)
    runner.cudagraph_manager = SimpleNamespace(
        needs_capture=lambda: True,
        capture=lambda *args, **kwargs: None,
        captured_token_counts=lambda: [1, 2],
    )
    runner.model_state = SimpleNamespace(
        supports_mm_inputs=False, capture_inner_cudagraphs=lambda *args: None
    )
    runner.input_buffers = runner.model = runner.intermediate_tensors = None
    runner.block_tables = runner.kv_cache_config = runner.lora_config = None
    runner.pcp_manager = runner.speculator = None
    runner.attn_groups = []
    runner.use_aux_hidden_state_outputs = False
    runner.kv_connector = SimpleNamespace(reset_capture_state=lambda: None)
    runner.adaptive_verification = Manager()
    runner.step_timing = StepTiming()
    runner._dummy_run = lambda **batch: calls.append(("dummy", batch))

    runner.capture_model(profile_only=profile_only)

    if profile_only:
        assert calls == []
    else:
        assert calls == [
            ("batches", [1, 2]),
            ("dummy", {"num_tokens": 2}),
            ("curves", ["sample"]),
        ]


@pytest.mark.skipif(not _has_musa_device(), reason="requires a MUSA device")
def test_v41_indexer_accepts_device_decided_query_lens_on_musa() -> None:
    from vllm.v1.attention.backends.mla import indexer

    assert indexer.DeepseekV41IndexerBackend.supports_device_cpu_query_lens_mismatch()
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(enable_adaptive_verification=True),
        num_speculative_tokens=1,
    )
    # next_n=2 decodes natively on MUSA unless adaptive verification is on.
    assert indexer._use_flattening(config)


def _builder(adaptive: bool):
    import torch
    from vllm.v1.attention.backends.mla import indexer

    builder = object.__new__(indexer.DeepseekV32IndexerMetadataBuilder)
    builder.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(enable_adaptive_verification=adaptive)
    )
    builder.supports_varlen = False
    builder.decode_seq_lens_buffer = torch.zeros(
        MAX_TOKENS, dtype=torch.int32, device="musa"
    )
    builder.decode_lens_buffer = torch.zeros_like(builder.decode_seq_lens_buffer)
    builder.arange_buffer = torch.arange(MAX_TOKENS, dtype=torch.int32, device="musa")
    builder.expanded_block_table_buffer = torch.zeros(
        MAX_TOKENS, BLOCKS_PER_REQ, dtype=torch.int32, device="musa"
    )
    return builder


def _decode_metadata(builder, device_lens, cpu_lens, seq_lens, block_table):
    """Flattened indexer decode rows; the CPU only knows the total of the lengths."""
    import torch

    assert sum(device_lens) == sum(cpu_lens)
    starts = [0, *torch.tensor(device_lens).cumsum(0).tolist()]
    query_start_loc = torch.tensor(starts, dtype=torch.int32, device="musa")
    decode_lens_cpu = torch.tensor(cpu_lens, dtype=torch.int32)
    rows, table, _, _, _ = builder._prepare_decode_tensors(
        seq_lens=seq_lens,
        block_table=block_table,
        decode_lens=torch.diff(query_start_loc),
        decode_lens_cpu=decode_lens_cpu,
        query_start_loc=query_start_loc[:-1],
        num_decodes=len(device_lens),
        num_decode_tokens=sum(device_lens),
        use_native=False,
        next_n=6,
        max_decode_len=int(decode_lens_cpu.max()),
    )
    return SimpleNamespace(seq_lens=rows, block_table=table, schedule_metadata=None)


def _indexer_inputs(num_tokens: int):
    import torch

    num_reqs = 4
    num_blocks = num_reqs * BLOCKS_PER_REQ
    fp8 = torch.float8_e4m3fn
    keys = torch.randn(num_blocks, BLOCK_SIZE, HEAD_DIM, device="musa").to(fp8)
    scales = torch.rand(num_blocks, BLOCK_SIZE, device="musa") + 0.5
    row_bytes = BLOCK_SIZE * (HEAD_DIM + 4)
    kv = torch.empty(num_blocks, row_bytes, dtype=torch.uint8, device="musa")
    kv[:, : BLOCK_SIZE * HEAD_DIM] = keys.view(torch.uint8).reshape(num_blocks, -1)
    kv[:, BLOCK_SIZE * HEAD_DIM :] = scales.view(torch.uint8).reshape(num_blocks, -1)
    q = torch.randn(num_tokens, HEADS, HEAD_DIM, device="musa").to(fp8)
    weights = torch.randn(num_tokens, HEADS, device="musa")
    blocks = torch.arange(num_blocks, dtype=torch.int32, device="musa")
    block_table = blocks.view(num_reqs, -1)
    seq_lens = torch.tensor([1500, 900, 2047, 600], dtype=torch.int32, device="musa")
    kv = kv.view(num_blocks, BLOCK_SIZE, HEAD_DIM + 4)
    return kv, q, weights, block_table, seq_lens


def _select_topk(q, kv, weights, metadata, out) -> None:
    from vllm.model_executor.layers import sparse_attn_indexer as sai

    out.fill_(-1)
    assert sai._musa_try_fill_decode_topk_from_indexer_cache_native(
        q, kv, weights, metadata, out, TOPK, HEAD_DIM, 8192
    )


@pytest.mark.skipif(not _has_musa_device(), reason="requires a MUSA device")
def test_trimmed_verify_batch_selects_the_exact_indexer_topk() -> None:
    import torch

    pytest.importorskip("torchada")
    torch.manual_seed(0)
    trimmed_a, trimmed_b, even = [6, 1, 3, 2], [2, 3, 1, 6], [3, 3, 3, 3]
    num_tokens = sum(even)
    kv, q, weights, block_table, seq_lens = _indexer_inputs(num_tokens)

    def reference(device_lens):
        meta = _decode_metadata(
            _builder(False), device_lens, device_lens, seq_lens, block_table
        )
        out = torch.empty(num_tokens, TOPK, dtype=torch.int32, device="musa")
        _select_topk(q, kv, weights, meta, out)
        return meta, out

    # Eager: device-decided lengths with an evenly split CPU copy.
    exact_meta, exact_topk = reference(trimmed_a)
    meta = _decode_metadata(_builder(True), trimmed_a, even, seq_lens, block_table)
    assert torch.equal(meta.seq_lens, exact_meta.seq_lens)
    assert torch.equal(meta.block_table, exact_meta.block_table)
    topk = torch.empty_like(exact_topk)
    _select_topk(q, kv, weights, meta, topk)
    assert torch.equal(topk, exact_topk)
    assert (topk >= 0).sum(dim=1).min() > 0

    # Without the adaptive branch the even CPU copy takes the uniform shortcut.
    wrong = _decode_metadata(_builder(False), trimmed_a, even, seq_lens, block_table)
    assert not torch.equal(wrong.seq_lens, exact_meta.seq_lens)

    # Graph replay: metadata is rebuilt eagerly into the captured buffers.
    builder = _builder(True)
    meta = _decode_metadata(builder, trimmed_a, even, seq_lens, block_table)
    out = torch.empty_like(exact_topk)
    _select_topk(q, kv, weights, meta, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _select_topk(q, kv, weights, meta, out)
    for device_lens in (trimmed_b, trimmed_a):
        _decode_metadata(builder, device_lens, even, seq_lens, block_table)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, reference(device_lens)[1])
