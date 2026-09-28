# SPDX-License-Identifier: Apache-2.0
"""Hardware checks for the Triton upgrade; source-contract tests are separate."""

import importlib

import pytest

pytest.importorskip("torchada")  # Activate MUSA mappings before torch/Triton.
import torch
import triton
import triton.language as tl


@pytest.fixture(scope="module", autouse=True)
def musa_device():
    if not getattr(torch.version, "musa", None):
        pytest.skip("requires a MUSA PyTorch build")
    assert torch.musa.is_available()
    torch.musa.set_device(0)


@triton.jit
def _copy_plus_one(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, other=0)
    tl.store(Y + offsets, x + 1, offsets < N)


@triton.jit
def _softmax(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, MANUAL: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, other=-float("inf"))
    if MANUAL:
        exp = tl.exp(x - tl.max(x, 0))
        result = exp / tl.sum(exp, 0)
    else:
        result = tl.softmax(x, dim=0)
    tl.store(Y + offsets, result, offsets < N)


@triton.jit
def _masked_uint16_bitcast(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    word = tl.load(X + offsets, offsets < N, other=0)
    bits = (word & 0xFFFF).to(tl.uint16)
    tl.store(Y + offsets, bits.to(tl.float16, bitcast=True), offsets < N)


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("n", [1, 31, 128, 513])
def test_softmax_dimension_and_manual_normalization(n, manual):
    x = torch.linspace(-8, 8, n, device="musa")
    y = torch.empty_like(x)
    _softmax[(1,)](x, y, n, triton.next_power_of_2(n), manual)
    torch.testing.assert_close(y.cpu(), torch.softmax(x.cpu(), 0), atol=1e-6, rtol=1e-5)


def test_uint16_mask_bitcast_preserves_low_word():
    words = torch.tensor([0x3C00, 0xBC00, 0x0400, 0x7BFF], dtype=torch.int32)
    words |= 0x12340000
    result = torch.empty(4, device="musa", dtype=torch.float16)
    _masked_uint16_bitcast[(1,)](words.to("musa"), result, 4, 32)
    expected = (words & 0xFFFF).to(torch.int16).view(torch.float16)
    torch.testing.assert_close(result.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "module",
    [
        "triton.knobs",
        "triton.experimental.gluon",
        "triton.experimental.gluon.language",
        "triton.tools.tensor_descriptor",
    ],
)
def test_upgrade_module_import(module):
    importlib.import_module(module)


def test_aggregate_and_cuda_extra_exports():
    from triton.language.core import _aggregate
    from vllm import triton_utils

    assert triton_utils.aggregate is _aggregate
    assert callable(tl.extra.cuda.libdevice.exp)


def test_triton_benchmarker_executes_kernel():
    from torch._inductor.runtime.benchmarking import TritonBenchmarker

    x = torch.arange(33, device="musa", dtype=torch.float32)
    y = torch.empty_like(x)

    def run():
        _copy_plus_one[(1,)](x, y, 33, 64)

    elapsed = TritonBenchmarker().triton_do_bench(run, warmup=1, rep=5)
    assert elapsed > 0
    torch.testing.assert_close(y.cpu(), x.cpu() + 1, atol=0, rtol=0)


def test_jit_monitor_hooks_observe_real_compilation(monkeypatch):
    from triton import knobs
    from triton.runtime import jit
    from vllm.utils import jit_monitor

    events = []
    monkeypatch.setattr(
        jit_monitor, "_log_triton_jit_compile", lambda *args: events.append(args)
    )
    monkeypatch.setattr(knobs.runtime, "jit_post_compile_hook", None)
    monkeypatch.setattr(
        jit, "serialize_specialization_data", jit.serialize_specialization_data
    )
    monkeypatch.setattr(knobs.autotuning, "print", False)
    monkeypatch.delenv("TRITON_PRINT_AUTOTUNING", raising=False)
    jit_monitor._setup_triton_autotuning_print()
    jit_monitor._setup_triton_jit_hook()
    assert knobs.autotuning.print
    x = torch.arange(71, device="musa", dtype=torch.float32)
    y = torch.empty_like(x)
    _copy_plus_one[(1,)](x, y, 71, 128)
    torch.testing.assert_close(y.cpu(), x.cpu() + 1, atol=0, rtol=0)
    assert events, "real JIT launch did not notify the vLLM monitor"


def test_wrap_triton_through_inductor():
    def wrapped(x):
        out = torch.empty_like(x)
        torch.library.wrap_triton(_copy_plus_one)[(1,)](x, out, x.numel(), 128)
        return out

    x = torch.arange(71, device="musa", dtype=torch.float32)
    compiled = torch.compile(wrapped, fullgraph=True)
    torch.testing.assert_close(compiled(x).cpu(), x.cpu() + 1, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["topk", "topp", "both"])
@pytest.mark.parametrize("vocab", [127, 1025, 16384])
def test_actual_topk_topp_kernel_against_pytorch(mode, vocab):
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
    from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

    torch.manual_seed(18018)
    logits = torch.randn(4, vocab, device="musa")
    k = torch.tensor([1, 7, 31, vocab], device="musa", dtype=torch.int32)
    p = torch.tensor([0.5, 0.8, 0.9, 1.0], device="musa")
    if mode == "topk":
        p = None
    if mode == "topp":
        k = None
    expected = apply_top_k_top_p_pytorch(logits.clone(), k, p)
    result = apply_top_k_top_p_triton(logits.clone(), k, p)
    torch.testing.assert_close(result.cpu(), expected.cpu(), atol=0, rtol=0)


@pytest.mark.parametrize("vocab", [127, 1025, 16384])
def test_actual_gumbel_fp32_replay_and_greedy(vocab):
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    torch.manual_seed(18018)
    logits = torch.randn(4, vocab, device="musa")
    mapping = torch.arange(4, device="musa", dtype=torch.int32)
    seeds = torch.full((4,), 12345, device="musa", dtype=torch.int64)
    pos = torch.arange(4, device="musa", dtype=torch.int64)
    temp = torch.ones(4, device="musa")
    first = gumbel_sample(logits, mapping, temp, seeds, pos, True, use_fp64=False)
    repeat = gumbel_sample(logits, mapping, temp, seeds, pos, True, use_fp64=False)
    torch.testing.assert_close(first.cpu(), repeat.cpu(), atol=0, rtol=0)
    assert ((first >= 0) & (first < vocab)).all().item()
    greedy = gumbel_sample(
        logits, mapping, temp.zero_(), seeds, pos, True, use_fp64=False
    )
    torch.testing.assert_close(greedy.cpu(), logits.cpu().argmax(-1), atol=0, rtol=0)


@pytest.mark.parametrize("temperature", [0.0, 1.0])
@pytest.mark.parametrize("block_verification", [False, True])
@pytest.mark.parametrize("rejected_step", [None, 0, 1])
def test_actual_rejection_accept_and_recover(
    temperature, block_verification, rejected_step
):
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

    # Point-mass distributions give an exact oracle for both stochastic and
    # greedy verification: accept the matching prefix, then emit the target.
    vocab = 128
    target = torch.full((3, vocab), -10000.0, device="musa")
    target[0, 1] = target[1, 2] = target[2, 3] = 0
    tokens = [1, 2]
    if rejected_step is not None:
        tokens[rejected_step] = 7
    draft = torch.full((1, 2, vocab), -10000.0, device="musa")
    for i, token in enumerate(tokens):
        draft[0, i, token] = 0
    sampled, counts = rejection_sample(
        target_logits=target,
        draft_logits=draft,
        draft_sampled=torch.tensor([0, *tokens], device="musa", dtype=torch.int64),
        cu_num_logits=torch.tensor([0, 3], device="musa", dtype=torch.int32),
        pos=torch.arange(3, device="musa", dtype=torch.int64),
        idx_mapping=torch.zeros(1, device="musa", dtype=torch.int32),
        expanded_idx_mapping=torch.zeros(3, device="musa", dtype=torch.int32),
        expanded_local_pos=torch.arange(3, device="musa", dtype=torch.int32),
        temperature=torch.tensor([temperature], device="musa"),
        seed=torch.tensor([18018], device="musa", dtype=torch.int64),
        num_speculative_steps=2,
        use_fp64=False,
        use_block_verification=block_verification,
    )
    expected_count = 3 if rejected_step is None else rejected_step + 1
    assert counts.cpu().tolist() == [expected_count]
    assert sampled.cpu()[0, :expected_count].tolist() == [1, 2, 3][:expected_count]


def test_gumbel_shards_match_global_offsets():
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    torch.manual_seed(18019)
    logits = torch.randn(4, 4096, device="musa")
    mapping = torch.arange(4, device="musa", dtype=torch.int32)
    seeds = torch.full((4,), 18019, device="musa", dtype=torch.int64)
    pos = torch.arange(4, device="musa", dtype=torch.int64)
    temp = torch.ones(4, device="musa")
    full = gumbel_sample(logits, mapping, temp, seeds, pos, True, use_fp64=False)
    ids, scores = [], []
    for start in (0, 2048):
        token, score = gumbel_sample(
            logits[:, start : start + 2048],
            mapping,
            temp,
            seeds,
            pos,
            True,
            use_fp64=False,
            vocab_start_index=start,
            return_values=True,
        )
        ids.append(token)  # gumbel_sample already returns global vocabulary IDs.
        scores.append(score)
    winner = torch.stack(scores, dim=1).argmax(dim=1, keepdim=True)
    combined = torch.stack(ids, dim=1).gather(1, winner).flatten()
    torch.testing.assert_close(combined.cpu(), full.cpu(), atol=0, rtol=0)


@pytest.mark.parametrize("use_td", [False, True])
@pytest.mark.parametrize("query_len", [1, 3])
def test_unified_attention_pointer_and_descriptor_paths(use_td, query_len):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    torch.manual_seed(18020)
    q = torch.randn(query_len, 4, 128, device="musa", dtype=torch.bfloat16)
    k = torch.randn(2, 16, 2, 128, device="musa", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=torch.tensor([0, query_len], device="musa", dtype=torch.int32),
        seqused_k=torch.tensor([32], device="musa", dtype=torch.int32),
        max_seqlen_q=query_len,
        max_seqlen_k=32,
        softmax_scale=128**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=torch.tensor([[0, 1]], device="musa", dtype=torch.int32),
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=0,
        use_td=use_td,
    )
    keys = k.cpu().float().reshape(32, 2, 128).repeat_interleave(2, dim=1)
    values = v.cpu().float().reshape(32, 2, 128).repeat_interleave(2, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.cpu().float(), keys) * (128**-0.5)
    mask = torch.arange(32)[None, :] > (
        32 - query_len + torch.arange(query_len)[:, None]
    )
    scores.masked_fill_(mask, -float("inf"))
    reference = torch.einsum("hqk,khd->qhd", scores.softmax(-1), values)
    torch.testing.assert_close(out.cpu().float(), reference, atol=0.02, rtol=0.02)
