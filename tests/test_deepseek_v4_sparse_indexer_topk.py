"""DeepSeek-V4 decode indexer top-k: dedicated radix op, GLM-5.2 on the shared op."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ATTENTION = ROOT / "csrc/musa/attention"
DSV4_TOPK = "csrc/musa/attention/deepseek_v4_sparse_indexer_topk.mu"
BINDINGS = ROOT / "csrc/musa/torch_bindings.cpp"
SERIES_PATCH = next(
    (ROOT / "vllm_musa/patches/series").glob(
        "*-MUSA-select-indexer-top-k-with-the-staged-radix-kern.patch"
    )
)
TOPK = 512


def test_radix_topk_lives_in_its_own_dsv4_kernel_file() -> None:
    for shared in ("glm52_indexer_topk.mu", "deepseek_v4_indexer_topk.mu"):
        source = (ATTENTION / shared).read_text()
        assert "sparse_indexer_topk_radix_kernel" not in source
        assert "radix_topk_supported" not in source
    dsv4 = (ROOT / DSV4_TOPK).read_text()
    assert "sparse_indexer_topk_radix_kernel" in dsv4
    assert "void deepseek_v4_sparse_indexer_topk_decode(" in dsv4
    assert f'"{DSV4_TOPK}"' in (ROOT / "setup.py").read_text()
    assert (
        '"deepseek_v4_sparse_indexer_topk_decode(Tensor logits' in BINDINGS.read_text()
    )


def test_series_routes_small_topk_to_the_staged_radix_op() -> None:
    added = [
        line[1:].strip()
        for line in SERIES_PATCH.read_text().splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    dispatch = added[added.index("if topk <= 512:") :]
    fallback = "_musa_custom_ops.sparse_indexer_topk_decode(logits, row_lens, topk_indices, topk)"
    dispatch = dispatch[: dispatch.index(fallback) + 1]
    assert "_musa_custom_ops.deepseek_v4_sparse_indexer_topk_decode(" in dispatch
    assert dispatch.index("return") < dispatch.index(fallback)


def _scores(torch, kind: str, rows: int, width: int, gen):
    if kind == "normal":
        return torch.randn(rows, width, generator=gen, device="musa")
    if kind == "relu_sum":
        x = torch.relu(torch.randn(rows, width, 16, generator=gen, device="musa"))
        w = torch.rand(16, generator=gen, device="musa")
        return (x * w).sum(-1)
    if kind == "ties":
        return (
            torch.round(torch.randn(rows, width, generator=gen, device="musa") * 4) / 4
        )
    if kind == "constant":
        return torch.full((rows, width), 3.0, device="musa")
    if kind == "narrow":  # one fp16 bin holds the whole row: stage overflow
        return 5.0 + torch.rand(rows, width, generator=gen, device="musa") * 0.5
    if kind == "neg_inf":
        return torch.full((rows, width), float("-inf"), device="musa")
    scores = torch.randn(rows, width, generator=gen, device="musa")
    scores[:, ::7] = 0.0
    scores[:, 3::7] = -0.0
    return scores


def _selected_sets(out, seq_lens, topk: int) -> list[list[int]]:
    sets = []
    for row in range(out.shape[0]):
        n = min(int(seq_lens[row]), topk)
        values = out[row, :topk].cpu().tolist()
        assert all(v == -1 for v in values[n:]), "padding must be -1"
        sets.append(sorted(values[:n]))
    return sets


def _reference_sets(torch, scores, seq_lens, topk: int) -> list[list[int]]:
    sets = []
    for row in range(scores.shape[0]):
        n = int(seq_lens[row])
        if n <= topk:
            sets.append(list(range(n)))
            continue
        order = torch.sort(-scores[row, :n].float().cpu(), stable=True).indices[:topk]
        sets.append(sorted(order.tolist()))
    return sets


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
@pytest.mark.parametrize("logits_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "kind", ["normal", "relu_sum", "ties", "constant", "narrow", "neg_zero", "neg_inf"]
)
def test_dsv4_topk_matches_the_shared_kernel(
    kind: str, index_dtype: str, logits_dtype: str, strided: bool
) -> None:
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        pytest.skip("MUSA-only test")
    from vllm_musa import _custom_ops as musa_ops

    dtype = getattr(torch, index_dtype)
    gen = torch.Generator(device="musa").manual_seed(0)
    for rows, width, lens in (
        (25, 8192, lambda r: 8000 - r % 5),
        (5, 32768, lambda r: 32000 - r),
        (7, 8192, lambda r: (100, 511, 512, 513, 600, 4097, 8192)[r]),
    ):
        scores = _scores(torch, kind, rows, width, gen).to(getattr(torch, logits_dtype))
        if strided:
            storage = torch.empty(rows, width + 32, dtype=scores.dtype, device="musa")
            storage[:, :width].copy_(scores)
            scores = storage[:, :width]
        seq_lens = torch.tensor(
            [lens(r) for r in range(rows)], dtype=torch.int32, device="musa"
        )
        shared = torch.full((rows, TOPK), -7, dtype=dtype, device="musa")
        dsv4 = torch.full((rows, TOPK), -7, dtype=dtype, device="musa")
        musa_ops.sparse_indexer_topk_decode(scores, seq_lens, shared, TOPK)
        musa_ops.deepseek_v4_sparse_indexer_topk_decode(scores, seq_lens, dsv4, TOPK)

        assert _selected_sets(dsv4, seq_lens, TOPK) == _reference_sets(
            torch, scores, seq_lens, TOPK
        )
        if kind in ("normal", "relu_sum", "constant", "narrow"):
            # Without signed zeros the output matches the shared kernel bit for bit.
            assert torch.equal(dsv4, shared)


@pytest.mark.parametrize("logits_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_sparse_prefill_topk_handles_windows(logits_dtype, index_dtype) -> None:
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        pytest.skip("MUSA-only test")
    from vllm_musa import _custom_ops as musa_ops

    scores = torch.randn(6, 8224, device="musa").to(getattr(torch, logits_dtype))[
        :, :8192
    ]
    starts = torch.tensor([0, 13, 300, 8192, 61, 17], dtype=torch.int32, device="musa")
    ends = torch.tensor(
        [0, 524, 813, 8192, 8192, 529], dtype=torch.int32, device="musa"
    )
    output = torch.full((6, TOPK), -7, dtype=getattr(torch, index_dtype), device="musa")
    musa_ops.sparse_indexer_topk(scores, starts, ends, output, TOPK)
    for row in range(6):
        start, end = int(starts[row]), int(ends[row])
        count = min(TOPK, end - start)
        expected = torch.argsort(-scores[row, start:end].float().cpu(), stable=True)[
            :count
        ]
        assert sorted(output[row, :count].cpu().tolist()) == sorted(expected.tolist())
        assert output[row, count:].eq(-1).all()


@pytest.mark.parametrize("logits_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
@pytest.mark.parametrize("topk", [1, 31, 32, 33, 255, 256, 511, 512])
def test_radix_topk_resolves_overflow_ties_in_position_order(
    logits_dtype, index_dtype, topk
) -> None:
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        pytest.skip("MUSA-only test")
    from vllm_musa import _custom_ops as musa_ops

    lengths = torch.tensor(
        [4095, 4096, 4097, 8191, 8192, 511, 512, 513, 0],
        dtype=torch.int64,
        device="musa",
    )
    storage = torch.full(
        (len(lengths), 8224), 5.0, dtype=getattr(torch, logits_dtype), device="musa"
    )
    scores = storage[:, :8192]
    scores[1::2, 3073:4097] = 5.25
    scores[2::3, 7168:] = 5.25
    output = torch.full(
        (len(lengths), topk), -7, dtype=getattr(torch, index_dtype), device="musa"
    )
    shared = torch.empty_like(output)
    musa_ops.deepseek_v4_sparse_indexer_topk_decode(scores, lengths, output, topk)
    musa_ops.sparse_indexer_topk_decode(scores, lengths, shared, topk)
    assert torch.equal(output, shared)
    assert _selected_sets(output, lengths, topk) == _reference_sets(
        torch, scores, lengths, topk
    )


@pytest.mark.parametrize("logits_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
@pytest.mark.parametrize("topk", [1, 31, 512])
def test_radix_topk_preserves_all_bf16_encodings(
    logits_dtype, index_dtype, topk
) -> None:
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        pytest.skip("MUSA-only test")
    from vllm_musa import _custom_ops as musa_ops

    encodings = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = encodings.view(torch.bfloat16).reshape(2, 32768).roll(127, dims=1)
    storage = torch.empty((2, 32800), dtype=getattr(torch, logits_dtype), device="musa")
    scores = storage[:, :32768]
    scores.copy_(values.to(getattr(torch, logits_dtype)))
    lengths = torch.full((2,), 32768, dtype=torch.int64, device="musa")
    output = torch.full((2, topk), -7, dtype=getattr(torch, index_dtype), device="musa")
    musa_ops.deepseek_v4_sparse_indexer_topk_decode(scores, lengths, output, topk)
    normalized = torch.nan_to_num(
        values.float(), nan=float("-inf"), posinf=float("inf"), neginf=float("-inf")
    )
    for row in range(2):
        positions = torch.argsort(normalized[row], descending=True, stable=True)[
            :topk
        ].tolist()
        positions.sort(key=lambda position: (position % 256, position))
        assert output[row].cpu().tolist() == positions


@pytest.mark.parametrize("logits_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("width", [8192, 16384, 65570])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
@pytest.mark.parametrize("topk", [1, 31, 255, 256, 511, 512])
def test_radix_topk_preserves_staged_owner_order(
    logits_dtype, width, index_dtype, topk
) -> None:
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        pytest.skip("MUSA-only test")
    from vllm_musa import _custom_ops as musa_ops

    generator = torch.Generator().manual_seed(3713)
    positions = torch.randperm(width, generator=generator)[: topk + 37].to("musa")
    storage = torch.full(
        (3, width + 37), -10.0, dtype=getattr(torch, logits_dtype), device="musa"
    )
    scores = storage[:, :width]
    scores[:, positions] = 5.0
    scores[:, positions[: topk // 2]] = 9.0
    lengths = torch.tensor([width, width - 1, topk - 1], device="musa")
    output = torch.full(
        (3, topk + 3), -7, dtype=getattr(torch, index_dtype), device="musa"
    )
    musa_ops.deepseek_v4_sparse_indexer_topk_decode(scores, lengths, output, topk)
    for row, length in enumerate(lengths.cpu().tolist()):
        if length <= topk:
            expected = list(range(length)) + [-1] * (topk - length)
        else:
            expected = torch.argsort(
                scores[row, :length].float().cpu(), descending=True, stable=True
            )[:topk].tolist()
            expected.sort(key=lambda position: (position % 256, position))
        assert output[row, :topk].cpu().tolist() == expected
    assert torch.all(output[:, topk:] == -7)


@pytest.mark.parametrize("logits_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("width", [8192, 65570])
def test_sparse_topk_cudagraph_replays_both_input_dtypes(logits_dtype, width) -> None:
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        pytest.skip("MUSA-only test")
    from vllm_musa import _custom_ops as musa_ops

    scores = torch.randn(32, width, device="musa").to(getattr(torch, logits_dtype))
    lengths = torch.full((32,), width, dtype=torch.int32, device="musa")
    output = torch.empty((32, TOPK), dtype=torch.int32, device="musa")
    musa_ops.deepseek_v4_sparse_indexer_topk_decode(scores, lengths, output, TOPK)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        musa_ops.deepseek_v4_sparse_indexer_topk_decode(scores, lengths, output, TOPK)
    scores.copy_(torch.randn_like(scores))
    graph.replay()
    torch.musa.synchronize()
    assert _selected_sets(output, lengths, TOPK) == _reference_sets(
        torch, scores, lengths, TOPK
    )
