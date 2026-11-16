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
@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize(
    "kind", ["normal", "relu_sum", "ties", "constant", "narrow", "neg_zero"]
)
def test_dsv4_topk_matches_the_shared_kernel(
    kind: str, index_dtype: str, topk: int
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
        (
            7,
            8192,
            lambda row: (0, 1, topk - 1, topk, topk + 1, 4097, 8192)[row],
        ),
    ):
        scores = _scores(torch, kind, rows, width, gen)
        seq_lens = torch.tensor(
            [lens(r) for r in range(rows)], dtype=torch.int32, device="musa"
        )
        shared = torch.full((rows, topk), -7, dtype=dtype, device="musa")
        dsv4 = torch.full((rows, topk), -7, dtype=dtype, device="musa")
        musa_ops.sparse_indexer_topk_decode(scores, seq_lens, shared, topk)
        musa_ops.deepseek_v4_sparse_indexer_topk_decode(scores, seq_lens, dsv4, topk)

        assert _selected_sets(dsv4, seq_lens, topk) == _reference_sets(
            torch, scores, seq_lens, topk
        )
        if kind in ("normal", "relu_sum", "constant", "narrow"):
            # Without signed zeros the output matches the shared kernel bit for bit.
            assert torch.equal(dsv4, shared)
