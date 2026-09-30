"""DeepSeek-V4 decode indexer top-k: dedicated radix op, GLM-5.2 on the shared op."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ATTENTION = ROOT / "csrc/musa/attention"
DSV4_TOPK = "csrc/musa/attention/deepseek_v4_sparse_indexer_topk.mu"
BINDINGS = ROOT / "csrc/musa/torch_bindings.cpp"
SERIES_PATCH = (
    ROOT / "vllm_musa/patches/series/"
    "0177-MUSA-route-DSV4-decode-indexer-top-k-to-its-own-op.patch"
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


def test_series_routes_only_dsv4_rows_to_the_dedicated_op() -> None:
    added = [
        line[1:].strip()
        for line in SERIES_PATCH.read_text().splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    assert "_musa_custom_ops.deepseek_v4_sparse_indexer_topk_decode" in added
    assert "if is_deepseek_v4" in added
    assert "else _musa_custom_ops.sparse_indexer_topk_decode" in added


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
@pytest.mark.parametrize(
    "kind", ["normal", "relu_sum", "ties", "constant", "narrow", "neg_zero"]
)
def test_dsv4_topk_matches_the_shared_kernel(kind: str, index_dtype: str) -> None:
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
        scores = _scores(torch, kind, rows, width, gen)
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
