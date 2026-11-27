# SPDX-License-Identifier: Apache-2.0
"""Multi-request sparse-indexer prefill chunks split into single-request chunks."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH = next(
    (ROOT / "vllm_musa/patches/series").glob(
        "*-MUSA-split-multi-request-indexer-prefill-chunks-per-*.patch"
    )
)


def _added_lines() -> list[str]:
    return [
        line[1:].strip()
        for line in PATCH.read_text().splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def test_patch_only_changes_the_sparse_indexer() -> None:
    text = PATCH.read_text()
    changed = {
        line[len("+++ b/") :] for line in text.splitlines() if line.startswith("+++ b/")
    }
    assert changed == {"vllm/model_executor/layers/sparse_attn_indexer.py"}
    assert "From: musa <musa@local>" in text


def test_refused_chunks_are_split_before_the_row_loop() -> None:
    added = _added_lines()
    split = added.index("sub_chunks = _musa_split_prefill_chunk_per_request(chunk)")
    requeue = added.index("chunks[:0] = sub_chunks")
    assert split < requeue
    assert "num_reqs=1," in added


def _chunk(seq_lens: list[int], query_lens: list[int]):
    torch = pytest.importorskip("torch")
    indexer = pytest.importorskip("vllm.v1.attention.backends.mla.indexer")

    offsets = [0]
    for seq_len in seq_lens:
        offsets.append(offsets[-1] + seq_len)
    ks: list[int] = []
    ke: list[int] = []
    for req, (seq_len, query_len) in enumerate(zip(seq_lens, query_lens)):
        for row in range(query_len):
            ks.append(offsets[req])
            ke.append(offsets[req] + max(0, seq_len - query_len + 1 + row))
    token_to_seq = [req for req, n in enumerate(seq_lens) for _ in range(n)]
    cu_seq_lens = torch.tensor(offsets, dtype=torch.int32)
    token_start = 3
    return indexer.DeepseekV32IndexerPrefillChunkMetadata(
        block_table=torch.arange(len(seq_lens) * 4, dtype=torch.int32).view(-1, 4),
        cu_seqlen_ks=torch.tensor(ks, dtype=torch.int32),
        cu_seqlen_ke=torch.tensor(ke, dtype=torch.int32),
        cu_seq_lens=cu_seq_lens,
        token_to_seq=torch.tensor(token_to_seq, dtype=torch.int32),
        total_seq_lens=offsets[-1],
        token_start=token_start,
        token_end=token_start + sum(query_lens),
        num_reqs=len(seq_lens),
        local_cu_seq_lens=cu_seq_lens,
        local_total_seq_lens=offsets[-1],
        max_local_total_seq_lens=offsets[-1],
    )


def test_split_selects_the_same_rows_as_the_whole_chunk() -> None:
    torch = pytest.importorskip("torch")
    sparse = pytest.importorskip("vllm.model_executor.layers.sparse_attn_indexer")

    chunk = _chunk(seq_lens=[37, 50, 23], query_lens=[5, 7, 4])
    sub_chunks = sparse._musa_split_prefill_chunk_per_request(chunk)

    assert [(c.token_start, c.token_end) for c in sub_chunks] == [
        (3, 8),
        (8, 15),
        (15, 19),
    ]
    for req, sub in enumerate(sub_chunks):
        assert sub.num_reqs == 1
        assert sub.cu_seq_lens.tolist() == [0, sub.total_seq_lens]
        assert sub.token_to_seq.tolist() == [0] * sub.total_seq_lens
        assert torch.equal(sub.block_table, chunk.block_table[req : req + 1])
    assert [c.total_seq_lens for c in sub_chunks] == [37, 50, 23]

    gen = torch.Generator().manual_seed(0)
    rows, heads, head_dim, topk = chunk.token_end, 4, 8, 6
    q = torch.randn(rows, heads, head_dim, generator=gen)
    k = torch.randn(chunk.total_seq_lens, head_dim, generator=gen)
    weights = torch.rand(rows, heads, generator=gen)
    whole = torch.full((rows, topk), -1, dtype=torch.int32)
    split = torch.full((rows, topk), -1, dtype=torch.int32)

    start, end = chunk.token_start, chunk.token_end
    sparse._musa_fill_topk_rows_from_indexer_logits(
        q[start:end],
        k,
        weights[start:end],
        chunk.cu_seqlen_ks,
        chunk.cu_seqlen_ke,
        whole[start:end],
        topk,
    )
    offsets = chunk.cu_seq_lens.tolist()
    for req, sub in enumerate(sub_chunks):
        start, end = sub.token_start, sub.token_end
        sparse._musa_fill_topk_rows_from_indexer_logits(
            q[start:end],
            k[offsets[req] : offsets[req + 1]],
            weights[start:end],
            sub.cu_seqlen_ks,
            sub.cu_seqlen_ke,
            split[start:end],
            topk,
        )
    assert torch.equal(whole, split)


def test_split_declines_single_and_keyless_requests() -> None:
    sparse = pytest.importorskip("vllm.model_executor.layers.sparse_attn_indexer")

    assert sparse._musa_split_prefill_chunk_per_request(_chunk([37], [5])) is None
    keyless = _chunk(seq_lens=[37, 0, 9], query_lens=[5, 2, 3])
    assert sparse._musa_split_prefill_chunk_per_request(keyless) is None
