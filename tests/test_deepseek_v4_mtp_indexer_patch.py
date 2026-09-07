from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0110-MUSA-adapt-DSV4-MTP-v028.patch"
)


def _patch_text() -> str:
    return PATCH.read_text(encoding="utf-8")


def test_mtp_indexer_preserves_cache_storage_stride() -> None:
    text = _patch_text()

    assert "_musa_indexer_cache_rows(kv_cache)" in text
    assert "cache_rows.index_select(0, page_ids.reshape(-1))" in text
    assert "page-tail scales" in text


def test_mtp_indexer_decodes_fp8_and_uses_bf16_gemm() -> None:
    text = _patch_text()

    assert "q_quant[:rows]\n+        .contiguous()" in text
    assert ".view(torch.float8_e4m3fn)" in text
    assert "per_head = torch.bmm(q, k.transpose(1, 2))" in text
    assert "_musa_try_fill_mtp_topk_bf16" in text


def test_mtp_indexer_patch_is_scoped_to_sparse_indexer() -> None:
    changed_files = {
        line.removeprefix("+++ b/")
        for line in _patch_text().splitlines()
        if line.startswith("+++ b/")
    }

    assert "vllm/model_executor/layers/sparse_attn_indexer.py" in changed_files
