from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative",
    [
        "csrc/musa/attention/deepseek_v4_sparse_indexer_topk.mu",
        "csrc/musa/attention/glm52_indexer_topk.mu",
    ],
)
def test_native_topk_supports_bf16_without_changing_fp32_dispatch(relative):
    source = (ROOT / relative).read_text()

    assert "#include <musa_bf16.h>" in source
    assert "template <typename LogitT, typename OutT>" in source
    assert "const LogitT *row_logits" in source
    assert "torch::kFloat32" in source
    assert "torch::kBFloat16" in source
    for input_type in ("float", "__mt_bfloat16"):
        for output_type in ("int32_t", "int64_t"):
            assert f"<{input_type}, {output_type}>" in source


def test_mate_mqa_bf16_patch_preserves_legacy_provider_and_avoids_fp32_cast():
    patches = list(
        (ROOT / "vllm_musa/patches/series").glob("*-Return-BF16-MQA-logits-*.patch")
    )
    assert len(patches) == 1
    added = "\n".join(
        line[1:]
        for line in patches[0].read_text().splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )

    assert 'provider_name == "mate.deep_gemm"' in added
    assert "_MUSA_INDEXER_LOGITS_DTYPE = torch.bfloat16" in added
    assert '"logits_dtype": _MUSA_INDEXER_LOGITS_DTYPE' in added
    assert "logits_dtype=_MUSA_INDEXER_LOGITS_DTYPE" in added
    assert "_MUSA_INDEXER_LOGITS_DTYPE.itemsize" in added
    assert "else {}" in added
    assert "torch.float32" not in added
