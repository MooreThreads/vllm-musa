from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa/patches/series/0174-perf-musa-extend-DSV4-score-DeepGEMM-to-decode-batch.patch"
)


def test_dsv4_score_deepgemm_covers_dspark4_decode_graph_ladder() -> None:
    source = PATCH.read_text(encoding="utf-8")
    assert "_MUSA_DEEPSEEK_V4_SCORE_FP32_DEEPGEMM_MAX_TOKENS = 80" in source
    assert "M=20, 40, and 80" in source
    assert "long-prefill" in source
