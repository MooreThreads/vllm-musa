from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = next(
    (ROOT / "vllm_musa/patches/series").glob(
        "*-perf-musa-extend-DSV4-score-DeepGEMM-to-decode-*.patch"
    )
)


def test_dsv4_score_deepgemm_covers_every_dspark4_decode_batch() -> None:
    source = PATCH.read_text(encoding="utf-8")
    # The decode bound is the multi-stream token threshold itself (64 DSpark-4
    # requests are 320 tokens); above it the prefill route takes over.
    assert "-_MUSA_DEEPSEEK_V4_SCORE_FP32_DEEPGEMM_MAX_TOKENS = 16" in source
    assert (
        "+        and a.shape[0] <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD"
        in source
    )
    assert "VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD" in source
