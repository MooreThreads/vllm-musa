# SPDX-License-Identifier: Apache-2.0
"""Source contracts for DeepSeek-V4 prefill score GEMM dispatch."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH = next(
    (REPO_ROOT / "vllm_musa/patches/series").glob(
        "*-MUSA-route-DeepSeek-V4-prefill-score-GEMM-to-DeepG*.patch"
    )
)


def _added_lines() -> list[str]:
    return [
        line[1:].strip()
        for line in PATCH.read_text().splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def test_small_m_gate_is_kept_and_prefill_rows_use_deepgemm():
    added = _added_lines()

    assert "a.shape[0] <= _MUSA_DEEPSEEK_V4_SCORE_FP32_DEEPGEMM_MAX_TOKENS" in added
    assert "and a.shape[0] > envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD" in added
    assert "vllm/models/deepseek_v4/attention.py" in PATCH.read_text()


def test_prefill_route_is_default_on_with_a_kill_switch():
    added = _added_lines()

    assert (
        'os.environ.get("VLLM_MUSA_DSV4_SCORE_DEEPGEMM_PREFILL", "1") == "1"'
        in added
    )
    assert "_MUSA_DEEPSEEK_V4_SCORE_DEEPGEMM_PREFILL" in added
    assert "From: musa <musa@local>" in PATCH.read_text()
