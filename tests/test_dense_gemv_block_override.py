# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEMV = ROOT / "csrc" / "musa" / "gemv.mu"


def test_dense_gemv_override_is_separate_and_precedes_exact_selectors() -> None:
    source = GEMV.read_text(encoding="utf-8")
    dense = source[
        source.index("void musa_fused_gemv(") : source.index(
            "void musa_fused_gemv_moe("
        )
    ]
    dispatch = dense[dense.index("BlockConfig forced_config") : dense.index("switch (")]

    assert 'kGemvBlockEnv = "VLLM_MUSA_GEMV_BLOCK"' in source
    assert "ParseDenseForcedBlockConfig(&dense_forced_config)" in dispatch
    assert dispatch.index(
        "ParseDenseForcedBlockConfig(&dense_forced_config)"
    ) < dispatch.index("SelectDeepSeekV4Fp8OProjTile(")
    assert dispatch.index("SelectDeepSeekV4Fp8OProjTile(") < dispatch.index(
        "ParseForcedBlockConfig(&forced_config)"
    )


def test_dense_override_validates_shape_and_keeps_moe_override_unchanged() -> None:
    source = GEMV.read_text(encoding="utf-8")
    helper = source[
        source.index("bool ParseDenseForcedBlockConfig(") : source.index(
            "bool IsForcedBlockConfigValid("
        )
    ]

    assert "std::getenv(kGemvBlockEnv)" in helper
    assert "must use '<block_n>x<block_k>'" in helper
    assert "block_n * block_k must be <= 512" in helper
    assert "IsForcedBlockConfigValid(dense_forced_config" in source
    assert 'kGemvMoeBlockEnv = "VLLM_MUSA_GEMV_MOE_BLOCK"' in source
