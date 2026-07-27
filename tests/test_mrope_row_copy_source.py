# SPDX-License-Identifier: Apache-2.0
"""Source contract for asynchronous MUSA MRoPE position uploads."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0089-MUSA-copy-MRoPE-positions-row-wise.patch"
)


def test_musa_mrope_copy_uses_contiguous_rows_without_a_gate() -> None:
    source = PATCH.read_text(encoding="utf-8")

    assert "+            if current_platform.is_musa():" in source
    assert "+                for row in range(" in source
    assert (
        "+                    self.mrope_positions.gpu["
        "row, :total_num_scheduled_tokens].copy_(" in source
    )
    assert (
        "+                        self.mrope_positions.cpu["
        "row, :total_num_scheduled_tokens]," in source
    )
    assert "VLLM_MUSA_MROPE_ROW_COPY" not in source
    assert "+import os" not in source
