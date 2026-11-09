# SPDX-License-Identifier: Apache-2.0
"""Source contract for the Triton 3.2 K-masked fused-MoE loads.

On the even-K path the fused-MoE kernel loads B without a K mask, and Triton 3.2
on MUSA pipelines those loads past the last K tile. The series routes Triton 3.2
to the K-masked path; this fails if that predicate is dropped or the change moves
out of the launcher.
"""

from pathlib import Path

SERIES = Path(__file__).resolve().parent.parent / "vllm_musa" / "patches" / "series"
MEMBER = "0213-MUSA-mask-fused-MoE-K-loads-on-Triton-3.2.patch"


def test_triton32_takes_the_k_masked_path() -> None:
    text = (SERIES / MEMBER).read_text()
    removed = [
        line[1:].strip() for line in text.splitlines() if line.startswith("-    ")
    ]
    added = [line[1:].strip() for line in text.splitlines() if line.startswith("+    ")]
    assert text.count("diff --git") == 1
    assert "def invoke_fused_moe_triton_kernel(" in text
    assert removed == ["even_Ks = B.size(2) % BLOCK_SIZE_K == 0"]
    assert (
        "even_Ks = B.size(2) % BLOCK_SIZE_K == 0 and not is_musa_triton_32()" in added
    )
