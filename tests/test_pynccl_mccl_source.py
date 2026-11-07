# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source contract of the MCCL pynccl library mapping in the vLLM series."""

from pathlib import Path

PATCH = (
    Path(__file__).parents[1]
    / "vllm_musa/patches/series/0214-MUSA-load-MCCL-as-the-pynccl-library.patch"
)


def test_series_maps_musa_to_mccl():
    text = PATCH.read_text()
    assert "diff --git a/vllm/utils/nccl.py b/vllm/utils/nccl.py" in text
    assert '+        elif getattr(torch.version, "musa", None) is not None:' in text
    assert '+            so_file = "libmccl.so.2"' in text
    assert "VLLM_NCCL_SO_PATH" not in "".join(
        line for line in text.splitlines(keepends=True) if line.startswith("-")
    )
