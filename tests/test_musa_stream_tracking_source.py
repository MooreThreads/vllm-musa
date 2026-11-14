# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source contract of the torch_musa stream tracking in the vLLM series."""

from pathlib import Path

PATCH = (
    Path(__file__).parents[1]
    / "vllm_musa/patches/series"
    / "0214-MUSA-track-torch_musa-streams-in-current_stream.patch"
)


def test_series_tracks_torch_musa_set_stream():
    text = PATCH.read_text()
    assert "diff --git a/vllm/utils/torch_utils.py b/vllm/utils/torch_utils.py" in text
    assert "+def _track_musa_set_stream() -> None:" in text
    assert "+        _current_stream_tls.value = stream" in text
    assert "+    torch_musa.set_stream = set_stream" in text
    assert '+                if current_platform.device_type == "musa":' in text
    assert "+                    _track_musa_set_stream()" in text
