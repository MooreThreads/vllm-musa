# SPDX-License-Identifier: Apache-2.0
"""Source contract for the MUSA-safe DSV4 auxiliary overlap hand-off."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / (
    "vllm_musa/patches/series/"
    "0126-MUSA-use-stream-waits-for-DSV4-overlap-hand-off.patch"
)


def test_dsv4_musa_overlap_uses_stream_waits() -> None:
    patch = PATCH.read_text()

    assert "use_stream_waits: bool = False" in patch
    assert "aux_stream.wait_stream(current_stream)" in patch
    assert "current_stream.wait_stream(aux_stream)" in patch
    assert "use_stream_waits=current_platform.is_musa()" in patch
    attention = (
        ROOT / "third_party/vllm/vllm/models/deepseek_v4/attention.py"
    ).read_text()
    v41_attention = (
        ROOT / "third_party/vllm/vllm/models/deepseek_v41/attention.py"
    ).read_text()
    assert attention.count("use_stream_waits=current_platform.is_musa()") == 4
    assert v41_attention.count("use_stream_waits=current_platform.is_musa()") == 3


def test_dsv4_overlap_does_not_serialize_decode_on_events() -> None:
    patch = PATCH.read_text()
    attention = (
        ROOT / "third_party/vllm/vllm/models/deepseek_v4/attention.py"
    ).read_text()

    assert "_musa_dsv4_graph_requires_serialized_overlap" not in patch
    assert "enable=aux_streams is not None" in attention
    assert "_musa_deepseek_v4_aux_overlap()" in attention
