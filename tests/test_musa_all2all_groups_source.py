# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source contract of the group-free DP/EP all2all collectives on MUSA."""

from pathlib import Path

PATCH = (
    Path(__file__).parents[1]
    / "vllm_musa/patches/series"
    / "0216-MUSA-avoid-MCCL-groups-in-DP-EP-all2all-collectives.patch"
)


def test_batched_all_gatherv_is_ungrouped_on_musa():
    text = PATCH.read_text()
    assert '+        grouped = current_platform.device_type != "musa"' in text
    assert "+        if grouped:\n+            pynccl_comm.group_start()" in text
    assert "+        if grouped:\n+            pynccl_comm.group_end()" in text


def test_uneven_collectives_are_padded_on_musa():
    text = PATCH.read_text()
    assert "+from vllm.platforms import current_platform" in text
    assert (
        "+            self._all_gatherv_padded(output_tensor, input_tensor, sizes, stream)"
        in text
    )
    assert "+            self._reduce_scatterv_padded(" in text
    assert "+            self.all_gather(gathered, padded, stream=stream)" in text
    assert (
        "+            self.reduce_scatter(reduced, padded, op=op, stream=stream)"
        in text
    )
