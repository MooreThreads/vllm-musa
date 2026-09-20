from pathlib import Path


def test_mtp_query_start_loc_patch_preserves_capture_pointer():
    patch = next(
        Path(__file__).parents[1]
        .joinpath("vllm_musa/patches/series")
        .glob("0155-MUSA-preserve-MTP-captured-query-start-pointer.patch")
    ).read_text()
    assert "query_start_loc[: batch_size + 1] = self.arange[" in patch
    assert "query_start_loc_cpu[: batch_size + 1]" in patch
    assert not any(
        line.startswith("+")
        and "query_start_loc = self.arange[: batch_size + 1]" in line
        for line in patch.splitlines()
    )
