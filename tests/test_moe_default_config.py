"""MUSA-100038: the tiny-M MoE default must not collapse occupancy on S5000.

At the tiny-M decode shape (7 tokens x topk 6 = 42 useful rows over 13-41 of 128
experts, one tile per expert) the upstream default pipelines the K loop 4 stages
deep at BLOCK_K=128, which reserves ~4x shared memory per block and costs resident
blocks.  Measured on S5000: -17.80% TPOT at 1 stage, byte-identical output.
"""

from vllm.model_executor.layers.fused_moe.fused_moe import get_default_config
from vllm.platforms import current_platform


def _cfg(M: int = 7) -> dict[str, int]:
    return get_default_config(M=M, E=128, N=1856, K=2688, topk=6, dtype="bfloat16")


def test_tiny_m_num_stages_follows_platform() -> None:
    assert _cfg()["num_stages"] == (1 if current_platform.is_musa() else 4)


def test_tiny_m_other_axes_unchanged() -> None:
    config = _cfg()
    assert config["BLOCK_SIZE_M"] == 16
    assert config["BLOCK_SIZE_N"] == 64
    assert config["BLOCK_SIZE_K"] == 128
    assert config["GROUP_SIZE_M"] == 1
    assert config["SPLIT_K"] == 1
    assert config["num_warps"] == 4


def test_all_buckets_up_to_32_take_one_stage_on_musa() -> None:
    for M in (1, 4, 7, 8, 16, 32):
        assert _cfg(M)["num_stages"] == (1 if current_platform.is_musa() else 4)


def test_larger_m_keeps_upstream_staging() -> None:
    assert _cfg(512)["num_stages"] == 3
