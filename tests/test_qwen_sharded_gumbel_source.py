from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "vllm_musa/patches/series/0099-perf-musa-gated-sharded-qwen-gumbel.patch"
SAMPLER = ROOT / "vllm_musa/v1/sample/topk_topp_sampler.py"


def test_sharded_gumbel_patch_keeps_the_upstream_default_contract() -> None:
    source = PATCH.read_text()
    assert "vocab_start_index: int = 0" in source
    assert "return_values: bool = False" in source
    assert 'getattr(self, "_musa_skip_tp_gather", False)' in source
    assert source.count("musa_compute_logits_if_eligible(") == 2


def test_sharded_gumbel_is_narrowly_gated_and_uses_ipc_pair_gather() -> None:
    source = SAMPLER.read_text()
    assert "_MUSA_QWEN_SHARDED_MIN_BATCH = 16" in source
    assert '"VLLM_MUSA_SHARDED_QWEN_GUMBEL"' in source
    assert "tp_size == 2" in source
    assert "get_pp_group().world_size != 1" in source
    assert "maybe_musa_jit_logits_all_gather(pair, dim=-1)" in source
    assert "scores = gathered[:, :, 0].contiguous()" in source
    assert "token_ids = gathered[:, :, 1].contiguous()" in source
    assert "_musa_qwen_shard_start_index" in source
