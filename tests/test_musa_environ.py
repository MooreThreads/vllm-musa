from vllm_musa.utils.environ import envs


def test_qwen_gdn_width4_split_env_bool_contract():
    with envs.VLLM_MUSA_QWEN_GDN_WIDTH4_PREFILL_SPLIT.override("yes"):
        assert envs.VLLM_MUSA_QWEN_GDN_WIDTH4_PREFILL_SPLIT.get() is True
    with envs.VLLM_MUSA_QWEN_GDN_WIDTH4_PREFILL_SPLIT.override("off"):
        assert envs.VLLM_MUSA_QWEN_GDN_WIDTH4_PREFILL_SPLIT.get() is False
