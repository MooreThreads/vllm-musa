from pathlib import Path


def _source(relative: str) -> str:
    return (Path(__file__).parents[1] / relative).read_text(encoding="utf-8")


def test_weighted_rmsnorm_provider_reads_frozen_hardware_and_exact_tactic():
    source = _source("vllm_musa/deepseek_v4_mhc.py")
    helper = source[
        source.index("def _try_mhc_weighted_rms_norm_musa(") : source.index(
            "def mhc_pre_musa_with_norm("
        )
    ]

    assert "get_primed_musa_kernel_hardware(device_index)" in helper
    assert "select_mhc_weighted_rmsnorm_tactic(" in helper
    assert "threads if tactic is None else tactic.threads" in helper
    assert "VLLM_MUSA_MHC_WEIGHTED_RMSNORM_THREADS" not in source
    assert "diagnostic_threads" not in helper
    assert "if not torch.compiler.is_compiling():" in helper
    assert "query_musa_kernel_hardware" not in helper
    assert "torch.musa.get_device_properties" not in helper
