from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FUSED_MOE = ROOT / "vllm_musa/model_executor/layers/fused_moe/fused_moe.py"


def test_musa_moe_sum_uses_fast_kernel_with_generic_fallback() -> None:
    source = FUSED_MOE.read_text()

    assert (
        "from vllm_musa.jit_kernel.csrc.moe import maybe_fast_moe_sum" in source
    )
    assert "if not maybe_fast_moe_sum(" in source
    assert "ops.moe_sum(moe_sum_input, curr_out_hidden_states)" in source


def test_musa_moe_sum_dispatch_has_no_ab_environment_gate() -> None:
    source = FUSED_MOE.read_text()
    dispatch = source[source.index("moe_sum_input =") : source.index("return out_hidden_states")]

    assert "os.environ" not in dispatch
    assert "getenv" not in dispatch
