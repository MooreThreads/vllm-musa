from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GEMV = ROOT / "csrc/musa/gemv.mu"


def test_deepseek_v4_w1_w2_use_shape_specific_tiles() -> None:
    source = GEMV.read_text()
    helper = source[
        source.index("bool ShouldUseDeepSeekV4Fp8MoeSplitTile(") : source.index(
            "void musa_fused_gemv("
        )
    ]
    moe = source[source.index("void musa_fused_gemv_moe(") :]

    assert "num_experts != 256" in helper
    assert "bseqlen != 1" in helper
    assert "topk == 6 && hidden_size == 4096" in helper
    assert "reduce_size == 512 && nr_n == 256" in helper
    assert "topk == 1 && hidden_size == 256" in helper
    assert "reduce_size == 4096 && nr_n == 4096" in helper
    assert "w1 ? BlockConfig{4, 32" in helper
    assert ": BlockConfig{32, 4" in helper

    dispatch = moe[moe.index("BlockConfig forced_config") : moe.index("switch (")]
    assert dispatch.index("ShouldUseDeepSeekV4Fp8MoeSplitTile(") < dispatch.index(
        "ParseForcedBlockConfig(&forced_config)"
    )
    assert "use_swigelu ? BlockConfig{4, 32" in dispatch
    assert ": BlockConfig{32, 4" in dispatch
