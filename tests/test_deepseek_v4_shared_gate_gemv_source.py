from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_shared_gate_gemv_is_layer_scoped() -> None:
    source = (
        ROOT / "vllm_musa/model_executor/kernels/linear/scaled_mm/deep_gemm.py"
    ).read_text()
    assert '".shared_experts.gate_up_proj"' in source
    assert 'getattr(layer, "tp_size", None) == 8' in source
    assert "tuple(params.weight.shape) == (512, 4096)" in source
    assert "use_deepseek_v4_shared_gate_up_gemv" in source


def test_shared_gate_native_tile_precedes_generic_override() -> None:
    source = (ROOT / "csrc/musa/gemv.mu").read_text()
    selector = source.index("SelectDeepSeekV4Fp8SharedGateUpTile(")
    dispatch = source.index("SelectDeepSeekV4Fp8SharedGateUpTile(", selector + 1)
    forced = source.index("ParseForcedBlockConfig(&forced_config)", dispatch)
    assert dispatch < forced
    assert "BlockConfig{4, 32, 0.f, true}" in source[selector:dispatch]
