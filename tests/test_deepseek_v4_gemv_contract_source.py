from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_contract_selected_gemv_block_reaches_custom_op_schema() -> None:
    bindings = _source("csrc/musa/torch_bindings.cpp")
    custom_ops = _source("vllm_musa/_custom_ops.py")
    fused_moe = _source("vllm_musa/model_executor/layers/fused_moe/fused_moe.py")

    assert "bool use_swigelu, int block_n=0, int block_k=0) -> ()" in bindings
    assert "block_n: int = 0" in custom_ops
    assert "block_k: int = 0" in custom_ops
    assert "block_n,\n        block_k," in custom_ops
    assert "_requested_gemv_block(shape)" in fused_moe
    assert "block_n=gemv_block_n" in fused_moe
    assert "block_k=gemv_block_k" in fused_moe


def test_requested_tile_preserves_dsv4_and_environment_precedence() -> None:
    gemv = _source("csrc/musa/gemv.mu")
    moe = gemv[gemv.index("void musa_fused_gemv_moe(") :]
    dispatch = moe[moe.index("BlockConfig forced_config") : moe.index("switch (")]

    assert "int64_t requested_block_n" in gemv
    assert "int64_t requested_block_k" in gemv
    assert dispatch.index("ShouldUseDeepSeekV4Fp8MoeSplitTile(") < dispatch.index(
        "ParseForcedBlockConfig(&forced_config)"
    )
    assert dispatch.index("ParseForcedBlockConfig(&forced_config)") < dispatch.index(
        "requested_config.valid"
    )
    assert "requested GEMV block must provide both block_n and block_k" in gemv


def test_contract_selector_does_not_forward_explicit_environment_override() -> None:
    fused_moe = _source("vllm_musa/model_executor/layers/fused_moe/fused_moe.py")

    helper = fused_moe[
        fused_moe.index("def _requested_gemv_block(") : fused_moe.index(
            "def _tensors_share_device("
        )
    ]
    assert "os.environ.get(_GEMV_MOE_BLOCK_ENV) is not None" in helper
    assert 'shape.gemv_block != "16x8"' in helper
    assert "return 0, 0" in helper
