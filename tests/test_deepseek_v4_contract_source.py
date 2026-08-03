from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_platform_uses_contract_without_model_derived_kernel_envs() -> None:
    source = _source("vllm_musa/platform.py")

    assert "DEEPSEEK_V4_TP8_FLASHMLA_SPARSE_PAGE256" in source
    assert "_is_deepseek_v4_model" not in source
    assert "VLLM_MUSA_GEMV_MOE_BLOCK" not in source
    assert "VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X" not in source


def test_shared_mlp_and_rmsnorm_bind_instance_contracts() -> None:
    linear = _source("vllm_musa/model_executor/layers/linear.py")
    layernorm = _source("vllm_musa/model_executor/layers/layernorm.py")
    model_patch = _source(
        "vllm_musa/patches/series/"
        "0101-MUSA-bind-DeepSeek-V4-optimization-contract.patch"
    )

    assert "DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8" in linear
    assert "prefers_optimization(" in linear
    assert "bind_optimization_contract(self.down_proj" in model_patch
    assert "bind_optimization_contract(self" in layernorm
    assert "DEEPSEEK_V4_TP8_FUSED_ADD_RMSNORM_BLOCK256" in layernorm
    assert "block_x=256" in layernorm


def test_sparse_indexer_contract_keeps_glm_entry_shape_local() -> None:
    patch = _source(
        "vllm_musa/patches/series/"
        "0101-MUSA-bind-DeepSeek-V4-optimization-contract.patch"
    )

    assert "DEEPSEEK_V4_NATIVE_SPARSE_INDEXER" in patch
    assert "DEEPSEEK_V4_MATERIALIZED_PREFILL_INDEXER" in patch
    assert "use_musa_materialized_prefill" in patch
    assert "q_quant.shape[1] == 32" in patch
    assert "and self.use_musa_native_indexer" in patch


def test_fused_add_rmsnorm_argument_is_graph_static_and_env_override_wins() -> None:
    schema = _source("csrc/musa/torch_bindings.cpp")
    kernel = _source("csrc/musa/fused_add_rmsnorm.mu")

    assert "float eps, int block_x=0) -> ()" in schema
    assert "int requested_block" in kernel
    assert "env_forced_block > 0 ? env_forced_block : requested_block" in kernel


def test_deepseek_tp8_moe_tile_is_selected_by_exact_runtime_shape() -> None:
    source = _source("vllm_musa/model_executor/layers/fused_moe/fused_moe.py")

    assert "is_deepseek_v4_flash_tp8_shape" in source
    assert 'gemv_block = "16x8" if is_deepseek_v4_flash_tp8_shape else "auto"' in source
    assert 'os.environ.get("VLLM_MUSA_GEMV_MOE_BLOCK")' in source


def test_custom_all_reduce_uses_contract_identity_and_dynamic_capture_guard() -> None:
    source = _source(
        "vllm_musa/distributed/device_communicators/musa_jit_custom_all_reduce.py"
    )

    assert "DEEPSEEK_V4_CAR_GRAPH_INPUT_CAPTURE_GUARD" in source
    assert "contract.model.family is not ModelFamily.DEEPSEEK_V4" in source
    assert "contract.supports(" in source
    assert "cudagraph_capture_sizes" in source
    assert '"DeepseekV4" in str(arch)' not in source
    assert 'getattr(hf_config, "model_type", None)' not in source
