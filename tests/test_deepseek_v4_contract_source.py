from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_platform_uses_contract_without_model_derived_kernel_envs() -> None:
    source = _source("vllm_musa/platform.py")
    policy = _source("vllm_musa/optimization_contract/policy.py")

    assert "DEEPSEEK_V4_TP8_FLASHMLA_SPARSE_PAGE256" in policy
    assert "deepseek_v4_flashmla_sparse_page_size" in policy
    assert "_is_deepseek_v4_model" not in source
    assert "VLLM_MUSA_GEMV_MOE_BLOCK" not in source
    assert "VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X" not in source


def test_shared_mlp_owner_guards_and_rmsnorm_instance_contract() -> None:
    linear = _source("vllm_musa/model_executor/layers/linear.py")
    layernorm = _source("vllm_musa/model_executor/layers/layernorm.py")
    model_patch = _source(
        "vllm_musa/patches/series/"
        "0101-MUSA-bind-DeepSeek-V4-optimization-contract.patch"
    )

    assert "def forward_swiglu_clamp(" in linear
    assert "This hook is only called by DeepSeek-V4's MLP" in linear
    assert "not envs.VLLM_BATCH_INVARIANT" in linear
    assert "_deepgemm_block_fp8(self.quant_method)" in linear
    assert (
        'tuple(getattr(self, "weight_block_size", None) or ()) == (128, 128)' in linear
    )
    assert "swiglu_limit == 10.0" in linear
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


def test_mtp_sparse_prefill_fixes_are_bound_at_the_dsv4_owner() -> None:
    patch = _source(
        "vllm_musa/patches/series/"
        "0102-MUSA-preserve-DeepSeek-V4-MTP-sparse-prefill-headroom.patch"
    )

    assert "DEEPSEEK_V4_TP8_MTP_SPARSE_DIRECT_OUT" in patch
    assert "allow_dsv4_tp8_mtp_direct_out=" in patch
    assert 'attention_backend_hint="flashmla"' in patch
    assert "deepseek_v4_mtp_sparse_prefill_headroom_bytes" in patch
    assert "musa_workspace_headroom_bytes" in patch
    assert "if current_platform.is_musa():" in patch


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
