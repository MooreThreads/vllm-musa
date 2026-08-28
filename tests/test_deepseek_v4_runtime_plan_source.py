import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_deepseek_runtime_defaults_are_declarative() -> None:
    provider = _source("vllm_musa/runtime_plan/deepseek_v4.py")
    profile = json.loads(_source("vllm_musa/runtime_plan/profiles/deepseek_v4.json"))

    assert "resolve_declarative_runtime_plan" in provider
    assert "RuntimeDecision." not in provider
    assert profile["schema_version"] == "musa.runtime_profile.v1"
    assert profile["id"] == "deepseek_v4"
    assert len(profile["decisions"]) == 11


def test_platform_uses_plan_without_model_derived_kernel_envs() -> None:
    source = _source("vllm_musa/platform.py")
    policy = _source("vllm_musa/runtime_plan/policy.py")

    assert "DEEPSEEK_V4_FLASHMLA_SPARSE_PAGE_SIZE" in policy
    assert "deepseek_v4_flashmla_sparse_page_size" in policy
    assert "_is_deepseek_v4_model" not in source
    assert "VLLM_MUSA_GEMV_MOE_BLOCK" not in source
    assert "VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X" not in source


def test_shared_mlp_owner_guards_and_rmsnorm_instance_plan() -> None:
    linear = _source("vllm_musa/model_executor/layers/linear.py")
    layernorm = _source("vllm_musa/model_executor/layers/layernorm.py")
    model_patch = _source(
        "vllm_musa/patches/series/"
        "0104-MUSA-bind-DeepSeek-V4-RuntimePlan-decisions.patch"
    )

    assert "def forward_swiglu_clamp(" in linear
    assert "This hook is only called by DeepSeek-V4's MLP" in linear
    assert "not envs.VLLM_BATCH_INVARIANT" in linear
    assert "_deepgemm_block_fp8(self.quant_method)" in linear
    assert (
        'tuple(getattr(self, "weight_block_size", None) or ()) == (128, 128)' in linear
    )
    assert "swiglu_limit == 10.0" in linear
    assert "bind_runtime_plan(self.down_proj" in model_patch
    assert "bind_runtime_plan(self" in layernorm
    assert "DEEPSEEK_V4_TP8_FUSED_ADD_RMSNORM_BLOCK256" in layernorm
    assert "self._musa_deepseek_v4_tp8_fused_add_rmsnorm_block256" in layernorm
    forward = layernorm[layernorm.index("    def forward_oot(") :]
    assert "plan.enabled(" not in forward
    assert "block_x=256" in layernorm


def test_sparse_indexer_plan_keeps_glm_entry_shape_local() -> None:
    patch = _source(
        "vllm_musa/patches/series/"
        "0104-MUSA-bind-DeepSeek-V4-RuntimePlan-decisions.patch"
    )

    assert "DEEPSEEK_V4_NATIVE_SPARSE_INDEXER" in patch
    assert "DEEPSEEK_V4_MATERIALIZED_PREFILL_INDEXER" in patch
    assert "use_musa_materialized_prefill" in patch
    assert "q_quant.shape[1] == 32" in patch
    assert "and self.use_musa_native_indexer" in patch


def test_mtp_sparse_prefill_fixes_are_bound_at_the_dsv4_owner() -> None:
    patch = _source(
        "vllm_musa/patches/series/"
        "0106-MUSA-preserve-DeepSeek-V4-MTP-sparse-prefill-headroo.patch"
    )

    assert "_allow_dsv4_tp8_mtp_direct_out" in patch
    assert "allow_dsv4_tp8_mtp_direct_out=" in patch
    assert "_musa_deepseek_v4_mtp_sparse_prefill_reserve_bytes" in patch
    assert "available_kv_cache_memory_bytes - sparse_prefill_reserve" in patch
    assert "current_platform.is_musa()" in patch

    queue_fence = _source(
        "vllm_musa/patches/series/"
        "0107-MUSA-fence-DeepSeek-V4-MTP-mixed-prefill-queues.patch"
    )
    assert "deepseek_v4_mtp_prefill_step_requires_sync" in queue_fence
    assert "scheduled_spec_decode_tokens" not in queue_fence
    assert "torch.musa.synchronize()" in queue_fence
    policy = _source("vllm_musa/runtime_plan/policy.py")
    assert "scheduled_spec_decode_tokens" not in policy
    assert "scheduled_new_reqs" in policy
    assert '0 in getattr(cached_reqs, "num_output_tokens", ())' in policy
    assert "_musa_dsv4_mtp_prefill_queue_fence" in queue_fence


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


def test_custom_all_reduce_uses_plan_identity_and_dynamic_capture_guard() -> None:
    source = _source(
        "vllm_musa/distributed/device_communicators/musa_jit_custom_all_reduce.py"
    )

    assert "DEEPSEEK_V4_CAR_GRAPH_INPUT_CAPTURE_GUARD" in source
    assert "deepseek_v4_mtp_car_graph_staging_plan" in source
    assert "plan.allows_descriptor(descriptor)" in source
    assert "_DSV4_MTP_MAX_CAPTURE_SIZE" not in source
    assert "self._pending_graph_inputs: list[torch.Tensor] = []" in source
    assert "self._graph_input_refs: list[torch.Tensor] = []" in source
    assert "plan.model.family is not ModelFamily.DEEPSEEK_V4" in source
    assert "plan.supports(" in source
    assert "cudagraph_capture_sizes" in source
    assert '"DeepseekV4" in str(arch)' not in source
    assert 'getattr(hf_config, "model_type", None)' not in source
