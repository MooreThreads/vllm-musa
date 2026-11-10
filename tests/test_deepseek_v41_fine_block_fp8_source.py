from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "vllm_musa" / "patches" / "series"
GEMV = ROOT / "vllm_musa" / "kernels" / "fp8_block_gemv.py"
LINEAR = (
    ROOT
    / "vllm_musa"
    / "model_executor"
    / "kernels"
    / "linear"
    / "scaled_mm"
    / "block_gemv.py"
)
EINSUM = ROOT / "vllm_musa" / "deepseek_v4_jit" / "fp8_einsum.py"
FUSED_MOE = (
    ROOT / "vllm_musa" / "model_executor" / "layers" / "fused_moe" / "fused_moe.py"
)
MHC = ROOT / "vllm_musa" / "deepseek_v4_mhc.py"
ACTIVATION = ROOT / "vllm_musa" / "model_executor" / "layers" / "activation.py"
LINEAR_LAYER = ROOT / "vllm_musa" / "model_executor" / "layers" / "linear.py"
GEMM = ROOT / "vllm_musa" / "kernels" / "fp8_block_gemm.py"
POLICY = (
    ROOT
    / "vllm_musa"
    / "model_executor"
    / "layers"
    / "fused_moe"
    / "dispatch_policy.py"
)
TILELANG = ROOT / "vllm_musa" / "deepseek_v4_jit" / "tilelang_kernels.py"
MHC_PRENORM = ROOT / "vllm_musa" / "kernels" / "mhc_prenorm.py"


def _patch(prefix: str) -> str:
    (path,) = SERIES.glob(f"{prefix}-*.patch")
    return path.read_text(encoding="utf-8")


def _added(text: str) -> str:
    return "\n".join(
        line[1:]
        for line in text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def _function(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    node = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return ast.get_source_segment(source, node)


def test_block_gemv_masks_invalid_experts_instead_of_returning() -> None:
    kernel = _function(GEMV, "_fp8_block_gemv_kernel")
    assert "return" not in kernel
    assert "valid = (expert >= 0) & (expert < num_experts)" in kernel
    assert "n_mask = n_mask & valid" in kernel
    assert "A_GROUP_K" in kernel and "B_GROUP_K" in kernel


def test_linear_kernel_takes_only_fine_k_blocks_and_small_m() -> None:
    source = LINEAR.read_text(encoding="utf-8")
    assert "weight_group.col not in (16, 32, 64)" in source
    assert "act_group.col != weight_group.col" in source
    assert "_small_m_path(B, Bs, A.shape[0])" in source
    assert "fp8_block_scaled_mm(A, As, B, Bs, self.config.out_dtype)" in source
    assert "_upcast_e8m0_to_fp32" in source
    assert "return super().apply_block_scaled_mm(A, B, As, Bs)" in source


def test_series_lists_block_gemv_between_deepgemm_and_triton() -> None:
    added = _added(_patch("0183"))
    assert "MUSAFp8BlockGemvScaledMMKernel," in added
    text = _patch("0183")
    order = [
        text.index(
            "MUSADeepGemmFp8BlockScaledMMKernel,\n+                MUSAFp8BlockGemvScaledMMKernel,"
        ),
        text.index("TritonFp8BlockScaledMMKernel,\n             ]"),
    ]
    assert order == sorted(order)


def test_prefill_indexer_rows_are_scored_without_host_syncs() -> None:
    text = _patch("0182")
    added = _added(text)
    removed = "\n".join(
        line[1:]
        for line in text.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    assert ".item()" not in added
    assert "int(cu_seqlen_ks[row].item())" in removed
    assert "torch.matmul(q_deq[row_start:row_end], keys_t)" in added
    assert 'masked_fill_((positions < lo) | (positions >= hi), float("-inf"))' in added
    assert "top - lo" in added


def test_o_proj_fine_blocks_run_before_the_dequant_fallback() -> None:
    dispatcher = _function(EINSUM, "try_musa_deepseek_v4_fp8_einsum")
    assert dispatcher.index("_try_musa_fp8_einsum_fine_blocks(") < dispatcher.index(
        "_dequant_weight("
    )
    fine = _function(EINSUM, "_try_musa_fp8_einsum_fine_blocks")
    assert "tokens <= _BLOCK_GEMV_MAX_TOKENS" in fine
    assert "fp8_block_splitk_config(weight[g], scales[g], tokens)" in fine
    assert (
        fine.index("fp8_block_gemv(")
        < fine.index("fp8_block_bf16_supported(")
        < fine.index("fp8_block_scaled_mm_supported(")
    )
    assert "repeat_interleave(act_group // block_k" in fine


def test_moe_block_gemv_gate_is_checked_before_backend_selection() -> None:
    dispatch = _function(FUSED_MOE, "_musa_fused_experts_impl_dispatch")
    assert dispatch.index("_can_use_musa_fp8_block_moe_gemv(") < dispatch.index(
        "backend = MusaFusedMoeBackend.UPSTREAM"
    )
    gate = _function(FUSED_MOE, "_can_use_musa_fp8_block_moe_gemv")
    assert "block_shape[1] not in (16, 32, 64)" in gate
    assert "apply_router_weight_on_input" in gate
    assert "expert_map is not None" in gate
    impl = _function(FUSED_MOE, "_musa_fp8_block_moe_gemv")
    assert "apply_moe_activation(MoEActivation.SILU, down_input, cache1)" in impl
    assert "swiglu_limit=gemm1_clamp_limit" in impl
    assert "sum_topk=top_k" in impl


def test_down_gemvs_apply_the_clamped_swiglu_in_kernel() -> None:
    dot = _function(GEMV, "_fp8_block_gemv_scaled_activation")
    assert dot.index("if SWIGLU:") < dot.index("if QUANT_A:")
    for kernel in ("_fp8_block_gemv_dot", "_fp8_block_gemv_rows_kernel"):
        assert "_fp8_block_gemv_scaled_activation(" in _function(GEMV, kernel)
    assert "tl.minimum(a, SWIGLU_LIMIT)" in dot
    assert "tl.minimum(tl.maximum(up, -SWIGLU_LIMIT), SWIGLU_LIMIT)" in dot
    assert "(gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)" in dot
    assert "(silu * up).to(tl.bfloat16).to(tl.float32)" in dot
    fused = _function(LINEAR, "apply_swiglu_clamp_weights")
    assert "if rows > _gemv_max_m(params.weight):" in fused
    assert "self._swiglu_clamp_block_mm(" in fused
    assert "gate_up.shape[-1] == 2 * params.weight.shape[1]" in fused
    assert "swiglu_limit=swiglu_limit" in fused
    hook = _function(LINEAR_LAYER, "forward_swiglu_clamp")
    assert hook.index("_block_gemv_fp8(self.quant_method)") < hook.index(
        "_deepgemm_block_fp8("
    )
    assert "tensor_model_parallel_all_reduce(output_parallel)" in hook


def test_measured_decode_tables_cover_the_v41_shapes() -> None:
    source = GEMV.read_text(encoding="utf-8")
    for key in (
        "(576, 5120, True)",
        "(576, 5120, False)",
        "(1792, 5120, False)",
        "(5120, 288, True)",
    ):
        assert key in source
    launch = _function(GEMV, "_launch")
    assert "_tile(n, k, has_experts, out_rows)" in launch
    assert "None if has_experts or sum_topk else _rows_tile(n, k, rows)" in launch
    assert "measured is not None and (routed or rows == 1)" in _function(GEMV, "_tile")
    split = _function(MHC, "select_mhc_prenorm_split_k")
    assert split.index("if hc_hidden_size == 16384:") < split.index(
        "if hc_hidden_size == 20480:"
    )
    v41 = split[split.index("if hc_hidden_size == 20480:") :]
    assert "return 40" in v41 and "return 20" in v41


def test_delayed_mhc_decode_fuse_needs_carried_pre_mix_and_norm() -> None:
    gate = _function(MHC, "_mhc_pre_delayed_decode_norm_fuse_supported")
    assert "pre_mix is None or norm_weight is None" in gate
    assert "hidden_size % 1024 == 0" in gate
    assert "num_tokens <= _MHC_PRE_DECODE_NORM_FUSE_MAX_TOKENS" in gate
    entry = _function(MHC, "mhc_pre_delayed_musa")
    assert entry.index("_mhc_pre_delayed_decode_norm_fuse(") < entry.index(
        "_mhc_pre_delayed_big_fuse("
    )


def test_clamped_swiglu_oot_uses_the_fused_kernel() -> None:
    source = ACTIVATION.read_text(encoding="utf-8")
    assert "@SiluAndMulWithClamp.register_oot" in source
    forward = source[source.index("class MusaSiluAndMulWithClamp") :]
    assert "torch.ops._C.silu_and_mul_with_clamp(" in forward
    assert "out, x, self.swiglu_limit, self.alpha, self.beta" in forward


def test_sub_block_gemm_keeps_one_scale_per_group_and_measured_shapes_only() -> None:
    kernel = _function(GEMM, "_fp8_block_scaled_mm_kernel")
    assert "for j in tl.static_range(BLOCK_K // GROUP_K):" in kernel
    assert "acc += tl.dot(a, b) * a_s[:, None] * b_s[None, :]" in kernel
    assert "mask=g < groups" in kernel
    supported = _function(GEMM, "fp8_block_scaled_mm_supported")
    assert "fp8_block_scaled_mm_config(m, n, k) is not None" in supported
    fine = _function(EINSUM, "_try_musa_fp8_einsum_fine_blocks")
    assert fine.index("fp8_block_scaled_mm_supported(") < fine.index(
        "w8a8_triton_block_scaled_mm("
    )


def test_block32_moe_policy_is_triton_only_and_leaves_other_shapes() -> None:
    policy = POLICY.read_text(encoding="utf-8")
    assert "block_size=32," in policy
    assert '"BLOCK_SIZE_K": 32,' in policy
    assert "for shape in _DSV41_TP8_BLOCK32_SHAPES" in policy
    dispatch = _function(FUSED_MOE, "_musa_fused_experts_impl_dispatch")
    assert "or has_triton_only_calibration(" in dispatch
    assert (
        '_env_int("VLLM_MUSA_FP8_BLOCK_MOE_GEMV_MAX_TOKENS", 5)'
        in FUSED_MOE.read_text(encoding="utf-8")
    )


def test_prefill_indexer_scores_fp8_keys_before_the_fp32_fallback() -> None:
    text = _patch("0184")
    added = _added(text)
    assert "def _musa_gather_indexer_fp8_cache_quantized(" in added
    assert "fp8_mqa_logits(" in added
    assert "_musa_custom_ops.sparse_indexer_topk(" in added
    call = added.index("if _musa_try_fill_prefill_topk_from_fp8_mqa_logits(")
    assert added.index("def _musa_try_fill_prefill_topk_from_fp8_mqa_logits(") < call
    assert (
        "if q_deq is None:"
        in text[text.index("if _musa_try_fill_prefill_topk_from_fp8_mqa_logits(") :]
    )


def test_decode_indexer_scores_fp8_tiles_with_one_dot() -> None:
    text = _patch("0185")
    added = _added(text)
    removed = "\n".join(
        line[1:]
        for line in text.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    assert "scores = tl.dot(k, q)" in added
    assert "if NUM_HEADS >= 16:" in added
    assert "k = k.to(tl.float8e4nv, bitcast=True)" in added
    assert "k = k.to(tl.float8e4nv, bitcast=True).to(tl.float32)" in removed


def test_indexer_topk_up_to_512_uses_the_staged_radix_select() -> None:
    text = _patch("0186")
    added = _added(text)
    removed = "\n".join(
        line[1:]
        for line in text.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    select = added[added.index("def _musa_select_indexer_topk(") :]
    assert "if topk <= 512:" in select
    assert "_musa_custom_ops.deepseek_v4_sparse_indexer_topk_decode(" in select
    windows = added[added.index("def _musa_indexer_windows_start_at_zero(") :]
    assert "not _musa_sparse_indexer_is_current_stream_capturing()" in windows
    assert "not bool(row_starts.any())" in windows
    assert added.count("_musa_select_indexer_topk_windows(") == 4
    assert added.count("_musa_select_indexer_topk(") == 4
    assert "_musa_custom_ops.sparse_indexer_topk_decode(logits, row_lens" in select
    assert "_musa_custom_ops.sparse_indexer_topk_decode(" in removed


def test_decode_mhc_prenorm_uses_fp32_fma_partials_for_measured_shapes() -> None:
    fuse = _function(MHC, "_mhc_pre_delayed_decode_norm_fuse")
    assert fuse.index("mhc_prenorm_fma_config(") < fuse.index(
        "_mhc_prenorm_gemm_sqrsum_deepgemm("
    )
    kernel = _function(MHC_PRENORM, "_mhc_prenorm_fma_kernel")
    assert "tl.dot" not in kernel
    assert "acc += f * x[None, :]" in kernel
    source = MHC_PRENORM.read_text(encoding="utf-8")
    assert "16384: ((8, (32, 128, 8)), (12, (32, 64, 4)))" in source
    assert "20480: ((2, (40, 128, 8))" in source
    wrapper = _function(MHC_PRENORM, "mhc_prenorm_fma")
    assert "split_len % block_k == 0" in wrapper


def test_delayed_mhc_norm_fuse_sums_split_partials_in_parallel_groups() -> None:
    kernel = _function(TILELANG, "mhc_pre_delayed_decode_norm_fuse_kernel")
    assert "groups = threads // n_cols" in kernel
    assert "for split_id in T.serial(n_splits)" not in kernel
    assert "rms[0] = col_sums[hc_mult3]" in kernel
