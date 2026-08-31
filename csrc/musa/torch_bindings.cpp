#include "cache.h"
#include "cuda_utils.h"
#include "musa_ops.h"
#include "core/registration.h"

#include <torch/library.h>
#include <torch/version.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _musa_ops), musa_ops) {
#ifdef USE_MUSA
  musa_ops.def(
      "musa_fused_gemv_moe(Tensor! A, Tensor! B, Tensor! C, Tensor? A_scale, Tensor? B_scale,"
      "Tensor! topk_weights, Tensor! topk_ids, bool mul_routed_weight, int topk, bool use_int4_w4a16,"
      "bool use_swigelu, int block_n=0, int block_k=0) -> ()");
  musa_ops.impl("musa_fused_gemv_moe", torch::kMUSA, &musa_fused_gemv_moe);

  musa_ops.def(
      "musa_fused_gemv(Tensor! A, Tensor! B, Tensor! C, Tensor? A_scale, Tensor? B_scale,"
      "bool use_int4_w4a16, bool use_swigelu, bool use_rms_norm, Tensor? gamma,"
      "float eps, int block_n=0, int block_k=0) -> ()");
  musa_ops.impl("musa_fused_gemv", torch::kMUSA, &musa_fused_gemv);

  musa_ops.def(
      "musa_fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, "
      "float eps, int block_x=0) -> ()");
  musa_ops.impl("musa_fused_add_rms_norm", torch::kMUSA,
                &musa_fused_add_rms_norm);

  musa_ops.def(
      "musa_reshape_and_cache_flash_nhd(Tensor key, Tensor value, "
      "Tensor! key_cache, Tensor! value_cache, Tensor slot_mapping, "
      "int block_x=0) -> ()");
  musa_ops.impl("musa_reshape_and_cache_flash_nhd", torch::kMUSA,
                &musa_reshape_and_cache_flash_nhd);

  musa_ops.def(
      "silu_and_mul_per_token_group_fp8_quant(Tensor input, Tensor! output_q, "
      "Tensor! output_s, int group_size, float eps, float fp8_min, "
      "float fp8_max, int groups_per_block=0) -> ()");
  musa_ops.impl("silu_and_mul_per_token_group_fp8_quant", torch::kMUSA,
                &silu_and_mul_per_token_group_fp8_quant);

  musa_ops.def(
      "silu_and_mul_clamp_per_token_group_fp8_quant(Tensor input, "
      "Tensor! output_q, Tensor! output_s, int group_size, float eps, "
      "float fp8_min, float fp8_max, float swiglu_limit, "
      "int groups_per_block=0) -> ()");
  musa_ops.impl("silu_and_mul_clamp_per_token_group_fp8_quant", torch::kMUSA,
                &silu_and_mul_clamp_per_token_group_fp8_quant);

  musa_ops.def(
      "fused_add_rms_norm_per_token_group_fp8_quant(Tensor input, "
      "Tensor residual, Tensor weight, Tensor! residual_out, "
      "Tensor! output_q, Tensor! output_scale, float epsilon) -> ()");
  musa_ops.impl("fused_add_rms_norm_per_token_group_fp8_quant", torch::kMUSA,
                &fused_add_rms_norm_per_token_group_fp8_quant);

  musa_ops.def(
      "per_token_group_quant_8bit_vec(Tensor input, Tensor! output_q, "
      "Tensor! output_s, int group_size, float eps, float min_8bit, "
      "float max_8bit, int groups_per_block=0) -> ()");
  musa_ops.impl("per_token_group_quant_8bit_vec", torch::kMUSA,
                &per_token_group_quant_8bit_vec);
  musa_ops.def(
      "musa_top_k_top_p_sampling_from_probs(Tensor probs, Tensor! output, Tensor? maybe_indices, Tensor? "
      "maybe_top_k_arr, "
      "float top_k_val, Tensor? maybe_top_p_arr, float top_p_val, bool deterministic, Generator? gen) -> ()");
  musa_ops.impl("musa_top_k_top_p_sampling_from_probs", torch::kMUSA, &musa_top_k_top_p_sampling_from_probs);

  musa_ops.def(
      "musa_chunked_min_p_sampling_from_probs(Tensor probs, Tensor! output, "
      "Tensor? maybe_indices, Tensor? maybe_min_p_arr, float min_p_val, "
      "bool deterministic, Generator? gen) -> ()");
  musa_ops.impl("musa_chunked_min_p_sampling_from_probs", torch::kMUSA,
                &musa_chunked_min_p_sampling_from_probs);

/*
* From FlashInfer
*/

  musa_ops.def("top_k_renorm_probs(Tensor probs, Tensor! renorm_probs, Tensor? maybe_top_k_arr, int top_k_val) -> ()");
  musa_ops.impl("top_k_renorm_probs", torch::kMUSA, &top_k_renorm_probs);

  musa_ops.def(
      "musa_rubymine_top_k_renorm_probs(Tensor probs, Tensor! renorm_probs, "
      "int top_k_val) -> ()");
  musa_ops.impl("musa_rubymine_top_k_renorm_probs", torch::kMUSA,
                &musa_rubymine_top_k_renorm_probs);

  musa_ops.def("top_p_renorm_probs(Tensor probs, Tensor! renorm_probs, Tensor? maybe_top_p_arr, float top_p_val) -> ()");
  musa_ops.impl("top_p_renorm_probs", torch::kMUSA, &top_p_renorm_probs);

  musa_ops.def(
      "min_p_sampling_from_probs(Tensor probs, Tensor! output, Tensor? maybe_indices, Tensor? maybe_min_p_arr, float "
      "min_p_val, bool deterministic, Generator? gen) -> ()");
  musa_ops.impl("min_p_sampling_from_probs", torch::kMUSA, &min_p_sampling_from_probs);

  musa_ops.def(
      "top_p_sampling_from_probs(Tensor probs, Tensor! output, Tensor? maybe_indices, Tensor? maybe_top_p_arr, "
      "float top_p_val, bool deterministic, Generator? gen) -> ()");
  musa_ops.impl("top_p_sampling_from_probs", torch::kMUSA, &top_p_sampling_from_probs);

  musa_ops.def(
      "deepseek_v4_store_sparse_kv(Tensor normed, Tensor! kv_cache, "
      "Tensor slot_mapping, Tensor write_mask) -> ()");
  musa_ops.impl("deepseek_v4_store_sparse_kv", torch::kMUSA,
                &deepseek_v4_store_sparse_kv);

  musa_ops.def(
      "deepseek_v4_qnorm_rope_kv_insert(Tensor! q, Tensor kv, "
      "Tensor! kv_cache, Tensor slot_mapping, Tensor positions, "
      "Tensor cos_sin_cache, float eps, int cache_block_size) -> ()");
  musa_ops.impl("deepseek_v4_qnorm_rope_kv_insert", torch::kMUSA,
                &deepseek_v4_qnorm_rope_kv_insert);

  musa_ops.def(
      "deepseek_v4_c4_indexer_compress_cache(Tensor state_cache, Tensor "
      "token_to_req_indices, Tensor positions, Tensor state_slot_mapping, "
      "Tensor block_table, Tensor rms_norm_weight, Tensor cos_sin_cache, "
      "Tensor! kv_cache, Tensor kv_slot_mapping, float rms_eps, int "
      "state_block_size, int state_width, int kv_block_size) -> ()");
  musa_ops.impl("deepseek_v4_c4_indexer_compress_cache", torch::kMUSA,
                &deepseek_v4_c4_indexer_compress_cache);

  musa_ops.def(
      "deepseek_v4_fused_q_kv_rmsnorm(Tensor q, Tensor kv, Tensor q_weight, "
      "Tensor kv_weight, float eps) -> (Tensor, Tensor)");
  musa_ops.impl("deepseek_v4_fused_q_kv_rmsnorm", torch::kMUSA,
                &deepseek_v4_fused_q_kv_rmsnorm);

  musa_ops.def(
      "deepseek_v4_dequantize_and_gather_k_cache(Tensor! out, Tensor k_cache, "
      "Tensor seq_lens, Tensor? gather_lens, Tensor block_table, int "
      "block_size, int offset) -> ()");
  musa_ops.impl("deepseek_v4_dequantize_and_gather_k_cache", torch::kMUSA,
                &deepseek_v4_dequantize_and_gather_k_cache);

  musa_ops.def(
      "deepseek_v4_compute_global_topk_indices_and_lens(Tensor topk_indices, "
      "Tensor token_to_req_indices, Tensor block_table, int block_size, "
      "Tensor is_valid_token) -> (Tensor, Tensor)");
  musa_ops.impl("deepseek_v4_compute_global_topk_indices_and_lens",
                torch::kMUSA,
                &deepseek_v4_compute_global_topk_indices_and_lens);

  musa_ops.def(
      "deepseek_v4_combine_topk_swa_indices(Tensor topk_indices, Tensor "
      "query_start_loc, Tensor seq_lens, Tensor gather_lens, int window_size, "
      "int compress_ratio, int topk, int M, int N) -> (Tensor, Tensor)");
  musa_ops.impl("deepseek_v4_combine_topk_swa_indices", torch::kMUSA,
                &deepseek_v4_combine_topk_swa_indices);

  musa_ops.def(
      "deepseek_v4_indexer_topk_decode(Tensor q_quant, Tensor kv_cache, "
      "Tensor weights, Tensor seq_lens, Tensor block_table, Tensor! "
      "topk_indices, int topk) -> ()");
  musa_ops.impl("deepseek_v4_indexer_topk_decode", torch::kMUSA,
                &deepseek_v4_indexer_topk_decode);

  musa_ops.def(
      "deepseek_v4_indexer_topk_prefill(Tensor q_quant, Tensor kv_cache, "
      "Tensor weights, Tensor block_table, Tensor cu_seq_lens, Tensor "
      "token_to_seq, Tensor cu_seqlen_ks, Tensor cu_seqlen_ke, Tensor! "
      "topk_indices, int topk) -> ()");
  musa_ops.impl("deepseek_v4_indexer_topk_prefill", torch::kMUSA,
                &deepseek_v4_indexer_topk_prefill);

  musa_ops.def(
      "deepseek_v4_indexer_rerank_prefill(Tensor q_quant, Tensor kv_cache, "
      "Tensor weights, Tensor block_table, Tensor cu_seq_lens, Tensor "
      "token_to_seq, Tensor cu_seqlen_ks, Tensor cu_seqlen_ke, Tensor "
      "candidate_abs_indices, Tensor! topk_indices, int topk) -> ()");
  musa_ops.impl("deepseek_v4_indexer_rerank_prefill", torch::kMUSA,
                &deepseek_v4_indexer_rerank_prefill);

  musa_ops.def(
      "sparse_indexer_fill_all(Tensor lengths, Tensor! topk_indices, int topk) "
      "-> ()");
  musa_ops.impl("sparse_indexer_fill_all", torch::kMUSA,
                &sparse_indexer_fill_all);

  musa_ops.def(
      "sparse_indexer_topk(Tensor logits, Tensor row_starts, Tensor row_ends, "
      "Tensor! topk_indices, int topk) -> ()");
  musa_ops.impl("sparse_indexer_topk", torch::kMUSA,
                &sparse_indexer_topk);

  musa_ops.def(
      "sparse_indexer_topk_decode(Tensor logits, Tensor seq_lens, Tensor! "
      "topk_indices, int topk) -> ()");
  musa_ops.impl("sparse_indexer_topk_decode", torch::kMUSA,
                &sparse_indexer_topk_decode);

  musa_ops.def(
      "glm52_indexer_topk_decode(Tensor q_quant, Tensor kv_cache, Tensor "
      "weights, Tensor seq_lens, Tensor block_table, Tensor! topk_indices, "
      "int topk) -> ()");
  musa_ops.impl("glm52_indexer_topk_decode", torch::kMUSA,
                &glm52_indexer_topk_decode);

  musa_ops.def(
      "glm52_indexer_topk_prefill(Tensor q_quant, Tensor kv_cache, Tensor "
      "weights, Tensor block_table, Tensor cu_seq_lens, Tensor token_to_seq, "
      "Tensor cu_seqlen_ks, Tensor cu_seqlen_ke, Tensor! topk_indices, int "
      "topk) -> ()");
  musa_ops.impl("glm52_indexer_topk_prefill", torch::kMUSA,
                &glm52_indexer_topk_prefill);

  musa_ops.def(
      "deepseek_v4_sparse_flashmla_decode(Tensor q, Tensor k_cache, "
      "Tensor indices, Tensor? topk_length, Tensor? attn_sink, "
      "Tensor? extra_k_cache, Tensor? extra_indices, Tensor? "
      "extra_topk_length, Tensor! out, float softmax_scale) -> "
      "(Tensor, Tensor)");
  musa_ops.impl("deepseek_v4_sparse_flashmla_decode", torch::kMUSA,
                &deepseek_v4_sparse_flashmla_decode);

  musa_ops.def(
      "deepseek_v4_fused_inv_rope_fp8_quant(Tensor o, Tensor positions, "
      "Tensor cos_sin_cache, int n_groups, int heads_per_group, int nope_dim, "
      "int rope_dim, int quant_group_size, bool tma_aligned_scales) -> "
      "(Tensor, Tensor)");
  musa_ops.impl("deepseek_v4_fused_inv_rope_fp8_quant", torch::kMUSA,
                &deepseek_v4_fused_inv_rope_fp8_quant);

  musa_ops.def(
      "deepseek_v4_topk_softplus_sqrt(Tensor! topk_weights, Tensor! "
      "topk_indices, Tensor! token_expert_indices, Tensor gating_output, bool "
      "renormalize, float routed_scaling_factor, Tensor? correction_bias, "
      "Tensor? input_ids, Tensor? hash_indices_table) -> ()");
  musa_ops.impl("deepseek_v4_topk_softplus_sqrt", torch::kMUSA,
                &deepseek_v4_topk_softplus_sqrt);

  musa_ops.def(
      "deepseek_v4_mhc_pre(Tensor residual, Tensor fn, Tensor hc_scale, "
      "Tensor hc_base, Tensor! post_mix, Tensor! comb_mix, Tensor! "
      "layer_input, float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps, "
      "float hc_post_mult_value, int sinkhorn_repeat) -> ()");
  musa_ops.impl("deepseek_v4_mhc_pre", torch::kMUSA,
                &deepseek_v4_mhc_pre);

#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
