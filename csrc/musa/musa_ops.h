#include <optional>
#include <tuple>
#include <torch/library.h>

#include "core/scalar_type.hpp"

#include <vector>

void musa_fused_gemv_moe(
    torch::Tensor &A,
    torch::Tensor &B,
    torch::Tensor &C,
    const c10::optional<torch::Tensor> &A_scale,
    const c10::optional<torch::Tensor> &B_scale,
    torch::Tensor &topk_weights,
    torch::Tensor &topk_ids,
    bool mul_routed_weight,
    int64_t topk,
    bool use_int4_w4a16,
    bool use_swigelu,
    int64_t block_n,
    int64_t block_k);

void musa_fused_gemv(
    torch::Tensor &A,
    torch::Tensor &B,
    torch::Tensor &C,
    const c10::optional<torch::Tensor> &A_scale,
    const c10::optional<torch::Tensor> &B_scale,
    bool use_int4_w4a16,
    bool use_swigelu,
    bool use_rms_norm,
    const c10::optional<torch::Tensor> &gamma,
    double eps);

void musa_fused_add_rms_norm(
    torch::Tensor &input,
    torch::Tensor &residual,
    torch::Tensor &weight,
    double eps,
    int64_t block_x);

void musa_reshape_and_cache_flash_nhd(
    torch::Tensor &key,
    torch::Tensor &value,
    torch::Tensor &key_cache,
    torch::Tensor &value_cache,
    torch::Tensor &slot_mapping);

void silu_and_mul_per_token_group_fp8_quant(
    const torch::Tensor& input,
    torch::Tensor& output_q, torch::Tensor& output_s,
    int64_t group_size, double eps, double fp8_min,
    double fp8_max);

void silu_and_mul_clamp_per_token_group_fp8_quant(
    const torch::Tensor& input,
    torch::Tensor& output_q, torch::Tensor& output_s,
    int64_t group_size, double eps, double fp8_min,
    double fp8_max, double swiglu_limit);

void fused_add_rms_norm_per_token_group_fp8_quant(
    const torch::Tensor& input, const torch::Tensor& residual,
    const torch::Tensor& weight, torch::Tensor& residual_out,
    torch::Tensor& output_q, torch::Tensor& output_scale, double epsilon);

void per_token_group_quant_8bit_vec(
    const torch::Tensor& input,
    torch::Tensor& output_q, torch::Tensor& output_s,
    int64_t group_size, double eps, double min_8bit,
    double max_8bit);

void musa_top_k_top_p_sampling_from_probs(
    at::Tensor probs,
    at::Tensor output,
    std::optional<at::Tensor> maybe_indices,
    std::optional<at::Tensor> maybe_top_k_arr,
    double top_k_val,
    std::optional<at::Tensor> maybe_top_p_arr,
    double top_p_val,
    bool deterministic,
    std::optional<at::Generator> gen_);

void musa_chunked_min_p_sampling_from_probs(
    at::Tensor probs, at::Tensor output,
    std::optional<at::Tensor> maybe_indices,
    std::optional<at::Tensor> maybe_min_p_arr, double min_p_val,
    bool deterministic, std::optional<at::Generator> gen_);

    /*
* From FlashInfer
*/
void min_p_sampling_from_probs(at::Tensor probs, at::Tensor output,
                               std::optional<at::Tensor> maybe_indices,
                               std::optional<at::Tensor> maybe_min_p_arr, double min_p_val,
                               bool deterministic, std::optional<at::Generator> gen_);

void top_p_sampling_from_probs(at::Tensor probs, at::Tensor output,
                               std::optional<at::Tensor> maybe_indices,
                               std::optional<at::Tensor> maybe_top_p_arr, double top_p_val,
                               bool deterministic, std::optional<at::Generator> gen_);

void top_p_renorm_probs(at::Tensor probs, at::Tensor renorm_probs,
                        std::optional<at::Tensor> maybe_top_p_arr, double top_p_val);

void top_k_renorm_probs(at::Tensor probs, at::Tensor renorm_probs,
                        std::optional<at::Tensor> maybe_top_k_arr, int64_t top_k_val);

void musa_rubymine_top_k_renorm_probs(at::Tensor probs,
                                      at::Tensor renorm_probs,
                                      int64_t top_k_val);

void deepseek_v4_store_sparse_kv(
    const torch::Tensor& normed,
    torch::Tensor& kv_cache,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& write_mask);
void deepseek_v4_qnorm_rope_kv_insert(
    torch::Tensor& q,
    const torch::Tensor& kv,
    torch::Tensor& kv_cache,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin_cache,
    double eps,
    int64_t cache_block_size);
void deepseek_v4_c4_indexer_compress_cache(
    const torch::Tensor& state_cache,
    const torch::Tensor& token_to_req_indices,
    const torch::Tensor& positions,
    const torch::Tensor& state_slot_mapping,
    const torch::Tensor& block_table,
    const torch::Tensor& rms_norm_weight,
    const torch::Tensor& cos_sin_cache,
    torch::Tensor& kv_cache,
    const torch::Tensor& kv_slot_mapping,
    double rms_eps,
    int64_t state_block_size,
    int64_t state_width,
    int64_t kv_block_size);
std::tuple<torch::Tensor, torch::Tensor> deepseek_v4_fused_q_kv_rmsnorm(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& q_weight,
    const torch::Tensor& kv_weight,
    double eps);
void deepseek_v4_dequantize_and_gather_k_cache(
    torch::Tensor &out, const torch::Tensor &k_cache,
    const torch::Tensor &seq_lens,
    const c10::optional<torch::Tensor> &gather_lens,
    const torch::Tensor &block_table, int64_t block_size, int64_t offset);
std::tuple<torch::Tensor, torch::Tensor>
deepseek_v4_compute_global_topk_indices_and_lens(
    const torch::Tensor &topk_indices,
    const torch::Tensor &token_to_req_indices, const torch::Tensor &block_table,
    int64_t block_size, const torch::Tensor &is_valid_token);
std::tuple<torch::Tensor, torch::Tensor> deepseek_v4_combine_topk_swa_indices(
    const torch::Tensor &topk_indices, const torch::Tensor &query_start_loc,
    const torch::Tensor &seq_lens, const torch::Tensor &gather_lens,
    int64_t window_size, int64_t compress_ratio, int64_t topk, int64_t M,
    int64_t N);
void deepseek_v4_indexer_topk_decode(
    const torch::Tensor &q_quant, const torch::Tensor &kv_cache,
    const torch::Tensor &weights, const torch::Tensor &seq_lens,
    const torch::Tensor &block_table, torch::Tensor &topk_indices,
    int64_t topk);
void deepseek_v4_indexer_topk_prefill(
    const torch::Tensor &q_quant, const torch::Tensor &kv_cache,
    const torch::Tensor &weights, const torch::Tensor &block_table,
    const torch::Tensor &cu_seq_lens, const torch::Tensor &token_to_seq,
    const torch::Tensor &cu_seqlen_ks, const torch::Tensor &cu_seqlen_ke,
    torch::Tensor &topk_indices, int64_t topk);
void deepseek_v4_indexer_rerank_prefill(
    const torch::Tensor &q_quant, const torch::Tensor &kv_cache,
    const torch::Tensor &weights, const torch::Tensor &block_table,
    const torch::Tensor &cu_seq_lens, const torch::Tensor &token_to_seq,
    const torch::Tensor &cu_seqlen_ks, const torch::Tensor &cu_seqlen_ke,
    const torch::Tensor &candidate_abs_indices, torch::Tensor &topk_indices,
    int64_t topk);
void sparse_indexer_fill_all(const torch::Tensor &lengths,
                             torch::Tensor &topk_indices, int64_t topk);
void sparse_indexer_topk(const torch::Tensor &logits,
                         const torch::Tensor &row_starts,
                         const torch::Tensor &row_ends,
                         torch::Tensor &topk_indices, int64_t topk);
void sparse_indexer_topk_decode(const torch::Tensor &logits,
                                const torch::Tensor &seq_lens,
                                torch::Tensor &topk_indices, int64_t topk);
void glm52_indexer_topk_decode(
    const torch::Tensor &q_quant, const torch::Tensor &kv_cache,
    const torch::Tensor &weights, const torch::Tensor &seq_lens,
    const torch::Tensor &block_table, torch::Tensor &topk_indices,
    int64_t topk);
void glm52_indexer_topk_prefill(
    const torch::Tensor &q_quant, const torch::Tensor &kv_cache,
    const torch::Tensor &weights, const torch::Tensor &block_table,
    const torch::Tensor &cu_seq_lens, const torch::Tensor &token_to_seq,
    const torch::Tensor &cu_seqlen_ks, const torch::Tensor &cu_seqlen_ke,
    torch::Tensor &topk_indices, int64_t topk);
std::tuple<torch::Tensor, torch::Tensor> deepseek_v4_sparse_flashmla_decode(
    const torch::Tensor &q, const torch::Tensor &k_cache,
    const torch::Tensor &indices,
    const c10::optional<torch::Tensor> &topk_length,
    const c10::optional<torch::Tensor> &attn_sink,
    const c10::optional<torch::Tensor> &extra_k_cache,
    const c10::optional<torch::Tensor> &extra_indices,
    const c10::optional<torch::Tensor> &extra_topk_length, torch::Tensor &out,
    double softmax_scale);
std::tuple<torch::Tensor, torch::Tensor> deepseek_v4_fused_inv_rope_fp8_quant(
    const torch::Tensor &o, const torch::Tensor &positions,
    const torch::Tensor &cos_sin_cache, int64_t n_groups,
    int64_t heads_per_group, int64_t nope_dim, int64_t rope_dim,
    int64_t quant_group_size, bool tma_aligned_scales);
void deepseek_v4_topk_softplus_sqrt(
    torch::Tensor &topk_weights, torch::Tensor &topk_indices,
    torch::Tensor &token_expert_indices, const torch::Tensor &gating_output,
    bool renormalize, double routed_scaling_factor,
    const c10::optional<torch::Tensor> &correction_bias,
    const c10::optional<torch::Tensor> &input_ids,
    const c10::optional<torch::Tensor> &hash_indices_table);
void deepseek_v4_mhc_pre(
    const torch::Tensor &residual, const torch::Tensor &fn,
    const torch::Tensor &hc_scale, const torch::Tensor &hc_base,
    torch::Tensor &post_mix, torch::Tensor &comb_mix,
    torch::Tensor &layer_input, double rms_eps, double hc_pre_eps,
    double hc_sinkhorn_eps, double hc_post_mult_value,
    int64_t sinkhorn_repeat);
