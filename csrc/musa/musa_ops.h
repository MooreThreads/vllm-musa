#include <optional>
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
    bool use_swigelu);

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
    double eps);

void musa_reshape_and_cache_flash_nhd(
    torch::Tensor &key,
    torch::Tensor &value,
    torch::Tensor &key_cache,
    torch::Tensor &value_cache,
    torch::Tensor &slot_mapping);

void per_token_group_quant_fp8(
    const torch::Tensor& input,
    torch::Tensor& output_q, torch::Tensor& output_s,
    int64_t group_size, double eps, double fp8_min,
    double fp8_max, bool scale_ue8m0,
    bool dummy_is_scale_transposed = false,
    bool dummy_is_tma_aligned = false);

void silu_and_mul_per_token_group_fp8_quant(
    const torch::Tensor& input,
    torch::Tensor& output_q, torch::Tensor& output_s,
    int64_t group_size, double eps, double fp8_min,
    double fp8_max);

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
