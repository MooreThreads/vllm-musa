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
      "bool use_swigelu) -> ()");
  musa_ops.impl("musa_fused_gemv_moe", torch::kMUSA, &musa_fused_gemv_moe);

  musa_ops.def(
      "musa_fused_gemv(Tensor! A, Tensor! B, Tensor! C, Tensor? A_scale, Tensor? B_scale,"
      "bool use_int4_w4a16, bool use_swigelu, bool use_rms_norm, Tensor? gamma,"
      "float eps) -> ()");
  musa_ops.impl("musa_fused_gemv", torch::kMUSA, &musa_fused_gemv);

  musa_ops.def(
      "musa_fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, "
      "float eps) -> ()");
  musa_ops.impl("musa_fused_add_rms_norm", torch::kMUSA,
                &musa_fused_add_rms_norm);

  musa_ops.def(
      "musa_reshape_and_cache_flash_nhd(Tensor key, Tensor value, "
      "Tensor! key_cache, Tensor! value_cache, Tensor slot_mapping) -> ()");
  musa_ops.impl("musa_reshape_and_cache_flash_nhd", torch::kMUSA,
                &musa_reshape_and_cache_flash_nhd);

  musa_ops.def(
      "per_token_group_fp8_quant(Tensor input, Tensor! output_q, Tensor! "
      "output_s, "
      "int group_size, float eps, float fp8_min, float fp8_max, bool "
      "scale_ue8m0, bool dummy_is_scale_transposed, bool dummy_is_tma_aligned "
      ") -> ()");
  musa_ops.impl("per_token_group_fp8_quant", torch::kMUSA,
           &per_token_group_quant_fp8);

  musa_ops.def(
      "silu_and_mul_per_token_group_fp8_quant(Tensor input, Tensor! output_q, "
      "Tensor! output_s, int group_size, float eps, float fp8_min, "
      "float fp8_max) -> ()");
  musa_ops.impl("silu_and_mul_per_token_group_fp8_quant", torch::kMUSA,
                &silu_and_mul_per_token_group_fp8_quant);
  musa_ops.def(
      "musa_top_k_top_p_sampling_from_probs(Tensor probs, Tensor! output, Tensor? maybe_indices, Tensor? "
      "maybe_top_k_arr, "
      "float top_k_val, Tensor? maybe_top_p_arr, float top_p_val, bool deterministic, Generator? gen) -> ()");
  musa_ops.impl("musa_top_k_top_p_sampling_from_probs", torch::kMUSA, &musa_top_k_top_p_sampling_from_probs);

/*
* From FlashInfer
*/

  musa_ops.def("top_k_renorm_probs(Tensor probs, Tensor! renorm_probs, Tensor? maybe_top_k_arr, int top_k_val) -> ()");
  musa_ops.impl("top_k_renorm_probs", torch::kMUSA, &top_k_renorm_probs);

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
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
