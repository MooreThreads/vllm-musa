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
      "per_token_group_fp8_quant(Tensor input, Tensor! output_q, Tensor! "
      "output_s, "
      "int group_size, float eps, float fp8_min, float fp8_max, bool "
      "scale_ue8m0, bool dummy_is_scale_transposed, bool dummy_is_tma_aligned "
      ") -> ()");
  musa_ops.impl("per_token_group_fp8_quant", torch::kMUSA,
           &per_token_group_quant_fp8);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
