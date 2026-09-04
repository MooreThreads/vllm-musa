#include <ATen/cuda/CUDAContext.h>

#include <cmath>
#include <cuda_fp8.h>
#include <torch/all.h>

#include "dispatch_utils.h"

// Warp-segment max reduce within a 16-thread group (THREADS_PER_GROUP=16).
__device__ __forceinline__ float GroupReduceMax(float val) {
  unsigned mask = threadIdx.x % 32 >= 16 ? 0xffff0000 : 0x0000ffff;

  val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
  return val;
}

inline int GetGroupsPerBlock(int64_t num_groups) {
  if (num_groups % 16 == 0) {
    return 16;
  }
  if (num_groups % 8 == 0) {
    return 8;
  }
  if (num_groups % 4 == 0) {
    return 4;
  }
  if (num_groups % 2 == 0) {
    return 2;
  }
  return 1;
}

template <typename T, typename DST_DTYPE, int GROUP_SIZE, bool APPLY_CLAMP>
__global__ void silu_and_mul_per_token_group_fp8_quant_kernel(
    const T* __restrict__ input, void* __restrict__ output_q,
    float* __restrict__ output_s, const int hidden, const int num_groups,
    const int groups_per_block, const float eps, const float min_8bit,
    const float max_8bit, const float swiglu_limit) {
  constexpr int THREADS_PER_GROUP = 16;
  constexpr int ELEMS_PER_THREAD = GROUP_SIZE / THREADS_PER_GROUP;

  const int local_group_id = threadIdx.x / THREADS_PER_GROUP;
  const int lane_id = threadIdx.x % THREADS_PER_GROUP;
  const int global_group_id = blockIdx.x * groups_per_block + local_group_id;
  if (global_group_id >= num_groups) {
    return;
  }

  const int groups_per_row = hidden / GROUP_SIZE;
  const int row = global_group_id / groups_per_row;
  const int group = global_group_id - row * groups_per_row;
  const int input_stride = hidden * 2;
  const int64_t input_base =
      static_cast<int64_t>(row) * input_stride + group * GROUP_SIZE;
  const int64_t output_base =
      static_cast<int64_t>(row) * hidden + group * GROUP_SIZE;

  float values[ELEMS_PER_THREAD];
  bool quantize_to_min[ELEMS_PER_THREAD];
  float local_absmax = eps;

#pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
    const int col = lane_id * ELEMS_PER_THREAD + i;
    float gate = static_cast<float>(input[input_base + col]);
    float up = static_cast<float>(input[input_base + hidden + col]);
    if constexpr (APPLY_CLAMP) {
      // The native path materializes a canonical BF16/FP16 NaN for a NaN
      // operand or for (-inf * sigmoid(-inf)).  With -ffast-math, keeping the
      // whole chain in one kernel can otherwise fold 0*NaN to zero.  Record
      // the exceptional result from IEEE bit patterns before doing arithmetic;
      // the reference quantizer maps that NaN through fmax(NaN, fp8_min) to
      // fp8_min.  Avoid isnan/isinf because fast-math may assume them false.
      const uint32_t gate_bits = __float_as_uint(gate);
      const uint32_t up_bits = __float_as_uint(up);
      const bool gate_nan =
          (gate_bits & 0x7fffffffu) > 0x7f800000u;
      const bool gate_neg_inf = gate_bits == 0xff800000u;
      const bool up_nan = (up_bits & 0x7fffffffu) > 0x7f800000u;
      quantize_to_min[i] = gate_nan || gate_neg_inf || up_nan;

      // SiluAndMulWithClamp materializes BF16/FP16 clamp outputs. Round the
      // scalar limit to the input dtype before applying it, and use ordered
      // comparisons so NaNs remain NaNs just as torch.clamp does.
      const float rounded_limit =
          static_cast<float>(static_cast<T>(swiglu_limit));
      gate = gate > rounded_limit ? rounded_limit : gate;
      up = up < -rounded_limit
               ? -rounded_limit
               : (up > rounded_limit ? rounded_limit : up);
    }
    T rounded;
    if constexpr (APPLY_CLAMP) {
      // MUSA dispatches SiluAndMulWithClamp through its OOT forward_native
      // implementation. Its sigmoid and two multiplies each materialize the
      // input dtype, so preserve those boundaries before FP8 absmax/packing.
      const T sigmoid_rounded =
          static_cast<T>(1.0f / (1.0f + expf(-gate)));
      const T silu_rounded =
          static_cast<T>(gate * static_cast<float>(sigmoid_rounded));
      rounded = static_cast<T>(static_cast<float>(silu_rounded) * up);
    } else {
      quantize_to_min[i] = false;
      const float silu = gate / (1.0f + expf(-gate));
      rounded = static_cast<T>(silu * up);
    }
    const float value = static_cast<float>(rounded);
    values[i] = value;
    local_absmax = fmaxf(local_absmax, fabsf(value));
  }

  const float group_absmax = GroupReduceMax(local_absmax);
  const float scale = group_absmax / max_8bit;

  if (lane_id == 0) {
    output_s[global_group_id] = scale;
  }

  DST_DTYPE* out = static_cast<DST_DTYPE*>(output_q);
#pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
    const int col = lane_id * ELEMS_PER_THREAD + i;
    float q = quantize_to_min[i] ? min_8bit : values[i] / scale;
    q = fminf(fmaxf(q, min_8bit), max_8bit);
    out[output_base + col] = DST_DTYPE(q);
  }
}

template <bool APPLY_CLAMP>
void silu_and_mul_per_token_group_fp8_quant_impl(
    const torch::Tensor& input, torch::Tensor& output_q,
    torch::Tensor& output_s, int64_t group_size, double eps, double fp8_min,
    double fp8_max, double swiglu_limit,
    int64_t requested_groups_per_block) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(output_q.is_contiguous());
  TORCH_CHECK(output_s.is_contiguous());
  TORCH_CHECK(input.dim() == 2);
  TORCH_CHECK(output_q.dim() == 2);
  TORCH_CHECK(output_s.dim() == 2);
  TORCH_CHECK(group_size == 128,
              "MUSA fused SiLU+FP8 quant currently supports group_size=128.");

  const int64_t hidden2 = input.size(1);
  TORCH_CHECK(hidden2 % 2 == 0, "input last dimension must be 2 * hidden.");
  const int64_t hidden = hidden2 / 2;
  TORCH_CHECK(hidden % group_size == 0,
              "hidden must be divisible by group_size.");
  TORCH_CHECK(output_q.size(0) == input.size(0) && output_q.size(1) == hidden);
  TORCH_CHECK(output_s.size(0) == input.size(0) &&
              output_s.size(1) == hidden / group_size);

  const int64_t num_groups = input.size(0) * (hidden / group_size);
  if (num_groups == 0) {
    return;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int THREADS_PER_GROUP = 16;
  int groups_per_block = GetGroupsPerBlock(num_groups);
  if (requested_groups_per_block != 0) {
    TORCH_CHECK(
        requested_groups_per_block == 1 || requested_groups_per_block == 2 ||
            requested_groups_per_block == 4 ||
            requested_groups_per_block == 8 ||
            requested_groups_per_block == 16,
        "groups_per_block must be one of 0,1,2,4,8,16.");
    groups_per_block = static_cast<int>(requested_groups_per_block);
  }
  const int num_blocks = (num_groups + groups_per_block - 1) / groups_per_block;
  const int num_threads = groups_per_block * THREADS_PER_GROUP;
  auto dst_type = output_q.scalar_type();

#define LAUNCH_SILU_QUANT_KERNEL(T, DST_DTYPE)                            \
  do {                                                                    \
    silu_and_mul_per_token_group_fp8_quant_kernel<T, DST_DTYPE, 128,      \
                                                    APPLY_CLAMP>           \
        <<<num_blocks, num_threads, 0, stream>>>(                         \
            static_cast<const T*>(input.data_ptr()), output_q.data_ptr(), \
            static_cast<float*>(output_s.data_ptr()),                     \
            static_cast<int>(hidden), static_cast<int>(num_groups),       \
            groups_per_block, static_cast<float>(eps),                    \
            static_cast<float>(fp8_min), static_cast<float>(fp8_max),     \
            static_cast<float>(swiglu_limit));                            \
  } while (0)

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "silu_and_mul_per_token_group_fp8_quant", ([&] {
        if (dst_type == at::ScalarType::Float8_e4m3fn) {
          LAUNCH_SILU_QUANT_KERNEL(scalar_t, __nv_fp8_e4m3);
        } else if (dst_type == at::ScalarType::Char) {
          LAUNCH_SILU_QUANT_KERNEL(scalar_t, int8_t);
        } else {
          TORCH_CHECK(false,
                      "silu_and_mul_per_token_group_fp8_quant only supports "
                      "FP8/INT8 outputs.");
        }
      }));

#undef LAUNCH_SILU_QUANT_KERNEL
}

void silu_and_mul_per_token_group_fp8_quant(
    const torch::Tensor& input, torch::Tensor& output_q,
    torch::Tensor& output_s, int64_t group_size, double eps, double fp8_min,
    double fp8_max, int64_t groups_per_block) {
  silu_and_mul_per_token_group_fp8_quant_impl<false>(
      input, output_q, output_s, group_size, eps, fp8_min, fp8_max, 0.0,
      groups_per_block);
}

void silu_and_mul_clamp_per_token_group_fp8_quant(
    const torch::Tensor& input, torch::Tensor& output_q,
    torch::Tensor& output_s, int64_t group_size, double eps, double fp8_min,
    double fp8_max, double swiglu_limit, int64_t groups_per_block) {
  TORCH_CHECK(std::isfinite(swiglu_limit) && swiglu_limit > 0.0,
              "swiglu_limit must be finite and positive.");
  silu_and_mul_per_token_group_fp8_quant_impl<true>(
      input, output_q, output_s, group_size, eps, fp8_min, fp8_max,
      swiglu_limit, groups_per_block);
}
