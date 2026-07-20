#include <cmath>
#include <cstdint>
#include <sstream>
#include <type_traits>

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_fp8.h>
#include <musa_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/musa/device_guard.h>
#include <tvm/ffi/function.h>

#include "../common.h"
#include "../device_utils.h"

// Copied and modified from DeepEP
__forceinline__ __device__ float fast_pow2(int x) {
  // We can ensure `-126 <= x and x <= 127`
  uint32_t bits_x = (x + 127) << 23;
  return *reinterpret_cast<float*>(&bits_x);
}

// Copied and modified from DeepEP
__forceinline__ __device__ int fast_log2_ceil(float x) {
  auto bits_x = *reinterpret_cast<uint32_t*>(&x);
  auto exp_x = (bits_x >> 23) & 0xff;
  auto man_bits = bits_x & ((1 << 23) - 1);
  return exp_x - 127 + (man_bits != 0);
}

// Copied and modified from DeepEP
template <bool ROUND_SCALE, typename dtype_info>
__forceinline__ __device__ void calculate_fp8_scales(float amax, float& scale, float& scale_inv) {
  constexpr float MAX_8BIT_INV = 1.0f / dtype_info::MAX;
  if constexpr (ROUND_SCALE) {
    auto exp_scale_inv = fast_log2_ceil(amax * MAX_8BIT_INV);
    scale = fast_pow2(-exp_scale_inv);
    scale_inv = fast_pow2(exp_scale_inv);
  } else {
    scale_inv = amax * MAX_8BIT_INV;
    scale = dtype_info::MAX / amax;
  }
}

// Copied and modified from DeepEP
template <bool SCALE_UE8M0, typename OUT_DTYPE_T = std::conditional_t<SCALE_UE8M0, uint8_t, float>>
__forceinline__ __device__ OUT_DTYPE_T extract_required_scale_format(float value) {
  if constexpr (SCALE_UE8M0) {
    return static_cast<uint8_t>((*reinterpret_cast<uint32_t*>(&value)) >> 23);
  } else {
    return value;
  }
}

template <typename T>
struct DtypeInfo;

template <>
struct DtypeInfo<int8_t> {
  static constexpr float MIN = -128;
  static constexpr float MAX = 127;
};

template <>
struct DtypeInfo<__mt_fp8_e4m3> {
  static constexpr float MIN = -448;
  static constexpr float MAX = 448;
};

template <bool FUSE_SILU_AND_MUL>
__device__ __forceinline__ int compute_input_group_start_offset(
    int expert_idx,
    int token_idx,
    int hidden_dim_group_idx,
    int hidden_size,
    int num_tokens_per_expert,
    int group_size) {
  return expert_idx * num_tokens_per_expert * hidden_size * (FUSE_SILU_AND_MUL ? 2 : 1) +
         token_idx * hidden_size * (FUSE_SILU_AND_MUL ? 2 : 1) + hidden_dim_group_idx * group_size;
}

constexpr float LOCAL_ABSMAX_ABS = 1e-10;
constexpr uint32_t INPUT_PRIMARY_VEC_NUM_BYTES = 32;
constexpr uint32_t FP8_FAST_VEC_ELEMS = 4;
constexpr int BF16_FP8_G128_THREADS_PER_GROUP = 16;
constexpr int kSiluActivation = 0;
constexpr int kGeluActivation = 1;
constexpr int kGeluTanhActivation = 2;

__device__ __forceinline__ float fast_erf_musa(float x) {
  constexpr float kP = 0.3275911f;
  constexpr float kA1 = 0.254829592f;
  constexpr float kA2 = -0.284496736f;
  constexpr float kA3 = 1.421413741f;
  constexpr float kA4 = -1.453152027f;
  constexpr float kA5 = 1.061405429f;
  const float sign = x < 0.0f ? -1.0f : 1.0f;
  const float ax = fabsf(x);
  const float t = 1.0f / (1.0f + kP * ax);
  const float poly =
      (((((kA5 * t + kA4) * t + kA3) * t + kA2) * t + kA1) * t);
  return sign * (1.0f - poly * expf(-ax * ax));
}

__device__ __forceinline__ float gelu_exact_musa(float x) {
  return 0.5f * x * (1.0f + fast_erf_musa(x * 0.7071067811865475f));
}

__device__ __forceinline__ float gelu_tanh_musa(float x) {
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  constexpr float kCoeff = 0.044715f;
  return 0.5f * x *
      (1.0f + tanhf(kSqrt2OverPi * (x + kCoeff * x * x * x)));
}

__device__ __forceinline__ float fast_silu_musa(float x) {
  const float half_x = 0.5f * x;
  return half_x * (1.0f + tanhf(half_x));
}

__device__ __forceinline__ float apply_fused_activation(
    float x,
    int activation_type) {
  if (activation_type == kGeluActivation) {
    return gelu_exact_musa(x);
  }
  if (activation_type == kGeluTanhActivation) {
    return gelu_tanh_musa(x);
  }
  return fast_silu_musa(x);
}

struct RowScheduler {
  static void compute_exec_config(
      int threads_per_subwarp,
      int num_local_experts,
      int hidden_dim_num_groups,
      int num_groups,
      int& subwarps_per_block,
      dim3& grid,
      dim3& block) {
    subwarps_per_block = ([=]() -> int {
      if (hidden_dim_num_groups % 16 == 0) {
        return 16;
      } else if (hidden_dim_num_groups % 8 == 0) {
        return 8;
      } else if (hidden_dim_num_groups % 4 == 0) {
        return 4;
      } else if (hidden_dim_num_groups % 2 == 0) {
        return 2;
      }
      return 1;
    })();
    const int num_tokens_per_expert = num_groups / hidden_dim_num_groups;
    grid = dim3(hidden_dim_num_groups / subwarps_per_block, num_tokens_per_expert);
    block = dim3(subwarps_per_block * threads_per_subwarp);
  }

  template <bool FUSE_SILU_AND_MUL, int GROUP_SIZE, int THREADS_PER_SUBWARP, typename FUNC>
  __device__ __forceinline__ static void execute(
      const int subwarps_per_block,
      const int hidden_dim_num_groups,
      const int32_t* masked_m,
      const int num_tokens_per_expert,
      FUNC fn) {
    constexpr int expert_idx = 0;

    const int64_t subwarp_id = threadIdx.x / THREADS_PER_SUBWARP;
    const int lane_id = threadIdx.x % THREADS_PER_SUBWARP;

    const int token_idx = blockIdx.y;
    const int hidden_dim_group_idx = blockIdx.x * subwarps_per_block + subwarp_id;
    const int hidden_size = hidden_dim_num_groups * GROUP_SIZE;
    const int64_t input_group_start_offset = compute_input_group_start_offset<FUSE_SILU_AND_MUL>(
        expert_idx, token_idx, hidden_dim_group_idx, hidden_size, num_tokens_per_expert, GROUP_SIZE);

    fn(expert_idx, token_idx, hidden_dim_group_idx, lane_id, input_group_start_offset);
  }
};

struct MaskedLayoutScheduler {
  // TODO can be dynamically determined (which may be good when num rank is small)
  static constexpr int TOKEN_DIM_BLOCK_NUM_PER_EXPERT = 1024;

  static void compute_exec_config(
      int threads_per_subwarp,
      int num_local_experts,
      int hidden_dim_num_groups,
      int num_groups,
      int& subwarps_per_block,
      dim3& grid,
      dim3& block) {
    subwarps_per_block = ([=]() -> int {
      if (hidden_dim_num_groups % 16 == 0) {
        return 16;
      } else if (hidden_dim_num_groups % 8 == 0) {
        return 8;
      } else if (hidden_dim_num_groups % 4 == 0) {
        return 4;
      } else if (hidden_dim_num_groups % 2 == 0) {
        return 2;
      }
      return 1;
    })();
    TORCH_CHECK(hidden_dim_num_groups % subwarps_per_block == 0);
    grid = dim3(hidden_dim_num_groups / subwarps_per_block, TOKEN_DIM_BLOCK_NUM_PER_EXPERT, num_local_experts);
    block = dim3(subwarps_per_block * threads_per_subwarp);
  }

  template <bool FUSE_SILU_AND_MUL, int GROUP_SIZE, int THREADS_PER_SUBWARP, typename FUNC>
  __device__ __forceinline__ static void execute(
      const int subwarps_per_block,
      const int hidden_dim_num_groups,
      const int32_t* masked_m,
      const int num_tokens_per_expert,
      FUNC fn) {
    const int64_t subwarp_id = threadIdx.x / THREADS_PER_SUBWARP;
    const int lane_id = threadIdx.x % THREADS_PER_SUBWARP;

    const int expert_idx = blockIdx.z;
    const int token_idx_start = blockIdx.y;

    const int64_t hidden_dim_group_idx = blockIdx.x * subwarps_per_block + subwarp_id;

    const int curr_expert_token_num = masked_m[expert_idx];

    for (int token_idx = token_idx_start; token_idx < curr_expert_token_num;
         token_idx += TOKEN_DIM_BLOCK_NUM_PER_EXPERT) {
      const int hidden_size = hidden_dim_num_groups * GROUP_SIZE;
      const int64_t input_group_start_offset = compute_input_group_start_offset<FUSE_SILU_AND_MUL>(
          expert_idx, token_idx, hidden_dim_group_idx, hidden_size, num_tokens_per_expert, GROUP_SIZE);
      fn(expert_idx, token_idx, hidden_dim_group_idx, lane_id, input_group_start_offset);
    }
  }
};

template <
    typename SCHEDULER,
    int GROUP_SIZE,
    int THREADS_PER_SUBWARP,
    typename T,
    typename DST_DTYPE,
    bool IS_COLUMN_MAJOR = false,
    bool SCALE_UE8M0 = false,
    bool FUSE_SILU_AND_MUL = false,
    typename scale_packed_t = std::conditional_t<SCALE_UE8M0, uint32_t, float>>
__global__ void per_token_group_quant_8bit_kernel(
    const T* __restrict__ input,
    DST_DTYPE* __restrict__ output_q,
    scale_packed_t* __restrict__ output_s,
    const int32_t* __restrict__ masked_m,
    const int subwarps_per_block,
    const int hidden_dim_num_groups,
    // TODO can this be removed?
    const int scale_expert_stride,
    const int scale_hidden_stride,
    const int num_tokens_per_expert,
    const int fused_activation_type) {
  using dst_dtype_info = DtypeInfo<DST_DTYPE>;
  using scale_element_t = std::conditional_t<SCALE_UE8M0, uint8_t, float>;

  SCHEDULER::template execute<FUSE_SILU_AND_MUL, GROUP_SIZE, THREADS_PER_SUBWARP>(
      subwarps_per_block,
      hidden_dim_num_groups,
      masked_m,
      num_tokens_per_expert,
      [&](const int expert_idx,
          const int token_idx,
          const int hidden_dim_group_idx,
          const int lane_id,
          const int input_group_start_offset) {
        constexpr uint32_t INPUT_PRIMARY_VEC_SIZE = INPUT_PRIMARY_VEC_NUM_BYTES / sizeof(T);
        constexpr uint32_t INPUT_PRIMARY_INT4_SIZE = INPUT_PRIMARY_VEC_NUM_BYTES / sizeof(int4);

        const int offset_num_groups = expert_idx * num_tokens_per_expert * hidden_dim_num_groups +
                                      token_idx * hidden_dim_num_groups + hidden_dim_group_idx;

        int4 input_primary_int4[INPUT_PRIMARY_INT4_SIZE];
        T* input_primary_vec = reinterpret_cast<T*>(input_primary_int4);

        int4 input_secondary_int4[INPUT_PRIMARY_INT4_SIZE];
        T* input_secondary_vec = reinterpret_cast<T*>(input_secondary_int4);

#pragma unroll
        for (uint32_t j = 0; j < INPUT_PRIMARY_INT4_SIZE; ++j) {
          input_primary_int4[j] = ld_global_nc(
              reinterpret_cast<const int4*>(input + input_group_start_offset + lane_id * INPUT_PRIMARY_VEC_SIZE) + j);
        }
        if constexpr (FUSE_SILU_AND_MUL) {
          const int secondary_offset = hidden_dim_num_groups * GROUP_SIZE;
#pragma unroll
          for (uint32_t j = 0; j < INPUT_PRIMARY_INT4_SIZE; ++j) {
            input_secondary_int4[j] = ld_global_nc(
                reinterpret_cast<const int4*>(
                    input + input_group_start_offset + lane_id * INPUT_PRIMARY_VEC_SIZE + secondary_offset) +
                j);
          }
        }

        constexpr int num_elems_per_pack = static_cast<int>(sizeof(scale_packed_t) / sizeof(scale_element_t));
        scale_element_t* scale_output;
        if constexpr (IS_COLUMN_MAJOR) {
          constexpr int scale_token_stride = 1;

          const int hidden_idx_packed = hidden_dim_group_idx / num_elems_per_pack;
          const int pack_idx = hidden_dim_group_idx % num_elems_per_pack;
          scale_output = reinterpret_cast<scale_element_t*>(output_s) +
                         (expert_idx * scale_expert_stride * num_elems_per_pack +
                          hidden_idx_packed * scale_hidden_stride * num_elems_per_pack +
                          token_idx * scale_token_stride * num_elems_per_pack + pack_idx);
        } else {
          scale_output = output_s + offset_num_groups;
        }

        // can speed up if too slow
        if constexpr (IS_COLUMN_MAJOR and SCALE_UE8M0) {
          const int remainder_num_groups = hidden_dim_num_groups % num_elems_per_pack;
          if ((remainder_num_groups != 0) and (hidden_dim_group_idx == hidden_dim_num_groups - 1) and
              (lane_id < num_elems_per_pack - remainder_num_groups)) {
            const int shift = 1 + lane_id;
            *(scale_output + shift) = 0;
          }
        }

        float local_absmax = LOCAL_ABSMAX_ABS;

#pragma unroll
        for (uint32_t j = 0; j < INPUT_PRIMARY_VEC_SIZE; ++j) {
          float val;
          if constexpr (FUSE_SILU_AND_MUL) {
            // TODO maybe vectorize
            T val_lowprec = static_cast<T>(apply_fused_activation(
                                static_cast<float>(input_primary_vec[j]),
                                fused_activation_type)) *
                            input_secondary_vec[j];
            val = static_cast<float>(val_lowprec);
            input_primary_vec[j] = val_lowprec;
          } else {
            val = static_cast<float>(input_primary_vec[j]);
          }

          float abs_val = fabsf(val);
          local_absmax = fmaxf(local_absmax, abs_val);
        }

        local_absmax = GroupReduceMax<THREADS_PER_SUBWARP>(local_absmax, lane_id);

        float y_scale, y_scale_inv;
        calculate_fp8_scales<SCALE_UE8M0, dst_dtype_info>(local_absmax, y_scale, y_scale_inv);
        if (lane_id == 0) {
          *scale_output = extract_required_scale_format<SCALE_UE8M0>(y_scale_inv);
        }

        int4 output_buf;

        if constexpr (std::is_same_v<DST_DTYPE, __mt_fp8_e4m3>) {
          const auto output_buf_ptr = reinterpret_cast<__mt_fp8x4_storage_t*>(&output_buf);

#pragma unroll
          for (uint32_t j = 0; j < INPUT_PRIMARY_VEC_SIZE; j += 4) {
            float4 outputx4 = {
                static_cast<float>(input_primary_vec[j]) * y_scale,
                static_cast<float>(input_primary_vec[j + 1]) * y_scale,
                static_cast<float>(input_primary_vec[j + 2]) * y_scale,
                static_cast<float>(input_primary_vec[j + 3]) * y_scale};
            output_buf_ptr[j / 4] = __musa_cvt_float4_to_fp8x4(outputx4, __MT_SATFINITE, __MT_E4M3);
          }
        } else {
          const auto output_buf_ptr = reinterpret_cast<DST_DTYPE*>(&output_buf);

#pragma unroll
          for (uint32_t j = 0; j < INPUT_PRIMARY_VEC_SIZE; ++j) {
            float val = static_cast<float>(input_primary_vec[j]);
            float q_val = fminf(fmaxf(val * y_scale, dst_dtype_info::MIN), dst_dtype_info::MAX);
            output_buf_ptr[j] = DST_DTYPE(q_val);
          }
        }

        st_global(
            reinterpret_cast<int4*>(output_q + offset_num_groups * GROUP_SIZE + lane_id * INPUT_PRIMARY_VEC_SIZE),
            output_buf);
      });
}

template <typename DST_DTYPE>
__device__ __forceinline__ uint32_t pack_quant4(const float values[FP8_FAST_VEC_ELEMS], const float y_scale) {
  if constexpr (std::is_same_v<DST_DTYPE, __mt_fp8_e4m3>) {
    float4 outputx4 = {values[0] * y_scale, values[1] * y_scale, values[2] * y_scale, values[3] * y_scale};
    return __musa_cvt_float4_to_fp8x4(outputx4, __MT_SATFINITE, __MT_E4M3);
  } else {
    DST_DTYPE packed[FP8_FAST_VEC_ELEMS];
#pragma unroll
    for (uint32_t j = 0; j < FP8_FAST_VEC_ELEMS; ++j) {
      const float q_val = fminf(fmaxf(values[j] * y_scale, DtypeInfo<DST_DTYPE>::MIN), DtypeInfo<DST_DTYPE>::MAX);
      packed[j] = DST_DTYPE(q_val);
    }
    return *reinterpret_cast<uint32_t*>(packed);
  }
}

template <
    int GROUP_SIZE,
    int THREADS_PER_GROUP,
    typename DST_DTYPE,
    bool IS_COLUMN_MAJOR = false,
    bool FUSE_SILU_AND_MUL = false>
__global__ void per_token_group_quant_8bit_fast_kernel(
    const half* __restrict__ input,
    DST_DTYPE* __restrict__ output_q,
    float* __restrict__ output_s,
    const int hidden_dim_num_groups,
    const int scale_hidden_stride,
    const int num_tokens_per_expert,
    const int fused_activation_type) {
  const int subwarp_id = threadIdx.x / THREADS_PER_GROUP;
  const int lane_id = threadIdx.x - subwarp_id * THREADS_PER_GROUP;
  const int subwarps_per_block = blockDim.x / THREADS_PER_GROUP;

  const int token_idx = blockIdx.y;
  const int hidden_dim_group_idx = blockIdx.x * subwarps_per_block + subwarp_id;
  const int hidden_size = hidden_dim_num_groups * GROUP_SIZE;
  const int input_group_start_offset = compute_input_group_start_offset<FUSE_SILU_AND_MUL>(
      0, token_idx, hidden_dim_group_idx, hidden_size, num_tokens_per_expert, GROUP_SIZE);
  const int elem_offset = lane_id * FP8_FAST_VEC_ELEMS;

  uint64_t input_primary_u64 = *reinterpret_cast<const uint64_t*>(input + input_group_start_offset + elem_offset);
  half* input_primary_vec = reinterpret_cast<half*>(&input_primary_u64);

  uint64_t input_secondary_u64;
  half* input_secondary_vec = reinterpret_cast<half*>(&input_secondary_u64);
  if constexpr (FUSE_SILU_AND_MUL) {
    input_secondary_u64 = *reinterpret_cast<const uint64_t*>(input + input_group_start_offset + hidden_size + elem_offset);
  }

  float values[FP8_FAST_VEC_ELEMS];
  float local_absmax = LOCAL_ABSMAX_ABS;

#pragma unroll
  for (uint32_t j = 0; j < FP8_FAST_VEC_ELEMS; ++j) {
    float val;
    if constexpr (FUSE_SILU_AND_MUL) {
      half val_lowprec = static_cast<half>(fast_silu_musa(static_cast<float>(input_primary_vec[j]))) *
                         input_secondary_vec[j];
      val = static_cast<float>(val_lowprec);
      values[j] = val;
    } else {
      val = static_cast<float>(input_primary_vec[j]);
      values[j] = val;
    }
    local_absmax = fmaxf(local_absmax, fabsf(val));
  }

  local_absmax = GroupReduceMax<THREADS_PER_GROUP>(local_absmax, lane_id);
  const float y_scale_inv = local_absmax * (1.0f / DtypeInfo<DST_DTYPE>::MAX);
  const float y_scale = DtypeInfo<DST_DTYPE>::MAX / local_absmax;

  if (lane_id == 0) {
    if constexpr (IS_COLUMN_MAJOR) {
      output_s[token_idx + hidden_dim_group_idx * scale_hidden_stride] = y_scale_inv;
    } else {
      output_s[token_idx * hidden_dim_num_groups + hidden_dim_group_idx] = y_scale_inv;
    }
  }

  const int output_offset = (token_idx * hidden_dim_num_groups + hidden_dim_group_idx) * GROUP_SIZE + elem_offset;
  const uint32_t output_buf = pack_quant4<DST_DTYPE>(values, y_scale);
  *reinterpret_cast<uint32_t*>(output_q + output_offset) = output_buf;
}

template <int GROUP_SIZE, int THREADS_PER_GROUP, int HIDDEN_GROUPS_PER_BLOCK, int TOKEN_GROUPS_PER_BLOCK>
__global__ void per_token_group_quant_8bit_i8_fused_tiled_kernel(
    const half* __restrict__ input,
    int8_t* __restrict__ output_q,
    float* __restrict__ output_s,
    const int hidden_dim_num_groups,
    const int num_tokens_per_expert) {
  const int subwarp_id = threadIdx.x / THREADS_PER_GROUP;
  const int lane_id = threadIdx.x - subwarp_id * THREADS_PER_GROUP;
  const int token_tile_idx = subwarp_id / HIDDEN_GROUPS_PER_BLOCK;
  const int hidden_tile_idx = subwarp_id - token_tile_idx * HIDDEN_GROUPS_PER_BLOCK;

  const int token_idx = blockIdx.y * TOKEN_GROUPS_PER_BLOCK + token_tile_idx;
  const int hidden_dim_group_idx = blockIdx.x * HIDDEN_GROUPS_PER_BLOCK + hidden_tile_idx;
  if (token_idx >= num_tokens_per_expert || hidden_dim_group_idx >= hidden_dim_num_groups) {
    return;
  }

  const int hidden_size = hidden_dim_num_groups * GROUP_SIZE;
  const int input_group_start_offset = token_idx * hidden_size * 2 + hidden_dim_group_idx * GROUP_SIZE;
  const int elem_offset = lane_id * FP8_FAST_VEC_ELEMS;

  uint64_t input_primary_u64 = *reinterpret_cast<const uint64_t*>(input + input_group_start_offset + elem_offset);
  half* input_primary_vec = reinterpret_cast<half*>(&input_primary_u64);
  uint64_t input_secondary_u64 =
      *reinterpret_cast<const uint64_t*>(input + input_group_start_offset + hidden_size + elem_offset);
  half* input_secondary_vec = reinterpret_cast<half*>(&input_secondary_u64);

  float values[FP8_FAST_VEC_ELEMS];
  float local_absmax = LOCAL_ABSMAX_ABS;

#pragma unroll
  for (uint32_t j = 0; j < FP8_FAST_VEC_ELEMS; ++j) {
    half val_lowprec = static_cast<half>(fast_silu_musa(static_cast<float>(input_primary_vec[j]))) *
                       input_secondary_vec[j];
    const float val = static_cast<float>(val_lowprec);
    values[j] = val;
    local_absmax = fmaxf(local_absmax, fabsf(val));
  }

  local_absmax = GroupReduceMax<THREADS_PER_GROUP>(local_absmax, lane_id);
  const float y_scale_inv = local_absmax * (1.0f / DtypeInfo<int8_t>::MAX);
  const float y_scale = DtypeInfo<int8_t>::MAX / local_absmax;

  if (lane_id == 0) {
    output_s[token_idx * hidden_dim_num_groups + hidden_dim_group_idx] = y_scale_inv;
  }

  const int output_offset = (token_idx * hidden_dim_num_groups + hidden_dim_group_idx) * GROUP_SIZE + elem_offset;
  const uint32_t output_buf = pack_quant4<int8_t>(values, y_scale);
  *reinterpret_cast<uint32_t*>(output_q + output_offset) = output_buf;
}

template <int GROUP_SIZE, int THREADS_PER_GROUP, int HIDDEN_GROUPS_PER_BLOCK, int TOKEN_GROUPS_PER_BLOCK>
__global__ void per_token_group_quant_8bit_fp8_col_tiled_kernel(
    const half* __restrict__ input,
    __mt_fp8_e4m3* __restrict__ output_q,
    float* __restrict__ output_s,
    const int hidden_dim_num_groups,
    const int scale_hidden_stride,
    const int num_tokens_per_expert) {
  const int subwarp_id = threadIdx.x / THREADS_PER_GROUP;
  const int lane_id = threadIdx.x - subwarp_id * THREADS_PER_GROUP;
  const int token_tile_idx = subwarp_id / HIDDEN_GROUPS_PER_BLOCK;
  const int hidden_tile_idx = subwarp_id - token_tile_idx * HIDDEN_GROUPS_PER_BLOCK;

  const int token_idx = blockIdx.y * TOKEN_GROUPS_PER_BLOCK + token_tile_idx;
  const int hidden_dim_group_idx = blockIdx.x * HIDDEN_GROUPS_PER_BLOCK + hidden_tile_idx;
  if (token_idx >= num_tokens_per_expert || hidden_dim_group_idx >= hidden_dim_num_groups) {
    return;
  }

  const int hidden_size = hidden_dim_num_groups * GROUP_SIZE;
  const int input_group_start_offset =
      token_idx * hidden_size + hidden_dim_group_idx * GROUP_SIZE + lane_id * FP8_FAST_VEC_ELEMS;

  uint64_t input_primary_u64 = *reinterpret_cast<const uint64_t*>(input + input_group_start_offset);
  half* input_primary_vec = reinterpret_cast<half*>(&input_primary_u64);

  float values[FP8_FAST_VEC_ELEMS];
  float local_absmax = LOCAL_ABSMAX_ABS;

#pragma unroll
  for (uint32_t j = 0; j < FP8_FAST_VEC_ELEMS; ++j) {
    const float val = static_cast<float>(input_primary_vec[j]);
    values[j] = val;
    local_absmax = fmaxf(local_absmax, fabsf(val));
  }

  local_absmax = GroupReduceMax<THREADS_PER_GROUP>(local_absmax, lane_id);
  const float y_scale_inv = local_absmax * (1.0f / DtypeInfo<__mt_fp8_e4m3>::MAX);
  const float y_scale = DtypeInfo<__mt_fp8_e4m3>::MAX / local_absmax;

  if (lane_id == 0) {
    output_s[token_idx + hidden_dim_group_idx * scale_hidden_stride] = y_scale_inv;
  }

  const int output_offset =
      (token_idx * hidden_dim_num_groups + hidden_dim_group_idx) * GROUP_SIZE + lane_id * FP8_FAST_VEC_ELEMS;
  uint32_t output_buf;
  auto output_buf_ptr = reinterpret_cast<__mt_fp8x4_storage_t*>(&output_buf);
  float4 outputx4 = {values[0] * y_scale, values[1] * y_scale, values[2] * y_scale, values[3] * y_scale};
  output_buf_ptr[0] = __musa_cvt_float4_to_fp8x4(outputx4, __MT_SATFINITE, __MT_E4M3);
  *reinterpret_cast<uint32_t*>(output_q + output_offset) = output_buf;
}

template <int THREADS_PER_GROUP, bool FUSE_SILU_AND_MUL>
__device__ __forceinline__ void quantize_bf16_fp8_g128_group(
    const __mt_bfloat16* __restrict__ input,
    __mt_fp8_e4m3* __restrict__ output_q,
    float* __restrict__ output_s,
    const int hidden_dim_num_groups,
    const int token_idx,
    const int hidden_dim_group_idx,
    const int fused_activation_type) {
  constexpr int GROUP_SIZE = 128;
  constexpr int ELEMS_PER_THREAD = GROUP_SIZE / THREADS_PER_GROUP;

  const int hidden_size = hidden_dim_num_groups * GROUP_SIZE;
  const int lane_id = threadIdx.x % THREADS_PER_GROUP;
  const int elem_offset = lane_id * ELEMS_PER_THREAD;
  const int input_offset =
      token_idx * hidden_size * (FUSE_SILU_AND_MUL ? 2 : 1) + hidden_dim_group_idx * GROUP_SIZE + elem_offset;

  float values[ELEMS_PER_THREAD];
  if constexpr (ELEMS_PER_THREAD >= 8) {
    constexpr int INPUT_INT4_SIZE = ELEMS_PER_THREAD * sizeof(__mt_bfloat16) / sizeof(int4);
    int4 input_int4[INPUT_INT4_SIZE];
    int4 input_secondary_int4[INPUT_INT4_SIZE];
    auto input_vec = reinterpret_cast<__mt_bfloat16*>(input_int4);
    auto input_secondary_vec = reinterpret_cast<__mt_bfloat16*>(input_secondary_int4);
#pragma unroll
    for (int j = 0; j < INPUT_INT4_SIZE; ++j) {
      input_int4[j] = ld_global_nc(reinterpret_cast<const int4*>(input + input_offset) + j);
      if constexpr (FUSE_SILU_AND_MUL) {
        input_secondary_int4[j] =
            ld_global_nc(reinterpret_cast<const int4*>(input + input_offset + hidden_size) + j);
      }
    }
#pragma unroll
    for (int j = 0; j < ELEMS_PER_THREAD; ++j) {
      if constexpr (FUSE_SILU_AND_MUL) {
        __mt_bfloat16 val_lowprec = static_cast<__mt_bfloat16>(
                                        fast_silu_musa(static_cast<float>(input_vec[j]))) *
                                    input_secondary_vec[j];
        values[j] = static_cast<float>(val_lowprec);
      } else {
        values[j] = static_cast<float>(input_vec[j]);
      }
    }
  } else {
    uint64_t input_u64 = *reinterpret_cast<const uint64_t*>(input + input_offset);
    auto input_vec = reinterpret_cast<__mt_bfloat16*>(&input_u64);
    uint64_t input_secondary_u64;
    auto input_secondary_vec = reinterpret_cast<__mt_bfloat16*>(&input_secondary_u64);
    if constexpr (FUSE_SILU_AND_MUL) {
      input_secondary_u64 = *reinterpret_cast<const uint64_t*>(input + input_offset + hidden_size);
    }
#pragma unroll
    for (int j = 0; j < ELEMS_PER_THREAD; ++j) {
      if constexpr (FUSE_SILU_AND_MUL) {
        __mt_bfloat16 val_lowprec = static_cast<__mt_bfloat16>(
                                        fast_silu_musa(static_cast<float>(input_vec[j]))) *
                                    input_secondary_vec[j];
        values[j] = static_cast<float>(val_lowprec);
      } else {
        values[j] = static_cast<float>(input_vec[j]);
      }
    }
  }

  float local_absmax = LOCAL_ABSMAX_ABS;
#pragma unroll
  for (int j = 0; j < ELEMS_PER_THREAD; ++j) {
    const float val = values[j];
    local_absmax = fmaxf(local_absmax, fabsf(val));
  }

  local_absmax = GroupReduceMax<THREADS_PER_GROUP>(local_absmax, lane_id);
  const float y_scale_inv = local_absmax * (1.0f / DtypeInfo<__mt_fp8_e4m3>::MAX);
  const float y_scale = 1.0f / y_scale_inv;

  if (lane_id == 0) {
    output_s[token_idx * hidden_dim_num_groups + hidden_dim_group_idx] = y_scale_inv;
  }

  const int output_offset = (token_idx * hidden_dim_num_groups + hidden_dim_group_idx) * GROUP_SIZE + elem_offset;
  if constexpr (ELEMS_PER_THREAD == 16) {
    float4 outputx4_0 = {values[0] * y_scale, values[1] * y_scale, values[2] * y_scale, values[3] * y_scale};
    float4 outputx4_1 = {values[4] * y_scale, values[5] * y_scale, values[6] * y_scale, values[7] * y_scale};
    float4 outputx4_2 = {values[8] * y_scale, values[9] * y_scale, values[10] * y_scale, values[11] * y_scale};
    float4 outputx4_3 = {values[12] * y_scale, values[13] * y_scale, values[14] * y_scale, values[15] * y_scale};
    auto output_u32 = reinterpret_cast<uint32_t*>(output_q + output_offset);
    output_u32[0] = __musa_cvt_float4_to_fp8x4(outputx4_0, __MT_SATFINITE, __MT_E4M3);
    output_u32[1] = __musa_cvt_float4_to_fp8x4(outputx4_1, __MT_SATFINITE, __MT_E4M3);
    output_u32[2] = __musa_cvt_float4_to_fp8x4(outputx4_2, __MT_SATFINITE, __MT_E4M3);
    output_u32[3] = __musa_cvt_float4_to_fp8x4(outputx4_3, __MT_SATFINITE, __MT_E4M3);
  } else {
    const float4 outputx4_0 = {values[0] * y_scale, values[1] * y_scale, values[2] * y_scale, values[3] * y_scale};
    const uint32_t output0 = __musa_cvt_float4_to_fp8x4(outputx4_0, __MT_SATFINITE, __MT_E4M3);
    if constexpr (ELEMS_PER_THREAD == 8) {
      const float4 outputx4_1 = {values[4] * y_scale, values[5] * y_scale, values[6] * y_scale, values[7] * y_scale};
      const uint64_t output_buf =
          static_cast<uint64_t>(output0) |
          (static_cast<uint64_t>(__musa_cvt_float4_to_fp8x4(outputx4_1, __MT_SATFINITE, __MT_E4M3)) << 32);
      st_global_b64_slc_new(reinterpret_cast<uint64_t*>(output_q + output_offset), output_buf);
    } else {
      *reinterpret_cast<uint32_t*>(output_q + output_offset) = output0;
    }
  }
}

template <int THREADS_PER_GROUP, bool FUSE_SILU_AND_MUL>
__global__ void per_token_group_quant_8bit_bf16_fp8_g128_row_kernel(
    const __mt_bfloat16* __restrict__ input,
    __mt_fp8_e4m3* __restrict__ output_q,
    float* __restrict__ output_s,
    const int hidden_dim_num_groups,
    const int fused_activation_type) {
  const int subwarp_id = threadIdx.x / THREADS_PER_GROUP;
  const int subwarps_per_block = blockDim.x / THREADS_PER_GROUP;
  const int token_idx = blockIdx.y;
  const int hidden_dim_group_idx = blockIdx.x * subwarps_per_block + subwarp_id;
  quantize_bf16_fp8_g128_group<THREADS_PER_GROUP, FUSE_SILU_AND_MUL>(
      input, output_q, output_s, hidden_dim_num_groups, token_idx, hidden_dim_group_idx, fused_activation_type);
}

template <int THREADS_PER_GROUP, bool FUSE_SILU_AND_MUL, int HIDDEN_GROUPS_PER_BLOCK, int TOKEN_GROUPS_PER_BLOCK>
__global__ void per_token_group_quant_8bit_bf16_fp8_g128_row_tiled_kernel(
    const __mt_bfloat16* __restrict__ input,
    __mt_fp8_e4m3* __restrict__ output_q,
    float* __restrict__ output_s,
    const int hidden_dim_num_groups,
    const int num_tokens_per_expert,
    const int fused_activation_type) {
  const int subwarp_id = threadIdx.x / THREADS_PER_GROUP;
  const int token_tile_idx = subwarp_id / HIDDEN_GROUPS_PER_BLOCK;
  const int hidden_tile_idx = subwarp_id - token_tile_idx * HIDDEN_GROUPS_PER_BLOCK;

  const int token_idx = blockIdx.y * TOKEN_GROUPS_PER_BLOCK + token_tile_idx;
  const int hidden_dim_group_idx = blockIdx.x * HIDDEN_GROUPS_PER_BLOCK + hidden_tile_idx;
  if (token_idx >= num_tokens_per_expert || hidden_dim_group_idx >= hidden_dim_num_groups) {
    return;
  }

  quantize_bf16_fp8_g128_group<THREADS_PER_GROUP, FUSE_SILU_AND_MUL>(
      input, output_q, output_s, hidden_dim_num_groups, token_idx, hidden_dim_group_idx, fused_activation_type);
}

inline int choose_subwarps_per_block(const int hidden_dim_num_groups) {
  if (hidden_dim_num_groups % 16 == 0) {
    return 16;
  } else if (hidden_dim_num_groups % 8 == 0) {
    return 8;
  } else if (hidden_dim_num_groups % 4 == 0) {
    return 4;
  } else if (hidden_dim_num_groups % 2 == 0) {
    return 2;
  }
  return 1;
}

template <int GROUP_SIZE>
struct FastQuantConfig;

template <>
struct FastQuantConfig<16> {
  static constexpr int THREADS_PER_GROUP = 4;
};

template <>
struct FastQuantConfig<32> {
  static constexpr int THREADS_PER_GROUP = 8;
};

template <>
struct FastQuantConfig<64> {
  static constexpr int THREADS_PER_GROUP = 16;
};

template <>
struct FastQuantConfig<128> {
  static constexpr int THREADS_PER_GROUP = 32;
};

template <int GROUP_SIZE, bool IS_COLUMN_MAJOR, bool FUSE_SILU_AND_MUL>
void launch_fp8_fast_kernel(
    ffi::TensorView input,
    ffi::TensorView output_q,
    ffi::TensorView output_s,
    int hidden_dim_num_groups,
    int scale_hidden_stride,
    int num_tokens_per_expert,
    int fused_activation_type,
    musaStream_t stream) {
  constexpr int THREADS_PER_GROUP = FastQuantConfig<GROUP_SIZE>::THREADS_PER_GROUP;
  const int subwarps_per_block = choose_subwarps_per_block(hidden_dim_num_groups);
  dim3 grid(hidden_dim_num_groups / subwarps_per_block, num_tokens_per_expert);
  dim3 block(subwarps_per_block * THREADS_PER_GROUP);
  per_token_group_quant_8bit_fast_kernel<
      GROUP_SIZE,
      THREADS_PER_GROUP,
      __mt_fp8_e4m3,
      IS_COLUMN_MAJOR,
      FUSE_SILU_AND_MUL><<<grid, block, 0, stream>>>(
      static_cast<half*>(input.data_ptr()),
      static_cast<__mt_fp8_e4m3*>(output_q.data_ptr()),
      static_cast<float*>(output_s.data_ptr()),
      hidden_dim_num_groups,
      scale_hidden_stride,
      num_tokens_per_expert,
      fused_activation_type);
}

template <bool IS_COLUMN_MAJOR, bool FUSE_SILU_AND_MUL>
void dispatch_fp8_fast_kernel(
    int64_t group_size,
    ffi::TensorView input,
    ffi::TensorView output_q,
    ffi::TensorView output_s,
    int hidden_dim_num_groups,
    int scale_hidden_stride,
    int num_tokens_per_expert,
    int fused_activation_type,
    musaStream_t stream) {
  switch (group_size) {
    case 16:
      launch_fp8_fast_kernel<16, IS_COLUMN_MAJOR, FUSE_SILU_AND_MUL>(
          input, output_q, output_s, hidden_dim_num_groups, scale_hidden_stride,
          num_tokens_per_expert, fused_activation_type, stream);
      break;
    case 32:
      launch_fp8_fast_kernel<32, IS_COLUMN_MAJOR, FUSE_SILU_AND_MUL>(
          input, output_q, output_s, hidden_dim_num_groups, scale_hidden_stride,
          num_tokens_per_expert, fused_activation_type, stream);
      break;
    case 64:
      launch_fp8_fast_kernel<64, IS_COLUMN_MAJOR, FUSE_SILU_AND_MUL>(
          input, output_q, output_s, hidden_dim_num_groups, scale_hidden_stride,
          num_tokens_per_expert, fused_activation_type, stream);
      break;
    case 128:
      launch_fp8_fast_kernel<128, IS_COLUMN_MAJOR, FUSE_SILU_AND_MUL>(
          input, output_q, output_s, hidden_dim_num_groups, scale_hidden_stride,
          num_tokens_per_expert, fused_activation_type, stream);
      break;
    default:
      TORCH_CHECK(false, "Unsupported group_size");
  }
}

template <int GROUP_SIZE, int HIDDEN_GROUPS_PER_BLOCK, int TOKEN_GROUPS_PER_BLOCK>
void launch_fp8_col_tiled_kernel(
    ffi::TensorView input,
    ffi::TensorView output_q,
    ffi::TensorView output_s,
    int hidden_dim_num_groups,
    int scale_hidden_stride,
    int num_tokens_per_expert,
    musaStream_t stream) {
  constexpr int THREADS_PER_GROUP = FastQuantConfig<GROUP_SIZE>::THREADS_PER_GROUP;
  dim3 grid(
      (hidden_dim_num_groups + HIDDEN_GROUPS_PER_BLOCK - 1) / HIDDEN_GROUPS_PER_BLOCK,
      (num_tokens_per_expert + TOKEN_GROUPS_PER_BLOCK - 1) / TOKEN_GROUPS_PER_BLOCK);
  dim3 block(THREADS_PER_GROUP * HIDDEN_GROUPS_PER_BLOCK * TOKEN_GROUPS_PER_BLOCK);
  per_token_group_quant_8bit_fp8_col_tiled_kernel<
      GROUP_SIZE,
      THREADS_PER_GROUP,
      HIDDEN_GROUPS_PER_BLOCK,
      TOKEN_GROUPS_PER_BLOCK><<<grid, block, 0, stream>>>(
      static_cast<half*>(input.data_ptr()),
      static_cast<__mt_fp8_e4m3*>(output_q.data_ptr()),
      static_cast<float*>(output_s.data_ptr()),
      hidden_dim_num_groups,
      scale_hidden_stride,
      num_tokens_per_expert);
}

void dispatch_fp8_col_nonfused_kernel(
    int64_t group_size,
    ffi::TensorView input,
    ffi::TensorView output_q,
    ffi::TensorView output_s,
    int hidden_dim_num_groups,
    int scale_hidden_stride,
    int num_tokens_per_expert,
    musaStream_t stream) {
  if (group_size == 32) {
    launch_fp8_col_tiled_kernel<32, 4, 4>(
        input, output_q, output_s, hidden_dim_num_groups, scale_hidden_stride, num_tokens_per_expert, stream);
    return;
  }

  dispatch_fp8_fast_kernel<true, false>(
      group_size, input, output_q, output_s, hidden_dim_num_groups,
      scale_hidden_stride, num_tokens_per_expert, kSiluActivation, stream);
}

template <int THREADS_PER_GROUP, bool FUSE_SILU_AND_MUL>
void launch_bf16_fp8_g128_row_kernel(
    ffi::TensorView input,
    ffi::TensorView output_q,
    ffi::TensorView output_s,
    int hidden_dim_num_groups,
    int num_tokens_per_expert,
    int fused_activation_type,
    musaStream_t stream) {
  int subwarps_per_block = choose_subwarps_per_block(hidden_dim_num_groups);
  if constexpr (THREADS_PER_GROUP == 32) {
    subwarps_per_block = subwarps_per_block > 8 ? 8 : subwarps_per_block;
  }
  dim3 grid(hidden_dim_num_groups / subwarps_per_block, num_tokens_per_expert);
  dim3 block(subwarps_per_block * THREADS_PER_GROUP);
  per_token_group_quant_8bit_bf16_fp8_g128_row_kernel<THREADS_PER_GROUP, FUSE_SILU_AND_MUL>
      <<<grid, block, 0, stream>>>(
      static_cast<__mt_bfloat16*>(input.data_ptr()),
      static_cast<__mt_fp8_e4m3*>(output_q.data_ptr()),
      static_cast<float*>(output_s.data_ptr()),
      hidden_dim_num_groups,
      fused_activation_type);
}

template <int THREADS_PER_GROUP, bool FUSE_SILU_AND_MUL, int HIDDEN_GROUPS_PER_BLOCK, int TOKEN_GROUPS_PER_BLOCK>
void launch_bf16_fp8_g128_row_tiled_kernel(
    ffi::TensorView input,
    ffi::TensorView output_q,
    ffi::TensorView output_s,
    int hidden_dim_num_groups,
    int num_tokens_per_expert,
    int fused_activation_type,
    musaStream_t stream) {
  dim3 grid(
      (hidden_dim_num_groups + HIDDEN_GROUPS_PER_BLOCK - 1) / HIDDEN_GROUPS_PER_BLOCK,
      (num_tokens_per_expert + TOKEN_GROUPS_PER_BLOCK - 1) / TOKEN_GROUPS_PER_BLOCK);
  dim3 block(THREADS_PER_GROUP * HIDDEN_GROUPS_PER_BLOCK * TOKEN_GROUPS_PER_BLOCK);
  per_token_group_quant_8bit_bf16_fp8_g128_row_tiled_kernel<
      THREADS_PER_GROUP,
      FUSE_SILU_AND_MUL,
      HIDDEN_GROUPS_PER_BLOCK,
      TOKEN_GROUPS_PER_BLOCK><<<grid, block, 0, stream>>>(
      static_cast<__mt_bfloat16*>(input.data_ptr()),
      static_cast<__mt_fp8_e4m3*>(output_q.data_ptr()),
      static_cast<float*>(output_s.data_ptr()),
      hidden_dim_num_groups,
      num_tokens_per_expert,
      fused_activation_type);
}

template <int THREADS_PER_GROUP, bool FUSE_SILU_AND_MUL>
bool try_launch_bf16_fp8_g128_row_tiled_kernel(
    ffi::TensorView input,
    ffi::TensorView output_q,
    ffi::TensorView output_s,
    int hidden_dim_num_groups,
    int num_tokens_per_expert,
    int fused_activation_type,
    musaStream_t stream) {
  if (hidden_dim_num_groups > 32) {
    return false;
  }
  if (num_tokens_per_expert < 16) {
    if constexpr (FUSE_SILU_AND_MUL) {
      return false;
    } else {
      if (num_tokens_per_expert < 8 || hidden_dim_num_groups < 2 || hidden_dim_num_groups > 8) {
        return false;
      }
    }
  }

  if constexpr (FUSE_SILU_AND_MUL) {
    if (hidden_dim_num_groups <= 4) {
      launch_bf16_fp8_g128_row_tiled_kernel<THREADS_PER_GROUP, FUSE_SILU_AND_MUL, 4, 2>(
          input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert, fused_activation_type, stream);
    } else if (hidden_dim_num_groups <= 8) {
      launch_bf16_fp8_g128_row_tiled_kernel<THREADS_PER_GROUP, FUSE_SILU_AND_MUL, 4, 2>(
          input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert, fused_activation_type, stream);
    } else if (hidden_dim_num_groups <= 16) {
      launch_bf16_fp8_g128_row_tiled_kernel<THREADS_PER_GROUP, FUSE_SILU_AND_MUL, 8, 2>(
          input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert, fused_activation_type, stream);
    } else {
      launch_bf16_fp8_g128_row_tiled_kernel<THREADS_PER_GROUP, FUSE_SILU_AND_MUL, 16, 1>(
          input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert, fused_activation_type, stream);
    }
    return true;
  }

  if (hidden_dim_num_groups <= 1) {
    launch_bf16_fp8_g128_row_tiled_kernel<THREADS_PER_GROUP, FUSE_SILU_AND_MUL, 1, 8>(
        input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert, fused_activation_type, stream);
  } else if (hidden_dim_num_groups <= 2) {
    launch_bf16_fp8_g128_row_tiled_kernel<THREADS_PER_GROUP, FUSE_SILU_AND_MUL, 2, 4>(
        input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert, fused_activation_type, stream);
  } else if (hidden_dim_num_groups <= 4) {
    launch_bf16_fp8_g128_row_tiled_kernel<THREADS_PER_GROUP, FUSE_SILU_AND_MUL, 4, 4>(
        input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert, fused_activation_type, stream);
  } else if (hidden_dim_num_groups <= 8) {
    launch_bf16_fp8_g128_row_tiled_kernel<THREADS_PER_GROUP, FUSE_SILU_AND_MUL, 8, 2>(
        input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert, fused_activation_type, stream);
  } else if (hidden_dim_num_groups <= 16) {
    launch_bf16_fp8_g128_row_tiled_kernel<THREADS_PER_GROUP, FUSE_SILU_AND_MUL, 16, 1>(
        input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert, fused_activation_type, stream);
  } else if (hidden_dim_num_groups % 4 == 0) {
    return false;
  } else if (hidden_dim_num_groups <= 24) {
    launch_bf16_fp8_g128_row_tiled_kernel<THREADS_PER_GROUP, FUSE_SILU_AND_MUL, 8, 2>(
        input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert, fused_activation_type, stream);
  } else {
    launch_bf16_fp8_g128_row_tiled_kernel<THREADS_PER_GROUP, FUSE_SILU_AND_MUL, 16, 1>(
        input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert, fused_activation_type, stream);
  }
  return true;
}

template <bool FUSE_SILU_AND_MUL>
void dispatch_bf16_fp8_g128_row_kernel(
    ffi::TensorView input,
    ffi::TensorView output_q,
    ffi::TensorView output_s,
    int hidden_dim_num_groups,
    int num_tokens_per_expert,
    int fused_activation_type,
    musaStream_t stream) {
  constexpr int THREADS_PER_GROUP = BF16_FP8_G128_THREADS_PER_GROUP;
  if (try_launch_bf16_fp8_g128_row_tiled_kernel<THREADS_PER_GROUP, FUSE_SILU_AND_MUL>(
          input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert, fused_activation_type, stream)) {
    return;
  }
  launch_bf16_fp8_g128_row_kernel<THREADS_PER_GROUP, FUSE_SILU_AND_MUL>(
      input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert, fused_activation_type, stream);
}

template <int GROUP_SIZE, bool FUSE_SILU_AND_MUL>
void launch_i8_fast_kernel(
    ffi::TensorView input,
    ffi::TensorView output_q,
    ffi::TensorView output_s,
    int hidden_dim_num_groups,
    int num_tokens_per_expert,
    int fused_activation_type,
    musaStream_t stream) {
  constexpr int THREADS_PER_GROUP = FastQuantConfig<GROUP_SIZE>::THREADS_PER_GROUP;
  const int subwarps_per_block = choose_subwarps_per_block(hidden_dim_num_groups);
  dim3 grid(hidden_dim_num_groups / subwarps_per_block, num_tokens_per_expert);
  dim3 block(subwarps_per_block * THREADS_PER_GROUP);
  per_token_group_quant_8bit_fast_kernel<GROUP_SIZE, THREADS_PER_GROUP, int8_t, false, FUSE_SILU_AND_MUL>
      <<<grid, block, 0, stream>>>(
          static_cast<half*>(input.data_ptr()),
          static_cast<int8_t*>(output_q.data_ptr()),
          static_cast<float*>(output_s.data_ptr()),
          hidden_dim_num_groups,
          0,
          num_tokens_per_expert,
          fused_activation_type);
}

template <int GROUP_SIZE, int HIDDEN_GROUPS_PER_BLOCK, int TOKEN_GROUPS_PER_BLOCK>
void launch_i8_fused_tiled_kernel(
    ffi::TensorView input,
    ffi::TensorView output_q,
    ffi::TensorView output_s,
    int hidden_dim_num_groups,
    int num_tokens_per_expert,
    musaStream_t stream) {
  constexpr int THREADS_PER_GROUP = FastQuantConfig<GROUP_SIZE>::THREADS_PER_GROUP;
  dim3 grid(
      (hidden_dim_num_groups + HIDDEN_GROUPS_PER_BLOCK - 1) / HIDDEN_GROUPS_PER_BLOCK,
      (num_tokens_per_expert + TOKEN_GROUPS_PER_BLOCK - 1) / TOKEN_GROUPS_PER_BLOCK);
  dim3 block(THREADS_PER_GROUP * HIDDEN_GROUPS_PER_BLOCK * TOKEN_GROUPS_PER_BLOCK);
  per_token_group_quant_8bit_i8_fused_tiled_kernel<
      GROUP_SIZE,
      THREADS_PER_GROUP,
      HIDDEN_GROUPS_PER_BLOCK,
      TOKEN_GROUPS_PER_BLOCK><<<grid, block, 0, stream>>>(
      static_cast<half*>(input.data_ptr()),
      static_cast<int8_t*>(output_q.data_ptr()),
      static_cast<float*>(output_s.data_ptr()),
      hidden_dim_num_groups,
      num_tokens_per_expert);
}

bool try_launch_i8_fused_tiled_kernel(
    int64_t group_size,
    ffi::TensorView input,
    ffi::TensorView output_q,
    ffi::TensorView output_s,
    int hidden_dim_num_groups,
    int num_tokens_per_expert,
    musaStream_t stream) {
  if (num_tokens_per_expert >= 512) {
    return false;
  }

  if (group_size == 64) {
    if (hidden_dim_num_groups <= 2 && num_tokens_per_expert >= 128) {
      launch_i8_fused_tiled_kernel<64, 2, 8>(
          input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert, stream);
    } else if (hidden_dim_num_groups <= 4 && num_tokens_per_expert >= 256) {
      launch_i8_fused_tiled_kernel<64, 4, 4>(
          input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert, stream);
    } else {
      return false;
    }
    return true;
  }

  if (group_size == 128) {
    if (hidden_dim_num_groups <= 1 && num_tokens_per_expert >= 256) {
      launch_i8_fused_tiled_kernel<128, 1, 8>(
          input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert, stream);
    } else if (hidden_dim_num_groups <= 2 && num_tokens_per_expert >= 256) {
      launch_i8_fused_tiled_kernel<128, 2, 4>(
          input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert, stream);
    } else if (hidden_dim_num_groups <= 4 && num_tokens_per_expert >= 128 && num_tokens_per_expert <= 384) {
      launch_i8_fused_tiled_kernel<128, 4, 2>(
          input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert, stream);
    } else {
      return false;
    }
    return true;
  }

  return false;
}

template <bool FUSE_SILU_AND_MUL>
void dispatch_i8_fast_kernel(
    int64_t group_size,
    ffi::TensorView input,
    ffi::TensorView output_q,
    ffi::TensorView output_s,
    int hidden_dim_num_groups,
    int num_tokens_per_expert,
    int fused_activation_type,
    musaStream_t stream) {
  switch (group_size) {
    case 16:
      launch_i8_fast_kernel<16, FUSE_SILU_AND_MUL>(
          input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert,
          fused_activation_type, stream);
      break;
    case 32:
      launch_i8_fast_kernel<32, FUSE_SILU_AND_MUL>(
          input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert,
          fused_activation_type, stream);
      break;
    case 64:
      launch_i8_fast_kernel<64, FUSE_SILU_AND_MUL>(
          input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert,
          fused_activation_type, stream);
      break;
    case 128:
      launch_i8_fast_kernel<128, FUSE_SILU_AND_MUL>(
          input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert,
          fused_activation_type, stream);
      break;
    default:
      TORCH_CHECK(false, "Unsupported group_size");
  }
}

void sgl_per_token_group_quant_8bit_v2(
    // vanilla: (num_tokens, hidden_size)
    // fuse_silu_and_mul: (num_tokens, hidden_size * 2)
    // fuse_silu_and_mul + masked_layout: (num_experts, num_tokens-with-padding, hidden_size * 2)
    ffi::TensorView input,
    ffi::TensorView output_q,
    ffi::TensorView output_s,
    int64_t group_size,
    double eps,
    double min_8bit,
    double max_8bit,
    bool scale_ue8m0,
    bool fuse_silu_and_mul,
    int64_t fused_activation_type,
    ffi::TensorView masked_m,
    bool has_masked_m) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  TVM_FFI_ICHECK_EQ(input.device().device_id, output_q.device().device_id);
  TVM_FFI_ICHECK_EQ(input.device().device_id, output_s.device().device_id);
  TVM_FFI_ICHECK(tensor_numel(input) > 0);

  TORCH_CHECK(std::abs(LOCAL_ABSMAX_ABS - eps) < 1e-13);

  CHECK_EQ(tensor_numel(input) % group_size, 0);
  const int num_groups = static_cast<int>(tensor_numel(input)) / group_size / (fuse_silu_and_mul ? 2 : 1);

  const bool masked_layout = has_masked_m;
  TORCH_CHECK(output_s.ndim() == (masked_layout ? 3 : 2));
  if (masked_layout) {
    TVM_FFI_ICHECK_EQ(masked_m.device().device_type, kDLExtDev);
    TVM_FFI_ICHECK_EQ(masked_m.device().device_id, input.device().device_id);
    TVM_FFI_ICHECK(masked_m.IsContiguous());
    TVM_FFI_ICHECK(dtype_equal(masked_m.dtype(), DLDataType{kDLInt, 32, 1}));
  }

  const int num_local_experts = masked_layout ? input.size(0) : 1;

  ffi::MUSADeviceGuard device_guard(input.device().device_id);
  musaStream_t stream = get_stream(input.device());

  const bool is_column_major = output_s.stride(-2) < output_s.stride(-1);
  const int hidden_dim_num_groups = static_cast<int>(output_q.size(-1)) / group_size;
  const int num_tokens_per_expert = static_cast<int>(output_q.size(-2));
  const int scale_expert_stride = masked_layout ? static_cast<int>(output_s.stride(0)) : 0;
  const int scale_hidden_stride = static_cast<int>(output_s.stride(-1));
  const int fused_act = static_cast<int>(fused_activation_type);
  TORCH_CHECK(
      fused_act == kSiluActivation || fused_act == kGeluActivation ||
      fused_act == kGeluTanhActivation,
      "Unsupported fused activation type");

#define LAUNCH_KERNEL_INNER(SCHEDULER, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, output_s_dtype, ...)           \
  do {                                                                                                               \
    int subwarps_per_block;                                                                                          \
    dim3 grid, block;                                                                                                \
    SCHEDULER::compute_exec_config(                                                                                  \
        THREADS_PER_SUBWARP, num_local_experts, hidden_dim_num_groups, num_groups, subwarps_per_block, grid, block); \
                                                                                                                     \
    per_token_group_quant_8bit_kernel<SCHEDULER, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, __VA_ARGS__>         \
        <<<grid, block, 0, stream>>>(                                                                                \
            static_cast<T*>(input.data_ptr()),                                                                       \
            static_cast<DST_DTYPE*>(output_q.data_ptr()),                                                            \
            static_cast<output_s_dtype*>(output_s.data_ptr()),                                                       \
            static_cast<int32_t*>(masked_layout ? masked_m.data_ptr() : nullptr),                                    \
            subwarps_per_block,                                                                                      \
            hidden_dim_num_groups,                                                                                   \
            scale_expert_stride,                                                                                     \
            scale_hidden_stride,                                                                                     \
            num_tokens_per_expert,                                                                                   \
            fused_act);                                                                                              \
  } while (0)

#define LAUNCH_KERNEL(GROUP_SIZE, T, DST_DTYPE)                                                                     \
  do {                                                                                                              \
    constexpr int THREADS_PER_SUBWARP = GROUP_SIZE / 16;                                                            \
    TORCH_CHECK(THREADS_PER_SUBWARP * INPUT_PRIMARY_VEC_NUM_BYTES == group_size * sizeof(T));                       \
                                                                                                                    \
    using dst_dtype_info = DtypeInfo<DST_DTYPE>;                                                                    \
    CHECK_EQ(dst_dtype_info::MIN, min_8bit);                                                                        \
    CHECK_EQ(dst_dtype_info::MAX, max_8bit);                                                                        \
                                                                                                                    \
    if (is_column_major) {                                                                                          \
      if (scale_ue8m0) {                                                                                            \
        if (fuse_silu_and_mul) {                                                                                    \
          if (masked_layout) {                                                                                      \
            LAUNCH_KERNEL_INNER(                                                                                    \
                MaskedLayoutScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, uint32_t, true, true, true);  \
          } else {                                                                                                  \
            LAUNCH_KERNEL_INNER(                                                                                    \
                RowScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, uint32_t, true, true, true);           \
          }                                                                                                         \
        } else {                                                                                                    \
          if (masked_layout) {                                                                                      \
            LAUNCH_KERNEL_INNER(                                                                                    \
                MaskedLayoutScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, uint32_t, true, true);        \
          } else {                                                                                                  \
            LAUNCH_KERNEL_INNER(RowScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, uint32_t, true, true); \
          }                                                                                                         \
        }                                                                                                           \
      } else {                                                                                                      \
        if (fuse_silu_and_mul) {                                                                                    \
          if (masked_layout) {                                                                                      \
            LAUNCH_KERNEL_INNER(                                                                                    \
                MaskedLayoutScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, true, false, true);    \
          } else {                                                                                                  \
            LAUNCH_KERNEL_INNER(                                                                                    \
                RowScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, true, false, true);             \
          }                                                                                                         \
        } else {                                                                                                    \
          if (masked_layout) {                                                                                      \
            LAUNCH_KERNEL_INNER(                                                                                    \
                MaskedLayoutScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, true);                 \
          } else {                                                                                                  \
            LAUNCH_KERNEL_INNER(RowScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, true);          \
          }                                                                                                         \
        }                                                                                                           \
      }                                                                                                             \
    } else {                                                                                                        \
      if (fuse_silu_and_mul) {                                                                                      \
        if (masked_layout) {                                                                                        \
          LAUNCH_KERNEL_INNER(                                                                                      \
              MaskedLayoutScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, false, false, true);     \
        } else {                                                                                                    \
          LAUNCH_KERNEL_INNER(                                                                                      \
              RowScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, false, false, true);              \
        }                                                                                                           \
      } else {                                                                                                      \
        if (masked_layout) {                                                                                        \
          LAUNCH_KERNEL_INNER(MaskedLayoutScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, false);  \
        } else {                                                                                                    \
          LAUNCH_KERNEL_INNER(RowScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, false);           \
        }                                                                                                           \
      }                                                                                                             \
    }                                                                                                               \
  } while (0)

#define LAUNCH_KERNEL_OUTER(...)                    \
  switch (group_size) {                             \
    case 16:                                        \
      LAUNCH_KERNEL(16, __VA_ARGS__);               \
      break;                                        \
    case 32:                                        \
      LAUNCH_KERNEL(32, __VA_ARGS__);               \
      break;                                        \
    case 64:                                        \
      LAUNCH_KERNEL(64, __VA_ARGS__);               \
      break;                                        \
    case 128:                                       \
      LAUNCH_KERNEL(128, __VA_ARGS__);              \
      break;                                        \
    default:                                        \
      TORCH_CHECK(false, "Unsupported group_size"); \
  }                                                 \
  while (0)

  if (dtype_equal(input.dtype(), dl_float16)) {
    using scalar_t = half;
    if (dtype_equal(output_q.dtype(), dl_int8)) {
      if (!is_column_major && !scale_ue8m0 && !masked_layout) {
        const bool prefer_generic_for_large_hidden =
            (group_size == 64 &&
             ((hidden_dim_num_groups >= 32 && num_tokens_per_expert >= 2048) ||
              (hidden_dim_num_groups >= 24 && num_tokens_per_expert >= 4096))) ||
            (group_size == 128 && hidden_dim_num_groups >= 12 && num_tokens_per_expert >= 2048);
        if (fuse_silu_and_mul && group_size <= 128 && !prefer_generic_for_large_hidden) {
          if (!try_launch_i8_fused_tiled_kernel(
                  group_size, input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert, stream)) {
            dispatch_i8_fast_kernel<true>(
                group_size, input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert, fused_act, stream);
          }
        } else if (!fuse_silu_and_mul && group_size == 16) {
          dispatch_i8_fast_kernel<false>(
              group_size, input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert, fused_act, stream);
        } else {
          LAUNCH_KERNEL_OUTER(scalar_t, int8_t);
        }
      } else {
        LAUNCH_KERNEL_OUTER(scalar_t, int8_t);
      }
    } else if (dtype_equal(output_q.dtype(), dl_float8_e4m3fn)) {
      if (!scale_ue8m0 && !masked_layout) {
        if (is_column_major) {
          if (fuse_silu_and_mul) {
            dispatch_fp8_fast_kernel<true, true>(
                group_size,
                input,
                output_q,
                output_s,
                hidden_dim_num_groups,
                scale_hidden_stride,
                num_tokens_per_expert,
                fused_act,
                stream);
          } else {
            dispatch_fp8_col_nonfused_kernel(
                group_size,
                input,
                output_q,
                output_s,
                hidden_dim_num_groups,
                scale_hidden_stride,
                num_tokens_per_expert,
                stream);
          }
        } else {
          if (fuse_silu_and_mul) {
            dispatch_fp8_fast_kernel<false, true>(
                group_size,
                input,
                output_q,
                output_s,
                hidden_dim_num_groups,
                scale_hidden_stride,
                num_tokens_per_expert,
                fused_act,
                stream);
          } else {
            dispatch_fp8_fast_kernel<false, false>(
                group_size,
                input,
                output_q,
                output_s,
                hidden_dim_num_groups,
                scale_hidden_stride,
                num_tokens_per_expert,
                fused_act,
                stream);
          }
        }
      } else {
        LAUNCH_KERNEL_OUTER(scalar_t, __mt_fp8_e4m3);
      }
    } else {
      TVM_FFI_THROW(ValueError) << "Unsupported output_q dtype";
    }
  } else if (dtype_equal(input.dtype(), dl_bfloat16)) {
    using scalar_t = __mt_bfloat16;
    if (dtype_equal(output_q.dtype(), dl_int8)) {
      LAUNCH_KERNEL_OUTER(scalar_t, int8_t);
    } else if (dtype_equal(output_q.dtype(), dl_float8_e4m3fn)) {
      if (group_size == 128 && !is_column_major && !scale_ue8m0 && !masked_layout) {
        if (fuse_silu_and_mul) {
          dispatch_bf16_fp8_g128_row_kernel<true>(
              input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert, fused_act, stream);
        } else {
          dispatch_bf16_fp8_g128_row_kernel<false>(
              input, output_q, output_s, hidden_dim_num_groups, num_tokens_per_expert, fused_act, stream);
        }
      } else {
        LAUNCH_KERNEL_OUTER(scalar_t, __mt_fp8_e4m3);
      }
    } else {
      TVM_FFI_THROW(ValueError) << "Unsupported output_q dtype";
    }
  } else {
    TVM_FFI_THROW(ValueError) << "Unsupported input dtype";
  }

#undef LAUNCH_KERNEL
#undef LAUNCH_KERNEL_INNER
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_per_token_group_quant_8bit_v2, sgl_per_token_group_quant_8bit_v2);
