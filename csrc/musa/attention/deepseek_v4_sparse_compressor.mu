#include <cfloat>
#include <cmath>
#include <cstdint>

#include <musa_bf16.h>
#include <musa_fp8.h>
#include <musa_runtime.h>
#include <torch/all.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace {

constexpr int64_t kHeadDim = 512;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kNopeDim = kHeadDim - kRopeDim;
constexpr int64_t kQuantBlock = 64;
constexpr int64_t kNopeBlocks = kNopeDim / kQuantBlock;
constexpr int64_t kTokenStride = 576;
constexpr int64_t kScaleDim = 8;
constexpr int64_t kTokenValueBytes = kNopeDim + kRopeDim * 2;
constexpr int64_t kTokenScaleBytes = kScaleDim;
constexpr int64_t kThreadsPerBlock = 128;
constexpr int64_t kWarpsPerBlock = 4;
// Non-overlap compression pools its rows with this many thread groups.
constexpr int kRowGroups = 4;
constexpr int64_t kMaxDecodeRows = 128;
constexpr float kFp8Max = 448.0f;
constexpr float kLog2E = 1.4426950408889634f;

constexpr int kIndexInt32 = 1;
constexpr int kIndexInt64 = 2;

static_assert(kTokenValueBytes == kTokenStride, "sparse MLA token data is 576 bytes");
static_assert(kNopeBlocks + 1 == kScaleDim, "7 UE8M0 scales plus 1 pad");

__device__ __forceinline__ int64_t load_index(const void* ptr, int kind,
                                              int64_t idx) {
  if (kind == kIndexInt32) {
    return static_cast<int64_t>(static_cast<const int32_t*>(ptr)[idx]);
  }
  return static_cast<const int64_t*>(ptr)[idx];
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
  value += __shfl_xor_sync(0xffffffffu, value, 16);
  value += __shfl_xor_sync(0xffffffffu, value, 8);
  value += __shfl_xor_sync(0xffffffffu, value, 4);
  value += __shfl_xor_sync(0xffffffffu, value, 2);
  value += __shfl_xor_sync(0xffffffffu, value, 1);
  return value;
}

__device__ __forceinline__ uint32_t half_warp_reduce_max_u32(uint32_t value,
                                                             int lane) {
  const unsigned mask = (lane < 16) ? 0x0000ffffu : 0xffff0000u;
  uint32_t peer = __shfl_xor_sync(mask, value, 8);
  value = value > peer ? value : peer;
  peer = __shfl_xor_sync(mask, value, 4);
  value = value > peer ? value : peer;
  peer = __shfl_xor_sync(mask, value, 2);
  value = value > peer ? value : peer;
  peer = __shfl_xor_sync(mask, value, 1);
  return value > peer ? value : peer;
}

__device__ __forceinline__ float weight_to_float(float value) { return value; }

__device__ __forceinline__ float weight_to_float(__mt_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ float clamp_fp8(float value) {
  return fminf(fmaxf(value, -kFp8Max), kFp8Max);
}

__device__ __forceinline__ void load_state_row(
    const float* __restrict__ state_cache, int64_t state_stride0,
    int64_t state_stride1, const void* __restrict__ block_table,
    int block_table_kind, int64_t block_table_stride, int64_t req_idx,
    int64_t source_position, int64_t head_offset, int64_t state_width,
    int64_t state_block_size, int64_t num_state_blocks, int64_t max_blocks,
    int lane, float* kv4, float* score4) {
  bool valid = source_position >= 0;
  int64_t physical_block = 0;
  const int64_t logical_block =
      valid ? source_position / state_block_size : 0;
  if (valid) {
    valid = logical_block >= 0 && logical_block < max_blocks;
  }
  if (valid) {
    physical_block = load_index(block_table, block_table_kind,
                                req_idx * block_table_stride + logical_block);
    valid = physical_block >= 0 && physical_block < num_state_blocks;
  }
  float4 kv_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float4 score_vec = make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
  if (valid) {
    const int64_t block_offset = source_position % state_block_size;
    const float* row_ptr = state_cache + physical_block * state_stride0 +
                           block_offset * state_stride1 + head_offset +
                           lane * 4;
    kv_vec = *reinterpret_cast<const float4*>(row_ptr);
    score_vec = *reinterpret_cast<const float4*>(row_ptr + state_width);
  }
  kv4[0] = kv_vec.x;
  kv4[1] = kv_vec.y;
  kv4[2] = kv_vec.z;
  kv4[3] = kv_vec.w;
  score4[0] = score_vec.x;
  score4[1] = score_vec.y;
  score4[2] = score_vec.z;
  score4[3] = score_vec.w;
}

__device__ __forceinline__ void store_kv_score_tile(
    float* __restrict__ base, int64_t state_width, const float* kv_row,
    const float* score_row, const float* ape_row, int offset) {
  *reinterpret_cast<float4*>(base + offset) =
      *reinterpret_cast<const float4*>(kv_row + offset);
  const float4 score_vec = *reinterpret_cast<const float4*>(score_row + offset);
  const float4 ape_vec = *reinterpret_cast<const float4*>(ape_row + offset);
  *reinterpret_cast<float4*>(base + state_width + offset) = make_float4(
      score_vec.x + ape_vec.x, score_vec.y + ape_vec.y, score_vec.z + ape_vec.z,
      score_vec.w + ape_vec.w);
}

template <bool kOverlap, int kStateBlockSize>
__device__ __forceinline__ void save_partial_row(
    float* __restrict__ state_cache, int64_t state_stride0,
    int64_t state_stride1, int64_t state_slot, int64_t state_width,
    const float* kv_row, const float* score_row, const float* ape_row,
    int tid) {
  const int64_t block_idx = state_slot / kStateBlockSize;
  const int64_t pos_in_block = state_slot % kStateBlockSize;
  float* base = state_cache + block_idx * state_stride0 +
                pos_in_block * state_stride1;
  const int offset = tid * 4;
  store_kv_score_tile(base, state_width, kv_row, score_row, ape_row, offset);
  if constexpr (kOverlap) {
    store_kv_score_tile(base, state_width, kv_row, score_row, ape_row,
                        kHeadDim + offset);
  }
}

template <int kCompressRatio, bool kOverlap, int kStateBlockSize>
__global__ __launch_bounds__(kThreadsPerBlock, 4) void
deepseek_v4_sparse_save_partial_kernel(
    float* __restrict__ state_cache, int64_t state_stride0,
    int64_t state_stride1, const void* __restrict__ positions,
    int position_kind, const void* __restrict__ state_slot_mapping,
    int state_slot_kind, int64_t num_tokens, int64_t num_state_blocks,
    int64_t state_width, const float* __restrict__ kv_states, int64_t kv_stride,
    const float* __restrict__ score_states, int64_t score_stride,
    const float* __restrict__ ape, int64_t ape_stride) {
  const int tid = threadIdx.x;
  const int64_t token = static_cast<int64_t>(blockIdx.x);
  if (token >= num_tokens) {
    return;
  }
  const int64_t state_slot =
      load_index(state_slot_mapping, state_slot_kind, token);
  if (state_slot < 0 ||
      state_slot >= num_state_blocks * kStateBlockSize) {
    return;
  }
  const int64_t position = load_index(positions, position_kind, token);
  const int64_t ape_row = position % kCompressRatio;
  save_partial_row<kOverlap, kStateBlockSize>(
      state_cache, state_stride0, state_stride1, state_slot, state_width,
      kv_states + token * kv_stride, score_states + token * score_stride,
      ape + ape_row * ape_stride, tid);
}

template <typename WeightT, int kCompressRatio, bool kOverlap,
          int kStateBlockSize>
__global__ __launch_bounds__(kThreadsPerBlock * kRowGroups) void
deepseek_v4_sparse_compressor_kernel(
    const float* __restrict__ state_cache, int64_t state_stride0,
    int64_t state_stride1, const void* __restrict__ token_to_req_indices,
    int token_to_req_kind, const void* __restrict__ positions,
    int position_kind, const void* __restrict__ state_slot_mapping,
    int state_slot_kind, const void* __restrict__ block_table,
    int block_table_kind, int64_t block_table_stride,
    const WeightT* __restrict__ rms_norm_weight,
    const float* __restrict__ cos_sin_cache, int64_t cos_sin_stride,
    uint8_t* __restrict__ kv_cache, int64_t kv_cache_stride0,
    const void* __restrict__ kv_slot_mapping, int kv_slot_kind, float rms_eps,
    int64_t num_tokens, int64_t num_state_blocks, int64_t num_reqs,
    int64_t max_blocks_per_req, int64_t num_kv_blocks, int64_t kv_block_size,
    int64_t state_width) {
  constexpr int kCompressRows = (1 + static_cast<int>(kOverlap)) * kCompressRatio;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int64_t token = static_cast<int64_t>(blockIdx.x);
  if (token >= num_tokens) {
    return;
  }

  const int64_t state_slot =
      load_index(state_slot_mapping, state_slot_kind, token);
  if (state_slot < 0 ||
      state_slot >= num_state_blocks * kStateBlockSize) {
    return;
  }
  const int64_t position = load_index(positions, position_kind, token);
  if (position < kCompressRatio - 1 ||
      (position + 1) % kCompressRatio != 0) {
    return;
  }
  const int64_t req_idx =
      load_index(token_to_req_indices, token_to_req_kind, token);
  if (req_idx < 0 || req_idx >= num_reqs) {
    return;
  }

  float compressed[4];
  if constexpr (kOverlap) {
    float kv[kCompressRows][4];
    float score[kCompressRows][4];
    const int64_t first_position = position - kCompressRows + 1;
#pragma unroll
    for (int row = 0; row < kCompressRows; ++row) {
      const int64_t head_offset = row >= kCompressRatio ? kHeadDim : 0;
      load_state_row(state_cache, state_stride0, state_stride1, block_table,
                     block_table_kind, block_table_stride, req_idx,
                     first_position + row, head_offset, state_width,
                     kStateBlockSize, num_state_blocks, max_blocks_per_req, tid,
                     kv[row], score[row]);
    }
    float max0 = score[0][0];
    float max1 = score[0][1];
    float max2 = score[0][2];
    float max3 = score[0][3];
#pragma unroll
    for (int row = 1; row < kCompressRows; ++row) {
      max0 = fmaxf(max0, score[row][0]);
      max1 = fmaxf(max1, score[row][1]);
      max2 = fmaxf(max2, score[row][2]);
      max3 = fmaxf(max3, score[row][3]);
    }
    float den0 = 0.0f, den1 = 0.0f, den2 = 0.0f, den3 = 0.0f;
    float num0 = 0.0f, num1 = 0.0f, num2 = 0.0f, num3 = 0.0f;
#pragma unroll
    for (int row = 0; row < kCompressRows; ++row) {
      const float e0 = exp2f((score[row][0] - max0) * kLog2E);
      const float e1 = exp2f((score[row][1] - max1) * kLog2E);
      const float e2 = exp2f((score[row][2] - max2) * kLog2E);
      const float e3 = exp2f((score[row][3] - max3) * kLog2E);
      num0 += kv[row][0] * e0;
      num1 += kv[row][1] * e1;
      num2 += kv[row][2] * e2;
      num3 += kv[row][3] * e3;
      den0 += e0;
      den1 += e1;
      den2 += e2;
      den3 += e3;
    }
    compressed[0] = num0 / den0;
    compressed[1] = num1 / den1;
    compressed[2] = num2 / den2;
    compressed[3] = num3 / den3;
  } else {
    // Each row group pools a strided row subset with an online softmax; the
    // partials merge in shared memory.  Group 0 owns the epilogue.
    const int group = tid / kThreadsPerBlock;
    const int dim_tid = tid % kThreadsPerBlock;
    float maxv[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
    float den[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float num[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int64_t first_position = position - kCompressRows + 1;
    for (int row = group; row < kCompressRows; row += kRowGroups) {
      float kv4[4];
      float score4[4];
      load_state_row(state_cache, state_stride0, state_stride1, block_table,
                     block_table_kind, block_table_stride, req_idx,
                     first_position + row, 0, state_width, kStateBlockSize,
                     num_state_blocks, max_blocks_per_req, dim_tid, kv4,
                     score4);
#pragma unroll
      for (int elem = 0; elem < 4; ++elem) {
        if (score4[elem] > maxv[elem]) {
          const float rescale = exp2f((maxv[elem] - score4[elem]) * kLog2E);
          num[elem] *= rescale;
          den[elem] *= rescale;
          maxv[elem] = score4[elem];
        }
        const float e = exp2f((score4[elem] - maxv[elem]) * kLog2E);
        num[elem] += kv4[elem] * e;
        den[elem] += e;
      }
    }
    __shared__ float partial_max[kRowGroups][kThreadsPerBlock * 4];
    __shared__ float partial_num[kRowGroups][kThreadsPerBlock * 4];
    __shared__ float partial_den[kRowGroups][kThreadsPerBlock * 4];
#pragma unroll
    for (int elem = 0; elem < 4; ++elem) {
      partial_max[group][dim_tid * 4 + elem] = maxv[elem];
      partial_num[group][dim_tid * 4 + elem] = num[elem];
      partial_den[group][dim_tid * 4 + elem] = den[elem];
    }
    __syncthreads();
#pragma unroll
    for (int elem = 0; elem < 4; ++elem) {
      const int idx = dim_tid * 4 + elem;
      float merged_max = partial_max[0][idx];
#pragma unroll
      for (int g = 1; g < kRowGroups; ++g) {
        merged_max = fmaxf(merged_max, partial_max[g][idx]);
      }
      float merged_num = 0.0f;
      float merged_den = 0.0f;
#pragma unroll
      for (int g = 0; g < kRowGroups; ++g) {
        const float scale = exp2f((partial_max[g][idx] - merged_max) * kLog2E);
        merged_num += partial_num[g][idx] * scale;
        merged_den += partial_den[g][idx] * scale;
      }
      compressed[elem] = merged_num / merged_den;
    }
  }

  float sum_of_squares = 0.0f;
#pragma unroll
  for (int elem = 0; elem < 4; ++elem) {
    sum_of_squares += compressed[elem] * compressed[elem];
  }
  sum_of_squares = warp_reduce_sum(sum_of_squares);

  __shared__ float warp_sums[kWarpsPerBlock];
  if (lane == 0 && warp < kWarpsPerBlock) {
    warp_sums[warp] = sum_of_squares;
  }
  __syncthreads();
  float reduced = tid < kWarpsPerBlock ? warp_sums[tid] : 0.0f;
  if (warp == 0) {
    reduced = warp_reduce_sum(reduced);
    if (lane == 0) {
      warp_sums[0] =
          rsqrtf(reduced / static_cast<float>(kHeadDim) + rms_eps);
    }
  }
  __syncthreads();
  const float norm_factor = warp_sums[0];
  if (tid >= kThreadsPerBlock) {
    return;
  }

  float output[4];
#pragma unroll
  for (int elem = 0; elem < 4; ++elem) {
    output[elem] = compressed[elem] * norm_factor *
                   weight_to_float(rms_norm_weight[tid * 4 + elem]);
  }

  const int half = lane >> 4;
  const int half_lane = lane & 15;
  const int tile = warp * 2 + half;

  const int64_t kv_slot = load_index(kv_slot_mapping, kv_slot_kind, token);
  if (kv_slot < 0 || kv_slot >= num_kv_blocks * kv_block_size) {
    return;
  }
  const int64_t kv_block_idx = kv_slot / kv_block_size;
  const int64_t kv_pos_in_block = kv_slot % kv_block_size;
  uint8_t* cache_block = kv_cache + kv_block_idx * kv_cache_stride0;
  uint8_t* value_ptr = cache_block + kv_pos_in_block * kTokenValueBytes;
  uint8_t* scale_ptr = cache_block + kv_block_size * kTokenValueBytes +
                       kv_pos_in_block * kTokenScaleBytes;

  if (tile < kNopeBlocks) {
    float quant[4];
#pragma unroll
    for (int elem = 0; elem < 4; ++elem) {
      quant[elem] = __bfloat162float(__float2bfloat16(output[elem]));
    }
    uint32_t local_absmax_bits = __float_as_uint(quant[0]) & 0x7fffffffu;
#pragma unroll
    for (int elem = 1; elem < 4; ++elem) {
      const uint32_t abs_bits = __float_as_uint(quant[elem]) & 0x7fffffffu;
      local_absmax_bits =
          local_absmax_bits > abs_bits ? local_absmax_bits : abs_bits;
    }
    const float absmax = fmaxf(
        1.0e-4f, __uint_as_float(half_warp_reduce_max_u32(local_absmax_bits,
                                                          lane)));
    const int exponent = static_cast<int>(ceilf(log2f(absmax / kFp8Max)));
    const float inv_scale = exp2f(static_cast<float>(-exponent));
    const int encoded = max(0, min(255, exponent + 127));
    const float4 quant_input =
        make_float4(clamp_fp8(quant[0] * inv_scale),
                    clamp_fp8(quant[1] * inv_scale),
                    clamp_fp8(quant[2] * inv_scale),
                    clamp_fp8(quant[3] * inv_scale));
    const __mt_fp8x4_e4m3 packed(quant_input);
    reinterpret_cast<uint32_t*>(value_ptr)[tid] =
        static_cast<uint32_t>(packed.__x);
    if (half_lane == 0) {
      scale_ptr[tile] = static_cast<uint8_t>(encoded);
    }
  } else {
    const int64_t pair0 = static_cast<int64_t>(half_lane) * 2;
    const int64_t compressed_position =
        (position / kCompressRatio) * kCompressRatio;
    const float* cos_ptr =
        cos_sin_cache + compressed_position * cos_sin_stride;
    const float* sin_ptr = cos_ptr + kRopeDim / 2;
    const float even0 = output[0];
    const float odd0 = output[1];
    const float even1 = output[2];
    const float odd1 = output[3];
    const float cos0 = cos_ptr[pair0];
    const float sin0 = sin_ptr[pair0];
    const float cos1 = cos_ptr[pair0 + 1];
    const float sin1 = sin_ptr[pair0 + 1];
    __mt_bfloat16* rope_ptr =
        reinterpret_cast<__mt_bfloat16*>(value_ptr + kNopeDim);
    rope_ptr[half_lane * 4 + 0] =
        __float2bfloat16(even0 * cos0 - odd0 * sin0);
    rope_ptr[half_lane * 4 + 1] =
        __float2bfloat16(odd0 * cos0 + even0 * sin0);
    rope_ptr[half_lane * 4 + 2] =
        __float2bfloat16(even1 * cos1 - odd1 * sin1);
    rope_ptr[half_lane * 4 + 3] =
        __float2bfloat16(odd1 * cos1 + even1 * sin1);
    if (half_lane == 0) {
      scale_ptr[kNopeBlocks] = 0;
    }
  }
}

int index_kind(const torch::Tensor& tensor, const char* name) {
  if (tensor.scalar_type() == torch::kInt32) {
    return kIndexInt32;
  }
  if (tensor.scalar_type() == torch::kInt64) {
    return kIndexInt64;
  }
  TORCH_CHECK(false, name, " must be int32 or int64");
}

void check_same_device(const torch::Tensor& reference,
                       const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device() == reference.device(), name,
              " must be on the same device as state_cache");
}

template <int kCompressRatio, bool kOverlap, int kStateBlockSize>
void launch_sparse_save_partial(torch::Tensor& state_cache,
                                const torch::Tensor& positions,
                                const torch::Tensor& state_slot_mapping,
                                int64_t state_width, const float* kv_states,
                                int64_t kv_stride, const float* score_states,
                                int64_t score_stride, const float* ape,
                                int64_t ape_stride, musaStream_t stream) {
  const int64_t num_tokens = state_slot_mapping.numel();
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(num_tokens));
  deepseek_v4_sparse_save_partial_kernel<kCompressRatio, kOverlap,
                                         kStateBlockSize>
      <<<grid, block, 0, stream>>>(
          static_cast<float*>(state_cache.data_ptr()), state_cache.stride(0),
          state_cache.stride(1), positions.data_ptr(),
          index_kind(positions, "positions"), state_slot_mapping.data_ptr(),
          index_kind(state_slot_mapping, "state_slot_mapping"), num_tokens,
          state_cache.size(0), state_width, kv_states, kv_stride, score_states,
          score_stride, ape, ape_stride);
}

template <typename WeightT, int kCompressRatio, bool kOverlap,
          int kStateBlockSize>
void launch_sparse_compressor(const torch::Tensor& state_cache,
                              const torch::Tensor& token_to_req_indices,
                              const torch::Tensor& positions,
                              const torch::Tensor& state_slot_mapping,
                              const torch::Tensor& block_table,
                              const torch::Tensor& rms_norm_weight,
                              const torch::Tensor& cos_sin_cache,
                              torch::Tensor& kv_cache,
                              const torch::Tensor& kv_slot_mapping,
                              float rms_eps, int64_t state_width,
                              musaStream_t stream) {
  const int64_t num_tokens = state_slot_mapping.numel();
  const dim3 block(kOverlap ? kThreadsPerBlock : kThreadsPerBlock * kRowGroups);
  const dim3 grid(static_cast<unsigned int>(num_tokens));
  deepseek_v4_sparse_compressor_kernel<WeightT, kCompressRatio, kOverlap,
                                       kStateBlockSize>
      <<<grid, block, 0, stream>>>(
          static_cast<const float*>(state_cache.data_ptr()),
          state_cache.stride(0), state_cache.stride(1),
          token_to_req_indices.data_ptr(),
          index_kind(token_to_req_indices, "token_to_req_indices"),
          positions.data_ptr(), index_kind(positions, "positions"),
          state_slot_mapping.data_ptr(),
          index_kind(state_slot_mapping, "state_slot_mapping"),
          block_table.data_ptr(), index_kind(block_table, "block_table"),
          block_table.stride(0),
          static_cast<const WeightT*>(rms_norm_weight.data_ptr()),
          static_cast<const float*>(cos_sin_cache.data_ptr()),
          cos_sin_cache.stride(0), static_cast<uint8_t*>(kv_cache.data_ptr()),
          kv_cache.stride(0), kv_slot_mapping.data_ptr(),
          index_kind(kv_slot_mapping, "kv_slot_mapping"), rms_eps, num_tokens,
          state_cache.size(0), block_table.size(0), block_table.size(1),
          kv_cache.size(0), kv_cache.size(1), state_width);
}

}  // namespace

void deepseek_v4_sparse_compress_cache(
    torch::Tensor& state_cache, const torch::Tensor& token_to_req_indices,
    const torch::Tensor& positions, const torch::Tensor& state_slot_mapping,
    const torch::Tensor& block_table, const torch::Tensor& rms_norm_weight,
    const torch::Tensor& cos_sin_cache, torch::Tensor& kv_cache,
    const torch::Tensor& kv_slot_mapping, double rms_eps,
    int64_t state_block_size, int64_t state_width, int64_t kv_block_size,
    int64_t compress_ratio, int64_t token_stride, int64_t scale_dim,
    int64_t quant_block, const c10::optional<torch::Tensor>& kv_states,
    const c10::optional<torch::Tensor>& score_states,
    const c10::optional<torch::Tensor>& ape) {
  TORCH_CHECK(state_cache.scalar_type() == torch::kFloat32,
              "state_cache must be float32");
  TORCH_CHECK(token_stride == kTokenStride, "sparse compressor requires 576-byte tokens");
  TORCH_CHECK(scale_dim == kScaleDim, "sparse compressor requires 8-byte UE8M0 scales");
  TORCH_CHECK(quant_block == kQuantBlock,
              "sparse compressor requires 64-wide UE8M0 groups");
  TORCH_CHECK(compress_ratio == 4 || compress_ratio == 128,
              "sparse compressor supports compress_ratio 4 or 128");
  const int64_t expected_block =
      compress_ratio == 4 ? int64_t{4} : int64_t{8};
  const int64_t expected_width =
      compress_ratio == 4 ? int64_t{1024} : int64_t{512};
  const int64_t expected_row = 2 * expected_width;
  TORCH_CHECK(state_block_size == expected_block && state_width == expected_width,
              "state block/width does not match compress_ratio");
  TORCH_CHECK(state_cache.dim() == 3 &&
                  state_cache.size(1) == expected_block &&
                  state_cache.size(2) == expected_row,
              "state_cache shape does not match the 512-d sparse compressor");
  TORCH_CHECK(state_cache.stride(2) == 1 && state_cache.stride(1) % 4 == 0 &&
                  state_cache.stride(0) % 4 == 0,
              "state_cache rows must support aligned float4 loads");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(state_cache.data_ptr()) % 16 == 0,
              "state_cache must be 16-byte aligned");
  TORCH_CHECK(state_slot_mapping.numel() > 0 &&
                  state_slot_mapping.numel() <= kMaxDecodeRows,
              "sparse native path supports 1..128 decode rows");

  const int64_t num_tokens = state_slot_mapping.numel();
  TORCH_CHECK(token_to_req_indices.numel() >= num_tokens,
              "token_to_req_indices does not cover every decode row");
  TORCH_CHECK(positions.numel() >= num_tokens,
              "positions does not cover every decode row");
  TORCH_CHECK(kv_slot_mapping.numel() >= num_tokens,
              "kv_slot_mapping does not cover every decode row");
  TORCH_CHECK(token_to_req_indices.is_contiguous() && positions.is_contiguous() &&
                  state_slot_mapping.is_contiguous() &&
                  kv_slot_mapping.is_contiguous(),
              "index tensors must be contiguous");
  TORCH_CHECK(block_table.dim() == 2 && block_table.stride(1) == 1 &&
                  block_table.size(0) > 0 && block_table.size(1) > 0,
              "block_table must be a non-empty row-contiguous 2D tensor");
  TORCH_CHECK(rms_norm_weight.dim() == 1 &&
                  rms_norm_weight.numel() == kHeadDim &&
                  rms_norm_weight.is_contiguous(),
              "rms_norm_weight must be contiguous with shape [512]");
  TORCH_CHECK(rms_norm_weight.scalar_type() == torch::kFloat32 ||
                  rms_norm_weight.scalar_type() == torch::kBFloat16,
              "rms_norm_weight must be float32 or bfloat16");
  TORCH_CHECK(cos_sin_cache.scalar_type() == torch::kFloat32 &&
                  cos_sin_cache.dim() == 2 &&
                  cos_sin_cache.size(1) == kRopeDim &&
                  cos_sin_cache.stride(1) == 1,
              "cos_sin_cache must be float32 [max_position, 64]");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8 && kv_cache.dim() >= 2 &&
                  kv_cache.size(0) > 0 && kv_cache.size(1) == kv_block_size &&
                  kv_cache.stride(-1) == 1,
              "kv_cache must be a uint8 paged cache with the requested block size");
  TORCH_CHECK(kv_block_size > 0, "kv_block_size must be positive");
  TORCH_CHECK(kv_cache.stride(0) >=
                  kv_block_size * (kTokenValueBytes + kTokenScaleBytes),
              "kv_cache page stride is smaller than the 584-byte token layout");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(kv_cache.data_ptr()) % 4 == 0 &&
                  kv_cache.stride(0) % 4 == 0,
              "kv_cache must support aligned fp8x4 stores");

  check_same_device(state_cache, token_to_req_indices, "token_to_req_indices");
  check_same_device(state_cache, positions, "positions");
  check_same_device(state_cache, state_slot_mapping, "state_slot_mapping");
  check_same_device(state_cache, block_table, "block_table");
  check_same_device(state_cache, rms_norm_weight, "rms_norm_weight");
  check_same_device(state_cache, cos_sin_cache, "cos_sin_cache");
  check_same_device(state_cache, kv_cache, "kv_cache");
  check_same_device(state_cache, kv_slot_mapping, "kv_slot_mapping");

  const bool fuse_save = kv_states.has_value();
  TORCH_CHECK(fuse_save == score_states.has_value() &&
                  fuse_save == ape.has_value(),
              "kv_states, score_states, and ape must be supplied together");
  const float* kv_ptr = nullptr;
  const float* score_ptr = nullptr;
  const float* ape_ptr = nullptr;
  int64_t kv_stride = 0;
  int64_t score_stride = 0;
  int64_t ape_stride = 0;
  if (fuse_save) {
    const torch::Tensor& kv_tensor = kv_states.value();
    const torch::Tensor& score_tensor = score_states.value();
    const torch::Tensor& ape_tensor = ape.value();
    TORCH_CHECK(kv_tensor.scalar_type() == torch::kFloat32 &&
                    score_tensor.scalar_type() == torch::kFloat32 &&
                    ape_tensor.scalar_type() == torch::kFloat32,
                "fused save inputs must be float32");
    TORCH_CHECK(kv_tensor.dim() == 2 && score_tensor.dim() == 2 &&
                    kv_tensor.size(0) >= num_tokens &&
                    score_tensor.size(0) >= num_tokens &&
                    kv_tensor.size(1) == state_width &&
                    score_tensor.size(1) == state_width &&
                    kv_tensor.stride(1) == 1 && score_tensor.stride(1) == 1,
                "fused save kv/score must be [tokens, state_width]");
    TORCH_CHECK(ape_tensor.dim() == 2 && ape_tensor.size(0) == compress_ratio &&
                    ape_tensor.size(1) == state_width &&
                    ape_tensor.stride(1) == 1,
                "fused save ape must be [compress_ratio, state_width]");
    check_same_device(state_cache, kv_tensor, "kv_states");
    check_same_device(state_cache, score_tensor, "score_states");
    check_same_device(state_cache, ape_tensor, "ape");
    kv_ptr = static_cast<const float*>(kv_tensor.data_ptr());
    score_ptr = static_cast<const float*>(score_tensor.data_ptr());
    ape_ptr = static_cast<const float*>(ape_tensor.data_ptr());
    kv_stride = kv_tensor.stride(0);
    score_stride = score_tensor.stride(0);
    ape_stride = ape_tensor.stride(0);
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(state_cache));
  musaStream_t stream = at::musa::getCurrentMUSAStream();

  if (fuse_save) {
    if (compress_ratio == 4) {
      launch_sparse_save_partial<4, true, 4>(
          state_cache, positions, state_slot_mapping, state_width, kv_ptr,
          kv_stride, score_ptr, score_stride, ape_ptr, ape_stride, stream);
    } else {
      launch_sparse_save_partial<128, false, 8>(
          state_cache, positions, state_slot_mapping, state_width, kv_ptr,
          kv_stride, score_ptr, score_stride, ape_ptr, ape_stride, stream);
    }
  }

#define LAUNCH_SPARSE(WEIGHT_T, RATIO, OVERLAP, BLOCK)                         \
  launch_sparse_compressor<WEIGHT_T, RATIO, OVERLAP, BLOCK>(                   \
      state_cache, token_to_req_indices, positions, state_slot_mapping,        \
      block_table, rms_norm_weight, cos_sin_cache, kv_cache, kv_slot_mapping,  \
      static_cast<float>(rms_eps), state_width, stream)

  if (rms_norm_weight.scalar_type() == torch::kFloat32) {
    if (compress_ratio == 4) {
      LAUNCH_SPARSE(float, 4, true, 4);
    } else {
      LAUNCH_SPARSE(float, 128, false, 8);
    }
  } else {
    if (compress_ratio == 4) {
      LAUNCH_SPARSE(__mt_bfloat16, 4, true, 4);
    } else {
      LAUNCH_SPARSE(__mt_bfloat16, 128, false, 8);
    }
  }

#undef LAUNCH_SPARSE

  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess,
              "deepseek_v4_sparse_compress_cache launch failed: ",
              musaGetErrorString(err));
}
