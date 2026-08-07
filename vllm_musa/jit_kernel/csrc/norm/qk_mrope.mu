#include <cstdint>

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/musa/device_guard.h>
#include <tvm/ffi/function.h>

#include "../common.h"
#include "../device_utils.h"
#include "common.mu"

void check_fused_qk_mrope_inputs(ffi::TensorView q, ffi::TensorView k,
                                 ffi::TensorView q_weight,
                                 ffi::TensorView k_weight,
                                 ffi::TensorView positions,
                                 ffi::TensorView cos_sin_cache,
                                 ffi::TensorView q_out, ffi::TensorView k_out) {
  CHECK_MUSA(q);
  CHECK_MUSA(k);
  CHECK_MUSA(q_weight);
  CHECK_MUSA(k_weight);
  CHECK_MUSA(positions);
  CHECK_MUSA(cos_sin_cache);
  CHECK_MUSA(q_out);
  CHECK_MUSA(k_out);
  TVM_FFI_ICHECK_EQ(q.ndim(), 3);
  TVM_FFI_ICHECK_EQ(k.ndim(), 3);
  TVM_FFI_ICHECK_EQ(q_out.ndim(), 3);
  TVM_FFI_ICHECK_EQ(k_out.ndim(), 3);
  TVM_FFI_ICHECK_EQ(positions.ndim(), 2);
  TVM_FFI_ICHECK_EQ(positions.size(0), 3);
  TVM_FFI_ICHECK_EQ(positions.size(1), q.size(0));
  TVM_FFI_ICHECK_EQ(cos_sin_cache.ndim(), 2);
  TVM_FFI_ICHECK_EQ(q.size(2), k.size(2));
  TVM_FFI_ICHECK_EQ(q.size(2) % 2, 0);
  TVM_FFI_ICHECK_EQ(q_weight.size(0), q.size(2));
  TVM_FFI_ICHECK_EQ(k_weight.size(0), k.size(2));
  TVM_FFI_ICHECK_GT(cos_sin_cache.size(1), 0);
  TVM_FFI_ICHECK_LE(cos_sin_cache.size(1), q.size(2));
  TVM_FFI_ICHECK_EQ(cos_sin_cache.size(1) % 2, 0);
  TVM_FFI_ICHECK_EQ(q_out.size(0), q.size(0));
  TVM_FFI_ICHECK_EQ(q_out.size(1), q.size(1));
  TVM_FFI_ICHECK_EQ(q_out.size(2), q.size(2));
  TVM_FFI_ICHECK_EQ(k_out.size(0), k.size(0));
  TVM_FFI_ICHECK_EQ(k_out.size(1), k.size(1));
  TVM_FFI_ICHECK_EQ(k_out.size(2), k.size(2));
  TVM_FFI_ICHECK_EQ(q.stride(2), 1);
  TVM_FFI_ICHECK_EQ(k.stride(2), 1);
  TVM_FFI_ICHECK_EQ(q_out.stride(2), 1);
  TVM_FFI_ICHECK_EQ(k_out.stride(2), 1);
  TVM_FFI_ICHECK_EQ(cos_sin_cache.stride(1), 1);
  TVM_FFI_ICHECK_EQ(q.device().device_id, k.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, q_weight.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, k_weight.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, positions.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, cos_sin_cache.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, q_out.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, k_out.device().device_id);
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), dl_bfloat16));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), k.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), q_weight.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), k_weight.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), cos_sin_cache.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), q_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), k_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(positions.dtype(), dl_int64));
}

void check_fused_qk_mrope_cache_inputs(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
    ffi::TensorView q_weight, ffi::TensorView k_weight,
    ffi::TensorView positions, ffi::TensorView cos_sin_cache,
    ffi::TensorView q_out, ffi::TensorView k_cache, ffi::TensorView v_cache,
    ffi::TensorView indices) {
  CHECK_MUSA(q);
  CHECK_MUSA(k);
  CHECK_MUSA(v);
  CHECK_MUSA(q_weight);
  CHECK_MUSA(k_weight);
  CHECK_MUSA(positions);
  CHECK_MUSA(cos_sin_cache);
  CHECK_MUSA(q_out);
  CHECK_MUSA(k_cache);
  CHECK_MUSA(v_cache);
  CHECK_MUSA(indices);
  TVM_FFI_ICHECK_EQ(q.ndim(), 3);
  TVM_FFI_ICHECK_EQ(k.ndim(), 3);
  TVM_FFI_ICHECK_EQ(v.ndim(), 3);
  TVM_FFI_ICHECK_EQ(q_out.ndim(), 3);
  TVM_FFI_ICHECK(k_cache.ndim() == 2 || k_cache.ndim() == 4);
  TVM_FFI_ICHECK_EQ(v_cache.ndim(), k_cache.ndim());
  TVM_FFI_ICHECK_EQ(positions.ndim(), 2);
  TVM_FFI_ICHECK_EQ(indices.ndim(), 1);
  TVM_FFI_ICHECK_EQ(positions.size(0), 3);
  TVM_FFI_ICHECK_EQ(positions.size(1), q.size(0));
  TVM_FFI_ICHECK_EQ(indices.size(0), q.size(0));
  TVM_FFI_ICHECK_EQ(cos_sin_cache.ndim(), 2);
  TVM_FFI_ICHECK_EQ(k.size(1), v.size(1));
  TVM_FFI_ICHECK_EQ(q.size(2), k.size(2));
  TVM_FFI_ICHECK_EQ(k.size(2), v.size(2));
  TVM_FFI_ICHECK_EQ(q.size(2) % 2, 0);
  TVM_FFI_ICHECK_EQ(q_weight.size(0), q.size(2));
  TVM_FFI_ICHECK_EQ(k_weight.size(0), k.size(2));
  TVM_FFI_ICHECK_GT(cos_sin_cache.size(1), 0);
  TVM_FFI_ICHECK_LE(cos_sin_cache.size(1), q.size(2));
  TVM_FFI_ICHECK_EQ(cos_sin_cache.size(1) % 2, 0);
  if (k_cache.ndim() == 2) {
    TVM_FFI_ICHECK_EQ(k_cache.size(1), k.size(1) * k.size(2));
    TVM_FFI_ICHECK_EQ(v_cache.size(1), v.size(1) * v.size(2));
  } else {
    TVM_FFI_ICHECK_EQ(k_cache.size(0), v_cache.size(0));
    TVM_FFI_ICHECK_EQ(k_cache.size(1), v_cache.size(1));
    TVM_FFI_ICHECK_EQ(k_cache.size(2), k.size(1));
    TVM_FFI_ICHECK_EQ(v_cache.size(2), v.size(1));
    TVM_FFI_ICHECK_EQ(k_cache.size(3), k.size(2));
    TVM_FFI_ICHECK_EQ(v_cache.size(3), v.size(2));
  }
  TVM_FFI_ICHECK_EQ(q_out.size(0), q.size(0));
  TVM_FFI_ICHECK_EQ(q_out.size(1), q.size(1));
  TVM_FFI_ICHECK_EQ(q_out.size(2), q.size(2));
  TVM_FFI_ICHECK_EQ(q.stride(2), 1);
  TVM_FFI_ICHECK_EQ(k.stride(2), 1);
  TVM_FFI_ICHECK_EQ(v.stride(2), 1);
  TVM_FFI_ICHECK_EQ(q_out.stride(2), 1);
  if (k_cache.ndim() == 2) {
    TVM_FFI_ICHECK_EQ(k_cache.stride(1), 1);
    TVM_FFI_ICHECK_EQ(v_cache.stride(1), 1);
  } else {
    TVM_FFI_ICHECK_EQ(k_cache.stride(3), 1);
    TVM_FFI_ICHECK_EQ(v_cache.stride(3), 1);
    TVM_FFI_ICHECK_EQ(k_cache.stride(2), k_cache.size(3));
    TVM_FFI_ICHECK_EQ(v_cache.stride(2), v_cache.size(3));
    TVM_FFI_ICHECK_EQ(k_cache.stride(1), k_cache.size(2) * k_cache.size(3));
    TVM_FFI_ICHECK_EQ(v_cache.stride(1), v_cache.size(2) * v_cache.size(3));
  }
  TVM_FFI_ICHECK_EQ(cos_sin_cache.stride(1), 1);
  TVM_FFI_ICHECK_EQ(q.device().device_id, k.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, v.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, q_weight.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, k_weight.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, positions.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, cos_sin_cache.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, q_out.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, k_cache.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, v_cache.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, indices.device().device_id);
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), dl_bfloat16));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), k.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), v.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), q_weight.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), k_weight.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), cos_sin_cache.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), q_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), k_cache.dtype()));
  TVM_FFI_ICHECK(dtype_equal(q.dtype(), v_cache.dtype()));
  TVM_FFI_ICHECK(dtype_equal(positions.dtype(), dl_int64));
  TVM_FFI_ICHECK(dtype_equal(indices.dtype(), dl_int32) ||
                 dtype_equal(indices.dtype(), dl_int64));
}


template <bool GEMMA, bool INTERLEAVED>
__global__ void fused_qk_rmsnorm_mrope_32q4kv_h128_bf16_kernel(
    const __mt_bfloat16 *__restrict__ q, const __mt_bfloat16 *__restrict__ k,
    const __mt_bfloat16 *__restrict__ q_weight,
    const __mt_bfloat16 *__restrict__ k_weight,
    const int64_t *__restrict__ positions,
    const __mt_bfloat16 *__restrict__ cos_sin_cache,
    __mt_bfloat16 *__restrict__ q_out, __mt_bfloat16 *__restrict__ k_out,
    int batch, int64_t position_stride, int64_t q_batch_stride,
    int64_t q_head_stride, int64_t k_batch_stride, int64_t k_head_stride,
    int64_t q_out_batch_stride, int64_t q_out_head_stride,
    int64_t k_out_batch_stride, int64_t k_out_head_stride, float eps) {
  constexpr int q_heads = 32;
  constexpr int k_heads = 4;
  constexpr int hidden = 128;
  constexpr int embed_dim = 64;
  constexpr int rot_dim = 128;
  constexpr int heads_per_block = 8;
  constexpr int groups_per_token = 5;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int token = (int)blockIdx.x / groups_per_token;
  const int group = (int)blockIdx.x - token * groups_per_token;
  if (token >= batch) {
    return;
  }

  const int global_head = group * heads_per_block + warp;
  if (global_head >= q_heads + k_heads) {
    return;
  }

  const bool is_q = global_head < q_heads;
  const int head = is_q ? global_head : global_head - q_heads;
  const __mt_bfloat16 *__restrict__ data = is_q ? q : k;
  const __mt_bfloat16 *__restrict__ weight = is_q ? q_weight : k_weight;
  __mt_bfloat16 *__restrict__ out = is_q ? q_out : k_out;
  const int64_t base =
      is_q ? ((int64_t)token * q_batch_stride + (int64_t)head * q_head_stride)
           : ((int64_t)token * k_batch_stride + (int64_t)head * k_head_stride);
  const int64_t out_base =
      is_q ? ((int64_t)token * q_out_batch_stride +
              (int64_t)head * q_out_head_stride)
           : ((int64_t)token * k_out_batch_stride +
              (int64_t)head * k_out_head_stride);

  const float data_0 = __bfloat162float(data[base + lane]);
  const float data_1 = __bfloat162float(data[base + lane + 32]);
  const float data_2 = __bfloat162float(data[base + lane + 64]);
  const float data_3 = __bfloat162float(data[base + lane + 96]);
  float sum =
      data_0 * data_0 + data_1 * data_1 + data_2 * data_2 + data_3 * data_3;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, mask);
  }
  const float scale = fast_rsqrt(sum * (1.0f / 128.0f) + eps);

  const int rot_offset0 = lane;
  const int axis0 = INTERLEAVED
                        ? mrope_24_20_20_interleaved_axis(rot_offset0)
                        : (rot_offset0 < 24 ? 0 : (rot_offset0 < 44 ? 1 : 2));
  const int64_t pos0 = positions[(int64_t)axis0 * position_stride + token];
  const __mt_bfloat16 *__restrict__ cache_ptr0 = cos_sin_cache + pos0 * rot_dim;
  const float cos_v0 = __bfloat162float(cache_ptr0[rot_offset0]);
  const float sin_v0 = __bfloat162float(cache_ptr0[embed_dim + rot_offset0]);
  const int x_index0 = rot_offset0;
  const int y_index0 = embed_dim + rot_offset0;
  const float x_weight0 =
      __bfloat162float(weight[x_index0]) + (GEMMA ? 1.0f : 0.0f);
  const float y_weight0 =
      __bfloat162float(weight[y_index0]) + (GEMMA ? 1.0f : 0.0f);
  const float x0 = data_0 * scale * x_weight0;
  const float y0 = data_2 * scale * y_weight0;
  out[out_base + x_index0] = __float2bfloat16_rn(x0 * cos_v0 - y0 * sin_v0);
  out[out_base + y_index0] = __float2bfloat16_rn(y0 * cos_v0 + x0 * sin_v0);

  const int rot_offset1 = lane + 32;
  const int axis1 = INTERLEAVED
                        ? mrope_24_20_20_interleaved_axis(rot_offset1)
                        : (rot_offset1 < 44 ? 1 : 2);
  const int64_t pos1 = positions[(int64_t)axis1 * position_stride + token];
  const __mt_bfloat16 *__restrict__ cache_ptr1 = cos_sin_cache + pos1 * rot_dim;
  const float cos_v1 = __bfloat162float(cache_ptr1[rot_offset1]);
  const float sin_v1 = __bfloat162float(cache_ptr1[embed_dim + rot_offset1]);
  const int x_index1 = rot_offset1;
  const int y_index1 = embed_dim + rot_offset1;
  const float x_weight1 =
      __bfloat162float(weight[x_index1]) + (GEMMA ? 1.0f : 0.0f);
  const float y_weight1 =
      __bfloat162float(weight[y_index1]) + (GEMMA ? 1.0f : 0.0f);
  const float x1 = data_1 * scale * x_weight1;
  const float y1 = data_3 * scale * y_weight1;
  out[out_base + x_index1] = __float2bfloat16_rn(x1 * cos_v1 - y1 * sin_v1);
  out[out_base + y_index1] = __float2bfloat16_rn(y1 * cos_v1 + x1 * sin_v1);
}

template <typename index_t, bool GEMMA, bool INTERLEAVED,
          bool STORE_K_OUT = true>
__global__ void fused_qk_rmsnorm_mrope_cache_32q4kv_h128_bf16_kernel(
    const __mt_bfloat16 *__restrict__ q, const __mt_bfloat16 *__restrict__ k,
    const __mt_bfloat16 *__restrict__ v,
    const __mt_bfloat16 *__restrict__ q_weight,
    const __mt_bfloat16 *__restrict__ k_weight,
    const int64_t *__restrict__ positions,
    const __mt_bfloat16 *__restrict__ cos_sin_cache,
    __mt_bfloat16 *__restrict__ q_out, __mt_bfloat16 *__restrict__ k_out,
    __mt_bfloat16 *__restrict__ k_cache, __mt_bfloat16 *__restrict__ v_cache,
    const index_t *__restrict__ indices, int batch, int64_t position_stride,
    int64_t q_batch_stride, int64_t q_head_stride, int64_t k_batch_stride,
    int64_t k_head_stride, int64_t v_batch_stride, int64_t v_head_stride,
    int64_t q_out_batch_stride, int64_t q_out_head_stride,
    int64_t k_out_batch_stride, int64_t k_out_head_stride,
    int64_t k_cache_row_stride, int64_t v_cache_row_stride,
    int64_t indices_stride, float eps) {
  constexpr int q_heads = 32;
  constexpr int k_heads = 4;
  constexpr int hidden = 128;
  constexpr int embed_dim = 64;
  constexpr int rot_dim = 128;
  constexpr int heads_per_block = 8;
  constexpr int groups_per_token = 5;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int token = (int)blockIdx.x / groups_per_token;
  const int group = (int)blockIdx.x - token * groups_per_token;
  if (token >= batch) {
    return;
  }

  const int64_t cache_idx =
      static_cast<int64_t>(indices[(int64_t)token * indices_stride]);
  const bool write_cache = cache_idx >= 0;

  if (group == 0 && write_cache) {
    constexpr int kVec = 8;
    constexpr int row_dim = k_heads * hidden;
    constexpr int vec_count = row_dim / kVec;
    const int64_t v_token_base = (int64_t)token * v_batch_stride;
    const int64_t v_out_base = cache_idx * v_cache_row_stride;
    for (int vec_idx = tid; vec_idx < vec_count; vec_idx += (int)blockDim.x) {
      const int head = vec_idx >> 4;
      const int col = (vec_idx & 15) * kVec;
      Vec8<__mt_bfloat16> v_vec = Vec8<__mt_bfloat16>::load(
          v + v_token_base + (int64_t)head * v_head_stride, col);
      *(Vec8<__mt_bfloat16> *)(v_cache + v_out_base + (int64_t)head * hidden +
                               col) = v_vec;
    }
  }

  const int global_head = group * heads_per_block + warp;
  if (global_head >= q_heads + k_heads) {
    return;
  }

  const bool is_q = global_head < q_heads;
  const int head = is_q ? global_head : global_head - q_heads;
  const __mt_bfloat16 *__restrict__ data = is_q ? q : k;
  const __mt_bfloat16 *__restrict__ weight = is_q ? q_weight : k_weight;
  const int64_t base =
      is_q ? ((int64_t)token * q_batch_stride + (int64_t)head * q_head_stride)
           : ((int64_t)token * k_batch_stride + (int64_t)head * k_head_stride);
  const int64_t q_out_base =
      (int64_t)token * q_out_batch_stride + (int64_t)head * q_out_head_stride;
  const int64_t k_out_base =
      (int64_t)token * k_out_batch_stride + (int64_t)head * k_out_head_stride;

  const float data_0 = __bfloat162float(data[base + lane]);
  const float data_1 = __bfloat162float(data[base + lane + 32]);
  const float data_2 = __bfloat162float(data[base + lane + 64]);
  const float data_3 = __bfloat162float(data[base + lane + 96]);
  float sum =
      data_0 * data_0 + data_1 * data_1 + data_2 * data_2 + data_3 * data_3;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, mask);
  }
  const float scale = fast_rsqrt(sum * (1.0f / 128.0f) + eps);

  const int rot_offset0 = lane;
  const int axis0 = INTERLEAVED
                        ? mrope_24_20_20_interleaved_axis(rot_offset0)
                        : (rot_offset0 < 24 ? 0 : (rot_offset0 < 44 ? 1 : 2));
  const int64_t pos0 = positions[(int64_t)axis0 * position_stride + token];
  const __mt_bfloat16 *__restrict__ cache_ptr0 = cos_sin_cache + pos0 * rot_dim;
  const float cos_v0 = __bfloat162float(cache_ptr0[rot_offset0]);
  const float sin_v0 = __bfloat162float(cache_ptr0[embed_dim + rot_offset0]);
  const int x_index0 = rot_offset0;
  const int y_index0 = embed_dim + rot_offset0;
  const float x_weight0 =
      __bfloat162float(weight[x_index0]) + (GEMMA ? 1.0f : 0.0f);
  const float y_weight0 =
      __bfloat162float(weight[y_index0]) + (GEMMA ? 1.0f : 0.0f);
  const float x0 = data_0 * scale * x_weight0;
  const float y0 = data_2 * scale * y_weight0;
  const __mt_bfloat16 x_rot0 = __float2bfloat16_rn(x0 * cos_v0 - y0 * sin_v0);
  const __mt_bfloat16 y_rot0 = __float2bfloat16_rn(y0 * cos_v0 + x0 * sin_v0);
  if (is_q) {
    q_out[q_out_base + x_index0] = x_rot0;
    q_out[q_out_base + y_index0] = y_rot0;
  } else {
    if (write_cache) {
      const int64_t k_cache_base =
          cache_idx * k_cache_row_stride + (int64_t)head * hidden;
      k_cache[k_cache_base + x_index0] = x_rot0;
      k_cache[k_cache_base + y_index0] = y_rot0;
    }
    if constexpr (STORE_K_OUT) {
      k_out[k_out_base + x_index0] = x_rot0;
      k_out[k_out_base + y_index0] = y_rot0;
    }
  }

  const int rot_offset1 = lane + 32;
  const int axis1 = INTERLEAVED
                        ? mrope_24_20_20_interleaved_axis(rot_offset1)
                        : (rot_offset1 < 44 ? 1 : 2);
  const int64_t pos1 = positions[(int64_t)axis1 * position_stride + token];
  const __mt_bfloat16 *__restrict__ cache_ptr1 = cos_sin_cache + pos1 * rot_dim;
  const float cos_v1 = __bfloat162float(cache_ptr1[rot_offset1]);
  const float sin_v1 = __bfloat162float(cache_ptr1[embed_dim + rot_offset1]);
  const int x_index1 = rot_offset1;
  const int y_index1 = embed_dim + rot_offset1;
  const float x_weight1 =
      __bfloat162float(weight[x_index1]) + (GEMMA ? 1.0f : 0.0f);
  const float y_weight1 =
      __bfloat162float(weight[y_index1]) + (GEMMA ? 1.0f : 0.0f);
  const float x1 = data_1 * scale * x_weight1;
  const float y1 = data_3 * scale * y_weight1;
  const __mt_bfloat16 x_rot1 = __float2bfloat16_rn(x1 * cos_v1 - y1 * sin_v1);
  const __mt_bfloat16 y_rot1 = __float2bfloat16_rn(y1 * cos_v1 + x1 * sin_v1);
  if (is_q) {
    q_out[q_out_base + x_index1] = x_rot1;
    q_out[q_out_base + y_index1] = y_rot1;
  } else {
    if (write_cache) {
      const int64_t k_cache_base =
          cache_idx * k_cache_row_stride + (int64_t)head * hidden;
      k_cache[k_cache_base + x_index1] = x_rot1;
      k_cache[k_cache_base + y_index1] = y_rot1;
    }
    if constexpr (STORE_K_OUT) {
      k_out[k_out_base + x_index1] = x_rot1;
      k_out[k_out_base + y_index1] = y_rot1;
    }
  }
}

template <typename index_t, int Q_HEADS, int K_HEADS, bool GEMMA,
          int HEADS_PER_BLOCK = 8, bool STORE_K_OUT = true,
          bool SAME_POSITION = false>
__global__ void fused_qk_rmsnorm_mrope_cache_h128_full_bf16_kernel(
    const __mt_bfloat16 *__restrict__ q, const __mt_bfloat16 *__restrict__ k,
    const __mt_bfloat16 *__restrict__ v,
    const __mt_bfloat16 *__restrict__ q_weight,
    const __mt_bfloat16 *__restrict__ k_weight,
    const int64_t *__restrict__ positions,
    const __mt_bfloat16 *__restrict__ cos_sin_cache,
    __mt_bfloat16 *__restrict__ q_out, __mt_bfloat16 *__restrict__ k_out,
    __mt_bfloat16 *__restrict__ k_cache, __mt_bfloat16 *__restrict__ v_cache,
    const index_t *__restrict__ indices, int batch, int64_t position_stride,
    int64_t q_batch_stride, int64_t q_head_stride, int64_t k_batch_stride,
    int64_t k_head_stride, int64_t v_batch_stride, int64_t v_head_stride,
    int64_t q_out_batch_stride, int64_t q_out_head_stride,
    int64_t k_out_batch_stride, int64_t k_out_head_stride,
    int64_t k_cache_row_stride, int64_t v_cache_row_stride,
    int64_t indices_stride, float eps) {
  constexpr int hidden = 128;
  constexpr int embed_dim = 64;
  constexpr int rot_dim = 128;
  constexpr int heads_per_block = HEADS_PER_BLOCK;
  constexpr int groups_per_token =
      (Q_HEADS + K_HEADS + heads_per_block - 1) / heads_per_block;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int token = (int)blockIdx.x / groups_per_token;
  const int group = (int)blockIdx.x - token * groups_per_token;
  if (token >= batch) {
    return;
  }

  const int64_t cache_idx =
      static_cast<int64_t>(indices[(int64_t)token * indices_stride]);
  const bool write_cache = cache_idx >= 0;

  if (group == 0 && write_cache) {
    constexpr int kVec = 8;
    constexpr int row_dim = K_HEADS * hidden;
    constexpr int vec_count = row_dim / kVec;
    const int64_t v_token_base = (int64_t)token * v_batch_stride;
    const int64_t v_out_base = cache_idx * v_cache_row_stride;
    for (int vec_idx = tid; vec_idx < vec_count; vec_idx += (int)blockDim.x) {
      const int head = vec_idx >> 4;
      const int col = (vec_idx & 15) * kVec;
      Vec8<__mt_bfloat16> v_vec = Vec8<__mt_bfloat16>::load(
          v + v_token_base + (int64_t)head * v_head_stride, col);
      *(Vec8<__mt_bfloat16> *)(v_cache + v_out_base + (int64_t)head * hidden +
                               col) = v_vec;
    }
  }

  const int global_head = group * heads_per_block + warp;
  if (global_head >= Q_HEADS + K_HEADS) {
    return;
  }

  const bool is_q = global_head < Q_HEADS;
  const int head = is_q ? global_head : global_head - Q_HEADS;
  const __mt_bfloat16 *__restrict__ data = is_q ? q : k;
  const __mt_bfloat16 *__restrict__ weight = is_q ? q_weight : k_weight;
  const int64_t base =
      is_q ? ((int64_t)token * q_batch_stride + (int64_t)head * q_head_stride)
           : ((int64_t)token * k_batch_stride + (int64_t)head * k_head_stride);

  const float data_0 = __bfloat162float(data[base + lane]);
  const float data_1 = __bfloat162float(data[base + lane + 32]);
  const float data_2 = __bfloat162float(data[base + lane + 64]);
  const float data_3 = __bfloat162float(data[base + lane + 96]);
  float sum =
      data_0 * data_0 + data_1 * data_1 + data_2 * data_2 + data_3 * data_3;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, mask);
  }
  const float scale = fast_rsqrt(sum * (1.0f / 128.0f) + eps);

  const int rot_offset0 = lane;
  int64_t pos0;
  int64_t pos1;
  if constexpr (SAME_POSITION) {
    pos0 = positions[token];
    pos1 = pos0;
  } else {
    constexpr unsigned int axis0_mask1 = 0x92492492U;
    constexpr unsigned int axis0_mask2 = 0x24924924U;
    constexpr unsigned int axis1_mask1 = 0x04924924U;
    constexpr unsigned int axis1_mask2 = 0x09249249U;
    const unsigned int lane_bit = 1U << lane;
    const int axis0 =
        ((axis0_mask1 & lane_bit) != 0U) +
        (((axis0_mask2 & lane_bit) != 0U) << 1);
    pos0 = positions[(int64_t)axis0 * position_stride + token];
    const int axis1 =
        ((axis1_mask1 & lane_bit) != 0U) +
        (((axis1_mask2 & lane_bit) != 0U) << 1);
    pos1 = positions[(int64_t)axis1 * position_stride + token];
  }
  const __mt_bfloat16 *__restrict__ cache_ptr0 = cos_sin_cache + pos0 * rot_dim;
  const float cos_v0 = __bfloat162float(cache_ptr0[rot_offset0]);
  const float sin_v0 = __bfloat162float(cache_ptr0[embed_dim + rot_offset0]);
  const float x_weight0 =
      __bfloat162float(weight[rot_offset0]) + (GEMMA ? 1.0f : 0.0f);
  const float y_weight0 =
      __bfloat162float(weight[embed_dim + rot_offset0]) +
      (GEMMA ? 1.0f : 0.0f);
  const float x0 = data_0 * scale * x_weight0;
  const float y0 = data_2 * scale * y_weight0;
  const __mt_bfloat16 x_rot0 = __float2bfloat16_rn(x0 * cos_v0 - y0 * sin_v0);
  const __mt_bfloat16 y_rot0 = __float2bfloat16_rn(y0 * cos_v0 + x0 * sin_v0);

  const int rot_offset1 = lane + 32;
  const __mt_bfloat16 *__restrict__ cache_ptr1 = cos_sin_cache + pos1 * rot_dim;
  const float cos_v1 = __bfloat162float(cache_ptr1[rot_offset1]);
  const float sin_v1 = __bfloat162float(cache_ptr1[embed_dim + rot_offset1]);
  const float x_weight1 =
      __bfloat162float(weight[rot_offset1]) + (GEMMA ? 1.0f : 0.0f);
  const float y_weight1 =
      __bfloat162float(weight[embed_dim + rot_offset1]) +
      (GEMMA ? 1.0f : 0.0f);
  const float x1 = data_1 * scale * x_weight1;
  const float y1 = data_3 * scale * y_weight1;
  const __mt_bfloat16 x_rot1 = __float2bfloat16_rn(x1 * cos_v1 - y1 * sin_v1);
  const __mt_bfloat16 y_rot1 = __float2bfloat16_rn(y1 * cos_v1 + x1 * sin_v1);

  if (is_q) {
    const int64_t q_out_base =
        (int64_t)token * q_out_batch_stride + (int64_t)head * q_out_head_stride;
    q_out[q_out_base + rot_offset0] = x_rot0;
    q_out[q_out_base + embed_dim + rot_offset0] = y_rot0;
    q_out[q_out_base + rot_offset1] = x_rot1;
    q_out[q_out_base + embed_dim + rot_offset1] = y_rot1;
  } else {
    if (write_cache) {
      const int64_t k_cache_base =
          cache_idx * k_cache_row_stride + (int64_t)head * hidden;
      k_cache[k_cache_base + rot_offset0] = x_rot0;
      k_cache[k_cache_base + embed_dim + rot_offset0] = y_rot0;
      k_cache[k_cache_base + rot_offset1] = x_rot1;
      k_cache[k_cache_base + embed_dim + rot_offset1] = y_rot1;
    }
    if constexpr (STORE_K_OUT) {
      const int64_t k_out_base = (int64_t)token * k_out_batch_stride +
                                 (int64_t)head * k_out_head_stride;
      k_out[k_out_base + rot_offset0] = x_rot0;
      k_out[k_out_base + embed_dim + rot_offset0] = y_rot0;
      k_out[k_out_base + rot_offset1] = x_rot1;
      k_out[k_out_base + embed_dim + rot_offset1] = y_rot1;
    }
  }
}

template <typename index_t, int Q_HEADS, int K_HEADS, bool GEMMA,
          int HEADS_PER_BLOCK = 8, bool STORE_K_OUT = true>
__global__ void fused_qk_rmsnorm_rope_cache_h128_full_bf16_kernel(
    const __mt_bfloat16 *__restrict__ q, const __mt_bfloat16 *__restrict__ k,
    const __mt_bfloat16 *__restrict__ v,
    const __mt_bfloat16 *__restrict__ q_weight,
    const __mt_bfloat16 *__restrict__ k_weight,
    const int64_t *__restrict__ positions,
    const __mt_bfloat16 *__restrict__ cos_sin_cache,
    __mt_bfloat16 *__restrict__ q_out, __mt_bfloat16 *__restrict__ k_out,
    __mt_bfloat16 *__restrict__ k_cache, __mt_bfloat16 *__restrict__ v_cache,
    const index_t *__restrict__ indices, int batch, int64_t position_stride,
    int64_t q_batch_stride, int64_t q_head_stride, int64_t k_batch_stride,
    int64_t k_head_stride, int64_t v_batch_stride, int64_t v_head_stride,
    int64_t q_out_batch_stride, int64_t q_out_head_stride,
    int64_t k_out_batch_stride, int64_t k_out_head_stride,
    int64_t k_cache_row_stride, int64_t v_cache_row_stride,
    int64_t indices_stride, float eps) {
  constexpr int hidden = 128;
  constexpr int embed_dim = 64;
  constexpr int rot_dim = 128;
  constexpr int heads_per_block = HEADS_PER_BLOCK;
  constexpr int groups_per_token =
      (Q_HEADS + K_HEADS + heads_per_block - 1) / heads_per_block;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int token = (int)blockIdx.x / groups_per_token;
  const int group = (int)blockIdx.x - token * groups_per_token;
  if (token >= batch) {
    return;
  }

  const int64_t cache_idx =
      static_cast<int64_t>(indices[(int64_t)token * indices_stride]);
  const bool write_cache = cache_idx >= 0;

  if (group == 0 && write_cache) {
    constexpr int kVec = 8;
    constexpr int row_dim = K_HEADS * hidden;
    constexpr int vec_count = row_dim / kVec;
    const int64_t v_token_base = (int64_t)token * v_batch_stride;
    const int64_t v_out_base = cache_idx * v_cache_row_stride;
    for (int vec_idx = tid; vec_idx < vec_count; vec_idx += (int)blockDim.x) {
      const int head = vec_idx >> 4;
      const int col = (vec_idx & 15) * kVec;
      Vec8<__mt_bfloat16> v_vec = Vec8<__mt_bfloat16>::load(
          v + v_token_base + (int64_t)head * v_head_stride, col);
      *(Vec8<__mt_bfloat16> *)(v_cache + v_out_base + (int64_t)head * hidden +
                               col) = v_vec;
    }
  }

  const int global_head = group * heads_per_block + warp;
  if (global_head >= Q_HEADS + K_HEADS) {
    return;
  }

  const bool is_q = global_head < Q_HEADS;
  const int head = is_q ? global_head : global_head - Q_HEADS;
  const __mt_bfloat16 *__restrict__ data = is_q ? q : k;
  const __mt_bfloat16 *__restrict__ weight = is_q ? q_weight : k_weight;
  const int64_t base =
      is_q ? ((int64_t)token * q_batch_stride + (int64_t)head * q_head_stride)
           : ((int64_t)token * k_batch_stride + (int64_t)head * k_head_stride);

  const float data_0 = __bfloat162float(data[base + lane]);
  const float data_1 = __bfloat162float(data[base + lane + 32]);
  const float data_2 = __bfloat162float(data[base + lane + 64]);
  const float data_3 = __bfloat162float(data[base + lane + 96]);
  float sum =
      data_0 * data_0 + data_1 * data_1 + data_2 * data_2 + data_3 * data_3;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, mask);
  }
  const float scale = fast_rsqrt(sum * (1.0f / 128.0f) + eps);

  const int64_t pos = positions[(int64_t)0 * position_stride + token];
  const __mt_bfloat16 *__restrict__ cache_ptr = cos_sin_cache + pos * rot_dim;

  const float cos_v0 = __bfloat162float(cache_ptr[lane]);
  const float sin_v0 = __bfloat162float(cache_ptr[embed_dim + lane]);
  const float x_weight0 =
      __bfloat162float(weight[lane]) + (GEMMA ? 1.0f : 0.0f);
  const float y_weight0 =
      __bfloat162float(weight[embed_dim + lane]) + (GEMMA ? 1.0f : 0.0f);
  const float x0 = data_0 * scale * x_weight0;
  const float y0 = data_2 * scale * y_weight0;
  const __mt_bfloat16 x_rot0 = __float2bfloat16_rn(x0 * cos_v0 - y0 * sin_v0);
  const __mt_bfloat16 y_rot0 = __float2bfloat16_rn(y0 * cos_v0 + x0 * sin_v0);

  const int rot_offset1 = lane + 32;
  const float cos_v1 = __bfloat162float(cache_ptr[rot_offset1]);
  const float sin_v1 = __bfloat162float(cache_ptr[embed_dim + rot_offset1]);
  const float x_weight1 =
      __bfloat162float(weight[rot_offset1]) + (GEMMA ? 1.0f : 0.0f);
  const float y_weight1 =
      __bfloat162float(weight[embed_dim + rot_offset1]) +
      (GEMMA ? 1.0f : 0.0f);
  const float x1 = data_1 * scale * x_weight1;
  const float y1 = data_3 * scale * y_weight1;
  const __mt_bfloat16 x_rot1 = __float2bfloat16_rn(x1 * cos_v1 - y1 * sin_v1);
  const __mt_bfloat16 y_rot1 = __float2bfloat16_rn(y1 * cos_v1 + x1 * sin_v1);

  if (is_q) {
    const int64_t q_out_base =
        (int64_t)token * q_out_batch_stride + (int64_t)head * q_out_head_stride;
    q_out[q_out_base + lane] = x_rot0;
    q_out[q_out_base + embed_dim + lane] = y_rot0;
    q_out[q_out_base + rot_offset1] = x_rot1;
    q_out[q_out_base + embed_dim + rot_offset1] = y_rot1;
  } else {
    if (write_cache) {
      const int64_t k_cache_base =
          cache_idx * k_cache_row_stride + (int64_t)head * hidden;
      k_cache[k_cache_base + lane] = x_rot0;
      k_cache[k_cache_base + embed_dim + lane] = y_rot0;
      k_cache[k_cache_base + rot_offset1] = x_rot1;
      k_cache[k_cache_base + embed_dim + rot_offset1] = y_rot1;
    }
    if constexpr (STORE_K_OUT) {
      const int64_t k_out_base = (int64_t)token * k_out_batch_stride +
                                 (int64_t)head * k_out_head_stride;
      k_out[k_out_base + lane] = x_rot0;
      k_out[k_out_base + embed_dim + lane] = y_rot0;
      k_out[k_out_base + rot_offset1] = x_rot1;
      k_out[k_out_base + embed_dim + rot_offset1] = y_rot1;
    }
  }
}

template <typename index_t, int Q_HEADS, int K_HEADS, bool GEMMA,
          int HEADS_PER_BLOCK = 8, bool STORE_K_OUT = true>
__global__ void fused_qk_rmsnorm_rope_cache_h128r64_bf16_kernel(
    const __mt_bfloat16 *__restrict__ q, const __mt_bfloat16 *__restrict__ k,
    const __mt_bfloat16 *__restrict__ v,
    const __mt_bfloat16 *__restrict__ q_weight,
    const __mt_bfloat16 *__restrict__ k_weight,
    const int64_t *__restrict__ positions,
    const __mt_bfloat16 *__restrict__ cos_sin_cache,
    __mt_bfloat16 *__restrict__ q_out, __mt_bfloat16 *__restrict__ k_out,
    __mt_bfloat16 *__restrict__ k_cache, __mt_bfloat16 *__restrict__ v_cache,
    const index_t *__restrict__ indices, int batch, int64_t position_stride,
    int64_t q_batch_stride, int64_t q_head_stride, int64_t k_batch_stride,
    int64_t k_head_stride, int64_t v_batch_stride, int64_t v_head_stride,
    int64_t q_out_batch_stride, int64_t q_out_head_stride,
    int64_t k_out_batch_stride, int64_t k_out_head_stride,
    int64_t k_cache_row_stride, int64_t v_cache_row_stride,
    int64_t indices_stride, float eps) {
  constexpr int hidden = 128;
  constexpr int embed_dim = 32;
  constexpr int rot_dim = 64;
  constexpr int warps_per_block = HEADS_PER_BLOCK;
  constexpr int heads_per_block = warps_per_block * 2;
  constexpr int groups_per_token =
      (Q_HEADS + K_HEADS + heads_per_block - 1) / heads_per_block;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int half = lane >> 4;
  const int sublane = lane & 15;
  const int pair_col = sublane * 2;
  const int token = (int)blockIdx.x / groups_per_token;
  const int group = (int)blockIdx.x - token * groups_per_token;
  if (token >= batch) {
    return;
  }

  const int64_t cache_idx =
      static_cast<int64_t>(indices[(int64_t)token * indices_stride]);
  const bool write_cache = cache_idx >= 0;

  if (group == 0 && write_cache) {
    constexpr int kVec = 8;
    constexpr int row_dim = K_HEADS * hidden;
    constexpr int vec_count = row_dim / kVec;
    const int64_t v_token_base = (int64_t)token * v_batch_stride;
    const int64_t v_out_base = cache_idx * v_cache_row_stride;
    for (int vec_idx = tid; vec_idx < vec_count; vec_idx += (int)blockDim.x) {
      const int head = vec_idx >> 4;
      const int col = (vec_idx & 15) * kVec;
      Vec8<__mt_bfloat16> v_vec = Vec8<__mt_bfloat16>::load(
          v + v_token_base + (int64_t)head * v_head_stride, col);
      *(Vec8<__mt_bfloat16> *)(v_cache + v_out_base + (int64_t)head * hidden +
                               col) = v_vec;
    }
  }

  const int global_head = group * heads_per_block + warp * 2 + half;
  if (global_head >= Q_HEADS + K_HEADS) {
    return;
  }

  const bool is_q = global_head < Q_HEADS;
  const int head = is_q ? global_head : global_head - Q_HEADS;
  const __mt_bfloat16 *__restrict__ data = is_q ? q : k;
  const __mt_bfloat16 *__restrict__ weight = is_q ? q_weight : k_weight;
  const int64_t base =
      is_q ? ((int64_t)token * q_batch_stride + (int64_t)head * q_head_stride)
           : ((int64_t)token * k_batch_stride + (int64_t)head * k_head_stride);

  const float2 data_0 = __bfloat1622float2(
      *reinterpret_cast<const __mt_bfloat162 *>(data + base + pair_col));
  const float2 data_1 = __bfloat1622float2(*reinterpret_cast<const __mt_bfloat162 *>(
      data + base + embed_dim + pair_col));
  const float2 data_2 = __bfloat1622float2(*reinterpret_cast<const __mt_bfloat162 *>(
      data + base + 64 + pair_col));
  const float2 data_3 = __bfloat1622float2(*reinterpret_cast<const __mt_bfloat162 *>(
      data + base + 96 + pair_col));
  float sum = data_0.x * data_0.x + data_0.y * data_0.y +
              data_1.x * data_1.x + data_1.y * data_1.y +
              data_2.x * data_2.x + data_2.y * data_2.y +
              data_3.x * data_3.x + data_3.y * data_3.y;
#pragma unroll
  for (int mask = 8; mask > 0; mask >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, mask);
  }
  const float scale = fast_rsqrt(sum * (1.0f / 128.0f) + eps);

  const int64_t pos = positions[(int64_t)0 * position_stride + token];
  const __mt_bfloat16 *__restrict__ cache_ptr = cos_sin_cache + pos * rot_dim;

  const float2 cos_v = __bfloat1622float2(
      *reinterpret_cast<const __mt_bfloat162 *>(cache_ptr + pair_col));
  const float2 sin_v = __bfloat1622float2(*reinterpret_cast<const __mt_bfloat162 *>(
      cache_ptr + embed_dim + pair_col));
  float2 x_weight = __bfloat1622float2(
      *reinterpret_cast<const __mt_bfloat162 *>(weight + pair_col));
  float2 y_weight = __bfloat1622float2(*reinterpret_cast<const __mt_bfloat162 *>(
      weight + embed_dim + pair_col));
  float2 pass_weight0 = __bfloat1622float2(*reinterpret_cast<const __mt_bfloat162 *>(
      weight + 64 + pair_col));
  float2 pass_weight1 = __bfloat1622float2(*reinterpret_cast<const __mt_bfloat162 *>(
      weight + 96 + pair_col));
  if constexpr (GEMMA) {
    x_weight.x += 1.0f;
    x_weight.y += 1.0f;
    y_weight.x += 1.0f;
    y_weight.y += 1.0f;
    pass_weight0.x += 1.0f;
    pass_weight0.y += 1.0f;
    pass_weight1.x += 1.0f;
    pass_weight1.y += 1.0f;
  }
  const float2 x = {data_0.x * scale * x_weight.x,
                    data_0.y * scale * x_weight.y};
  const float2 y = {data_1.x * scale * y_weight.x,
                    data_1.y * scale * y_weight.y};
  const float2 x_rot_f = {x.x * cos_v.x - y.x * sin_v.x,
                          x.y * cos_v.y - y.y * sin_v.y};
  const float2 y_rot_f = {y.x * cos_v.x + x.x * sin_v.x,
                          y.y * cos_v.y + x.y * sin_v.y};
  const float2 pass0_f = {data_2.x * scale * pass_weight0.x,
                          data_2.y * scale * pass_weight0.y};
  const float2 pass1_f = {data_3.x * scale * pass_weight1.x,
                          data_3.y * scale * pass_weight1.y};
  const __mt_bfloat162 x_rot = __float22bfloat162_rn(x_rot_f);
  const __mt_bfloat162 y_rot = __float22bfloat162_rn(y_rot_f);
  const __mt_bfloat162 pass0 = __float22bfloat162_rn(pass0_f);
  const __mt_bfloat162 pass1 = __float22bfloat162_rn(pass1_f);

  if (is_q) {
    const int64_t q_out_base =
        (int64_t)token * q_out_batch_stride + (int64_t)head * q_out_head_stride;
    *reinterpret_cast<__mt_bfloat162 *>(q_out + q_out_base + pair_col) = x_rot;
    *reinterpret_cast<__mt_bfloat162 *>(q_out + q_out_base + embed_dim + pair_col) =
        y_rot;
    *reinterpret_cast<__mt_bfloat162 *>(q_out + q_out_base + 64 + pair_col) =
        pass0;
    *reinterpret_cast<__mt_bfloat162 *>(q_out + q_out_base + 96 + pair_col) =
        pass1;
  } else {
    if (write_cache) {
      const int64_t k_cache_base =
          cache_idx * k_cache_row_stride + (int64_t)head * hidden;
      *reinterpret_cast<__mt_bfloat162 *>(k_cache + k_cache_base + pair_col) =
          x_rot;
      *reinterpret_cast<__mt_bfloat162 *>(k_cache + k_cache_base + embed_dim +
                                          pair_col) = y_rot;
      *reinterpret_cast<__mt_bfloat162 *>(k_cache + k_cache_base + 64 + pair_col) =
          pass0;
      *reinterpret_cast<__mt_bfloat162 *>(k_cache + k_cache_base + 96 + pair_col) =
          pass1;
    }
    if constexpr (STORE_K_OUT) {
      const int64_t k_out_base = (int64_t)token * k_out_batch_stride +
                                 (int64_t)head * k_out_head_stride;
      *reinterpret_cast<__mt_bfloat162 *>(k_out + k_out_base + pair_col) =
          x_rot;
      *reinterpret_cast<__mt_bfloat162 *>(k_out + k_out_base + embed_dim +
                                          pair_col) = y_rot;
      *reinterpret_cast<__mt_bfloat162 *>(k_out + k_out_base + 64 + pair_col) =
          pass0;
      *reinterpret_cast<__mt_bfloat162 *>(k_out + k_out_base + 96 + pair_col) =
          pass1;
    }
  }
}

template <typename index_t, int Q_HEADS, int K_HEADS,
          int HEADS_PER_BLOCK = 8, bool STORE_K_OUT = true>
__global__ void fused_qk_rmsnorm_mrope_cache_h256r64_bf16_kernel(
    const __mt_bfloat16 *__restrict__ q, const __mt_bfloat16 *__restrict__ k,
    const __mt_bfloat16 *__restrict__ v,
    const __mt_bfloat16 *__restrict__ q_weight,
    const __mt_bfloat16 *__restrict__ k_weight,
    const int64_t *__restrict__ positions,
    const __mt_bfloat16 *__restrict__ cos_sin_cache,
    __mt_bfloat16 *__restrict__ q_out, __mt_bfloat16 *__restrict__ k_out,
    __mt_bfloat16 *__restrict__ k_cache, __mt_bfloat16 *__restrict__ v_cache,
    const index_t *__restrict__ indices, int batch, int64_t position_stride,
    int64_t q_batch_stride, int64_t q_head_stride, int64_t k_batch_stride,
    int64_t k_head_stride, int64_t v_batch_stride, int64_t v_head_stride,
    int64_t q_out_batch_stride, int64_t q_out_head_stride,
    int64_t k_out_batch_stride, int64_t k_out_head_stride,
    int cache_block_size, int64_t k_cache_block_stride,
    int64_t k_cache_token_stride, int64_t v_cache_block_stride,
    int64_t v_cache_token_stride,
    int64_t indices_stride, float eps) {
  constexpr int hidden = 256;
  constexpr int embed_dim = 32;
  constexpr int rot_dim = 64;
  constexpr int heads_per_block = HEADS_PER_BLOCK;
  constexpr int groups_per_token =
      (Q_HEADS + K_HEADS + heads_per_block - 1) / heads_per_block;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int token = (int)blockIdx.x / groups_per_token;
  const int group = (int)blockIdx.x - token * groups_per_token;
  if (token >= batch) {
    return;
  }

  const int64_t cache_idx =
      static_cast<int64_t>(indices[(int64_t)token * indices_stride]);
  const bool write_cache = cache_idx >= 0;
  const int64_t cache_block =
      cache_block_size > 0 ? cache_idx / cache_block_size : 0;
  const int64_t cache_token =
      cache_block_size > 0 ? cache_idx % cache_block_size : cache_idx;
  const int64_t k_cache_base =
      cache_block * k_cache_block_stride + cache_token * k_cache_token_stride;
  const int64_t v_cache_base =
      cache_block * v_cache_block_stride + cache_token * v_cache_token_stride;

  if (group == 0 && write_cache) {
    constexpr int kVec = 8;
    constexpr int row_dim = K_HEADS * hidden;
    constexpr int vec_count = row_dim / kVec;
    const int64_t v_token_base = (int64_t)token * v_batch_stride;
    const int64_t v_out_base = v_cache_base;
    for (int vec_idx = tid; vec_idx < vec_count; vec_idx += (int)blockDim.x) {
      const int head = vec_idx >> 5;
      const int col = (vec_idx & 31) * kVec;
      Vec8<__mt_bfloat16> v_vec = Vec8<__mt_bfloat16>::load(
          v + v_token_base + (int64_t)head * v_head_stride, col);
      *(Vec8<__mt_bfloat16> *)(v_cache + v_out_base + (int64_t)head * hidden +
                               col) = v_vec;
    }
  }

  const int global_head = group * heads_per_block + warp;
  if (global_head >= Q_HEADS + K_HEADS) {
    return;
  }

  const bool is_q = global_head < Q_HEADS;
  const int head = is_q ? global_head : global_head - Q_HEADS;
  const __mt_bfloat16 *__restrict__ data = is_q ? q : k;
  const __mt_bfloat16 *__restrict__ weight = is_q ? q_weight : k_weight;
  const int64_t base =
      is_q ? ((int64_t)token * q_batch_stride + (int64_t)head * q_head_stride)
           : ((int64_t)token * k_batch_stride + (int64_t)head * k_head_stride);

  const float data_0 = __bfloat162float(data[base + lane]);
  const float data_1 = __bfloat162float(data[base + lane + 32]);
  const float data_2 = __bfloat162float(data[base + lane + 64]);
  const float data_3 = __bfloat162float(data[base + lane + 96]);
  const float data_4 = __bfloat162float(data[base + lane + 128]);
  const float data_5 = __bfloat162float(data[base + lane + 160]);
  const float data_6 = __bfloat162float(data[base + lane + 192]);
  const float data_7 = __bfloat162float(data[base + lane + 224]);
  float sum = data_0 * data_0 + data_1 * data_1 + data_2 * data_2 +
              data_3 * data_3 + data_4 * data_4 + data_5 * data_5 +
              data_6 * data_6 + data_7 * data_7;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, mask);
  }
  const float scale = fast_rsqrt(sum * (1.0f / 256.0f) + eps);

  const int rot_offset = lane;
  const int axis = mrope_11_11_10_interleaved_axis(rot_offset);
  const int64_t pos = positions[(int64_t)axis * position_stride + token];
  const __mt_bfloat16 *__restrict__ cache_ptr = cos_sin_cache + pos * rot_dim;
  const float cos_v = __bfloat162float(cache_ptr[rot_offset]);
  const float sin_v = __bfloat162float(cache_ptr[embed_dim + rot_offset]);
  const float x_weight = __bfloat162float(weight[rot_offset]) + 1.0f;
  const float y_weight = __bfloat162float(weight[embed_dim + rot_offset]) + 1.0f;
  const float x = data_0 * scale * x_weight;
  const float y = data_1 * scale * y_weight;
  const __mt_bfloat16 x_rot = __float2bfloat16_rn(x * cos_v - y * sin_v);
  const __mt_bfloat16 y_rot = __float2bfloat16_rn(y * cos_v + x * sin_v);

  const __mt_bfloat16 out_2 = __float2bfloat16_rn(
      data_2 * scale * (__bfloat162float(weight[lane + 64]) + 1.0f));
  const __mt_bfloat16 out_3 = __float2bfloat16_rn(
      data_3 * scale * (__bfloat162float(weight[lane + 96]) + 1.0f));
  const __mt_bfloat16 out_4 = __float2bfloat16_rn(
      data_4 * scale * (__bfloat162float(weight[lane + 128]) + 1.0f));
  const __mt_bfloat16 out_5 = __float2bfloat16_rn(
      data_5 * scale * (__bfloat162float(weight[lane + 160]) + 1.0f));
  const __mt_bfloat16 out_6 = __float2bfloat16_rn(
      data_6 * scale * (__bfloat162float(weight[lane + 192]) + 1.0f));
  const __mt_bfloat16 out_7 = __float2bfloat16_rn(
      data_7 * scale * (__bfloat162float(weight[lane + 224]) + 1.0f));

  if (is_q) {
    const int64_t q_out_base =
        (int64_t)token * q_out_batch_stride + (int64_t)head * q_out_head_stride;
    q_out[q_out_base + lane] = x_rot;
    q_out[q_out_base + lane + 32] = y_rot;
    q_out[q_out_base + lane + 64] = out_2;
    q_out[q_out_base + lane + 96] = out_3;
    q_out[q_out_base + lane + 128] = out_4;
    q_out[q_out_base + lane + 160] = out_5;
    q_out[q_out_base + lane + 192] = out_6;
    q_out[q_out_base + lane + 224] = out_7;
  } else {
    if (write_cache) {
      const int64_t head_cache_base = k_cache_base + (int64_t)head * hidden;
      k_cache[head_cache_base + lane] = x_rot;
      k_cache[head_cache_base + lane + 32] = y_rot;
      k_cache[head_cache_base + lane + 64] = out_2;
      k_cache[head_cache_base + lane + 96] = out_3;
      k_cache[head_cache_base + lane + 128] = out_4;
      k_cache[head_cache_base + lane + 160] = out_5;
      k_cache[head_cache_base + lane + 192] = out_6;
      k_cache[head_cache_base + lane + 224] = out_7;
    }
    if constexpr (STORE_K_OUT) {
      const int64_t k_out_base = (int64_t)token * k_out_batch_stride +
                                 (int64_t)head * k_out_head_stride;
      k_out[k_out_base + lane] = x_rot;
      k_out[k_out_base + lane + 32] = y_rot;
      k_out[k_out_base + lane + 64] = out_2;
      k_out[k_out_base + lane + 96] = out_3;
      k_out[k_out_base + lane + 128] = out_4;
      k_out[k_out_base + lane + 160] = out_5;
      k_out[k_out_base + lane + 192] = out_6;
      k_out[k_out_base + lane + 224] = out_7;
    }
  }
}

template <int Q_HEADS, int K_HEADS, int HEADS_PER_BLOCK = 8>
__global__ void fused_qk_rmsnorm_mrope_h256r64_bf16_kernel(
    const __mt_bfloat16 *__restrict__ q, const __mt_bfloat16 *__restrict__ k,
    const __mt_bfloat16 *__restrict__ q_weight,
    const __mt_bfloat16 *__restrict__ k_weight,
    const int64_t *__restrict__ positions,
    const __mt_bfloat16 *__restrict__ cos_sin_cache,
    __mt_bfloat16 *__restrict__ q_out, __mt_bfloat16 *__restrict__ k_out,
    int batch, int64_t position_stride, int64_t q_batch_stride,
    int64_t q_head_stride, int64_t k_batch_stride, int64_t k_head_stride,
    int64_t q_out_batch_stride, int64_t q_out_head_stride,
    int64_t k_out_batch_stride, int64_t k_out_head_stride, float eps) {
  constexpr int hidden = 256;
  constexpr int embed_dim = 32;
  constexpr int rot_dim = 64;
  constexpr int heads_per_block = HEADS_PER_BLOCK;
  constexpr int groups_per_token =
      (Q_HEADS + K_HEADS + heads_per_block - 1) / heads_per_block;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int token = (int)blockIdx.x / groups_per_token;
  const int group = (int)blockIdx.x - token * groups_per_token;
  if (token >= batch) {
    return;
  }

  const int global_head = group * heads_per_block + warp;
  if (global_head >= Q_HEADS + K_HEADS) {
    return;
  }

  const bool is_q = global_head < Q_HEADS;
  const int head = is_q ? global_head : global_head - Q_HEADS;
  const __mt_bfloat16 *__restrict__ data = is_q ? q : k;
  const __mt_bfloat16 *__restrict__ weight = is_q ? q_weight : k_weight;
  __mt_bfloat16 *__restrict__ out = is_q ? q_out : k_out;
  const int64_t base =
      is_q ? ((int64_t)token * q_batch_stride + (int64_t)head * q_head_stride)
           : ((int64_t)token * k_batch_stride + (int64_t)head * k_head_stride);
  const int64_t out_base =
      is_q ? ((int64_t)token * q_out_batch_stride +
              (int64_t)head * q_out_head_stride)
           : ((int64_t)token * k_out_batch_stride +
              (int64_t)head * k_out_head_stride);

  const float data_0 = __bfloat162float(data[base + lane]);
  const float data_1 = __bfloat162float(data[base + lane + 32]);
  const float data_2 = __bfloat162float(data[base + lane + 64]);
  const float data_3 = __bfloat162float(data[base + lane + 96]);
  const float data_4 = __bfloat162float(data[base + lane + 128]);
  const float data_5 = __bfloat162float(data[base + lane + 160]);
  const float data_6 = __bfloat162float(data[base + lane + 192]);
  const float data_7 = __bfloat162float(data[base + lane + 224]);
  float sum = data_0 * data_0 + data_1 * data_1 + data_2 * data_2 +
              data_3 * data_3 + data_4 * data_4 + data_5 * data_5 +
              data_6 * data_6 + data_7 * data_7;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, mask);
  }
  const float scale = fast_rsqrt(sum * (1.0f / 256.0f) + eps);

  const int rot_offset = lane;
  const int axis = mrope_11_11_10_interleaved_axis(rot_offset);
  const int64_t pos = positions[(int64_t)axis * position_stride + token];
  const __mt_bfloat16 *__restrict__ cache_ptr = cos_sin_cache + pos * rot_dim;
  const float cos_v = __bfloat162float(cache_ptr[rot_offset]);
  const float sin_v = __bfloat162float(cache_ptr[embed_dim + rot_offset]);
  const float x_weight = __bfloat162float(weight[rot_offset]) + 1.0f;
  const float y_weight = __bfloat162float(weight[embed_dim + rot_offset]) + 1.0f;
  const float x = data_0 * scale * x_weight;
  const float y = data_1 * scale * y_weight;
  out[out_base + lane] = __float2bfloat16_rn(x * cos_v - y * sin_v);
  out[out_base + lane + 32] = __float2bfloat16_rn(y * cos_v + x * sin_v);
  out[out_base + lane + 64] = __float2bfloat16_rn(
      data_2 * scale * (__bfloat162float(weight[lane + 64]) + 1.0f));
  out[out_base + lane + 96] = __float2bfloat16_rn(
      data_3 * scale * (__bfloat162float(weight[lane + 96]) + 1.0f));
  out[out_base + lane + 128] = __float2bfloat16_rn(
      data_4 * scale * (__bfloat162float(weight[lane + 128]) + 1.0f));
  out[out_base + lane + 160] = __float2bfloat16_rn(
      data_5 * scale * (__bfloat162float(weight[lane + 160]) + 1.0f));
  out[out_base + lane + 192] = __float2bfloat16_rn(
      data_6 * scale * (__bfloat162float(weight[lane + 192]) + 1.0f));
  out[out_base + lane + 224] = __float2bfloat16_rn(
      data_7 * scale * (__bfloat162float(weight[lane + 224]) + 1.0f));
}

template <int Q_HEADS, int K_HEADS, int HEADS_PER_BLOCK = 8>
__global__ void fused_qk_rmsnorm_mrope_h128_full_bf16_kernel(
    const __mt_bfloat16 *__restrict__ q, const __mt_bfloat16 *__restrict__ k,
    const __mt_bfloat16 *__restrict__ q_weight,
    const __mt_bfloat16 *__restrict__ k_weight,
    const int64_t *__restrict__ positions,
    const __mt_bfloat16 *__restrict__ cos_sin_cache,
    __mt_bfloat16 *__restrict__ q_out, __mt_bfloat16 *__restrict__ k_out,
    int batch, int64_t position_stride, int64_t q_batch_stride,
    int64_t q_head_stride, int64_t k_batch_stride, int64_t k_head_stride,
    int64_t q_out_batch_stride, int64_t q_out_head_stride,
    int64_t k_out_batch_stride, int64_t k_out_head_stride, float eps) {
  constexpr int hidden = 128;
  constexpr int embed_dim = 64;
  constexpr int rot_dim = 128;
  constexpr int heads_per_block = HEADS_PER_BLOCK;
  constexpr int groups_per_token =
      (Q_HEADS + K_HEADS + heads_per_block - 1) / heads_per_block;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int token = (int)blockIdx.x / groups_per_token;
  const int group = (int)blockIdx.x - token * groups_per_token;
  if (token >= batch) {
    return;
  }

  const int global_head = group * heads_per_block + warp;
  if (global_head >= Q_HEADS + K_HEADS) {
    return;
  }

  const bool is_q = global_head < Q_HEADS;
  const int head = is_q ? global_head : global_head - Q_HEADS;
  const __mt_bfloat16 *__restrict__ data = is_q ? q : k;
  const __mt_bfloat16 *__restrict__ weight = is_q ? q_weight : k_weight;
  __mt_bfloat16 *__restrict__ out = is_q ? q_out : k_out;
  const int64_t base =
      is_q ? ((int64_t)token * q_batch_stride + (int64_t)head * q_head_stride)
           : ((int64_t)token * k_batch_stride + (int64_t)head * k_head_stride);
  const int64_t out_base =
      is_q ? ((int64_t)token * q_out_batch_stride +
              (int64_t)head * q_out_head_stride)
           : ((int64_t)token * k_out_batch_stride +
              (int64_t)head * k_out_head_stride);

  const float data_0 = __bfloat162float(data[base + lane]);
  const float data_1 = __bfloat162float(data[base + lane + 32]);
  const float data_2 = __bfloat162float(data[base + lane + 64]);
  const float data_3 = __bfloat162float(data[base + lane + 96]);
  float sum =
      data_0 * data_0 + data_1 * data_1 + data_2 * data_2 + data_3 * data_3;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, mask);
  }
  const float scale = fast_rsqrt(sum * (1.0f / 128.0f) + eps);

  const int rot_offset0 = lane;
  const int axis0 = mrope_24_20_20_interleaved_axis(rot_offset0);
  const int64_t pos0 = positions[(int64_t)axis0 * position_stride + token];
  const __mt_bfloat16 *__restrict__ cache_ptr0 = cos_sin_cache + pos0 * rot_dim;
  const float cos_v0 = __bfloat162float(cache_ptr0[rot_offset0]);
  const float sin_v0 = __bfloat162float(cache_ptr0[embed_dim + rot_offset0]);
  const float x_weight0 = __bfloat162float(weight[rot_offset0]) + 1.0f;
  const float y_weight0 =
      __bfloat162float(weight[embed_dim + rot_offset0]) + 1.0f;
  const float x0 = data_0 * scale * x_weight0;
  const float y0 = data_2 * scale * y_weight0;
  out[out_base + rot_offset0] =
      __float2bfloat16_rn(x0 * cos_v0 - y0 * sin_v0);
  out[out_base + embed_dim + rot_offset0] =
      __float2bfloat16_rn(y0 * cos_v0 + x0 * sin_v0);

  const int rot_offset1 = lane + 32;
  const int axis1 = mrope_24_20_20_interleaved_axis(rot_offset1);
  const int64_t pos1 = positions[(int64_t)axis1 * position_stride + token];
  const __mt_bfloat16 *__restrict__ cache_ptr1 = cos_sin_cache + pos1 * rot_dim;
  const float cos_v1 = __bfloat162float(cache_ptr1[rot_offset1]);
  const float sin_v1 = __bfloat162float(cache_ptr1[embed_dim + rot_offset1]);
  const float x_weight1 = __bfloat162float(weight[rot_offset1]) + 1.0f;
  const float y_weight1 =
      __bfloat162float(weight[embed_dim + rot_offset1]) + 1.0f;
  const float x1 = data_1 * scale * x_weight1;
  const float y1 = data_3 * scale * y_weight1;
  out[out_base + rot_offset1] =
      __float2bfloat16_rn(x1 * cos_v1 - y1 * sin_v1);
  out[out_base + embed_dim + rot_offset1] =
      __float2bfloat16_rn(y1 * cos_v1 + x1 * sin_v1);
}

template <int Q_HEADS, int K_HEADS>
__global__ void fused_qk_rmsnorm_mrope_h128_full_halfwarp_bf16_kernel(
    const __mt_bfloat16 *__restrict__ q, const __mt_bfloat16 *__restrict__ k,
    const __mt_bfloat16 *__restrict__ q_weight,
    const __mt_bfloat16 *__restrict__ k_weight,
    const int64_t *__restrict__ positions,
    const __mt_bfloat16 *__restrict__ cos_sin_cache,
    __mt_bfloat16 *__restrict__ q_out, __mt_bfloat16 *__restrict__ k_out,
    int batch, int64_t position_stride, int64_t q_batch_stride,
    int64_t q_head_stride, int64_t k_batch_stride, int64_t k_head_stride,
    int64_t q_out_batch_stride, int64_t q_out_head_stride,
    int64_t k_out_batch_stride, int64_t k_out_head_stride, float eps) {
  constexpr int hidden = 128;
  constexpr int embed_dim = 64;
  constexpr int rot_dim = 128;
  constexpr int heads_per_block = 16;
  constexpr int groups_per_token =
      (Q_HEADS + K_HEADS + heads_per_block - 1) / heads_per_block;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int half = lane >> 4;
  const int sublane = lane & 15;
  const int token = (int)blockIdx.x / groups_per_token;
  const int group = (int)blockIdx.x - token * groups_per_token;
  if (token >= batch) {
    return;
  }

  const int global_head = group * heads_per_block + warp * 2 + half;
  if (global_head >= Q_HEADS + K_HEADS) {
    return;
  }

  const bool is_q = global_head < Q_HEADS;
  const int head = is_q ? global_head : global_head - Q_HEADS;
  const __mt_bfloat16 *__restrict__ data = is_q ? q : k;
  const __mt_bfloat16 *__restrict__ weight = is_q ? q_weight : k_weight;
  __mt_bfloat16 *__restrict__ out = is_q ? q_out : k_out;
  const int64_t base =
      is_q ? ((int64_t)token * q_batch_stride + (int64_t)head * q_head_stride)
           : ((int64_t)token * k_batch_stride + (int64_t)head * k_head_stride);
  const int64_t out_base =
      is_q ? ((int64_t)token * q_out_batch_stride +
              (int64_t)head * q_out_head_stride)
           : ((int64_t)token * k_out_batch_stride +
              (int64_t)head * k_out_head_stride);

  const int o0 = sublane;
  const int o1 = sublane + 16;
  const int o2 = sublane + 32;
  const int o3 = sublane + 48;
  const float data_0 = __bfloat162float(data[base + o0]);
  const float data_1 = __bfloat162float(data[base + o1]);
  const float data_2 = __bfloat162float(data[base + o2]);
  const float data_3 = __bfloat162float(data[base + o3]);
  const float data_4 = __bfloat162float(data[base + o0 + embed_dim]);
  const float data_5 = __bfloat162float(data[base + o1 + embed_dim]);
  const float data_6 = __bfloat162float(data[base + o2 + embed_dim]);
  const float data_7 = __bfloat162float(data[base + o3 + embed_dim]);
  float sum = data_0 * data_0 + data_1 * data_1 + data_2 * data_2 +
              data_3 * data_3 + data_4 * data_4 + data_5 * data_5 +
              data_6 * data_6 + data_7 * data_7;
#pragma unroll
  for (int mask = 8; mask > 0; mask >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, mask);
  }
  const float scale = fast_rsqrt(sum * (1.0f / 128.0f) + eps);

#define SGLANG_H128_FULL_HALF_ROTATE(ROT_OFFSET, DATA_X, DATA_Y)              \
  do {                                                                        \
    const int axis = mrope_24_20_20_interleaved_axis(ROT_OFFSET);             \
    const int64_t pos = positions[(int64_t)axis * position_stride + token];   \
    const __mt_bfloat16 *__restrict__ cache_ptr = cos_sin_cache + pos * rot_dim; \
    const float cos_v = __bfloat162float(cache_ptr[ROT_OFFSET]);              \
    const float sin_v = __bfloat162float(cache_ptr[embed_dim + ROT_OFFSET]);  \
    const float x_weight = __bfloat162float(weight[ROT_OFFSET]) + 1.0f;       \
    const float y_weight =                                                       \
        __bfloat162float(weight[embed_dim + ROT_OFFSET]) + 1.0f;              \
    const float x = (DATA_X)*scale * x_weight;                                \
    const float y = (DATA_Y)*scale * y_weight;                                \
    out[out_base + ROT_OFFSET] = __float2bfloat16_rn(x * cos_v - y * sin_v);  \
    out[out_base + embed_dim + ROT_OFFSET] =                                  \
        __float2bfloat16_rn(y * cos_v + x * sin_v);                           \
  } while (0)

  SGLANG_H128_FULL_HALF_ROTATE(o0, data_0, data_4);
  SGLANG_H128_FULL_HALF_ROTATE(o1, data_1, data_5);
  SGLANG_H128_FULL_HALF_ROTATE(o2, data_2, data_6);
  SGLANG_H128_FULL_HALF_ROTATE(o3, data_3, data_7);
#undef SGLANG_H128_FULL_HALF_ROTATE
}

template <typename index_t, bool WITH_CACHE, bool GEMMA>
__global__ void fused_qk_rmsnorm_mrope_generic_bf16_kernel(
    const __mt_bfloat16 *__restrict__ q, const __mt_bfloat16 *__restrict__ k,
    const __mt_bfloat16 *__restrict__ v,
    const __mt_bfloat16 *__restrict__ q_weight,
    const __mt_bfloat16 *__restrict__ k_weight,
    const int64_t *__restrict__ positions,
    const __mt_bfloat16 *__restrict__ cos_sin_cache,
    __mt_bfloat16 *__restrict__ q_out, __mt_bfloat16 *__restrict__ k_out,
    __mt_bfloat16 *__restrict__ k_cache, __mt_bfloat16 *__restrict__ v_cache,
    const index_t *__restrict__ indices, int batch, int q_heads, int k_heads,
    int hidden, int rot_dim, int64_t position_stride, int64_t q_batch_stride,
    int64_t q_head_stride, int64_t k_batch_stride, int64_t k_head_stride,
    int64_t v_batch_stride, int64_t v_head_stride, int64_t q_out_batch_stride,
    int64_t q_out_head_stride, int64_t k_out_batch_stride,
    int64_t k_out_head_stride, int64_t k_cache_row_stride,
    int64_t v_cache_row_stride, int64_t indices_stride, int mrope_section_t,
    int mrope_section_h, int mrope_section_w, bool is_interleaved, float eps) {
  constexpr int heads_per_block = 8;
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int total_heads = q_heads + k_heads;
  const int groups_per_token =
      (total_heads + heads_per_block - 1) / heads_per_block;
  const int token = (int)blockIdx.x / groups_per_token;
  const int group = (int)blockIdx.x - token * groups_per_token;
  if (token >= batch) {
    return;
  }

  int64_t cache_idx = 0;
  bool write_cache = false;
  if constexpr (WITH_CACHE) {
    cache_idx = static_cast<int64_t>(indices[(int64_t)token * indices_stride]);
    write_cache = cache_idx >= 0;
    if (group == 0 && write_cache) {
      const int row_dim = k_heads * hidden;
      const int64_t v_base = (int64_t)token * v_batch_stride;
      const int64_t v_dst = cache_idx * v_cache_row_stride;
      for (int col = tid; col < row_dim; col += (int)blockDim.x) {
        const int head = col / hidden;
        const int head_col = col - head * hidden;
        v_cache[v_dst + col] =
            v[v_base + (int64_t)head * v_head_stride + head_col];
      }
    }
  }

  const int global_head = group * heads_per_block + warp;
  if (global_head >= total_heads) {
    return;
  }
  const bool is_q = global_head < q_heads;
  const int head = is_q ? global_head : global_head - q_heads;
  const __mt_bfloat16 *__restrict__ data = is_q ? q : k;
  const __mt_bfloat16 *__restrict__ weight = is_q ? q_weight : k_weight;
  const int64_t base =
      is_q ? ((int64_t)token * q_batch_stride + (int64_t)head * q_head_stride)
           : ((int64_t)token * k_batch_stride + (int64_t)head * k_head_stride);
  const int embed_dim = rot_dim >> 1;

  float sum = 0.0f;
  for (int col = lane; col < hidden; col += 32) {
    const float value = __bfloat162float(data[base + col]);
    sum += value * value;
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, mask);
  }
  const float scale = fast_rsqrt(sum / static_cast<float>(hidden) + eps);

  for (int rot_offset = lane; rot_offset < embed_dim; rot_offset += 32) {
    int axis;
    if (is_interleaved) {
      const bool use_h =
          (rot_offset % 3 == 1) && (rot_offset <= 3 * mrope_section_h);
      const bool use_w =
          (rot_offset % 3 == 2) && (rot_offset <= 3 * mrope_section_w);
      axis = use_h ? 1 : (use_w ? 2 : 0);
    } else {
      axis = rot_offset < mrope_section_t
                 ? 0
                 : (rot_offset < mrope_section_t + mrope_section_h ? 1 : 2);
    }
    const int64_t pos = positions[(int64_t)axis * position_stride + token];
    const __mt_bfloat16 *__restrict__ cache_ptr = cos_sin_cache + pos * rot_dim;
    const int x_index = rot_offset;
    const int y_index = embed_dim + rot_offset;
    const float cos_v = __bfloat162float(cache_ptr[x_index]);
    const float sin_v = __bfloat162float(cache_ptr[y_index]);
    const float x_weight = __bfloat162float(weight[x_index]) +
                           (GEMMA ? 1.0f : 0.0f);
    const float y_weight = __bfloat162float(weight[y_index]) +
                           (GEMMA ? 1.0f : 0.0f);
    const float x = __bfloat162float(data[base + x_index]) * scale * x_weight;
    const float y = __bfloat162float(data[base + y_index]) * scale * y_weight;
    const __mt_bfloat16 x_rot = __float2bfloat16_rn(x * cos_v - y * sin_v);
    const __mt_bfloat16 y_rot = __float2bfloat16_rn(y * cos_v + x * sin_v);
    if (is_q) {
      const int64_t out_base =
          (int64_t)token * q_out_batch_stride +
          (int64_t)head * q_out_head_stride;
      q_out[out_base + x_index] = x_rot;
      q_out[out_base + y_index] = y_rot;
    } else if constexpr (WITH_CACHE) {
      if (write_cache) {
        const int64_t cache_base =
            cache_idx * k_cache_row_stride + (int64_t)head * hidden;
        k_cache[cache_base + x_index] = x_rot;
        k_cache[cache_base + y_index] = y_rot;
      }
      if (k_out != nullptr) {
        const int64_t out_base =
            (int64_t)token * k_out_batch_stride +
            (int64_t)head * k_out_head_stride;
        k_out[out_base + x_index] = x_rot;
        k_out[out_base + y_index] = y_rot;
      }
    } else {
      const int64_t out_base =
          (int64_t)token * k_out_batch_stride +
          (int64_t)head * k_out_head_stride;
      k_out[out_base + x_index] = x_rot;
      k_out[out_base + y_index] = y_rot;
    }
  }

  for (int col = rot_dim + lane; col < hidden; col += 32) {
    const float weight_value =
        __bfloat162float(weight[col]) + (GEMMA ? 1.0f : 0.0f);
    const __mt_bfloat16 out_value =
        __float2bfloat16_rn(__bfloat162float(data[base + col]) * scale * weight_value);
    if (is_q) {
      const int64_t out_base =
          (int64_t)token * q_out_batch_stride +
          (int64_t)head * q_out_head_stride;
      q_out[out_base + col] = out_value;
    } else if constexpr (WITH_CACHE) {
      if (write_cache) {
        const int64_t cache_base =
            cache_idx * k_cache_row_stride + (int64_t)head * hidden;
        k_cache[cache_base + col] = out_value;
      }
      if (k_out != nullptr) {
        const int64_t out_base =
            (int64_t)token * k_out_batch_stride +
            (int64_t)head * k_out_head_stride;
        k_out[out_base + col] = out_value;
      }
    } else {
      const int64_t out_base =
          (int64_t)token * k_out_batch_stride +
          (int64_t)head * k_out_head_stride;
      k_out[out_base + col] = out_value;
    }
  }
}

void launch_qk_mrope_kernel(ffi::TensorView q, ffi::TensorView k,
                            ffi::TensorView q_weight,
                            ffi::TensorView k_weight,
                            ffi::TensorView positions,
                            ffi::TensorView cos_sin_cache,
                            ffi::TensorView q_out, ffi::TensorView k_out,
                            int batch, bool gemma, bool is_interleaved, float eps,
                            musaStream_t stream) {
  constexpr int threads = 256;
  constexpr int groups_per_token = 5;
  if (gemma && is_interleaved) {
    fused_qk_rmsnorm_mrope_32q4kv_h128_bf16_kernel<true, true>
        <<<batch * groups_per_token, threads, 0, stream>>>(
            static_cast<const __mt_bfloat16 *>(q.data_ptr()),
            static_cast<const __mt_bfloat16 *>(k.data_ptr()),
            static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
            static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
            static_cast<const int64_t *>(positions.data_ptr()),
            static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
            static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
            static_cast<__mt_bfloat16 *>(k_out.data_ptr()), batch,
            static_cast<int64_t>(positions.stride(0)),
            static_cast<int64_t>(q.stride(0)), static_cast<int64_t>(q.stride(1)),
            static_cast<int64_t>(k.stride(0)), static_cast<int64_t>(k.stride(1)),
            static_cast<int64_t>(q_out.stride(0)),
            static_cast<int64_t>(q_out.stride(1)),
            static_cast<int64_t>(k_out.stride(0)),
            static_cast<int64_t>(k_out.stride(1)), eps);
    return;
  }
  if (gemma) {
    fused_qk_rmsnorm_mrope_32q4kv_h128_bf16_kernel<true, false>
        <<<batch * groups_per_token, threads, 0, stream>>>(
            static_cast<const __mt_bfloat16 *>(q.data_ptr()),
            static_cast<const __mt_bfloat16 *>(k.data_ptr()),
            static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
            static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
            static_cast<const int64_t *>(positions.data_ptr()),
            static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
            static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
            static_cast<__mt_bfloat16 *>(k_out.data_ptr()), batch,
            static_cast<int64_t>(positions.stride(0)),
            static_cast<int64_t>(q.stride(0)), static_cast<int64_t>(q.stride(1)),
            static_cast<int64_t>(k.stride(0)), static_cast<int64_t>(k.stride(1)),
            static_cast<int64_t>(q_out.stride(0)),
            static_cast<int64_t>(q_out.stride(1)),
            static_cast<int64_t>(k_out.stride(0)),
            static_cast<int64_t>(k_out.stride(1)), eps);
    return;
  }
  if (is_interleaved) {
    fused_qk_rmsnorm_mrope_32q4kv_h128_bf16_kernel<false, true>
        <<<batch * groups_per_token, threads, 0, stream>>>(
            static_cast<const __mt_bfloat16 *>(q.data_ptr()),
            static_cast<const __mt_bfloat16 *>(k.data_ptr()),
            static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
            static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
            static_cast<const int64_t *>(positions.data_ptr()),
            static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
            static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
            static_cast<__mt_bfloat16 *>(k_out.data_ptr()), batch,
            static_cast<int64_t>(positions.stride(0)),
            static_cast<int64_t>(q.stride(0)), static_cast<int64_t>(q.stride(1)),
            static_cast<int64_t>(k.stride(0)), static_cast<int64_t>(k.stride(1)),
            static_cast<int64_t>(q_out.stride(0)),
            static_cast<int64_t>(q_out.stride(1)),
            static_cast<int64_t>(k_out.stride(0)),
            static_cast<int64_t>(k_out.stride(1)), eps);
    return;
  }
  fused_qk_rmsnorm_mrope_32q4kv_h128_bf16_kernel<false, false>
      <<<batch * groups_per_token, threads, 0, stream>>>(
          static_cast<const __mt_bfloat16 *>(q.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k.data_ptr()),
          static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
          static_cast<const int64_t *>(positions.data_ptr()),
          static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
          static_cast<__mt_bfloat16 *>(k_out.data_ptr()), batch,
          static_cast<int64_t>(positions.stride(0)),
          static_cast<int64_t>(q.stride(0)), static_cast<int64_t>(q.stride(1)),
          static_cast<int64_t>(k.stride(0)), static_cast<int64_t>(k.stride(1)),
          static_cast<int64_t>(q_out.stride(0)),
          static_cast<int64_t>(q_out.stride(1)),
          static_cast<int64_t>(k_out.stride(0)),
          static_cast<int64_t>(k_out.stride(1)), eps);
}

template <int Q_HEADS, int K_HEADS, int HEADS_PER_BLOCK = 8>
void launch_h256r64_mrope_kernel(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView q_weight,
    ffi::TensorView k_weight, ffi::TensorView positions,
    ffi::TensorView cos_sin_cache, ffi::TensorView q_out,
    ffi::TensorView k_out, int batch, float eps, musaStream_t stream) {
  constexpr int heads_per_block = HEADS_PER_BLOCK;
  constexpr int threads = 32 * heads_per_block;
  constexpr int groups_per_token =
      (Q_HEADS + K_HEADS + heads_per_block - 1) / heads_per_block;
  fused_qk_rmsnorm_mrope_h256r64_bf16_kernel<Q_HEADS, K_HEADS,
                                                   HEADS_PER_BLOCK>
      <<<batch * groups_per_token, threads, 0, stream>>>(
          static_cast<const __mt_bfloat16 *>(q.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k.data_ptr()),
          static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
          static_cast<const int64_t *>(positions.data_ptr()),
          static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
          static_cast<__mt_bfloat16 *>(k_out.data_ptr()), batch,
          static_cast<int64_t>(positions.stride(0)),
          static_cast<int64_t>(q.stride(0)),
          static_cast<int64_t>(q.stride(1)),
          static_cast<int64_t>(k.stride(0)),
          static_cast<int64_t>(k.stride(1)),
          static_cast<int64_t>(q_out.stride(0)),
          static_cast<int64_t>(q_out.stride(1)),
          static_cast<int64_t>(k_out.stride(0)),
          static_cast<int64_t>(k_out.stride(1)), eps);
}

template <int Q_HEADS, int K_HEADS, int HEADS_PER_BLOCK = 8>
void launch_h128_full_mrope_kernel(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView q_weight,
    ffi::TensorView k_weight, ffi::TensorView positions,
    ffi::TensorView cos_sin_cache, ffi::TensorView q_out,
    ffi::TensorView k_out, int batch, float eps, musaStream_t stream) {
  constexpr int warps_per_block = HEADS_PER_BLOCK;
  constexpr int heads_per_block = warps_per_block * 2;
  constexpr int threads = 32 * warps_per_block;
  constexpr int groups_per_token =
      (Q_HEADS + K_HEADS + heads_per_block - 1) / heads_per_block;
  fused_qk_rmsnorm_mrope_h128_full_bf16_kernel<Q_HEADS, K_HEADS,
                                               HEADS_PER_BLOCK>
      <<<batch * groups_per_token, threads, 0, stream>>>(
          static_cast<const __mt_bfloat16 *>(q.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k.data_ptr()),
          static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
          static_cast<const int64_t *>(positions.data_ptr()),
          static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
          static_cast<__mt_bfloat16 *>(k_out.data_ptr()), batch,
          static_cast<int64_t>(positions.stride(0)),
          static_cast<int64_t>(q.stride(0)),
          static_cast<int64_t>(q.stride(1)),
          static_cast<int64_t>(k.stride(0)),
          static_cast<int64_t>(k.stride(1)),
          static_cast<int64_t>(q_out.stride(0)),
          static_cast<int64_t>(q_out.stride(1)),
          static_cast<int64_t>(k_out.stride(0)),
          static_cast<int64_t>(k_out.stride(1)), eps);
}

template <int Q_HEADS, int K_HEADS>
void launch_h128_full_halfwarp_mrope_kernel(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView q_weight,
    ffi::TensorView k_weight, ffi::TensorView positions,
    ffi::TensorView cos_sin_cache, ffi::TensorView q_out,
    ffi::TensorView k_out, int batch, float eps, musaStream_t stream) {
  constexpr int threads = 256;
  constexpr int heads_per_block = 16;
  constexpr int groups_per_token =
      (Q_HEADS + K_HEADS + heads_per_block - 1) / heads_per_block;
  fused_qk_rmsnorm_mrope_h128_full_halfwarp_bf16_kernel<Q_HEADS, K_HEADS>
      <<<batch * groups_per_token, threads, 0, stream>>>(
          static_cast<const __mt_bfloat16 *>(q.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k.data_ptr()),
          static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
          static_cast<const int64_t *>(positions.data_ptr()),
          static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
          static_cast<__mt_bfloat16 *>(k_out.data_ptr()), batch,
          static_cast<int64_t>(positions.stride(0)),
          static_cast<int64_t>(q.stride(0)),
          static_cast<int64_t>(q.stride(1)),
          static_cast<int64_t>(k.stride(0)),
          static_cast<int64_t>(k.stride(1)),
          static_cast<int64_t>(q_out.stride(0)),
          static_cast<int64_t>(q_out.stride(1)),
          static_cast<int64_t>(k_out.stride(0)),
          static_cast<int64_t>(k_out.stride(1)), eps);
}

void launch_fused_qk_rmsnorm_mrope_bf16(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView q_weight,
    ffi::TensorView k_weight, ffi::TensorView positions,
    ffi::TensorView cos_sin_cache, ffi::TensorView q_out, ffi::TensorView k_out,
    int mrope_section_t, int mrope_section_h, int mrope_section_w,
    bool is_interleaved, bool gemma, float eps) {
  const int batch = static_cast<int>(q.size(0));
  if (batch == 0) {
    return;
  }
  musaStream_t stream = get_stream(q.device());
  const int q_heads = static_cast<int>(q.size(1));
  const int k_heads = static_cast<int>(k.size(1));
  const int hidden = static_cast<int>(q.size(2));
  const int rot_dim = static_cast<int>(cos_sin_cache.size(1));
  const bool use_32q4kv_h128_mrope_specialization =
      q_heads == 32 && k_heads == 4 && hidden == 128 && rot_dim == 128 &&
      mrope_section_t == 24 && mrope_section_h == 20 &&
      mrope_section_w == 20;
  const bool use_h256r64_mrope_specialization =
      gemma && is_interleaved && hidden == 256 && rot_dim == 64 &&
      mrope_section_t == 11 && mrope_section_h == 11 &&
      mrope_section_w == 10;
  if (use_h256r64_mrope_specialization) {
    if (q_heads == 4 && k_heads == 1) {
      launch_h256r64_mrope_kernel<4, 1>(
          q, k, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          batch, eps, stream);
      return;
    }
    if (q_heads == 8 && k_heads == 1) {
      launch_h256r64_mrope_kernel<8, 1, 16>(
          q, k, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          batch, eps, stream);
      return;
    }
    if (q_heads == 16 && k_heads == 1) {
      launch_h256r64_mrope_kernel<16, 1>(
          q, k, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          batch, eps, stream);
      return;
    }
    if (q_heads == 16 && k_heads == 2) {
      launch_h256r64_mrope_kernel<16, 2>(
          q, k, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          batch, eps, stream);
      return;
    }
    if (q_heads == 16 && k_heads == 4) {
      launch_h256r64_mrope_kernel<16, 4>(
          q, k, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          batch, eps, stream);
      return;
    }
    if (q_heads == 24 && k_heads == 4) {
      launch_h256r64_mrope_kernel<24, 4>(
          q, k, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          batch, eps, stream);
      return;
    }
    if (q_heads == 32 && k_heads == 2) {
      launch_h256r64_mrope_kernel<32, 2>(
          q, k, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          batch, eps, stream);
      return;
    }
    if (q_heads == 32 && k_heads == 4) {
      launch_h256r64_mrope_kernel<32, 4>(
          q, k, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          batch, eps, stream);
      return;
    }
    if (q_heads == 64 && k_heads == 4) {
      launch_h256r64_mrope_kernel<64, 4>(
          q, k, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          batch, eps, stream);
      return;
    }
    if (q_heads == 64 && k_heads == 8) {
      launch_h256r64_mrope_kernel<64, 8>(
          q, k, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          batch, eps, stream);
      return;
    }
  }
  if (use_32q4kv_h128_mrope_specialization && batch < 128) {
    launch_qk_mrope_kernel(q, k, q_weight, k_weight, positions, cos_sin_cache,
                           q_out, k_out, batch, gemma, is_interleaved, eps,
                           stream);
    return;
  }
  const bool use_h128_full_mrope_specialization =
      gemma && is_interleaved && hidden == 128 && rot_dim == 128 &&
      mrope_section_t == 24 && mrope_section_h == 20 &&
      mrope_section_w == 20;
  if (use_h128_full_mrope_specialization) {
    if (q_heads == 2 && k_heads == 2) {
      launch_h128_full_mrope_kernel<2, 2>(
          q, k, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          batch, eps, stream);
      return;
    }
    if (q_heads == 4 && k_heads == 4) {
      launch_h128_full_halfwarp_mrope_kernel<4, 4>(
          q, k, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          batch, eps, stream);
      return;
    }
    if (q_heads == 8 && k_heads == 8) {
      launch_h128_full_halfwarp_mrope_kernel<8, 8>(
          q, k, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          batch, eps, stream);
      return;
    }
    if (q_heads == 32 && k_heads == 32) {
      launch_h128_full_halfwarp_mrope_kernel<32, 32>(
          q, k, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          batch, eps, stream);
      return;
    }
    if (q_heads == 16 && k_heads == 16) {
      launch_h128_full_halfwarp_mrope_kernel<16, 16>(
          q, k, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          batch, eps, stream);
      return;
    }
    if (q_heads == 32 && k_heads == 4) {
      launch_h128_full_halfwarp_mrope_kernel<32, 4>(
          q, k, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          batch, eps, stream);
      return;
    }
  }
  if (!use_32q4kv_h128_mrope_specialization) {
    constexpr int threads = 256;
    constexpr int heads_per_block = 8;
    const int groups_per_token =
        (q_heads + k_heads + heads_per_block - 1) / heads_per_block;
    if (gemma) {
      fused_qk_rmsnorm_mrope_generic_bf16_kernel<int32_t, false, true>
          <<<batch * groups_per_token, threads, 0, stream>>>(
              static_cast<const __mt_bfloat16 *>(q.data_ptr()),
              static_cast<const __mt_bfloat16 *>(k.data_ptr()),
              static_cast<const __mt_bfloat16 *>(nullptr),
              static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
              static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
              static_cast<const int64_t *>(positions.data_ptr()),
              static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
              static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
              static_cast<__mt_bfloat16 *>(k_out.data_ptr()),
              static_cast<__mt_bfloat16 *>(nullptr),
              static_cast<__mt_bfloat16 *>(nullptr),
              static_cast<const int32_t *>(nullptr), batch, q_heads, k_heads,
              hidden, rot_dim, static_cast<int64_t>(positions.stride(0)),
              static_cast<int64_t>(q.stride(0)),
              static_cast<int64_t>(q.stride(1)),
              static_cast<int64_t>(k.stride(0)),
              static_cast<int64_t>(k.stride(1)), static_cast<int64_t>(0),
              static_cast<int64_t>(0), static_cast<int64_t>(q_out.stride(0)),
              static_cast<int64_t>(q_out.stride(1)),
              static_cast<int64_t>(k_out.stride(0)),
              static_cast<int64_t>(k_out.stride(1)), static_cast<int64_t>(0),
              static_cast<int64_t>(0), static_cast<int64_t>(0),
              mrope_section_t, mrope_section_h, mrope_section_w,
              is_interleaved, eps);
      return;
    }
    fused_qk_rmsnorm_mrope_generic_bf16_kernel<int32_t, false, false>
        <<<batch * groups_per_token, threads, 0, stream>>>(
            static_cast<const __mt_bfloat16 *>(q.data_ptr()),
            static_cast<const __mt_bfloat16 *>(k.data_ptr()),
            static_cast<const __mt_bfloat16 *>(nullptr),
            static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
            static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
            static_cast<const int64_t *>(positions.data_ptr()),
            static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
            static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
            static_cast<__mt_bfloat16 *>(k_out.data_ptr()),
            static_cast<__mt_bfloat16 *>(nullptr),
            static_cast<__mt_bfloat16 *>(nullptr),
            static_cast<const int32_t *>(nullptr), batch, q_heads, k_heads,
            hidden, rot_dim, static_cast<int64_t>(positions.stride(0)),
            static_cast<int64_t>(q.stride(0)), static_cast<int64_t>(q.stride(1)),
            static_cast<int64_t>(k.stride(0)), static_cast<int64_t>(k.stride(1)),
            static_cast<int64_t>(0), static_cast<int64_t>(0),
            static_cast<int64_t>(q_out.stride(0)),
            static_cast<int64_t>(q_out.stride(1)),
            static_cast<int64_t>(k_out.stride(0)),
            static_cast<int64_t>(k_out.stride(1)), static_cast<int64_t>(0),
            static_cast<int64_t>(0), static_cast<int64_t>(0), mrope_section_t,
            mrope_section_h, mrope_section_w, is_interleaved, eps);
    return;
  }
  launch_qk_mrope_kernel(q, k, q_weight, k_weight, positions, cos_sin_cache,
                         q_out, k_out, batch, gemma, is_interleaved, eps,
                         stream);
}

template <typename index_t>
void launch_qk_mrope_cache_kernel(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
    ffi::TensorView q_weight, ffi::TensorView k_weight,
    ffi::TensorView positions, ffi::TensorView cos_sin_cache,
    ffi::TensorView q_out, ffi::TensorView k_cache, ffi::TensorView v_cache,
    ffi::TensorView indices, int batch, bool gemma, bool is_interleaved, float eps,
    musaStream_t stream) {
  constexpr int threads = 256;
  constexpr int groups_per_token = 5;
  if (gemma && is_interleaved) {
    fused_qk_rmsnorm_mrope_cache_32q4kv_h128_bf16_kernel<index_t, true, true,
                                                          false>
        <<<batch * groups_per_token, threads, 0, stream>>>(
            static_cast<const __mt_bfloat16 *>(q.data_ptr()),
            static_cast<const __mt_bfloat16 *>(k.data_ptr()),
            static_cast<const __mt_bfloat16 *>(v.data_ptr()),
            static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
            static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
            static_cast<const int64_t *>(positions.data_ptr()),
            static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
            static_cast<__mt_bfloat16 *>(q_out.data_ptr()), nullptr,
            static_cast<__mt_bfloat16 *>(k_cache.data_ptr()),
            static_cast<__mt_bfloat16 *>(v_cache.data_ptr()),
            static_cast<const index_t *>(indices.data_ptr()), batch,
            static_cast<int64_t>(positions.stride(0)),
            static_cast<int64_t>(q.stride(0)), static_cast<int64_t>(q.stride(1)),
            static_cast<int64_t>(k.stride(0)), static_cast<int64_t>(k.stride(1)),
            static_cast<int64_t>(v.stride(0)), static_cast<int64_t>(v.stride(1)),
            static_cast<int64_t>(q_out.stride(0)),
            static_cast<int64_t>(q_out.stride(1)), 0, 0,
            static_cast<int64_t>(k_cache.stride(0)),
            static_cast<int64_t>(v_cache.stride(0)),
            static_cast<int64_t>(indices.stride(0)), eps);
    return;
  }
  if (gemma) {
    fused_qk_rmsnorm_mrope_cache_32q4kv_h128_bf16_kernel<index_t, true, false,
                                                          false>
        <<<batch * groups_per_token, threads, 0, stream>>>(
            static_cast<const __mt_bfloat16 *>(q.data_ptr()),
            static_cast<const __mt_bfloat16 *>(k.data_ptr()),
            static_cast<const __mt_bfloat16 *>(v.data_ptr()),
            static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
            static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
            static_cast<const int64_t *>(positions.data_ptr()),
            static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
            static_cast<__mt_bfloat16 *>(q_out.data_ptr()), nullptr,
            static_cast<__mt_bfloat16 *>(k_cache.data_ptr()),
            static_cast<__mt_bfloat16 *>(v_cache.data_ptr()),
            static_cast<const index_t *>(indices.data_ptr()), batch,
            static_cast<int64_t>(positions.stride(0)),
            static_cast<int64_t>(q.stride(0)), static_cast<int64_t>(q.stride(1)),
            static_cast<int64_t>(k.stride(0)), static_cast<int64_t>(k.stride(1)),
            static_cast<int64_t>(v.stride(0)), static_cast<int64_t>(v.stride(1)),
            static_cast<int64_t>(q_out.stride(0)),
            static_cast<int64_t>(q_out.stride(1)), 0, 0,
            static_cast<int64_t>(k_cache.stride(0)),
            static_cast<int64_t>(v_cache.stride(0)),
            static_cast<int64_t>(indices.stride(0)), eps);
    return;
  }
  if (is_interleaved) {
    fused_qk_rmsnorm_mrope_cache_32q4kv_h128_bf16_kernel<index_t, false, true,
                                                          false>
        <<<batch * groups_per_token, threads, 0, stream>>>(
            static_cast<const __mt_bfloat16 *>(q.data_ptr()),
            static_cast<const __mt_bfloat16 *>(k.data_ptr()),
            static_cast<const __mt_bfloat16 *>(v.data_ptr()),
            static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
            static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
            static_cast<const int64_t *>(positions.data_ptr()),
            static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
            static_cast<__mt_bfloat16 *>(q_out.data_ptr()), nullptr,
            static_cast<__mt_bfloat16 *>(k_cache.data_ptr()),
            static_cast<__mt_bfloat16 *>(v_cache.data_ptr()),
            static_cast<const index_t *>(indices.data_ptr()), batch,
            static_cast<int64_t>(positions.stride(0)),
            static_cast<int64_t>(q.stride(0)), static_cast<int64_t>(q.stride(1)),
            static_cast<int64_t>(k.stride(0)), static_cast<int64_t>(k.stride(1)),
            static_cast<int64_t>(v.stride(0)), static_cast<int64_t>(v.stride(1)),
            static_cast<int64_t>(q_out.stride(0)),
            static_cast<int64_t>(q_out.stride(1)), 0, 0,
            static_cast<int64_t>(k_cache.stride(0)),
            static_cast<int64_t>(v_cache.stride(0)),
            static_cast<int64_t>(indices.stride(0)), eps);
    return;
  }
  fused_qk_rmsnorm_mrope_cache_32q4kv_h128_bf16_kernel<index_t, false, false,
                                                        false>
      <<<batch * groups_per_token, threads, 0, stream>>>(
          static_cast<const __mt_bfloat16 *>(q.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k.data_ptr()),
          static_cast<const __mt_bfloat16 *>(v.data_ptr()),
          static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
          static_cast<const int64_t *>(positions.data_ptr()),
          static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(q_out.data_ptr()), nullptr,
          static_cast<__mt_bfloat16 *>(k_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(v_cache.data_ptr()),
          static_cast<const index_t *>(indices.data_ptr()), batch,
          static_cast<int64_t>(positions.stride(0)),
          static_cast<int64_t>(q.stride(0)), static_cast<int64_t>(q.stride(1)),
          static_cast<int64_t>(k.stride(0)), static_cast<int64_t>(k.stride(1)),
          static_cast<int64_t>(v.stride(0)), static_cast<int64_t>(v.stride(1)),
          static_cast<int64_t>(q_out.stride(0)),
	          static_cast<int64_t>(q_out.stride(1)), 0, 0,
	          static_cast<int64_t>(k_cache.stride(0)),
	          static_cast<int64_t>(v_cache.stride(0)),
	          static_cast<int64_t>(indices.stride(0)), eps);
}

template <typename index_t, int Q_HEADS, int K_HEADS, bool GEMMA,
          int HEADS_PER_BLOCK = 8>
void launch_h128_full_mrope_cache_kernel(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
    ffi::TensorView q_weight, ffi::TensorView k_weight,
    ffi::TensorView positions, ffi::TensorView cos_sin_cache,
    ffi::TensorView q_out, ffi::TensorView k_cache, ffi::TensorView v_cache,
    ffi::TensorView indices, int batch, float eps, musaStream_t stream) {
  constexpr int heads_per_block = HEADS_PER_BLOCK;
  constexpr int threads = 32 * heads_per_block;
  constexpr int groups_per_token =
      (Q_HEADS + K_HEADS + heads_per_block - 1) / heads_per_block;
#define LAUNCH_H128_FULL_MROPE_CACHE(SAME_POSITION_VALUE)                    \
  fused_qk_rmsnorm_mrope_cache_h128_full_bf16_kernel<                        \
      index_t, Q_HEADS, K_HEADS, GEMMA, HEADS_PER_BLOCK, false,              \
      SAME_POSITION_VALUE>                                                   \
      <<<batch * groups_per_token, threads, 0, stream>>>(                    \
          static_cast<const __mt_bfloat16 *>(q.data_ptr()),                  \
          static_cast<const __mt_bfloat16 *>(k.data_ptr()),                  \
          static_cast<const __mt_bfloat16 *>(v.data_ptr()),                  \
          static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),           \
          static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),           \
          static_cast<const int64_t *>(positions.data_ptr()),                \
          static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),      \
          static_cast<__mt_bfloat16 *>(q_out.data_ptr()), nullptr,           \
          static_cast<__mt_bfloat16 *>(k_cache.data_ptr()),                  \
          static_cast<__mt_bfloat16 *>(v_cache.data_ptr()),                  \
          static_cast<const index_t *>(indices.data_ptr()), batch,           \
          static_cast<int64_t>(positions.stride(0)),                         \
          static_cast<int64_t>(q.stride(0)),                                 \
          static_cast<int64_t>(q.stride(1)),                                 \
          static_cast<int64_t>(k.stride(0)),                                 \
          static_cast<int64_t>(k.stride(1)),                                 \
          static_cast<int64_t>(v.stride(0)),                                 \
          static_cast<int64_t>(v.stride(1)),                                 \
          static_cast<int64_t>(q_out.stride(0)),                             \
          static_cast<int64_t>(q_out.stride(1)), 0, 0,                       \
          static_cast<int64_t>(k_cache.stride(0)),                           \
          static_cast<int64_t>(v_cache.stride(0)),                           \
          static_cast<int64_t>(indices.stride(0)), eps)
  if (positions.stride(0) == 0) {
    LAUNCH_H128_FULL_MROPE_CACHE(true);
  } else {
    LAUNCH_H128_FULL_MROPE_CACHE(false);
  }
#undef LAUNCH_H128_FULL_MROPE_CACHE
}

template <typename index_t, int Q_HEADS, int K_HEADS, bool GEMMA,
          int HEADS_PER_BLOCK = 8>
void launch_h128_full_mrope_cache_out_kernel(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
    ffi::TensorView q_weight, ffi::TensorView k_weight,
    ffi::TensorView positions, ffi::TensorView cos_sin_cache,
    ffi::TensorView q_out, ffi::TensorView k_out, ffi::TensorView k_cache,
    ffi::TensorView v_cache, ffi::TensorView indices, int batch, float eps,
    musaStream_t stream) {
  constexpr int heads_per_block = HEADS_PER_BLOCK;
  constexpr int threads = 32 * heads_per_block;
  constexpr int groups_per_token =
      (Q_HEADS + K_HEADS + heads_per_block - 1) / heads_per_block;
#define LAUNCH_H128_FULL_MROPE_CACHE_OUT(SAME_POSITION_VALUE)                \
  fused_qk_rmsnorm_mrope_cache_h128_full_bf16_kernel<                        \
      index_t, Q_HEADS, K_HEADS, GEMMA, HEADS_PER_BLOCK, true,               \
      SAME_POSITION_VALUE>                                                   \
      <<<batch * groups_per_token, threads, 0, stream>>>(                    \
          static_cast<const __mt_bfloat16 *>(q.data_ptr()),                  \
          static_cast<const __mt_bfloat16 *>(k.data_ptr()),                  \
          static_cast<const __mt_bfloat16 *>(v.data_ptr()),                  \
          static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),           \
          static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),           \
          static_cast<const int64_t *>(positions.data_ptr()),                \
          static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),      \
          static_cast<__mt_bfloat16 *>(q_out.data_ptr()),                    \
          static_cast<__mt_bfloat16 *>(k_out.data_ptr()),                    \
          static_cast<__mt_bfloat16 *>(k_cache.data_ptr()),                  \
          static_cast<__mt_bfloat16 *>(v_cache.data_ptr()),                  \
          static_cast<const index_t *>(indices.data_ptr()), batch,           \
          static_cast<int64_t>(positions.stride(0)),                         \
          static_cast<int64_t>(q.stride(0)),                                 \
          static_cast<int64_t>(q.stride(1)),                                 \
          static_cast<int64_t>(k.stride(0)),                                 \
          static_cast<int64_t>(k.stride(1)),                                 \
          static_cast<int64_t>(v.stride(0)),                                 \
          static_cast<int64_t>(v.stride(1)),                                 \
          static_cast<int64_t>(q_out.stride(0)),                             \
          static_cast<int64_t>(q_out.stride(1)),                             \
          static_cast<int64_t>(k_out.stride(0)),                             \
          static_cast<int64_t>(k_out.stride(1)),                             \
          static_cast<int64_t>(k_cache.stride(0)),                           \
          static_cast<int64_t>(v_cache.stride(0)),                           \
          static_cast<int64_t>(indices.stride(0)), eps)
  if (positions.stride(0) == 0) {
    LAUNCH_H128_FULL_MROPE_CACHE_OUT(true);
  } else {
    LAUNCH_H128_FULL_MROPE_CACHE_OUT(false);
  }
#undef LAUNCH_H128_FULL_MROPE_CACHE_OUT
}

template <typename index_t, int Q_HEADS, int K_HEADS, bool GEMMA,
          int HEADS_PER_BLOCK = 8>
void launch_h128_full_rope_cache_kernel(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
    ffi::TensorView q_weight, ffi::TensorView k_weight,
    ffi::TensorView positions, ffi::TensorView cos_sin_cache,
    ffi::TensorView q_out, ffi::TensorView k_cache, ffi::TensorView v_cache,
    ffi::TensorView indices, int batch, float eps, musaStream_t stream) {
  constexpr int heads_per_block = HEADS_PER_BLOCK;
  constexpr int threads = 32 * heads_per_block;
  constexpr int groups_per_token =
      (Q_HEADS + K_HEADS + heads_per_block - 1) / heads_per_block;
  fused_qk_rmsnorm_rope_cache_h128_full_bf16_kernel<
      index_t, Q_HEADS, K_HEADS, GEMMA, HEADS_PER_BLOCK, false>
      <<<batch * groups_per_token, threads, 0, stream>>>(
          static_cast<const __mt_bfloat16 *>(q.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k.data_ptr()),
          static_cast<const __mt_bfloat16 *>(v.data_ptr()),
          static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
          static_cast<const int64_t *>(positions.data_ptr()),
          static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(q_out.data_ptr()), nullptr,
          static_cast<__mt_bfloat16 *>(k_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(v_cache.data_ptr()),
          static_cast<const index_t *>(indices.data_ptr()), batch,
          static_cast<int64_t>(positions.stride(0)),
          static_cast<int64_t>(q.stride(0)),
          static_cast<int64_t>(q.stride(1)),
          static_cast<int64_t>(k.stride(0)),
          static_cast<int64_t>(k.stride(1)),
          static_cast<int64_t>(v.stride(0)),
          static_cast<int64_t>(v.stride(1)),
          static_cast<int64_t>(q_out.stride(0)),
          static_cast<int64_t>(q_out.stride(1)), 0, 0,
          static_cast<int64_t>(k_cache.stride(0)),
          static_cast<int64_t>(v_cache.stride(0)),
          static_cast<int64_t>(indices.stride(0)), eps);
}

template <typename index_t, int Q_HEADS, int K_HEADS, bool GEMMA,
          int HEADS_PER_BLOCK = 8>
void launch_h128_full_rope_cache_out_kernel(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
    ffi::TensorView q_weight, ffi::TensorView k_weight,
    ffi::TensorView positions, ffi::TensorView cos_sin_cache,
    ffi::TensorView q_out, ffi::TensorView k_out, ffi::TensorView k_cache,
    ffi::TensorView v_cache, ffi::TensorView indices, int batch, float eps,
    musaStream_t stream) {
  constexpr int heads_per_block = HEADS_PER_BLOCK;
  constexpr int threads = 32 * heads_per_block;
  constexpr int groups_per_token =
      (Q_HEADS + K_HEADS + heads_per_block - 1) / heads_per_block;
  fused_qk_rmsnorm_rope_cache_h128_full_bf16_kernel<
      index_t, Q_HEADS, K_HEADS, GEMMA, HEADS_PER_BLOCK, true>
      <<<batch * groups_per_token, threads, 0, stream>>>(
          static_cast<const __mt_bfloat16 *>(q.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k.data_ptr()),
          static_cast<const __mt_bfloat16 *>(v.data_ptr()),
          static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
          static_cast<const int64_t *>(positions.data_ptr()),
          static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
          static_cast<__mt_bfloat16 *>(k_out.data_ptr()),
          static_cast<__mt_bfloat16 *>(k_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(v_cache.data_ptr()),
          static_cast<const index_t *>(indices.data_ptr()), batch,
          static_cast<int64_t>(positions.stride(0)),
          static_cast<int64_t>(q.stride(0)),
          static_cast<int64_t>(q.stride(1)),
          static_cast<int64_t>(k.stride(0)),
          static_cast<int64_t>(k.stride(1)),
          static_cast<int64_t>(v.stride(0)),
          static_cast<int64_t>(v.stride(1)),
          static_cast<int64_t>(q_out.stride(0)),
          static_cast<int64_t>(q_out.stride(1)),
          static_cast<int64_t>(k_out.stride(0)),
          static_cast<int64_t>(k_out.stride(1)),
          static_cast<int64_t>(k_cache.stride(0)),
          static_cast<int64_t>(v_cache.stride(0)),
          static_cast<int64_t>(indices.stride(0)), eps);
}

template <typename index_t, int Q_HEADS, int K_HEADS, bool GEMMA,
          int HEADS_PER_BLOCK = 4>
void launch_h128r64_rope_cache_kernel(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
    ffi::TensorView q_weight, ffi::TensorView k_weight,
    ffi::TensorView positions, ffi::TensorView cos_sin_cache,
    ffi::TensorView q_out, ffi::TensorView k_cache, ffi::TensorView v_cache,
    ffi::TensorView indices, int batch, float eps, musaStream_t stream) {
  constexpr int heads_per_block = HEADS_PER_BLOCK;
  constexpr int threads = 32 * heads_per_block;
  constexpr int groups_per_token =
      (Q_HEADS + K_HEADS + heads_per_block - 1) / heads_per_block;
  fused_qk_rmsnorm_rope_cache_h128r64_bf16_kernel<
      index_t, Q_HEADS, K_HEADS, GEMMA, HEADS_PER_BLOCK, false>
      <<<batch * groups_per_token, threads, 0, stream>>>(
          static_cast<const __mt_bfloat16 *>(q.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k.data_ptr()),
          static_cast<const __mt_bfloat16 *>(v.data_ptr()),
          static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
          static_cast<const int64_t *>(positions.data_ptr()),
          static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(q_out.data_ptr()), nullptr,
          static_cast<__mt_bfloat16 *>(k_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(v_cache.data_ptr()),
          static_cast<const index_t *>(indices.data_ptr()), batch,
          static_cast<int64_t>(positions.stride(0)),
          static_cast<int64_t>(q.stride(0)),
          static_cast<int64_t>(q.stride(1)),
          static_cast<int64_t>(k.stride(0)),
          static_cast<int64_t>(k.stride(1)),
          static_cast<int64_t>(v.stride(0)),
          static_cast<int64_t>(v.stride(1)),
          static_cast<int64_t>(q_out.stride(0)),
          static_cast<int64_t>(q_out.stride(1)), 0, 0,
          static_cast<int64_t>(k_cache.stride(0)),
          static_cast<int64_t>(v_cache.stride(0)),
          static_cast<int64_t>(indices.stride(0)), eps);
}

template <typename index_t, int Q_HEADS, int K_HEADS, bool GEMMA,
          int HEADS_PER_BLOCK = 4>
void launch_h128r64_rope_cache_out_kernel(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
    ffi::TensorView q_weight, ffi::TensorView k_weight,
    ffi::TensorView positions, ffi::TensorView cos_sin_cache,
    ffi::TensorView q_out, ffi::TensorView k_out, ffi::TensorView k_cache,
    ffi::TensorView v_cache, ffi::TensorView indices, int batch, float eps,
    musaStream_t stream) {
  constexpr int heads_per_block = HEADS_PER_BLOCK;
  constexpr int threads = 32 * heads_per_block;
  constexpr int groups_per_token =
      (Q_HEADS + K_HEADS + heads_per_block - 1) / heads_per_block;
  fused_qk_rmsnorm_rope_cache_h128r64_bf16_kernel<
      index_t, Q_HEADS, K_HEADS, GEMMA, HEADS_PER_BLOCK, true>
      <<<batch * groups_per_token, threads, 0, stream>>>(
          static_cast<const __mt_bfloat16 *>(q.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k.data_ptr()),
          static_cast<const __mt_bfloat16 *>(v.data_ptr()),
          static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
          static_cast<const int64_t *>(positions.data_ptr()),
          static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
          static_cast<__mt_bfloat16 *>(k_out.data_ptr()),
          static_cast<__mt_bfloat16 *>(k_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(v_cache.data_ptr()),
          static_cast<const index_t *>(indices.data_ptr()), batch,
          static_cast<int64_t>(positions.stride(0)),
          static_cast<int64_t>(q.stride(0)),
          static_cast<int64_t>(q.stride(1)),
          static_cast<int64_t>(k.stride(0)),
          static_cast<int64_t>(k.stride(1)),
          static_cast<int64_t>(v.stride(0)),
          static_cast<int64_t>(v.stride(1)),
          static_cast<int64_t>(q_out.stride(0)),
          static_cast<int64_t>(q_out.stride(1)),
          static_cast<int64_t>(k_out.stride(0)),
          static_cast<int64_t>(k_out.stride(1)),
          static_cast<int64_t>(k_cache.stride(0)),
          static_cast<int64_t>(v_cache.stride(0)),
          static_cast<int64_t>(indices.stride(0)), eps);
}

template <typename index_t, int Q_HEADS, int K_HEADS, int HEADS_PER_BLOCK = 8>
void launch_h256r64_mrope_cache_kernel(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
    ffi::TensorView q_weight, ffi::TensorView k_weight,
    ffi::TensorView positions, ffi::TensorView cos_sin_cache,
    ffi::TensorView q_out, ffi::TensorView k_cache, ffi::TensorView v_cache,
    ffi::TensorView indices, int batch, float eps, musaStream_t stream) {
  constexpr int heads_per_block = HEADS_PER_BLOCK;
  constexpr int threads = 32 * heads_per_block;
  constexpr int groups_per_token =
      (Q_HEADS + K_HEADS + heads_per_block - 1) / heads_per_block;
  fused_qk_rmsnorm_mrope_cache_h256r64_bf16_kernel<
      index_t, Q_HEADS, K_HEADS, HEADS_PER_BLOCK, false>
      <<<batch * groups_per_token, threads, 0, stream>>>(
          static_cast<const __mt_bfloat16 *>(q.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k.data_ptr()),
          static_cast<const __mt_bfloat16 *>(v.data_ptr()),
          static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
          static_cast<const int64_t *>(positions.data_ptr()),
          static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(q_out.data_ptr()), nullptr,
          static_cast<__mt_bfloat16 *>(k_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(v_cache.data_ptr()),
          static_cast<const index_t *>(indices.data_ptr()), batch,
          static_cast<int64_t>(positions.stride(0)),
          static_cast<int64_t>(q.stride(0)), static_cast<int64_t>(q.stride(1)),
          static_cast<int64_t>(k.stride(0)), static_cast<int64_t>(k.stride(1)),
          static_cast<int64_t>(v.stride(0)), static_cast<int64_t>(v.stride(1)),
          static_cast<int64_t>(q_out.stride(0)),
          static_cast<int64_t>(q_out.stride(1)), 0, 0,
          0, 0, static_cast<int64_t>(k_cache.stride(0)), 0,
          static_cast<int64_t>(v_cache.stride(0)),
          static_cast<int64_t>(indices.stride(0)), eps);
}

template <typename index_t, int Q_HEADS, int K_HEADS, int HEADS_PER_BLOCK = 8>
void launch_h256r64_mrope_cache_out_kernel(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
    ffi::TensorView q_weight, ffi::TensorView k_weight,
    ffi::TensorView positions, ffi::TensorView cos_sin_cache,
    ffi::TensorView q_out, ffi::TensorView k_out, ffi::TensorView k_cache,
    ffi::TensorView v_cache, ffi::TensorView indices, int batch, float eps,
    musaStream_t stream) {
  constexpr int heads_per_block = HEADS_PER_BLOCK;
  constexpr int threads = 32 * heads_per_block;
  constexpr int groups_per_token =
      (Q_HEADS + K_HEADS + heads_per_block - 1) / heads_per_block;
  const int cache_block_size =
      k_cache.ndim() == 4 ? static_cast<int>(k_cache.size(1)) : 0;
  const int64_t k_cache_token_stride =
      k_cache.ndim() == 4 ? static_cast<int64_t>(k_cache.stride(1))
                          : static_cast<int64_t>(k_cache.stride(0));
  const int64_t v_cache_token_stride =
      v_cache.ndim() == 4 ? static_cast<int64_t>(v_cache.stride(1))
                          : static_cast<int64_t>(v_cache.stride(0));
  fused_qk_rmsnorm_mrope_cache_h256r64_bf16_kernel<
      index_t, Q_HEADS, K_HEADS, HEADS_PER_BLOCK>
      <<<batch * groups_per_token, threads, 0, stream>>>(
          static_cast<const __mt_bfloat16 *>(q.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k.data_ptr()),
          static_cast<const __mt_bfloat16 *>(v.data_ptr()),
          static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
          static_cast<const int64_t *>(positions.data_ptr()),
          static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
          static_cast<__mt_bfloat16 *>(k_out.data_ptr()),
          static_cast<__mt_bfloat16 *>(k_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(v_cache.data_ptr()),
          static_cast<const index_t *>(indices.data_ptr()), batch,
          static_cast<int64_t>(positions.stride(0)),
          static_cast<int64_t>(q.stride(0)), static_cast<int64_t>(q.stride(1)),
          static_cast<int64_t>(k.stride(0)), static_cast<int64_t>(k.stride(1)),
          static_cast<int64_t>(v.stride(0)), static_cast<int64_t>(v.stride(1)),
          static_cast<int64_t>(q_out.stride(0)),
          static_cast<int64_t>(q_out.stride(1)),
          static_cast<int64_t>(k_out.stride(0)),
          static_cast<int64_t>(k_out.stride(1)),
          cache_block_size, static_cast<int64_t>(k_cache.stride(0)),
          k_cache_token_stride, static_cast<int64_t>(v_cache.stride(0)),
          v_cache_token_stride,
          static_cast<int64_t>(indices.stride(0)), eps);
}

template <typename index_t>
void launch_fused_qk_rmsnorm_mrope_cache_bf16(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
    ffi::TensorView q_weight, ffi::TensorView k_weight,
    ffi::TensorView positions, ffi::TensorView cos_sin_cache,
    ffi::TensorView q_out, ffi::TensorView k_cache, ffi::TensorView v_cache,
    ffi::TensorView indices, int mrope_section_t, int mrope_section_h,
    int mrope_section_w, bool is_interleaved, bool gemma, float eps) {
  const int batch = static_cast<int>(q.size(0));
  if (batch == 0) {
    return;
  }
  musaStream_t stream = get_stream(q.device());
  const int q_heads = static_cast<int>(q.size(1));
  const int k_heads = static_cast<int>(k.size(1));
  const int hidden = static_cast<int>(q.size(2));
  const int rot_dim = static_cast<int>(cos_sin_cache.size(1));
  const bool use_32q4kv_h128_mrope_specialization =
      q_heads == 32 && k_heads == 4 && hidden == 128 && rot_dim == 128 &&
      mrope_section_t == 24 && mrope_section_h == 20 &&
      mrope_section_w == 20;
  const bool use_h256r64_mrope_specialization =
      gemma && is_interleaved && hidden == 256 && rot_dim == 64 &&
      mrope_section_t == 11 && mrope_section_h == 11 &&
      mrope_section_w == 10;
  if (use_h256r64_mrope_specialization) {
    if (q_heads == 4 && k_heads == 1) {
      launch_h256r64_mrope_cache_kernel<index_t, 4, 1>(
          q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_cache,
          v_cache, indices, batch, eps, stream);
      return;
    }
    if (q_heads == 8 && k_heads == 1) {
      launch_h256r64_mrope_cache_kernel<index_t, 8, 1, 16>(
          q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_cache,
          v_cache, indices, batch, eps, stream);
      return;
    }
    if (q_heads == 16 && k_heads == 1) {
      launch_h256r64_mrope_cache_kernel<index_t, 16, 1>(
          q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_cache,
          v_cache, indices, batch, eps, stream);
      return;
    }
    if (q_heads == 16 && k_heads == 2) {
      launch_h256r64_mrope_cache_kernel<index_t, 16, 2>(
          q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_cache,
          v_cache, indices, batch, eps, stream);
      return;
    }
    if (q_heads == 16 && k_heads == 4) {
      launch_h256r64_mrope_cache_kernel<index_t, 16, 4>(
          q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_cache,
          v_cache, indices, batch, eps, stream);
      return;
    }
    if (q_heads == 24 && k_heads == 4) {
      launch_h256r64_mrope_cache_kernel<index_t, 24, 4>(
          q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_cache,
          v_cache, indices, batch, eps, stream);
      return;
    }
    if (q_heads == 32 && k_heads == 2) {
      launch_h256r64_mrope_cache_kernel<index_t, 32, 2>(
          q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_cache,
          v_cache, indices, batch, eps, stream);
      return;
    }
    if (q_heads == 32 && k_heads == 4) {
      launch_h256r64_mrope_cache_kernel<index_t, 32, 4>(
          q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_cache,
          v_cache, indices, batch, eps, stream);
      return;
    }
    if (q_heads == 64 && k_heads == 4) {
      launch_h256r64_mrope_cache_kernel<index_t, 64, 4>(
          q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_cache,
          v_cache, indices, batch, eps, stream);
      return;
    }
    if (q_heads == 64 && k_heads == 8) {
      launch_h256r64_mrope_cache_kernel<index_t, 64, 8>(
          q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_cache,
          v_cache, indices, batch, eps, stream);
      return;
    }
  }
  const bool use_h128_full_mrope_specialization =
      is_interleaved && hidden == 128 && rot_dim == 128 && mrope_section_t == 24 &&
      mrope_section_h == 20 && mrope_section_w == 20;
  if (use_h128_full_mrope_specialization) {
#define LAUNCH_H128_MROPE_CACHE(QH, KH)                                       \
  do {                                                                        \
    if (q_heads == (QH) && k_heads == (KH)) {                                 \
      if (gemma) {                                                            \
        launch_h128_full_mrope_cache_kernel<index_t, QH, KH, true>(           \
            q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out,     \
            k_cache, v_cache, indices, batch, eps, stream);                   \
      } else {                                                                \
        launch_h128_full_mrope_cache_kernel<index_t, QH, KH, false>(          \
            q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out,     \
            k_cache, v_cache, indices, batch, eps, stream);                   \
      }                                                                       \
      return;                                                                 \
    }                                                                         \
  } while (0)
    LAUNCH_H128_MROPE_CACHE(2, 1);
    LAUNCH_H128_MROPE_CACHE(4, 1);
    LAUNCH_H128_MROPE_CACHE(4, 2);
    LAUNCH_H128_MROPE_CACHE(8, 2);
    LAUNCH_H128_MROPE_CACHE(8, 4);
    LAUNCH_H128_MROPE_CACHE(16, 4);
    LAUNCH_H128_MROPE_CACHE(16, 8);
    LAUNCH_H128_MROPE_CACHE(32, 8);
    LAUNCH_H128_MROPE_CACHE(2, 2);
    LAUNCH_H128_MROPE_CACHE(4, 4);
    LAUNCH_H128_MROPE_CACHE(8, 8);
    LAUNCH_H128_MROPE_CACHE(16, 16);
    LAUNCH_H128_MROPE_CACHE(32, 32);
#undef LAUNCH_H128_MROPE_CACHE
  }
  const bool use_h128_full_rope_specialization =
      hidden == 128 && rot_dim == 128 && mrope_section_t == 64 &&
      mrope_section_h == 0 && mrope_section_w == 0 && !is_interleaved;
  if (use_h128_full_rope_specialization) {
#define LAUNCH_H128_ROPE_CACHE(QH, KH)                                        \
  do {                                                                        \
    if (q_heads == (QH) && k_heads == (KH)) {                                 \
      if (gemma) {                                                            \
        launch_h128_full_rope_cache_kernel<index_t, QH, KH, true>(            \
            q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out,     \
            k_cache, v_cache, indices, batch, eps, stream);                   \
      } else {                                                                \
        launch_h128_full_rope_cache_kernel<index_t, QH, KH, false>(           \
            q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out,     \
            k_cache, v_cache, indices, batch, eps, stream);                   \
      }                                                                       \
      return;                                                                 \
    }                                                                         \
  } while (0)
    LAUNCH_H128_ROPE_CACHE(8, 8);
    LAUNCH_H128_ROPE_CACHE(16, 8);
    LAUNCH_H128_ROPE_CACHE(24, 8);
    LAUNCH_H128_ROPE_CACHE(32, 8);
    LAUNCH_H128_ROPE_CACHE(40, 8);
    LAUNCH_H128_ROPE_CACHE(48, 8);
    LAUNCH_H128_ROPE_CACHE(64, 8);
    LAUNCH_H128_ROPE_CACHE(5, 1);
    LAUNCH_H128_ROPE_CACHE(8, 1);
    LAUNCH_H128_ROPE_CACHE(10, 2);
    LAUNCH_H128_ROPE_CACHE(16, 1);
    LAUNCH_H128_ROPE_CACHE(16, 2);
    LAUNCH_H128_ROPE_CACHE(16, 4);
    LAUNCH_H128_ROPE_CACHE(20, 4);
    LAUNCH_H128_ROPE_CACHE(24, 4);
    LAUNCH_H128_ROPE_CACHE(32, 2);
    LAUNCH_H128_ROPE_CACHE(32, 4);
    LAUNCH_H128_ROPE_CACHE(40, 4);
    LAUNCH_H128_ROPE_CACHE(4, 1);
    LAUNCH_H128_ROPE_CACHE(8, 2);
#undef LAUNCH_H128_ROPE_CACHE
  }
  const bool use_h128r64_rope_specialization =
      hidden == 128 && rot_dim == 64 && mrope_section_t == 32 &&
      mrope_section_h == 0 && mrope_section_w == 0 && !is_interleaved;
  if (use_h128r64_rope_specialization) {
#define LAUNCH_H128R64_ROPE_CACHE(QH, KH)                                     \
  do {                                                                        \
    if (q_heads == (QH) && k_heads == (KH)) {                                 \
      if (gemma) {                                                            \
        launch_h128r64_rope_cache_kernel<index_t, QH, KH, true>(              \
            q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out,     \
            k_cache, v_cache, indices, batch, eps, stream);                   \
      } else {                                                                \
        launch_h128r64_rope_cache_kernel<index_t, QH, KH, false>(             \
            q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out,     \
            k_cache, v_cache, indices, batch, eps, stream);                   \
      }                                                                       \
      return;                                                                 \
    }                                                                         \
  } while (0)
    LAUNCH_H128R64_ROPE_CACHE(6, 1);
    LAUNCH_H128R64_ROPE_CACHE(8, 1);
    LAUNCH_H128R64_ROPE_CACHE(12, 1);
    LAUNCH_H128R64_ROPE_CACHE(12, 2);
    LAUNCH_H128R64_ROPE_CACHE(16, 2);
    LAUNCH_H128R64_ROPE_CACHE(24, 2);
    LAUNCH_H128R64_ROPE_CACHE(24, 4);
    LAUNCH_H128R64_ROPE_CACHE(32, 4);
    LAUNCH_H128R64_ROPE_CACHE(48, 4);
    LAUNCH_H128R64_ROPE_CACHE(48, 8);
    LAUNCH_H128R64_ROPE_CACHE(64, 8);
    LAUNCH_H128R64_ROPE_CACHE(96, 8);
#undef LAUNCH_H128R64_ROPE_CACHE
  }
  if (!use_32q4kv_h128_mrope_specialization) {
    constexpr int threads = 256;
    constexpr int heads_per_block = 8;
    const int groups_per_token =
        (q_heads + k_heads + heads_per_block - 1) / heads_per_block;
    if (gemma) {
      fused_qk_rmsnorm_mrope_generic_bf16_kernel<index_t, true, true>
          <<<batch * groups_per_token, threads, 0, stream>>>(
              static_cast<const __mt_bfloat16 *>(q.data_ptr()),
              static_cast<const __mt_bfloat16 *>(k.data_ptr()),
              static_cast<const __mt_bfloat16 *>(v.data_ptr()),
              static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
              static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
              static_cast<const int64_t *>(positions.data_ptr()),
              static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
              static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
              static_cast<__mt_bfloat16 *>(nullptr),
              static_cast<__mt_bfloat16 *>(k_cache.data_ptr()),
              static_cast<__mt_bfloat16 *>(v_cache.data_ptr()),
              static_cast<const index_t *>(indices.data_ptr()), batch, q_heads,
              k_heads, hidden, rot_dim, static_cast<int64_t>(positions.stride(0)),
              static_cast<int64_t>(q.stride(0)),
              static_cast<int64_t>(q.stride(1)),
              static_cast<int64_t>(k.stride(0)),
              static_cast<int64_t>(k.stride(1)),
              static_cast<int64_t>(v.stride(0)),
              static_cast<int64_t>(v.stride(1)),
              static_cast<int64_t>(q_out.stride(0)),
              static_cast<int64_t>(q_out.stride(1)), static_cast<int64_t>(0),
              static_cast<int64_t>(0), static_cast<int64_t>(k_cache.stride(0)),
              static_cast<int64_t>(v_cache.stride(0)),
              static_cast<int64_t>(indices.stride(0)), mrope_section_t,
              mrope_section_h, mrope_section_w, is_interleaved, eps);
      return;
    }
    fused_qk_rmsnorm_mrope_generic_bf16_kernel<index_t, true, false>
        <<<batch * groups_per_token, threads, 0, stream>>>(
            static_cast<const __mt_bfloat16 *>(q.data_ptr()),
            static_cast<const __mt_bfloat16 *>(k.data_ptr()),
            static_cast<const __mt_bfloat16 *>(v.data_ptr()),
            static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
            static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
            static_cast<const int64_t *>(positions.data_ptr()),
            static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
            static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
            static_cast<__mt_bfloat16 *>(nullptr),
            static_cast<__mt_bfloat16 *>(k_cache.data_ptr()),
            static_cast<__mt_bfloat16 *>(v_cache.data_ptr()),
            static_cast<const index_t *>(indices.data_ptr()), batch, q_heads,
            k_heads, hidden, rot_dim, static_cast<int64_t>(positions.stride(0)),
            static_cast<int64_t>(q.stride(0)), static_cast<int64_t>(q.stride(1)),
            static_cast<int64_t>(k.stride(0)), static_cast<int64_t>(k.stride(1)),
            static_cast<int64_t>(v.stride(0)), static_cast<int64_t>(v.stride(1)),
            static_cast<int64_t>(q_out.stride(0)),
            static_cast<int64_t>(q_out.stride(1)), static_cast<int64_t>(0),
            static_cast<int64_t>(0), static_cast<int64_t>(k_cache.stride(0)),
            static_cast<int64_t>(v_cache.stride(0)),
            static_cast<int64_t>(indices.stride(0)), mrope_section_t,
            mrope_section_h, mrope_section_w, is_interleaved, eps);
    return;
  }
  launch_qk_mrope_cache_kernel<index_t>(
      q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_cache,
      v_cache, indices, batch, gemma, is_interleaved, eps, stream);
}

template <typename index_t>
void launch_qk_mrope_cache_out_kernel(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
    ffi::TensorView q_weight, ffi::TensorView k_weight,
    ffi::TensorView positions, ffi::TensorView cos_sin_cache,
    ffi::TensorView q_out, ffi::TensorView k_out, ffi::TensorView k_cache,
    ffi::TensorView v_cache, ffi::TensorView indices, int batch, bool gemma,
    bool is_interleaved, float eps, musaStream_t stream) {
  constexpr int threads = 256;
  constexpr int groups_per_token = 5;
#define LAUNCH_CACHE_OUT_32Q4(GEMMA_VALUE, INTERLEAVED_VALUE)                 \
  fused_qk_rmsnorm_mrope_cache_32q4kv_h128_bf16_kernel<                       \
      index_t, GEMMA_VALUE, INTERLEAVED_VALUE>                                \
      <<<batch * groups_per_token, threads, 0, stream>>>(                     \
          static_cast<const __mt_bfloat16 *>(q.data_ptr()),                   \
          static_cast<const __mt_bfloat16 *>(k.data_ptr()),                   \
          static_cast<const __mt_bfloat16 *>(v.data_ptr()),                   \
          static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),            \
          static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),            \
          static_cast<const int64_t *>(positions.data_ptr()),                 \
          static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),       \
          static_cast<__mt_bfloat16 *>(q_out.data_ptr()),                     \
          static_cast<__mt_bfloat16 *>(k_out.data_ptr()),                     \
          static_cast<__mt_bfloat16 *>(k_cache.data_ptr()),                   \
          static_cast<__mt_bfloat16 *>(v_cache.data_ptr()),                   \
          static_cast<const index_t *>(indices.data_ptr()), batch,            \
          static_cast<int64_t>(positions.stride(0)),                          \
          static_cast<int64_t>(q.stride(0)),                                  \
          static_cast<int64_t>(q.stride(1)),                                  \
          static_cast<int64_t>(k.stride(0)),                                  \
          static_cast<int64_t>(k.stride(1)),                                  \
          static_cast<int64_t>(v.stride(0)),                                  \
          static_cast<int64_t>(v.stride(1)),                                  \
          static_cast<int64_t>(q_out.stride(0)),                              \
          static_cast<int64_t>(q_out.stride(1)),                              \
          static_cast<int64_t>(k_out.stride(0)),                              \
          static_cast<int64_t>(k_out.stride(1)),                              \
          static_cast<int64_t>(k_cache.stride(0)),                            \
          static_cast<int64_t>(v_cache.stride(0)),                            \
          static_cast<int64_t>(indices.stride(0)), eps)

  if (gemma && is_interleaved) {
    LAUNCH_CACHE_OUT_32Q4(true, true);
    return;
  }
  if (gemma) {
    LAUNCH_CACHE_OUT_32Q4(true, false);
    return;
  }
  if (is_interleaved) {
    LAUNCH_CACHE_OUT_32Q4(false, true);
    return;
  }
  LAUNCH_CACHE_OUT_32Q4(false, false);
#undef LAUNCH_CACHE_OUT_32Q4
}

void sgl_musa_fused_qk_rmsnorm_mrope(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView q_weight,
    ffi::TensorView k_weight, ffi::TensorView positions,
    ffi::TensorView cos_sin_cache, ffi::TensorView q_out, ffi::TensorView k_out,
    bool is_neox, int mrope_section_t, int mrope_section_h,
    int mrope_section_w, bool is_interleaved, double eps, bool gemma) {
  check_fused_qk_mrope_inputs(q, k, q_weight, k_weight, positions,
                              cos_sin_cache, q_out, k_out);
  TVM_FFI_ICHECK(is_neox) << "MUSA fused MRoPE path requires NeoX style";
  TVM_FFI_ICHECK_EQ(mrope_section_t + mrope_section_h + mrope_section_w,
                    cos_sin_cache.size(1) / 2);
  ffi::MUSADeviceGuard device_guard(q.device().device_id);
  launch_fused_qk_rmsnorm_mrope_bf16(
      q, k, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
      mrope_section_t, mrope_section_h, mrope_section_w, is_interleaved, gemma,
      static_cast<float>(eps));
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA fused_qk_rmsnorm_mrope kernel failed: "
      << musaGetErrorString(err);
}

void sgl_musa_fused_qk_rmsnorm_mrope_cache(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
    ffi::TensorView q_weight, ffi::TensorView k_weight,
    ffi::TensorView positions, ffi::TensorView cos_sin_cache,
    ffi::TensorView q_out, ffi::TensorView k_cache, ffi::TensorView v_cache,
    ffi::TensorView indices, bool is_neox, int mrope_section_t,
    int mrope_section_h, int mrope_section_w, bool is_interleaved, double eps,
    bool gemma) {
  check_fused_qk_mrope_cache_inputs(q, k, v, q_weight, k_weight, positions,
                                    cos_sin_cache, q_out, k_cache, v_cache,
                                    indices);
  TVM_FFI_ICHECK(is_neox)
      << "MUSA fused MRoPE cache path requires NeoX style";
  TVM_FFI_ICHECK_EQ(mrope_section_t + mrope_section_h + mrope_section_w,
                    cos_sin_cache.size(1) / 2);
  ffi::MUSADeviceGuard device_guard(q.device().device_id);
  if (dtype_equal(indices.dtype(), dl_int32)) {
    launch_fused_qk_rmsnorm_mrope_cache_bf16<int32_t>(
        q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_cache,
        v_cache, indices, mrope_section_t, mrope_section_h, mrope_section_w,
        is_interleaved, gemma, static_cast<float>(eps));
  } else if (dtype_equal(indices.dtype(), dl_int64)) {
    launch_fused_qk_rmsnorm_mrope_cache_bf16<int64_t>(
        q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_cache,
        v_cache, indices, mrope_section_t, mrope_section_h, mrope_section_w,
        is_interleaved, gemma, static_cast<float>(eps));
  } else {
    TVM_FFI_THROW(ValueError) << "indices must be int32 or int64";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA fused_qk_rmsnorm_mrope_cache kernel failed: "
      << musaGetErrorString(err);
}

template <typename index_t>
void launch_fused_qk_rmsnorm_mrope_cache_out_bf16(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
    ffi::TensorView q_weight, ffi::TensorView k_weight,
    ffi::TensorView positions, ffi::TensorView cos_sin_cache,
    ffi::TensorView q_out, ffi::TensorView k_out, ffi::TensorView k_cache,
    ffi::TensorView v_cache, ffi::TensorView indices, int mrope_section_t,
    int mrope_section_h, int mrope_section_w, bool is_interleaved, bool gemma,
    float eps) {
  const int batch = static_cast<int>(q.size(0));
  if (batch == 0) {
    return;
  }
  musaStream_t stream = get_stream(q.device());
  const int q_heads = static_cast<int>(q.size(1));
  const int k_heads = static_cast<int>(k.size(1));
  const int hidden = static_cast<int>(q.size(2));
  const int rot_dim = static_cast<int>(cos_sin_cache.size(1));
  const bool use_32q4kv_h128_mrope_specialization =
      q_heads == 32 && k_heads == 4 && hidden == 128 && rot_dim == 128 &&
      mrope_section_t == 24 && mrope_section_h == 20 &&
      mrope_section_w == 20;
  const bool use_h256r64_mrope_specialization =
      gemma && is_interleaved && hidden == 256 && rot_dim == 64 &&
      mrope_section_t == 11 && mrope_section_h == 11 &&
      mrope_section_w == 10;
  TVM_FFI_ICHECK(k_cache.ndim() == 2 || use_h256r64_mrope_specialization)
      << "paged cache-out requires the h256r64 specialization";
  if (use_h256r64_mrope_specialization) {
    if (q_heads == 4 && k_heads == 1) {
      launch_h256r64_mrope_cache_out_kernel<index_t, 4, 1>(
          q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          k_cache, v_cache, indices, batch, eps, stream);
      return;
    }
    if (q_heads == 8 && k_heads == 1) {
      launch_h256r64_mrope_cache_out_kernel<index_t, 8, 1, 16>(
          q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          k_cache, v_cache, indices, batch, eps, stream);
      return;
    }
    if (q_heads == 16 && k_heads == 1) {
      launch_h256r64_mrope_cache_out_kernel<index_t, 16, 1>(
          q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          k_cache, v_cache, indices, batch, eps, stream);
      return;
    }
    if (q_heads == 16 && k_heads == 2) {
      launch_h256r64_mrope_cache_out_kernel<index_t, 16, 2>(
          q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          k_cache, v_cache, indices, batch, eps, stream);
      return;
    }
    if (q_heads == 16 && k_heads == 4) {
      launch_h256r64_mrope_cache_out_kernel<index_t, 16, 4>(
          q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          k_cache, v_cache, indices, batch, eps, stream);
      return;
    }
    if (q_heads == 24 && k_heads == 4) {
      launch_h256r64_mrope_cache_out_kernel<index_t, 24, 4>(
          q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          k_cache, v_cache, indices, batch, eps, stream);
      return;
    }
    if (q_heads == 32 && k_heads == 2) {
      launch_h256r64_mrope_cache_out_kernel<index_t, 32, 2>(
          q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          k_cache, v_cache, indices, batch, eps, stream);
      return;
    }
    if (q_heads == 32 && k_heads == 4) {
      launch_h256r64_mrope_cache_out_kernel<index_t, 32, 4>(
          q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          k_cache, v_cache, indices, batch, eps, stream);
      return;
    }
    if (q_heads == 64 && k_heads == 4) {
      launch_h256r64_mrope_cache_out_kernel<index_t, 64, 4>(
          q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          k_cache, v_cache, indices, batch, eps, stream);
      return;
    }
    if (q_heads == 64 && k_heads == 8) {
      launch_h256r64_mrope_cache_out_kernel<index_t, 64, 8>(
          q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
          k_cache, v_cache, indices, batch, eps, stream);
      return;
    }
  }
  const bool use_h128_full_mrope_specialization =
      is_interleaved && hidden == 128 && rot_dim == 128 && mrope_section_t == 24 &&
      mrope_section_h == 20 && mrope_section_w == 20;
  if (use_h128_full_mrope_specialization) {
#define LAUNCH_H128_MROPE_CACHE_OUT(QH, KH)                                   \
  do {                                                                        \
    if (q_heads == (QH) && k_heads == (KH)) {                                 \
      if (gemma) {                                                            \
        launch_h128_full_mrope_cache_out_kernel<index_t, QH, KH, true>(       \
            q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out,     \
            k_out, k_cache, v_cache, indices, batch, eps, stream);            \
      } else {                                                                \
        launch_h128_full_mrope_cache_out_kernel<index_t, QH, KH, false>(      \
            q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out,     \
            k_out, k_cache, v_cache, indices, batch, eps, stream);            \
      }                                                                       \
      return;                                                                 \
    }                                                                         \
  } while (0)
    LAUNCH_H128_MROPE_CACHE_OUT(2, 1);
    LAUNCH_H128_MROPE_CACHE_OUT(4, 1);
    LAUNCH_H128_MROPE_CACHE_OUT(4, 2);
    LAUNCH_H128_MROPE_CACHE_OUT(8, 2);
    LAUNCH_H128_MROPE_CACHE_OUT(8, 4);
    LAUNCH_H128_MROPE_CACHE_OUT(16, 4);
    LAUNCH_H128_MROPE_CACHE_OUT(16, 8);
    LAUNCH_H128_MROPE_CACHE_OUT(32, 8);
    LAUNCH_H128_MROPE_CACHE_OUT(2, 2);
    LAUNCH_H128_MROPE_CACHE_OUT(4, 4);
    LAUNCH_H128_MROPE_CACHE_OUT(8, 8);
    LAUNCH_H128_MROPE_CACHE_OUT(16, 16);
    LAUNCH_H128_MROPE_CACHE_OUT(32, 32);
#undef LAUNCH_H128_MROPE_CACHE_OUT
  }
  const bool use_h128_full_rope_specialization =
      hidden == 128 && rot_dim == 128 && mrope_section_t == 64 &&
      mrope_section_h == 0 && mrope_section_w == 0 && !is_interleaved;
  if (use_h128_full_rope_specialization) {
#define LAUNCH_H128_ROPE_CACHE_OUT(QH, KH)                                    \
  do {                                                                        \
    if (q_heads == (QH) && k_heads == (KH)) {                                 \
      if (gemma) {                                                            \
        launch_h128_full_rope_cache_out_kernel<index_t, QH, KH, true>(        \
            q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out,     \
            k_out, k_cache, v_cache, indices, batch, eps, stream);            \
      } else {                                                                \
        launch_h128_full_rope_cache_out_kernel<index_t, QH, KH, false>(       \
            q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out,     \
            k_out, k_cache, v_cache, indices, batch, eps, stream);            \
      }                                                                       \
      return;                                                                 \
    }                                                                         \
  } while (0)
    LAUNCH_H128_ROPE_CACHE_OUT(8, 8);
    LAUNCH_H128_ROPE_CACHE_OUT(16, 8);
    LAUNCH_H128_ROPE_CACHE_OUT(24, 8);
    LAUNCH_H128_ROPE_CACHE_OUT(32, 8);
    LAUNCH_H128_ROPE_CACHE_OUT(40, 8);
    LAUNCH_H128_ROPE_CACHE_OUT(48, 8);
    LAUNCH_H128_ROPE_CACHE_OUT(64, 8);
    LAUNCH_H128_ROPE_CACHE_OUT(5, 1);
    LAUNCH_H128_ROPE_CACHE_OUT(8, 1);
    LAUNCH_H128_ROPE_CACHE_OUT(10, 2);
    LAUNCH_H128_ROPE_CACHE_OUT(16, 1);
    LAUNCH_H128_ROPE_CACHE_OUT(16, 2);
    LAUNCH_H128_ROPE_CACHE_OUT(16, 4);
    LAUNCH_H128_ROPE_CACHE_OUT(20, 4);
    LAUNCH_H128_ROPE_CACHE_OUT(24, 4);
    LAUNCH_H128_ROPE_CACHE_OUT(32, 2);
    LAUNCH_H128_ROPE_CACHE_OUT(32, 4);
    LAUNCH_H128_ROPE_CACHE_OUT(40, 4);
    LAUNCH_H128_ROPE_CACHE_OUT(4, 1);
    LAUNCH_H128_ROPE_CACHE_OUT(8, 2);
#undef LAUNCH_H128_ROPE_CACHE_OUT
  }
  const bool use_h128r64_rope_specialization =
      hidden == 128 && rot_dim == 64 && mrope_section_t == 32 &&
      mrope_section_h == 0 && mrope_section_w == 0 && !is_interleaved;
  if (use_h128r64_rope_specialization) {
#define LAUNCH_H128R64_ROPE_CACHE_OUT(QH, KH)                                 \
  do {                                                                        \
    if (q_heads == (QH) && k_heads == (KH)) {                                 \
      if (gemma) {                                                            \
        launch_h128r64_rope_cache_out_kernel<index_t, QH, KH, true>(          \
            q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out,     \
            k_out, k_cache, v_cache, indices, batch, eps, stream);            \
      } else {                                                                \
        launch_h128r64_rope_cache_out_kernel<index_t, QH, KH, false>(         \
            q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out,     \
            k_out, k_cache, v_cache, indices, batch, eps, stream);            \
      }                                                                       \
      return;                                                                 \
    }                                                                         \
  } while (0)
    LAUNCH_H128R64_ROPE_CACHE_OUT(6, 1);
    LAUNCH_H128R64_ROPE_CACHE_OUT(8, 1);
    LAUNCH_H128R64_ROPE_CACHE_OUT(12, 1);
    LAUNCH_H128R64_ROPE_CACHE_OUT(12, 2);
    LAUNCH_H128R64_ROPE_CACHE_OUT(16, 2);
    LAUNCH_H128R64_ROPE_CACHE_OUT(24, 2);
    LAUNCH_H128R64_ROPE_CACHE_OUT(24, 4);
    LAUNCH_H128R64_ROPE_CACHE_OUT(32, 4);
    LAUNCH_H128R64_ROPE_CACHE_OUT(48, 4);
    LAUNCH_H128R64_ROPE_CACHE_OUT(48, 8);
    LAUNCH_H128R64_ROPE_CACHE_OUT(64, 8);
    LAUNCH_H128R64_ROPE_CACHE_OUT(96, 8);
#undef LAUNCH_H128R64_ROPE_CACHE_OUT
  }
  if (use_32q4kv_h128_mrope_specialization) {
    launch_qk_mrope_cache_out_kernel<index_t>(
        q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
        k_cache, v_cache, indices, batch, gemma, is_interleaved, eps, stream);
    return;
  }
  constexpr int threads = 256;
  constexpr int heads_per_block = 8;
  const int groups_per_token =
      (q_heads + k_heads + heads_per_block - 1) / heads_per_block;
  if (gemma) {
    fused_qk_rmsnorm_mrope_generic_bf16_kernel<index_t, true, true>
        <<<batch * groups_per_token, threads, 0, stream>>>(
            static_cast<const __mt_bfloat16 *>(q.data_ptr()),
            static_cast<const __mt_bfloat16 *>(k.data_ptr()),
            static_cast<const __mt_bfloat16 *>(v.data_ptr()),
            static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
            static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
            static_cast<const int64_t *>(positions.data_ptr()),
            static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
            static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
            static_cast<__mt_bfloat16 *>(k_out.data_ptr()),
            static_cast<__mt_bfloat16 *>(k_cache.data_ptr()),
            static_cast<__mt_bfloat16 *>(v_cache.data_ptr()),
            static_cast<const index_t *>(indices.data_ptr()), batch, q_heads,
            k_heads, hidden, rot_dim, static_cast<int64_t>(positions.stride(0)),
            static_cast<int64_t>(q.stride(0)),
            static_cast<int64_t>(q.stride(1)),
            static_cast<int64_t>(k.stride(0)),
            static_cast<int64_t>(k.stride(1)),
            static_cast<int64_t>(v.stride(0)),
            static_cast<int64_t>(v.stride(1)),
            static_cast<int64_t>(q_out.stride(0)),
            static_cast<int64_t>(q_out.stride(1)),
            static_cast<int64_t>(k_out.stride(0)),
            static_cast<int64_t>(k_out.stride(1)),
            static_cast<int64_t>(k_cache.stride(0)),
            static_cast<int64_t>(v_cache.stride(0)),
            static_cast<int64_t>(indices.stride(0)), mrope_section_t,
            mrope_section_h, mrope_section_w, is_interleaved, eps);
    return;
  }
  fused_qk_rmsnorm_mrope_generic_bf16_kernel<index_t, true, false>
      <<<batch * groups_per_token, threads, 0, stream>>>(
          static_cast<const __mt_bfloat16 *>(q.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k.data_ptr()),
          static_cast<const __mt_bfloat16 *>(v.data_ptr()),
          static_cast<const __mt_bfloat16 *>(q_weight.data_ptr()),
          static_cast<const __mt_bfloat16 *>(k_weight.data_ptr()),
          static_cast<const int64_t *>(positions.data_ptr()),
          static_cast<const __mt_bfloat16 *>(cos_sin_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(q_out.data_ptr()),
          static_cast<__mt_bfloat16 *>(k_out.data_ptr()),
          static_cast<__mt_bfloat16 *>(k_cache.data_ptr()),
          static_cast<__mt_bfloat16 *>(v_cache.data_ptr()),
          static_cast<const index_t *>(indices.data_ptr()), batch, q_heads,
          k_heads, hidden, rot_dim, static_cast<int64_t>(positions.stride(0)),
          static_cast<int64_t>(q.stride(0)), static_cast<int64_t>(q.stride(1)),
          static_cast<int64_t>(k.stride(0)), static_cast<int64_t>(k.stride(1)),
          static_cast<int64_t>(v.stride(0)), static_cast<int64_t>(v.stride(1)),
          static_cast<int64_t>(q_out.stride(0)),
          static_cast<int64_t>(q_out.stride(1)),
          static_cast<int64_t>(k_out.stride(0)),
          static_cast<int64_t>(k_out.stride(1)),
          static_cast<int64_t>(k_cache.stride(0)),
          static_cast<int64_t>(v_cache.stride(0)),
          static_cast<int64_t>(indices.stride(0)), mrope_section_t,
          mrope_section_h, mrope_section_w, is_interleaved, eps);
}

void sgl_musa_fused_qk_rmsnorm_mrope_cache_out(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
    ffi::TensorView q_weight, ffi::TensorView k_weight,
    ffi::TensorView positions, ffi::TensorView cos_sin_cache,
    ffi::TensorView q_out, ffi::TensorView k_out, ffi::TensorView k_cache,
    ffi::TensorView v_cache, ffi::TensorView indices, bool is_neox,
    int mrope_section_t, int mrope_section_h, int mrope_section_w,
    bool is_interleaved, double eps, bool gemma) {
  check_fused_qk_mrope_inputs(q, k, q_weight, k_weight, positions,
                              cos_sin_cache, q_out, k_out);
  check_fused_qk_mrope_cache_inputs(q, k, v, q_weight, k_weight, positions,
                                    cos_sin_cache, q_out, k_cache, v_cache,
                                    indices);
  TVM_FFI_ICHECK(is_neox)
      << "MUSA fused MRoPE cache-out path requires NeoX style";
  TVM_FFI_ICHECK_EQ(mrope_section_t + mrope_section_h + mrope_section_w,
                    cos_sin_cache.size(1) / 2);
  ffi::MUSADeviceGuard device_guard(q.device().device_id);
  if (dtype_equal(indices.dtype(), dl_int32)) {
    launch_fused_qk_rmsnorm_mrope_cache_out_bf16<int32_t>(
        q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
        k_cache, v_cache, indices, mrope_section_t, mrope_section_h,
        mrope_section_w, is_interleaved, gemma, static_cast<float>(eps));
  } else if (dtype_equal(indices.dtype(), dl_int64)) {
    launch_fused_qk_rmsnorm_mrope_cache_out_bf16<int64_t>(
        q, k, v, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out,
        k_cache, v_cache, indices, mrope_section_t, mrope_section_h,
        mrope_section_w, is_interleaved, gemma, static_cast<float>(eps));
  } else {
    TVM_FFI_THROW(ValueError) << "indices must be int32 or int64";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA fused_qk_rmsnorm_mrope_cache_out kernel failed: "
      << musaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_fused_qk_rmsnorm_mrope,
                              sgl_musa_fused_qk_rmsnorm_mrope);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_fused_qk_rmsnorm_mrope_cache,
                              sgl_musa_fused_qk_rmsnorm_mrope_cache);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_fused_qk_rmsnorm_mrope_cache_out,
                              sgl_musa_fused_qk_rmsnorm_mrope_cache_out);
