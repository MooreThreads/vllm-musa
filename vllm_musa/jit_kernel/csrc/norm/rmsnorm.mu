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

template <typename T>
struct __align__(16) Vec8Storage {
  T elem[8];
};

struct __align__(32) Float8Storage {
  float elem[8];
};

template <typename T>
struct __align__(16) Vec8 {
  union {
    Vec8Storage<T> storage;
    T elem[8];
  } val;

  __device__ __forceinline__ Vec8() {}

  template <typename Offset>
  static __device__ __forceinline__ Vec8 load(const T* ptr, Offset idx) {
    return *(const Vec8*)(ptr + idx);
  }

  template <typename Offset>
  static __device__ __forceinline__ Vec8 load_byp_slc(const T* ptr, Offset idx) {
#if ((defined __MUSA_ARCH__) && (__MUSA_ARCH__ == 310))
    Vec8 dst;
    const T* addr = ptr + idx;
    asm volatile(
        "LSU.LD.B128 %0, %1, _, 16, 1, 1, inner_persist=0, outer_persist=2, "
        "chrnt=l2_l3, slc=byp, persist=0, stride_add_first=0"
        : "=R"(dst)
        : "R"(addr));
    return dst;
#else
    return *(const Vec8*)(ptr + idx);
#endif
  }
};

struct __align__(32) Float8 {
  union {
    Float8Storage storage;
    float elem[8];
  } val;

  __device__ __forceinline__ Float8() {}
};

__device__ __forceinline__ float fast_rsqrt(float value) {
#if ((defined __MUSA_ARCH__) && (__MUSA_ARCH__ == 310))
  const float half_value = 0.5f * value;
  float y = __frsqrt_rn(value);
  y = y * (1.5f - half_value * y * y);
  return y;
#else
  return rsqrtf(value);
#endif
}

__device__ __forceinline__ float block_sum(float value, float* warp_sums) {
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int num_warps = ((int)blockDim.x + 31) >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value = tid < num_warps ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  __syncthreads_lm();
  return warp_sums[0];
}

__device__ __forceinline__ float block_sum_8warps(float value, float* warp_sums) {
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value = lane < 8 ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  __syncthreads_lm();
  return warp_sums[0];
}

__device__ __forceinline__ float block_sum_4warps(float value, float* warp_sums) {
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value = lane < 4 ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  __syncthreads_lm();
  return warp_sums[0];
}

template <typename T, bool GEMMA, bool CACHE>
__launch_bounds__(1024, 1)
__global__ void rmsnorm_vec8_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ out,
    int rows,
    int hidden,
    int64_t input_row_stride,
    int64_t out_row_stride,
    float inv_hidden,
    float eps) {
  constexpr int kVec = 8;
  extern __shared__ __align__(16) float smem[];
  float* cached = smem;
  float* warp_sums = smem + (CACHE ? hidden : 0);

  const int row = (int)blockIdx.x;
  const int tid = (int)threadIdx.x;
  const int vec_count = hidden / kVec;
  const int64_t input_base = (int64_t)row * input_row_stride;
  const int64_t out_base = (int64_t)row * out_row_stride;
  float sum = 0.0f;

  for (int vec_idx = tid; vec_idx < vec_count; vec_idx += (int)blockDim.x) {
    const int col = vec_idx * kVec;
    Vec8<T> x = Vec8<T>::load(input + input_base, col);
    Float8 x_float;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float value = to_float<T>(x.val.elem[i]);
      sum += value * value;
      x_float.val.elem[i] = value;
    }
    if constexpr (CACHE) {
      *(Float8*)(cached + col) = x_float;
    }
  }

  sum = block_sum(sum, warp_sums);

  const float scale = fast_rsqrt(sum * inv_hidden + eps);
  for (int vec_idx = tid; vec_idx < vec_count; vec_idx += (int)blockDim.x) {
    const int col = vec_idx * kVec;
    Float8 x_float;
    if constexpr (CACHE) {
      x_float = *(Float8*)(cached + col);
    } else {
      Vec8<T> x = Vec8<T>::load(input + input_base, col);
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        x_float.val.elem[i] = to_float<T>(x.val.elem[i]);
      }
    }
    Vec8<T> w = Vec8<T>::load(weight, col);
    Vec8<T> dst;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float weight_value = to_float<T>(w.val.elem[i]) + (GEMMA ? 1.0f : 0.0f);
      dst.val.elem[i] = from_float<T>(x_float.val.elem[i] * scale * weight_value);
    }
    *(Vec8<T>*)(out + out_base + col) = dst;
  }
}

template <typename T, bool GEMMA, int H, int WARPS>
__launch_bounds__(256, 1)
__global__ void rmsnorm_small_h_vec8_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ out,
    int rows,
    int hidden,
    int64_t input_row_stride,
    int64_t out_row_stride,
    float inv_hidden,
    float eps) {
  constexpr int kVec = 8;
  constexpr int vec_count = H / kVec;
  extern __shared__ __align__(16) float smem[];
  float* cached = smem;
  float* warp_sums = smem + H;

  const int row = (int)blockIdx.x;
  const int tid = (int)threadIdx.x;
  const int64_t input_base = (int64_t)row * input_row_stride;
  const int64_t out_base = (int64_t)row * out_row_stride;
  float sum = 0.0f;

  for (int vec_idx = tid; vec_idx < vec_count; vec_idx += (int)blockDim.x) {
    const int col = vec_idx * kVec;
    Vec8<T> x = Vec8<T>::load(input + input_base, col);
    Float8 x_float;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float value = to_float<T>(x.val.elem[i]);
      sum += value * value;
      x_float.val.elem[i] = value;
    }
    *(Float8*)(cached + col) = x_float;
  }

  if constexpr (WARPS == 4) {
    sum = block_sum_4warps(sum, warp_sums);
  } else {
    sum = block_sum_8warps(sum, warp_sums);
  }

  const float scale = fast_rsqrt(sum * inv_hidden + eps);
  for (int vec_idx = tid; vec_idx < vec_count; vec_idx += (int)blockDim.x) {
    const int col = vec_idx * kVec;
    Float8 x_float = *(Float8*)(cached + col);
    Vec8<T> w = Vec8<T>::load(weight, col);
    Vec8<T> dst;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float weight_value = to_float<T>(w.val.elem[i]) + (GEMMA ? 1.0f : 0.0f);
      dst.val.elem[i] = from_float<T>(x_float.val.elem[i] * scale * weight_value);
    }
    *(Vec8<T>*)(out + out_base + col) = dst;
  }
}

template <typename T, bool GEMMA, int H, int WARPS>
__launch_bounds__(256, 1)
__global__ void rmsnorm_small_h_one_vec_register_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ out,
    int rows,
    int hidden,
    int64_t input_row_stride,
    int64_t out_row_stride,
    float inv_hidden,
    float eps) {
  constexpr int kVec = 8;
  extern __shared__ float warp_sums[];

  const int row = (int)blockIdx.x;
  const int tid = (int)threadIdx.x;
  const int col = tid * kVec;
  const int64_t input_base = (int64_t)row * input_row_stride;
  const int64_t out_base = (int64_t)row * out_row_stride;
  float sum = 0.0f;
  Float8 x_float;

  Vec8<T> x = Vec8<T>::load(input + input_base, col);
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    const float value = to_float<T>(x.val.elem[i]);
    sum += value * value;
    x_float.val.elem[i] = value;
  }

  if constexpr (WARPS == 4) {
    sum = block_sum_4warps(sum, warp_sums);
  } else {
    sum = block_sum_8warps(sum, warp_sums);
  }

  const float scale = fast_rsqrt(sum * inv_hidden + eps);
  Vec8<T> w = Vec8<T>::load(weight, col);
  Vec8<T> dst;
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    const float weight_value = to_float<T>(w.val.elem[i]) + (GEMMA ? 1.0f : 0.0f);
    dst.val.elem[i] = from_float<T>(x_float.val.elem[i] * scale * weight_value);
  }
  *(Vec8<T>*)(out + out_base + col) = dst;
}

template <typename T, bool GEMMA, bool CACHE>
__launch_bounds__(1024, 1)
__global__ void fused_add_rmsnorm_vec8_kernel(
    T* __restrict__ input,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    int rows,
    int hidden,
    int64_t input_row_stride,
    int64_t residual_row_stride,
    float inv_hidden,
    float eps) {
  constexpr int kVec = 8;
  extern __shared__ __align__(16) float smem[];
  float* cached = smem;
  float* warp_sums = smem + (CACHE ? hidden : 0);

  const int row = (int)blockIdx.x;
  const int tid = (int)threadIdx.x;
  const int vec_count = hidden / kVec;
  const int64_t input_base = (int64_t)row * input_row_stride;
  const int64_t residual_base = (int64_t)row * residual_row_stride;
  float sum = 0.0f;

  for (int vec_idx = tid; vec_idx < vec_count; vec_idx += (int)blockDim.x) {
    const int col = vec_idx * kVec;
    Vec8<T> x = Vec8<T>::load_byp_slc(input + input_base, col);
    Vec8<T> r = Vec8<T>::load_byp_slc(residual + residual_base, col);
    Vec8<T> residual_out;
    Float8 sum_float;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float value = to_float<T>(x.val.elem[i]) + to_float<T>(r.val.elem[i]);
      sum += value * value;
      residual_out.val.elem[i] = from_float<T>(value);
      sum_float.val.elem[i] = value;
    }
    *(Vec8<T>*)(residual + residual_base + col) = residual_out;
    if constexpr (CACHE) {
      *(Float8*)(cached + col) = sum_float;
    }
  }

  sum = block_sum(sum, warp_sums);

  const float scale = fast_rsqrt(sum * inv_hidden + eps);
  for (int vec_idx = tid; vec_idx < vec_count; vec_idx += (int)blockDim.x) {
    const int col = vec_idx * kVec;
    Float8 sum_float;
    if constexpr (CACHE) {
      sum_float = *(Float8*)(cached + col);
    } else {
      Vec8<T> r = Vec8<T>::load(residual + residual_base, col);
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        sum_float.val.elem[i] = to_float<T>(r.val.elem[i]);
      }
    }
    Vec8<T> w = Vec8<T>::load(weight, col);
    Vec8<T> dst;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float weight_value = to_float<T>(w.val.elem[i]) + (GEMMA ? 1.0f : 0.0f);
      dst.val.elem[i] = from_float<T>(sum_float.val.elem[i] * scale * weight_value);
    }
    *(Vec8<T>*)(input + input_base + col) = dst;
  }
}

template <typename T, bool GEMMA>
__global__ void rmsnorm_scalar_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ out,
    int rows,
    int hidden,
    int64_t input_row_stride,
    int64_t out_row_stride,
    float inv_hidden,
    float eps) {
  extern __shared__ float warp_sums[];
  const int row = (int)blockIdx.x;
  const int tid = (int)threadIdx.x;
  const int64_t input_base = (int64_t)row * input_row_stride;
  const int64_t out_base = (int64_t)row * out_row_stride;
  float sum = 0.0f;

  for (int col = tid; col < hidden; col += (int)blockDim.x) {
    const float value = to_float<T>(input[input_base + col]);
    sum += value * value;
  }
  sum = block_sum(sum, warp_sums);

  const float scale = fast_rsqrt(sum * inv_hidden + eps);
  for (int col = tid; col < hidden; col += (int)blockDim.x) {
    const float weight_value = to_float<T>(weight[col]) + (GEMMA ? 1.0f : 0.0f);
    out[out_base + col] = from_float<T>(to_float<T>(input[input_base + col]) * scale * weight_value);
  }
}

template <typename T, bool GEMMA>
__global__ void fused_add_rmsnorm_scalar_kernel(
    T* __restrict__ input,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    int rows,
    int hidden,
    int64_t input_row_stride,
    int64_t residual_row_stride,
    float inv_hidden,
    float eps) {
  extern __shared__ float warp_sums[];
  const int row = (int)blockIdx.x;
  const int tid = (int)threadIdx.x;
  const int64_t input_base = (int64_t)row * input_row_stride;
  const int64_t residual_base = (int64_t)row * residual_row_stride;
  float sum = 0.0f;

  for (int col = tid; col < hidden; col += (int)blockDim.x) {
    const float value = to_float<T>(input[input_base + col]) + to_float<T>(residual[residual_base + col]);
    residual[residual_base + col] = from_float<T>(value);
    sum += value * value;
  }
  sum = block_sum(sum, warp_sums);

  const float scale = fast_rsqrt(sum * inv_hidden + eps);
  for (int col = tid; col < hidden; col += (int)blockDim.x) {
    const float weight_value = to_float<T>(weight[col]) + (GEMMA ? 1.0f : 0.0f);
    input[input_base + col] = from_float<T>(to_float<T>(residual[residual_base + col]) * scale * weight_value);
  }
}

inline int vec8_block_threads(int hidden) {
  const int vec_count = hidden / 8;
  const int rounded = ((vec_count + 31) / 32) * 32;
  return rounded < 1024 ? rounded : 1024;
}

inline int rmsnorm_block_threads(int rows, int hidden) {
  if (hidden <= 512) {
    return 64;
  }
  if (hidden <= 4096) {
    if (rows <= 16) {
      const int threads = vec8_block_threads(hidden);
      return threads < 512 ? threads : 512;
    }
    if (rows <= 256) {
      const int threads = vec8_block_threads(hidden);
      return threads < 256 ? threads : 256;
    }
    return 128;
  }
  if (hidden <= 8192) {
    const int threads = vec8_block_threads(hidden);
    return threads < 512 ? threads : 512;
  }
  const int threads = vec8_block_threads(hidden);
  return threads < 896 ? threads : 896;
}

inline int fused_block_threads(int hidden) {
  const int threads = vec8_block_threads(hidden);
  return threads;
}

inline int cached_vec8_shared_bytes(int hidden, int block_threads, int cache_hidden_limit) {
  const int reduce_floats = (block_threads + 31) / 32;
  const int cached_floats = hidden <= cache_hidden_limit ? hidden : 0;
  return (cached_floats + reduce_floats) * static_cast<int>(sizeof(float));
}

void check_rmsnorm_inputs(ffi::TensorView input, ffi::TensorView weight, ffi::TensorView out) {
  CHECK_MUSA(input);
  CHECK_MUSA(weight);
  CHECK_MUSA(out);
  TVM_FFI_ICHECK_EQ(input.ndim(), 2);
  TVM_FFI_ICHECK_EQ(out.ndim(), 2);
  TVM_FFI_ICHECK_EQ(input.stride(1), 1);
  TVM_FFI_ICHECK_EQ(out.stride(1), 1);
  TVM_FFI_ICHECK_EQ(weight.ndim(), 1);
  TVM_FFI_ICHECK_EQ(weight.stride(0), 1);
  TVM_FFI_ICHECK_EQ(input.size(0), out.size(0));
  TVM_FFI_ICHECK_EQ(input.size(1), out.size(1));
  TVM_FFI_ICHECK_EQ(input.size(1), weight.size(0));
  TVM_FFI_ICHECK_GE(input.stride(0), input.size(1));
  TVM_FFI_ICHECK_GE(out.stride(0), out.size(1));
  TVM_FFI_ICHECK_EQ(input.device().device_id, weight.device().device_id);
  TVM_FFI_ICHECK_EQ(input.device().device_id, out.device().device_id);
  TVM_FFI_ICHECK(dtype_equal(input.dtype(), out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(input.dtype(), weight.dtype()));
}

void check_fused_inputs(ffi::TensorView input, ffi::TensorView residual, ffi::TensorView weight) {
  CHECK_MUSA(input);
  CHECK_MUSA(residual);
  CHECK_MUSA(weight);
  TVM_FFI_ICHECK_EQ(input.ndim(), 2);
  TVM_FFI_ICHECK_EQ(residual.ndim(), 2);
  TVM_FFI_ICHECK_EQ(input.stride(1), 1);
  TVM_FFI_ICHECK_EQ(residual.stride(1), 1);
  TVM_FFI_ICHECK_EQ(weight.ndim(), 1);
  TVM_FFI_ICHECK_EQ(weight.stride(0), 1);
  TVM_FFI_ICHECK_EQ(input.size(0), residual.size(0));
  TVM_FFI_ICHECK_EQ(input.size(1), residual.size(1));
  TVM_FFI_ICHECK_EQ(input.size(1), weight.size(0));
  TVM_FFI_ICHECK_GE(input.stride(0), input.size(1));
  TVM_FFI_ICHECK_GE(residual.stride(0), residual.size(1));
  TVM_FFI_ICHECK_EQ(input.device().device_id, residual.device().device_id);
  TVM_FFI_ICHECK_EQ(input.device().device_id, weight.device().device_id);
  TVM_FFI_ICHECK(dtype_equal(input.dtype(), residual.dtype()));
  TVM_FFI_ICHECK(dtype_equal(input.dtype(), weight.dtype()));
}

template <typename T, bool GEMMA>
void launch_rmsnorm(ffi::TensorView input, ffi::TensorView weight, ffi::TensorView out, float eps) {
  const int rows = static_cast<int>(input.size(0));
  const int hidden = static_cast<int>(input.size(1));
  const int64_t input_row_stride = static_cast<int64_t>(input.stride(0));
  const int64_t out_row_stride = static_cast<int64_t>(out.stride(0));
  const float inv_hidden = 1.0f / static_cast<float>(hidden);
  musaStream_t stream = get_stream(input.device());

  if ((hidden % 8) == 0 && hidden <= 32768) {
    if (rows <= 16 && hidden == 1024) {
      constexpr int threads = 128;
      constexpr int smem_bytes = 4 * static_cast<int>(sizeof(float));
      rmsnorm_small_h_one_vec_register_kernel<T, GEMMA, 1024, 4><<<rows, threads, smem_bytes, stream>>>(
          static_cast<const T*>(input.data_ptr()),
          static_cast<const T*>(weight.data_ptr()),
          static_cast<T*>(out.data_ptr()),
          rows,
          hidden,
          input_row_stride,
          out_row_stride,
          inv_hidden,
          eps);
      return;
    }
    if (rows <= 16 && hidden == 2048) {
      constexpr int threads = 256;
      constexpr int smem_bytes = 8 * static_cast<int>(sizeof(float));
      rmsnorm_small_h_one_vec_register_kernel<T, GEMMA, 2048, 8><<<rows, threads, smem_bytes, stream>>>(
          static_cast<const T*>(input.data_ptr()),
          static_cast<const T*>(weight.data_ptr()),
          static_cast<T*>(out.data_ptr()),
          rows,
          hidden,
          input_row_stride,
          out_row_stride,
          inv_hidden,
          eps);
      return;
    }
    const int threads = rmsnorm_block_threads(rows, hidden);
    if (hidden <= 8192) {
      rmsnorm_vec8_kernel<T, GEMMA, true><<<rows, threads, cached_vec8_shared_bytes(hidden, threads, 8192), stream>>>(
          static_cast<const T*>(input.data_ptr()),
          static_cast<const T*>(weight.data_ptr()),
          static_cast<T*>(out.data_ptr()),
          rows,
          hidden,
          input_row_stride,
          out_row_stride,
          inv_hidden,
          eps);
    } else {
      rmsnorm_vec8_kernel<T, GEMMA, false><<<rows, threads, cached_vec8_shared_bytes(hidden, threads, 8192), stream>>>(
          static_cast<const T*>(input.data_ptr()),
          static_cast<const T*>(weight.data_ptr()),
          static_cast<T*>(out.data_ptr()),
          rows,
          hidden,
          input_row_stride,
          out_row_stride,
          inv_hidden,
          eps);
    }
  } else {
    constexpr int threads = 256;
    constexpr int smem_bytes = ((threads + 31) / 32) * static_cast<int>(sizeof(float));
    rmsnorm_scalar_kernel<T, GEMMA><<<rows, threads, smem_bytes, stream>>>(
        static_cast<const T*>(input.data_ptr()),
        static_cast<const T*>(weight.data_ptr()),
        static_cast<T*>(out.data_ptr()),
        rows,
        hidden,
        input_row_stride,
        out_row_stride,
        inv_hidden,
        eps);
  }
}

template <typename T, bool GEMMA>
void launch_fused_add_rmsnorm(ffi::TensorView input, ffi::TensorView residual, ffi::TensorView weight, float eps) {
  const int rows = static_cast<int>(input.size(0));
  const int hidden = static_cast<int>(input.size(1));
  const int64_t input_row_stride = static_cast<int64_t>(input.stride(0));
  const int64_t residual_row_stride = static_cast<int64_t>(residual.stride(0));
  const float inv_hidden = 1.0f / static_cast<float>(hidden);
  musaStream_t stream = get_stream(input.device());

  if ((hidden % 8) == 0 && hidden <= 32768) {
    const int threads = fused_block_threads(hidden);
    if (hidden <= 32768) {
      fused_add_rmsnorm_vec8_kernel<T, GEMMA, true><<<rows, threads, cached_vec8_shared_bytes(hidden, threads, 32768), stream>>>(
          static_cast<T*>(input.data_ptr()),
          static_cast<T*>(residual.data_ptr()),
          static_cast<const T*>(weight.data_ptr()),
          rows,
          hidden,
          input_row_stride,
          residual_row_stride,
          inv_hidden,
          eps);
    } else {
      fused_add_rmsnorm_vec8_kernel<T, GEMMA, false><<<rows, threads, cached_vec8_shared_bytes(hidden, threads, 32768), stream>>>(
          static_cast<T*>(input.data_ptr()),
          static_cast<T*>(residual.data_ptr()),
          static_cast<const T*>(weight.data_ptr()),
          rows,
          hidden,
          input_row_stride,
          residual_row_stride,
          inv_hidden,
          eps);
    }
  } else {
    constexpr int threads = 256;
    constexpr int smem_bytes = ((threads + 31) / 32) * static_cast<int>(sizeof(float));
    fused_add_rmsnorm_scalar_kernel<T, GEMMA><<<rows, threads, smem_bytes, stream>>>(
        static_cast<T*>(input.data_ptr()),
        static_cast<T*>(residual.data_ptr()),
        static_cast<const T*>(weight.data_ptr()),
        rows,
        hidden,
        input_row_stride,
        residual_row_stride,
        inv_hidden,
        eps);
  }
}

void sgl_musa_rmsnorm(ffi::TensorView input, ffi::TensorView weight, ffi::TensorView out, double eps, bool gemma) {
  check_rmsnorm_inputs(input, weight, out);
  ffi::MUSADeviceGuard device_guard(input.device().device_id);
  if (dtype_equal(input.dtype(), dl_float16)) {
    if (gemma) {
      launch_rmsnorm<half, true>(input, weight, out, static_cast<float>(eps));
    } else {
      launch_rmsnorm<half, false>(input, weight, out, static_cast<float>(eps));
    }
  } else if (dtype_equal(input.dtype(), dl_bfloat16)) {
    if (gemma) {
      launch_rmsnorm<__mt_bfloat16, true>(input, weight, out, static_cast<float>(eps));
    } else {
      launch_rmsnorm<__mt_bfloat16, false>(input, weight, out, static_cast<float>(eps));
    }
  } else {
    TVM_FFI_THROW(ValueError) << "sgl_musa_rmsnorm only supports fp16/bf16";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess) << "MUSA rmsnorm kernel failed: " << musaGetErrorString(err);
}

void sgl_musa_fused_add_rmsnorm(
    ffi::TensorView input, ffi::TensorView residual, ffi::TensorView weight, double eps, bool gemma) {
  check_fused_inputs(input, residual, weight);
  ffi::MUSADeviceGuard device_guard(input.device().device_id);
  if (dtype_equal(input.dtype(), dl_float16)) {
    if (gemma) {
      launch_fused_add_rmsnorm<half, true>(input, residual, weight, static_cast<float>(eps));
    } else {
      launch_fused_add_rmsnorm<half, false>(input, residual, weight, static_cast<float>(eps));
    }
  } else if (dtype_equal(input.dtype(), dl_bfloat16)) {
    if (gemma) {
      launch_fused_add_rmsnorm<__mt_bfloat16, true>(input, residual, weight, static_cast<float>(eps));
    } else {
      launch_fused_add_rmsnorm<__mt_bfloat16, false>(input, residual, weight, static_cast<float>(eps));
    }
  } else {
    TVM_FFI_THROW(ValueError) << "sgl_musa_fused_add_rmsnorm only supports fp16/bf16";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess) << "MUSA fused_add_rmsnorm kernel failed: " << musaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_rmsnorm, sgl_musa_rmsnorm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_fused_add_rmsnorm, sgl_musa_fused_add_rmsnorm);
