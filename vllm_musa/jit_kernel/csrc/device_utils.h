#pragma once

#include <cstdint>
#include <type_traits>

#include <musa_bf16.h>
#include <musa_fp16.h>

template <typename T> __device__ __forceinline__ float to_float(T value) {
  if constexpr (std::is_same_v<T, half>) {
    return __half2float(value);
  } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
    return __bfloat162float(value);
  } else {
    return static_cast<float>(value);
  }
}

template <typename T> __device__ __forceinline__ T from_float(float value) {
  if constexpr (std::is_same_v<T, half>) {
    return __float2half_rn(value);
  } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
    return __float2bfloat16_rn(value);
  } else {
    return static_cast<T>(value);
  }
}

template <int THREADS_PER_SUBWARP>
__device__ __forceinline__ float GroupReduceMax(float val, const int tid) {
  constexpr unsigned subwarp_mask_bits = []() constexpr {
    if constexpr (THREADS_PER_SUBWARP == 32) {
      return 0xffffffffu;
    } else {
      return (1u << THREADS_PER_SUBWARP) - 1u;
    }
  }();
  const unsigned mask = subwarp_mask_bits << (threadIdx.x - tid);

  if constexpr (THREADS_PER_SUBWARP >= 32) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 16));
  }
  if constexpr (THREADS_PER_SUBWARP >= 16) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  }
  if constexpr (THREADS_PER_SUBWARP >= 8) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  }
  if constexpr (THREADS_PER_SUBWARP >= 4) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  }
  if constexpr (THREADS_PER_SUBWARP >= 2) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
  }
  return val;
}

__device__ __forceinline__ float silu(const float &val) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 1000)
  float half_val = 0.5f * val;
  float t = __tanhf(half_val);
  return half_val * (1.0f + t);
#else
  return val / (1.0f + __expf(-val));
#endif
}

__device__ __forceinline__ float2 fmul2_rn(float2 a, float2 b) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 1000)
  return __fmul2_rn(a, b);
#else
  float2 result;
  result.x = a.x * b.x;
  result.y = a.y * b.y;
  return result;
#endif
}

__device__ __forceinline__ void st_global(const int4 *ptr, const int4 &value) {
  int4 *p = const_cast<int4 *>(ptr);
  *p = value;
}

__device__ __forceinline__ int4 ld_global_nc(const int4 *ptr) { return *ptr; }

__device__ __forceinline__ void st_global_b64_slc_new(uint64_t *ptr,
                                                      const uint64_t value) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 310)
  asm volatile(
      "LSU.ST.B64 %1, %0, _, 8, 1, 1, inner_persist=4, outer_persist=2, "
      "chrnt=l2_l3, slc=new, persist=0, stride_add_first=0"
      :
      : "R"(ptr), "R"(value));
#else
  *ptr = value;
#endif
}
