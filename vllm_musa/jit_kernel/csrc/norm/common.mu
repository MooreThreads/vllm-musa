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

__device__ __forceinline__ int mrope_24_20_20_interleaved_axis(int rot_offset) {
  constexpr unsigned long long axis1_mask = 0x492492492492492ULL;
  constexpr unsigned long long axis2_mask = 0x924924924924924ULL;
  const unsigned long long bit = 1ULL << rot_offset;
  return ((axis1_mask & bit) != 0ULL) + (((axis2_mask & bit) != 0ULL) << 1);
}

__device__ __forceinline__ int mrope_11_11_10_interleaved_axis(int rot_offset) {
  constexpr unsigned int axis1_mask = 0x92492492U;
  constexpr unsigned int axis2_mask = 0x24924924U;
  const unsigned int bit = 1U << rot_offset;
  return ((axis1_mask & bit) != 0U) + (((axis2_mask & bit) != 0U) << 1);
}

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
