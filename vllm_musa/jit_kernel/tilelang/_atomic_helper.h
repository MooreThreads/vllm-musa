#pragma once

#include <stdint.h>

#include <tl_templates/musa/cvt.h>

__device__ __forceinline__ int sgl_tl_atomic_add_offset(int *base, int offset,
                                                        int val) {
  return atomicAdd(base + offset, val);
}

__device__ __forceinline__ void sgl_tl_store_fp8e4m3x4(fp8_e4_t *base,
                                                       int64_t offset, float x0,
                                                       float x1, float x2,
                                                       float x3) {
  const float4 values = {x0, x1, x2, x3};
  const fp8_e4_4_t packed = tl::cvt_float_to_fp8e4m3_x4(values);
  *reinterpret_cast<unsigned int *>(base + offset) =
      *reinterpret_cast<const unsigned int *>(&packed);
}

__device__ __forceinline__ void sgl_tl_copy_bf16x8(bfloat16_t *dst,
                                                   const bfloat16_t *src,
                                                   int64_t dst_offset,
                                                   int64_t src_offset) {
  const int4 value = *reinterpret_cast<const int4 *>(src + src_offset);
  *reinterpret_cast<int4 *>(dst + dst_offset) = value;
}

__device__ __forceinline__ void sgl_tl_copy_fp8x16(fp8_e4_t *dst,
                                                   const fp8_e4_t *src,
                                                   int64_t dst_offset,
                                                   int64_t src_offset) {
  const int4 value = *reinterpret_cast<const int4 *>(src + src_offset);
  *reinterpret_cast<int4 *>(dst + dst_offset) = value;
}
