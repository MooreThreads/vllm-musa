// Small-M GEMVs for FP8 weights with 32x32 block scales.
//
// The activation is quantized once per call (per_token_group_quant_fp8 with 32-wide
// groups), then each weight byte is read once for all activation rows. A lane reads
// 16 weight bytes per step; the 16 bytes lie in one scale group, so each partial dot
// is scaled once. CTAs start their K sweep at different steps: a weight row stride
// that is a multiple of the DRAM channel interleave would otherwise send every row
// of a step to the same channels.
#include <algorithm>
#include <cstdint>

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

namespace ffi = tvm::ffi;

namespace {

constexpr int kWarpSize = 32;
constexpr int kVec = 16;
constexpr int kStep = kWarpSize * kVec;
constexpr int kGroup = 32;
constexpr float kFp8Max = 448.0f;
constexpr float kQuantEps = 1e-10f;
constexpr int kMaxSmemBytes = 160 * 1024;
constexpr int kMaxMapSlots = 512;
constexpr int kMapWarps = kMaxMapSlots / kWarpSize;
constexpr int kMapRows = 16;
constexpr int kMapWidth = 2 + kMapRows;

typedef _Float16 half4_t __attribute__((ext_vector_type(4)));
using fp8x4_t = unsigned char __attribute__((vector_size(4)));

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = kWarpSize / 2; o > 0; o >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, o);
  }
  return v;
}

__device__ __forceinline__ void fp8x16_to_float(const uint4 raw, float (&dst)[kVec]) {
  const fp8x4_t* q = reinterpret_cast<const fp8x4_t*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const half4_t h = __musa_e4m32f16_rn_bst4(q[i]);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      dst[i * 4 + j] = static_cast<float>(h[j]);
    }
  }
}

__device__ __forceinline__ float4 fp8x4_to_float4(uint32_t raw, float scale) {
  fp8x4_t q4;
  __builtin_memcpy(&q4, &raw, 4);
  const half4_t h = __musa_e4m32f16_rn_bst4(q4);
  return make_float4(static_cast<float>(h[0]) * scale, static_cast<float>(h[1]) * scale,
                     static_cast<float>(h[2]) * scale, static_cast<float>(h[3]) * scale);
}

// IEEE division and exp: per_token_group_quant_fp8 divides by the group scale, and
// silu_and_mul_with_clamp rounds an exact SiLU.
#pragma float_control(precise, on, push)
__device__ __forceinline__ float div_rn(float x, float y) { return x / y; }
__device__ __forceinline__ float exp_rn(float x) { return expf(x); }
#pragma float_control(pop)

// per_token_group_quant_fp8 of a [M, K] BF16 activation, or of the clamped SwiGLU of a
// [M, 2K] [gate | up] activation. One lane per 16 elements, so a group spans two lanes.
template <bool SWIGLU>
__global__ void __launch_bounds__(128)
    act_quant_kernel(const uint16_t* __restrict__ x, int64_t x_stride, uint8_t* __restrict__ xq,
                     int64_t xq_stride, float* __restrict__ xs, int64_t xs_stride, int rows, int K,
                     int chunks, float swiglu_limit) {
  const int gw = blockIdx.x * 4 + threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int m = gw / chunks;
  const int k = (gw - m * chunks) * kStep + lane * kVec;
  const bool active = m < rows && k < K;
  float v[kVec];
  if (active) {
    const uint4* src = reinterpret_cast<const uint4*>(x + m * x_stride + k);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint4 raw = src[i];
      const uint32_t w4[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        v[i * 8 + j * 2] = __uint_as_float(w4[j] << 16);
        v[i * 8 + j * 2 + 1] = __uint_as_float(w4[j] & 0xffff0000u);
      }
    }
    if constexpr (SWIGLU) {
      const uint4* up_src = reinterpret_cast<const uint4*>(x + m * x_stride + K + k);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint4 raw = up_src[i];
        const uint32_t w4[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int e = i * 8 + j * 2 + h;
            float up = __uint_as_float(h ? (w4[j] & 0xffff0000u) : (w4[j] << 16));
            up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
            const float gate = fminf(v[e], swiglu_limit);
            const float silu =
                __bfloat162float(__float2bfloat16(div_rn(gate, 1.0f + exp_rn(-gate))));
            v[e] = __bfloat162float(__float2bfloat16(silu * up));
          }
        }
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < kVec; ++i) v[i] = 0.0f;
  }
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < kVec; ++i) amax = fmaxf(amax, fabsf(v[i]));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  if (!active) return;
  const float s = fmaxf(amax, kQuantEps) / kFp8Max;
  uint32_t packed[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float4 f4;
    f4.x = fminf(fmaxf(div_rn(v[i * 4], s), -kFp8Max), kFp8Max);
    f4.y = fminf(fmaxf(div_rn(v[i * 4 + 1], s), -kFp8Max), kFp8Max);
    f4.z = fminf(fmaxf(div_rn(v[i * 4 + 2], s), -kFp8Max), kFp8Max);
    f4.w = fminf(fmaxf(div_rn(v[i * 4 + 3], s), -kFp8Max), kFp8Max);
    packed[i] = __musa_cvt_float4_to_fp8x4(f4, __MT_SATFINITE, __MT_E4M3);
  }
  *reinterpret_cast<uint4*>(xq + m * xq_stride + k) =
      make_uint4(packed[0], packed[1], packed[2], packed[3]);
  if ((lane & 1) == 0) xs[m * xs_stride + k / kGroup] = s;
}

// CTA 0 partitions up to kMaxMapSlots routed expert ids into disjoint 16-row MMA
// work-units in first-use order: map[u * 18] = [expert, chunk_rows, row_0, ...] and
// map[map_slots * 18] = num_units. CTAs past CTA 0 concurrently quantize the [M, K]
// BF16 MoE input per 32-wide group.
__global__ void __launch_bounds__(kMaxMapSlots)
    moe_route_prep_kernel(const int32_t* __restrict__ ids, int total_rows, int num_experts,
                          int map_slots, int32_t* __restrict__ route_map,
                          const uint16_t* __restrict__ x, int64_t x_stride,
                          uint8_t* __restrict__ xq, int64_t xq_stride,
                          float* __restrict__ xs, int64_t xs_stride, int m_rows, int K,
                          int chunks) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  if (blockIdx.x > 0) {
    const int gw = (blockIdx.x - 1) * kMapWarps + warp;
    const int m = gw / chunks;
    const int k = (gw - m * chunks) * kStep + lane * kVec;
    const bool active = m < m_rows && k < K;
    float v[kVec];
    if (active) {
      const uint4* src = reinterpret_cast<const uint4*>(x + m * x_stride + k);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint4 raw = src[i];
        const uint32_t w4[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          v[i * 8 + j * 2] = __uint_as_float(w4[j] << 16);
          v[i * 8 + j * 2 + 1] = __uint_as_float(w4[j] & 0xffff0000u);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < kVec; ++i) v[i] = 0.0f;
    }
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < kVec; ++i) amax = fmaxf(amax, fabsf(v[i]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    if (!active) return;
    const float s = fmaxf(amax, kQuantEps) / kFp8Max;
    uint32_t packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float4 f4;
      f4.x = fminf(fmaxf(div_rn(v[i * 4], s), -kFp8Max), kFp8Max);
      f4.y = fminf(fmaxf(div_rn(v[i * 4 + 1], s), -kFp8Max), kFp8Max);
      f4.z = fminf(fmaxf(div_rn(v[i * 4 + 2], s), -kFp8Max), kFp8Max);
      f4.w = fminf(fmaxf(div_rn(v[i * 4 + 3], s), -kFp8Max), kFp8Max);
      packed[i] = __musa_cvt_float4_to_fp8x4(f4, __MT_SATFINITE, __MT_E4M3);
    }
    *reinterpret_cast<uint4*>(xq + m * xq_stride + k) =
        make_uint4(packed[0], packed[1], packed[2], packed[3]);
    if ((lane & 1) == 0) xs[m * xs_stride + k / kGroup] = s;
    return;
  }

  __shared__ int s_ids[kMaxMapSlots];
  __shared__ int s_uniq[kMaxMapSlots];
  __shared__ unsigned s_first[kMapWarps];

  const int r = threadIdx.x;
  int id = (r < total_rows) ? ids[r] : -1;
  if (id < 0 || id >= num_experts) {
    id = -1;
  }
  s_ids[r] = id;
  __syncthreads();

  int earlier = 0;
  int count = 0;
  int leader_row = r;
  const bool valid = (r < total_rows) && (id >= 0);
  if (valid) {
    for (int q = 0; q < total_rows; ++q) {
      if (s_ids[q] == id) {
        if (q <= r && (count % kMapRows) == 0) {
          leader_row = q;
        }
        if (q == r) {
          earlier = count;
        }
        ++count;
      }
    }
  }
  const int slot = earlier % kMapRows;
  const bool first_unit = valid && (slot == 0);
  const unsigned first_mask = __ballot_sync(0xffffffffu, first_unit);
  if (lane == 0) {
    s_first[warp] = first_mask;
  }
  __syncthreads();

  int uniq = __popc(first_mask & ((1u << lane) - 1u));
  for (int w = 0; w < warp; ++w) {
    uniq += __popc(s_first[w]);
  }
  if (first_unit) {
    s_uniq[r] = uniq;
    route_map[uniq * kMapWidth] = id;
    const int rem = count - earlier;
    route_map[uniq * kMapWidth + 1] = rem < kMapRows ? rem : kMapRows;
  }
  __syncthreads();

  if (valid) {
    route_map[s_uniq[leader_row] * kMapWidth + 2 + slot] = r;
  }
  if (r == 0) {
    int total_units = 0;
#pragma unroll
    for (int w = 0; w < kMapWarps; ++w) {
      total_units += __popc(s_first[w]);
    }
    route_map[map_slots * kMapWidth] = total_units;
  }
}

// out[m, n] = sum_k dequant(xq[m, k]) * dequant(w[n, k]) for M rows, activation read
// from global memory. A row group of RPW weight rows is split over KS warps.
template <int M, int RPW, int WARPS, int KS>
__global__ void __launch_bounds__(WARPS * kWarpSize)
    dense_reg_kernel(const uint8_t* __restrict__ xq, int64_t x_stride,
                     const float* __restrict__ x_scale, int64_t xs_stride,
                     const uint8_t* __restrict__ w, int64_t w_stride,
                     const float* __restrict__ w_scale, int64_t ws_stride,
                     __mt_bfloat16* __restrict__ out, int64_t out_stride, int N, int K,
                     int group_n, int a_group) {
  constexpr int kGroupsPerCta = WARPS / KS;
  __shared__ float part_sm[KS > 1 ? WARPS * M * RPW : 1];
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int row_group = warp / KS;
  const int ks = warp % KS;
  const int n_base = (blockIdx.x * kGroupsPerCta + row_group) * RPW;
  float acc[M][RPW];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int r = 0; r < RPW; ++r) acc[m][r] = 0.0f;
  int64_t row_off[RPW];
  int64_t srow_off[RPW];
#pragma unroll
  for (int r = 0; r < RPW; ++r) {
    const int n = min(n_base + r, N - 1);
    row_off[r] = static_cast<int64_t>(n) * w_stride;
    srow_off[r] = static_cast<int64_t>(n / group_n) * ws_stride;
  }
  const int steps = (K + kStep - 1) / kStep;
  const int my_steps = (steps - ks + KS - 1) / KS;
  const int rot = (blockIdx.x * kGroupsPerCta + row_group) % steps;
  auto k_of = [&](int t) { return ((t * KS + ks + rot) % steps) * kStep + lane * kVec; };
  uint4 cur[RPW];
  {
    const int k = k_of(0);
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      cur[r] = (my_steps > 0 && k < K) ? *reinterpret_cast<const uint4*>(w + row_off[r] + k)
                                       : make_uint4(0, 0, 0, 0);
    }
  }
  for (int t = 0; t < my_steps; ++t) {
    const int k = k_of(t);
    const int k_next = k_of(t + 1);
    uint4 nxt[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      nxt[r] = (t + 1 < my_steps && k_next < K)
                   ? *reinterpret_cast<const uint4*>(w + row_off[r] + k_next)
                   : make_uint4(0, 0, 0, 0);
    }
    if (k < K) {
      float bs[RPW];
      float wf[RPW][kVec];
#pragma unroll
      for (int r = 0; r < RPW; ++r) {
        bs[r] = w_scale[srow_off[r] + k / kGroup];
        fp8x16_to_float(cur[r], wf[r]);
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float a[kVec];
        fp8x16_to_float(*reinterpret_cast<const uint4*>(xq + m * x_stride + k), a);
        const float as = x_scale[m * xs_stride + k / a_group];
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
          float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
#pragma unroll
          for (int j = 0; j < kVec; j += 4) {
            p0 = fmaf(wf[r][j], a[j], p0);
            p1 = fmaf(wf[r][j + 1], a[j + 1], p1);
            p2 = fmaf(wf[r][j + 2], a[j + 2], p2);
            p3 = fmaf(wf[r][j + 3], a[j + 3], p3);
          }
          acc[m][r] = fmaf((p0 + p1) + (p2 + p3), as * bs[r], acc[m][r]);
        }
      }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) cur[r] = nxt[r];
  }
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int r = 0; r < RPW; ++r) acc[m][r] = warp_sum(acc[m][r]);
  if constexpr (KS > 1) {
    if (lane == 0) {
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int r = 0; r < RPW; ++r) part_sm[(warp * M + m) * RPW + r] = acc[m][r];
    }
    __syncthreads();
    if (ks != 0 || lane != 0) return;
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int j = 1; j < KS; ++j) acc[m][r] += part_sm[((warp + j) * M + m) * RPW + r];
  }
  if (lane == 0) {
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int r = 0; r < RPW; ++r) {
        const int n = n_base + r;
        if (n < N) out[m * out_stride + n] = __float2bfloat16(acc[m][r]);
      }
  }
}

// dense_reg_kernel with the dequantized activation staged once per CTA in shared
// memory. Element k = s * 512 + lane * 16 + j4 * 4 + i of a row sits at float4
// (s * 4 + j4) * 32 + lane, so the LDS.128 reads of a warp are conflict-free.
template <int M, int RPW, int WARPS, int KS>
__global__ void __launch_bounds__(WARPS * kWarpSize)
    dense_smem_kernel(const uint8_t* __restrict__ xq, int64_t x_stride,
                      const float* __restrict__ x_scale, int64_t xs_stride,
                      const uint8_t* __restrict__ w, int64_t w_stride,
                      const float* __restrict__ w_scale, int64_t ws_stride,
                      __mt_bfloat16* __restrict__ out, int64_t out_stride, int N, int K,
                      int group_n, int a_group) {
  extern __shared__ float4 act4[];
  constexpr int kGroupsPerCta = WARPS / KS;
  __shared__ float part_sm[KS > 1 ? WARPS * M * RPW : 1];
  const int steps = (K + kStep - 1) / kStep;
  const int per_row4 = steps * kStep / 4;
  for (int idx = threadIdx.x; idx < M * per_row4; idx += WARPS * kWarpSize) {
    const int m = idx / per_row4;
    const int k = (idx - m * per_row4) * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (k < K) {
      v = fp8x4_to_float4(*reinterpret_cast<const uint32_t*>(xq + m * x_stride + k),
                          x_scale[m * xs_stride + k / a_group]);
    }
    const int st = k / kStep;
    const int ln = (k % kStep) / kVec;
    const int j4 = (k % kVec) / 4;
    act4[m * per_row4 + (st * 4 + j4) * kWarpSize + ln] = v;
  }
  __syncthreads();

  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int row_group = warp / KS;
  const int ks = warp % KS;
  const int n_base = (blockIdx.x * kGroupsPerCta + row_group) * RPW;
  float acc[M][RPW];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int r = 0; r < RPW; ++r) acc[m][r] = 0.0f;
  int64_t row_off[RPW];
  int64_t srow_off[RPW];
#pragma unroll
  for (int r = 0; r < RPW; ++r) {
    const int n = min(n_base + r, N - 1);
    row_off[r] = static_cast<int64_t>(n) * w_stride;
    srow_off[r] = static_cast<int64_t>(n / group_n) * ws_stride;
  }
  const int my_steps = (steps - ks + KS - 1) / KS;
  const int rot = (blockIdx.x * kGroupsPerCta + row_group) % steps;
  auto step_of = [&](int t) { return (t * KS + ks + rot) % steps; };
  uint4 cur[RPW];
  {
    const int k = step_of(0) * kStep + lane * kVec;
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      cur[r] = (my_steps > 0 && k < K) ? *reinterpret_cast<const uint4*>(w + row_off[r] + k)
                                       : make_uint4(0, 0, 0, 0);
    }
  }
  for (int t = 0; t < my_steps; ++t) {
    const int st = step_of(t);
    const int k = st * kStep + lane * kVec;
    const int k_next = step_of(t + 1) * kStep + lane * kVec;
    uint4 nxt[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      nxt[r] = (t + 1 < my_steps && k_next < K)
                   ? *reinterpret_cast<const uint4*>(w + row_off[r] + k_next)
                   : make_uint4(0, 0, 0, 0);
    }
    if (k < K) {
      float wf[RPW][kVec];
      float bs[RPW];
#pragma unroll
      for (int r = 0; r < RPW; ++r) {
        fp8x16_to_float(cur[r], wf[r]);
        bs[r] = w_scale[srow_off[r] + k / kGroup];
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const float4* ap = act4 + m * per_row4 + st * 4 * kWarpSize + lane;
        float p[RPW][2];
#pragma unroll
        for (int r = 0; r < RPW; ++r) p[r][0] = p[r][1] = 0.0f;
#pragma unroll
        for (int j4 = 0; j4 < 4; ++j4) {
          const float4 a = ap[j4 * kWarpSize];
#pragma unroll
          for (int r = 0; r < RPW; ++r) {
            p[r][j4 & 1] = fmaf(wf[r][j4 * 4], a.x, p[r][j4 & 1]);
            p[r][j4 & 1] = fmaf(wf[r][j4 * 4 + 1], a.y, p[r][j4 & 1]);
            p[r][j4 & 1] = fmaf(wf[r][j4 * 4 + 2], a.z, p[r][j4 & 1]);
            p[r][j4 & 1] = fmaf(wf[r][j4 * 4 + 3], a.w, p[r][j4 & 1]);
          }
        }
#pragma unroll
        for (int r = 0; r < RPW; ++r) acc[m][r] = fmaf(p[r][0] + p[r][1], bs[r], acc[m][r]);
      }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) cur[r] = nxt[r];
  }
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int r = 0; r < RPW; ++r) acc[m][r] = warp_sum(acc[m][r]);
  if constexpr (KS > 1) {
    if (lane == 0) {
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int r = 0; r < RPW; ++r) part_sm[(warp * M + m) * RPW + r] = acc[m][r];
    }
    __syncthreads();
    if (ks != 0 || lane != 0) return;
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int j = 1; j < KS; ++j) acc[m][r] += part_sm[((warp + j) * M + m) * RPW + r];
  }
  if (lane == 0) {
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int r = 0; r < RPW; ++r) {
        const int n = n_base + r;
        if (n < N) out[m * out_stride + n] = __float2bfloat16(acc[m][r]);
      }
  }
}

// Routed rows of the MoE gate/up projection: out[j] = W1[ids[j]] @ dequant(xq[j / topk]).
// Rows routed outside [0, E) are zero.
template <int RPW, int WARPS>
__global__ void __launch_bounds__(WARPS * kWarpSize)
    moe_w1_kernel(const uint8_t* __restrict__ xq, int64_t x_stride,
                  const float* __restrict__ x_scale, int64_t xs_stride,
                  const uint8_t* __restrict__ w, int64_t w_estride, int64_t w_stride,
                  const float* __restrict__ w_scale, int64_t ws_estride, int64_t ws_stride,
                  const int32_t* __restrict__ expert_ids, int topk, int num_experts,
                  __mt_bfloat16* __restrict__ out, int64_t out_stride, int N, int K, int group_n) {
  const int j = blockIdx.y;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int n_base = (blockIdx.x * WARPS + warp) * RPW;
  const int e = expert_ids[j];
  float acc[RPW];
#pragma unroll
  for (int r = 0; r < RPW; ++r) acc[r] = 0.0f;
  if (e >= 0 && e < num_experts) {
    const uint8_t* we = w + static_cast<int64_t>(e) * w_estride;
    const float* wse = w_scale + static_cast<int64_t>(e) * ws_estride;
    const uint8_t* xr = xq + static_cast<int64_t>(j / topk) * x_stride;
    const float* xsr = x_scale + static_cast<int64_t>(j / topk) * xs_stride;
    int64_t row_off[RPW];
    int64_t srow_off[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      const int n = min(n_base + r, N - 1);
      row_off[r] = static_cast<int64_t>(n) * w_stride;
      srow_off[r] = static_cast<int64_t>(n / group_n) * ws_stride;
    }
    const int steps = (K + kStep - 1) / kStep;
    const int rot = (blockIdx.x * WARPS + warp + j * 3) % steps;
    auto k_of = [&](int t) { return ((t + rot) % steps) * kStep + lane * kVec; };
    uint4 cur[RPW];
    {
      const int k = k_of(0);
#pragma unroll
      for (int r = 0; r < RPW; ++r) {
        cur[r] = k < K ? *reinterpret_cast<const uint4*>(we + row_off[r] + k)
                       : make_uint4(0, 0, 0, 0);
      }
    }
    for (int t = 0; t < steps; ++t) {
      const int k = k_of(t);
      const int k_next = k_of(t + 1);
      uint4 nxt[RPW];
#pragma unroll
      for (int r = 0; r < RPW; ++r) {
        nxt[r] = (t + 1 < steps && k_next < K)
                     ? *reinterpret_cast<const uint4*>(we + row_off[r] + k_next)
                     : make_uint4(0, 0, 0, 0);
      }
      if (k < K) {
        float a[kVec];
        fp8x16_to_float(*reinterpret_cast<const uint4*>(xr + k), a);
        const float as = xsr[k / kGroup];
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
          float wf[kVec];
          fp8x16_to_float(cur[r], wf);
          float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
#pragma unroll
          for (int i = 0; i < kVec; i += 4) {
            p0 = fmaf(wf[i], a[i], p0);
            p1 = fmaf(wf[i + 1], a[i + 1], p1);
            p2 = fmaf(wf[i + 2], a[i + 2], p2);
            p3 = fmaf(wf[i + 3], a[i + 3], p3);
          }
          acc[r] = fmaf((p0 + p1) + (p2 + p3), as * wse[srow_off[r] + k / kGroup], acc[r]);
        }
      }
#pragma unroll
      for (int r = 0; r < RPW; ++r) cur[r] = nxt[r];
    }
  }
#pragma unroll
  for (int r = 0; r < RPW; ++r) {
    const float v = warp_sum(acc[r]);
    const int n = n_base + r;
    if (lane == 0 && n < N) out[static_cast<int64_t>(j) * out_stride + n] = __float2bfloat16(v);
  }
}

// MoE down projection summed over each token's routed rows, rounded once:
// out[t] = sum_s rw[t, s] * W2[ids[t, s]] @ dequant(aq[t * TOPK + s]).
// Lanes walk the (slot, 16-element chunk) units of a token, CPR chunks per row. The
// scaled activation of a token is staged in shared memory: unit u = it * 32 + lane
// keeps its 16 values at float4 (it * 4 + j4) * 32 + lane.
template <int TOPK, int CPR, int RPW, int WARPS>
__global__ void __launch_bounds__(WARPS * kWarpSize)
    moe_w2_kernel(const uint8_t* __restrict__ aq, int64_t a_stride,
                  const float* __restrict__ a_scale, int64_t as_stride,
                  const uint8_t* __restrict__ w, int64_t w_estride, int64_t w_stride,
                  const float* __restrict__ w_scale, int64_t ws_estride, int64_t ws_stride,
                  const int32_t* __restrict__ expert_ids, const float* __restrict__ routed_w,
                  int num_experts, __mt_bfloat16* __restrict__ out, int64_t out_stride, int N,
                  int group_n) {
  constexpr int kUnits = TOPK * CPR;
  constexpr int kIters = (kUnits + kWarpSize - 1) / kWarpSize;
  __shared__ float4 act4[kIters * 4 * kWarpSize];
  __shared__ int64_t ebase_sm[TOPK];
  __shared__ int64_t sbase_sm[TOPK];
  const int t = blockIdx.y;
  if (threadIdx.x < TOPK) {
    const int e = expert_ids[t * TOPK + threadIdx.x];
    const bool valid = e >= 0 && e < num_experts;
    ebase_sm[threadIdx.x] = valid ? static_cast<int64_t>(e) * w_estride : -1;
    sbase_sm[threadIdx.x] = valid ? static_cast<int64_t>(e) * ws_estride : -1;
  }
  for (int idx = threadIdx.x; idx < kIters * 4 * kWarpSize; idx += WARPS * kWarpSize) {
    const int it = idx / (4 * kWarpSize);
    const int rem = idx - it * 4 * kWarpSize;
    const int j4 = rem / kWarpSize;
    const int u = it * kWarpSize + rem - j4 * kWarpSize;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (u < kUnits) {
      const int sl = u / CPR;
      const int k = (u - sl * CPR) * kVec + j4 * 4;
      const int64_t row = static_cast<int64_t>(t) * TOPK + sl;
      v = fp8x4_to_float4(*reinterpret_cast<const uint32_t*>(aq + row * a_stride + k),
                          a_scale[row * as_stride + k / kGroup] * routed_w[row]);
    }
    act4[idx] = v;
  }
  __syncthreads();
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int n_base = (blockIdx.x * WARPS + warp) * RPW;
  // RPW divides group_n, so a warp's rows share one weight-scale row.
  const int64_t scale_row = static_cast<int64_t>(min(n_base, N - 1) / group_n) * ws_stride;
  uint4 wraw[kIters][RPW];
  float bs[kIters];
#pragma unroll
  for (int it = 0; it < kIters; ++it) {
    const int u = it * kWarpSize + lane;
    const int sl = min(u / CPR, TOPK - 1);
    const int k = (u - (u / CPR) * CPR) * kVec;
    const int64_t eb = ebase_sm[sl];
    const bool valid = u < kUnits && eb >= 0;
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      const int n = min(n_base + r, N - 1);
      wraw[it][r] = valid ? *reinterpret_cast<const uint4*>(w + eb + static_cast<int64_t>(n) * w_stride + k)
                          : make_uint4(0, 0, 0, 0);
    }
    bs[it] = valid ? w_scale[sbase_sm[sl] + scale_row + k / kGroup] : 0.0f;
  }
  float acc[RPW];
#pragma unroll
  for (int r = 0; r < RPW; ++r) acc[r] = 0.0f;
#pragma unroll
  for (int it = 0; it < kIters; ++it) {
    float4 a[4];
#pragma unroll
    for (int j4 = 0; j4 < 4; ++j4) a[j4] = act4[(it * 4 + j4) * kWarpSize + lane];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      float wf[kVec];
      fp8x16_to_float(wraw[it][r], wf);
      float p0 = 0.0f, p1 = 0.0f;
#pragma unroll
      for (int j4 = 0; j4 < 4; j4 += 2) {
        p0 = fmaf(wf[j4 * 4], a[j4].x, p0);
        p0 = fmaf(wf[j4 * 4 + 1], a[j4].y, p0);
        p0 = fmaf(wf[j4 * 4 + 2], a[j4].z, p0);
        p0 = fmaf(wf[j4 * 4 + 3], a[j4].w, p0);
        p1 = fmaf(wf[j4 * 4 + 4], a[j4 + 1].x, p1);
        p1 = fmaf(wf[j4 * 4 + 5], a[j4 + 1].y, p1);
        p1 = fmaf(wf[j4 * 4 + 6], a[j4 + 1].z, p1);
        p1 = fmaf(wf[j4 * 4 + 7], a[j4 + 1].w, p1);
      }
      acc[r] = fmaf(p0 + p1, bs[it], acc[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < RPW; ++r) {
    const float v = warp_sum(acc[r]);
    const int n = n_base + r;
    if (lane == 0 && n < N) out[static_cast<int64_t>(t) * out_stride + n] = __float2bfloat16(v);
  }
}


struct DenseArgs {
  const uint8_t* xq;
  int64_t x_stride;
  const float* xs;
  int64_t xs_stride;
  const uint8_t* w;
  int64_t w_stride;
  const float* ws;
  int64_t ws_stride;
  __mt_bfloat16* out;
  int64_t out_stride;
  int n;
  int k;
  int group_n;
  int a_group;
};

template <int M, int RPW, int WARPS, int KS, bool SMEM>
void launch_dense(const DenseArgs& a, musaStream_t stream) {
  const int rows_per_cta = WARPS / KS * RPW;
  const int grid = (a.n + rows_per_cta - 1) / rows_per_cta;
  if constexpr (SMEM) {
    const int steps = (a.k + kStep - 1) / kStep;
    const int smem = M * steps * kStep * static_cast<int>(sizeof(float));
    TVM_FFI_ICHECK_LE(smem, kMaxSmemBytes) << "activation does not fit in shared memory";
    auto kernel = dense_smem_kernel<M, RPW, WARPS, KS>;
    musaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                         musaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<grid, WARPS * kWarpSize, smem, stream>>>(a.xq, a.x_stride, a.xs, a.xs_stride, a.w,
                                                     a.w_stride, a.ws, a.ws_stride, a.out,
                                                     a.out_stride, a.n, a.k, a.group_n,
                                                     a.a_group);
  } else {
    dense_reg_kernel<M, RPW, WARPS, KS><<<grid, WARPS * kWarpSize, 0, stream>>>(
        a.xq, a.x_stride, a.xs, a.xs_stride, a.w, a.w_stride, a.ws, a.ws_stride, a.out,
        a.out_stride, a.n, a.k, a.group_n, a.a_group);
  }
}

// Config ids: 0-5 keep the activation in registers, 6-10 stage it in shared memory.
// Each tuple is (RPW, WARPS, KS).
template <int M>
void dispatch_dense(int cfg, const DenseArgs& a, musaStream_t stream) {
  switch (cfg) {
    case 0: launch_dense<M, 1, 4, 1, false>(a, stream); return;
    case 1: launch_dense<M, 2, 4, 1, false>(a, stream); return;
    case 2: launch_dense<M, 2, 4, 2, false>(a, stream); return;
    case 3: launch_dense<M, 2, 8, 2, false>(a, stream); return;
    case 4: launch_dense<M, 4, 8, 2, false>(a, stream); return;
    case 5: launch_dense<M, 4, 4, 1, false>(a, stream); return;
    case 6: launch_dense<M, 1, 16, 1, true>(a, stream); return;
    case 7: launch_dense<M, 2, 16, 1, true>(a, stream); return;
    case 8: launch_dense<M, 1, 32, 1, true>(a, stream); return;
    case 9: launch_dense<M, 1, 32, 2, true>(a, stream); return;
    case 10: launch_dense<M, 2, 32, 2, true>(a, stream); return;
    default: TVM_FFI_THROW(ValueError) << "unknown GEMV config " << cfg;
  }
}

musaStream_t stream_of(ffi::TensorView t) {
  return static_cast<musaStream_t>(TVMFFIEnvGetStream(t.device().device_type, t.device().device_id));
}

int group_rows(ffi::TensorView weight, ffi::TensorView weight_scale, int dim) {
  const int rows = static_cast<int>(weight.size(dim));
  const int blocks = static_cast<int>(weight_scale.size(dim));
  return (rows + blocks - 1) / blocks;
}

void check_launch(const char* name) {
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess) << name << " failed: " << musaGetErrorString(err);
}

}  // namespace

void fp8_act_quant(ffi::TensorView x, ffi::TensorView xq, ffi::TensorView xs, bool swiglu,
                   double swiglu_limit) {
  ffi::MUSADeviceGuard device_guard(x.device().device_id);
  const int m = static_cast<int>(xq.size(0));
  const int k = static_cast<int>(xq.size(1));
  TVM_FFI_ICHECK_EQ(k % kGroup, 0);
  TVM_FFI_ICHECK_EQ(x.stride(1), 1);
  TVM_FFI_ICHECK_EQ(x.size(1), swiglu ? 2 * k : k);
  const int chunks = (k + kStep - 1) / kStep;
  const int grid = (m * chunks + 3) / 4;
  const auto* xp = static_cast<const uint16_t*>(x.data_ptr());
  auto* qp = static_cast<uint8_t*>(xq.data_ptr());
  auto* sp = static_cast<float*>(xs.data_ptr());
  const float lim = static_cast<float>(swiglu_limit);
  if (swiglu) {
    act_quant_kernel<true><<<grid, 128, 0, stream_of(x)>>>(
        xp, x.stride(0), qp, xq.stride(0), sp, xs.stride(0), m, k, chunks, lim);
  } else {
    act_quant_kernel<false><<<grid, 128, 0, stream_of(x)>>>(
        xp, x.stride(0), qp, xq.stride(0), sp, xs.stride(0), m, k, chunks, lim);
  }
  check_launch("fp8_act_quant");
}

void fp8_dense_gemv(ffi::TensorView xq, ffi::TensorView xs, ffi::TensorView w,
                    ffi::TensorView ws, ffi::TensorView out, int64_t cfg) {
  ffi::MUSADeviceGuard device_guard(xq.device().device_id);
  TVM_FFI_ICHECK_EQ(w.stride(1), 1);
  TVM_FFI_ICHECK_EQ(ws.stride(1), 1);
  TVM_FFI_ICHECK_EQ(xq.stride(1), 1);
  TVM_FFI_ICHECK_EQ(out.stride(1), 1);
  TVM_FFI_ICHECK_EQ(ws.size(1) * kGroup, w.size(1));
  const int a_group = static_cast<int>(w.size(1) / xs.size(1));
  TVM_FFI_ICHECK_EQ(xs.size(1) * a_group, w.size(1));
  TVM_FFI_ICHECK(a_group == 32 || a_group == 64 || a_group == 128);
  const DenseArgs a{static_cast<const uint8_t*>(xq.data_ptr()), xq.stride(0),
                    static_cast<const float*>(xs.data_ptr()), xs.stride(0),
                    static_cast<const uint8_t*>(w.data_ptr()), w.stride(0),
                    static_cast<const float*>(ws.data_ptr()), ws.stride(0),
                    static_cast<__mt_bfloat16*>(out.data_ptr()), out.stride(0),
                    static_cast<int>(w.size(0)), static_cast<int>(w.size(1)),
                    group_rows(w, ws, 0), a_group};
  const int c = static_cast<int>(cfg);
  musaStream_t stream = stream_of(xq);
  switch (xq.size(0)) {
    case 1: dispatch_dense<1>(c, a, stream); break;
    case 2: dispatch_dense<2>(c, a, stream); break;
    case 3: dispatch_dense<3>(c, a, stream); break;
    case 4: dispatch_dense<4>(c, a, stream); break;
    case 5: dispatch_dense<5>(c, a, stream); break;
    case 6: dispatch_dense<6>(c, a, stream); break;
    case 7: dispatch_dense<7>(c, a, stream); break;
    case 8: dispatch_dense<8>(c, a, stream); break;
    default: TVM_FFI_THROW(ValueError) << "rows must be 1..8, got " << xq.size(0);
  }
  check_launch("fp8_dense_gemv");
}

void fp8_moe_w1_gemv(ffi::TensorView xq, ffi::TensorView xs, ffi::TensorView w, ffi::TensorView ws,
                     ffi::TensorView expert_ids, ffi::TensorView out, int64_t topk) {
  ffi::MUSADeviceGuard device_guard(xq.device().device_id);
  TVM_FFI_ICHECK_EQ(w.stride(2), 1);
  TVM_FFI_ICHECK_EQ(ws.stride(2), 1);
  TVM_FFI_ICHECK_EQ(ws.size(2) * kGroup, w.size(2));
  const int rows = static_cast<int>(expert_ids.size(0));
  const int n = static_cast<int>(w.size(1));
  constexpr int kRpw = 2;
  constexpr int kWarps = 4;
  const dim3 grid((n + kWarps * kRpw - 1) / (kWarps * kRpw), rows);
  moe_w1_kernel<kRpw, kWarps><<<grid, kWarps * kWarpSize, 0, stream_of(xq)>>>(
      static_cast<const uint8_t*>(xq.data_ptr()), xq.stride(0),
      static_cast<const float*>(xs.data_ptr()), xs.stride(0),
      static_cast<const uint8_t*>(w.data_ptr()), w.stride(0), w.stride(1),
      static_cast<const float*>(ws.data_ptr()), ws.stride(0), ws.stride(1),
      static_cast<const int32_t*>(expert_ids.data_ptr()), static_cast<int>(topk),
      static_cast<int>(w.size(0)), static_cast<__mt_bfloat16*>(out.data_ptr()), out.stride(0), n,
      static_cast<int>(w.size(2)), group_rows(w, ws, 1));
  check_launch("fp8_moe_w1_gemv");
}

void fp8_moe_w2_gemv(ffi::TensorView aq, ffi::TensorView as, ffi::TensorView w, ffi::TensorView ws,
                     ffi::TensorView expert_ids, ffi::TensorView routed_w, ffi::TensorView out) {
  ffi::MUSADeviceGuard device_guard(aq.device().device_id);
  TVM_FFI_ICHECK_EQ(w.stride(2), 1);
  TVM_FFI_ICHECK_EQ(ws.stride(2), 1);
  const int tokens = static_cast<int>(out.size(0));
  const int topk = static_cast<int>(expert_ids.size(0)) / tokens;
  const int n = static_cast<int>(w.size(1));
  const int k = static_cast<int>(w.size(2));
  const int group_n = group_rows(w, ws, 1);
  constexpr int kRpw = 2;
  constexpr int kWarps = 4;
  TVM_FFI_ICHECK_EQ(group_n % kRpw, 0);
  const dim3 grid((n + kWarps * kRpw - 1) / (kWarps * kRpw), tokens);
  musaStream_t stream = stream_of(aq);
#define MOE_W2_CASE(TOPK, K)                                                                     \
  if (topk == TOPK && k == K) {                                                                  \
    moe_w2_kernel<TOPK, K / kVec, kRpw, kWarps><<<grid, kWarps * kWarpSize, 0, stream>>>(          \
        static_cast<const uint8_t*>(aq.data_ptr()), aq.stride(0),                                \
        static_cast<const float*>(as.data_ptr()), as.stride(0),                                  \
        static_cast<const uint8_t*>(w.data_ptr()), w.stride(0), w.stride(1),                     \
        static_cast<const float*>(ws.data_ptr()), ws.stride(0), ws.stride(1),                    \
        static_cast<const int32_t*>(expert_ids.data_ptr()),                                      \
        static_cast<const float*>(routed_w.data_ptr()), static_cast<int>(w.size(0)),             \
        static_cast<__mt_bfloat16*>(out.data_ptr()), out.stride(0), n, group_n);                 \
    check_launch("fp8_moe_w2_gemv");                                                             \
    return;                                                                                      \
  }
  MOE_W2_CASE(6, 288)
  MOE_W2_CASE(3, 288)
#undef MOE_W2_CASE
  TVM_FFI_THROW(ValueError) << "unsupported MoE down shape: top_k " << topk << ", K " << k;
}

void fp8_moe_route_prep(ffi::TensorView ids, ffi::TensorView route_map, ffi::TensorView x,
                        ffi::TensorView xq, ffi::TensorView xs, int64_t num_experts,
                        int64_t map_slots, bool quant_x) {
  ffi::MUSADeviceGuard device_guard(ids.device().device_id);
  const int total_rows = static_cast<int>(ids.size(0));
  const int slots = static_cast<int>(map_slots);
  TVM_FFI_ICHECK_EQ(ids.stride(0), 1);
  TVM_FFI_ICHECK_EQ(route_map.stride(0), 1);
  TVM_FFI_ICHECK_GE(slots, total_rows);
  TVM_FFI_ICHECK_LE(slots, kMaxMapSlots);
  TVM_FFI_ICHECK_GE(route_map.size(0), (slots + 1) * kMapWidth);

  int m_rows = 0;
  int K = 0;
  int chunks = 1;
  int quant_ctas = 0;
  const uint16_t* xp = nullptr;
  int64_t x_stride = 0;
  uint8_t* qp = nullptr;
  int64_t xq_stride = 0;
  float* sp = nullptr;
  int64_t xs_stride = 0;
  if (quant_x) {
    m_rows = static_cast<int>(xq.size(0));
    K = static_cast<int>(xq.size(1));
    TVM_FFI_ICHECK_EQ(K % kGroup, 0);
    TVM_FFI_ICHECK_EQ(x.size(0), m_rows);
    TVM_FFI_ICHECK_EQ(x.size(1), K);
    TVM_FFI_ICHECK_EQ(x.stride(1), 1);
    TVM_FFI_ICHECK_EQ(x.stride(0) % 8, 0);
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16, 0);
    chunks = (K + kStep - 1) / kStep;
    quant_ctas = (m_rows * chunks + kMapWarps - 1) / kMapWarps;
    xp = static_cast<const uint16_t*>(x.data_ptr());
    x_stride = x.stride(0);
    qp = static_cast<uint8_t*>(xq.data_ptr());
    xq_stride = xq.stride(0);
    sp = static_cast<float*>(xs.data_ptr());
    xs_stride = xs.stride(0);
  }
  const dim3 grid(1 + quant_ctas);
  moe_route_prep_kernel<<<grid, kMaxMapSlots, 0, stream_of(ids)>>>(
      static_cast<const int32_t*>(ids.data_ptr()), total_rows, static_cast<int>(num_experts),
      slots, static_cast<int32_t*>(route_map.data_ptr()), xp, x_stride, qp, xq_stride, sp,
      xs_stride, m_rows, K, chunks);
  check_launch("fp8_moe_route_prep");
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_act_quant, fp8_act_quant);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_dense_gemv, fp8_dense_gemv);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_moe_w1_gemv, fp8_moe_w1_gemv);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_moe_w2_gemv, fp8_moe_w2_gemv);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_moe_route_prep, fp8_moe_route_prep);
