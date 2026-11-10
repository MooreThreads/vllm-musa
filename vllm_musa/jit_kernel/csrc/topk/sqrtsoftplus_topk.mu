// DeepSeek-V4 routing: top-k of sqrt(softplus(logit)) + bias. The selected experts
// carry their unbiased scores, renormalized to sum to the routed scaling factor.
//
// The arithmetic is that of the vLLM topkGatingSoftplusSqrt kernel, so weights and
// ids match it bit for bit: ties go to the lower expert id and a picked score is
// cleared to -10000. One warp handles one token; lane l holds experts l, l + 32, ...
// Image tokens (five consecutive sentinel ids from image_lo) select with bias_vl
// instead of the correction bias. An optional constant slot (a shared expert folded
// into the routed expert stack) follows the routed top-k. The logits are FP32 or the
// BF16 gate output, which is widened exactly.
#include <cstdint>

#include <musa_bf16.h>
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
constexpr int kWarps = 4;
constexpr float kCleared = -10000.0f;
constexpr float kAbsent = -3.0e38f;
constexpr int kQuantVec = 16;
constexpr float kFp8Max = 448.0f;
constexpr float kQuantEps = 1e-10f;

#pragma float_control(precise, on, push)
__device__ __forceinline__ float div_rn(float x, float y) { return x / y; }
#pragma float_control(pop)

// per_token_group_quant_fp8 (32-wide groups) of 16 BF16 values per lane; the two
// lanes of a group are adjacent. Every lane of the warp calls it.
__device__ __forceinline__ void quant16(const uint16_t* __restrict__ x, uint8_t* __restrict__ xq,
                                        float* __restrict__ xs, int lane, bool active) {
  float v[kQuantVec];
  const uint4* src = reinterpret_cast<const uint4*>(x);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const uint4 raw = active ? src[i] : make_uint4(0u, 0u, 0u, 0u);
    const uint32_t w4[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[i * 8 + j * 2] = __uint_as_float(w4[j] << 16);
      v[i * 8 + j * 2 + 1] = __uint_as_float(w4[j] & 0xffff0000u);
    }
  }
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < kQuantVec; ++i) {
    amax = fmaxf(amax, fabsf(v[i]));
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  if (!active) {
    return;
  }
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
  *reinterpret_cast<uint4*>(xq) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
  if ((lane & 1) == 0) {
    *xs = s;
  }
}

__device__ __forceinline__ float load_logit(const float* p) { return *p; }

__device__ __forceinline__ float load_logit(const uint16_t* p) {
  return __uint_as_float(static_cast<uint32_t>(*p) << 16);
}

struct RowBias {
  const float* bias;
  const float* bias_vl;
  const void* input_ids;
  bool ids_64;
  int64_t image_lo;
};

// Routes one token held by one warp: the scaled weights and ids of its top-k.
template <int VPT, int TOPK, typename LT>
__device__ __forceinline__ void route_row(const LT* __restrict__ x, const RowBias& rb,
                                          int row, int experts, int lane, float scaling,
                                          bool renormalize, float (&wk)[TOPK],
                                          int (&ek)[TOPK]) {
  // Issue every load before any arithmetic, the token id and both biases included:
  // the scores and their biases stay in registers, and a picked expert's bias is
  // read back from its lane.
  const bool image_bias = rb.bias_vl != nullptr && rb.image_lo > 0;
  int64_t token = 0;
  if (image_bias) {
    token = rb.ids_64 ? static_cast<const int64_t*>(rb.input_ids)[row]
                      : static_cast<const int32_t*>(rb.input_ids)[row];
  }
  float xv[VPT];
  float bv[VPT];
  float bvl[VPT];
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    const int e = min(lane + j * kWarpSize, experts - 1);
    xv[j] = load_logit(x + e);
    bv[j] = rb.bias[e];
    bvl[j] = image_bias ? rb.bias_vl[e] : 0.0f;
  }
  if (image_bias && token >= rb.image_lo && token < rb.image_lo + 5) {
#pragma unroll
    for (int j = 0; j < VPT; ++j) {
      bv[j] = bvl[j];
    }
  }
  float v[VPT];
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    float s = xv[j] > 20.0f ? xv[j] : __logf(1.0f + __expf(xv[j]));
    s = sqrtf(s);
    if (isnan(s)) {
      s = 0.0f;
    }
    v[j] = lane + j * kWarpSize < experts ? s + bv[j] : kAbsent;
  }

  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    float best = v[0];
    int arg = lane;
#pragma unroll
    for (int j = 1; j < VPT; ++j) {
      if (v[j] > best) {
        best = v[j];
        arg = lane + j * kWarpSize;
      }
    }
#pragma unroll
    for (int o = kWarpSize / 2; o > 0; o >>= 1) {
      const float other = __shfl_xor_sync(0xffffffffu, best, o);
      const int other_arg = __shfl_xor_sync(0xffffffffu, arg, o);
      if (other > best || (other == best && other_arg < arg)) {
        best = other;
        arg = other_arg;
      }
    }
    float picked = 0.0f;
#pragma unroll
    for (int j = 0; j < VPT; ++j) {
      if (lane + j * kWarpSize == arg) {
        v[j] = kCleared;
        picked = best - bv[j];
      }
    }
    const float w = __shfl_sync(0xffffffffu, picked, arg % kWarpSize);
    wk[k] = w;
    ek[k] = arg;
    if (renormalize) {
      sum += w;
    }
  }
  float scale = scaling;
  if (renormalize) {
    const float denom = sum > 0.0f ? sum : 1.0f;
    scale /= denom;
  }
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    wk[k] = wk[k] * scale;
  }
}

template <int VPT, int TOPK, typename LT>
__global__ void __launch_bounds__(kWarps * kWarpSize)
    sqrtsoftplus_topk_kernel(const LT* __restrict__ logits, int64_t logits_stride,
                             const RowBias bias, const bool* __restrict__ is_padding,
                             float* __restrict__ out_w, int32_t* __restrict__ out_ids,
                             int64_t out_stride, int rows, int experts, float scaling,
                             bool renormalize, int shared_id, float shared_weight) {
  const int lane = threadIdx.x % kWarpSize;
  const int row = blockIdx.x * kWarps + threadIdx.x / kWarpSize;
  if (row >= rows) {
    return;
  }
  float* w_out = out_w + row * out_stride;
  int32_t* id_out = out_ids + row * out_stride;
  const int cols = TOPK + (shared_id >= 0 ? 1 : 0);
  // A padding row is routed like any other and its outputs replaced, which keeps
  // the mask load off the routing's critical path.
  const bool pad = is_padding != nullptr && is_padding[row];
  float wk[TOPK];
  int ek[TOPK];
  route_row<VPT, TOPK, LT>(logits + row * logits_stride, bias, row, experts, lane, scaling,
                           renormalize, wk, ek);
  if (pad) {
    if (lane < cols) {
      w_out[lane] = 0.0f;
      id_out[lane] = -1;
    }
    return;
  }
  if (lane == 0) {
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      w_out[k] = wk[k];
      id_out[k] = ek[k];
    }
    if (shared_id >= 0) {
      w_out[TOPK] = shared_weight;
      id_out[TOPK] = shared_id;
    }
  }
}

// Up to kMapTokens tokens in one CTA that also writes the expert route map of the
// small-batch FP8 MoE: distinct experts in first-use order, map[u] = [expert, rows,
// row_0, ...] with the routed rows (flat ids indices) of expert u in increasing
// order, and map[kMapSlots] = [distinct experts]. With x, more CTAs of the launch
// quantize the MoE input per 32-wide group.
constexpr int kMapTokens = 16;
constexpr int kMapSlots = 64;
constexpr int kMapRows = 16;
constexpr int kMapWidth = 2 + kMapRows;

template <int VPT, int TOPK, typename LT>
__global__ void __launch_bounds__(kMapTokens * kWarpSize)
    sqrtsoftplus_route_map_kernel(const LT* __restrict__ logits, int64_t logits_stride,
                                  const RowBias bias, const bool* __restrict__ is_padding,
                                  float* __restrict__ out_w,
                                  int32_t* __restrict__ out_ids, int64_t out_stride,
                                  int32_t* __restrict__ route_map,
                                  const uint16_t* __restrict__ x, int64_t x_stride,
                                  uint8_t* __restrict__ xq, float* __restrict__ xs, int hidden,
                                  int rows, int experts, float scaling, bool renormalize,
                                  int shared_id, float shared_weight) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  // CTAs past the first quantize the MoE input, one 16-value chunk per lane.
  if (blockIdx.x > 0) {
    const int per_row = hidden / kQuantVec;
    const int t = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    const int base = t - lane;
    if (base < rows * per_row) {
      const bool active = t < rows * per_row;
      const int m = active ? t / per_row : 0;
      const int k = active ? (t - m * per_row) * kQuantVec : 0;
      quant16(x + m * x_stride + k, xq + m * hidden + k, xs + m * (hidden / 32) + k / 32, lane,
              active);
    }
    return;
  }

  __shared__ int s_ids[kMapSlots];
  __shared__ int s_uniq[kMapSlots];
  __shared__ unsigned s_first[kMapSlots / kWarpSize];
  const int cols = TOPK + (shared_id >= 0 ? 1 : 0);
  if (warp < rows) {
    float* w_out = out_w + warp * out_stride;
    int32_t* id_out = out_ids + warp * out_stride;
    const bool pad = is_padding != nullptr && is_padding[warp];
    float wk[TOPK];
    int ek[TOPK];
    route_row<VPT, TOPK, LT>(logits + warp * logits_stride, bias, warp, experts, lane,
                             scaling, renormalize, wk, ek);
    if (pad) {
      if (lane < cols) {
        w_out[lane] = 0.0f;
        id_out[lane] = -1;
        s_ids[warp * cols + lane] = -1;
      }
    } else {
      if (lane == 0) {
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
          w_out[k] = wk[k];
          id_out[k] = ek[k];
          s_ids[warp * cols + k] = ek[k];
        }
        if (shared_id >= 0) {
          w_out[TOPK] = shared_weight;
          id_out[TOPK] = shared_id;
          s_ids[warp * cols + TOPK] = shared_id;
        }
      }
    }
  }
  __syncthreads();

  const int r = threadIdx.x;
  const int total = rows * cols;
  int id = -1;
  int earlier = 0;
  int count = 0;
  int first_row = r;
  if (r < total) {
    id = s_ids[r];
    for (int q = 0; q < total; ++q) {
      if (s_ids[q] == id) {
        if (q < r) {
          first_row = earlier == 0 ? q : first_row;
          ++earlier;
        }
        ++count;
      }
    }
  }
  const bool valid = r < total && id >= 0;
  const bool first = valid && earlier == 0;
  const unsigned first_mask = __ballot_sync(0xffffffffu, first);
  if (warp < kMapSlots / kWarpSize && lane == 0) {
    s_first[warp] = first_mask;
  }
  __syncthreads();
  int uniq = __popc(first_mask & ((1u << lane) - 1u));
  for (int w = 0; w < warp && w < kMapSlots / kWarpSize; ++w) {
    uniq += __popc(s_first[w]);
  }
  if (first) {
    s_uniq[r] = uniq;
    route_map[uniq * kMapWidth] = id;
    route_map[uniq * kMapWidth + 1] = count;
  }
  __syncthreads();
  if (valid && earlier < kMapRows) {
    route_map[s_uniq[first_row] * kMapWidth + 2 + earlier] = r;
  }
  if (threadIdx.x == 0) {
    int distinct = 0;
    for (int w = 0; w < kMapSlots / kWarpSize; ++w) {
      distinct += __popc(s_first[w]);
    }
    route_map[kMapSlots * kMapWidth] = distinct;
  }
}

template <int VPT, typename LT>
void launch_topk(int top_k, dim3 grid, musaStream_t stream, const LT* logits,
                 int64_t logits_stride, const RowBias& bias, const bool* is_padding, float* out_w,
                 int32_t* out_ids, int64_t out_stride, int32_t* route_map, const uint16_t* x,
                 int64_t x_stride, uint8_t* xq, float* xs, int hidden, int rows, int experts,
                 float scaling, bool renormalize, int shared_id, float shared_weight) {
  const int quant_ctas =
      x == nullptr ? 0 : (rows * (hidden / kQuantVec) + kMapTokens * kWarpSize - 1) /
                             (kMapTokens * kWarpSize);
  const dim3 map_grid(1 + quant_ctas);
#define TOPK_CASE(K)                                                                           \
  case K:                                                                                      \
    if (route_map != nullptr) {                                                                \
      sqrtsoftplus_route_map_kernel<VPT, K, LT>                                                \
          <<<map_grid, kMapTokens * kWarpSize, 0, stream>>>(                                   \
          logits, logits_stride, bias, is_padding, out_w, out_ids, out_stride, route_map, x,   \
          x_stride, xq, xs, hidden, rows, experts, scaling, renormalize, shared_id,            \
          shared_weight);                                                                      \
    } else {                                                                                   \
      sqrtsoftplus_topk_kernel<VPT, K, LT><<<grid, kWarps * kWarpSize, 0, stream>>>(           \
          logits, logits_stride, bias, is_padding, out_w, out_ids, out_stride, rows, experts,  \
          scaling, renormalize, shared_id, shared_weight);                                     \
    }                                                                                          \
    return;
  switch (top_k) {
    TOPK_CASE(1)
    TOPK_CASE(2)
    TOPK_CASE(3)
    TOPK_CASE(4)
    TOPK_CASE(5)
    TOPK_CASE(6)
    TOPK_CASE(7)
    TOPK_CASE(8)
    default:
      TVM_FFI_THROW(ValueError) << "top_k must be 1..8, got " << top_k;
  }
#undef TOPK_CASE
}

template <typename LT>
void launch_for_experts(const void* logits, int64_t logits_stride, int top_k, dim3 grid,
                        musaStream_t stream, const RowBias& rb, const bool* pp, float* wp,
                        int32_t* ip, int64_t out_stride, int32_t* mp, const uint16_t* xp,
                        int64_t x_stride, uint8_t* qp, float* sp, int hidden, int rows,
                        int experts, float sc, bool renormalize, int sid, float sw) {
  const auto* lp = static_cast<const LT*>(logits);
  if (experts <= 128) {
    launch_topk<4>(top_k, grid, stream, lp, logits_stride, rb, pp, wp, ip, out_stride, mp, xp,
                   x_stride, qp, sp, hidden, rows, experts, sc, renormalize, sid, sw);
  } else if (experts <= 256) {
    launch_topk<8>(top_k, grid, stream, lp, logits_stride, rb, pp, wp, ip, out_stride, mp, xp,
                   x_stride, qp, sp, hidden, rows, experts, sc, renormalize, sid, sw);
  } else if (experts <= 384) {
    launch_topk<12>(top_k, grid, stream, lp, logits_stride, rb, pp, wp, ip, out_stride, mp, xp,
                    x_stride, qp, sp, hidden, rows, experts, sc, renormalize, sid, sw);
  } else if (experts <= 512) {
    launch_topk<16>(top_k, grid, stream, lp, logits_stride, rb, pp, wp, ip, out_stride, mp, xp,
                    x_stride, qp, sp, hidden, rows, experts, sc, renormalize, sid, sw);
  } else {
    TVM_FFI_THROW(ValueError) << "at most 512 experts, got " << experts;
  }
}

}  // namespace

void sqrtsoftplus_topk(ffi::TensorView logits, ffi::TensorView bias, ffi::TensorView bias_vl,
                       ffi::TensorView input_ids, int64_t image_sentinel_lo,
                       ffi::TensorView is_padding, bool has_padding, ffi::TensorView out_w,
                       ffi::TensorView out_ids, ffi::TensorView route_map, bool has_route_map,
                       ffi::TensorView x, ffi::TensorView xq, ffi::TensorView xs, bool quant_x,
                       bool renormalize, double scaling, int64_t shared_id,
                       double shared_weight) {
  ffi::MUSADeviceGuard device_guard(logits.device().device_id);
  const int rows = static_cast<int>(logits.size(0));
  const int experts = static_cast<int>(logits.size(1));
  const int cols = static_cast<int>(out_w.size(1));
  const int top_k = cols - (shared_id >= 0 ? 1 : 0);
  const DLDataType ldt = logits.dtype();
  const bool bf16_logits = ldt.code == kDLBfloat && ldt.bits == 16;
  TVM_FFI_ICHECK(bf16_logits || (ldt.code == kDLFloat && ldt.bits == 32))
      << "logits must be float32 or bfloat16";
  TVM_FFI_ICHECK_EQ(logits.stride(1), 1);
  TVM_FFI_ICHECK_EQ(out_w.stride(0), out_ids.stride(0));
  TVM_FFI_ICHECK_EQ(out_w.stride(1), 1);
  TVM_FFI_ICHECK_EQ(out_ids.stride(1), 1);
  TVM_FFI_ICHECK_EQ(bias.size(0), experts);
  TVM_FFI_ICHECK(top_k >= 1 && top_k <= experts);
  const bool image_bias = image_sentinel_lo > 0;
  if (image_bias) {
    TVM_FFI_ICHECK_EQ(bias_vl.size(0), experts);
    TVM_FFI_ICHECK_GE(input_ids.size(0), rows);
    TVM_FFI_ICHECK(input_ids.dtype().bits == 32 || input_ids.dtype().bits == 64);
  }
  if (has_route_map) {
    TVM_FFI_ICHECK_LE(rows, kMapTokens);
    TVM_FFI_ICHECK_LE(rows * cols, kMapSlots);
    TVM_FFI_ICHECK_GE(route_map.size(0), (kMapSlots + 1) * kMapWidth);
  }
  int hidden = 0;
  if (quant_x) {
    TVM_FFI_ICHECK(has_route_map) << "the activation quant rides on the route-map kernel";
    hidden = static_cast<int>(x.size(1));
    TVM_FFI_ICHECK_EQ(x.size(0), rows);
    TVM_FFI_ICHECK_EQ(x.stride(1), 1);
    TVM_FFI_ICHECK_EQ(hidden % 32, 0);
    TVM_FFI_ICHECK_EQ(x.stride(0) % 8, 0);
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16, 0);
    TVM_FFI_ICHECK_EQ(xq.size(0), rows);
    TVM_FFI_ICHECK_EQ(xq.size(1), hidden);
    TVM_FFI_ICHECK_EQ(xq.stride(0), hidden);
    TVM_FFI_ICHECK_EQ(xs.size(1), hidden / 32);
    TVM_FFI_ICHECK_EQ(xs.stride(0), hidden / 32);
  }
  if (rows == 0) {
    return;
  }
  const dim3 grid((rows + kWarps - 1) / kWarps);
  const musaStream_t stream = static_cast<musaStream_t>(
      TVMFFIEnvGetStream(logits.device().device_type, logits.device().device_id));
  const RowBias rb{static_cast<const float*>(bias.data_ptr()),
                   image_bias ? static_cast<const float*>(bias_vl.data_ptr()) : nullptr,
                   image_bias ? input_ids.data_ptr() : nullptr,
                   image_bias && input_ids.dtype().bits == 64, image_sentinel_lo};
  const bool* pp = has_padding ? static_cast<const bool*>(is_padding.data_ptr()) : nullptr;
  auto* wp = static_cast<float*>(out_w.data_ptr());
  auto* ip = static_cast<int32_t*>(out_ids.data_ptr());
  auto* mp = has_route_map ? static_cast<int32_t*>(route_map.data_ptr()) : nullptr;
  const auto* xp = quant_x ? static_cast<const uint16_t*>(x.data_ptr()) : nullptr;
  const int64_t x_stride = quant_x ? x.stride(0) : 0;
  auto* qp = quant_x ? static_cast<uint8_t*>(xq.data_ptr()) : nullptr;
  auto* sp = quant_x ? static_cast<float*>(xs.data_ptr()) : nullptr;
  const int sid = static_cast<int>(shared_id);
  const float sw = static_cast<float>(shared_weight);
  const float sc = static_cast<float>(scaling);
  if (bf16_logits) {
    launch_for_experts<uint16_t>(logits.data_ptr(), logits.stride(0), top_k, grid, stream, rb, pp,
                                 wp, ip, out_w.stride(0), mp, xp, x_stride, qp, sp, hidden, rows,
                                 experts, sc, renormalize, sid, sw);
  } else {
    launch_for_experts<float>(logits.data_ptr(), logits.stride(0), top_k, grid, stream, rb, pp, wp,
                              ip, out_w.stride(0), mp, xp, x_stride, qp, sp, hidden, rows, experts,
                              sc, renormalize, sid, sw);
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess) << "sqrtsoftplus_topk failed: " << musaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sqrtsoftplus_topk, sqrtsoftplus_topk);
