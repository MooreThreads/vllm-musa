#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/musa/device_guard.h>
#include <tvm/ffi/function.h>

namespace ffi = tvm::ffi;

constexpr DLDataType dl_float16{kDLFloat, 16, 1};
constexpr DLDataType dl_float32{kDLFloat, 32, 1};
constexpr DLDataType dl_bfloat16{kDLBfloat, 16, 1};
constexpr DLDataType dl_int64{kDLInt, 64, 1};

inline bool dtype_equal(DLDataType lhs, DLDataType rhs) {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

inline musaStream_t get_stream(DLDevice device) {
  return static_cast<musaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));
}

#define TORCH_CHECK(cond, ...) TVM_FFI_ICHECK(cond)
#define CHECK_MUSA(x) TVM_FFI_ICHECK_EQ((x).device().device_type, kDLExtDev)

#if defined(__MUSA_ARCH__)
#define VLLM_MUSA_ARCH __MUSA_ARCH__
#else
#define VLLM_MUSA_ARCH 0
#endif

#define VLLM_MUSA_ARCH_MP31 (VLLM_MUSA_ARCH == 310)

#ifndef VLLM_MUSA_ROPE_STORE_HINT
#define VLLM_MUSA_ROPE_STORE_HINT 0
#endif

template <typename T>
__device__ __forceinline__ float to_float(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float to_float<half>(half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float<__mt_bfloat16>(__mt_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ half from_float<half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __mt_bfloat16 from_float<__mt_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

__device__ __forceinline__ void store_half2(half2* ptr, half2 value) {
#if VLLM_MUSA_ROPE_STORE_HINT == 1
  __stwb(ptr, value);
#elif VLLM_MUSA_ROPE_STORE_HINT == 2
  __stcg(ptr, value);
#elif VLLM_MUSA_ROPE_STORE_HINT == 3
  __stcs(ptr, value);
#else
  *ptr = value;
#endif
}

__device__ __forceinline__ void store_bfloat162(__mt_bfloat162* ptr, __mt_bfloat162 value) {
#if VLLM_MUSA_ROPE_STORE_HINT == 1
  __stwb(ptr, value);
#elif VLLM_MUSA_ROPE_STORE_HINT == 2
  __stcg(ptr, value);
#elif VLLM_MUSA_ROPE_STORE_HINT == 3
  __stcs(ptr, value);
#else
  *ptr = value;
#endif
}

template <typename T, bool IS_NEOX>
__device__ __forceinline__ void apply_one(
    T* data,
    const T* cache,
    int base,
    int head_stride,
    int dim_stride,
    int head_idx,
    int rot_offset,
    int half_rot_dim) {
  const int x_idx = IS_NEOX ? rot_offset : 2 * rot_offset;
  const int y_idx = IS_NEOX ? half_rot_dim + rot_offset : x_idx + 1;
  const int head_base = base + head_idx * head_stride;
  const int x_offset = head_base + x_idx * dim_stride;
  const int y_offset = head_base + y_idx * dim_stride;

  const float x = to_float(data[x_offset]);
  const float y = to_float(data[y_offset]);
  const float c = to_float(cache[rot_offset]);
  const float s = to_float(cache[half_rot_dim + rot_offset]);
  data[x_offset] = from_float<T>(x * c - y * s);
  data[y_offset] = from_float<T>(y * c + x * s);
}

template <typename T, bool IS_NEOX, bool HAS_KEY>
__global__ void rotary_embedding_decode_kernel(
    const int64_t* __restrict__ positions,
    T* __restrict__ query,
    T* __restrict__ key,
    const T* __restrict__ cos_sin_cache,
    int num_heads,
    int num_kv_heads,
    int rot_dim,
    int seq_len,
    int query_batch_stride,
    int key_batch_stride,
    int query_stride,
    int key_stride,
    int query_head_stride,
    int key_head_stride,
    int query_dim_stride,
    int key_dim_stride) {
  const int token_idx = blockIdx.x;
  const int pair_idx = blockIdx.y * blockDim.x + threadIdx.x;
  const int half_rot_dim = rot_dim / 2;
  const int pos = static_cast<int>(positions[token_idx]);
  const T* cache = cos_sin_cache + pos * rot_dim;

  int q_token_base;
  int k_token_base;
  if (seq_len > 0) {
    const int batch_idx = token_idx / seq_len;
    const int seq_idx = token_idx - batch_idx * seq_len;
    q_token_base = batch_idx * query_batch_stride + seq_idx * query_stride;
    k_token_base = batch_idx * key_batch_stride + seq_idx * key_stride;
  } else {
    q_token_base = token_idx * query_stride;
    k_token_base = token_idx * key_stride;
  }

  if (pair_idx < num_heads * half_rot_dim) {
    const int q_head_idx = pair_idx / half_rot_dim;
    const int q_rot_offset = pair_idx - q_head_idx * half_rot_dim;
    apply_one<T, IS_NEOX>(
        query,
        cache,
        q_token_base,
        query_head_stride,
        query_dim_stride,
        q_head_idx,
        q_rot_offset,
        half_rot_dim);
  }
  if constexpr (HAS_KEY) {
  if (pair_idx < num_kv_heads * half_rot_dim) {
    const int k_head_idx = pair_idx / half_rot_dim;
    const int k_rot_offset = pair_idx - k_head_idx * half_rot_dim;
    apply_one<T, IS_NEOX>(
        key,
        cache,
        k_token_base,
        key_head_stride,
        key_dim_stride,
        k_head_idx,
        k_rot_offset,
        half_rot_dim);
  }
  }
}

template <typename T, bool IS_NEOX, bool HAS_KEY>
__global__ void rotary_embedding_prefill_kernel(
    const int64_t* __restrict__ positions,
    T* __restrict__ query,
    T* __restrict__ key,
    const T* __restrict__ cos_sin_cache,
    int num_heads,
    int num_kv_heads,
    int rot_dim,
    int seq_len,
    int query_batch_stride,
    int key_batch_stride,
    int query_stride,
    int key_stride,
    int query_head_stride,
    int key_head_stride,
    int query_dim_stride,
    int key_dim_stride) {
  extern __shared__ unsigned char smem[];
  T* cache_shared = reinterpret_cast<T*>(smem);

  const int token_idx = blockIdx.x;
  const int half_rot_dim = rot_dim / 2;
  const int max_head_pairs = max(num_heads, num_kv_heads) * half_rot_dim;
  const int pos = static_cast<int>(positions[token_idx]);
  const T* cache = cos_sin_cache + pos * rot_dim;

  for (int i = threadIdx.x; i < rot_dim; i += blockDim.x) {
    cache_shared[i] = cache[i];
  }
  __syncthreads();

  int q_token_base;
  int k_token_base;
  if (seq_len > 0) {
    const int batch_idx = token_idx / seq_len;
    const int seq_idx = token_idx - batch_idx * seq_len;
    q_token_base = batch_idx * query_batch_stride + seq_idx * query_stride;
    k_token_base = batch_idx * key_batch_stride + seq_idx * key_stride;
  } else {
    q_token_base = token_idx * query_stride;
    k_token_base = token_idx * key_stride;
  }

  for (int pair_idx = threadIdx.x; pair_idx < max_head_pairs; pair_idx += blockDim.x) {
    if (pair_idx < num_heads * half_rot_dim) {
      const int q_head_idx = pair_idx / half_rot_dim;
      const int q_rot_offset = pair_idx - q_head_idx * half_rot_dim;
      apply_one<T, IS_NEOX>(
          query,
          cache_shared,
          q_token_base,
          query_head_stride,
          query_dim_stride,
          q_head_idx,
          q_rot_offset,
          half_rot_dim);
    }
    if constexpr (HAS_KEY) {
    if (pair_idx < num_kv_heads * half_rot_dim) {
      const int k_head_idx = pair_idx / half_rot_dim;
      const int k_rot_offset = pair_idx - k_head_idx * half_rot_dim;
      apply_one<T, IS_NEOX>(
          key,
          cache_shared,
          k_token_base,
          key_head_stride,
          key_dim_stride,
          k_head_idx,
          k_rot_offset,
          half_rot_dim);
    }
    }
  }
}

template <typename T, bool IS_NEOX, bool HAS_KEY>
__global__ void rotary_embedding_prefill_direct_kernel(
    const int64_t* __restrict__ positions,
    T* __restrict__ query,
    T* __restrict__ key,
    const T* __restrict__ cos_sin_cache,
    int num_heads,
    int num_kv_heads,
    int rot_dim,
    int seq_len,
    int query_batch_stride,
    int key_batch_stride,
    int query_stride,
    int key_stride,
    int query_head_stride,
    int key_head_stride,
    int query_dim_stride,
    int key_dim_stride) {
  const int token_idx = blockIdx.x;
  const int half_rot_dim = rot_dim / 2;
  const int max_head_pairs = max(num_heads, num_kv_heads) * half_rot_dim;
  const int pos = static_cast<int>(positions[token_idx]);
  const T* cache = cos_sin_cache + pos * rot_dim;

  int q_token_base;
  int k_token_base;
  if (seq_len > 0) {
    const int batch_idx = token_idx / seq_len;
    const int seq_idx = token_idx - batch_idx * seq_len;
    q_token_base = batch_idx * query_batch_stride + seq_idx * query_stride;
    k_token_base = batch_idx * key_batch_stride + seq_idx * key_stride;
  } else {
    q_token_base = token_idx * query_stride;
    k_token_base = token_idx * key_stride;
  }

  for (int pair_idx = threadIdx.x; pair_idx < max_head_pairs; pair_idx += blockDim.x) {
    if (pair_idx < num_heads * half_rot_dim) {
      const int q_head_idx = pair_idx / half_rot_dim;
      const int q_rot_offset = pair_idx - q_head_idx * half_rot_dim;
      apply_one<T, IS_NEOX>(
          query, cache, q_token_base, query_head_stride, query_dim_stride, q_head_idx, q_rot_offset, half_rot_dim);
    }
    if constexpr (HAS_KEY) {
    if (pair_idx < num_kv_heads * half_rot_dim) {
      const int k_head_idx = pair_idx / half_rot_dim;
      const int k_rot_offset = pair_idx - k_head_idx * half_rot_dim;
      apply_one<T, IS_NEOX>(
          key, cache, k_token_base, key_head_stride, key_dim_stride, k_head_idx, k_rot_offset, half_rot_dim);
    }
    }
  }
}

template <typename T, int ROT_DIM, bool HAS_KEY, bool HAS_SEQ>
__global__ void rotary_embedding_prefill_neox_static_kernel(
    const int64_t* __restrict__ positions,
    T* __restrict__ query,
    T* __restrict__ key,
    const T* __restrict__ cos_sin_cache,
    int num_heads,
    int num_kv_heads,
    int seq_len,
    int query_batch_stride,
    int key_batch_stride,
    int query_stride,
    int key_stride,
    int query_head_stride,
    int key_head_stride) {
  constexpr int half_rot_dim = ROT_DIM / 2;
  const int token_idx = blockIdx.x;
  const int max_head_pairs = max(num_heads, num_kv_heads) * half_rot_dim;
  const int pos = static_cast<int>(positions[token_idx]);
  const T* cache = cos_sin_cache + pos * ROT_DIM;

  int q_token_base;
  int k_token_base;
  if constexpr (HAS_SEQ) {
    const int batch_idx = token_idx / seq_len;
    const int seq_idx = token_idx - batch_idx * seq_len;
    q_token_base = batch_idx * query_batch_stride + seq_idx * query_stride;
    k_token_base = batch_idx * key_batch_stride + seq_idx * key_stride;
  } else {
    q_token_base = token_idx * query_stride;
    k_token_base = token_idx * key_stride;
  }

  for (int pair_idx = threadIdx.x; pair_idx < max_head_pairs; pair_idx += blockDim.x) {
  if (pair_idx < num_heads * half_rot_dim) {
    const int q_head_idx = pair_idx / half_rot_dim;
    const int q_rot_offset = pair_idx & (half_rot_dim - 1);
    apply_one<T, true>(query, cache, q_token_base, query_head_stride, 1, q_head_idx, q_rot_offset, half_rot_dim);
  }
  if constexpr (HAS_KEY) {
  if (pair_idx < num_kv_heads * half_rot_dim) {
    const int k_head_idx = pair_idx / half_rot_dim;
    const int k_rot_offset = pair_idx & (half_rot_dim - 1);
    apply_one<T, true>(key, cache, k_token_base, key_head_stride, 1, k_head_idx, k_rot_offset, half_rot_dim);
  }
  }
  }
}

template <int ROT_DIM>
__device__ __forceinline__ void apply_neox_fp16_half2(
    half* __restrict__ data,
    const half* __restrict__ cache,
    int base,
    int head_stride,
    int pair_idx) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  const int head_idx = pair_idx / HALF_ROT_DIM;
  const int rot = pair_idx & (HALF_ROT_DIM - 1);
  const int head_base = base + head_idx * head_stride;
  const half2 c = *reinterpret_cast<const half2*>(cache + rot);
  const half2 s = *reinterpret_cast<const half2*>(cache + HALF_ROT_DIM + rot);
  const half2 x = *reinterpret_cast<const half2*>(data + head_base + rot);
  const half2 y = *reinterpret_cast<const half2*>(data + head_base + HALF_ROT_DIM + rot);
  const half2 out_x = __hfma2(y, __hneg2(s), __hmul2(x, c));
  const half2 out_y = __hfma2(x, s, __hmul2(y, c));
  store_half2(reinterpret_cast<half2*>(data + head_base + rot), out_x);
  store_half2(reinterpret_cast<half2*>(data + head_base + HALF_ROT_DIM + rot), out_y);
}

template <int ROT_DIM>
__device__ __forceinline__ void apply_neox_fp16_half2_hs128(
    half* __restrict__ data,
    const half* __restrict__ cache,
    int base,
    int pair_idx) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  const int head_idx = pair_idx / HALF_ROT_DIM;
  const int rot = pair_idx & (HALF_ROT_DIM - 1);
  const int head_base = base + (head_idx << 7);
  const half2 c = *reinterpret_cast<const half2*>(cache + rot);
  const half2 s = *reinterpret_cast<const half2*>(cache + HALF_ROT_DIM + rot);
  const half2 x = *reinterpret_cast<const half2*>(data + head_base + rot);
  const half2 y = *reinterpret_cast<const half2*>(data + head_base + HALF_ROT_DIM + rot);
  const half2 out_x = __hfma2(y, __hneg2(s), __hmul2(x, c));
  const half2 out_y = __hfma2(x, s, __hmul2(y, c));
  store_half2(reinterpret_cast<half2*>(data + head_base + rot), out_x);
  store_half2(reinterpret_cast<half2*>(data + head_base + HALF_ROT_DIM + rot), out_y);
}

template <int ROT_DIM>
__device__ __forceinline__ void apply_neox_bf16_bfloat162(
    __mt_bfloat16* __restrict__ data,
    const __mt_bfloat16* __restrict__ cache,
    int base,
    int head_stride,
    int pair_idx) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  const int head_idx = pair_idx / HALF_ROT_DIM;
  const int rot = pair_idx & (HALF_ROT_DIM - 1);
  const int head_base = base + head_idx * head_stride;
  const __mt_bfloat162 c = *reinterpret_cast<const __mt_bfloat162*>(cache + rot);
  const __mt_bfloat162 s = *reinterpret_cast<const __mt_bfloat162*>(cache + HALF_ROT_DIM + rot);
  const __mt_bfloat162 x = *reinterpret_cast<const __mt_bfloat162*>(data + head_base + rot);
  const __mt_bfloat162 y = *reinterpret_cast<const __mt_bfloat162*>(data + head_base + HALF_ROT_DIM + rot);
  const __mt_bfloat162 out_x = __hfma2(y, __hneg2(s), __hmul2(x, c));
  const __mt_bfloat162 out_y = __hfma2(x, s, __hmul2(y, c));
  store_bfloat162(reinterpret_cast<__mt_bfloat162*>(data + head_base + rot), out_x);
  store_bfloat162(reinterpret_cast<__mt_bfloat162*>(data + head_base + HALF_ROT_DIM + rot), out_y);
}

template <int ROT_DIM>
__device__ __forceinline__ void apply_neox_bf16_bfloat162_hs128(
    __mt_bfloat16* __restrict__ data,
    const __mt_bfloat16* __restrict__ cache,
    int base,
    int pair_idx) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  const int head_idx = pair_idx / HALF_ROT_DIM;
  const int rot = pair_idx & (HALF_ROT_DIM - 1);
  const int head_base = base + (head_idx << 7);
  const __mt_bfloat162 c = *reinterpret_cast<const __mt_bfloat162*>(cache + rot);
  const __mt_bfloat162 s = *reinterpret_cast<const __mt_bfloat162*>(cache + HALF_ROT_DIM + rot);
  const __mt_bfloat162 x = *reinterpret_cast<const __mt_bfloat162*>(data + head_base + rot);
  const __mt_bfloat162 y = *reinterpret_cast<const __mt_bfloat162*>(data + head_base + HALF_ROT_DIM + rot);
  const __mt_bfloat162 out_x = __hfma2(y, __hneg2(s), __hmul2(x, c));
  const __mt_bfloat162 out_y = __hfma2(x, s, __hmul2(y, c));
  store_bfloat162(reinterpret_cast<__mt_bfloat162*>(data + head_base + rot), out_x);
  store_bfloat162(reinterpret_cast<__mt_bfloat162*>(data + head_base + HALF_ROT_DIM + rot), out_y);
}

template <int ROT_DIM, bool HAS_KEY>
__global__ void rotary_embedding_prefill_neox_fp16_h2_kernel(
    const int64_t* __restrict__ positions,
    half* __restrict__ query,
    half* __restrict__ key,
    const half* __restrict__ cos_sin_cache,
    int num_heads,
    int num_kv_heads,
    int query_stride,
    int key_stride,
    int query_head_stride,
    int key_head_stride) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  const int token_idx = blockIdx.x;
  const int pos = static_cast<int>(positions[token_idx]);
  const half* cache = cos_sin_cache + pos * ROT_DIM;
  const int q_token_base = token_idx * query_stride;
  const int k_token_base = token_idx * key_stride;
  const int q_pairs = num_heads * HALF_ROT_DIM;
  const int k_pairs = num_kv_heads * HALF_ROT_DIM;
  const int max_pairs = HAS_KEY ? max(q_pairs, k_pairs) : q_pairs;

  for (int pair_idx = threadIdx.x * 2; pair_idx < max_pairs; pair_idx += blockDim.x * 2) {
    if (pair_idx < q_pairs) {
      apply_neox_fp16_half2<ROT_DIM>(query, cache, q_token_base, query_head_stride, pair_idx);
    }
    if constexpr (HAS_KEY) {
    if (pair_idx < k_pairs) {
      apply_neox_fp16_half2<ROT_DIM>(key, cache, k_token_base, key_head_stride, pair_idx);
    }
    }
  }
}

template <int ROT_DIM, bool HAS_KEY>
__global__ void rotary_embedding_decode_neox_fp16_h2_split_kernel(
    const int64_t* __restrict__ positions,
    half* __restrict__ query,
    half* __restrict__ key,
    const half* __restrict__ cos_sin_cache,
    int num_heads,
    int num_kv_heads,
    int query_stride,
    int key_stride,
    int query_head_stride,
    int key_head_stride) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  const int token_idx = blockIdx.x;
  const int pair_idx = (blockIdx.y * blockDim.x + threadIdx.x) * 2;
  const int q_pairs = num_heads * HALF_ROT_DIM;
  const int k_pairs = num_kv_heads * HALF_ROT_DIM;
  const int pos = static_cast<int>(positions[token_idx]);
  const half* cache = cos_sin_cache + pos * ROT_DIM;
  const int q_token_base = token_idx * query_stride;
  const int k_token_base = token_idx * key_stride;

  if (pair_idx < q_pairs) {
    apply_neox_fp16_half2<ROT_DIM>(query, cache, q_token_base, query_head_stride, pair_idx);
  }
  if constexpr (HAS_KEY) {
  if (pair_idx < k_pairs) {
    apply_neox_fp16_half2<ROT_DIM>(key, cache, k_token_base, key_head_stride, pair_idx);
  }
  }
}

template <int ROT_DIM>
__global__ void rotary_embedding_decode_neox_fp16_h2_32x8_hs128_kernel(
    const int64_t* __restrict__ positions,
    half* __restrict__ query,
    half* __restrict__ key,
    const half* __restrict__ cos_sin_cache) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  constexpr int Q_PAIRS = 32 * HALF_ROT_DIM;
  constexpr int K_PAIRS = 8 * HALF_ROT_DIM;
  const int token_idx = blockIdx.x;
  const int pair_idx = (blockIdx.y * blockDim.x + threadIdx.x) * 2;
  const int pos = static_cast<int>(positions[token_idx]);
  const half* cache = cos_sin_cache + pos * ROT_DIM;
  const int q_token_base = token_idx << 12;
  const int k_token_base = token_idx << 10;

  if (pair_idx < Q_PAIRS) {
    apply_neox_fp16_half2_hs128<ROT_DIM>(query, cache, q_token_base, pair_idx);
  }
  if (pair_idx < K_PAIRS) {
    apply_neox_fp16_half2_hs128<ROT_DIM>(key, cache, k_token_base, pair_idx);
  }
}

__global__ void rotary_embedding_decode_neox_fp16_h2_rd64_32x8_hs128_kernel(
    const int64_t* __restrict__ positions,
    half* __restrict__ query,
    half* __restrict__ key,
    const half* __restrict__ cos_sin_cache) {
  const int token_idx = blockIdx.x;
  const int pair_idx = (blockIdx.y * blockDim.x + threadIdx.x) * 2;
  const int pos = static_cast<int>(positions[token_idx]);
  const half* cache = cos_sin_cache + pos * 64;
  const int q_token_base = token_idx << 12;
  const int k_token_base = token_idx << 10;

  apply_neox_fp16_half2_hs128<64>(query, cache, q_token_base, pair_idx);
  if (blockIdx.y == 0) {
    apply_neox_fp16_half2_hs128<64>(key, cache, k_token_base, pair_idx);
  }
}

template <int ROT_DIM, int TOKENS_PER_BLOCK>
__global__ void rotary_embedding_decode_neox_fp16_h2_32x8_hs128_mtile_kernel(
    const int64_t* __restrict__ positions,
    half* __restrict__ query,
    half* __restrict__ key,
    const half* __restrict__ cos_sin_cache,
    int num_tokens) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  constexpr int Q_PAIRS = 32 * HALF_ROT_DIM;
  constexpr int K_PAIRS = 8 * HALF_ROT_DIM;
  const int token_base = blockIdx.x * TOKENS_PER_BLOCK;
  const int pair_idx = (blockIdx.y * blockDim.x + threadIdx.x) * 2;

#pragma unroll
  for (int t = 0; t < TOKENS_PER_BLOCK; ++t) {
    const int token_idx = token_base + t;
    if (token_idx >= num_tokens) {
      return;
    }
    const int pos = static_cast<int>(positions[token_idx]);
    const half* cache = cos_sin_cache + pos * ROT_DIM;
    const int q_token_base = token_idx << 12;
    const int k_token_base = token_idx << 10;
    if (pair_idx < Q_PAIRS) {
      apply_neox_fp16_half2_hs128<ROT_DIM>(query, cache, q_token_base, pair_idx);
    }
    if (pair_idx < K_PAIRS) {
      apply_neox_fp16_half2_hs128<ROT_DIM>(key, cache, k_token_base, pair_idx);
    }
  }
}

template <int TOKENS_PER_BLOCK>
__global__ void rotary_embedding_decode_neox_fp16_h2_rd64_32x8_hs128_mtile_kernel(
    const int64_t* __restrict__ positions,
    half* __restrict__ query,
    half* __restrict__ key,
    const half* __restrict__ cos_sin_cache,
    int num_tokens) {
  const int token_base = blockIdx.x * TOKENS_PER_BLOCK;
  const int pair_idx = (blockIdx.y * blockDim.x + threadIdx.x) * 2;

#pragma unroll
  for (int t = 0; t < TOKENS_PER_BLOCK; ++t) {
    const int token_idx = token_base + t;
    if (token_idx >= num_tokens) {
      return;
    }
    const int pos = static_cast<int>(positions[token_idx]);
    const half* cache = cos_sin_cache + pos * 64;
    const int q_token_base = token_idx << 12;
    const int k_token_base = token_idx << 10;
    apply_neox_fp16_half2_hs128<64>(query, cache, q_token_base, pair_idx);
    if (blockIdx.y == 0) {
      apply_neox_fp16_half2_hs128<64>(key, cache, k_token_base, pair_idx);
    }
  }
}

template <int ROT_DIM, bool HAS_KEY>
__global__ void rotary_embedding_prefill_neox_bf16_h2_kernel(
    const int64_t* __restrict__ positions,
    __mt_bfloat16* __restrict__ query,
    __mt_bfloat16* __restrict__ key,
    const __mt_bfloat16* __restrict__ cos_sin_cache,
    int num_heads,
    int num_kv_heads,
    int query_stride,
    int key_stride,
    int query_head_stride,
    int key_head_stride) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  const int token_idx = blockIdx.x;
  const int pos = static_cast<int>(positions[token_idx]);
  const __mt_bfloat16* cache = cos_sin_cache + pos * ROT_DIM;
  const int q_token_base = token_idx * query_stride;
  const int k_token_base = token_idx * key_stride;
  const int q_pairs = num_heads * HALF_ROT_DIM;
  const int k_pairs = num_kv_heads * HALF_ROT_DIM;
  const int max_pairs = HAS_KEY ? max(q_pairs, k_pairs) : q_pairs;

  for (int pair_idx = threadIdx.x * 2; pair_idx < max_pairs; pair_idx += blockDim.x * 2) {
    if (pair_idx < q_pairs) {
      apply_neox_bf16_bfloat162<ROT_DIM>(query, cache, q_token_base, query_head_stride, pair_idx);
    }
    if constexpr (HAS_KEY) {
    if (pair_idx < k_pairs) {
      apply_neox_bf16_bfloat162<ROT_DIM>(key, cache, k_token_base, key_head_stride, pair_idx);
    }
    }
  }
}

template <int ROT_DIM, bool HAS_KEY>
__global__ void rotary_embedding_decode_neox_bf16_h2_split_kernel(
    const int64_t* __restrict__ positions,
    __mt_bfloat16* __restrict__ query,
    __mt_bfloat16* __restrict__ key,
    const __mt_bfloat16* __restrict__ cos_sin_cache,
    int num_heads,
    int num_kv_heads,
    int query_stride,
    int key_stride,
    int query_head_stride,
    int key_head_stride) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  const int token_idx = blockIdx.x;
  const int pair_idx = (blockIdx.y * blockDim.x + threadIdx.x) * 2;
  const int q_pairs = num_heads * HALF_ROT_DIM;
  const int k_pairs = num_kv_heads * HALF_ROT_DIM;
  const int pos = static_cast<int>(positions[token_idx]);
  const __mt_bfloat16* cache = cos_sin_cache + pos * ROT_DIM;
  const int q_token_base = token_idx * query_stride;
  const int k_token_base = token_idx * key_stride;

  if (pair_idx < q_pairs) {
    apply_neox_bf16_bfloat162<ROT_DIM>(query, cache, q_token_base, query_head_stride, pair_idx);
  }
  if constexpr (HAS_KEY) {
  if (pair_idx < k_pairs) {
    apply_neox_bf16_bfloat162<ROT_DIM>(key, cache, k_token_base, key_head_stride, pair_idx);
  }
  }
}

template <int ROT_DIM>
__global__ void rotary_embedding_decode_neox_bf16_h2_32x8_hs128_kernel(
    const int64_t* __restrict__ positions,
    __mt_bfloat16* __restrict__ query,
    __mt_bfloat16* __restrict__ key,
    const __mt_bfloat16* __restrict__ cos_sin_cache) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  constexpr int Q_PAIRS = 32 * HALF_ROT_DIM;
  constexpr int K_PAIRS = 8 * HALF_ROT_DIM;
  const int token_idx = blockIdx.x;
  const int pair_idx = (blockIdx.y * blockDim.x + threadIdx.x) * 2;
  const int pos = static_cast<int>(positions[token_idx]);
  const __mt_bfloat16* cache = cos_sin_cache + pos * ROT_DIM;
  const int q_token_base = token_idx << 12;
  const int k_token_base = token_idx << 10;

  if (pair_idx < Q_PAIRS) {
    apply_neox_bf16_bfloat162_hs128<ROT_DIM>(query, cache, q_token_base, pair_idx);
  }
  if (pair_idx < K_PAIRS) {
    apply_neox_bf16_bfloat162_hs128<ROT_DIM>(key, cache, k_token_base, pair_idx);
  }
}

template <int ROT_DIM>
__global__ void rotary_embedding_prefill_neox_fp16_q_h2x2_kernel(
    const int64_t* __restrict__ positions,
    half* __restrict__ query,
    const half* __restrict__ cos_sin_cache,
    int num_heads,
    int query_stride,
    int query_head_stride) {
  constexpr int HALF_ROT_DIM = ROT_DIM / 2;
  const int token_idx = blockIdx.x;
  const int pos = static_cast<int>(positions[token_idx]);
  const half* cache = cos_sin_cache + pos * ROT_DIM;
  const int q_token_base = token_idx * query_stride;
  const int q_pairs = num_heads * HALF_ROT_DIM;

  for (int pair_idx = threadIdx.x * 4; pair_idx < q_pairs; pair_idx += blockDim.x * 4) {
    apply_neox_fp16_half2<ROT_DIM>(query, cache, q_token_base, query_head_stride, pair_idx);
    const int pair_idx_1 = pair_idx + 2;
    if (pair_idx_1 < q_pairs) {
      apply_neox_fp16_half2<ROT_DIM>(query, cache, q_token_base, query_head_stride, pair_idx_1);
    }
  }
}

template <typename T, bool IS_NEOX, bool HAS_KEY>
void launch_rotary_embedding(
    ffi::TensorView positions,
    ffi::TensorView query,
    ffi::TensorView key,
    ffi::TensorView cos_sin_cache,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int rot_dim,
    int seq_len,
    int query_batch_stride,
    int key_batch_stride,
    int query_stride,
    int key_stride,
    int query_head_stride,
    int key_head_stride,
    int query_dim_stride,
    int key_dim_stride,
    musaStream_t stream) {
  constexpr int kDecodeThreshold = 64;
  const int half_rot_dim = rot_dim / 2;
  const int max_head_pairs = std::max(num_heads, num_kv_heads) * half_rot_dim;
  if constexpr (IS_NEOX) {
    if constexpr (std::is_same<T, half>::value && HAS_KEY) {
      if (seq_len == 0 && query_dim_stride == 1 && key_dim_stride == 1 &&
          (rot_dim == 64 || rot_dim == 128) &&
          (query_stride & 1) == 0 && (key_stride & 1) == 0 &&
          (query_head_stride & 1) == 0 && (key_head_stride & 1) == 0) {
        if (num_tokens <= 128) {
          const int block_threads = num_tokens < 128 ? 128 : 256;
          const int h2_pairs = (max_head_pairs + 1) / 2;
          const dim3 grid(num_tokens, (h2_pairs + block_threads - 1) / block_threads);
          const dim3 block(block_threads);
          if (VLLM_MUSA_ARCH_MP31 && num_heads == 32 && num_kv_heads == 8 && query_stride == 4096 &&
              key_stride == 1024 && query_head_stride == 128 && key_head_stride == 128) {
            if (false && rot_dim == 64 && num_tokens > 1 && num_tokens <= 64) {
              constexpr int token_tile = 8;
              const dim3 mtile_grid((num_tokens + token_tile - 1) / token_tile, (h2_pairs + block_threads - 1) / block_threads);
              rotary_embedding_decode_neox_fp16_h2_rd64_32x8_hs128_mtile_kernel<token_tile><<<mtile_grid, block, 0, stream>>>(
                  static_cast<const int64_t*>(positions.data_ptr()),
                  static_cast<half*>(query.data_ptr()),
                  static_cast<half*>(key.data_ptr()),
                  static_cast<const half*>(cos_sin_cache.data_ptr()),
                  num_tokens);
            } else if (rot_dim == 64) {
              rotary_embedding_decode_neox_fp16_h2_32x8_hs128_kernel<64><<<grid, block, 0, stream>>>(
                  static_cast<const int64_t*>(positions.data_ptr()),
                  static_cast<half*>(query.data_ptr()),
                  static_cast<half*>(key.data_ptr()),
                  static_cast<const half*>(cos_sin_cache.data_ptr()));
            } else {
              rotary_embedding_decode_neox_fp16_h2_32x8_hs128_kernel<128><<<grid, block, 0, stream>>>(
                  static_cast<const int64_t*>(positions.data_ptr()),
                  static_cast<half*>(query.data_ptr()),
                  static_cast<half*>(key.data_ptr()),
                  static_cast<const half*>(cos_sin_cache.data_ptr()));
            }
          } else if (rot_dim == 64) {
            rotary_embedding_decode_neox_fp16_h2_split_kernel<64, true><<<grid, block, 0, stream>>>(
                static_cast<const int64_t*>(positions.data_ptr()),
                static_cast<half*>(query.data_ptr()),
                static_cast<half*>(key.data_ptr()),
                static_cast<const half*>(cos_sin_cache.data_ptr()),
                num_heads,
                num_kv_heads,
                query_stride,
                key_stride,
                query_head_stride,
                key_head_stride);
          } else {
            rotary_embedding_decode_neox_fp16_h2_split_kernel<128, true><<<grid, block, 0, stream>>>(
                static_cast<const int64_t*>(positions.data_ptr()),
                static_cast<half*>(query.data_ptr()),
                static_cast<half*>(key.data_ptr()),
                static_cast<const half*>(cos_sin_cache.data_ptr()),
                num_heads,
                num_kv_heads,
                query_stride,
                key_stride,
                query_head_stride,
                key_head_stride);
          }
          return;
        }
        const dim3 grid(num_tokens);
        const dim3 block(num_tokens <= 256 ? 256 : 512);
        if (rot_dim == 64) {
          rotary_embedding_prefill_neox_fp16_h2_kernel<64, true><<<grid, block, 0, stream>>>(
              static_cast<const int64_t*>(positions.data_ptr()),
              static_cast<half*>(query.data_ptr()),
              static_cast<half*>(key.data_ptr()),
              static_cast<const half*>(cos_sin_cache.data_ptr()),
              num_heads,
              num_kv_heads,
              query_stride,
              key_stride,
              query_head_stride,
              key_head_stride);
        } else {
          rotary_embedding_prefill_neox_fp16_h2_kernel<128, true><<<grid, block, 0, stream>>>(
              static_cast<const int64_t*>(positions.data_ptr()),
              static_cast<half*>(query.data_ptr()),
              static_cast<half*>(key.data_ptr()),
              static_cast<const half*>(cos_sin_cache.data_ptr()),
              num_heads,
              num_kv_heads,
              query_stride,
              key_stride,
              query_head_stride,
              key_head_stride);
        }
        return;
      }
    }
    if constexpr (std::is_same<T, __mt_bfloat16>::value && HAS_KEY) {
      if (seq_len == 0 && query_dim_stride == 1 && key_dim_stride == 1 &&
          (rot_dim == 64 || rot_dim == 128) &&
          (query_stride & 1) == 0 && (key_stride & 1) == 0 &&
          (query_head_stride & 1) == 0 && (key_head_stride & 1) == 0) {
        if (num_tokens <= 128) {
          const int block_threads = num_tokens < 128 ? 128 : 256;
          const int h2_pairs = (max_head_pairs + 1) / 2;
          const dim3 grid(num_tokens, (h2_pairs + block_threads - 1) / block_threads);
          const dim3 block(block_threads);
          if (VLLM_MUSA_ARCH_MP31 && num_heads == 32 && num_kv_heads == 8 && query_stride == 4096 &&
              key_stride == 1024 && query_head_stride == 128 && key_head_stride == 128) {
            if (rot_dim == 64) {
              rotary_embedding_decode_neox_bf16_h2_32x8_hs128_kernel<64><<<grid, block, 0, stream>>>(
                  static_cast<const int64_t*>(positions.data_ptr()),
                  static_cast<__mt_bfloat16*>(query.data_ptr()),
                  static_cast<__mt_bfloat16*>(key.data_ptr()),
                  static_cast<const __mt_bfloat16*>(cos_sin_cache.data_ptr()));
            } else {
              rotary_embedding_decode_neox_bf16_h2_32x8_hs128_kernel<128><<<grid, block, 0, stream>>>(
                  static_cast<const int64_t*>(positions.data_ptr()),
                  static_cast<__mt_bfloat16*>(query.data_ptr()),
                  static_cast<__mt_bfloat16*>(key.data_ptr()),
                  static_cast<const __mt_bfloat16*>(cos_sin_cache.data_ptr()));
            }
          } else if (rot_dim == 64) {
            rotary_embedding_decode_neox_bf16_h2_split_kernel<64, true><<<grid, block, 0, stream>>>(
                static_cast<const int64_t*>(positions.data_ptr()),
                static_cast<__mt_bfloat16*>(query.data_ptr()),
                static_cast<__mt_bfloat16*>(key.data_ptr()),
                static_cast<const __mt_bfloat16*>(cos_sin_cache.data_ptr()),
                num_heads,
                num_kv_heads,
                query_stride,
                key_stride,
                query_head_stride,
                key_head_stride);
          } else {
            rotary_embedding_decode_neox_bf16_h2_split_kernel<128, true><<<grid, block, 0, stream>>>(
                static_cast<const int64_t*>(positions.data_ptr()),
                static_cast<__mt_bfloat16*>(query.data_ptr()),
                static_cast<__mt_bfloat16*>(key.data_ptr()),
                static_cast<const __mt_bfloat16*>(cos_sin_cache.data_ptr()),
                num_heads,
                num_kv_heads,
                query_stride,
                key_stride,
                query_head_stride,
                key_head_stride);
          }
          return;
        }
        const dim3 grid(num_tokens);
        const dim3 block(num_tokens <= 256 ? 256 : 512);
        if (rot_dim == 64) {
          rotary_embedding_prefill_neox_bf16_h2_kernel<64, true><<<grid, block, 0, stream>>>(
              static_cast<const int64_t*>(positions.data_ptr()),
              static_cast<__mt_bfloat16*>(query.data_ptr()),
              static_cast<__mt_bfloat16*>(key.data_ptr()),
              static_cast<const __mt_bfloat16*>(cos_sin_cache.data_ptr()),
              num_heads,
              num_kv_heads,
              query_stride,
              key_stride,
              query_head_stride,
              key_head_stride);
        } else {
          rotary_embedding_prefill_neox_bf16_h2_kernel<128, true><<<grid, block, 0, stream>>>(
              static_cast<const int64_t*>(positions.data_ptr()),
              static_cast<__mt_bfloat16*>(query.data_ptr()),
              static_cast<__mt_bfloat16*>(key.data_ptr()),
              static_cast<const __mt_bfloat16*>(cos_sin_cache.data_ptr()),
              num_heads,
              num_kv_heads,
              query_stride,
              key_stride,
              query_head_stride,
              key_head_stride);
        }
        return;
      }
    }
  }
  if (num_tokens <= kDecodeThreshold) {
    const int block_pairs = std::min(512, std::max(32, num_heads * half_rot_dim));
    const dim3 grid(num_tokens, (max_head_pairs + block_pairs - 1) / block_pairs);
    const dim3 block(block_pairs);
    rotary_embedding_decode_kernel<T, IS_NEOX, HAS_KEY><<<grid, block, 0, stream>>>(
        static_cast<const int64_t*>(positions.data_ptr()),
        static_cast<T*>(query.data_ptr()),
        static_cast<T*>(key.data_ptr()),
        static_cast<const T*>(cos_sin_cache.data_ptr()),
        num_heads,
        num_kv_heads,
        rot_dim,
        seq_len,
        query_batch_stride,
        key_batch_stride,
        query_stride,
        key_stride,
        query_head_stride,
        key_head_stride,
        query_dim_stride,
        key_dim_stride);
  } else {
    if constexpr (IS_NEOX) {
      if constexpr (std::is_same<T, half>::value && HAS_KEY) {
        if (seq_len == 0 && query_dim_stride == 1 && key_dim_stride == 1 &&
            (rot_dim == 64 || rot_dim == 128) &&
            (query_stride & 1) == 0 && (key_stride & 1) == 0 &&
            (query_head_stride & 1) == 0 && (key_head_stride & 1) == 0) {
          const dim3 grid(num_tokens);
          const dim3 block(num_tokens <= 256 ? 256 : 512);
          if (rot_dim == 64) {
            rotary_embedding_prefill_neox_fp16_h2_kernel<64, true><<<grid, block, 0, stream>>>(
                static_cast<const int64_t*>(positions.data_ptr()),
                static_cast<half*>(query.data_ptr()),
                static_cast<half*>(key.data_ptr()),
                static_cast<const half*>(cos_sin_cache.data_ptr()),
                num_heads,
                num_kv_heads,
                query_stride,
                key_stride,
                query_head_stride,
                key_head_stride);
          } else {
            rotary_embedding_prefill_neox_fp16_h2_kernel<128, true><<<grid, block, 0, stream>>>(
                static_cast<const int64_t*>(positions.data_ptr()),
                static_cast<half*>(query.data_ptr()),
                static_cast<half*>(key.data_ptr()),
                static_cast<const half*>(cos_sin_cache.data_ptr()),
                num_heads,
                num_kv_heads,
                query_stride,
                key_stride,
                query_head_stride,
                key_head_stride);
          }
          return;
        }
      }
      if constexpr (std::is_same<T, half>::value && !HAS_KEY) {
        if (seq_len == 0 && query_dim_stride == 1 && (rot_dim == 64 || rot_dim == 128) &&
            (query_stride & 1) == 0 && (query_head_stride & 1) == 0) {
          if (rot_dim == 64) {
            const dim3 grid(num_tokens);
            const dim3 block(256);
            rotary_embedding_prefill_neox_fp16_h2_kernel<64, false><<<grid, block, 0, stream>>>(
                static_cast<const int64_t*>(positions.data_ptr()),
                static_cast<half*>(query.data_ptr()),
                static_cast<half*>(key.data_ptr()),
                static_cast<const half*>(cos_sin_cache.data_ptr()),
                num_heads,
                num_kv_heads,
                query_stride,
                key_stride,
                query_head_stride,
                key_head_stride);
          } else {
            const dim3 grid(num_tokens);
            const dim3 block(512);
            rotary_embedding_prefill_neox_fp16_q_h2x2_kernel<128><<<grid, block, 0, stream>>>(
                static_cast<const int64_t*>(positions.data_ptr()),
                static_cast<half*>(query.data_ptr()),
                static_cast<const half*>(cos_sin_cache.data_ptr()),
                num_heads,
                query_stride,
                query_head_stride);
          }
          return;
        }
      }
      if constexpr (std::is_same<T, __mt_bfloat16>::value) {
        if (seq_len == 0 && query_dim_stride == 1 && key_dim_stride == 1 &&
            (rot_dim == 64 || rot_dim == 128) &&
            (query_stride & 1) == 0 && (key_stride & 1) == 0 &&
            (query_head_stride & 1) == 0 && (key_head_stride & 1) == 0) {
          const dim3 grid(num_tokens);
          const dim3 block(512);
          if constexpr (HAS_KEY) {
            if (rot_dim == 64) {
              rotary_embedding_prefill_neox_bf16_h2_kernel<64, true><<<grid, block, 0, stream>>>(
                  static_cast<const int64_t*>(positions.data_ptr()),
                  static_cast<__mt_bfloat16*>(query.data_ptr()),
                  static_cast<__mt_bfloat16*>(key.data_ptr()),
                  static_cast<const __mt_bfloat16*>(cos_sin_cache.data_ptr()),
                  num_heads,
                  num_kv_heads,
                  query_stride,
                  key_stride,
                  query_head_stride,
                  key_head_stride);
            } else {
              rotary_embedding_prefill_neox_bf16_h2_kernel<128, true><<<grid, block, 0, stream>>>(
                  static_cast<const int64_t*>(positions.data_ptr()),
                  static_cast<__mt_bfloat16*>(query.data_ptr()),
                  static_cast<__mt_bfloat16*>(key.data_ptr()),
                  static_cast<const __mt_bfloat16*>(cos_sin_cache.data_ptr()),
                  num_heads,
                  num_kv_heads,
                  query_stride,
                  key_stride,
                  query_head_stride,
                  key_head_stride);
            }
            return;
          } else if (rot_dim == 64) {
            rotary_embedding_prefill_neox_bf16_h2_kernel<64, false><<<grid, block, 0, stream>>>(
                static_cast<const int64_t*>(positions.data_ptr()),
                static_cast<__mt_bfloat16*>(query.data_ptr()),
                static_cast<__mt_bfloat16*>(key.data_ptr()),
                static_cast<const __mt_bfloat16*>(cos_sin_cache.data_ptr()),
                num_heads,
                num_kv_heads,
                query_stride,
                key_stride,
                query_head_stride,
                key_head_stride);
            return;
          }
        }
      }

      if (query_dim_stride == 1 && key_dim_stride == 1 && (rot_dim == 64 || rot_dim == 128)) {
        int block_pairs = std::min(512, std::max(32, num_heads * half_rot_dim));
        if (num_tokens > 16384) {
          block_pairs = std::min(256, block_pairs);
        }
        const dim3 grid(num_tokens);
        const dim3 block(block_pairs);
        if (rot_dim == 64) {
          if (seq_len > 0) {
            rotary_embedding_prefill_neox_static_kernel<T, 64, HAS_KEY, true><<<grid, block, 0, stream>>>(
                static_cast<const int64_t*>(positions.data_ptr()),
                static_cast<T*>(query.data_ptr()),
                static_cast<T*>(key.data_ptr()),
                static_cast<const T*>(cos_sin_cache.data_ptr()),
                num_heads,
                num_kv_heads,
                seq_len,
                query_batch_stride,
                key_batch_stride,
                query_stride,
                key_stride,
                query_head_stride,
                key_head_stride);
          } else {
            rotary_embedding_prefill_neox_static_kernel<T, 64, HAS_KEY, false><<<grid, block, 0, stream>>>(
                static_cast<const int64_t*>(positions.data_ptr()),
                static_cast<T*>(query.data_ptr()),
                static_cast<T*>(key.data_ptr()),
                static_cast<const T*>(cos_sin_cache.data_ptr()),
                num_heads,
                num_kv_heads,
                0,
                query_batch_stride,
                key_batch_stride,
                query_stride,
                key_stride,
                query_head_stride,
                key_head_stride);
          }
        } else {
          if (seq_len > 0) {
            rotary_embedding_prefill_neox_static_kernel<T, 128, HAS_KEY, true><<<grid, block, 0, stream>>>(
                static_cast<const int64_t*>(positions.data_ptr()),
                static_cast<T*>(query.data_ptr()),
                static_cast<T*>(key.data_ptr()),
                static_cast<const T*>(cos_sin_cache.data_ptr()),
                num_heads,
                num_kv_heads,
                seq_len,
                query_batch_stride,
                key_batch_stride,
                query_stride,
                key_stride,
                query_head_stride,
                key_head_stride);
          } else {
            rotary_embedding_prefill_neox_static_kernel<T, 128, HAS_KEY, false><<<grid, block, 0, stream>>>(
              static_cast<const int64_t*>(positions.data_ptr()),
              static_cast<T*>(query.data_ptr()),
              static_cast<T*>(key.data_ptr()),
              static_cast<const T*>(cos_sin_cache.data_ptr()),
              num_heads,
              num_kv_heads,
              0,
              query_batch_stride,
              key_batch_stride,
              query_stride,
              key_stride,
              query_head_stride,
              key_head_stride);
          }
        }
        return;
      }
    }

    int block_pairs = std::min(512, std::max(32, num_heads * half_rot_dim));
    if (num_tokens > 16384) {
      block_pairs = std::min(256, block_pairs);
    }
    const dim3 grid(num_tokens);
    const dim3 block(block_pairs);
    rotary_embedding_prefill_direct_kernel<T, IS_NEOX, HAS_KEY><<<grid, block, 0, stream>>>(
        static_cast<const int64_t*>(positions.data_ptr()),
        static_cast<T*>(query.data_ptr()),
        static_cast<T*>(key.data_ptr()),
        static_cast<const T*>(cos_sin_cache.data_ptr()),
        num_heads,
        num_kv_heads,
        rot_dim,
        seq_len,
        query_batch_stride,
        key_batch_stride,
        query_stride,
        key_stride,
        query_head_stride,
        key_head_stride,
        query_dim_stride,
        key_dim_stride);
  }
}

template <typename T>
void dispatch_rotary_embedding(
    ffi::TensorView positions,
    ffi::TensorView query,
    ffi::TensorView key,
    ffi::TensorView cos_sin_cache,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int rot_dim,
    int seq_len,
    int query_batch_stride,
    int key_batch_stride,
    int query_stride,
    int key_stride,
    int query_head_stride,
    int key_head_stride,
    int query_dim_stride,
    int key_dim_stride,
    bool is_neox,
    bool has_key,
    musaStream_t stream) {
#define DISPATCH_ROPE(IS_NEOX_VALUE, HAS_KEY_VALUE)                 \
  launch_rotary_embedding<T, IS_NEOX_VALUE, HAS_KEY_VALUE>(         \
      positions,                                                    \
      query,                                                        \
      key,                                                          \
      cos_sin_cache,                                                \
      num_tokens,                                                   \
      num_heads,                                                    \
      num_kv_heads,                                                 \
      rot_dim,                                                      \
      seq_len,                                                      \
      query_batch_stride,                                           \
      key_batch_stride,                                             \
      query_stride,                                                 \
      key_stride,                                                   \
      query_head_stride,                                            \
      key_head_stride,                                              \
      query_dim_stride,                                             \
      key_dim_stride,                                               \
      stream)

  if (is_neox) {
    if (has_key) {
      DISPATCH_ROPE(true, true);
    } else {
      DISPATCH_ROPE(true, false);
    }
  } else {
    if (has_key) {
      DISPATCH_ROPE(false, true);
    } else {
      DISPATCH_ROPE(false, false);
    }
  }
#undef DISPATCH_ROPE
}

void sgl_rotary_embedding(
    ffi::TensorView positions,
    ffi::TensorView query,
    ffi::TensorView key,
    ffi::TensorView cos_sin_cache,
    int64_t num_tokens,
    int64_t head_size,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t rot_dim,
    int64_t seq_len,
    int64_t query_batch_stride,
    int64_t key_batch_stride,
    int64_t query_stride,
    int64_t key_stride,
    int64_t query_head_stride,
    int64_t key_head_stride,
    int64_t query_dim_stride,
    int64_t key_dim_stride,
    bool is_neox,
    bool has_key) {
  CHECK_MUSA(positions);
  CHECK_MUSA(query);
  CHECK_MUSA(key);
  CHECK_MUSA(cos_sin_cache);
  TVM_FFI_ICHECK(positions.IsContiguous());
  TVM_FFI_ICHECK(cos_sin_cache.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(positions.dtype(), dl_int64));
  TVM_FFI_ICHECK(dtype_equal(query.dtype(), cos_sin_cache.dtype()));
  TVM_FFI_ICHECK(dtype_equal(key.dtype(), query.dtype()));
  TVM_FFI_ICHECK_EQ(positions.device().device_id, query.device().device_id);
  TVM_FFI_ICHECK_EQ(positions.device().device_id, key.device().device_id);
  TVM_FFI_ICHECK_EQ(positions.device().device_id, cos_sin_cache.device().device_id);
  TVM_FFI_ICHECK_GT(num_tokens, 0);
  TVM_FFI_ICHECK_GT(head_size, 0);
  TVM_FFI_ICHECK_GT(num_heads, 0);
  TVM_FFI_ICHECK_GT(num_kv_heads, 0);
  TVM_FFI_ICHECK_GT(rot_dim, 0);
  TVM_FFI_ICHECK_EQ(rot_dim % 2, 0);
  TVM_FFI_ICHECK_LE(rot_dim, head_size);

  ffi::MUSADeviceGuard device_guard(query.device().device_id);
  musaStream_t stream = get_stream(query.device());

#define DISPATCH_DTYPE(T)                                                       \
  dispatch_rotary_embedding<T>(                                                 \
      positions,                                                                \
      query,                                                                    \
      key,                                                                      \
      cos_sin_cache,                                                            \
      static_cast<int>(num_tokens),                                             \
      static_cast<int>(num_heads),                                              \
      static_cast<int>(num_kv_heads),                                           \
      static_cast<int>(rot_dim),                                                \
      static_cast<int>(seq_len),                                                \
      static_cast<int>(query_batch_stride),                                     \
      static_cast<int>(key_batch_stride),                                       \
      static_cast<int>(query_stride),                                           \
      static_cast<int>(key_stride),                                             \
      static_cast<int>(query_head_stride),                                      \
      static_cast<int>(key_head_stride),                                        \
      static_cast<int>(query_dim_stride),                                       \
      static_cast<int>(key_dim_stride),                                         \
      is_neox,                                                                  \
      has_key,                                                                  \
      stream)

  if (dtype_equal(query.dtype(), dl_float16)) {
    DISPATCH_DTYPE(half);
  } else if (dtype_equal(query.dtype(), dl_bfloat16)) {
    DISPATCH_DTYPE(__mt_bfloat16);
  } else if (dtype_equal(query.dtype(), dl_float32)) {
    DISPATCH_DTYPE(float);
  } else {
    TVM_FFI_THROW(ValueError) << "Unsupported rotary_embedding dtype";
  }
#undef DISPATCH_DTYPE
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_rotary_embedding, sgl_rotary_embedding);
