#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <torch/all.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

#include <cstdlib>

#include "vec_utils.muh"

namespace vllm_musa {

template <typename T, int BLOCK_X, int TOKENS_PER_BLOCK>
__global__ void __launch_bounds__(BLOCK_X, 1)
    reshape_and_cache_flash_nhd_kernel(
        const T* __restrict__ key, const T* __restrict__ value,
        T* __restrict__ key_cache, T* __restrict__ value_cache,
        const int64_t* __restrict__ slot_mapping, int num_tokens,
        int vecs_per_token, int block_size, int64_t key_stride,
        int64_t value_stride, int64_t block_stride, int64_t page_stride) {
  constexpr int VEC_SIZE = 8;
  const int base_token = blockIdx.x * TOKENS_PER_BLOCK;
  const int total_vecs = TOKENS_PER_BLOCK * vecs_per_token;

  for (int linear = threadIdx.x; linear < total_vecs; linear += BLOCK_X) {
    const int local_token = linear / vecs_per_token;
    const int token_idx = base_token + local_token;
    if (token_idx >= num_tokens) {
      continue;
    }

    const int64_t slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) {
      continue;
    }

    const int vec_idx = linear - local_token * vecs_per_token;
    const int elem_offset = vec_idx * VEC_SIZE;
    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx - block_idx * block_size;

    const int64_t src_k = static_cast<int64_t>(token_idx) * key_stride +
                          elem_offset;
    const int64_t src_v = static_cast<int64_t>(token_idx) * value_stride +
                          elem_offset;
    const int64_t dst = block_idx * block_stride + block_offset * page_stride +
                        elem_offset;

    Vec16<T> key_vec = vload16_byp_slc(key + src_k);
    Vec16<T> value_vec = vload16_byp_slc(value + src_v);
    vstore16(key_cache + dst, key_vec);
    vstore16(value_cache + dst, value_vec);
  }
}

template <typename T, int BLOCK_X>
void launch_reshape_and_cache_flash_nhd(
    const T* key, const T* value, T* key_cache, T* value_cache,
    const int64_t* slot_mapping, int num_tokens, int num_heads, int head_size,
    int block_size, int64_t key_stride, int64_t value_stride,
    int64_t block_stride, int64_t page_stride, musaStream_t stream,
    int forced_tokens_per_block) {
  const int vecs_per_token = (num_heads * head_size) / 8;

#define LAUNCH(TPB)                                                        \
  reshape_and_cache_flash_nhd_kernel<T, BLOCK_X, TPB>                      \
      <<<(num_tokens + TPB - 1) / TPB, BLOCK_X, 0, stream>>>(              \
          key, value, key_cache, value_cache, slot_mapping, num_tokens,    \
          vecs_per_token, block_size, key_stride, value_stride,            \
          block_stride, page_stride)

  if (forced_tokens_per_block == 1) {
    LAUNCH(1);
  } else if (forced_tokens_per_block == 2) {
    LAUNCH(2);
  } else if (forced_tokens_per_block == 4) {
    LAUNCH(4);
  } else if (forced_tokens_per_block == 8) {
    LAUNCH(8);
  } else if (vecs_per_token <= 64) {
    LAUNCH(8);
  } else if (vecs_per_token <= 128) {
    LAUNCH(4);
  } else if (vecs_per_token <= 256) {
    LAUNCH(2);
  } else {
    LAUNCH(1);
  }

#undef LAUNCH
}

template <typename T>
void dispatch_reshape_and_cache_flash_nhd(
    const T* key, const T* value, T* key_cache, T* value_cache,
    const int64_t* slot_mapping, int num_tokens, int num_heads, int head_size,
    int block_size, int64_t key_stride, int64_t value_stride,
    int64_t block_stride, int64_t page_stride, musaStream_t stream,
    int64_t requested_block_x) {
  // Nonzero BLOCK_X values are a per-call benchmark seam. Production callers
  // pass 0, which preserves the fixed 512-thread launch.
  const int block_x = requested_block_x == 0 ? 512 : requested_block_x;
  static const int forced_tokens_per_block = []() {
    const char* env = std::getenv("VLLM_MUSA_RESHAPE_CACHE_TOKENS_PER_BLOCK");
    return env == nullptr ? 0 : std::atoi(env);
  }();

#define DISPATCH(BLOCK_X)                                                  \
  launch_reshape_and_cache_flash_nhd<T, BLOCK_X>(                          \
      key, value, key_cache, value_cache, slot_mapping, num_tokens,        \
      num_heads, head_size, block_size, key_stride, value_stride,          \
      block_stride, page_stride, stream, forced_tokens_per_block)

  if (block_x == 128) {
    DISPATCH(128);
  } else if (block_x == 256) {
    DISPATCH(256);
  } else if (block_x == 512) {
    DISPATCH(512);
  } else {
    DISPATCH(1024);
  }

#undef DISPATCH
}

}  // namespace vllm_musa

void musa_reshape_and_cache_flash_nhd(torch::Tensor& key, torch::Tensor& value,
                                      torch::Tensor& key_cache,
                                      torch::Tensor& value_cache,
                                      torch::Tensor& slot_mapping,
                                      int64_t block_x) {
  TORCH_CHECK(block_x == 0 || block_x == 128 || block_x == 256 ||
                  block_x == 512 || block_x == 1024,
              "block_x must be 0 (production default) or one of 128, 256, "
              "512, or 1024");
  TORCH_CHECK(key.device().is_privateuseone(), "key must be a MUSA tensor");
  TORCH_CHECK(value.device().is_privateuseone(), "value must be a MUSA tensor");
  TORCH_CHECK(key_cache.device().is_privateuseone(),
              "key_cache must be a MUSA tensor");
  TORCH_CHECK(value_cache.device().is_privateuseone(),
              "value_cache must be a MUSA tensor");
  TORCH_CHECK(slot_mapping.device().is_privateuseone(),
              "slot_mapping must be a MUSA tensor");
  TORCH_CHECK(key.dim() == 3, "key must be [tokens, heads, head_size]");
  TORCH_CHECK(value.dim() == 3, "value must be [tokens, heads, head_size]");
  TORCH_CHECK(key_cache.dim() == 4,
              "key_cache must be [blocks, block_size, heads, head_size]");
  TORCH_CHECK(value_cache.dim() == 4,
              "value_cache must be [blocks, block_size, heads, head_size]");
  TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be 1-D");
  TORCH_CHECK(key.scalar_type() == value.scalar_type(),
              "key/value dtype mismatch");
  TORCH_CHECK(key.scalar_type() == key_cache.scalar_type(),
              "key/key_cache dtype mismatch");
  TORCH_CHECK(value.scalar_type() == value_cache.scalar_type(),
              "value/value_cache dtype mismatch");
  TORCH_CHECK(slot_mapping.scalar_type() == at::ScalarType::Long,
              "slot_mapping must be int64");
  TORCH_CHECK(key.stride(2) == 1 && key.stride(1) == key.size(2),
              "key head/head_size dimensions must be contiguous");
  TORCH_CHECK(value.stride(2) == 1 && value.stride(1) == value.size(2),
              "value head/head_size dimensions must be contiguous");
  TORCH_CHECK(key_cache.size(0) == value_cache.size(0),
              "key/value cache block count mismatch");
  TORCH_CHECK(key_cache.size(1) == value_cache.size(1),
              "key/value cache block size mismatch");
  TORCH_CHECK(key_cache.size(2) == value_cache.size(2),
              "key/value cache head count mismatch");
  TORCH_CHECK(key_cache.size(3) == value_cache.size(3),
              "key/value cache head size mismatch");
  TORCH_CHECK(key.size(1) == key_cache.size(2), "head count mismatch");
  TORCH_CHECK(value.size(1) == value_cache.size(2), "value head count mismatch");
  TORCH_CHECK(key.size(2) == key_cache.size(3), "head size mismatch");
  TORCH_CHECK(value.size(2) == value_cache.size(3), "value head size mismatch");
  TORCH_CHECK(slot_mapping.size(0) <= key.size(0),
              "slot_mapping longer than key rows");
  TORCH_CHECK(slot_mapping.size(0) <= value.size(0),
              "slot_mapping longer than value rows");
  TORCH_CHECK(key.size(2) % 8 == 0, "head_size must be a multiple of 8");
  TORCH_CHECK(key_cache.stride(3) == 1 && value_cache.stride(3) == 1,
              "cache head dimension must be contiguous");
  TORCH_CHECK(key_cache.stride(2) == key_cache.size(3),
              "key_cache must use NHD layout");
  TORCH_CHECK(value_cache.stride(2) == value_cache.size(3),
              "value_cache must use NHD layout");
  TORCH_CHECK(key_cache.stride(1) == key_cache.size(2) * key_cache.size(3),
              "key_cache page stride must be contiguous NHD");
  TORCH_CHECK(value_cache.stride(1) ==
                  value_cache.size(2) * value_cache.size(3),
              "value_cache page stride must be contiguous NHD");

  const c10::musa::OptionalMUSAGuard guard(device_of(key));
  musaStream_t stream = c10::musa::getCurrentMUSAStream().stream();
  const int num_tokens = static_cast<int>(slot_mapping.size(0));
  const int num_heads = static_cast<int>(key.size(1));
  const int head_size = static_cast<int>(key.size(2));
  const int block_size = static_cast<int>(key_cache.size(1));
  const int64_t key_stride = key.stride(0);
  const int64_t value_stride = value.stride(0);
  const int64_t block_stride = key_cache.stride(0);
  const int64_t page_stride = key_cache.stride(1);

  if (num_tokens == 0) {
    return;
  }

  if (key.scalar_type() == at::ScalarType::Half) {
    vllm_musa::dispatch_reshape_and_cache_flash_nhd<__half>(
        static_cast<const __half*>(key.data_ptr()),
        static_cast<const __half*>(value.data_ptr()),
        static_cast<__half*>(key_cache.data_ptr()),
        static_cast<__half*>(value_cache.data_ptr()),
        slot_mapping.data_ptr<int64_t>(), num_tokens, num_heads, head_size,
        block_size, key_stride, value_stride, block_stride, page_stride,
        stream, block_x);
  } else if (key.scalar_type() == at::ScalarType::BFloat16) {
    vllm_musa::dispatch_reshape_and_cache_flash_nhd<__mt_bfloat16>(
        static_cast<const __mt_bfloat16*>(key.data_ptr()),
        static_cast<const __mt_bfloat16*>(value.data_ptr()),
        static_cast<__mt_bfloat16*>(key_cache.data_ptr()),
        static_cast<__mt_bfloat16*>(value_cache.data_ptr()),
        slot_mapping.data_ptr<int64_t>(), num_tokens, num_heads, head_size,
        block_size, key_stride, value_stride, block_stride, page_stride,
        stream, block_x);
  } else {
    TORCH_CHECK(false, "only fp16 and bf16 are supported");
  }
}
