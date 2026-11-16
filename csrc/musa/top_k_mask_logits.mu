/*
 * Copyright (c) 2020-2026, Moore Threads Technology Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stateless logits-domain top-k masking by radix selection over a monotone
 * IEEE-754 key. Elements ranked inside the top-k keep their value; the rest
 * become -inf, so callers can feed the result straight into a sampler without
 * a softmax pass. Values tied with the k-th largest stay eligible, matching
 * topk-threshold masking. NaN never wins and is always masked out.
 */

#include <cstdint>
#include <limits>

#include <musa_runtime.h>
#include <torch/all.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

#include "musa_ops.h"

namespace {

constexpr int kRadixBits = 8;
constexpr int kRadixBins = 1 << kRadixBits;
constexpr int kBinsPerLane = kRadixBins / 32;
constexpr int kMaxTopK = 1024;
constexpr int kCandidateCapacity = 2048;

// NaN detection on the raw bit pattern; a float self-comparison is not
// reliable here because the compiler may fold it under fast-math.
__device__ __forceinline__ bool is_nan_bits(float value) {
  return (__float_as_uint(value) & 0x7fffffffu) > 0x7f800000u;
}

// Monotone 32-bit key: unsigned comparison of keys matches float ordering for
// every finite value, with -inf at 0x007fffff and +inf at 0xff800000. NaN maps
// to the -inf key so it never ranks into the top-k.
__device__ __forceinline__ unsigned int logit_key(float value) {
  const unsigned int bits = __float_as_uint(value);
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    return 0x007fffffu;
  }
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Select the rank-th largest bin of a 256-bin histogram. Bins are descending
// ordered, so lane 31 owns the largest bins.
__device__ __forceinline__ void select_radix_desc(
    const unsigned int* histogram, unsigned int* selected, int* rank) {
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    const int begin = lane * kBinsPerLane;
    int count = 0;
    #pragma unroll
    for (int i = 0; i < kBinsPerLane; ++i) {
      count += static_cast<int>(histogram[begin + i]);
    }
    int suffix = count;
    #pragma unroll
    for (int delta = 1; delta < 32; delta <<= 1) {
      const int next = __shfl_down_sync(0xffffffffu, suffix, delta);
      if (lane + delta < 32) {
        suffix += next;
      }
    }
    const int above = suffix - count;
    const int wanted = *rank;
    const unsigned int mask =
        __ballot_sync(0xffffffffu, wanted > above && wanted <= suffix);
    const int winner = __ffs(static_cast<int>(mask)) - 1;
    if (lane == winner) {
      int local_rank = wanted - above;
      for (int bin = begin + kBinsPerLane - 1; bin >= begin; --bin) {
        const int bin_count = static_cast<int>(histogram[bin]);
        if (local_rank > bin_count) {
          local_rank -= bin_count;
        } else {
          *selected = static_cast<unsigned int>(bin);
          *rank = local_rank;
          break;
        }
      }
    }
  }
  __syncthreads();
}

template <typename scalar_t, int BlockSize>
__global__ __launch_bounds__(BlockSize, 1) void top_k_mask_logits_kernel(
    const scalar_t* __restrict__ input, scalar_t* __restrict__ output,
    const int* __restrict__ top_k_arr, int top_k_val, int rows, int vocab) {
  const int thread = threadIdx.x;
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ unsigned int histogram[kRadixBins];
  __shared__ unsigned int prefix;
  __shared__ unsigned int selected_bin;
  __shared__ int rank;
  __shared__ int candidate_count;
  __shared__ float candidates[kCandidateCapacity];
  __shared__ int candidate_indices[kCandidateCapacity];

  const int64_t row_base = static_cast<int64_t>(row) * vocab;
  const scalar_t* row_in = input + row_base;
  scalar_t* row_out = output + row_base;

  if (thread == 0) {
    prefix = 0;
    selected_bin = 0;
    int wanted = top_k_arr != nullptr ? top_k_arr[row] : top_k_val;
    wanted = wanted < 1 ? vocab : wanted;
    rank = wanted > vocab ? vocab : wanted;
    candidate_count = 0;
  }
  __syncthreads();

  for (int i = thread; i < kRadixBins; i += BlockSize) {
    histogram[i] = 0;
  }
  __syncthreads();

  // Level 0: histogram of the leading 8 key bits over the whole row.
  for (int i = thread; i < vocab; i += BlockSize) {
    atomicAdd(&histogram[logit_key(static_cast<float>(row_in[i])) >> 24], 1);
  }
  __syncthreads();
  select_radix_desc(histogram, &selected_bin, &rank);
  if (thread == 0) {
    prefix = selected_bin;
  }
  __syncthreads();

  // Level 1: histogram of the next 8 bits, caching the surviving elements so
  // the remaining levels stay in shared memory.
  for (int i = thread; i < kRadixBins; i += BlockSize) {
    histogram[i] = 0;
  }
  __syncthreads();
  for (int i = thread; i < vocab; i += BlockSize) {
    const float value = static_cast<float>(row_in[i]);
    const unsigned int key = logit_key(value);
    if ((key >> 24) != prefix) {
      continue;
    }
    const int index = atomicAdd(&candidate_count, 1);
    if (index < kCandidateCapacity) {
      candidates[index] = value;
      candidate_indices[index] = i;
    }
    atomicAdd(&histogram[(key >> 16) & 0xffu], 1);
  }
  __syncthreads();
  const bool cached = candidate_count <= kCandidateCapacity;
  select_radix_desc(histogram, &selected_bin, &rank);
  if (thread == 0) {
    prefix = (prefix << 8) | selected_bin;
  }
  __syncthreads();

  // Levels 2 and 3 refine the remaining 16 bits.
  #pragma unroll
  for (int level = 2; level < 4; ++level) {
    const int shift = 32 - (level + 1) * kRadixBits;
    const unsigned int upper_mask = 0xffffffffu << (shift + kRadixBits);
    const unsigned int upper_bits = prefix << (shift + kRadixBits);
    for (int i = thread; i < kRadixBins; i += BlockSize) {
      histogram[i] = 0;
    }
    __syncthreads();
    if (cached) {
      for (int i = thread; i < candidate_count; i += BlockSize) {
        const unsigned int key = logit_key(candidates[i]);
        if ((key & upper_mask) != upper_bits) {
          continue;
        }
        atomicAdd(&histogram[(key >> shift) & 0xffu], 1);
      }
    } else {
      for (int i = thread; i < vocab; i += BlockSize) {
        const unsigned int key = logit_key(static_cast<float>(row_in[i]));
        if ((key & upper_mask) != upper_bits) {
          continue;
        }
        atomicAdd(&histogram[(key >> shift) & 0xffu], 1);
      }
    }
    __syncthreads();
    select_radix_desc(histogram, &selected_bin, &rank);
    if (thread == 0) {
      prefix = (prefix << 8) | selected_bin;
    }
    __syncthreads();
  }

  const unsigned int threshold = prefix;
  const scalar_t neg_inf =
      static_cast<scalar_t>(-std::numeric_limits<float>::infinity());
  for (int i = thread; i < vocab; i += BlockSize) {
    const float value = static_cast<float>(row_in[i]);
    const bool keep = !is_nan_bits(value) && logit_key(value) >= threshold;
    row_out[i] = keep ? row_in[i] : neg_inf;
  }
}

template <typename scalar_t>
void launch_mask(const scalar_t* input, scalar_t* output, const int* top_k_arr,
                 int top_k_val, int rows, int vocab, musaStream_t stream) {
  constexpr int kSmallBlock = 1024;
  constexpr int kLargeBlock = 512;
  const dim3 grid(rows);
  if (rows <= 8) {
    top_k_mask_logits_kernel<scalar_t, kSmallBlock>
        <<<grid, kSmallBlock, 0, stream>>>(input, output, top_k_arr, top_k_val,
                                           rows, vocab);
  } else {
    top_k_mask_logits_kernel<scalar_t, kLargeBlock>
        <<<grid, kLargeBlock, 0, stream>>>(input, output, top_k_arr, top_k_val,
                                           rows, vocab);
  }
}

}  // namespace

void musa_top_k_mask_logits(const at::Tensor& logits, at::Tensor& output,
                            const c10::optional<at::Tensor>& maybe_top_k_arr,
                            int64_t top_k_val) {
  TORCH_CHECK(logits.device() == output.device(),
              "top-k mask output must be on the input device");
  TORCH_CHECK(logits.scalar_type() == output.scalar_type(),
              "top-k mask output dtype must match the input");
  TORCH_CHECK(logits.dim() == 2 && output.sizes() == logits.sizes(),
              "top-k mask expects matching 2-D tensors");
  TORCH_CHECK(logits.is_contiguous() && output.is_contiguous(),
              "top-k mask expects contiguous tensors");
  TORCH_CHECK(top_k_val >= 0 && top_k_val <= kMaxTopK,
              "top-k mask supports 0 (no top-k) or 1 <= k <= ", kMaxTopK);
  const int rows = static_cast<int>(logits.size(0));
  const int vocab = static_cast<int>(logits.size(1));
  TORCH_CHECK(vocab > 0 && (vocab % 4) == 0,
              "top-k mask requires a non-empty vocab divisible by 4");
  const int* top_k_arr =
      maybe_top_k_arr.has_value() ? maybe_top_k_arr->data_ptr<int>() : nullptr;
  if (rows == 0) {
    return;
  }

  const c10::musa::OptionalMUSAGuard device_guard(logits.device());
  auto stream = at::musa::getCurrentMUSAStream();
  if (logits.scalar_type() == at::kBFloat16) {
    launch_mask<at::BFloat16>(logits.data_ptr<at::BFloat16>(),
                              output.data_ptr<at::BFloat16>(), top_k_arr,
                              static_cast<int>(top_k_val), rows, vocab, stream);
  } else if (logits.scalar_type() == at::kFloat) {
    launch_mask<float>(logits.data_ptr<float>(), output.data_ptr<float>(),
                       top_k_arr, static_cast<int>(top_k_val), rows, vocab,
                       stream);
  } else {
    TORCH_CHECK(false, "top-k mask supports float32 and bfloat16 logits");
  }
  const musaError_t error = musaGetLastError();
  TORCH_CHECK(error == musaSuccess, "top-k mask kernel failed: ",
              musaGetErrorString(error));
}
