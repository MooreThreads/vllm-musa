/*
 * Copyright (c) 2020-2026, Moore Threads Technology Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Logits-domain top-k + top-p masking by radix selection over a monotone
 * IEEE-754 key, so the sampler never materializes a softmax or sorts a
 * vocabulary-wide row.
 *
 * Phase A resolves the k-th largest key by counting radix (4 x 8-bit levels).
 * Phase B masks everything below it and caches the survivors. Phase C walks
 * four radix levels again, this time accumulating exp(value - max) mass per
 * bin, and stops on the bin where the descending cumulative mass first reaches
 * p * total. Phase D masks the survivors below that key. Values tied with a
 * boundary stay eligible as a group, matching the top-k mask policy.
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
// The sort-based reference walks a sequential fp32 `cumsum`, while this kernel
// sums per-bin masses; the two agree only to ~1e-5 of the row total, which is
// enough to move the boundary across a whole bf16 tie group. Biasing the target
// up by this fraction guarantees the kept mass still reaches `p` (the reference
// contract is "cumulative mass >= p") and bounds the excess to 5e-5, at the cost
// of occasionally keeping one extra quantum of boundary tokens.
constexpr float kMassTargetBias = 5e-5f;

// NaN detection on the raw bit pattern; a float self-comparison is not
// reliable here because the compiler may fold it under fast-math.
__device__ __forceinline__ bool is_nan_bits(float value) {
  return (__float_as_uint(value) & 0x7fffffffu) > 0x7f800000u;
}

// Monotone 32-bit key: unsigned comparison of keys matches float ordering for
// every finite value, with -inf at 0x007fffff and +inf at 0xff800000. NaN maps
// to the -inf key so it never ranks inside a filter.
__device__ __forceinline__ unsigned int logit_key(float value) {
  const unsigned int bits = __float_as_uint(value);
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    return 0x007fffffu;
  }
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Select the bin holding the rank-th largest element. Lane 31 owns the largest
// bins, so the suffix scan accumulates everything above the lane.
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

// Same selection, but over the probability mass carried by each bin. On return
// ``remaining`` holds the mass still needed inside the selected bin.
__device__ __forceinline__ void select_mass_desc(const float* histogram,
                                                float* remaining,
                                                unsigned int* selected) {
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    const int begin = lane * kBinsPerLane;
    float mass = 0.0f;
    #pragma unroll
    for (int i = 0; i < kBinsPerLane; ++i) {
      mass += histogram[begin + i];
    }
    float suffix = mass;
    #pragma unroll
    for (int delta = 1; delta < 32; delta <<= 1) {
      const float next = __shfl_down_sync(0xffffffffu, suffix, delta);
      if (lane + delta < 32) {
        suffix += next;
      }
    }
    const float above = suffix - mass;
    const float all_mass = __shfl_sync(0xffffffffu, suffix, 0);
    float wanted = *remaining;
    wanted = wanted > all_mass ? all_mass : wanted;
    const unsigned int mask =
        __ballot_sync(0xffffffffu, wanted > above && wanted <= suffix);
    const int winner = __ffs(static_cast<int>(mask)) - 1;
    if (lane == winner) {
      float local = wanted - above;
      local = local > 0.0f ? local : 0.0f;
      for (int bin = begin + kBinsPerLane - 1; bin >= begin; --bin) {
        const float bin_mass = histogram[bin];
        if (local > bin_mass) {
          local -= bin_mass;
        } else {
          *selected = static_cast<unsigned int>(bin);
          *remaining = local;
          break;
        }
      }
    }
  }
  __syncthreads();
}

template <typename scalar_t, int BlockSize>
__global__ __launch_bounds__(BlockSize, 1) void top_k_top_p_mask_kernel(
    const scalar_t* __restrict__ input, scalar_t* __restrict__ output,
    const int* __restrict__ top_k_arr, int top_k_val,
    const float* __restrict__ top_p_arr, float top_p_val, int rows,
    int vocab) {
  const int thread = threadIdx.x;
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ unsigned int histogram[kRadixBins];
  __shared__ float mass_histogram[kRadixBins];
  __shared__ unsigned int prefix;
  __shared__ unsigned int selected_bin;
  __shared__ unsigned int survivor_max_key;
  __shared__ int rank;
  __shared__ int candidate_count;
  __shared__ float candidates[kCandidateCapacity];
  __shared__ int candidate_indices[kCandidateCapacity];
  __shared__ float remaining_mass;
  __shared__ float mass_total;

  const int64_t row_base = static_cast<int64_t>(row) * vocab;
  const scalar_t* row_in = input + row_base;
  scalar_t* row_out = output + row_base;
  const scalar_t neg_inf =
      static_cast<scalar_t>(-std::numeric_limits<float>::infinity());

  int top_k = top_k_arr != nullptr ? top_k_arr[row] : top_k_val;
  top_k = top_k < 1 ? vocab : top_k;
  top_k = top_k > vocab ? vocab : top_k;
  float top_p = top_p_arr != nullptr ? top_p_arr[row] : top_p_val;
  top_p = top_p > 1.0f ? 1.0f : top_p;
  top_p = top_p < 1e-6f ? 1e-6f : top_p;

  if (thread == 0) {
    prefix = 0;
    selected_bin = 0;
    survivor_max_key = 0;
    candidate_count = 0;
    remaining_mass = 0.0f;
    mass_total = 0.0f;
    rank = top_k;
  }
  __syncthreads();

  // Phase A: k-th largest key. A k covering the vocabulary means "no top-k".
  if (top_k < vocab) {
    for (int i = thread; i < kRadixBins; i += BlockSize) {
      histogram[i] = 0;
    }
    __syncthreads();
    for (int i = thread; i < vocab; i += BlockSize) {
      atomicAdd(&histogram[logit_key(static_cast<float>(row_in[i])) >> 24], 1);
    }
    __syncthreads();
    select_radix_desc(histogram, &selected_bin, &rank);
    if (thread == 0) {
      prefix = selected_bin;
    }
    __syncthreads();

    for (int i = thread; i < kRadixBins; i += BlockSize) {
      histogram[i] = 0;
    }
    __syncthreads();
    for (int i = thread; i < vocab; i += BlockSize) {
      const unsigned int key = logit_key(static_cast<float>(row_in[i]));
      if ((key >> 24) != prefix) {
        continue;
      }
      atomicAdd(&histogram[(key >> 16) & 0xffu], 1);
    }
    __syncthreads();
    select_radix_desc(histogram, &selected_bin, &rank);
    if (thread == 0) {
      prefix = (prefix << 8) | selected_bin;
    }
    __syncthreads();

    #pragma unroll
    for (int level = 2; level < 4; ++level) {
      const int shift = 32 - (level + 1) * kRadixBits;
      const unsigned int upper_mask = 0xffffffffu << (shift + kRadixBits);
      const unsigned int upper_bits = prefix << (shift + kRadixBits);
      for (int i = thread; i < kRadixBins; i += BlockSize) {
        histogram[i] = 0;
      }
      __syncthreads();
      for (int i = thread; i < vocab; i += BlockSize) {
        const unsigned int key = logit_key(static_cast<float>(row_in[i]));
        if ((key & upper_mask) != upper_bits) {
          continue;
        }
        atomicAdd(&histogram[(key >> shift) & 0xffu], 1);
      }
      __syncthreads();
      select_radix_desc(histogram, &selected_bin, &rank);
      if (thread == 0) {
        prefix = (prefix << 8) | selected_bin;
      }
      __syncthreads();
    }
  }
  const unsigned int kth_key = top_k < vocab ? prefix : 0u;

  // Phase B: mask below the k-th key and cache the survivors. The same sweep
  // records the largest surviving key, which is the softmax stabilizer.
  // Only the top-k path can fit the cache (k <= kMaxTopK < kCandidateCapacity),
  // and a per-element atomic increment is expensive on a vocabulary-wide sweep,
  // so a top-p-only row skips it and rescans instead.
  const bool want_cache = top_k < vocab;
  unsigned int local_max_key = 0u;
  for (int i = thread; i < vocab; i += BlockSize) {
    const float value = static_cast<float>(row_in[i]);
    const unsigned int key = logit_key(value);
    const bool keep =
        top_k >= vocab || (key >= kth_key && !is_nan_bits(value));
    if (!keep) {
      row_out[i] = neg_inf;
      continue;
    }
    row_out[i] = row_in[i];
    local_max_key = key > local_max_key ? key : local_max_key;
    if (want_cache) {
      const int slot = atomicAdd(&candidate_count, 1);
      if (slot < kCandidateCapacity) {
        candidates[slot] = value;
        candidate_indices[slot] = i;
      }
    }
  }
  if (local_max_key != 0u) {
    atomicMax(&survivor_max_key, local_max_key);
  }
  __syncthreads();
  if (top_p >= 1.0f) {
    return;
  }
  const bool cached = want_cache && candidate_count <= kCandidateCapacity;
  const float row_max = __uint_as_float(
      (survivor_max_key & 0x80000000u) ? (survivor_max_key ^ 0x80000000u)
                                       : ~survivor_max_key);
  if (thread == 0) {
    remaining_mass = 0.0f;
    prefix = 0;
  }
  __syncthreads();

  // Phase C: resolve the top-p threshold over the surviving mass.
  for (int level = 0; level < 4; ++level) {
    for (int i = thread; i < kRadixBins; i += BlockSize) {
      mass_histogram[i] = 0.0f;
    }
    __syncthreads();
    const int shift = 32 - (level + 1) * kRadixBits;
    const int upper_shift = shift + kRadixBits;
    const unsigned int upper_mask =
        upper_shift >= 32 ? 0u : (0xffffffffu << upper_shift);
    const unsigned int upper_bits =
        upper_shift >= 32 ? 0u : (prefix << upper_shift);
    if (cached) {
      for (int i = thread; i < candidate_count; i += BlockSize) {
        const unsigned int key = logit_key(candidates[i]);
        if (level > 0 && (key & upper_mask) != upper_bits) {
          continue;
        }
        const float weight = expf(candidates[i] - row_max);
        atomicAdd(&mass_histogram[(key >> shift) & 0xffu], weight);
      }
    } else {
      for (int i = thread; i < vocab; i += BlockSize) {
        const float value = static_cast<float>(row_in[i]);
        const unsigned int key = logit_key(value);
        if (top_k < vocab && key < kth_key) {
          continue;
        }
        if (level > 0 && (key & upper_mask) != upper_bits) {
          continue;
        }
        const float weight = is_nan_bits(value) ? 0.0f : expf(value - row_max);
        atomicAdd(&mass_histogram[(key >> shift) & 0xffu], weight);
      }
    }
    __syncthreads();
    if (level == 0) {
      float total = 0.0f;
      for (int i = thread; i < kRadixBins; i += BlockSize) {
        total += mass_histogram[i];
      }
      atomicAdd(&mass_total, total);
      __syncthreads();
      if (thread == 0) {
        const float biased_p = top_p + kMassTargetBias;
        remaining_mass =
            mass_total * (biased_p < 1.0f ? biased_p : 1.0f);
      }
      __syncthreads();
      if (remaining_mass <= 0.0f) {
        return;
      }
    }
    // A residual that flushed to zero finds no winning lane, which means the
    // need is already met: fall back to the bottom bin so the prefix keeps every
    // token it still covers instead of reusing the previous level's bin.
    if (thread == 0) {
      selected_bin = 0;
    }
    __syncthreads();
    select_mass_desc(mass_histogram, &remaining_mass, &selected_bin);
    if (thread == 0) {
      prefix = (prefix << 8) | selected_bin;
    }
    __syncthreads();
  }
  const unsigned int p_key = prefix;

  // Phase D: mask survivors that fall below the top-p threshold.
  if (cached) {
    for (int i = thread; i < candidate_count; i += BlockSize) {
      if (logit_key(candidates[i]) < p_key) {
        row_out[candidate_indices[i]] = neg_inf;
      }
    }
  } else {
    for (int i = thread; i < vocab; i += BlockSize) {
      const float value = static_cast<float>(row_in[i]);
      const unsigned int key = logit_key(value);
      if (top_k < vocab && key < kth_key) {
        continue;
      }
      if (key < p_key) {
        row_out[i] = neg_inf;
      }
    }
  }
}

template <typename scalar_t>
void launch_mask(const scalar_t* input, scalar_t* output, const int* top_k_arr,
                 int top_k_val, const float* top_p_arr, float top_p_val,
                 int rows, int vocab, musaStream_t stream) {
  constexpr int kSmallBlock = 1024;
  constexpr int kLargeBlock = 512;
  const dim3 grid(rows);
  if (rows <= 8) {
    top_k_top_p_mask_kernel<scalar_t, kSmallBlock>
        <<<grid, kSmallBlock, 0, stream>>>(input, output, top_k_arr, top_k_val,
                                           top_p_arr, top_p_val, rows, vocab);
  } else {
    top_k_top_p_mask_kernel<scalar_t, kLargeBlock>
        <<<grid, kLargeBlock, 0, stream>>>(input, output, top_k_arr, top_k_val,
                                           top_p_arr, top_p_val, rows, vocab);
  }
}

}  // namespace

void musa_top_p_mask_logits(const at::Tensor& logits, at::Tensor& output,
                            const c10::optional<at::Tensor>& maybe_top_k_arr,
                            int64_t top_k_val,
                            const c10::optional<at::Tensor>& maybe_top_p_arr,
                            double top_p_val) {
  TORCH_CHECK(logits.device() == output.device(),
              "top-p mask output must be on the input device");
  TORCH_CHECK(logits.scalar_type() == output.scalar_type(),
              "top-p mask output dtype must match the input");
  TORCH_CHECK(logits.dim() == 2 && output.sizes() == logits.sizes(),
              "top-p mask expects matching 2-D tensors");
  TORCH_CHECK(logits.is_contiguous() && output.is_contiguous(),
              "top-p mask expects contiguous tensors");
  TORCH_CHECK(top_k_val >= 0 && top_k_val <= kMaxTopK,
              "top-p mask supports 0 (no top-k) or 1 <= k <= ", kMaxTopK);
  const int rows = static_cast<int>(logits.size(0));
  const int vocab = static_cast<int>(logits.size(1));
  TORCH_CHECK(vocab > 0, "top-p mask requires a non-empty vocabulary");
  const int* top_k_arr =
      maybe_top_k_arr.has_value() ? maybe_top_k_arr->data_ptr<int>() : nullptr;
  const float* top_p_arr = maybe_top_p_arr.has_value()
                               ? maybe_top_p_arr->data_ptr<float>()
                               : nullptr;
  if (rows == 0) {
    return;
  }

  const c10::musa::OptionalMUSAGuard device_guard(logits.device());
  auto stream = at::musa::getCurrentMUSAStream();
  if (logits.scalar_type() == at::kBFloat16) {
    launch_mask<at::BFloat16>(logits.data_ptr<at::BFloat16>(),
                              output.data_ptr<at::BFloat16>(), top_k_arr,
                              static_cast<int>(top_k_val), top_p_arr,
                              static_cast<float>(top_p_val), rows, vocab,
                              stream);
  } else if (logits.scalar_type() == at::kFloat) {
    launch_mask<float>(logits.data_ptr<float>(), output.data_ptr<float>(),
                       top_k_arr, static_cast<int>(top_k_val), top_p_arr,
                       static_cast<float>(top_p_val), rows, vocab, stream);
  } else {
    TORCH_CHECK(false, "top-p mask supports float32 and bfloat16 logits");
  }
  const musaError_t error = musaGetLastError();
  TORCH_CHECK(error == musaSuccess, "top-p mask kernel failed: ",
              musaGetErrorString(error));
}
