#include <cstdint>

#include <musa_fp16.h>
#include <musa_runtime.h>
#include <torch/all.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

#include "../musa_ops.h"

namespace {

constexpr int64_t kMaxTopK = 512;
// Block width of the shared sparse_indexer_topk_kernel; the radix select emits
// its per-thread output order.
constexpr int kThreads = 256;
constexpr int kIndexInt32 = 1;
constexpr int kIndexInt64 = 2;

__device__ __forceinline__ int64_t load_index(const void *ptr, int kind,
                                              int64_t idx) {
  if (kind == kIndexInt32) {
    return static_cast<int64_t>(static_cast<const int32_t *>(ptr)[idx]);
  }
  return static_cast<int64_t>(static_cast<const int64_t *>(ptr)[idx]);
}

int index_kind(const torch::Tensor &tensor, const char *name) {
  if (tensor.scalar_type() == torch::kInt32) {
    return kIndexInt32;
  }
  if (tensor.scalar_type() == torch::kInt64) {
    return kIndexInt64;
  }
  TORCH_CHECK(false, name, " must be int32 or int64");
}

void check_musa_tensor(const torch::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.device().is_privateuseone(), name,
              " must be a MUSA tensor");
}

void check_same_device(const torch::Tensor &a, const torch::Tensor &b,
                       const char *b_name) {
  TORCH_CHECK(a.device() == b.device(), b_name, " must be on the same device");
}

template <typename OutT>
__device__ __forceinline__ void fill_all_indices(
    OutT *__restrict__ topk_indices, int64_t topk_stride0,
    int64_t topk_stride1, int64_t row, int64_t row_len, int64_t topk) {
  for (int64_t rank = threadIdx.x; rank < topk; rank += blockDim.x) {
    topk_indices[row * topk_stride0 + rank * topk_stride1] =
        static_cast<OutT>(rank < row_len ? rank : -1);
  }
}

// Decode top-k: a coarse histogram on the fp16 top byte finds the threshold
// bin, whose candidates are staged in shared memory for an exact 32-bit radix
// select.  Rows whose threshold bin overflows the stage stream from global
// memory instead.  The output (set and order) matches the shared
// sparse_indexer_topk_decode: NaN ranks lowest, signed zeros are equal, and
// exact ties resolve to the lowest positions.
constexpr int kRadixThreads = 512;
constexpr int kRadixWarps = kRadixThreads / 32;
constexpr int kRadixBins = 256;
constexpr int kRadixHistGroups = 4;
constexpr int kRadixCandidates = 4096;
constexpr int kRadixTies = 512;
static_assert(kRadixThreads == 2 * kThreads,
              "two radix threads per sparse_indexer_topk_kernel output owner");

// Integer normalization stays exact under -ffast-math/-fno-signed-zeros.
__device__ __forceinline__ uint32_t score_bits(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t magnitude = bits & 0x7FFFFFFFU;
  if (magnitude > 0x7F800000U) {
    return 0xFF800000U;
  }
  return magnitude == 0U ? 0U : bits;
}

__device__ __forceinline__ uint32_t score_key(uint32_t bits) {
  return (bits & 0x80000000U) ? ~bits : (bits | 0x80000000U);
}

// Rounding to fp16 is monotonic, so a higher coarse bin is a strictly larger
// score.
__device__ __forceinline__ uint32_t coarse_score_bin(uint32_t bits) {
  const uint16_t half_bits = __half_as_ushort(__float2half_rn(__uint_as_float(bits)));
  const uint16_t key = (half_bits & 0x8000U)
                           ? static_cast<uint16_t>(~half_bits)
                           : static_cast<uint16_t>(half_bits | 0x8000U);
  return static_cast<uint32_t>(key >> 8);
}

__device__ __forceinline__ uint32_t warp_inclusive_sum(uint32_t lane,
                                                       uint32_t value) {
#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset <<= 1) {
    const uint32_t other = __shfl_up_sync(0xFFFFFFFFU, value, offset);
    if (lane >= offset) {
      value += other;
    }
  }
  return value;
}

struct RadixSelection {
  uint32_t bin;
  uint32_t above;
  uint32_t in_bin;
};

// Merge the group histograms, clear them for the next round, and publish the
// unique bin b with #(bin > b) < need <= #(bin >= b).
__device__ __forceinline__ void radix_find_bin(
    uint32_t (*__restrict__ hist)[kRadixBins], uint32_t need,
    uint32_t *__restrict__ warp_sums, RadixSelection *__restrict__ selection) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 31U;
  uint32_t count = 0;
  uint32_t inclusive = 0;
  if (tid < kRadixBins) {
#pragma unroll
    for (int group = 0; group < kRadixHistGroups; ++group) {
      count += hist[group][tid];
      hist[group][tid] = 0;
    }
    inclusive = warp_inclusive_sum(lane, count);
    if (lane == 31U) {
      warp_sums[tid >> 5] = inclusive;
    }
  }
  __syncthreads();
  if (tid < kRadixBins) {
    uint32_t total = 0;
    uint32_t before = 0;
#pragma unroll
    for (uint32_t warp = 0; warp < kRadixBins / 32; ++warp) {
      const uint32_t part = warp_sums[warp];
      total += part;
      before += warp < (tid >> 5) ? part : 0U;
    }
    const uint32_t at_or_above = total - (before + inclusive) + count;
    const uint32_t above = at_or_above - count;
    if (at_or_above >= need && above < need) {
      selection->bin = tid;
      selection->above = above;
      selection->in_bin = count;
    }
  }
  __syncthreads();
}

template <typename OutT>
__global__ void sparse_indexer_topk_radix_kernel(
    const float *__restrict__ logits, int64_t logits_stride0,
    int64_t logits_stride1, int64_t columns, const void *__restrict__ row_ends,
    int row_ends_kind, int64_t row_ends_stride, OutT *__restrict__ topk_indices,
    int64_t topk_stride0, int64_t topk_stride1, int64_t rows, int64_t topk) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  __shared__ uint32_t hist[kRadixHistGroups][kRadixBins];
  __shared__ uint32_t warp_sums[kRadixWarps];
  __shared__ uint32_t cand_keys[kRadixCandidates];
  __shared__ int32_t cand_pos[kRadixCandidates];
  __shared__ int32_t tie_pos[kRadixTies];
  __shared__ uint32_t thread_counts[kRadixThreads];
  __shared__ uint32_t owner_counts[kThreads];
  __shared__ RadixSelection selection;
  __shared__ uint32_t s_num_cand;
  __shared__ uint32_t s_num_ties;
  __shared__ int64_t s_cutoff;

  const int tid = threadIdx.x;
  int64_t row_len = load_index(row_ends, row_ends_kind, row * row_ends_stride);
  row_len = row_len < 0 ? 0 : (row_len > columns ? columns : row_len);
  if (row_len <= topk) {
    fill_all_indices(topk_indices, topk_stride0, topk_stride1, row, row_len,
                     topk);
    return;
  }

  const float *row_logits = logits + row * logits_stride0;
  const int group = tid / (kRadixThreads / kRadixHistGroups);
  for (int i = tid; i < kRadixHistGroups * kRadixBins; i += kRadixThreads) {
    hist[i / kRadixBins][i % kRadixBins] = 0;
  }
  if (tid == 0) {
    s_num_cand = 0;
    s_num_ties = 0;
    s_cutoff = INT64_MAX;
  }
  __syncthreads();

  for (int64_t pos = tid; pos < row_len; pos += kRadixThreads) {
    const uint32_t bits = score_bits(row_logits[pos * logits_stride1]);
    atomicAdd(&hist[group][coarse_score_bin(bits)], 1U);
  }
  __syncthreads();
  uint32_t need = static_cast<uint32_t>(topk);
  radix_find_bin(hist, need, warp_sums, &selection);
  const uint32_t coarse_bin = selection.bin;
  need -= selection.above;

  // Stage the threshold-bin candidates; count this thread's sure selections.
  uint32_t selected = 0;
  for (int64_t pos = tid; pos < row_len; pos += kRadixThreads) {
    const uint32_t bits = score_bits(row_logits[pos * logits_stride1]);
    const uint32_t bin = coarse_score_bin(bits);
    if (bin > coarse_bin) {
      ++selected;
    } else if (bin == coarse_bin) {
      const uint32_t slot = atomicAdd(&s_num_cand, 1U);
      if (slot < kRadixCandidates) {
        cand_keys[slot] = score_key(bits);
        cand_pos[slot] = static_cast<int32_t>(pos);
      }
    }
  }
  thread_counts[tid] = selected;
  __syncthreads();
  const uint32_t num_cand = s_num_cand;
  const bool staged = num_cand <= static_cast<uint32_t>(kRadixCandidates);

  uint32_t prefix = 0U;
  uint32_t prefix_mask = 0U;
  uint32_t num_ties = 0U;
#pragma unroll 1
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (staged) {
      for (uint32_t i = tid; i < num_cand; i += kRadixThreads) {
        const uint32_t key = cand_keys[i];
        if ((key & prefix_mask) == prefix) {
          atomicAdd(&hist[group][(key >> shift) & 0xFFU], 1U);
        }
      }
    } else {
      for (int64_t pos = tid; pos < row_len; pos += kRadixThreads) {
        const uint32_t bits = score_bits(row_logits[pos * logits_stride1]);
        const uint32_t key = score_key(bits);
        if (coarse_score_bin(bits) == coarse_bin &&
            (key & prefix_mask) == prefix) {
          atomicAdd(&hist[group][(key >> shift) & 0xFFU], 1U);
        }
      }
    }
    __syncthreads();
    radix_find_bin(hist, need, warp_sums, &selection);
    prefix |= selection.bin << shift;
    prefix_mask |= 0xFFU << shift;
    need -= selection.above;
    num_ties = selection.in_bin;
  }
  const uint32_t threshold = prefix;
  const uint32_t ties_needed = need;

  if (ties_needed < num_ties) {
    if (staged && num_ties <= static_cast<uint32_t>(kRadixTies)) {
      for (uint32_t i = tid; i < num_cand; i += kRadixThreads) {
        if (cand_keys[i] == threshold) {
          tie_pos[atomicAdd(&s_num_ties, 1U)] = cand_pos[i];
        }
      }
      __syncthreads();
      for (uint32_t i = tid; i < num_ties; i += kRadixThreads) {
        const int32_t pos = tie_pos[i];
        uint32_t rank = 0;
        for (uint32_t j = 0; j < num_ties; ++j) {
          rank += tie_pos[j] < pos ? 1U : 0U;
        }
        if (rank + 1 == ties_needed) {
          s_cutoff = pos;
        }
      }
    } else if (tid == 0) {
      uint32_t remaining = ties_needed;
      int64_t cutoff = -1;
      for (int64_t pos = 0; pos < row_len && remaining > 0; ++pos) {
        if (score_key(score_bits(row_logits[pos * logits_stride1])) ==
            threshold) {
          cutoff = pos;
          --remaining;
        }
      }
      s_cutoff = cutoff;
    }
    __syncthreads();
  }
  const int64_t cutoff = s_cutoff;

  // Emit sparse_indexer_topk_kernel's order: kThreads owners, owner =
  // pos % kThreads, each writing its selected positions in ascending order.
  if (!staged) {
    for (int64_t pos = tid; pos < row_len; pos += kRadixThreads) {
      const uint32_t bits = score_bits(row_logits[pos * logits_stride1]);
      const uint32_t key = score_key(bits);
      if (coarse_score_bin(bits) == coarse_bin &&
          (key > threshold || (key == threshold && pos <= cutoff))) {
        ++selected;
      }
    }
    thread_counts[tid] = selected;
  }
  __syncthreads();
  if (tid < kThreads) {
    owner_counts[tid] = thread_counts[tid] + thread_counts[tid + kThreads];
  }
  __syncthreads();
  if (staged) {
    for (uint32_t i = tid; i < num_cand; i += kRadixThreads) {
      const uint32_t key = cand_keys[i];
      const int64_t pos = cand_pos[i];
      if (key > threshold || (key == threshold && pos <= cutoff)) {
        atomicAdd(&owner_counts[pos % kThreads], 1U);
      }
    }
    __syncthreads();
  }

  const uint32_t lane = static_cast<uint32_t>(tid) & 31U;
  uint32_t owned = 0;
  uint32_t inclusive = 0;
  if (tid < kThreads) {
    owned = owner_counts[tid];
    inclusive = warp_inclusive_sum(lane, owned);
    if (lane == 31U) {
      warp_sums[tid >> 5] = inclusive;
    }
  }
  __syncthreads();
  if (tid >= kThreads) {
    return;
  }
  uint32_t before = 0;
  for (int warp = 0; warp < (tid >> 5); ++warp) {
    before += warp_sums[warp];
  }
  int64_t slot = static_cast<int64_t>(before + inclusive - owned);
  for (int64_t pos = tid; pos < row_len; pos += kThreads) {
    const uint32_t key = score_key(score_bits(row_logits[pos * logits_stride1]));
    if (key > threshold || (key == threshold && pos <= cutoff)) {
      if (slot < topk) {
        topk_indices[row * topk_stride0 + slot * topk_stride1] =
            static_cast<OutT>(pos);
      }
      ++slot;
    }
  }
}

// The radix select relies on 32-lane warp shuffles and lane masks.
bool radix_topk_supported() {
  static const bool supported = [] {
    int device = 0;
    musaDeviceProp prop;
    return musaGetDevice(&device) == musaSuccess &&
           musaGetDeviceProperties(&prop, device) == musaSuccess &&
           prop.warpSize == 32;
  }();
  return supported;
}

template <typename OutT>
void launch_sparse_indexer_topk_radix(const torch::Tensor &logits,
                                      const torch::Tensor &seq_lens,
                                      torch::Tensor &topk_indices,
                                      int64_t topk, musaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(logits.size(0)));
  const dim3 block(kRadixThreads);
  sparse_indexer_topk_radix_kernel<OutT><<<grid, block, 0, stream>>>(
      static_cast<const float *>(logits.data_ptr()), logits.stride(0),
      logits.stride(1), logits.size(1), seq_lens.data_ptr(),
      index_kind(seq_lens, "seq_lens"), seq_lens.stride(0),
      static_cast<OutT *>(topk_indices.data_ptr()), topk_indices.stride(0),
      topk_indices.stride(1), logits.size(0), topk);
}

} // namespace

void deepseek_v4_sparse_indexer_topk_decode(const torch::Tensor &logits,
                                            const torch::Tensor &seq_lens,
                                            torch::Tensor &topk_indices,
                                            int64_t topk) {
  check_musa_tensor(logits, "logits");
  check_same_device(logits, seq_lens, "seq_lens");
  check_same_device(logits, topk_indices, "topk_indices");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32,
              "logits must be float32");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2-D");
  TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.numel() >= logits.size(0),
              "seq_lens must have at least one entry per logit row");
  TORCH_CHECK(topk_indices.dim() == 2 &&
                  topk_indices.size(0) >= logits.size(0),
              "topk_indices must have at least one output row per logit row");
  TORCH_CHECK(topk >= 0 && topk <= topk_indices.size(1),
              "topk must fit topk_indices width");
  TORCH_CHECK(topk <= kMaxTopK, "topk > 512 is not supported");
  TORCH_CHECK(logits.stride(1) == 1,
              "logits last dimension must be contiguous");
  TORCH_CHECK(topk_indices.stride(1) == 1,
              "topk_indices last dimension must be contiguous");
  index_kind(seq_lens, "seq_lens");
  if (!radix_topk_supported()) {
    sparse_indexer_topk_decode(logits, seq_lens, topk_indices, topk);
    return;
  }
  if (logits.size(0) == 0 || topk == 0) {
    return;
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(logits));
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  if (topk_indices.scalar_type() == torch::kInt32) {
    launch_sparse_indexer_topk_radix<int32_t>(logits, seq_lens, topk_indices,
                                              topk, stream);
  } else if (topk_indices.scalar_type() == torch::kInt64) {
    launch_sparse_indexer_topk_radix<int64_t>(logits, seq_lens, topk_indices,
                                              topk, stream);
  } else {
    TORCH_CHECK(false, "topk_indices must be int32 or int64");
  }
  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess,
              "deepseek_v4_sparse_indexer_topk_decode launch failed: ",
              musaGetErrorString(err));
}
