// Radix-select fast top-k, adapted from sgl-kernel's AOT topk.cu (itself
// adapted from tilelang's topk_selector). Ported to the JIT layer so that
// kTopK = 512 support ships with the sglang python package instead of
// requiring an sgl-kernel wheel release.
//
// Semantics match the AOT fast_topk_v2 op: for each row b, select the
// kTopK largest scores in [row_starts[b], row_starts[b] + lengths[b]) and
// write their indices relative to row_starts[b]. Output order within a row
// is unspecified (atomic collection order), matching the AOT kernel.
#include <tvm/ffi/container/tensor.h>
#include <musa_fp16.h>
#include "../common.h"


namespace fast_topk_detail {

constexpr uint32_t kThreadsPerBlock = 1024;
// Each radix pass needs at most ~kTopK candidates in the threshold bin, so
// 4K entries per round (2 rounds = 8K entries = 32KB) is sufficient.
constexpr size_t kSmemBytes = 8 * 1024 * sizeof(uint32_t);  // 32KB

struct FastTopKParams {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B]
  int32_t* __restrict__ indices;           // [B, kTopK]
  const int32_t* __restrict__ lengths;     // [B]
  int64_t input_stride;
};

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  const __half h = __float2half_rn(x);
  const uint16_t bits = __half_as_ushort(h);
  const uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                       : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// When length <= kTopK, write the indices directly.
template <int kTopK>
__device__ __forceinline__ void naive_topk(
    const float* __restrict__ score, int32_t* __restrict__ indice, int32_t length) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < kTopK; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

// Radix-select top-k. Assumes length > kTopK (checked by the caller).
template <int kTopK>
__device__ __forceinline__ void radix_select_topk(
    const float* __restrict__ input, int* __restrict__ index, int row_start, int length) {
  int topk = kTopK;
  constexpr auto BLOCK_SIZE = kThreadsPerBlock;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = kSmemBytes / (2 * sizeof(int));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];

  auto& s_histogram = s_histogram_buf[0];
  // allocate for two rounds
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  // stage 1: 8bit coarse histogram
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx + row_start]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (tx < RADIX) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];

  if (topk == 0) {
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto bin = static_cast<int>(convert_to_uint8(input[idx + row_start]));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      }
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto raw_input = input[idx + row_start];
      const auto bin = static_cast<int>(convert_to_uint8(raw_input));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        // fuse the histogram computation here
        if (pos < int(SMEM_INPUT_SIZE)) {
          s_input_idx[0][pos] = idx;
          const auto bin = convert_to_uint32(raw_input);
          const auto sub_bin = (bin >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

  // stage 2: refine with 8bit radix passes
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int s_last_remain;
    const auto r_idx = round % 2;

    // clip here to prevent overflow
    const auto _raw_num_input = s_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE))
                               ? _raw_num_input
                               : int(SMEM_INPUT_SIZE);

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = topk - s_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    topk -= s_histogram[threshold_bin + 1];

    if (topk == 0) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(input[idx + row_start]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
      __syncthreads();
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = input[idx + row_start];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              index[kTopK - pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (pos < int(SMEM_INPUT_SIZE)) {
              // fuse the histogram computation here
              s_input_idx[r_idx ^ 1][pos] = idx;
              const auto bin = convert_to_uint32(raw_input);
              const auto sub_bin = (bin >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

template <int kTopK, bool kUsePDL>
__global__ __launch_bounds__(fast_topk_detail::kThreadsPerBlock) void fast_topk_kernel(
    const fast_topk_detail::FastTopKParams params) {
  using namespace fast_topk_detail;

  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto row_start = params.row_starts == nullptr ? 0 : params.row_starts[bid];
  const auto length = params.lengths[bid];
  const auto indice = params.indices + bid * kTopK;
  const auto score = params.input + bid * params.input_stride;
  if (length <= kTopK) {
    naive_topk<kTopK>(score, indice, length);
  } else {
    radix_select_topk<kTopK>(score, indice, row_start, length);
  }

  }

}  // namespace fast_topk_detail



template <int kTopK>
void launch_qsa_fast_topk(ffi::TensorView score, ffi::TensorView row_starts,
                          ffi::TensorView indices, ffi::TensorView lengths) {
  CHECK_MUSA_CONTIGUOUS(score);
  CHECK_MUSA_CONTIGUOUS(row_starts);
  CHECK_MUSA_CONTIGUOUS(indices);
  CHECK_MUSA_CONTIGUOUS(lengths);
  TVM_FFI_ICHECK_EQ(score.ndim(), 2);
  TVM_FFI_ICHECK_EQ(row_starts.ndim(), 1);
  TVM_FFI_ICHECK_EQ(indices.ndim(), 2);
  TVM_FFI_ICHECK_EQ(lengths.ndim(), 1);
  TVM_FFI_ICHECK_EQ(score.size(0), row_starts.size(0));
  TVM_FFI_ICHECK_EQ(score.size(0), indices.size(0));
  TVM_FFI_ICHECK_EQ(score.size(0), lengths.size(0));
  TVM_FFI_ICHECK_EQ(indices.size(1), kTopK);
  TVM_FFI_ICHECK(dtype_equal(score.dtype(), dl_float32));
  TVM_FFI_ICHECK(dtype_equal(row_starts.dtype(), dl_int32));
  TVM_FFI_ICHECK(dtype_equal(indices.dtype(), dl_int32));
  TVM_FFI_ICHECK(dtype_equal(lengths.dtype(), dl_int32));
  const int64_t rows = score.size(0);
  if (rows == 0) return;
  fast_topk_detail::FastTopKParams params{
      static_cast<const float*>(score.data_ptr()),
      static_cast<const int32_t*>(row_starts.data_ptr()),
      static_cast<int32_t*>(indices.data_ptr()),
      static_cast<const int32_t*>(lengths.data_ptr()), score.stride(0)};
  TVM_FFI_ICHECK_EQ(musaSetDevice(score.device().device_id), musaSuccess);
  auto stream = get_stream(score.device());
  fast_topk_detail::fast_topk_kernel<kTopK, false>
      <<<static_cast<unsigned>(rows), fast_topk_detail::kThreadsPerBlock,
         fast_topk_detail::kSmemBytes, stream>>>(params);
  TVM_FFI_ICHECK_EQ(musaGetLastError(), musaSuccess)
      << "MUSA QSA fast top-k launch failed";
}

void vllm_musa_qsa_fast_topk(ffi::TensorView score,
                             ffi::TensorView row_starts,
                             ffi::TensorView indices,
                             ffi::TensorView lengths,
                             int topk) {
  if (topk == 512) {
    launch_qsa_fast_topk<512>(score, row_starts, indices, lengths);
  } else if (topk == 2048) {
    launch_qsa_fast_topk<2048>(score, row_starts, indices, lengths);
  } else {
    TVM_FFI_THROW(ValueError) << "unsupported QSA top-k: " << topk;
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(vllm_musa_qsa_fast_topk, vllm_musa_qsa_fast_topk);
