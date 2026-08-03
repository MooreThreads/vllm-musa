#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <torch/all.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

#include <cstdlib>
#include <type_traits>

#include "vec_utils.muh"

extern "C" {
extern __device__ void __musa_memcpy_g2s(
    __attribute__((address_space(3))) void* dst,
    __attribute__((address_space(1))) const void* src, int size, int prefetch);
extern __device__ void __musa_memcpy_g2s_wait();
}

namespace vllm_musa {

template <int BLOCK_X>
__device__ __forceinline__ float block_reduce_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }
  if constexpr (BLOCK_X <= 32) {
    return value;
  }

  __shared__ float shared[BLOCK_X / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();

  value = threadIdx.x < (BLOCK_X / 32) ? shared[threadIdx.x] : 0.0f;
  if (warp == 0) {
    for (int offset = (BLOCK_X / 32) >> 1; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    if (threadIdx.x == 0) {
      shared[0] = value;
    }
  }
  __syncthreads();
  return shared[0];
}

template <typename T, int BLOCK_X, int CHUNK>
__global__ void __launch_bounds__(BLOCK_X, 1) fused_add_rmsnorm_kernel(
    T* __restrict__ input, T* __restrict__ residual,
    const T* __restrict__ weight, int hidden_size, int vec_hidden_size,
    float epsilon) {
  constexpr int VEC_SIZE = 8;
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  T* input_row = input + static_cast<size_t>(row) * hidden_size;
  T* residual_row = residual + static_cast<size_t>(row) * hidden_size;

  __shared__ T shared_weight[CHUNK * BLOCK_X * VEC_SIZE];

#pragma unroll
  for (int chunk = 0; chunk < CHUNK; ++chunk) {
    const int vec_idx = chunk * BLOCK_X + tid;
    if (vec_idx < vec_hidden_size) {
      __attribute__((address_space(3))) void* smem_ptr =
          (__attribute__((address_space(3))) void*)(
              shared_weight + vec_idx * VEC_SIZE);
      __attribute__((address_space(1))) const void* gmem_ptr =
          (__attribute__((address_space(1))) const void*)(
              weight + vec_idx * VEC_SIZE);
      __musa_memcpy_g2s(smem_ptr, gmem_ptr, 16, 128);
    }
  }

  Vec16<T> fused_local[CHUNK];
  float variance = 0.0f;
#pragma unroll
  for (int chunk = 0; chunk < CHUNK; ++chunk) {
    const int vec_idx = chunk * BLOCK_X + tid;
    if (vec_idx < vec_hidden_size) {
      Vec16<T> input_vec = vload16_byp_slc(input_row + vec_idx * VEC_SIZE);
      Vec16<T> residual_vec =
          vload16_byp_slc(residual_row + vec_idx * VEC_SIZE);
      Vec16<T> sum_vec;
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        const float value =
            static_cast<float>(input_vec.x[i]) + static_cast<float>(residual_vec.x[i]);
        sum_vec.x[i] = static_cast<T>(value);
        variance += value * value;
      }
      fused_local[chunk] = sum_vec;
      vstore16(residual_row + vec_idx * VEC_SIZE, sum_vec);
    }
  }

  const float total = block_reduce_sum<BLOCK_X>(variance);
  const float inv_rms = rsqrtf(total / static_cast<float>(hidden_size) + epsilon);

  __musa_memcpy_g2s_wait();
  __syncthreads();

#pragma unroll
  for (int chunk = 0; chunk < CHUNK; ++chunk) {
    const int vec_idx = chunk * BLOCK_X + tid;
    if (vec_idx < vec_hidden_size) {
      Vec16<T> weight_vec =
          *reinterpret_cast<Vec16<T>*>(shared_weight + vec_idx * VEC_SIZE);
      Vec16<T> out_vec;
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        out_vec.x[i] = static_cast<T>(static_cast<float>(fused_local[chunk].x[i]) *
                                      inv_rms *
                                      static_cast<float>(weight_vec.x[i]));
      }
      vstore16(input_row + vec_idx * VEC_SIZE, out_vec);
    }
  }
}

template <typename T>
void dispatch_fused_add_rmsnorm(T* input, T* residual, const T* weight,
                                int rows, int hidden_size, float epsilon,
                                musaStream_t stream, int requested_block) {
  const int vec_hidden_size = hidden_size / 8;

  static const int env_forced_block = []() {
    const char* env = std::getenv("VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X");
    return env == nullptr ? 0 : std::atoi(env);
  }();
  const int forced_block =
      env_forced_block > 0 ? env_forced_block : requested_block;

  int block_x;
  if (forced_block > 0) {
    TORCH_CHECK(forced_block == 128 || forced_block == 256 ||
                    forced_block == 512 || forced_block == 1024,
                "VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X must be one of "
                "128, 256, 512, or 1024");
    block_x = forced_block;
  } else if (rows >= 512 && vec_hidden_size <= 640) {
    block_x = 128;
  } else if (rows >= 256) {
    block_x = 256;
  } else if (rows >= 128 && vec_hidden_size >= 896) {
    block_x = 512;
  } else if (rows >= 128) {
    block_x = 256;
  } else if (rows >= 8 && vec_hidden_size >= 1280) {
    block_x = 512;
  } else if (std::is_same<T, __mt_bfloat16>::value && rows > 0 &&
             rows <= 16 && hidden_size == 896) {
    // S5000/Qwen2: reduce idle threads and reduction overhead for small-row
    // H=896.
    block_x = 256;
  } else {
    block_x = 1024;
  }

#define LAUNCH(BLOCK, CHUNK)                                                \
  fused_add_rmsnorm_kernel<T, BLOCK, CHUNK>                                 \
      <<<rows, BLOCK, 0, stream>>>(input, residual, weight, hidden_size,     \
                                   vec_hidden_size, epsilon)

  if (block_x == 128) {
    if (vec_hidden_size <= 128) LAUNCH(128, 1);
    else if (vec_hidden_size <= 256) LAUNCH(128, 2);
    else if (vec_hidden_size <= 384) LAUNCH(128, 3);
    else if (vec_hidden_size <= 512) LAUNCH(128, 4);
    else if (vec_hidden_size <= 640) LAUNCH(128, 5);
    else if (vec_hidden_size <= 768) LAUNCH(128, 6);
    else if (vec_hidden_size <= 896) LAUNCH(128, 7);
    else if (vec_hidden_size <= 1024) LAUNCH(128, 8);
    else if (vec_hidden_size <= 1536) LAUNCH(128, 12);
    else TORCH_CHECK(false, "BLOCK=128 hidden_size too large");
  } else if (block_x == 256) {
    if (vec_hidden_size <= 256) LAUNCH(256, 1);
    else if (vec_hidden_size <= 512) LAUNCH(256, 2);
    else if (vec_hidden_size <= 768) LAUNCH(256, 3);
    else if (vec_hidden_size <= 1024) LAUNCH(256, 4);
    else if (vec_hidden_size <= 1536) LAUNCH(256, 6);
    else if (vec_hidden_size <= 2048) LAUNCH(256, 8);
    else TORCH_CHECK(false, "BLOCK=256 hidden_size too large");
  } else if (block_x == 512) {
    if (vec_hidden_size <= 512) LAUNCH(512, 1);
    else if (vec_hidden_size <= 1024) LAUNCH(512, 2);
    else if (vec_hidden_size <= 1536) LAUNCH(512, 3);
    else if (vec_hidden_size <= 2048) LAUNCH(512, 4);
    else TORCH_CHECK(false, "BLOCK=512 hidden_size too large");
  } else {
    if (vec_hidden_size <= 1024) LAUNCH(1024, 1);
    else if (vec_hidden_size <= 2048) LAUNCH(1024, 2);
    else TORCH_CHECK(false, "BLOCK=1024 hidden_size too large");
  }

#undef LAUNCH
}

}  // namespace vllm_musa

void musa_fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                             torch::Tensor& weight, double epsilon,
                             int64_t block_x) {
  TORCH_CHECK(input.device().is_privateuseone(), "input must be a MUSA tensor");
  TORCH_CHECK(residual.device().is_privateuseone(),
              "residual must be a MUSA tensor");
  TORCH_CHECK(weight.device().is_privateuseone(), "weight must be a MUSA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(input.dim() == 2, "input must be 2-D");
  TORCH_CHECK(residual.dim() == 2, "residual must be 2-D");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1-D");
  TORCH_CHECK(input.sizes() == residual.sizes(), "input/residual shape mismatch");
  TORCH_CHECK(input.size(1) == weight.size(0), "weight size mismatch");
  TORCH_CHECK(input.scalar_type() == residual.scalar_type(),
              "input/residual dtype mismatch");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
              "input/weight dtype mismatch");
  TORCH_CHECK(input.size(1) % 8 == 0,
              "hidden_size must be a multiple of 8");
  TORCH_CHECK(input.size(1) <= 16384,
              "hidden_size must be <= 16384");

  const c10::musa::OptionalMUSAGuard guard(device_of(input));
  musaStream_t stream = c10::musa::getCurrentMUSAStream().stream();
  const int rows = static_cast<int>(input.size(0));
  const int hidden_size = static_cast<int>(input.size(1));

  if (input.scalar_type() == at::ScalarType::Half) {
    vllm_musa::dispatch_fused_add_rmsnorm<__half>(
        static_cast<__half*>(input.data_ptr()),
        static_cast<__half*>(residual.data_ptr()),
        static_cast<const __half*>(weight.data_ptr()), rows, hidden_size,
        static_cast<float>(epsilon), stream, static_cast<int>(block_x));
  } else if (input.scalar_type() == at::ScalarType::BFloat16) {
    vllm_musa::dispatch_fused_add_rmsnorm<__mt_bfloat16>(
        static_cast<__mt_bfloat16*>(input.data_ptr()),
        static_cast<__mt_bfloat16*>(residual.data_ptr()),
        static_cast<const __mt_bfloat16*>(weight.data_ptr()), rows, hidden_size,
        static_cast<float>(epsilon), stream, static_cast<int>(block_x));
  } else {
    TORCH_CHECK(false, "only fp16 and bf16 are supported");
  }
}
