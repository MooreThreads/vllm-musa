#include <ATen/cuda/CUDAContext.h>

#include <cmath>
#include <cstdint>
#include <cuda_fp8.h>
#include <torch/all.h>

// Register-resident per-token-group 8-bit quantization.
//
// One 16-thread group cooperates on a 128-element quant group: each lane loads
// its 8 source elements (16 B) with a single 128-bit load into registers,
// reduces the group absmax with warp shuffles, then quantizes straight from
// registers and writes the 8 output bytes with a single 64-bit store. No shared
// memory and no __syncthreads(), unlike the shared-memory staging path.
//
// Scales are written row- or column-major from the output_s strides; the column
// layout matches the block-FP8 GEMM scale expectations.

namespace {

constexpr int THREADS_PER_GROUP = 16;

// Warp-segment max reduce within a 16-thread group.
__device__ __forceinline__ float GroupReduceMax(float val) {
  unsigned mask = threadIdx.x % 32 >= 16 ? 0xffff0000 : 0x0000ffff;

  val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
  return val;
}

inline int GetGroupsPerBlock(int64_t num_groups) {
  if (num_groups % 16 == 0) return 16;
  if (num_groups % 8 == 0) return 8;
  if (num_groups % 4 == 0) return 4;
  if (num_groups % 2 == 0) return 2;
  return 1;
}

template <typename T, typename DST_DTYPE, bool IS_COLUMN_MAJOR>
__global__ void per_token_group_quant_8bit_vec_kernel(
    const T* __restrict__ input, void* __restrict__ output_q,
    float* __restrict__ output_s, const int group_size, const int num_groups,
    const int groups_per_block, const float eps, const float min_8bit,
    const float max_8bit, const int num_groups_per_row, const int scale_stride) {
  static_assert(sizeof(T) == 2, "vec quant kernel expects a 2-byte input type");
  constexpr int VEC_SIZE = 16 / sizeof(T);  // 8 elements = 16 B per lane
  static_assert(sizeof(T) * VEC_SIZE == 16, "one 128-bit load per lane");

  const int local_group_id = threadIdx.x / THREADS_PER_GROUP;
  const int lane_id = threadIdx.x % THREADS_PER_GROUP;
  const int64_t global_group_id =
      static_cast<int64_t>(blockIdx.x) * groups_per_block + local_group_id;
  if (global_group_id >= num_groups) return;

  const int64_t group_offset = global_group_id * group_size;
  const T* group_input = input + group_offset + lane_id * VEC_SIZE;
  DST_DTYPE* group_output =
      static_cast<DST_DTYPE*>(output_q) + group_offset + lane_id * VEC_SIZE;

  // One 128-bit load of this lane's 8 source elements into registers.
  alignas(16) T regs[VEC_SIZE];
  *reinterpret_cast<uint4*>(&regs[0]) =
      *reinterpret_cast<const uint4*>(group_input);

  float local_absmax = eps;
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    local_absmax = fmaxf(local_absmax, fabsf(static_cast<float>(regs[i])));
  }
  local_absmax = GroupReduceMax(local_absmax);

  const float y_s = local_absmax / max_8bit;

  float* scale_output;
  if constexpr (IS_COLUMN_MAJOR) {
    const int row_idx = global_group_id / num_groups_per_row;
    const int col_idx = global_group_id % num_groups_per_row;
    scale_output = output_s + col_idx * scale_stride + row_idx;
  } else {
    scale_output = output_s + global_group_id;
  }
  if (lane_id == 0) *scale_output = y_s;

  // Quantize from registers and pack into a single 64-bit store.
  union {
    uint8_t b[VEC_SIZE];
    uint2 v;
  } pack;
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    float q = fminf(fmaxf(static_cast<float>(regs[i]) / y_s, min_8bit), max_8bit);
    DST_DTYPE qb = DST_DTYPE(q);
    pack.b[i] = *reinterpret_cast<const uint8_t*>(&qb);
  }
  *reinterpret_cast<uint2*>(group_output) = pack.v;
}

#define VEC_DISPATCH_HALF_TYPES(TYPE, NAME, ...)          \
  AT_DISPATCH_SWITCH(TYPE, NAME,                          \
                     AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
                         AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__))

}  // namespace

// Drop-in for per_token_group_fp8_quant on the group_size==128 bf16/fp16 path.
// Column-major scales are inferred from output_s strides.
void per_token_group_quant_8bit_vec(const torch::Tensor& input,
                                    torch::Tensor& output_q,
                                    torch::Tensor& output_s, int64_t group_size,
                                    double eps, double min_8bit,
                                    double max_8bit,
                                    int64_t requested_groups_per_block) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(output_q.is_contiguous());
  TORCH_CHECK(output_s.dim() == 2);
  TORCH_CHECK(group_size == 128,
              "per_token_group_quant_8bit_vec supports group_size==128.");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half ||
                  input.scalar_type() == at::ScalarType::BFloat16,
              "per_token_group_quant_8bit_vec supports bf16/fp16 input.");
  TORCH_CHECK(input.numel() % group_size == 0);
  TORCH_CHECK(output_s.scalar_type() == at::ScalarType::Float,
              "per_token_group_quant_8bit_vec requires a float32 output_s.");
  // The kernel loads each group with a 128-bit vector op, which needs a
  // 16-byte aligned base; a contiguous view with an odd storage offset is not.
  TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0,
              "per_token_group_quant_8bit_vec requires a 16-byte aligned input.");

  const int num_groups = static_cast<int>(input.numel() / group_size);
  if (num_groups == 0) return;

  int groups_per_block = GetGroupsPerBlock(num_groups);
  if (requested_groups_per_block != 0) {
    TORCH_CHECK(
        requested_groups_per_block == 1 || requested_groups_per_block == 2 ||
            requested_groups_per_block == 4 ||
            requested_groups_per_block == 8 ||
            requested_groups_per_block == 16,
        "groups_per_block must be one of 0,1,2,4,8,16.");
    TORCH_CHECK(num_groups % requested_groups_per_block == 0,
                "num_groups must be divisible by requested groups_per_block.");
    groups_per_block = static_cast<int>(requested_groups_per_block);
  }
  const int num_blocks = num_groups / groups_per_block;
  const int num_threads = groups_per_block * THREADS_PER_GROUP;

  const bool is_column_major = output_s.stride(0) < output_s.stride(1);
  const int num_groups_per_row =
      static_cast<int>(input.size(input.dim() - 1)) / group_size;
  const int scale_stride = static_cast<int>(output_s.stride(1));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto dst_type = output_q.scalar_type();

#define LAUNCH_VEC_QUANT_KERNEL(T, DST_DTYPE, IS_COL)                          \
  per_token_group_quant_8bit_vec_kernel<T, DST_DTYPE, IS_COL>                 \
      <<<num_blocks, num_threads, 0, stream>>>(                              \
          static_cast<const T*>(input.data_ptr()), output_q.data_ptr(),      \
          static_cast<float*>(output_s.data_ptr()), group_size, num_groups,  \
          groups_per_block, static_cast<float>(eps),                         \
          static_cast<float>(min_8bit), static_cast<float>(max_8bit),        \
          num_groups_per_row, scale_stride)

#define LAUNCH_VEC_QUANT_DST(T, DST_DTYPE)      \
  do {                                          \
    if (is_column_major) {                      \
      LAUNCH_VEC_QUANT_KERNEL(T, DST_DTYPE, true);  \
    } else {                                    \
      LAUNCH_VEC_QUANT_KERNEL(T, DST_DTYPE, false); \
    }                                           \
  } while (0)

  VEC_DISPATCH_HALF_TYPES(
      input.scalar_type(), "per_token_group_quant_8bit_vec", ([&] {
        if (dst_type == at::ScalarType::Float8_e4m3fn) {
          LAUNCH_VEC_QUANT_DST(scalar_t, __nv_fp8_e4m3);
        } else if (dst_type == at::ScalarType::Char) {
          LAUNCH_VEC_QUANT_DST(scalar_t, int8_t);
        } else {
          TORCH_CHECK(false,
                      "per_token_group_quant_8bit_vec only supports FP8/INT8 "
                      "outputs.");
        }
      }));

#undef LAUNCH_VEC_QUANT_DST
#undef LAUNCH_VEC_QUANT_KERNEL
}
