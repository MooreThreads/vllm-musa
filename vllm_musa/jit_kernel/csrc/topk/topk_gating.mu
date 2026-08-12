#include <cmath>
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

#include "../common.h"
#include "../device_utils.h"

constexpr float kFloatMinimum = -10000.0f;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerCta = 4;

__device__ __forceinline__ float stable_sigmoid(float x) {
  const bool positive = x >= 0.0f;
  const float z = __expf(positive ? -x : x);
  const float inv = 1.0f / (1.0f + z);
  return positive ? inv : z * inv;
}

__device__ __forceinline__ void warp_argmax(float &val, int &idx) {
#pragma unroll
  for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    const float other_val = __shfl_xor_sync(0xffffffff, val, mask);
    const int other_idx = __shfl_xor_sync(0xffffffff, idx, mask);
    if (other_val > val || (other_val == val && other_idx < idx)) {
      val = other_val;
      idx = other_idx;
    }
  }
}

__device__ __forceinline__ float warp_sum(float val) {
#pragma unroll
  for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

__device__ __forceinline__ float warp_max(float val) {
#pragma unroll
  for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

__device__ __forceinline__ int lane_id() {
  int lane;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
  return lane;
}

template <typename T, int NumExperts, int ValuesPerThread>
__global__
__launch_bounds__(kWarpsPerCta *kWarpSize, 1) void topk_softmax_warp_kernel(
    const T *__restrict__ gating_output, float *__restrict__ topk_weights,
    int32_t *__restrict__ topk_ids, const float *__restrict__ correction_bias,
    int num_tokens, int topk, bool renormalize, float moe_softcapping,
    bool has_correction_bias) {
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane = lane_id();
  const int row = blockIdx.x * kWarpsPerCta + warp_id;
  if (row >= num_tokens) {
    return;
  }

  float probs[ValuesPerThread];
  float thread_max = kFloatMinimum;

#pragma unroll
  for (int i = 0; i < ValuesPerThread; ++i) {
    const int expert = lane + i * kWarpSize;
    float val = kFloatMinimum;
    val = to_float(gating_output[row * NumExperts + expert]);
    if (has_correction_bias) {
      val += correction_bias[expert];
    }
    if (moe_softcapping > 0.0f) {
      val = tanhf(val / moe_softcapping) * moe_softcapping;
    }
    probs[i] = val;
    thread_max = fmaxf(thread_max, val);
  }

  const float row_max = warp_max(thread_max);
  float thread_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ValuesPerThread; ++i) {
    probs[i] = __expf(probs[i] - row_max);
    thread_sum += probs[i];
  }
  const float inv_row_sum = 1.0f / warp_sum(thread_sum);
#pragma unroll
  for (int i = 0; i < ValuesPerThread; ++i) {
    probs[i] *= inv_row_sum;
  }

  float selected_sum = 0.0f;
  for (int k_idx = 0; k_idx < topk; ++k_idx) {
    float max_val = probs[0];
    int max_idx = lane;
#pragma unroll
    for (int i = 1; i < ValuesPerThread; ++i) {
      const int expert = lane + i * kWarpSize;
      const float val = probs[i];
      if (val > max_val) {
        max_val = val;
        max_idx = expert;
      }
    }
    warp_argmax(max_val, max_idx);

    if (lane == 0) {
      const int out_idx = row * topk + k_idx;
      topk_weights[out_idx] = max_val;
      topk_ids[out_idx] = max_idx;
      selected_sum += max_val;
    }

#pragma unroll
    for (int i = 0; i < ValuesPerThread; ++i) {
      const int expert = lane + i * kWarpSize;
      if (expert == max_idx) {
        probs[i] = kFloatMinimum;
      }
    }
  }

  if (renormalize && lane == 0) {
    const float inv_selected_sum = 1.0f / selected_sum;
    for (int k_idx = 0; k_idx < topk; ++k_idx) {
      const int out_idx = row * topk + k_idx;
      topk_weights[out_idx] *= inv_selected_sum;
    }
  }
}

template <typename T, int NumExperts, int ValuesPerThread>
__global__
__launch_bounds__(kWarpsPerCta *kWarpSize, 1) void topk_sigmoid_warp_kernel(
    const T *__restrict__ gating_output, float *__restrict__ topk_weights,
    int32_t *__restrict__ topk_ids, const float *__restrict__ correction_bias,
    int num_tokens, int topk, bool renormalize, bool has_correction_bias) {
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane = lane_id();
  const int row = blockIdx.x * kWarpsPerCta + warp_id;
  if (row >= num_tokens) {
    return;
  }

  float choice[ValuesPerThread];

#pragma unroll
  for (int i = 0; i < ValuesPerThread; ++i) {
    const int expert = lane + i * kWarpSize;
    float val = kFloatMinimum;
    val = stable_sigmoid(to_float(gating_output[row * NumExperts + expert]));
    if (has_correction_bias) {
      val += correction_bias[expert];
    }
    choice[i] = val;
  }

  float selected_sum = 0.0f;
  for (int k_idx = 0; k_idx < topk; ++k_idx) {
    float max_val = choice[0];
    int max_idx = lane;
#pragma unroll
    for (int i = 1; i < ValuesPerThread; ++i) {
      const int expert = lane + i * kWarpSize;
      const float val = choice[i];
      if (val > max_val) {
        max_val = val;
        max_idx = expert;
      }
    }
    warp_argmax(max_val, max_idx);

    const float selected_prob =
        has_correction_bias ? max_val - correction_bias[max_idx] : max_val;
    if (lane == 0) {
      const int out_idx = row * topk + k_idx;
      topk_weights[out_idx] = selected_prob;
      topk_ids[out_idx] = max_idx;
      selected_sum += selected_prob;
    }

#pragma unroll
    for (int i = 0; i < ValuesPerThread; ++i) {
      const int expert = lane + i * kWarpSize;
      if (expert == max_idx) {
        choice[i] = kFloatMinimum;
      }
    }
  }

  if (renormalize && lane == 0) {
    const float inv_selected_sum = 1.0f / selected_sum;
    for (int k_idx = 0; k_idx < topk; ++k_idx) {
      const int out_idx = row * topk + k_idx;
      topk_weights[out_idx] *= inv_selected_sum;
    }
  }
}

template <typename T, int NumExperts, int ValuesPerThread>
__global__ __launch_bounds__(
    kWarpsPerCta *kWarpSize,
    1) void topk_sigmoid_no_bias_warp_kernel(const T
                                                 *__restrict__ gating_output,
                                             float *__restrict__ topk_weights,
                                             int32_t *__restrict__ topk_ids,
                                             int num_tokens, int topk,
                                             bool renormalize) {
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane = lane_id();
  const int row = blockIdx.x * kWarpsPerCta + warp_id;
  if (row >= num_tokens) {
    return;
  }

  float logits[ValuesPerThread];

#pragma unroll
  for (int i = 0; i < ValuesPerThread; ++i) {
    const int expert = lane + i * kWarpSize;
    logits[i] = to_float(gating_output[row * NumExperts + expert]);
  }

  float selected_sum = 0.0f;
  for (int k_idx = 0; k_idx < topk; ++k_idx) {
    float max_val = logits[0];
    int max_idx = lane;
#pragma unroll
    for (int i = 1; i < ValuesPerThread; ++i) {
      const int expert = lane + i * kWarpSize;
      const float val = logits[i];
      if (val > max_val) {
        max_val = val;
        max_idx = expert;
      }
    }
    warp_argmax(max_val, max_idx);

    const float selected_prob = stable_sigmoid(max_val);
    if (lane == 0) {
      const int out_idx = row * topk + k_idx;
      topk_weights[out_idx] = selected_prob;
      topk_ids[out_idx] = max_idx;
      selected_sum += selected_prob;
    }

#pragma unroll
    for (int i = 0; i < ValuesPerThread; ++i) {
      const int expert = lane + i * kWarpSize;
      if (expert == max_idx) {
        logits[i] = kFloatMinimum;
      }
    }
  }

  if (renormalize && lane == 0) {
    const float inv_selected_sum = 1.0f / selected_sum;
    for (int k_idx = 0; k_idx < topk; ++k_idx) {
      const int out_idx = row * topk + k_idx;
      topk_weights[out_idx] *= inv_selected_sum;
    }
  }
}

template <typename T, int NumExperts, int ValuesPerThread, int TopK>
__global__ __launch_bounds__(
    kWarpsPerCta *kWarpSize,
    1) void topk_softmax_no_bias_warp_kernel_fixed_k(const T
                                                         *__restrict__ gating_output,
                                                     float
                                                         *__restrict__ topk_weights,
                                                     int32_t
                                                         *__restrict__ topk_ids,
                                                     int num_tokens,
                                                     bool renormalize) {
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane = lane_id();
  const int row = blockIdx.x * kWarpsPerCta + warp_id;
  if (row >= num_tokens) {
    return;
  }

  float probs[ValuesPerThread];
  float thread_max = kFloatMinimum;

#pragma unroll
  for (int i = 0; i < ValuesPerThread; ++i) {
    const int expert = lane + i * kWarpSize;
    const float val = to_float(gating_output[row * NumExperts + expert]);
    probs[i] = val;
    thread_max = fmaxf(thread_max, val);
  }

  const float row_max = warp_max(thread_max);
  float thread_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ValuesPerThread; ++i) {
    probs[i] = __expf(probs[i] - row_max);
    thread_sum += probs[i];
  }
  const float inv_row_sum = 1.0f / warp_sum(thread_sum);
#pragma unroll
  for (int i = 0; i < ValuesPerThread; ++i) {
    probs[i] *= inv_row_sum;
  }

  float selected_sum = 0.0f;
#pragma unroll
  for (int k_idx = 0; k_idx < TopK; ++k_idx) {
    float max_val = probs[0];
    int max_idx = lane;
#pragma unroll
    for (int i = 1; i < ValuesPerThread; ++i) {
      const int expert = lane + i * kWarpSize;
      const float val = probs[i];
      if (val > max_val) {
        max_val = val;
        max_idx = expert;
      }
    }
    warp_argmax(max_val, max_idx);

    if (lane == 0) {
      const int out_idx = row * TopK + k_idx;
      topk_weights[out_idx] = max_val;
      topk_ids[out_idx] = max_idx;
      selected_sum += max_val;
    }

#pragma unroll
    for (int i = 0; i < ValuesPerThread; ++i) {
      const int expert = lane + i * kWarpSize;
      if (expert == max_idx) {
        probs[i] = kFloatMinimum;
      }
    }
  }

  if (renormalize && lane == 0) {
    const float inv_selected_sum = 1.0f / selected_sum;
#pragma unroll
    for (int k_idx = 0; k_idx < TopK; ++k_idx) {
      const int out_idx = row * TopK + k_idx;
      topk_weights[out_idx] *= inv_selected_sum;
    }
  }
}

template <typename T, typename SharedT, int NumExperts, int ValuesPerThread,
          int TopK>
__global__
__launch_bounds__(kWarpsPerCta *kWarpSize, 1) void topk_softmax_no_bias_warp_shared1_kernel_fixed_k(
    const T *__restrict__ gating_output,
    const SharedT *__restrict__ shared_gate_output,
    float *__restrict__ topk_weights, int32_t *__restrict__ topk_ids,
    int num_tokens, bool renormalize) {
  constexpr int OutTopK = TopK + 1;
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane = lane_id();
  const int row = blockIdx.x * kWarpsPerCta + warp_id;
  if (row >= num_tokens) {
    return;
  }

  float probs[ValuesPerThread];
  float thread_max = kFloatMinimum;

#pragma unroll
  for (int i = 0; i < ValuesPerThread; ++i) {
    const int expert = lane + i * kWarpSize;
    const float val = to_float(gating_output[row * NumExperts + expert]);
    probs[i] = val;
    thread_max = fmaxf(thread_max, val);
  }

  const float row_max = warp_max(thread_max);
  float thread_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ValuesPerThread; ++i) {
    probs[i] = __expf(probs[i] - row_max);
    thread_sum += probs[i];
  }
  const float inv_row_sum = 1.0f / warp_sum(thread_sum);
#pragma unroll
  for (int i = 0; i < ValuesPerThread; ++i) {
    probs[i] *= inv_row_sum;
  }

  float selected_sum = 0.0f;
#pragma unroll
  for (int k_idx = 0; k_idx < TopK; ++k_idx) {
    float max_val = probs[0];
    int max_idx = lane;
#pragma unroll
    for (int i = 1; i < ValuesPerThread; ++i) {
      const int expert = lane + i * kWarpSize;
      const float val = probs[i];
      if (val > max_val) {
        max_val = val;
        max_idx = expert;
      }
    }
    warp_argmax(max_val, max_idx);

    if (lane == 0) {
      const int out_idx = row * OutTopK + k_idx;
      topk_weights[out_idx] = max_val;
      topk_ids[out_idx] = max_idx;
      selected_sum += max_val;
    }

#pragma unroll
    for (int i = 0; i < ValuesPerThread; ++i) {
      const int expert = lane + i * kWarpSize;
      if (expert == max_idx) {
        probs[i] = kFloatMinimum;
      }
    }
  }

  if (lane == 0) {
    if (renormalize) {
      const float inv_selected_sum = 1.0f / selected_sum;
#pragma unroll
      for (int k_idx = 0; k_idx < TopK; ++k_idx) {
        topk_weights[row * OutTopK + k_idx] *= inv_selected_sum;
      }
    }
    const int shared_out_idx = row * OutTopK + TopK;
    const SharedT shared_weight =
        from_float<SharedT>(stable_sigmoid(to_float(shared_gate_output[row])));
    topk_weights[shared_out_idx] = to_float(shared_weight);
    topk_ids[shared_out_idx] = NumExperts;
  }
}

template <typename T, int NumExperts, int ValuesPerThread, int TopK>
__global__
__launch_bounds__(kWarpsPerCta *kWarpSize, 1) void topk_softmax_no_bias_renorm_warp_kernel_fixed_k(
    const T *__restrict__ gating_output, float *__restrict__ topk_weights,
    int32_t *__restrict__ topk_ids, int num_tokens) {
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane = lane_id();
  const int row = blockIdx.x * kWarpsPerCta + warp_id;
  if (row >= num_tokens) {
    return;
  }

  float logits[ValuesPerThread];

#pragma unroll
  for (int i = 0; i < ValuesPerThread; ++i) {
    const int expert = lane + i * kWarpSize;
    const float val = to_float(gating_output[row * NumExperts + expert]);
    logits[i] = val;
  }

  float selected_logits[TopK];
  int selected_ids[TopK];

#pragma unroll
  for (int k_idx = 0; k_idx < TopK; ++k_idx) {
    float max_val = logits[0];
    int max_idx = lane;
#pragma unroll
    for (int i = 1; i < ValuesPerThread; ++i) {
      const int expert = lane + i * kWarpSize;
      const float val = logits[i];
      if (val > max_val) {
        max_val = val;
        max_idx = expert;
      }
    }
    warp_argmax(max_val, max_idx);

    if (lane == 0) {
      selected_logits[k_idx] = max_val;
      selected_ids[k_idx] = max_idx;
    }

#pragma unroll
    for (int i = 0; i < ValuesPerThread; ++i) {
      const int expert = lane + i * kWarpSize;
      if (expert == max_idx) {
        logits[i] = kFloatMinimum;
      }
    }
  }

  if (lane == 0) {
    const float selected_max = selected_logits[0];
    selected_logits[0] = 1.0f;
    float selected_sum = 1.0f;
#pragma unroll
    for (int k_idx = 1; k_idx < TopK; ++k_idx) {
      selected_logits[k_idx] = __expf(selected_logits[k_idx] - selected_max);
      selected_sum += selected_logits[k_idx];
    }
    const float inv_selected_sum = 1.0f / selected_sum;
#pragma unroll
    for (int k_idx = 0; k_idx < TopK; ++k_idx) {
      const int out_idx = row * TopK + k_idx;
      topk_weights[out_idx] = selected_logits[k_idx] * inv_selected_sum;
      topk_ids[out_idx] = selected_ids[k_idx];
    }
  }
}

template <typename T, int NumExperts, int ValuesPerThread, int TopK>
__global__ __launch_bounds__(
    kWarpsPerCta * kWarpSize,
    1) void topk_softmax_no_bias_renorm_warp_combined_shared1_kernel_fixed_k(
    const T *__restrict__ combined_gating_output,
    float *__restrict__ topk_weights, int32_t *__restrict__ topk_ids,
    int num_tokens) {
  constexpr int InputExperts = NumExperts + 1;
  constexpr int OutTopK = TopK + 1;

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane = lane_id();
  const int row = blockIdx.x * kWarpsPerCta + warp_id;
  if (row >= num_tokens) {
    return;
  }

  float logits[ValuesPerThread];
#pragma unroll
  for (int i = 0; i < ValuesPerThread; ++i) {
    const int expert = lane + i * kWarpSize;
    logits[i] =
        to_float(combined_gating_output[row * InputExperts + expert]);
  }

  float selected_logits[TopK];
  int selected_ids[TopK];
#pragma unroll
  for (int k_idx = 0; k_idx < TopK; ++k_idx) {
    float max_val = logits[0];
    int max_idx = lane;
#pragma unroll
    for (int i = 1; i < ValuesPerThread; ++i) {
      const int expert = lane + i * kWarpSize;
      const float val = logits[i];
      if (val > max_val) {
        max_val = val;
        max_idx = expert;
      }
    }
    warp_argmax(max_val, max_idx);
    if (lane == 0) {
      selected_logits[k_idx] = max_val;
      selected_ids[k_idx] = max_idx;
    }
#pragma unroll
    for (int i = 0; i < ValuesPerThread; ++i) {
      const int expert = lane + i * kWarpSize;
      if (expert == max_idx) {
        logits[i] = kFloatMinimum;
      }
    }
  }

  if (lane == 0) {
    const float selected_max = selected_logits[0];
    selected_logits[0] = 1.0f;
    float selected_sum = 1.0f;
#pragma unroll
    for (int k_idx = 1; k_idx < TopK; ++k_idx) {
      selected_logits[k_idx] = __expf(selected_logits[k_idx] - selected_max);
      selected_sum += selected_logits[k_idx];
    }
    const float inv_selected_sum = 1.0f / selected_sum;
#pragma unroll
    for (int k_idx = 0; k_idx < TopK; ++k_idx) {
      const int out_idx = row * OutTopK + k_idx;
      topk_weights[out_idx] = selected_logits[k_idx] * inv_selected_sum;
      topk_ids[out_idx] = selected_ids[k_idx];
    }

    const int shared_out_idx = row * OutTopK + TopK;
    const float shared_logit =
        to_float(combined_gating_output[row * InputExperts + NumExperts]);
    const T rounded_shared_weight = from_float<T>(stable_sigmoid(shared_logit));
    topk_weights[shared_out_idx] = to_float(rounded_shared_weight);
    topk_ids[shared_out_idx] = NumExperts;
  }
}

template <typename T, int NumExperts, int TopK>
__global__
__launch_bounds__(kWarpsPerCta *kWarpSize, 1) void topk_softmax_no_bias_renorm_halfwarp_kernel_fixed_k(
    const T *__restrict__ gating_output, float *__restrict__ topk_weights,
    int32_t *__restrict__ topk_ids, int num_tokens) {
  constexpr int ThreadsPerRow = 16;
  constexpr int ValuesPerThread = NumExperts / ThreadsPerRow;
  constexpr int RowsPerWarp = kWarpSize / ThreadsPerRow;
  constexpr int RowsPerCta = kWarpsPerCta * RowsPerWarp;

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane = lane_id();
  const int row_in_warp = lane / ThreadsPerRow;
  const int lane_in_row = lane - row_in_warp * ThreadsPerRow;
  const int row = blockIdx.x * RowsPerCta + warp_id * RowsPerWarp + row_in_warp;
  if (row >= num_tokens) {
    return;
  }

  float logits[ValuesPerThread];

#pragma unroll
  for (int i = 0; i < ValuesPerThread; ++i) {
    const int expert = lane_in_row + i * ThreadsPerRow;
    const float val = to_float(gating_output[row * NumExperts + expert]);
    logits[i] = val;
  }

  float selected_logits[TopK];
  int selected_ids[TopK];

#pragma unroll
  for (int k_idx = 0; k_idx < TopK; ++k_idx) {
    float max_val = logits[0];
    int max_idx = lane_in_row;
#pragma unroll
    for (int i = 1; i < ValuesPerThread; ++i) {
      const int expert = lane_in_row + i * ThreadsPerRow;
      const float val = logits[i];
      if (val > max_val) {
        max_val = val;
        max_idx = expert;
      }
    }

#pragma unroll
    for (int mask = ThreadsPerRow / 2; mask > 0; mask >>= 1) {
      const float other_val =
          __shfl_xor_sync(0xffffffff, max_val, mask, ThreadsPerRow);
      const int other_idx =
          __shfl_xor_sync(0xffffffff, max_idx, mask, ThreadsPerRow);
      if (other_val > max_val ||
          (other_val == max_val && other_idx < max_idx)) {
        max_val = other_val;
        max_idx = other_idx;
      }
    }

    if (lane_in_row == 0) {
      selected_logits[k_idx] = max_val;
      selected_ids[k_idx] = max_idx;
    }

#pragma unroll
    for (int i = 0; i < ValuesPerThread; ++i) {
      const int expert = lane_in_row + i * ThreadsPerRow;
      if (expert == max_idx) {
        logits[i] = kFloatMinimum;
      }
    }
  }

  if (lane_in_row == 0) {
    const float selected_max = selected_logits[0];
    selected_logits[0] = 1.0f;
    float selected_sum = 1.0f;
#pragma unroll
    for (int k_idx = 1; k_idx < TopK; ++k_idx) {
      selected_logits[k_idx] = __expf(selected_logits[k_idx] - selected_max);
      selected_sum += selected_logits[k_idx];
    }
    const float inv_selected_sum = 1.0f / selected_sum;
#pragma unroll
    for (int k_idx = 0; k_idx < TopK; ++k_idx) {
      const int out_idx = row * TopK + k_idx;
      topk_weights[out_idx] = selected_logits[k_idx] * inv_selected_sum;
      topk_ids[out_idx] = selected_ids[k_idx];
    }
  }
}

template <int TopK>
__global__
__launch_bounds__(kWarpsPerCta *kWarpSize, 1) void topk_softmax_half_e1024_no_bias_renorm_warp_kernel_fixed_k(
    const half *__restrict__ gating_output, float *__restrict__ topk_weights,
    int32_t *__restrict__ topk_ids, int num_tokens) {
  constexpr int NumExperts = 1024;
  constexpr int ValuesPerThread = 32;
  constexpr int Half2PerThread = ValuesPerThread / 2;

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane = lane_id();
  const int row = blockIdx.x * kWarpsPerCta + warp_id;
  if (row >= num_tokens) {
    return;
  }

  float logits[ValuesPerThread];
  const __half2 *row_ptr =
      reinterpret_cast<const __half2 *>(gating_output + row * NumExperts);

#pragma unroll
  for (int i = 0; i < Half2PerThread; ++i) {
    const int pair_idx = lane + i * kWarpSize;
    const __half2 packed = row_ptr[pair_idx];
    const float2 vals = __half22float2(packed);
    logits[i * 2] = vals.x;
    logits[i * 2 + 1] = vals.y;
  }

  float selected_logits[TopK];
  int selected_ids[TopK];

#pragma unroll
  for (int k_idx = 0; k_idx < TopK; ++k_idx) {
    float max_val = logits[0];
    int max_idx = lane * 2;
#pragma unroll
    for (int i = 1; i < ValuesPerThread; ++i) {
      const int pair_idx = i >> 1;
      const int expert = lane * 2 + pair_idx * (kWarpSize * 2) + (i & 1);
      const float val = logits[i];
      if (val > max_val) {
        max_val = val;
        max_idx = expert;
      }
    }
    warp_argmax(max_val, max_idx);

    if (lane == 0) {
      selected_logits[k_idx] = max_val;
      selected_ids[k_idx] = max_idx;
    }

#pragma unroll
    for (int i = 0; i < ValuesPerThread; ++i) {
      const int pair_idx = i >> 1;
      const int expert = lane * 2 + pair_idx * (kWarpSize * 2) + (i & 1);
      if (expert == max_idx) {
        logits[i] = kFloatMinimum;
      }
    }
  }

  if (lane == 0) {
    const float selected_max = selected_logits[0];
    selected_logits[0] = 1.0f;
    float selected_sum = 1.0f;
#pragma unroll
    for (int k_idx = 1; k_idx < TopK; ++k_idx) {
      selected_logits[k_idx] = __expf(selected_logits[k_idx] - selected_max);
      selected_sum += selected_logits[k_idx];
    }
    const float inv_selected_sum = 1.0f / selected_sum;
#pragma unroll
    for (int k_idx = 0; k_idx < TopK; ++k_idx) {
      const int out_idx = row * TopK + k_idx;
      topk_weights[out_idx] = selected_logits[k_idx] * inv_selected_sum;
      topk_ids[out_idx] = selected_ids[k_idx];
    }
  }
}

template <typename T, int NumExperts, int ValuesPerThread, int TopK>
__global__ __launch_bounds__(
    kWarpsPerCta *kWarpSize,
    1) void topk_sigmoid_no_bias_warp_kernel_fixed_k(const T
                                                         *__restrict__ gating_output,
                                                     float
                                                         *__restrict__ topk_weights,
                                                     int32_t
                                                         *__restrict__ topk_ids,
                                                     int num_tokens,
                                                     bool renormalize) {
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane = lane_id();
  const int row = blockIdx.x * kWarpsPerCta + warp_id;
  if (row >= num_tokens) {
    return;
  }

  float logits[ValuesPerThread];

#pragma unroll
  for (int i = 0; i < ValuesPerThread; ++i) {
    const int expert = lane + i * kWarpSize;
    logits[i] = to_float(gating_output[row * NumExperts + expert]);
  }

  float selected_sum = 0.0f;
#pragma unroll
  for (int k_idx = 0; k_idx < TopK; ++k_idx) {
    float max_val = logits[0];
    int max_idx = lane;
#pragma unroll
    for (int i = 1; i < ValuesPerThread; ++i) {
      const int expert = lane + i * kWarpSize;
      const float val = logits[i];
      if (val > max_val || (val == max_val && expert < max_idx)) {
        max_val = val;
        max_idx = expert;
      }
    }
    warp_argmax(max_val, max_idx);

    const float selected_prob = stable_sigmoid(max_val);
    if (lane == 0) {
      const int out_idx = row * TopK + k_idx;
      topk_weights[out_idx] = selected_prob;
      topk_ids[out_idx] = max_idx;
      selected_sum += selected_prob;
    }

#pragma unroll
    for (int i = 0; i < ValuesPerThread; ++i) {
      const int expert = lane + i * kWarpSize;
      if (expert == max_idx) {
        logits[i] = kFloatMinimum;
      }
    }
  }

  if (renormalize && lane == 0) {
    const float inv_selected_sum = 1.0f / selected_sum;
#pragma unroll
    for (int k_idx = 0; k_idx < TopK; ++k_idx) {
      const int out_idx = row * TopK + k_idx;
      topk_weights[out_idx] *= inv_selected_sum;
    }
  }
}

template <typename T, bool IsSoftmax>
__global__ void topk_block_kernel(
    const T *__restrict__ gating_output, float *__restrict__ topk_weights,
    int32_t *__restrict__ topk_ids, const float *__restrict__ correction_bias,
    int num_tokens, int num_experts, int topk, int block_width,
    bool renormalize, float moe_softcapping, bool has_correction_bias) {
  extern __shared__ unsigned char smem[];
  float *logits_or_probs = reinterpret_cast<float *>(smem);
  float *choice = logits_or_probs + block_width;
  float *reduce_vals = choice + block_width;
  int32_t *reduce_idxs = reinterpret_cast<int32_t *>(reduce_vals + block_width);
  float *selected_sum = reinterpret_cast<float *>(reduce_idxs + block_width);

  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const int row_base = row * num_experts;

  float val = kFloatMinimum;
  if (tid < num_experts) {
    val = to_float(gating_output[row_base + tid]);
    if constexpr (IsSoftmax) {
      if (has_correction_bias) {
        val += correction_bias[tid];
      }
      if (moe_softcapping > 0.0f) {
        val = tanhf(val / moe_softcapping) * moe_softcapping;
      }
    } else {
      val = stable_sigmoid(val);
    }
  }
  logits_or_probs[tid] = val;
  reduce_vals[tid] = (tid < num_experts) ? val : kFloatMinimum;
  reduce_idxs[tid] = tid;
  __syncthreads();

  if constexpr (IsSoftmax) {
    for (int stride = block_width >> 1; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduce_vals[tid] = fmaxf(reduce_vals[tid], reduce_vals[tid + stride]);
      }
      __syncthreads();
    }

    const float row_max = reduce_vals[0];
    float prob = 0.0f;
    if (tid < num_experts) {
      prob = __expf(logits_or_probs[tid] - row_max);
    }
    logits_or_probs[tid] = prob;
    reduce_vals[tid] = prob;
    __syncthreads();

    for (int stride = block_width >> 1; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduce_vals[tid] += reduce_vals[tid + stride];
      }
      __syncthreads();
    }

    if (tid < num_experts) {
      logits_or_probs[tid] *= 1.0f / reduce_vals[0];
      choice[tid] = logits_or_probs[tid];
    } else {
      choice[tid] = kFloatMinimum;
    }
  } else {
    if (tid < num_experts) {
      choice[tid] = has_correction_bias ? val + correction_bias[tid] : val;
    } else {
      choice[tid] = kFloatMinimum;
    }
  }

  if (tid == 0) {
    selected_sum[0] = 0.0f;
  }
  __syncthreads();

  for (int k_idx = 0; k_idx < topk; ++k_idx) {
    reduce_vals[tid] = choice[tid];
    reduce_idxs[tid] = tid;
    __syncthreads();

    for (int stride = block_width >> 1; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const float rhs_val = reduce_vals[tid + stride];
        const int rhs_idx = reduce_idxs[tid + stride];
        if (rhs_val > reduce_vals[tid] ||
            (rhs_val == reduce_vals[tid] && rhs_idx < reduce_idxs[tid])) {
          reduce_vals[tid] = rhs_val;
          reduce_idxs[tid] = rhs_idx;
        }
      }
      __syncthreads();
    }

    if (tid == 0) {
      const int out_idx = row * topk + k_idx;
      const int expert = reduce_idxs[0];
      const float selected_prob = logits_or_probs[expert];
      topk_weights[out_idx] = selected_prob;
      topk_ids[out_idx] = expert;
      selected_sum[0] += selected_prob;
      choice[expert] = kFloatMinimum;
    }
    __syncthreads();
  }

  if (renormalize && tid < topk) {
    const int out_idx = row * topk + tid;
    topk_weights[out_idx] *= 1.0f / selected_sum[0];
  }
}

int next_power_of_2(int value) {
  int out = 1;
  while (out < value) {
    out <<= 1;
  }
  return out;
}

template <typename T, typename SharedT>
void launch_topk_shared1(ffi::TensorView topk_weights, ffi::TensorView topk_ids,
                         ffi::TensorView gating_output, bool renormalize,
                         ffi::TensorView shared_gate_output) {
  const int num_tokens = static_cast<int>(gating_output.size(0));
  if (num_tokens == 0) {
    return;
  }

  const T *input_ptr = static_cast<const T *>(gating_output.data_ptr());
  const SharedT *shared_ptr =
      static_cast<const SharedT *>(shared_gate_output.data_ptr());
  float *weights_ptr = static_cast<float *>(topk_weights.data_ptr());
  int32_t *ids_ptr = static_cast<int32_t *>(topk_ids.data_ptr());
  ffi::MUSADeviceGuard device_guard(gating_output.device().device_id);
  musaStream_t stream = get_stream(gating_output.device());
  const int blocks = (num_tokens + kWarpsPerCta - 1) / kWarpsPerCta;
  topk_softmax_no_bias_warp_shared1_kernel_fixed_k<T, SharedT, 256, 8, 8>
      <<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
          input_ptr, shared_ptr, weights_ptr, ids_ptr, num_tokens, renormalize);

  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA topk fused shared-expert kernel failed: "
      << musaGetErrorString(err);
}

template <typename T, bool IsSoftmax>
void launch_topk(ffi::TensorView topk_weights, ffi::TensorView topk_ids,
                 ffi::TensorView gating_output, bool renormalize,
                 float moe_softcapping, ffi::TensorView correction_bias,
                 bool has_correction_bias, int num_fused_shared_experts) {
  const int num_tokens = static_cast<int>(gating_output.size(0));
  const int num_experts = static_cast<int>(gating_output.size(1));
  const int topk = static_cast<int>(topk_weights.size(1));
  if (num_tokens == 0 || topk == 0) {
    return;
  }
  const float *bias_ptr =
      has_correction_bias
          ? static_cast<const float *>(correction_bias.data_ptr())
          : nullptr;
  const T *input_ptr = static_cast<const T *>(gating_output.data_ptr());
  float *weights_ptr = static_cast<float *>(topk_weights.data_ptr());
  int32_t *ids_ptr = static_cast<int32_t *>(topk_ids.data_ptr());

  ffi::MUSADeviceGuard device_guard(gating_output.device().device_id);
  musaStream_t stream = get_stream(gating_output.device());

  if constexpr (IsSoftmax) {
    if (num_fused_shared_experts == 1 && num_experts == 257 && topk == 9 &&
        renormalize && !has_correction_bias && moe_softcapping <= 0.0f) {
      constexpr int values_per_thread = 8;
      const int blocks = (num_tokens + kWarpsPerCta - 1) / kWarpsPerCta;
      topk_softmax_no_bias_renorm_warp_combined_shared1_kernel_fixed_k<
          T, 256, values_per_thread, 8>
          <<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
              input_ptr, weights_ptr, ids_ptr, num_tokens);
      const musaError_t err = musaGetLastError();
      TVM_FFI_ICHECK_EQ(err, musaSuccess)
          << "MUSA fused-shared topk kernel failed: "
          << musaGetErrorString(err);
      return;
    }
  }

  TVM_FFI_ICHECK_EQ(num_fused_shared_experts, 0)
      << "Fused shared top-k currently supports only softmax-renorm "
         "E=256, topk=8, shared=1.";

  if (num_experts == 128) {
    constexpr int values_per_thread = 4;
    const int blocks = (num_tokens + kWarpsPerCta - 1) / kWarpsPerCta;
    if constexpr (IsSoftmax) {
      if (topk == 8 && !has_correction_bias && moe_softcapping <= 0.0f) {
        if (renormalize) {
          const int halfwarp_rows_per_cta = kWarpsPerCta * 2;
          const int halfwarp_blocks =
              (num_tokens + halfwarp_rows_per_cta - 1) / halfwarp_rows_per_cta;
          topk_softmax_no_bias_renorm_halfwarp_kernel_fixed_k<T, 128, 8>
              <<<halfwarp_blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
                  input_ptr, weights_ptr, ids_ptr, num_tokens);
        } else {
          topk_softmax_no_bias_warp_kernel_fixed_k<T, 128, values_per_thread, 8>
              <<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
                  input_ptr, weights_ptr, ids_ptr, num_tokens, renormalize);
        }
      } else {
        topk_softmax_warp_kernel<T, 128, values_per_thread>
            <<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
                input_ptr, weights_ptr, ids_ptr, bias_ptr, num_tokens, topk,
                renormalize, moe_softcapping, has_correction_bias);
      }
    } else if (!has_correction_bias) {
      topk_sigmoid_no_bias_warp_kernel<T, 128, values_per_thread>
          <<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
              input_ptr, weights_ptr, ids_ptr, num_tokens, topk, renormalize);
    } else {
      topk_sigmoid_warp_kernel<T, 128, values_per_thread>
          <<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
              input_ptr, weights_ptr, ids_ptr, bias_ptr, num_tokens, topk,
              renormalize, has_correction_bias);
    }
  } else if (num_experts == 256) {
    constexpr int values_per_thread = 8;
    const int blocks = (num_tokens + kWarpsPerCta - 1) / kWarpsPerCta;
    if constexpr (IsSoftmax) {
      if (topk == 8 && !has_correction_bias && moe_softcapping <= 0.0f) {
        if (renormalize) {
          // MUSA: e256 renorm goes one-warp-per-row, matching the e1024 path.
          topk_softmax_no_bias_renorm_warp_kernel_fixed_k<T, 256,
                                                          values_per_thread, 8>
              <<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
                  input_ptr, weights_ptr, ids_ptr, num_tokens);
        } else {
          topk_softmax_no_bias_warp_kernel_fixed_k<T, 256, values_per_thread, 8>
              <<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
                  input_ptr, weights_ptr, ids_ptr, num_tokens, renormalize);
        }
      } else {
        topk_softmax_warp_kernel<T, 256, values_per_thread>
            <<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
                input_ptr, weights_ptr, ids_ptr, bias_ptr, num_tokens, topk,
                renormalize, moe_softcapping, has_correction_bias);
      }
    } else if (!has_correction_bias) {
      topk_sigmoid_no_bias_warp_kernel<T, 256, values_per_thread>
          <<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
              input_ptr, weights_ptr, ids_ptr, num_tokens, topk, renormalize);
    } else {
      topk_sigmoid_warp_kernel<T, 256, values_per_thread>
          <<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
              input_ptr, weights_ptr, ids_ptr, bias_ptr, num_tokens, topk,
              renormalize, has_correction_bias);
    }
  } else if (num_experts == 512) {
    constexpr int values_per_thread = 16;
    const int blocks = (num_tokens + kWarpsPerCta - 1) / kWarpsPerCta;
    if constexpr (IsSoftmax) {
      if (topk == 8 && !has_correction_bias && moe_softcapping <= 0.0f) {
        if (renormalize) {
          const int halfwarp_rows_per_cta = kWarpsPerCta * 2;
          const int halfwarp_blocks =
              (num_tokens + halfwarp_rows_per_cta - 1) / halfwarp_rows_per_cta;
          topk_softmax_no_bias_renorm_halfwarp_kernel_fixed_k<T, 512, 8>
              <<<halfwarp_blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
                  input_ptr, weights_ptr, ids_ptr, num_tokens);
        } else {
          topk_softmax_no_bias_warp_kernel_fixed_k<T, 512, values_per_thread, 8>
              <<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
                  input_ptr, weights_ptr, ids_ptr, num_tokens, renormalize);
        }
      } else {
        topk_softmax_warp_kernel<T, 512, values_per_thread>
            <<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
                input_ptr, weights_ptr, ids_ptr, bias_ptr, num_tokens, topk,
                renormalize, moe_softcapping, has_correction_bias);
      }
    } else if (!has_correction_bias) {
      topk_sigmoid_no_bias_warp_kernel<T, 512, values_per_thread>
          <<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
              input_ptr, weights_ptr, ids_ptr, num_tokens, topk, renormalize);
    } else {
      topk_sigmoid_warp_kernel<T, 512, values_per_thread>
          <<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
              input_ptr, weights_ptr, ids_ptr, bias_ptr, num_tokens, topk,
              renormalize, has_correction_bias);
    }
  } else if (num_experts == 1024) {
    constexpr int values_per_thread = 32;
    const int blocks = (num_tokens + kWarpsPerCta - 1) / kWarpsPerCta;
    if constexpr (IsSoftmax) {
      if (topk == 8 && !has_correction_bias && moe_softcapping <= 0.0f) {
        if (renormalize) {
          if constexpr (std::is_same_v<T, half>) {
            topk_softmax_half_e1024_no_bias_renorm_warp_kernel_fixed_k<8>
                <<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
                    input_ptr, weights_ptr, ids_ptr, num_tokens);
          } else {
            topk_softmax_no_bias_renorm_warp_kernel_fixed_k<
                T, 1024, values_per_thread, 8>
                <<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
                    input_ptr, weights_ptr, ids_ptr, num_tokens);
          }
        } else {
          topk_softmax_no_bias_warp_kernel_fixed_k<T, 1024, values_per_thread,
                                                   8>
              <<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
                  input_ptr, weights_ptr, ids_ptr, num_tokens, renormalize);
        }
      } else {
        topk_softmax_warp_kernel<T, 1024, values_per_thread>
            <<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
                input_ptr, weights_ptr, ids_ptr, bias_ptr, num_tokens, topk,
                renormalize, moe_softcapping, has_correction_bias);
      }
    } else if (!has_correction_bias) {
      if (topk == 8 && num_tokens <= 512) {
        topk_sigmoid_no_bias_warp_kernel_fixed_k<T, 1024, values_per_thread, 8>
            <<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
                input_ptr, weights_ptr, ids_ptr, num_tokens, renormalize);
      } else {
        topk_sigmoid_no_bias_warp_kernel<T, 1024, values_per_thread>
            <<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
                input_ptr, weights_ptr, ids_ptr, num_tokens, topk, renormalize);
      }
    } else {
      topk_sigmoid_warp_kernel<T, 1024, values_per_thread>
          <<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
              input_ptr, weights_ptr, ids_ptr, bias_ptr, num_tokens, topk,
              renormalize, has_correction_bias);
    }
  } else {
    const int block_width = next_power_of_2(num_experts);
    const size_t smem_bytes = 3 * block_width * sizeof(float) +
                              block_width * sizeof(int32_t) + sizeof(float);
    topk_block_kernel<T, IsSoftmax>
        <<<num_tokens, block_width, smem_bytes, stream>>>(
            input_ptr, weights_ptr, ids_ptr, bias_ptr, num_tokens, num_experts,
            topk, block_width, renormalize, moe_softcapping,
            has_correction_bias);
  }

  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA topk kernel failed: " << musaGetErrorString(err);
}

void check_topk_inputs(ffi::TensorView topk_weights, ffi::TensorView topk_ids,
                       ffi::TensorView gating_output,
                       ffi::TensorView correction_bias,
                       bool has_correction_bias,
                       ffi::TensorView shared_gate_output,
                       int num_fused_shared_experts, bool has_shared_experts) {
  CHECK_MUSA_CONTIGUOUS(topk_weights);
  CHECK_MUSA_CONTIGUOUS(topk_ids);
  CHECK_MUSA_CONTIGUOUS(gating_output);
  TVM_FFI_ICHECK_EQ(topk_weights.device().device_id,
                    gating_output.device().device_id);
  TVM_FFI_ICHECK_EQ(topk_ids.device().device_id,
                    gating_output.device().device_id);
  TVM_FFI_ICHECK_EQ(gating_output.ndim(), 2);
  TVM_FFI_ICHECK_EQ(topk_weights.ndim(), 2);
  TVM_FFI_ICHECK_EQ(topk_ids.ndim(), 2);
  TVM_FFI_ICHECK_EQ(topk_weights.size(0), gating_output.size(0));
  TVM_FFI_ICHECK_EQ(topk_ids.size(0), gating_output.size(0));
  TVM_FFI_ICHECK_EQ(topk_ids.size(1), topk_weights.size(1));
  const int routed_topk =
      (has_shared_experts || num_fused_shared_experts > 0)
          ? static_cast<int>(topk_weights.size(1)) - num_fused_shared_experts
          : static_cast<int>(topk_weights.size(1));
  TVM_FFI_ICHECK_GE(routed_topk, 0);
  TVM_FFI_ICHECK_LE(routed_topk, gating_output.size(1));
  TVM_FFI_ICHECK_LE(gating_output.size(1), 1024);
  TVM_FFI_ICHECK(dtype_equal(topk_weights.dtype(), dl_float32));
  TVM_FFI_ICHECK(dtype_equal(topk_ids.dtype(), dl_int32));
  if (has_correction_bias) {
    CHECK_MUSA_CONTIGUOUS(correction_bias);
    TVM_FFI_ICHECK_EQ(correction_bias.device().device_id,
                      gating_output.device().device_id);
    TVM_FFI_ICHECK_EQ(correction_bias.ndim(), 1);
    TVM_FFI_ICHECK_EQ(correction_bias.size(0), gating_output.size(1));
    TVM_FFI_ICHECK(dtype_equal(correction_bias.dtype(), dl_float32));
  }
  if (has_shared_experts) {
    CHECK_MUSA_CONTIGUOUS(shared_gate_output);
    TVM_FFI_ICHECK_EQ(shared_gate_output.device().device_id,
                      gating_output.device().device_id);
    TVM_FFI_ICHECK_EQ(shared_gate_output.ndim(), 2);
    TVM_FFI_ICHECK_EQ(shared_gate_output.size(0), gating_output.size(0));
    TVM_FFI_ICHECK_EQ(shared_gate_output.size(1), 1);
    TVM_FFI_ICHECK_EQ(num_fused_shared_experts, 1);
    TVM_FFI_ICHECK_EQ(gating_output.size(1), 256);
    TVM_FFI_ICHECK_EQ(routed_topk, 8);
    TVM_FFI_ICHECK_EQ(topk_weights.size(1), 9);
    TVM_FFI_ICHECK(dtype_equal(shared_gate_output.dtype(), dl_float32) ||
                   dtype_equal(shared_gate_output.dtype(), dl_float16) ||
                   dtype_equal(shared_gate_output.dtype(), dl_bfloat16));
  }
  if (!has_shared_experts && num_fused_shared_experts > 0) {
    TVM_FFI_ICHECK_EQ(num_fused_shared_experts, 1);
    TVM_FFI_ICHECK_EQ(gating_output.size(1), 257);
    TVM_FFI_ICHECK_EQ(routed_topk, 8);
    TVM_FFI_ICHECK_EQ(topk_weights.size(1), 9);
  }
}

template <bool IsSoftmax>
void dispatch_topk(ffi::TensorView topk_weights, ffi::TensorView topk_ids,
                   ffi::TensorView gating_output, bool renormalize,
                   float moe_softcapping, ffi::TensorView correction_bias,
                   bool has_correction_bias, ffi::TensorView shared_gate_output,
                   int num_fused_shared_experts, bool has_shared_experts) {
  if (has_shared_experts) {
    if constexpr (!IsSoftmax) {
      TVM_FFI_THROW(ValueError)
          << "Fused shared-expert topk only supports softmax routing";
    }
    TVM_FFI_ICHECK(!has_correction_bias);
    TVM_FFI_ICHECK(moe_softcapping <= 0.0f);

#define DISPATCH_SHARED(GateT)                                                 \
  if (dtype_equal(shared_gate_output.dtype(), dl_float32)) {                   \
    launch_topk_shared1<GateT, float>(topk_weights, topk_ids, gating_output,   \
                                      renormalize, shared_gate_output);        \
  } else if (dtype_equal(shared_gate_output.dtype(), dl_float16)) {            \
    launch_topk_shared1<GateT, half>(topk_weights, topk_ids, gating_output,    \
                                     renormalize, shared_gate_output);         \
  } else if (dtype_equal(shared_gate_output.dtype(), dl_bfloat16)) {           \
    launch_topk_shared1<GateT, __mt_bfloat16>(topk_weights, topk_ids,          \
                                              gating_output, renormalize,      \
                                              shared_gate_output);             \
  } else {                                                                     \
    TVM_FFI_THROW(ValueError) << "Unsupported shared_gate_output dtype";       \
  }

    if (dtype_equal(gating_output.dtype(), dl_float32)) {
      DISPATCH_SHARED(float)
    } else if (dtype_equal(gating_output.dtype(), dl_float16)) {
      DISPATCH_SHARED(half)
    } else if (dtype_equal(gating_output.dtype(), dl_bfloat16)) {
      DISPATCH_SHARED(__mt_bfloat16)
    } else {
      TVM_FFI_THROW(ValueError) << "Unsupported gating_output dtype";
    }
#undef DISPATCH_SHARED
    return;
  }

  if (dtype_equal(gating_output.dtype(), dl_float32)) {
    launch_topk<float, IsSoftmax>(topk_weights, topk_ids, gating_output,
                                  renormalize, moe_softcapping, correction_bias,
                                  has_correction_bias,
                                  num_fused_shared_experts);
  } else if (dtype_equal(gating_output.dtype(), dl_float16)) {
    launch_topk<half, IsSoftmax>(topk_weights, topk_ids, gating_output,
                                 renormalize, moe_softcapping, correction_bias,
                                 has_correction_bias,
                                 num_fused_shared_experts);
  } else if (dtype_equal(gating_output.dtype(), dl_bfloat16)) {
    launch_topk<__mt_bfloat16, IsSoftmax>(topk_weights, topk_ids, gating_output,
                                          renormalize, moe_softcapping,
                                          correction_bias, has_correction_bias,
                                          num_fused_shared_experts);
  } else {
    TVM_FFI_THROW(ValueError) << "Unsupported gating_output dtype";
  }
}

void sgl_musa_topk_softmax(
    ffi::TensorView topk_weights, ffi::TensorView topk_ids,
    ffi::TensorView gating_output, bool renormalize, double moe_softcapping,
    ffi::TensorView correction_bias, bool has_correction_bias,
    ffi::TensorView shared_gate_output, int num_fused_shared_experts,
    bool has_shared_experts) {
  check_topk_inputs(topk_weights, topk_ids, gating_output, correction_bias,
                    has_correction_bias, shared_gate_output,
                    num_fused_shared_experts, has_shared_experts);
  dispatch_topk<true>(topk_weights, topk_ids, gating_output, renormalize,
                      static_cast<float>(moe_softcapping), correction_bias,
                      has_correction_bias, shared_gate_output,
                      num_fused_shared_experts, has_shared_experts);
}

void sgl_musa_topk_sigmoid(ffi::TensorView topk_weights,
                           ffi::TensorView topk_ids,
                           ffi::TensorView gating_output, bool renormalize,
                           ffi::TensorView correction_bias,
                           bool has_correction_bias,
                           ffi::TensorView shared_gate_output,
                           int num_fused_shared_experts,
                           bool has_shared_experts) {
  check_topk_inputs(topk_weights, topk_ids, gating_output, correction_bias,
                    has_correction_bias, shared_gate_output,
                    num_fused_shared_experts, has_shared_experts);
  dispatch_topk<false>(topk_weights, topk_ids, gating_output, renormalize, 0.0f,
                       correction_bias, has_correction_bias, shared_gate_output,
                       num_fused_shared_experts, has_shared_experts);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_topk_softmax, sgl_musa_topk_softmax);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_topk_sigmoid, sgl_musa_topk_sigmoid);
