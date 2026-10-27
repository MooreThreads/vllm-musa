// CAR v2 and fused CAR+RMSNorm are deliberately separate translation units
// selected by SGL_CUSTOM_AR_FUSED_RMSNORM.  The two implementations define
// overlapping Signal/RankData helpers; keeping one branch per JIT module
// prevents ODR/type-layout collisions and preserves the pure v2 ABI.
#if !defined(SGL_CUSTOM_AR_FUSED_RMSNORM) || !(SGL_CUSTOM_AR_FUSED_RMSNORM)
#include "../common.h"
#include "../device_utils.h"

#include <musa_runtime.h>
#include <musa_bf16.h>
#include <musa_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace {

#ifndef SGL_CUSTOM_AR_THREADS
#define SGL_CUSTOM_AR_THREADS 512
#endif

#ifndef SGL_CUSTOM_AR_BLOCKS
#define SGL_CUSTOM_AR_BLOCKS 36
#endif

#ifndef SGL_CUSTOM_AR_VECTOR_LOAD
#define SGL_CUSTOM_AR_VECTOR_LOAD 0
#endif

#ifndef SGL_CUSTOM_AR_ATOMIC_BARRIER
#define SGL_CUSTOM_AR_ATOMIC_BARRIER 1
#endif

#ifndef SGL_CUSTOM_AR_MAX_BLOCKS
#define SGL_CUSTOM_AR_MAX_BLOCKS 120
#endif

#ifndef SGL_CUSTOM_AR_DYNAMIC_BLOCKS
#define SGL_CUSTOM_AR_DYNAMIC_BLOCKS 1
#endif

#ifndef SGL_CUSTOM_AR_PUSH_POLLING
#define SGL_CUSTOM_AR_PUSH_POLLING 0
#endif

#ifndef SGL_CUSTOM_AR_PUSH_16B_ASM
#define SGL_CUSTOM_AR_PUSH_16B_ASM 0
#endif

#ifndef SGL_CUSTOM_AR_2SHOT_DOUBLE_STORE
#define SGL_CUSTOM_AR_2SHOT_DOUBLE_STORE 0
#endif

#ifndef SGL_CUSTOM_AR_1SHOT_2RANK_SPECIAL
#define SGL_CUSTOM_AR_1SHOT_2RANK_SPECIAL 1
#endif

#ifndef SGL_CUSTOM_AR_PUSH_SKIP_START_BARRIER
#define SGL_CUSTOM_AR_PUSH_SKIP_START_BARRIER 0
#endif

constexpr int kMaxBlocks = SGL_CUSTOM_AR_MAX_BLOCKS;
// Signal is shared by independently JIT-compiled algorithm variants.  Its ABI
// must not depend on a per-shot launch tuning value.
constexpr int kSignalMaxBlocks = 512;
static_assert(kMaxBlocks <= kSignalMaxBlocks);
constexpr int kMaxThreadsPerBlock = 1024;
constexpr int kDefaultThreads = SGL_CUSTOM_AR_THREADS;
constexpr int kDefaultBlockLimit = SGL_CUSTOM_AR_BLOCKS;
constexpr int kMaxRanks = 8;
using FlagType = uint32_t;

struct alignas(128) PullSemaphore {
  FlagType flag;
  FlagType counter;
  FlagType self_counter[kMaxRanks];
  FlagType peer_counter[2][kMaxRanks];
};
static_assert(sizeof(PullSemaphore) == 128);

// Pull/fused and push controllers use distinct allocations, but every JIT
// variant must agree on the signal ABI. Overlay controller-specific state so
// both allocations stay compact and fused RMSNorm can share the pull layout.
struct alignas(128) Signal {
  union alignas(128) {
    PullSemaphore pull[kSignalMaxBlocks];
    FlagType push_epoch[kSignalMaxBlocks];
  };
  union alignas(128) {
    PullSemaphore publish[kSignalMaxBlocks];
    PullSemaphore push_ready[kSignalMaxBlocks];
  };
};
static_assert(
    sizeof(Signal) == 2 * kSignalMaxBlocks * sizeof(PullSemaphore));

template <typename T>
__device__ __forceinline__ T* get_tmp_buf(Signal* signal) {
  return reinterpret_cast<T*>(signal + 1);
}

struct __align__(16) RankData {
  const void* ptrs[kMaxRanks];
};

struct __align__(16) RankSignals {
  Signal* signals[kMaxRanks];
};

template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

template <typename T>
struct packed_t {
  using P = array_t<T, 16 / sizeof(T)>;
  using A = array_t<float, 16 / sizeof(T)>;
};

template <typename T>
struct fp_trait {};

template <>
struct fp_trait<half> {
  using type = uint16_t;
  static constexpr uint16_t pos_zero = 0x0000u;
  static constexpr uint16_t neg_zero = 0x8000u;
};

template <>
struct fp_trait<__mt_bfloat16> {
  using type = uint16_t;
  static constexpr uint16_t pos_zero = 0x0000u;
  static constexpr uint16_t neg_zero = 0x8000u;
};

template <>
struct fp_trait<float> {
  using type = uint32_t;
  static constexpr uint32_t pos_zero = 0x00000000u;
  static constexpr uint32_t neg_zero = 0x80000000u;
};

template <typename DType>
__device__ __forceinline__ void clear_pos_zero(DType& val) {
  using Trait = fp_trait<DType>;
  auto* ptr = reinterpret_cast<typename Trait::type*>(&val);
  if (*ptr == Trait::pos_zero) *ptr = Trait::neg_zero;
}

template <typename DType>
__device__ __forceinline__ bool is_pos_zero(const DType& val) {
  using Trait = fp_trait<DType>;
  const auto* ptr = reinterpret_cast<const typename Trait::type*>(&val);
  return *ptr == Trait::pos_zero;
}

template <typename DType>
__device__ __forceinline__ DType get_pos_zero() {
  using Trait = fp_trait<DType>;
  const auto value = Trait::pos_zero;
  return *reinterpret_cast<const DType*>(&value);
}

template <typename T, int N>
__device__ __forceinline__ void clear_pos_zero(array_t<T, N>& val) {
#pragma unroll
  for (int i = 0; i < N; ++i) {
    clear_pos_zero(val.data[i]);
  }
}

template <typename T, int N>
__device__ __forceinline__ bool has_pos_zero(array_t<T, N> val) {
  bool found = false;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    found |= is_pos_zero(val.data[i]);
  }
  return found;
}

template <typename P>
__device__ __forceinline__ P make_pos_zero_packet() {
  P out;
#pragma unroll
  for (int i = 0; i < P::size; ++i) {
    out.data[i] = get_pos_zero<typename P::type>();
  }
  return out;
}

template <typename P>
__device__ __forceinline__ P load_volatile_packet(const P* ptr) {
  static_assert(sizeof(P) == 16 && alignof(P) == 16);
#if SGL_CUSTOM_AR_PUSH_16B_ASM
  using i4 = int __attribute__((ext_vector_type(4)));
  uint4 raw;
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 310)
  i4 tmp = __musa_ldcv_v4i32(reinterpret_cast<const i4*>(ptr));
  raw = *reinterpret_cast<uint4*>(&tmp);
#else
  raw = *reinterpret_cast<const uint4*>(ptr);
#endif
  return *reinterpret_cast<const P*>(&raw);
#else
  uint4 raw;
  const auto* src = reinterpret_cast<const volatile uint32_t*>(ptr);
  raw.x = src[0];
  raw.y = src[1];
  raw.z = src[2];
  raw.w = src[3];
  return *reinterpret_cast<const P*>(&raw);
#endif
}

template <typename P>
__device__ __forceinline__ void store_volatile_packet(P* ptr, P value) {
  static_assert(sizeof(P) == 16 && alignof(P) == 16);
  const auto raw = *reinterpret_cast<const uint4*>(&value);
#if SGL_CUSTOM_AR_PUSH_16B_ASM
  using i4 = int __attribute__((ext_vector_type(4)));
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 310)
  __musa_stwb_v4i32(*reinterpret_cast<const i4*>(&raw), reinterpret_cast<i4*>(ptr));
#else
  *reinterpret_cast<uint4*>(ptr) = raw;
#endif
#else
  auto* dst = reinterpret_cast<volatile uint32_t*>(ptr);
  dst[0] = raw.x;
  dst[1] = raw.y;
  dst[2] = raw.z;
  dst[3] = raw.w;
#endif
}

template <typename P>
__device__ __forceinline__ P load_volatile_half_packet(const P* ptr) {
  static_assert(sizeof(P) == 8 && alignof(P) == 8);
  P value;
  const auto* src = reinterpret_cast<const volatile uint32_t*>(ptr);
  auto* dst = reinterpret_cast<uint32_t*>(&value);
  dst[0] = src[0];
  dst[1] = src[1];
  return value;
}

template <typename P>
__device__ __forceinline__ void store_volatile_half_packet(P* ptr, P value) {
  static_assert(sizeof(P) == 8 && alignof(P) == 8);
  const auto* src = reinterpret_cast<const uint32_t*>(&value);
  auto* dst = reinterpret_cast<volatile uint32_t*>(ptr);
  dst[0] = src[0];
  dst[1] = src[1];
}

__device__ __forceinline__ float upcast_s(half value) {
  return __half2float(value);
}

__device__ __forceinline__ float upcast_s(__mt_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ float upcast_s(float value) {
  return value;
}

template <typename T>
__device__ __forceinline__ T downcast_s(float value);

template <>
__device__ __forceinline__ half downcast_s<half>(float value) {
  return __float2half(value);
}

template <>
__device__ __forceinline__ __mt_bfloat16 downcast_s<__mt_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <>
__device__ __forceinline__ float downcast_s<float>(float value) {
  return value;
}

template <typename T>
__device__ __forceinline__ T& assign_add(T& a, T b) {
  a = downcast_s<T>(upcast_s(a) + upcast_s(b));
  return a;
}

template <>
__device__ __forceinline__ float& assign_add<float>(float& a, float b) {
  a += b;
  return a;
}

template <typename T, int N>
__device__ __forceinline__ array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; ++i) {
    assign_add(a.data[i], b.data[i]);
  }
  return a;
}

template <typename T, int N>
__device__ __forceinline__ array_t<float, N> upcast(array_t<T, N> value) {
  if constexpr (std::is_same<T, float>::value) {
    return value;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      out.data[i] = upcast_s(value.data[i]);
    }
    return out;
  }
}

template <typename O>
__device__ __forceinline__ O downcast(array_t<float, O::size> value) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return value;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; ++i) {
      out.data[i] = downcast_s<typename O::type>(value.data[i]);
    }
    return out;
  }
}

__device__ __forceinline__ void signal_store(FlagType* ptr, FlagType value) {
  volatile_store(static_cast<uint32_t>(value), reinterpret_cast<uint32_t*>(ptr));
}

__device__ __forceinline__ void signal_store_volatile(FlagType* ptr, FlagType value) {
  *reinterpret_cast<volatile FlagType*>(ptr) = value;
}

__device__ __forceinline__ FlagType signal_load(FlagType* ptr) {
  flushInv_byp();
  return static_cast<uint32_t>(volatile_load(reinterpret_cast<uint32_t*>(ptr)));
}

__device__ __forceinline__ FlagType signal_load_volatile(FlagType* ptr) {
  return *reinterpret_cast<volatile FlagType*>(ptr);
}

template <int nranks, bool start, bool fence = false>
__device__ __forceinline__ void multi_rank_barrier(const RankSignals& sg, Signal* self_sg, int rank) {
  static_assert(!(start && fence));
  if constexpr (!start) {
    __syncthreads_lm();
  }
  if (threadIdx.x < nranks) {
    const int target_rank = threadIdx.x;
    PullSemaphore* local = &self_sg->pull[blockIdx.x];
    PullSemaphore* target = &sg.signals[target_rank]->pull[blockIdx.x];
#if SGL_CUSTOM_AR_ATOMIC_BARRIER
    const auto flag = atomicAdd(&local->self_counter[target_rank], 1) + 1;
    atomicExch(&target->peer_counter[flag & 1][rank], flag);
    while (atomicAdd(&local->peer_counter[flag & 1][target_rank], 0) != flag) {
    }
#else
    const auto flag = local->self_counter[target_rank] + 1;
    local->self_counter[target_rank] = flag;
    auto* peer = &target->peer_counter[flag & 1][rank];
    auto* local_peer = &local->peer_counter[flag & 1][target_rank];
    if constexpr (fence) {
      signal_store(peer, flag);
      while (signal_load(local_peer) != flag) {
      }
    } else {
      signal_store_volatile(peer, flag);
      while (signal_load_volatile(local_peer) != flag) {
      }
    }
#endif
    if constexpr (start || fence) {
      __musa_fence_acq();
    }
  }
  if constexpr (start || fence) {
    __syncthreads_lm();
  }
}

template <bool start>
__device__ __forceinline__ void multi_rank_barrier_2rank(
    Signal* rank0_sg, Signal* rank1_sg, Signal* self_sg, int rank) {
  if constexpr (!start) {
    __syncthreads_lm();
  }
  if (threadIdx.x < 2) {
    const int target_rank = threadIdx.x;
    Signal* target_sg = target_rank == 0 ? rank0_sg : rank1_sg;
    PullSemaphore* target = &target_sg->pull[blockIdx.x];
    if (target_rank == rank) {
      PullSemaphore* local = &self_sg->pull[blockIdx.x];
      constexpr FlagType phase = 2;
      const FlagType base = start
          ? atomicAdd(&local->counter, 2 * phase)
          : static_cast<FlagType>(atomicAdd(&local->counter, 0) - phase);
      atomicAdd(&target->flag, 1);
      while (static_cast<FlagType>(atomicAdd(&local->flag, 0) - base) <
             phase) {
      }
      if constexpr (start) {
        __musa_fence_acq();
      }
    } else {
      atomicAdd(&target->flag, 1);
    }
  }
  if constexpr (start) {
    __syncthreads_lm();
  }
}

template <int nranks>
__device__ __forceinline__ void push_publish_barrier(
    const RankSignals& sg, Signal* self_sg, int rank) {
  __syncthreads_lm();
  __musa_barrier_slc();
  if (threadIdx.x == 0) {
    __musa_fence_rel();
  }
  __syncthreads_lm();
  if (threadIdx.x < nranks) {
    const int target_rank = threadIdx.x;
    PullSemaphore* target =
        &sg.signals[target_rank]->push_ready[blockIdx.x];
    if (target_rank == rank) {
      PullSemaphore* local = &self_sg->push_ready[blockIdx.x];
      const FlagType phase = static_cast<FlagType>(nranks);
      const FlagType base = atomicAdd(&local->counter, phase);
      atomicAdd(&target->flag, 1);
      while (static_cast<FlagType>(atomicAdd(&local->flag, 0) - base) <
             phase) {
      }
      __musa_fence_acq();
    } else {
      atomicAdd(&target->flag, 1);
    }
  }
  __syncthreads_lm();
}

template <typename P, int nranks, typename A>
__device__ __forceinline__ P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < nranks; ++i) {
    packed_assign_add(tmp, upcast(ptrs[i][idx]));
  }
  return downcast<P>(tmp);
}

template <typename T, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) cross_device_reduce_1stage(
    RankData data, const RankData* data_ptr, RankSignals sg, Signal* self_sg,
    T* __restrict__ out, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  multi_rank_barrier<nranks, true>(sg, self_sg, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    const void* rank0_data =
        data_ptr != nullptr ? data_ptr->ptrs[0] : data.ptrs[0];
    A reduced =
        upcast(reinterpret_cast<const P*>(rank0_data)[idx]);
#pragma unroll
    for (int i = 1; i < nranks; ++i) {
      const void* rank_data =
          data_ptr != nullptr ? data_ptr->ptrs[i] : data.ptrs[i];
      packed_assign_add(
          reduced, upcast(reinterpret_cast<const P*>(rank_data)[idx]));
    }
    reinterpret_cast<P*>(out)[idx] = downcast<P>(reduced);
  }
  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <typename T>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) cross_device_reduce_1stage_2rank(
    const void* local_ptr, const void* peer_ptr, const RankData* data_ptr,
    Signal* rank0_sg, Signal* rank1_sg, Signal* self_sg, T* __restrict__ out,
    int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  if (data_ptr != nullptr) {
    local_ptr = data_ptr->ptrs[rank];
    peer_ptr = data_ptr->ptrs[rank ^ 1];
  }
  const auto* local = reinterpret_cast<const P*>(local_ptr);
  const auto* peer = reinterpret_cast<const P*>(peer_ptr);
  auto* dst = reinterpret_cast<P*>(out);
  multi_rank_barrier_2rank<true>(rank0_sg, rank1_sg, self_sg, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x) {
    A tmp = upcast(local[idx]);
    packed_assign_add(tmp, upcast(peer[idx]));
    dst[idx] = downcast<P>(tmp);
  }
  multi_rank_barrier_2rank<false>(rank0_sg, rank1_sg, self_sg, rank);
}

template <typename T, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) cross_device_push_1stage(
#if SGL_CUSTOM_AR_PUSH_POLLING
    const RankData* data_ptr,
#else
    RankData data, const RankData* data_ptr, RankSignals sg,
#endif
    Signal* self_sg,
    const T* __restrict__ input, T* __restrict__ out, int rank, int size,
    int slot_bytes) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const int slot_size = slot_bytes / static_cast<int>(sizeof(P));
  const P* src = reinterpret_cast<const P*>(input);

#if !SGL_CUSTOM_AR_PUSH_POLLING
  if constexpr (nranks == 2) {
#if !SGL_CUSTOM_AR_PUSH_SKIP_START_BARRIER
    multi_rank_barrier<2, true>(sg, self_sg, rank);
#endif
    const void* peer_data =
        data_ptr != nullptr ? data_ptr->ptrs[rank ^ 1] : data.ptrs[rank ^ 1];
    P* peer_dst = reinterpret_cast<P*>(
        reinterpret_cast<char*>(const_cast<void*>(peer_data)) +
        rank * slot_bytes);
    for (int idx = tid; idx < size; idx += stride) {
      peer_dst[idx] = src[idx];
    }
    __musa_barrier_slc();
    __syncthreads_lm();
    if (threadIdx.x == 0) {
      __threadfence_system_noflush();
    }
    push_publish_barrier<2>(sg, self_sg, rank);

    const void* local_data =
        data_ptr != nullptr ? data_ptr->ptrs[rank] : data.ptrs[rank];
    const auto* local_buffer = reinterpret_cast<const char*>(local_data);
    const P* peer_src = reinterpret_cast<const P*>(
        local_buffer + (rank ^ 1) * slot_bytes);
    auto* dst = reinterpret_cast<P*>(out);
    for (int idx = tid; idx < size; idx += stride) {
      A tmp = upcast(src[idx]);
      packed_assign_add(tmp, upcast(peer_src[idx]));
      dst[idx] = downcast<P>(tmp);
    }
    multi_rank_barrier<2, false>(sg, self_sg, rank);
    return;
  }

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);
#pragma unroll
  for (int peer = 0; peer < nranks; ++peer) {
    const void* peer_data =
        data_ptr != nullptr ? data_ptr->ptrs[peer] : data.ptrs[peer];
    P* dst = reinterpret_cast<P*>(
        reinterpret_cast<char*>(const_cast<void*>(peer_data)) +
        rank * slot_bytes);
    for (int idx = tid; idx < size; idx += stride) {
      dst[idx] = src[idx];
    }
  }
  __musa_barrier_slc();
  __syncthreads_lm();
  if (threadIdx.x == 0) {
    __threadfence_system_noflush();
  }
  push_publish_barrier<nranks>(sg, self_sg, rank);

  const P* ptrs[nranks];
  const void* local_data =
      data_ptr != nullptr ? data_ptr->ptrs[rank] : data.ptrs[rank];
  const auto* local_buffer = reinterpret_cast<const char*>(local_data);
#pragma unroll
  for (int i = 0; i < nranks; ++i) {
    ptrs[i] = reinterpret_cast<const P*>(local_buffer + i * slot_bytes);
  }
  for (int idx = tid; idx < size; idx += stride) {
    reinterpret_cast<P*>(out)[idx] = packed_reduce<P, nranks, A>(ptrs, idx);
  }
  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
#else
  using PollP = array_t<T, 8 / sizeof(T)>;
  using PollA = array_t<float, 8 / sizeof(T)>;
  const int epoch = self_sg->push_epoch[blockIdx.x] & 1;
  const int poll_slot_size = slot_bytes / static_cast<int>(sizeof(PollP));
  const PollP* poll_src = reinterpret_cast<const PollP*>(input);
  // Push data uses a v2-style two-stage ring:
  //   slot = epoch * nranks + source_rank
#pragma unroll
  for (int peer = 0; peer < nranks; ++peer) {
    PollP* dst = reinterpret_cast<PollP*>(
        reinterpret_cast<char*>(const_cast<void*>(data_ptr->ptrs[peer])) +
        (epoch * nranks + rank) * slot_bytes);
    for (int idx = tid; idx < size; idx += stride) {
      PollP value = poll_src[idx];
      clear_pos_zero(value);
      store_volatile_half_packet(&dst[idx], value);
    }
  }

  __musa_barrier_slc();
  __syncthreads_lm();
  if (threadIdx.x == 0) {
    __threadfence_system_noflush();
  }
  __syncthreads_lm();

  auto* local_buffer =
      reinterpret_cast<char*>(const_cast<void*>(data_ptr->ptrs[rank]));
  for (int idx = tid; idx < size; idx += stride) {
    PollA tmp{};
#pragma unroll
    for (int i = 0; i < nranks; ++i) {
      const PollP* peer_src = reinterpret_cast<const PollP*>(
          local_buffer + (epoch * nranks + i) * slot_bytes);
      PollP value;
      do {
        flushInv_byp();
        value = load_volatile_half_packet(&peer_src[idx]);
      } while (has_pos_zero(value));
      if (i == 0) {
        tmp = upcast(value);
      } else {
        packed_assign_add(tmp, upcast(value));
      }
    }
    reinterpret_cast<PollP*>(out)[idx] = downcast<PollP>(tmp);
    PollP* reset_ptr = reinterpret_cast<PollP*>(
        local_buffer + epoch * nranks * slot_bytes);
    const PollP pos_zero = make_pos_zero_packet<PollP>();
#pragma unroll
    for (int i = 0; i < nranks; ++i) {
      reset_ptr[i * poll_slot_size + idx] = pos_zero;
    }
  }
  __syncthreads_lm();
  if (threadIdx.x == 0) {
    self_sg->push_epoch[blockIdx.x] = (epoch + 1) & 1;
  }
#endif
}

template <typename T>
__global__ void __launch_bounds__(kDefaultThreads, 1)
cross_device_push_1stage_2rank_polling(
    const void* rank0_data,
    const void* rank1_data,
    const RankData* data_ptr,
    Signal* self_sg,
    const T* __restrict__ input,
    T* __restrict__ out,
    int rank,
    int size,
    int slot_bytes) {
  if (data_ptr != nullptr) {
    rank0_data = data_ptr->ptrs[0];
    rank1_data = data_ptr->ptrs[1];
  }
  // The 16-byte polling packet leaves one packet in private memory on MP31.
  // Two 8-byte packets preserve the same slot protocol while keeping the
  // polling value and fp32 accumulator in registers.
  using P = array_t<T, 8 / sizeof(T)>;
  using A = array_t<float, 8 / sizeof(T)>;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const int slot_size = slot_bytes / static_cast<int>(sizeof(P));
  const int epoch = self_sg->push_epoch[blockIdx.x] & 1;
  const P* src = reinterpret_cast<const P*>(input);

  auto* rank0_buffer = reinterpret_cast<char*>(const_cast<void*>(rank0_data));
  auto* rank1_buffer = reinterpret_cast<char*>(const_cast<void*>(rank1_data));
  P* rank0_dst = reinterpret_cast<P*>(
      rank0_buffer + (epoch * 2 + rank) * slot_bytes);
  P* rank1_dst = reinterpret_cast<P*>(
      rank1_buffer + (epoch * 2 + rank) * slot_bytes);
  for (int idx = tid; idx < size; idx += stride) {
    P value = src[idx];
    clear_pos_zero(value);
    store_volatile_half_packet(&rank0_dst[idx], value);
    store_volatile_half_packet(&rank1_dst[idx], value);
  }

  __musa_barrier_slc();
  __syncthreads_lm();
  if (threadIdx.x == 0) {
    __threadfence_system_noflush();
  }
  __syncthreads_lm();

  char* local_buffer = rank == 0 ? rank0_buffer : rank1_buffer;
  const P* src0 = reinterpret_cast<const P*>(
      local_buffer + epoch * 2 * slot_bytes);
  const P* src1 = reinterpret_cast<const P*>(
      local_buffer + (epoch * 2 + 1) * slot_bytes);
  P* reset0 = reinterpret_cast<P*>(
      local_buffer + epoch * 2 * slot_bytes);
  P* reset1 = reset0 + slot_size;
  for (int idx = tid; idx < size; idx += stride) {
    P value0;
    P value1;
    do {
      flushInv_byp();
      value0 = load_volatile_half_packet(&src0[idx]);
      value1 = load_volatile_half_packet(&src1[idx]);
    } while (has_pos_zero(value0) || has_pos_zero(value1));
    A tmp = upcast(value0);
    packed_assign_add(tmp, upcast(value1));
    reinterpret_cast<P*>(out)[idx] = downcast<P>(tmp);
    const P pos_zero = make_pos_zero_packet<P>();
    reset0[idx] = pos_zero;
    reset1[idx] = pos_zero;
  }
  __syncthreads_lm();
  if (threadIdx.x == 0) {
    self_sg->push_epoch[blockIdx.x] = (epoch + 1) & 1;
  }
}

template <typename T, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) cross_device_reduce_2stage(
    RankData data, const RankData* data_ptr, RankSignals sg, Signal* self_sg,
    T* __restrict__ out, int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  if (data_ptr != nullptr) data = *data_ptr;
  // Qwen3.5 TP4 decode at bs=128 is exactly 512 KiB.  The direct
  // broadcast path is measurably faster for this aligned payload, while
  // Gemma's non-aligned 405,504-byte payload needs the peer-gather path
  // below for both correctness and throughput.
  if constexpr (nranks == 4) {
    constexpr int kDirectBroadcastSize = (512 * 1024) / sizeof(P);
    if (size == kDirectBroadcastSize) {
      constexpr int warp = 32;
      const int vec_per_rank =
          ((size + nranks - 1) / nranks + warp - 1) / warp * warp;
      const int start = min(rank * vec_per_rank, size);
      const int end = min(start + vec_per_rank, size);
      multi_rank_barrier<nranks, true>(sg, self_sg, rank);
      for (int idx = start + tid; idx < end; idx += stride) {
        A reduced_acc = upcast(
            reinterpret_cast<const P*>(data.ptrs[0])[idx]);
#pragma unroll
        for (int i = 1; i < nranks; ++i) {
          packed_assign_add(
              reduced_acc,
              upcast(reinterpret_cast<const P*>(data.ptrs[i])[idx]));
        }
        const P reduced = downcast<P>(reduced_acc);
#pragma unroll
        for (int i = 0; i < nranks; ++i) {
          auto* peer = reinterpret_cast<P*>(
              const_cast<void*>(data.ptrs[i]));
          peer[idx] = reduced;
        }
      }
      multi_rank_barrier<nranks, false, true>(sg, self_sg, rank);
      return;
    }
  }
  int part = size / nranks;
  int start = rank * part;
  int end = rank == nranks - 1 ? size : start + part;
  int largest_part = part + size % nranks;
  const P* ptrs[nranks];
  P* tmp = get_tmp_buf<P>(sg.signals[rank]);
#pragma unroll
  for (int i = 0; i < nranks; ++i) {
    ptrs[i] = reinterpret_cast<const P*>(data.ptrs[i]);
  }
  multi_rank_barrier<nranks, true>(sg, self_sg, rank);
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp[idx - start] = packed_reduce<P, nranks, A>(ptrs, idx);
  }
  multi_rank_barrier<nranks, false, true>(sg, self_sg, rank);
  const int local_part = rank == nranks - 1 ? largest_part : part;
  for (int idx = tid; idx < local_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < nranks; ++i) {
      auto* peer = reinterpret_cast<P*>(const_cast<void*>(data.ptrs[i]));
      peer[rank * part + idx] = tmp[idx];
    }
  }
}

template <typename T, int nranks, int vlen = 8>
__device__ __forceinline__ void shfl_reduce(float* res) {
  if constexpr (nranks >= 4) {
#pragma unroll
    for (int i = 0; i < vlen; ++i) {
      res[i] += __shfl_xor_sync(0xffffffff, res[i], 16);
    }
  }
#pragma unroll
  for (int i = 0; i < vlen; ++i) {
    res[i] += __shfl_xor_sync(0xffffffff, res[i], 8);
  }
}

template <typename T, int nranks, int vlen = 8>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) custom_all_reduce_2shot(
    RankData data,
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    T* __restrict__ out,
    int rank,
    int size) {
  if (data_ptr != nullptr) {
    data = *data_ptr;
  }
  constexpr int nranks_sft = (nranks >> 1) - (nranks >> 3);
  constexpr int coalesce_num = 8;
  constexpr int coalesce_sft = 3;
  constexpr int group_stride_sft = nranks_sft + coalesce_sft;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int target_rank = (tid >> coalesce_sft) & (nranks - 1);
  const int group_id = tid >> group_stride_sft;
  const int coalesce_tid = tid & (coalesce_num - 1);
  const int stride = gridDim.x * blockDim.x;

  typedef int16_t Vec __attribute__((vector_size(16)));
  int idx_base = blockIdx.x * blockDim.x;
  int idx_in_blk = coalesce_tid + (rank << coalesce_sft) + (group_id << group_stride_sft);

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);
  Vec* target_ptr =
      reinterpret_cast<Vec*>(const_cast<void*>(data.ptrs[target_rank]));
  Vec* buffer_ptr = get_tmp_buf<Vec>(sg.signals[rank]);
  do {
    int idx = idx_in_blk + idx_base;
    float acc[vlen] = {0};
    if (idx < size) {
#if SGL_CUSTOM_AR_VECTOR_LOAD
      Vec raw = target_ptr[idx];
      const T* src = reinterpret_cast<const T*>(&raw);
#else
      const T* src = reinterpret_cast<const T*>(&target_ptr[idx]);
#endif
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        acc[i] = upcast_s(src[i]);
      }
    }
    shfl_reduce<T, nranks, vlen>(acc);
    if constexpr (nranks == 8) {
      __shared__ float smem[kMaxThreadsPerBlock << 1];
      if (lane < coalesce_num) {
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          smem[warp * vlen * coalesce_num + coalesce_tid * vlen + i] = acc[i];
        }
      }
      __syncthreads_lm();
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        acc[i] += smem[(warp ^ 1) * vlen * coalesce_num + coalesce_tid * vlen + i];
      }
    }
    if (rank == target_rank && idx < size) {
      Vec res;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        reinterpret_cast<T*>(&res)[i] = downcast_s<T>(acc[i]);
      }
      buffer_ptr[idx] = res;
#if SGL_CUSTOM_AR_2SHOT_DOUBLE_STORE
      buffer_ptr[idx] = res;
#endif
    }
    idx_base += stride;
  } while (idx_base < size);

  multi_rank_barrier<nranks, false, true>(sg, self_sg, rank);

  buffer_ptr = get_tmp_buf<Vec>(sg.signals[target_rank]);
  idx_in_blk =
      coalesce_tid + (target_rank << coalesce_sft) +
      (group_id << group_stride_sft);
  idx_base = blockIdx.x * blockDim.x;
  do {
    int idx = idx_in_blk + idx_base;
    if (idx < size) {
      reinterpret_cast<Vec*>(out)[idx] = buffer_ptr[idx];
    }
    idx_base += stride;
  } while (idx_base < size);
}

template <typename T, int nranks>
int select_block_limit(int packed_size, int shot) {
  int limit = kDefaultBlockLimit;
#if SGL_CUSTOM_AR_DYNAMIC_BLOCKS
  if (shot == 0) {
    if constexpr (nranks == 2) {
      if (packed_size < 32 * 1024) {
        limit = std::min(limit, 12);
      }
    }
  } else if (shot == 1) {
    if constexpr (nranks == 2) {
      if (packed_size <= 128 * 1024) {
        limit = std::min(limit, 24);
      }
    }
  } else if ((shot == 2 || shot == 4) && !std::is_same<T, float>::value) {
    if constexpr (nranks == 4) {
      if (packed_size >= 4 * 1024 * 1024) {
        limit = 40;
      } else if (packed_size >= 1024 * 1024) {
        limit = 80;
      }
    } else if constexpr (nranks == 8) {
      if (packed_size <= 128 * 1024) {
        limit = 60;
      }
    }
  }
#endif
  return std::min(limit, kMaxBlocks);
}

template <typename T, int nranks>
void launch_ar(
    RankData data,
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* input,
    T* out,
    int rank,
    int size,
    int shot,
    int push_buffer_bytes,
    musaStream_t stream) {
  const int pack = packed_t<T>::P::size;
  TVM_FFI_ICHECK_EQ(size % pack, 0);
  int packed_size = size / pack;
  int block_limit = select_block_limit<T, nranks>(packed_size, shot);
  int blocks = std::min(block_limit, (packed_size + kDefaultThreads - 1) / kDefaultThreads);
#if SGL_CUSTOM_AR_PUSH_POLLING
  // CUDA v2 binds push to a fixed counter grid. Keep every block's epoch in
  // lockstep across changing payload sizes; otherwise a vector may move to a
  // block whose double-buffer phase did not advance on the previous call.
  if (shot == 0 || shot == 3) {
    blocks = block_limit;
  }
#endif
  if (blocks <= 0) {
    return;
  }
  if (shot == 0 || shot == 3) {
#if SGL_CUSTOM_AR_PUSH_POLLING
    TVM_FFI_ICHECK(data_ptr != nullptr);
    cross_device_push_1stage<T, nranks>
          <<<blocks, kDefaultThreads, 0, stream>>>(
              data_ptr, self_sg, input, out, rank,
              packed_size * 2, push_buffer_bytes);
#else
    cross_device_push_1stage<T, nranks>
        <<<blocks, kDefaultThreads, 0, stream>>>(
            data, data_ptr, sg, self_sg, input, out, rank, packed_size,
            push_buffer_bytes);
#endif
  } else if (shot == 1) {
    if constexpr (nranks == 2 && SGL_CUSTOM_AR_1SHOT_2RANK_SPECIAL) {
      cross_device_reduce_1stage_2rank<T>
          <<<blocks, kDefaultThreads, 0, stream>>>(
              data.ptrs[rank], data.ptrs[rank ^ 1], data_ptr, sg.signals[0],
              sg.signals[1], self_sg, out, rank, packed_size);
    } else {
      cross_device_reduce_1stage<T, nranks><<<blocks, kDefaultThreads, 0, stream>>>(data, data_ptr, sg, self_sg, out, rank, packed_size);
    }
  } else if (shot == 2) {
    // CUDA-v2 reduce-scatter plus direct broadcast; unlike the older shuffle
    // variant this is also valid for non-power-of-two rank counts.
    cross_device_reduce_2stage<T, nranks>
        <<<blocks, kDefaultThreads, 0, stream>>>(
            data, data_ptr, sg, self_sg, out, rank, packed_size);
  } else if (shot == 4) {
    if constexpr (nranks == 4 || nranks == 8) {
      custom_all_reduce_2shot<T, nranks>
          <<<blocks, kDefaultThreads, 0, stream>>>(
              data, data_ptr, sg, self_sg, out, rank, packed_size);
    } else {
      TVM_FFI_THROW(ValueError) << "shot 4 requires world size 4 or 8";
    }
  } else {
    TVM_FFI_THROW(ValueError) << "shot must be 0, 1, 2, 3, or 4";
  }
}

template <typename T>
void dispatch_world_size(
    RankData data,
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* input,
    T* out,
    int rank,
    int world_size,
    int size,
    int shot,
    int push_buffer_bytes,
    musaStream_t stream) {
  switch (world_size) {
    case 2:
      launch_ar<T, 2>(data, data_ptr, sg, self_sg, input, out, rank, size, shot, push_buffer_bytes, stream);
      break;
    case 4:
      launch_ar<T, 4>(data, data_ptr, sg, self_sg, input, out, rank, size, shot, push_buffer_bytes, stream);
      break;
    case 6:
      launch_ar<T, 6>(data, data_ptr, sg, self_sg, input, out, rank, size, shot, push_buffer_bytes, stream);
      break;
    case 8:
      launch_ar<T, 8>(data, data_ptr, sg, self_sg, input, out, rank, size, shot, push_buffer_bytes, stream);
      break;
    default:
      TVM_FFI_THROW(ValueError) << "world_size must be one of 2/4/6/8";
  }
}

template <typename T>
void dispatch_world_size(
    RankData data,
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* input,
    T* out,
    int rank,
    int world_size,
    int size,
    int shot,
    musaStream_t stream) {
  // Device-descriptor callers derive their push slot size from the payload.
  dispatch_world_size(
      data, data_ptr, sg, self_sg, input, out, rank, world_size, size, shot,
      size * static_cast<int>(sizeof(T)), stream);
}

}  // namespace

void vllm_musa_custom_ar_launch_device_descriptor(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView out,
    int64_t self_signal_ptr,
    int64_t rank,
    int64_t world_size,
    int64_t shot) {
  CHECK_MUSA_CONTIGUOUS(out);
  TVM_FFI_ICHECK_EQ(
      rank_data.device().device_type, out.device().device_type);
  TVM_FFI_ICHECK_EQ(rank_data.device().device_id, out.device().device_id);
  TVM_FFI_ICHECK(rank_data.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(rank_data.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(rank_data.size(0), kMaxRanks);
  TVM_FFI_ICHECK_EQ(signal_ptrs_cpu.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(signal_ptrs_cpu.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(signal_ptrs_cpu.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(signal_ptrs_cpu.size(0), world_size);
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size);
  TVM_FFI_ICHECK(shot == 0 || shot == 3);

  RankSignals sg{};
  const auto* ptrs = static_cast<const int64_t*>(signal_ptrs_cpu.data_ptr());
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(ptrs[i]);
  }
  RankData data{};
  const auto* device_data_ptr =
      reinterpret_cast<const RankData*>(rank_data.data_ptr());
  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(out.device());
  const int64_t numel64 = tensor_numel(out);
  TVM_FFI_ICHECK_LE(numel64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  const int size = static_cast<int>(numel64);

  if (dtype_equal(out.dtype(), dl_float16)) {
    auto* out_ptr = static_cast<half*>(out.data_ptr());
    dispatch_world_size(data, device_data_ptr, sg, self_sg, out_ptr, out_ptr, static_cast<int>(rank), static_cast<int>(world_size), size, static_cast<int>(shot), stream);
  } else if (dtype_equal(out.dtype(), dl_bfloat16)) {
    auto* out_ptr = static_cast<__mt_bfloat16*>(out.data_ptr());
    dispatch_world_size(data, device_data_ptr, sg, self_sg, out_ptr, out_ptr, static_cast<int>(rank), static_cast<int>(world_size), size, static_cast<int>(shot), stream);
  } else if (dtype_equal(out.dtype(), dl_float32)) {
    auto* out_ptr = static_cast<float*>(out.data_ptr());
    dispatch_world_size(data, device_data_ptr, sg, self_sg, out_ptr, out_ptr, static_cast<int>(rank), static_cast<int>(world_size), size, static_cast<int>(shot), stream);
  } else {
    TVM_FFI_THROW(ValueError) << "custom ar only supports fp16/bf16/fp32";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess) << "MUSA custom AR kernel failed: " << musaGetErrorString(err);
}

extern "C" void vllm_musa_custom_ar_launch_unregistered_raw_nocheck(
    const int64_t* rank_ptrs,
    const int64_t* signal_ptrs,
    const void* inp,
    void* out,
    int64_t self_signal_ptr,
    int64_t self_buffer_ptr,
    int64_t push_buffer_bytes,
    int64_t numel,
    int dtype_code,
    int64_t stream_value,
    int64_t rank,
    int64_t world_size,
    int64_t shot) {
  RankSignals sg{};
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(signal_ptrs[i]);
  }
  RankData data{};
  for (int i = 0; i < kMaxRanks; ++i) {
    data.ptrs[i] = reinterpret_cast<const void*>(rank_ptrs[i]);
  }
  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = reinterpret_cast<musaStream_t>(stream_value);
  const int size = static_cast<int>(numel);
  const int64_t nbytes = dtype_code == 2 ? numel * 4 : numel * 2;
  if (shot != 0 && shot != 3) {
    musaMemcpyAsync(
        reinterpret_cast<void*>(self_buffer_ptr),
        inp,
        static_cast<size_t>(nbytes),
        musaMemcpyDeviceToDevice,
        stream);
  }

  if (dtype_code == 0) {
    dispatch_world_size(data, nullptr, sg, self_sg, static_cast<const half*>(inp), static_cast<half*>(out), static_cast<int>(rank), static_cast<int>(world_size), size, static_cast<int>(shot), static_cast<int>(push_buffer_bytes), stream);
  } else if (dtype_code == 1) {
    dispatch_world_size(data, nullptr, sg, self_sg, static_cast<const __mt_bfloat16*>(inp), static_cast<__mt_bfloat16*>(out), static_cast<int>(rank), static_cast<int>(world_size), size, static_cast<int>(shot), static_cast<int>(push_buffer_bytes), stream);
  } else {
    dispatch_world_size(data, nullptr, sg, self_sg, static_cast<const float*>(inp), static_cast<float*>(out), static_cast<int>(rank), static_cast<int>(world_size), size, static_cast<int>(shot), static_cast<int>(push_buffer_bytes), stream);
  }
  // The generic shot-2 kernel publishes its final result through the pull
  // buffer. The optimized shot-4 kernel gathers peer scratch directly into
  // ``out``; copying the local pull buffer here would overwrite the reduced
  // value with this rank's original input.
  if (shot == 2) {
    musaMemcpyAsync(
        out,
        reinterpret_cast<void*>(self_buffer_ptr),
        static_cast<size_t>(nbytes),
        musaMemcpyDeviceToDevice,
        stream);
  }
}

extern "C" void vllm_musa_custom_ar_launch_registered_raw_nocheck(
    const int64_t* rank_ptrs,
    const int64_t* signal_ptrs,
    void* out,
    int64_t self_signal_ptr,
    int64_t numel,
    int dtype_code,
    int64_t stream_value,
    int64_t rank,
    int64_t world_size,
    int64_t shot) {
  RankSignals sg{};
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(signal_ptrs[i]);
  }
  RankData data{};
  const auto* device_data_ptr =
      reinterpret_cast<const RankData*>(rank_ptrs);
  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = reinterpret_cast<musaStream_t>(stream_value);
  const int size = static_cast<int>(numel);

  if (dtype_code == 0) {
    auto* out_ptr = static_cast<half*>(out);
    dispatch_world_size(data, device_data_ptr, sg, self_sg, out_ptr, out_ptr,
                        static_cast<int>(rank), static_cast<int>(world_size),
                        size, static_cast<int>(shot), stream);
  } else if (dtype_code == 1) {
    auto* out_ptr = static_cast<__mt_bfloat16*>(out);
    dispatch_world_size(data, device_data_ptr, sg, self_sg, out_ptr, out_ptr,
                        static_cast<int>(rank), static_cast<int>(world_size),
                        size, static_cast<int>(shot), stream);
  } else {
    auto* out_ptr = static_cast<float*>(out);
    dispatch_world_size(data, device_data_ptr, sg, self_sg, out_ptr, out_ptr,
                        static_cast<int>(rank), static_cast<int>(world_size),
                        size, static_cast<int>(shot), stream);
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    vllm_musa_custom_ar_launch_device_descriptor,
    vllm_musa_custom_ar_launch_device_descriptor);

#else  // SGL_CUSTOM_AR_FUSED_RMSNORM
// Ordinary AR+RMSNorm source reconstructed from MR577 target lines 1-4162
// and exports at target lines 4455-4469 (Gemma-only section excluded).
// Source segment SHA256 before the vLLM ABI adaptation:
// 8508b7bcf849bc362beb09c6431842a99730316709e4dbd3c7c944642ba0fd72
// The complete target Git blob is not reconstructed; source-map evidence is
// kept in the MUSA-100062 takeover artifacts. Adaptations: optional raw output,
// optional residual, and vLLM dtype materialization before RMSNorm.
#include "../common.h"
#include "../device_utils.h"
#include "../norm/common.mu"

#include <musa_runtime.h>
#include <musa_bf16.h>
#include <musa_fp16.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

#ifndef SGL_CUSTOM_AR_THREADS
#define SGL_CUSTOM_AR_THREADS 512
#endif

#ifndef SGL_CUSTOM_AR_BLOCKS
#define SGL_CUSTOM_AR_BLOCKS 36
#endif

#ifndef SGL_CUSTOM_AR_VECTOR_LOAD
#define SGL_CUSTOM_AR_VECTOR_LOAD 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_H8192_BLOCKS
#define SGL_CUSTOM_AR_FUSED_RMSNORM_H8192_BLOCKS 16
#endif

#ifndef SGL_CUSTOM_AR_ATOMIC_BARRIER
#define SGL_CUSTOM_AR_ATOMIC_BARRIER 1
#endif

#ifndef SGL_CUSTOM_AR_MAX_BLOCKS
#define SGL_CUSTOM_AR_MAX_BLOCKS 120
#endif

#ifndef SGL_CUSTOM_AR_LAUNCH_MAX_BLOCKS
#define SGL_CUSTOM_AR_LAUNCH_MAX_BLOCKS SGL_CUSTOM_AR_MAX_BLOCKS
#endif

#ifndef SGL_CUSTOM_AR_DYNAMIC_BLOCKS
#define SGL_CUSTOM_AR_DYNAMIC_BLOCKS 1
#endif

#ifndef SGL_CUSTOM_AR_RMSNORM_CACHE_HIDDEN_LIMIT
#define SGL_CUSTOM_AR_RMSNORM_CACHE_HIDDEN_LIMIT 2048
#endif

#ifndef SGL_CUSTOM_AR_RMSNORM_T_CACHE_HIDDEN_LIMIT
#define SGL_CUSTOM_AR_RMSNORM_T_CACHE_HIDDEN_LIMIT 8192
#endif

#ifndef SGL_CUSTOM_AR_RMSNORM_WEIGHT_CACHE_HIDDEN_LIMIT
#define SGL_CUSTOM_AR_RMSNORM_WEIGHT_CACHE_HIDDEN_LIMIT 8192
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_TOKEN_2STAGE
#define SGL_CUSTOM_AR_FUSED_RMSNORM_TOKEN_2STAGE 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_PACKED_1STAGE
#define SGL_CUSTOM_AR_FUSED_RMSNORM_PACKED_1STAGE 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_SAFE_PACKED_1STAGE
#define SGL_CUSTOM_AR_FUSED_RMSNORM_SAFE_PACKED_1STAGE 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_VEC2RANK_1STAGE
#define SGL_CUSTOM_AR_FUSED_RMSNORM_VEC2RANK_1STAGE 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_PARTIAL_PACKED_NON8
#define SGL_CUSTOM_AR_FUSED_RMSNORM_PARTIAL_PACKED_NON8 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_VEC2_NON8
#define SGL_CUSTOM_AR_FUSED_RMSNORM_VEC2_NON8 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_VEC4_NON8
#define SGL_CUSTOM_AR_FUSED_RMSNORM_VEC4_NON8 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_REGCACHE_2RANK
#define SGL_CUSTOM_AR_FUSED_RMSNORM_REGCACHE_2RANK 1
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WEIGHT_CACHE
#define SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WEIGHT_CACHE 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WEIGHT_CACHE_MIN_ROWS
#define SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WEIGHT_CACHE_MIN_ROWS 2048
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_SHARED_INV_RMS
#define SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_SHARED_INV_RMS 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WARP_INV_RMS
#define SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WARP_INV_RMS 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD
#define SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_SMALL_ROWS
#define SGL_CUSTOM_AR_FUSED_RMSNORM_SMALL_ROWS 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_SKIP_END_BARRIER
#define SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_SKIP_END_BARRIER 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_PEER_PUBLISH
#define SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_PEER_PUBLISH 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_SKIP_SELF_SIGNAL
#define SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_SKIP_SELF_SIGNAL 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_PUSH
#define SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_PUSH 0
#endif

#ifndef SGL_CUSTOM_AR_PUSH_16B_ASM
#define SGL_CUSTOM_AR_PUSH_16B_ASM 0
#endif

constexpr int kMaxBlocks = SGL_CUSTOM_AR_MAX_BLOCKS;
constexpr int kLaunchMaxBlocks =
    (SGL_CUSTOM_AR_LAUNCH_MAX_BLOCKS < kMaxBlocks)
        ? SGL_CUSTOM_AR_LAUNCH_MAX_BLOCKS
        : kMaxBlocks;
static_assert(kLaunchMaxBlocks > 0, "launch block cap must be positive");
// Keep the semaphore ABI independent of shape-specific JIT block tuning.  The
// fused communicator shares one signal allocation across all compiled shapes.
constexpr int kSignalMaxBlocks = 512;
static_assert(kMaxBlocks <= kSignalMaxBlocks);
constexpr int kMaxThreadsPerBlock = 1024;
constexpr int kDefaultThreads = SGL_CUSTOM_AR_THREADS;
constexpr int kDefaultBlockLimit = SGL_CUSTOM_AR_BLOCKS;
constexpr int kH4096BlockLimit = 32;
constexpr int kH8192BlockLimit = SGL_CUSTOM_AR_FUSED_RMSNORM_H8192_BLOCKS;
constexpr int kCacheHiddenLimit = SGL_CUSTOM_AR_RMSNORM_CACHE_HIDDEN_LIMIT;
constexpr int kTypedCacheHiddenLimit = SGL_CUSTOM_AR_RMSNORM_T_CACHE_HIDDEN_LIMIT;
constexpr int kWeightCacheHiddenLimit = SGL_CUSTOM_AR_RMSNORM_WEIGHT_CACHE_HIDDEN_LIMIT;
constexpr bool kRowSharedInvRms = SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_SHARED_INV_RMS != 0;
constexpr bool kRowWarpInvRms = SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WARP_INV_RMS != 0;
constexpr int kMaxRanks = 8;
constexpr int kOneShotMaxToken = 128;
using FlagType = uint32_t;

__device__ __host__ __forceinline__ bool rmsnorm_cache_weight_hidden(
    int hidden,
    int rows) {
  return (hidden == 8192 || hidden == 1536) && hidden <= kWeightCacheHiddenLimit;
}

struct alignas(128) PullSemaphore {
  FlagType flag;
  FlagType counter;
  FlagType self_counter[kMaxRanks];
  FlagType peer_counter[2][kMaxRanks];
};
static_assert(sizeof(PullSemaphore) == 128);

struct alignas(128) Signal {
  union alignas(128) {
    PullSemaphore pull[kSignalMaxBlocks];
    FlagType push_epoch[kSignalMaxBlocks];
  };
  // Two-stage kernels publish peer-visible scratch between the start and end
  // phases. Keep that generation independent from the pull controller while
  // preserving the common pull/push signal ABI.
  union alignas(128) {
    PullSemaphore publish[kSignalMaxBlocks];
    PullSemaphore push_ready[kSignalMaxBlocks];
  };
};
static_assert(
    sizeof(Signal) == 2 * kSignalMaxBlocks * sizeof(PullSemaphore));

struct __align__(16) RankData {
  const void* ptrs[kMaxRanks];
};

struct __align__(16) RankSignals {
  Signal* signals[kMaxRanks];
};

int get_musa_sm_count() {
  static int sm_count = 0;
  if (sm_count == 0) {
    int device_id = 0;
    if (musaGetDevice(&device_id) == musaSuccess) {
      int queried = 0;
      if (musaDeviceGetAttribute(
              &queried, musaDevAttrMultiProcessorCount, device_id) ==
              musaSuccess &&
          queried > 0) {
        sm_count = queried;
      }
    }
    if (sm_count <= 0) {
      sm_count = kDefaultBlockLimit;
    }
  }
  return sm_count;
}

template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

template <typename T>
struct packed_t {
  using P = array_t<T, 16 / sizeof(T)>;
  using A = array_t<float, 16 / sizeof(T)>;
};

template <typename T>
struct fp_trait {};

template <>
struct fp_trait<half> {
  using type = uint16_t;
  static constexpr uint16_t pos_zero = 0x0000u;
  static constexpr uint16_t neg_zero = 0x8000u;
};

template <>
struct fp_trait<__mt_bfloat16> {
  using type = uint16_t;
  static constexpr uint16_t pos_zero = 0x0000u;
  static constexpr uint16_t neg_zero = 0x8000u;
};

template <>
struct fp_trait<float> {
  using type = uint32_t;
  static constexpr uint32_t pos_zero = 0x00000000u;
  static constexpr uint32_t neg_zero = 0x80000000u;
};

template <typename DType>
__device__ __forceinline__ void clear_pos_zero(DType& val) {
  using Trait = fp_trait<DType>;
  auto* ptr = reinterpret_cast<typename Trait::type*>(&val);
  if (*ptr == Trait::pos_zero) {
    *ptr = Trait::neg_zero;
  }
}

template <typename DType>
__device__ __forceinline__ bool is_pos_zero(const DType& val) {
  using Trait = fp_trait<DType>;
  const auto* ptr = reinterpret_cast<const typename Trait::type*>(&val);
  return *ptr == Trait::pos_zero;
}

template <typename DType>
__device__ __forceinline__ DType get_pos_zero() {
  using Trait = fp_trait<DType>;
  const auto value = Trait::pos_zero;
  return *reinterpret_cast<const DType*>(&value);
}

template <typename T, int N>
__device__ __forceinline__ void clear_pos_zero(array_t<T, N>& val) {
#pragma unroll
  for (int i = 0; i < N; ++i) {
    clear_pos_zero(val.data[i]);
  }
}

template <typename T, int N>
__device__ __forceinline__ bool has_pos_zero(array_t<T, N> val) {
  bool found = false;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    found |= is_pos_zero(val.data[i]);
  }
  return found;
}

template <typename P>
__device__ __forceinline__ P make_pos_zero_packet() {
  P out;
#pragma unroll
  for (int i = 0; i < P::size; ++i) {
    out.data[i] = get_pos_zero<typename P::type>();
  }
  return out;
}

template <typename P>
__device__ __forceinline__ P load_volatile_packet(const P* ptr) {
  static_assert(sizeof(P) == 16 && alignof(P) == 16);
#if SGL_CUSTOM_AR_PUSH_16B_ASM
  using i4 = int __attribute__((ext_vector_type(4)));
  uint4 raw;
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 310)
  i4 tmp = __musa_ldcv_v4i32(reinterpret_cast<const i4*>(ptr));
  raw = *reinterpret_cast<uint4*>(&tmp);
#else
  raw = *reinterpret_cast<const uint4*>(ptr);
#endif
  return *reinterpret_cast<const P*>(&raw);
#else
  uint4 raw;
  const auto* src = reinterpret_cast<const volatile uint32_t*>(ptr);
  raw.x = src[0];
  raw.y = src[1];
  raw.z = src[2];
  raw.w = src[3];
  return *reinterpret_cast<const P*>(&raw);
#endif
}

template <typename P>
__device__ __forceinline__ void store_volatile_packet(P* ptr, P value) {
  static_assert(sizeof(P) == 16 && alignof(P) == 16);
  const auto raw = *reinterpret_cast<const uint4*>(&value);
#if SGL_CUSTOM_AR_PUSH_16B_ASM
  using i4 = int __attribute__((ext_vector_type(4)));
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 310)
  __musa_stwb_v4i32(*reinterpret_cast<const i4*>(&raw), reinterpret_cast<i4*>(ptr));
#else
  *reinterpret_cast<uint4*>(ptr) = raw;
#endif
#else
  auto* dst = reinterpret_cast<volatile uint32_t*>(ptr);
  dst[0] = raw.x;
  dst[1] = raw.y;
  dst[2] = raw.z;
  dst[3] = raw.w;
#endif
}

__device__ __forceinline__ float upcast_s(half value) {
  return __half2float(value);
}

__device__ __forceinline__ float upcast_s(__mt_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ float upcast_s(float value) {
  return value;
}

template <typename T>
__device__ __forceinline__ T downcast_s(float value);

template <>
__device__ __forceinline__ half downcast_s<half>(float value) {
  return __float2half(value);
}

template <>
__device__ __forceinline__ __mt_bfloat16 downcast_s<__mt_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <>
__device__ __forceinline__ float downcast_s<float>(float value) {
  return value;
}


#ifndef VLLM_MUSA_FUSED_HAS_RESIDUAL
#define VLLM_MUSA_FUSED_HAS_RESIDUAL 1
#endif
#ifndef VLLM_MUSA_FUSED_WRITE_RAW
#define VLLM_MUSA_FUSED_WRITE_RAW 0
#endif
#ifndef VLLM_MUSA_FUSED_GEMMA
#define VLLM_MUSA_FUSED_GEMMA 0
#endif
#if VLLM_MUSA_FUSED_WRITE_RAW && SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
#error "raw output requires scratch to retain the pre-residual all-reduce value"
#endif

// Preserve vLLM IR's materialization points: AR rounds to the model dtype;
// fused-add RMSNorm accumulates residual in fp32 while storing residual_out
// separately in T. Raw is never recovered by subtracting rounded residuals.
template <typename T>
__device__ __forceinline__ float vllm_fused_norm_value(
    float reduced, float residual, T* raw_out, int offset) {
  const T raw = downcast_s<T>(reduced);
#if VLLM_MUSA_FUSED_WRITE_RAW
  raw_out[offset] = raw;
#endif
#if VLLM_MUSA_FUSED_HAS_RESIDUAL
  return upcast_s(raw) + residual;
#else
  return upcast_s(raw);
#endif
}

// Gemma applies its zero-centered weight in fp32. Ordinary RMSNorm casts the
// normalized value to the weight dtype before multiplying by the weight.
template <typename T>
__device__ __forceinline__ T vllm_fused_apply_weight(float normalized, float weight) {
#if VLLM_MUSA_FUSED_GEMMA
  return downcast_s<T>(normalized * (weight + 1.0f));
#else
  return downcast_s<T>(upcast_s(downcast_s<T>(normalized)) * weight);
#endif
}

// The bypass-load implementation uses volatile assembly. Gate the load
// itself, rather than only its arithmetic consumer, for norm-only modules.
template <typename T, bool Bypass, typename Offset>
__device__ __forceinline__ Vec8<T> vllm_fused_load_residual(
    const T* ptr, Offset offset) {
#if VLLM_MUSA_FUSED_HAS_RESIDUAL
  if constexpr (Bypass) {
    return Vec8<T>::load_byp_slc(ptr, offset);
  } else {
    return Vec8<T>::load(ptr, offset);
  }
#else
  Vec8<T> zero;
#pragma unroll
  for (int i = 0; i < 8; ++i) zero.val.elem[i] = downcast_s<T>(0.0f);
  return zero;
#endif
}

template <typename T>
__device__ __forceinline__ T& assign_add(T& a, T b) {
  a = downcast_s<T>(upcast_s(a) + upcast_s(b));
  return a;
}

template <>
__device__ __forceinline__ float& assign_add<float>(float& a, float b) {
  a += b;
  return a;
}

template <typename T, int N>
__device__ __forceinline__ array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; ++i) {
    assign_add(a.data[i], b.data[i]);
  }
  return a;
}

template <typename T, int N>
__device__ __forceinline__ array_t<float, N> upcast(array_t<T, N> value) {
  if constexpr (std::is_same<T, float>::value) {
    return value;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      out.data[i] = upcast_s(value.data[i]);
    }
    return out;
  }
}

template <typename O>
__device__ __forceinline__ O downcast(array_t<float, O::size> value) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return value;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; ++i) {
      out.data[i] = downcast_s<typename O::type>(value.data[i]);
    }
    return out;
  }
}

template <typename P, int nranks, typename A>
__device__ __forceinline__ P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < nranks; ++i) {
    packed_assign_add(tmp, upcast(ptrs[i][idx]));
  }
  return downcast<P>(tmp);
}

template <typename P>
__device__ __forceinline__ P* get_tmp_buf(Signal* signal) {
  return reinterpret_cast<P*>(signal + 1);
}

template <typename T, int nranks, int vlen = 8>
__device__ __forceinline__ void shfl_reduce(float* res) {
  if constexpr (nranks >= 4) {
#pragma unroll
    for (int i = 0; i < vlen; ++i) {
      res[i] += __shfl_xor_sync(0xffffffff, res[i], 16);
    }
  }
#pragma unroll
  for (int i = 0; i < vlen; ++i) {
    res[i] += __shfl_xor_sync(0xffffffff, res[i], 8);
  }
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  return value;
}

__device__ __forceinline__ float block_rms_scale_16warps(
    float value,
    float* warp_sums,
    float* inv_rms,
    float inv_hidden,
    float eps) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value = lane < 16 ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
    if (lane == 0) {
      *inv_rms = fast_rsqrt(value * inv_hidden + eps);
    }
  }
  __syncthreads_lm();
  return *inv_rms;
}

__device__ __forceinline__ float block_sum_16warps_tid0(float value, float* warp_sums) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value = lane < 16 ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
  }
  return value;
}

__device__ __forceinline__ float block_sum_12warps(float value, float* warp_sums) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0 && warp < 12) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value = lane < 12 ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  __syncthreads_lm();
  return warp_sums[0];
}

__device__ __forceinline__ float block_sum_12warps_tid0(float value, float* warp_sums) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0 && warp < 12) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value = lane < 12 ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
  }
  return value;
}

template <int kActiveWarps>
__device__ __forceinline__ float block_sum_nwarps(float value, float* warp_sums) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0 && warp < kActiveWarps) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value = lane < kActiveWarps ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  __syncthreads_lm();
  return warp_sums[0];
}

template <int kActiveWarps>
__device__ __forceinline__ float block_sum_nwarps_tid0(
    float value,
    float* warp_sums) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0 && warp < kActiveWarps) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value = lane < kActiveWarps ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
  }
  return value;
}

__device__ __forceinline__ float block_sum_tid0(float value, float* warp_sums) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int num_warps = (static_cast<int>(blockDim.x) + 31) >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value = tid < num_warps ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
  }
  return value;
}

__device__ __forceinline__ float row_group_sum_tid0(
    float value,
    float* warp_sums,
    int packed_hidden) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int tid_in_row = tid % packed_hidden;
  const int row_group = tid / packed_hidden;
  const int warp_in_row = tid_in_row >> 5;
  const int warps_per_row = packed_hidden >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value =
      (warp_in_row == 0 && lane < warps_per_row)
          ? warp_sums[row_group * warps_per_row + lane]
          : 0.0f;
  if (warp_in_row == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
  }
  return value;
}

__device__ __forceinline__ float row_group_sum_all(
    float value,
    float* warp_sums,
    int packed_hidden) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row_group = tid / packed_hidden;
  const int warps_per_row = packed_hidden >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (i < warps_per_row) {
      sum += warp_sums[row_group * warps_per_row + i];
    }
  }
  __syncthreads_lm();
  return sum;
}

__device__ __forceinline__ float row_group_rms_scale_shared(
    float value,
    float* warp_sums,
    float* inv_rms,
    float inv_hidden,
    float eps,
    int packed_hidden) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warps_per_row = packed_hidden >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value = lane < warps_per_row ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
    if (lane == 0) {
      *inv_rms = fast_rsqrt(value * inv_hidden + eps);
    }
  }
  __syncthreads_lm();
  return *inv_rms;
}

__device__ __forceinline__ float row_group_rms_scale_all(
    float value,
    float* warp_sums,
    float* inv_rms,
    float inv_hidden,
    float eps,
    int packed_hidden) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row_group = tid / packed_hidden;
  const int warp_in_row = (tid % packed_hidden) >> 5;
  const int warps_per_row = packed_hidden >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  if (warp_in_row == 0) {
    value = lane < warps_per_row
                ? warp_sums[row_group * warps_per_row + lane]
                : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
    if (lane == 0) {
      inv_rms[row_group] = fast_rsqrt(value * inv_hidden + eps);
    }
  }
  __syncthreads_lm();
  return inv_rms[row_group];
}

__device__ __forceinline__ float row_group_sum_all_wide(
    float value,
    float* warp_sums,
    int packed_hidden) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row_group = tid / packed_hidden;
  const int warps_per_row = packed_hidden >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    if (i < warps_per_row) {
      sum += warp_sums[row_group * warps_per_row + i];
    }
  }
  __syncthreads_lm();
  return sum;
}

constexpr size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

template <int nranks, bool start>
__device__ __forceinline__ void multi_rank_barrier(const RankSignals& sg, Signal* self_sg, int rank) {
  if constexpr (!start) {
    __syncthreads_lm();
  }
  if (threadIdx.x < nranks &&
      (!SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_SKIP_SELF_SIGNAL ||
       threadIdx.x != rank)) {
    const int target_rank = threadIdx.x;
    PullSemaphore* local = &self_sg->pull[blockIdx.x];
    PullSemaphore* target = &sg.signals[target_rank]->pull[blockIdx.x];
    const FlagType generation =
        atomicAdd(&local->self_counter[target_rank], 1) + 1;
    atomicExch(&target->peer_counter[generation & 1][rank], generation);
    while (atomicAdd(
               &local->peer_counter[generation & 1][target_rank], 0) !=
           generation) {
    }
    if constexpr (start) {
      __musa_fence_acq();
    }
  }
  if constexpr (start) {
    __syncthreads_lm();
  }
}

template <int nranks>
__device__ __forceinline__ void multi_rank_start_barrier_no_end(
    const RankSignals& sg, Signal* self_sg, int rank) {
  multi_rank_barrier<nranks, true>(sg, self_sg, rank);
}

template <int nranks>
__device__ __forceinline__ void row_multi_rank_end_barrier(
    const RankSignals& sg, Signal* self_sg, int rank) {
#if SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_SKIP_END_BARRIER
  (void)sg;
  (void)self_sg;
  (void)rank;
#else
  if constexpr (nranks != 4) {
    multi_rank_barrier<nranks, false>(sg, self_sg, rank);
  }
#endif
}

template <int nranks>
__device__ __forceinline__ void publish_barrier(
    const RankSignals& sg, Signal* self_sg, int rank) {
  __syncthreads_lm();
  __musa_barrier_slc();
  if (threadIdx.x == 0) {
    __musa_fence_rel();
  }
  __syncthreads_lm();
  if (threadIdx.x < nranks &&
      (!SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_SKIP_SELF_SIGNAL ||
       threadIdx.x != rank)) {
    const int target_rank = threadIdx.x;
    PullSemaphore* local = &self_sg->publish[blockIdx.x];
    PullSemaphore* target = &sg.signals[target_rank]->publish[blockIdx.x];
    const FlagType generation =
        atomicAdd(&local->self_counter[target_rank], 1) + 1;
    atomicExch(&target->peer_counter[generation & 1][rank], generation);
    while (atomicAdd(
               &local->peer_counter[generation & 1][target_rank], 0) !=
           generation) {
    }
    __musa_fence_acq();
  }
  __syncthreads_lm();
}

template <int nranks>
__device__ __forceinline__ void publish_barrier_peer_slots(
    const RankSignals& sg, Signal* self_sg, int rank) {
#if SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_PEER_PUBLISH
  __syncthreads_lm();
  __musa_barrier_slc();
  if (threadIdx.x == 0) {
    __musa_fence_rel();
  }
  __syncthreads_lm();
  if (threadIdx.x < nranks) {
    const int target_rank = threadIdx.x;
    PullSemaphore* target = &sg.signals[target_rank]->publish[blockIdx.x];
    if (target_rank == rank) {
      PullSemaphore* local = &self_sg->publish[blockIdx.x];
      const FlagType phase = static_cast<FlagType>(nranks);
      const FlagType base = atomicAdd(&local->counter, phase);
      atomicAdd(&target->flag, 1);
      while (static_cast<FlagType>(atomicAdd(&local->flag, 0) - base) < phase) {
      }
      __musa_fence_acq();
    } else {
      atomicAdd(&target->flag, 1);
    }
  }
  __syncthreads_lm();
#else
  publish_barrier<nranks>(sg, self_sg, rank);
#endif
}

template <typename T, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
custom_all_reduce_residual_rmsnorm_kernel(
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual_in,
    T* __restrict__ residual_out,
    T* __restrict__ norm_out,
    T* __restrict__ raw_out,
    const T* __restrict__ weight,
    int rank,
    int rows,
    int hidden,
    float eps) {
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ float inv_rms;
  __shared__ RankData data;
  const int tid = threadIdx.x;
  if (tid < nranks) {
    data.ptrs[tid] = data_ptr->ptrs[tid];
  }
  __syncthreads_lm();
  const bool cache_float_values =
#if VLLM_MUSA_FUSED_HAS_RESIDUAL
      true;
#else
      hidden <= kCacheHiddenLimit &&
      !(nranks == 2 && hidden == 2048) &&
      !(nranks == 2 && hidden == 4096 && rows < 2048);
#endif
  const bool cache_t_values =
      !cache_float_values && !(nranks == 2 && hidden == 2048) &&
      hidden != 4096 && hidden <= kTypedCacheHiddenLimit;
  const bool cache_weight = rmsnorm_cache_weight_hidden(hidden, rows);
  float* cached = reinterpret_cast<float*>(smem);
  T* cached_t = reinterpret_cast<T*>(smem);
  const size_t cache_bytes =
      cache_float_values
          ? static_cast<size_t>(hidden) * sizeof(float)
          : (cache_t_values ? static_cast<size_t>(hidden) * sizeof(T) : 0);
  const size_t weight_offset = align_up(cache_bytes, alignof(T));
  T* cached_weight = reinterpret_cast<T*>(smem + weight_offset);
  const size_t weight_bytes = cache_weight ? static_cast<size_t>(hidden) * sizeof(T) : 0;
  float* warp_sums =
      reinterpret_cast<float*>(smem + align_up(weight_offset + weight_bytes, alignof(float)));
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack = P::size;
  const int packed_hidden = hidden / pack;
  const bool use_packed =
      (SGL_CUSTOM_AR_FUSED_RMSNORM_PACKED_1STAGE ||
       SGL_CUSTOM_AR_FUSED_RMSNORM_SAFE_PACKED_1STAGE ||
       SGL_CUSTOM_AR_FUSED_RMSNORM_VEC2RANK_1STAGE) &&
      (hidden % pack) == 0;
  const bool use_partial_packed_non8 =
      (nranks == 4 || nranks == 8) && !use_packed &&
      SGL_CUSTOM_AR_FUSED_RMSNORM_VEC2RANK_1STAGE &&
      SGL_CUSTOM_AR_FUSED_RMSNORM_PARTIAL_PACKED_NON8 &&
      hidden >= pack;
  const bool use_vec4_non8 =
      (nranks == 4 || nranks == 8) && !use_packed &&
      SGL_CUSTOM_AR_FUSED_RMSNORM_VEC4_NON8 &&
      !use_partial_packed_non8 && sizeof(T) == 2 && ((hidden & 3) == 0);
  const bool use_vec2_non8 =
      (nranks == 4 || nranks == 8) && !use_packed &&
      SGL_CUSTOM_AR_FUSED_RMSNORM_VEC2_NON8 &&
      !use_partial_packed_non8 && !use_vec4_non8 && sizeof(T) == 2 &&
      ((hidden & 1) == 0);

  if (cache_weight && use_packed) {
    for (int packed_col = tid; packed_col < packed_hidden; packed_col += blockDim.x) {
      const int col0 = packed_col * pack;
      *(Vec8<T>*)(cached_weight + col0) = Vec8<T>::load(weight, col0);
    }
  }

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);

  if constexpr (nranks == 2) {
    if (SGL_CUSTOM_AR_FUSED_RMSNORM_VEC2RANK_1STAGE &&
        hidden <= 2048 && (hidden % pack) == 0 &&
        (blockDim.x % packed_hidden) == 0) {
      const int rows_per_block = blockDim.x / packed_hidden;
      const int row_slot = tid / packed_hidden;
      const int packed_col = tid - row_slot * packed_hidden;
      const auto* local = reinterpret_cast<const P*>(data.ptrs[rank]);
      const auto* peer = reinterpret_cast<const P*>(data.ptrs[rank ^ 1]);
      const auto* residual = reinterpret_cast<const P*>(residual_in);
      auto* residual_dst = reinterpret_cast<P*>(residual_out);
      auto* norm_dst = reinterpret_cast<P*>(norm_out);
      const auto* weight_vec = reinterpret_cast<const P*>(weight);

      for (int row_base = blockIdx.x * rows_per_block; row_base < rows;
           row_base += gridDim.x * rows_per_block) {
        const int row = row_base + row_slot;
        float values[pack] = {0.0f};
        float square_sum = 0.0f;
        P residual_packet_out;
        if (row < rows) {
          const int packed_idx = row * packed_hidden + packed_col;
          const P local_vec = local[packed_idx];
          const P peer_vec = peer[packed_idx];
          const P residual_vec = residual[packed_idx];
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            float val = upcast_s(local_vec.data[i]);
            val += upcast_s(peer_vec.data[i]);
            val = vllm_fused_norm_value<T>(val, upcast_s(residual_vec.data[i]), raw_out, packed_idx * pack + i);
            values[i] = val;
            residual_packet_out.data[i] = downcast_s<T>(val);
            square_sum += val * val;
          }
          residual_dst[packed_idx] = residual_packet_out;
        }

        const float row_square_sum =
            row_group_sum_all(square_sum, warp_sums, packed_hidden);
        const float scale =
            fast_rsqrt(row_square_sum / static_cast<float>(hidden) + eps);

        if (row < rows) {
          const int packed_idx = row * packed_hidden + packed_col;
          const P w = weight_vec[packed_col];
          P norm_packet;
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            const float gamma = upcast_s(w.data[i]);
            norm_packet.data[i] =
                vllm_fused_apply_weight<T>(values[i] * scale, gamma);
          }
          norm_dst[packed_idx] = norm_packet;
        }
      }
      multi_rank_barrier<nranks, false>(sg, self_sg, rank);
      return;
    }
  }

  if constexpr (nranks == 4 || nranks == 8) {
    if (SGL_CUSTOM_AR_FUSED_RMSNORM_VEC2RANK_1STAGE &&
        hidden <= 2048 && (hidden % pack) == 0 &&
        blockDim.x >= packed_hidden && (blockDim.x % packed_hidden) == 0) {
      const int rows_per_block = blockDim.x / packed_hidden;
      const int row_slot = tid / packed_hidden;
      const int packed_col = tid - row_slot * packed_hidden;
      const int col0 = packed_col * pack;
      for (int row_base = blockIdx.x * rows_per_block; row_base < rows;
           row_base += gridDim.x * rows_per_block) {
        const int row = row_base + row_slot;
        Float8 sum_float;
        float square_sum_row = 0.0f;
        if (row < rows) {
          float acc[pack] = {0.0f};
#pragma unroll
          for (int r = 0; r < nranks; ++r) {
            const auto* peer = reinterpret_cast<const T*>(data.ptrs[r]);
            Vec8<T> x = Vec8<T>::load(peer + row * hidden, col0);
#pragma unroll
            for (int i = 0; i < pack; ++i) {
              acc[i] += to_float<T>(x.val.elem[i]);
            }
          }
          Vec8<T> residual_vec = vllm_fused_load_residual<T, false>(residual_in + row * hidden, col0);
          Vec8<T> residual_vec_out;
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            float val = vllm_fused_norm_value<T>(acc[i], to_float<T>(residual_vec.val.elem[i]), raw_out, row * hidden + col0 + i);
            sum_float.val.elem[i] = val;
            residual_vec_out.val.elem[i] = from_float<T>(val);
            square_sum_row += val * val;
          }
          *(Vec8<T>*)(residual_out + row * hidden + col0) = residual_vec_out;
        }

        const float row_square_sum =
            row_group_sum_all(square_sum_row, warp_sums, packed_hidden);
        const float scale =
            fast_rsqrt(row_square_sum / static_cast<float>(hidden) + eps);

        if (row < rows) {
          Vec8<T> w = Vec8<T>::load(weight, col0);
          Vec8<T> dst;
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            const float gamma = to_float<T>(w.val.elem[i]);
            dst.val.elem[i] =
                vllm_fused_apply_weight<T>(sum_float.val.elem[i] * scale, gamma);
          }
          *(Vec8<T>*)(norm_out + row * hidden + col0) = dst;
        }
      }
      multi_rank_barrier<nranks, false>(sg, self_sg, rank);
      return;
    }

    if (SGL_CUSTOM_AR_FUSED_RMSNORM_VEC2RANK_1STAGE &&
        (hidden == 4096 || hidden == 8192) && (hidden % pack) == 0 &&
        blockDim.x >= packed_hidden && (blockDim.x % packed_hidden) == 0) {
      const int rows_per_block = static_cast<int>(blockDim.x) / packed_hidden;
      const int row_slot = tid / packed_hidden;
      const int packed_col = tid - row_slot * packed_hidden;
      const int col0 = packed_col * pack;
      for (int row_base = blockIdx.x * rows_per_block; row_base < rows;
           row_base += gridDim.x * rows_per_block) {
        const int row = row_base + row_slot;
        Float8 sum_float;
        float square_sum_row = 0.0f;
        if (row < rows) {
          float acc[pack] = {0.0f};
#pragma unroll
          for (int r = 0; r < nranks; ++r) {
            const auto* peer = reinterpret_cast<const T*>(data.ptrs[r]);
            Vec8<T> x = Vec8<T>::load(peer + row * hidden, col0);
#pragma unroll
            for (int i = 0; i < pack; ++i) {
              acc[i] += to_float<T>(x.val.elem[i]);
            }
          }
          Vec8<T> residual_vec = vllm_fused_load_residual<T, false>(residual_in + row * hidden, col0);
          Vec8<T> residual_vec_out;
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            float val = vllm_fused_norm_value<T>(acc[i], to_float<T>(residual_vec.val.elem[i]), raw_out, row * hidden + col0 + i);
            sum_float.val.elem[i] = val;
            residual_vec_out.val.elem[i] = from_float<T>(val);
            square_sum_row += val * val;
          }
          *(Vec8<T>*)(residual_out + row * hidden + col0) = residual_vec_out;
        }

        const float row_square_sum =
            packed_hidden <= 256
                ? row_group_sum_all(square_sum_row, warp_sums, packed_hidden)
                : row_group_sum_all_wide(square_sum_row, warp_sums, packed_hidden);
        const float scale =
            fast_rsqrt(row_square_sum / static_cast<float>(hidden) + eps);

        if (row < rows) {
          Vec8<T> w = Vec8<T>::load(weight, col0);
          Vec8<T> dst;
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            const float gamma = to_float<T>(w.val.elem[i]);
            dst.val.elem[i] =
                vllm_fused_apply_weight<T>(sum_float.val.elem[i] * scale, gamma);
          }
          *(Vec8<T>*)(norm_out + row * hidden + col0) = dst;
        }
      }
      multi_rank_barrier<nranks, false>(sg, self_sg, rank);
      return;
    }
  }

  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int base = row * hidden;
    float square_sum = 0.0f;
    if constexpr (nranks == 2) {
      if (SGL_CUSTOM_AR_FUSED_RMSNORM_REGCACHE_2RANK &&
          (hidden % pack) == 0 && blockDim.x == packed_hidden &&
          packed_hidden <= kMaxThreadsPerBlock &&
          (hidden == 4096 || (hidden & (hidden - 1)) != 0)) {
        const auto* local = reinterpret_cast<const T*>(data.ptrs[rank]);
        const auto* peer = reinterpret_cast<const T*>(data.ptrs[rank ^ 1]);
        const int col0 = tid * pack;
        Float8 sum_float;
        Vec8<T> local_vec = Vec8<T>::load(local + base, col0);
        Vec8<T> peer_vec = Vec8<T>::load(peer + base, col0);
        Vec8<T> residual_vec = vllm_fused_load_residual<T, false>(residual_in + base, col0);
        Vec8<T> residual_vec_out;
#pragma unroll
        for (int i = 0; i < pack; ++i) {
          float val = to_float<T>(local_vec.val.elem[i]);
          val += to_float<T>(peer_vec.val.elem[i]);
          val = vllm_fused_norm_value<T>(val, to_float<T>(residual_vec.val.elem[i]), raw_out, base + col0 + i);
          sum_float.val.elem[i] = val;
          residual_vec_out.val.elem[i] = from_float<T>(val);
          square_sum += val * val;
        }
        *(Vec8<T>*)(residual_out + base + col0) = residual_vec_out;

        const float row_square_sum = block_sum(square_sum, warp_sums);
        if (tid == 0) {
          inv_rms = fast_rsqrt(row_square_sum / static_cast<float>(hidden) + eps);
        }
        __syncthreads_lm();

        Vec8<T> w = Vec8<T>::load(weight, col0);
        Vec8<T> dst;
#pragma unroll
        for (int i = 0; i < pack; ++i) {
          const float gamma = to_float<T>(w.val.elem[i]);
          dst.val.elem[i] = vllm_fused_apply_weight<T>(sum_float.val.elem[i] * inv_rms, gamma);
        }
        *(Vec8<T>*)(norm_out + base + col0) = dst;
        if (row + gridDim.x < rows) {
          __syncthreads_lm();
        }
        continue;
      }
    }
    if constexpr (nranks == 2) {
      if (SGL_CUSTOM_AR_FUSED_RMSNORM_VEC2RANK_1STAGE && (hidden % pack) == 0) {
        const auto* local = reinterpret_cast<const T*>(data.ptrs[rank]);
        const auto* peer = reinterpret_cast<const T*>(data.ptrs[rank ^ 1]);
        for (int col0 = tid * pack; col0 < hidden; col0 += blockDim.x * pack) {
          Vec8<T> local_vec = Vec8<T>::load(local + base, col0);
          Vec8<T> peer_vec = Vec8<T>::load(peer + base, col0);
          Vec8<T> residual_vec = vllm_fused_load_residual<T, false>(residual_in + base, col0);
          Vec8<T> residual_vec_out;
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            const int col = col0 + i;
            float val = to_float<T>(local_vec.val.elem[i]);
            val += to_float<T>(peer_vec.val.elem[i]);
            val = vllm_fused_norm_value<T>(val, to_float<T>(residual_vec.val.elem[i]), raw_out, base + col0 + i);
            residual_vec_out.val.elem[i] = from_float<T>(val);
            if (cache_float_values) {
              cached[col] = val;
            } else if (cache_t_values) {
              cached_t[col] = residual_vec_out.val.elem[i];
            }
            square_sum += val * val;
          }
          *(Vec8<T>*)(residual_out + base + col0) = residual_vec_out;
        }
      } else if (use_packed && SGL_CUSTOM_AR_FUSED_RMSNORM_PACKED_1STAGE) {
        const int packed_base = row * packed_hidden;
        for (int packed_col = tid; packed_col < packed_hidden; packed_col += blockDim.x) {
          const int col0 = packed_col * pack;
          float acc[pack] = {0.0f};
#pragma unroll
          for (int r = 0; r < nranks; ++r) {
            const auto* peer = reinterpret_cast<const T*>(data.ptrs[r]);
            Vec8<T> x = Vec8<T>::load_byp_slc(peer + base, col0);
#pragma unroll
            for (int i = 0; i < pack; ++i) {
              acc[i] += to_float<T>(x.val.elem[i]);
            }
          }
          Vec8<T> residual_vec = vllm_fused_load_residual<T, true>(residual_in + base, col0);
          Vec8<T> residual_vec_out;
          Float8 sum_float;
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            const int col = col0 + i;
            float val = vllm_fused_norm_value<T>(acc[i], to_float<T>(residual_vec.val.elem[i]), raw_out, base + col0 + i);
            residual_vec_out.val.elem[i] = from_float<T>(val);
            sum_float.val.elem[i] = val;
            if (cache_float_values) {
              cached[col] = val;
            } else if (cache_t_values) {
              cached_t[col] = residual_vec_out.val.elem[i];
            }
            square_sum += val * val;
          }
          *(Vec8<T>*)(residual_out + base + col0) = residual_vec_out;
        }
      } else {
        for (int col = tid; col < hidden; col += blockDim.x) {
          float ar_sum = 0.0f;
#pragma unroll
          for (int r = 0; r < nranks; ++r) {
            const auto* peer = reinterpret_cast<const T*>(data.ptrs[r]);
            ar_sum += to_float(peer[base + col]);
          }
          const float val = vllm_fused_norm_value<T>(ar_sum, to_float(residual_in[base + col]), raw_out, base + col);
          const T residual_value = from_float<T>(val);
          residual_out[base + col] = residual_value;
          if (cache_float_values) {
            cached[col] = val;
          } else if (cache_t_values) {
            cached_t[col] = residual_value;
          }
          square_sum += val * val;
        }
      }
    } else if (use_packed && SGL_CUSTOM_AR_FUSED_RMSNORM_PACKED_1STAGE) {
      const int packed_base = row * packed_hidden;
      for (int packed_col = tid; packed_col < packed_hidden; packed_col += blockDim.x) {
        const int col0 = packed_col * pack;
        float acc[pack] = {0.0f};
#pragma unroll
        for (int r = 0; r < nranks; ++r) {
          const auto* peer = reinterpret_cast<const T*>(data.ptrs[r]);
          Vec8<T> x = Vec8<T>::load_byp_slc(peer + base, col0);
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            acc[i] += to_float<T>(x.val.elem[i]);
          }
        }
        Vec8<T> residual_vec = vllm_fused_load_residual<T, true>(residual_in + base, col0);
        Vec8<T> residual_vec_out;
        Float8 sum_float;
#pragma unroll
        for (int i = 0; i < pack; ++i) {
          const int col = col0 + i;
          float val = vllm_fused_norm_value<T>(acc[i], to_float<T>(residual_vec.val.elem[i]), raw_out, base + col0 + i);
          residual_vec_out.val.elem[i] = from_float<T>(val);
          sum_float.val.elem[i] = val;
          if (cache_float_values) {
            cached[col] = val;
          } else if (cache_t_values) {
            cached_t[col] = residual_vec_out.val.elem[i];
          }
          square_sum += val * val;
        }
        *(Vec8<T>*)(residual_out + base + col0) = residual_vec_out;
      }
    } else if (use_packed) {
      const int packed_base = row * packed_hidden;
      for (int packed_col = tid; packed_col < packed_hidden; packed_col += blockDim.x) {
        const int packed_idx = packed_base + packed_col;
        const int col0 = packed_col * pack;
        A reduced_acc = upcast(
            reinterpret_cast<const P*>(data.ptrs[0])[packed_idx]);
#pragma unroll
        for (int r = 1; r < nranks; ++r) {
          packed_assign_add(
              reduced_acc,
              upcast(reinterpret_cast<const P*>(data.ptrs[r])[packed_idx]));
        }
        P reduced = downcast<P>(reduced_acc);
        P residual_packet = reinterpret_cast<const P*>(residual_in)[packed_idx];
        P residual_packet_out;
#pragma unroll
        for (int i = 0; i < pack; ++i) {
          const int col = col0 + i;
          float val = upcast_s(reduced.data[i]);
          val = vllm_fused_norm_value<T>(val, upcast_s(residual_packet.data[i]), raw_out, packed_idx * pack + i);
          residual_packet_out.data[i] = downcast_s<T>(val);
          if (cache_float_values) {
            cached[col] = val;
          } else if (cache_t_values) {
            cached_t[col] = residual_packet_out.data[i];
          }
          square_sum += val * val;
        }
        reinterpret_cast<P*>(residual_out)[packed_idx] = residual_packet_out;
      }
    } else if (use_partial_packed_non8) {
      const int row_mod = base & (pack - 1);
      const int aligned_start = row_mod == 0 ? 0 : min(hidden, pack - row_mod);
      const int aligned_packed_hidden = (hidden - aligned_start) / pack;
      const int aligned_end = aligned_start + aligned_packed_hidden * pack;
      for (int col = tid; col < aligned_start; col += blockDim.x) {
        float ar_sum = 0.0f;
#pragma unroll
        for (int r = 0; r < nranks; ++r) {
          const auto* peer = reinterpret_cast<const T*>(data.ptrs[r]);
          ar_sum += to_float(peer[base + col]);
        }
        const float val = vllm_fused_norm_value<T>(ar_sum, to_float(residual_in[base + col]), raw_out, base + col);
        const T residual_value = from_float<T>(val);
        residual_out[base + col] = residual_value;
        if (cache_float_values) {
          cached[col] = val;
        } else if (cache_t_values) {
          cached_t[col] = residual_value;
        }
        square_sum += val * val;
      }
      for (int packed_col = tid; packed_col < aligned_packed_hidden;
           packed_col += blockDim.x) {
        const int col0 = aligned_start + packed_col * pack;
        float acc[pack] = {0.0f};
#pragma unroll
        for (int r = 0; r < nranks; ++r) {
          const auto* peer = reinterpret_cast<const T*>(data.ptrs[r]);
          Vec8<T> x = Vec8<T>::load_byp_slc(peer + base, col0);
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            acc[i] += to_float<T>(x.val.elem[i]);
          }
        }
        Vec8<T> residual_vec = vllm_fused_load_residual<T, true>(residual_in + base, col0);
        Vec8<T> residual_vec_out;
#pragma unroll
        for (int i = 0; i < pack; ++i) {
          const int col = col0 + i;
          float val = vllm_fused_norm_value<T>(acc[i], to_float<T>(residual_vec.val.elem[i]), raw_out, base + col0 + i);
          residual_vec_out.val.elem[i] = from_float<T>(val);
          if (cache_float_values) {
            cached[col] = val;
          } else if (cache_t_values) {
            cached_t[col] = residual_vec_out.val.elem[i];
          }
          square_sum += val * val;
        }
        *(Vec8<T>*)(residual_out + base + col0) = residual_vec_out;
      }
      for (int col = aligned_end + tid; col < hidden; col += blockDim.x) {
        float ar_sum = 0.0f;
#pragma unroll
        for (int r = 0; r < nranks; ++r) {
          const auto* peer = reinterpret_cast<const T*>(data.ptrs[r]);
          ar_sum += to_float(peer[base + col]);
        }
        const float val = vllm_fused_norm_value<T>(ar_sum, to_float(residual_in[base + col]), raw_out, base + col);
        const T residual_value = from_float<T>(val);
        residual_out[base + col] = residual_value;
        if (cache_float_values) {
          cached[col] = val;
        } else if (cache_t_values) {
          cached_t[col] = residual_value;
        }
        square_sum += val * val;
      }
    } else if (use_vec4_non8) {
      using Vec4 = int64_t;
      const int packed4_hidden = hidden >> 2;
      const int base4 = row * packed4_hidden;
      const Vec4* residual4 = reinterpret_cast<const Vec4*>(residual_in);
      Vec4* residual_out4 = reinterpret_cast<Vec4*>(residual_out);
      Vec4* cached_t4 = reinterpret_cast<Vec4*>(cached_t);
      for (int packed_col = tid; packed_col < packed4_hidden;
           packed_col += blockDim.x) {
        float vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int r = 0; r < nranks; ++r) {
          const Vec4 raw = reinterpret_cast<const Vec4*>(data.ptrs[r])[
              base4 + packed_col];
          const T* src = reinterpret_cast<const T*>(&raw);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            vals[i] += upcast_s(src[i]);
          }
        }
        const Vec4 residual_raw = residual4[base4 + packed_col];
        const T* residual_src = reinterpret_cast<const T*>(&residual_raw);
        Vec4 out;
        T* dst = reinterpret_cast<T*>(&out);
        const int col0 = packed_col << 2;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          vals[i] = vllm_fused_norm_value<T>(vals[i], upcast_s(residual_src[i]), raw_out, base + (packed_col << 2) + i);
          dst[i] = downcast_s<T>(vals[i]);
          if (cache_float_values) {
            cached[col0 + i] = vals[i];
          }
          square_sum += vals[i] * vals[i];
        }
        residual_out4[base4 + packed_col] = out;
        if (cache_t_values) {
          cached_t4[packed_col] = out;
        }
      }
    } else if (use_vec2_non8) {
      using Vec2 = int32_t;
      const int packed2_hidden = hidden >> 1;
      const int base2 = row * packed2_hidden;
      const Vec2* residual2 = reinterpret_cast<const Vec2*>(residual_in);
      Vec2* residual_out2 = reinterpret_cast<Vec2*>(residual_out);
      Vec2* cached_t2 = reinterpret_cast<Vec2*>(cached_t);
      for (int packed_col = tid; packed_col < packed2_hidden;
           packed_col += blockDim.x) {
        float vals[2] = {0.0f, 0.0f};
#pragma unroll
        for (int r = 0; r < nranks; ++r) {
          const Vec2 raw = reinterpret_cast<const Vec2*>(data.ptrs[r])[
              base2 + packed_col];
          const T* src = reinterpret_cast<const T*>(&raw);
          vals[0] += upcast_s(src[0]);
          vals[1] += upcast_s(src[1]);
        }
        const Vec2 residual_raw = residual2[base2 + packed_col];
        const T* residual_src = reinterpret_cast<const T*>(&residual_raw);
        vals[0] = vllm_fused_norm_value<T>(vals[0], upcast_s(residual_src[0]), raw_out, base + (packed_col << 1) + 0);
        vals[1] = vllm_fused_norm_value<T>(vals[1], upcast_s(residual_src[1]), raw_out, base + (packed_col << 1) + 1);

        Vec2 out;
        T* dst = reinterpret_cast<T*>(&out);
        dst[0] = downcast_s<T>(vals[0]);
        dst[1] = downcast_s<T>(vals[1]);
        residual_out2[base2 + packed_col] = out;
        const int col0 = packed_col << 1;
        if (cache_float_values) {
          cached[col0] = vals[0];
          cached[col0 + 1] = vals[1];
        } else if (cache_t_values) {
          cached_t2[packed_col] = out;
        }
        square_sum += vals[0] * vals[0] + vals[1] * vals[1];
      }
    } else {
      for (int col = tid; col < hidden; col += blockDim.x) {
        float ar_sum = 0.0f;
#pragma unroll
        for (int r = 0; r < nranks; ++r) {
          const auto* peer = reinterpret_cast<const T*>(data.ptrs[r]);
          ar_sum += to_float(peer[base + col]);
        }
        const float val = vllm_fused_norm_value<T>(ar_sum, to_float(residual_in[base + col]), raw_out, base + col);
        const T residual_value = from_float<T>(val);
        residual_out[base + col] = residual_value;
        if (cache_float_values) {
          cached[col] = val;
        } else if (cache_t_values) {
          cached_t[col] = residual_value;
        }
        square_sum += val * val;
      }
    }

    const float row_square_sum = block_sum(square_sum, warp_sums);
    if (tid == 0) {
      inv_rms = fast_rsqrt(row_square_sum / static_cast<float>(hidden) + eps);
    }
    __syncthreads_lm();

    if (use_packed) {
      for (int packed_col = tid; packed_col < packed_hidden; packed_col += blockDim.x) {
        const int col0 = packed_col * pack;
        Float8 sum_float;
        if (cache_float_values) {
          sum_float = *(Float8*)(cached + col0);
        } else if (cache_t_values) {
          Vec8<T> r = *(Vec8<T>*)(cached_t + col0);
          #pragma unroll
          for (int i = 0; i < pack; ++i) {
            sum_float.val.elem[i] = to_float<T>(r.val.elem[i]);
          }
        } else {
          Vec8<T> r = Vec8<T>::load(residual_out + base, col0);
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            sum_float.val.elem[i] = to_float<T>(r.val.elem[i]);
          }
        }
        Vec8<T> w = cache_weight ? *(Vec8<T>*)(cached_weight + col0) : Vec8<T>::load(weight, col0);
        Vec8<T> dst;
#pragma unroll
        for (int i = 0; i < pack; ++i) {
          const float gamma = to_float<T>(w.val.elem[i]);
          dst.val.elem[i] = vllm_fused_apply_weight<T>(sum_float.val.elem[i] * inv_rms, gamma);
        }
        *(Vec8<T>*)(norm_out + base + col0) = dst;
      }
    } else if (use_partial_packed_non8) {
      const int row_mod = base & (pack - 1);
      const int aligned_start = row_mod == 0 ? 0 : min(hidden, pack - row_mod);
      const int aligned_packed_hidden = (hidden - aligned_start) / pack;
      const int aligned_end = aligned_start + aligned_packed_hidden * pack;
      for (int col = tid; col < aligned_start; col += blockDim.x) {
        float val;
        if (cache_float_values) {
          val = cached[col];
        } else if (cache_t_values) {
          val = to_float(cached_t[col]);
        } else {
          val = to_float(residual_out[base + col]);
        }
        const float gamma = to_float(weight[col]);
        norm_out[base + col] = vllm_fused_apply_weight<T>(val * inv_rms, gamma);
      }
      for (int packed_col = tid; packed_col < aligned_packed_hidden;
           packed_col += blockDim.x) {
        const int col0 = aligned_start + packed_col * pack;
        Float8 sum_float;
        if (cache_float_values) {
          sum_float = *(Float8*)(cached + col0);
        } else if (cache_t_values) {
          Vec8<T> r = *(Vec8<T>*)(cached_t + col0);
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            sum_float.val.elem[i] = to_float<T>(r.val.elem[i]);
          }
        } else {
          Vec8<T> r = Vec8<T>::load(residual_out + base, col0);
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            sum_float.val.elem[i] = to_float<T>(r.val.elem[i]);
          }
        }
        Vec8<T> w = Vec8<T>::load(weight, col0);
        Vec8<T> dst;
#pragma unroll
        for (int i = 0; i < pack; ++i) {
          const float gamma = to_float<T>(w.val.elem[i]);
          dst.val.elem[i] = vllm_fused_apply_weight<T>(sum_float.val.elem[i] * inv_rms, gamma);
        }
        *(Vec8<T>*)(norm_out + base + col0) = dst;
      }
      for (int col = aligned_end + tid; col < hidden; col += blockDim.x) {
        float val;
        if (cache_float_values) {
          val = cached[col];
        } else if (cache_t_values) {
          val = to_float(cached_t[col]);
        } else {
          val = to_float(residual_out[base + col]);
        }
        const float gamma = to_float(weight[col]);
        norm_out[base + col] = vllm_fused_apply_weight<T>(val * inv_rms, gamma);
      }
    } else if (use_vec4_non8) {
      using Vec4 = int64_t;
      const int packed4_hidden = hidden >> 2;
      const int base4 = row * packed4_hidden;
      const Vec4* cached_t4 = reinterpret_cast<const Vec4*>(cached_t);
      const Vec4* residual_out4 = reinterpret_cast<const Vec4*>(residual_out);
      const Vec4* weight4 = reinterpret_cast<const Vec4*>(weight);
      Vec4* norm_out4 = reinterpret_cast<Vec4*>(norm_out);
      for (int packed_col = tid; packed_col < packed4_hidden;
           packed_col += blockDim.x) {
        float vals[4];
        const int col0 = packed_col << 2;
        if (cache_float_values) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            vals[i] = cached[col0 + i];
          }
        } else {
          const Vec4 raw = cache_t_values ? cached_t4[packed_col]
                                          : residual_out4[base4 + packed_col];
          const T* src = reinterpret_cast<const T*>(&raw);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            vals[i] = upcast_s(src[i]);
          }
        }

        const Vec4 weight_raw = weight4[packed_col];
        const T* weight_src = reinterpret_cast<const T*>(&weight_raw);
        Vec4 out;
        T* dst = reinterpret_cast<T*>(&out);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          dst[i] = vllm_fused_apply_weight<T>(vals[i] * inv_rms, upcast_s(weight_src[i]));
        }
        norm_out4[base4 + packed_col] = out;
      }
    } else if (use_vec2_non8) {
      using Vec2 = int32_t;
      const int packed2_hidden = hidden >> 1;
      const int base2 = row * packed2_hidden;
      const Vec2* cached_t2 = reinterpret_cast<const Vec2*>(cached_t);
      const Vec2* residual_out2 = reinterpret_cast<const Vec2*>(residual_out);
      const Vec2* weight2 = reinterpret_cast<const Vec2*>(weight);
      Vec2* norm_out2 = reinterpret_cast<Vec2*>(norm_out);
      for (int packed_col = tid; packed_col < packed2_hidden;
           packed_col += blockDim.x) {
        float vals[2];
        if (cache_float_values) {
          const int col0 = packed_col << 1;
          vals[0] = cached[col0];
          vals[1] = cached[col0 + 1];
        } else {
          const Vec2 raw = cache_t_values ? cached_t2[packed_col]
                                          : residual_out2[base2 + packed_col];
          const T* src = reinterpret_cast<const T*>(&raw);
          vals[0] = upcast_s(src[0]);
          vals[1] = upcast_s(src[1]);
        }

        const Vec2 weight_raw = weight2[packed_col];
        const T* weight_src = reinterpret_cast<const T*>(&weight_raw);
        Vec2 out;
        T* dst = reinterpret_cast<T*>(&out);
        dst[0] = vllm_fused_apply_weight<T>(vals[0] * inv_rms, upcast_s(weight_src[0]));
        dst[1] = vllm_fused_apply_weight<T>(vals[1] * inv_rms, upcast_s(weight_src[1]));
        norm_out2[base2 + packed_col] = out;
      }
    } else {
      for (int col = tid; col < hidden; col += blockDim.x) {
        float val;
        if (cache_float_values) {
          val = cached[col];
        } else if (cache_t_values) {
          val = to_float(cached_t[col]);
        } else {
          val = to_float(residual_out[base + col]);
        }
        const float gamma = to_float(weight[col]);
        norm_out[base + col] = vllm_fused_apply_weight<T>(val * inv_rms, gamma);
      }
    }
    if (row + gridDim.x < rows) {
      __syncthreads_lm();
    }
  }

  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <typename T, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
custom_all_reduce_residual_rmsnorm_2stage_kernel(
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual_in,
    T* __restrict__ residual_out,
    T* __restrict__ norm_out,
    T* __restrict__ raw_out,
    const T* __restrict__ weight,
    int rank,
    int rows,
    int hidden,
    float eps,
    int packed_size) {
  const RankData& data = *data_ptr;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack = P::size;

  extern __shared__ __align__(16) unsigned char smem[];

  const int tid = threadIdx.x;
  const bool cache_float_values = hidden <= kCacheHiddenLimit;
  const bool cache_t_values =
      !cache_float_values && hidden != 4096 && hidden <= kTypedCacheHiddenLimit;
  const bool cache_weight = rmsnorm_cache_weight_hidden(hidden, rows);
  float* cached = reinterpret_cast<float*>(smem);
  T* cached_t = reinterpret_cast<T*>(smem);
  const size_t cache_bytes =
      cache_float_values
          ? static_cast<size_t>(hidden) * sizeof(float)
          : (cache_t_values ? static_cast<size_t>(hidden) * sizeof(T) : 0);
  const size_t weight_offset = align_up(cache_bytes, alignof(T));
  T* cached_weight = reinterpret_cast<T*>(smem + weight_offset);
  const size_t weight_bytes = cache_weight ? static_cast<size_t>(hidden) * sizeof(T) : 0;
  float* warp_sums =
      reinterpret_cast<float*>(smem + align_up(weight_offset + weight_bytes, alignof(float)));

  const P* ptrs[nranks];
#pragma unroll
  for (int i = 0; i < nranks; ++i) {
    ptrs[i] = reinterpret_cast<const P*>(data.ptrs[i]);
  }
  const P* residual_p = reinterpret_cast<const P*>(residual_in);
  const P* weight_p = reinterpret_cast<const P*>(weight);
  P* residual_out_p = reinterpret_cast<P*>(residual_out);
  P* norm_out_p = reinterpret_cast<P*>(norm_out);
  P* self_tmp = get_tmp_buf<P>(self_sg);

  if (cache_weight) {
    for (int packed_col = tid; packed_col < hidden / pack; packed_col += blockDim.x) {
      const int col0 = packed_col * pack;
      *(Vec8<T>*)(cached_weight + col0) = Vec8<T>::load(weight, col0);
    }
  }

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);

  const int packed_hidden = hidden / pack;
  const int rows_per_rank = rows / nranks;
  const int row_remainder = rows % nranks;

  const int owner_begin =
      rank * rows_per_rank + (row_remainder > rank ? rank : row_remainder);
  const int owner_rows = rows_per_rank + (row_remainder > rank ? 1 : 0);
  const int owner_end = owner_begin + owner_rows;

  // The producer and consumer of a row must use the same block-local
  // cross-rank barrier. Index from the global block row and filter by owner;
  // owner_begin + blockIdx.x would shift the producer block whenever an
  // owner's shard does not start on a gridDim boundary.
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    if (row < owner_begin || row >= owner_end) {
      continue;
    }
    const int packed_base = row * packed_hidden;
    for (int packed_col = tid; packed_col < packed_hidden; packed_col += blockDim.x) {
      const int packed_idx = packed_base + packed_col;
      P reduced = packed_reduce<P, nranks, A>(ptrs, packed_idx);
      self_tmp[packed_idx] = reduced;
    }
  }

  __musa_barrier_slc();
  __syncthreads_lm();
  if (tid == 0) {
    __threadfence_system_noflush();
  }
  publish_barrier<nranks>(sg, self_sg, rank);

  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int base = row * hidden;
    const int packed_base = row * packed_hidden;
    float square_sum = 0.0f;
    int owner = row_remainder == 0 ? row / rows_per_rank : nranks - 1;
    int scan_begin = 0;
    if (row_remainder != 0) {
#pragma unroll
      for (int r = 0; r < nranks; ++r) {
        const int scan_rows = rows_per_rank + (row_remainder > r ? 1 : 0);
        if (row >= scan_begin && row < scan_begin + scan_rows) {
          owner = r;
        }
        scan_begin += scan_rows;
      }
    }
    const P* owner_tmp = get_tmp_buf<P>(sg.signals[owner]);

    for (int packed_col = tid; packed_col < packed_hidden; packed_col += blockDim.x) {
      const int packed_idx = packed_base + packed_col;
      P reduced = owner_tmp[packed_idx];
      P residual_packet = residual_p[packed_idx];
      P residual_packet_out;

#pragma unroll
      for (int i = 0; i < pack; ++i) {
        const int col = packed_col * pack + i;
        float val = upcast_s(reduced.data[i]);
        val = vllm_fused_norm_value<T>(val, upcast_s(residual_packet.data[i]), raw_out, packed_idx * pack + i);
        T residual_value = downcast_s<T>(val);
        residual_packet_out.data[i] = residual_value;
        if (cache_float_values) {
          cached[col] = val;
        } else if (cache_t_values) {
          cached_t[col] = residual_value;
        }
        square_sum += val * val;
      }
      residual_out_p[packed_idx] = residual_packet_out;
    }

    const float row_square_sum = block_sum(square_sum, warp_sums);
    const float inv_rms = fast_rsqrt(row_square_sum / static_cast<float>(hidden) + eps);

    for (int packed_col = tid; packed_col < packed_hidden; packed_col += blockDim.x) {
      const int packed_idx = packed_base + packed_col;
      P weight_packet =
          cache_weight ? reinterpret_cast<const P*>(cached_weight)[packed_col]
                       : weight_p[packed_col];
      P norm_packet;
#pragma unroll
      for (int i = 0; i < pack; ++i) {
        const int col = packed_col * pack + i;
        float val;
        if (cache_float_values) {
          val = cached[col];
        } else if (cache_t_values) {
          val = upcast_s(cached_t[col]);
        } else {
          val = upcast_s(residual_out[base + col]);
        }
        const float gamma = upcast_s(weight_packet.data[i]);
        norm_packet.data[i] = vllm_fused_apply_weight<T>(val * inv_rms, gamma);
      }
      norm_out_p[packed_idx] = norm_packet;
    }
    if (hidden != 3072) {
      __syncthreads_lm();
    }
  }

  if (hidden != 3072) {
    multi_rank_barrier<nranks, false>(sg, self_sg, rank);
  }
}

template <typename T, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
custom_all_reduce_residual_kernel(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual_in,
    T* __restrict__ residual_out,
    int rank,
    int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack = P::size;
  const int packed_size = size / pack;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  const P* ptrs[nranks];
#pragma unroll
  for (int i = 0; i < nranks; ++i) {
    ptrs[i] = reinterpret_cast<const P*>(data.ptrs[i]);
  }
  const P* residual = reinterpret_cast<const P*>(residual_in);
  P* dst = reinterpret_cast<P*>(residual_out);

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);
  for (int idx = tid; idx < packed_size; idx += stride) {
    P reduced = packed_reduce<P, nranks, A>(ptrs, idx);
    P residual_packet = residual[idx];
    P out;
#pragma unroll
    for (int i = 0; i < pack; ++i) {
      out.data[i] =
          downcast_s<T>(upcast_s(reduced.data[i]) + upcast_s(residual_packet.data[i]));
    }
    dst[idx] = out;
  }
  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <typename T, int nranks, int vlen = 8>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
custom_all_reduce_residual_2shot_kernel(
    RankData data,
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual_in,
    T* __restrict__ residual_out,
    int rank,
    int packed_size) {
  constexpr int nranks_sft = (nranks >> 1) - (nranks >> 3);
  constexpr int coalesce_num = 8;
  constexpr int coalesce_sft = 3;
  constexpr int group_stride_sft = nranks_sft + coalesce_sft;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int target_rank = (tid >> coalesce_sft) & (nranks - 1);
  const int group_id = tid >> group_stride_sft;
  const int coalesce_tid = tid & (coalesce_num - 1);
  const int stride = gridDim.x * blockDim.x;

  using Vec = array_t<T, vlen>;
  int idx_base = blockIdx.x * blockDim.x;
  int idx_in_blk = coalesce_tid + (rank << coalesce_sft) + (group_id << group_stride_sft);

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);
  const void* target_data =
      data_ptr != nullptr ? data_ptr->ptrs[target_rank]
                          : data.ptrs[target_rank];
  Vec* target_ptr =
      reinterpret_cast<Vec*>(const_cast<void*>(target_data));
  Vec* buffer_ptr = get_tmp_buf<Vec>(sg.signals[rank]);
  do {
    const int idx = idx_in_blk + idx_base;
    float acc[vlen] = {0};
    if (idx < packed_size) {
#if SGL_CUSTOM_AR_VECTOR_LOAD
      Vec raw = target_ptr[idx];
      const T* src = reinterpret_cast<const T*>(&raw);
#else
      const T* src = reinterpret_cast<const T*>(&target_ptr[idx]);
#endif
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        acc[i] = upcast_s(src[i]);
      }
    }
    shfl_reduce<T, nranks, vlen>(acc);
    if constexpr (nranks == 8) {
      __shared__ float smem[kMaxThreadsPerBlock << 1];
      if (lane < coalesce_num) {
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          smem[warp * vlen * coalesce_num + coalesce_tid * vlen + i] = acc[i];
        }
      }
      __syncthreads_lm();
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        acc[i] += smem[(warp ^ 1) * vlen * coalesce_num + coalesce_tid * vlen + i];
      }
    }
    if (rank == target_rank && idx < packed_size) {
      Vec res;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        reinterpret_cast<T*>(&res)[i] = downcast_s<T>(acc[i]);
      }
      buffer_ptr[idx] = res;
    }
    idx_base += stride;
  } while (idx_base < packed_size);

  publish_barrier<nranks>(sg, self_sg, rank);

  buffer_ptr = get_tmp_buf<Vec>(sg.signals[target_rank]);
  const Vec* residual = reinterpret_cast<const Vec*>(residual_in);
  Vec* dst = reinterpret_cast<Vec*>(residual_out);
  idx_in_blk = coalesce_tid + (target_rank << coalesce_sft) + (group_id << group_stride_sft);
  idx_base = blockIdx.x * blockDim.x;
  do {
    const int idx = idx_in_blk + idx_base;
    if (idx < packed_size) {
      const Vec reduced = buffer_ptr[idx];
      const Vec residual_packet = residual[idx];
      Vec out;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        reinterpret_cast<T*>(&out)[i] = downcast_s<T>(
            upcast_s(reinterpret_cast<const T*>(&reduced)[i]) +
            upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]));
      }
      dst[idx] = out;
    }
    idx_base += stride;
  } while (idx_base < packed_size);

  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <typename T, int nranks, int vlen = 8>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
custom_all_reduce_residual_sums_direct_kernel(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual_in,
    T* __restrict__ residual_out,
    float* __restrict__ row_sums,
    int rank,
    int rows,
    int hidden) {
  static_assert(nranks == 2, "direct residual sums is specialized for TP2");
  const int tid = threadIdx.x;
  constexpr int pack = vlen;
  const int packed_hidden = hidden / pack;
  __shared__ float warp_sums[(kMaxThreadsPerBlock + 31) / 32];

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);

  const auto* local = reinterpret_cast<const T*>(data.ptrs[rank]);
  const auto* peer = reinterpret_cast<const T*>(data.ptrs[rank ^ 1]);
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int base = row * hidden;
    float square_sum = 0.0f;
    for (int packed_col = tid; packed_col < packed_hidden;
         packed_col += blockDim.x) {
      const int col0 = packed_col * pack;
      Vec8<T> local_vec = Vec8<T>::load_byp_slc(local + base, col0);
      Vec8<T> peer_vec = Vec8<T>::load_byp_slc(peer + base, col0);
      Vec8<T> residual_vec = vllm_fused_load_residual<T, true>(residual_in + base, col0);
      Vec8<T> residual_vec_out;
#pragma unroll
      for (int i = 0; i < pack; ++i) {
        const float val = to_float<T>(local_vec.val.elem[i]) +
                          to_float<T>(peer_vec.val.elem[i]) +
                          to_float<T>(residual_vec.val.elem[i]);
        residual_vec_out.val.elem[i] = from_float<T>(val);
        square_sum += val * val;
      }
      *(Vec8<T>*)(residual_out + base + col0) = residual_vec_out;
    }

    const float row_square_sum = block_sum_tid0(square_sum, warp_sums);
    if (tid == 0) {
      row_sums[row] = row_square_sum;
    }
  }

  __musa_barrier_slc();
  __syncthreads_lm();
  if (tid == 0) {
    __threadfence_system_noflush();
  }
  __syncthreads_lm();
  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <typename T, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
custom_all_reduce_residual_sums_scalar_kernel(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual_in,
    T* __restrict__ residual_out,
    float* __restrict__ row_sums,
    int rank,
    int rows,
    int hidden) {
  const int tid = threadIdx.x;
  __shared__ float warp_sums[(kMaxThreadsPerBlock + 31) / 32];
  T* self_tmp = get_tmp_buf<T>(self_sg);

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);

  const T* ptrs[nranks];
#pragma unroll
  for (int r = 0; r < nranks; ++r) {
    ptrs[r] = reinterpret_cast<const T*>(data.ptrs[r]);
  }

  if (rows > 1) {
    if ((hidden & 1) == 0) {
      using Vec2 = int32_t;
      const int packed2_hidden = hidden >> 1;
      const Vec2* ptrs2[nranks];
#pragma unroll
      for (int r = 0; r < nranks; ++r) {
        ptrs2[r] = reinterpret_cast<const Vec2*>(ptrs[r]);
      }
      const Vec2* residual2 = reinterpret_cast<const Vec2*>(residual_in);
      Vec2* residual_out2 = reinterpret_cast<Vec2*>(residual_out);

      for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const int base2 = row * packed2_hidden;
        float square_sum = 0.0f;
        for (int packed_col = tid; packed_col < packed2_hidden;
             packed_col += blockDim.x) {
          float vals[2] = {0.0f, 0.0f};
#pragma unroll
          for (int r = 0; r < nranks; ++r) {
            const Vec2 raw = ptrs2[r][base2 + packed_col];
            const T* src = reinterpret_cast<const T*>(&raw);
            vals[0] += upcast_s(src[0]);
            vals[1] += upcast_s(src[1]);
          }
          const Vec2 residual_raw = residual2[base2 + packed_col];
          const T* residual_src = reinterpret_cast<const T*>(&residual_raw);
          vals[0] += upcast_s(residual_src[0]);
          vals[1] += upcast_s(residual_src[1]);

          Vec2 out;
          T* dst = reinterpret_cast<T*>(&out);
          dst[0] = downcast_s<T>(vals[0]);
          dst[1] = downcast_s<T>(vals[1]);
          residual_out2[base2 + packed_col] = out;
          square_sum += vals[0] * vals[0] + vals[1] * vals[1];
        }

        const float row_square_sum = block_sum_tid0(square_sum, warp_sums);
        if (tid == 0) {
          row_sums[row] = row_square_sum;
        }
      }

      __musa_barrier_slc();
      __syncthreads_lm();
      if (tid == 0) {
        __threadfence_system_noflush();
      }
      __syncthreads_lm();
      multi_rank_barrier<nranks, false>(sg, self_sg, rank);
      return;
    }

    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
      const int base = row * hidden;
      float square_sum = 0.0f;
      for (int col = tid; col < hidden; col += blockDim.x) {
        float val = upcast_s(residual_in[base + col]);
#pragma unroll
        for (int r = 0; r < nranks; ++r) {
          val += upcast_s(ptrs[r][base + col]);
        }
        residual_out[base + col] = downcast_s<T>(val);
        square_sum += val * val;
      }

      const float row_square_sum = block_sum_tid0(square_sum, warp_sums);
      if (tid == 0) {
        row_sums[row] = row_square_sum;
      }
    }

    __musa_barrier_slc();
    __syncthreads_lm();
    if (tid == 0) {
      __threadfence_system_noflush();
    }
    __syncthreads_lm();
    multi_rank_barrier<nranks, false>(sg, self_sg, rank);
    return;
  }

  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int base = row * hidden;
    for (int col = rank + tid * nranks; col < hidden;
         col += blockDim.x * nranks) {
      float val = 0.0f;
#pragma unroll
      for (int r = 0; r < nranks; ++r) {
        val += upcast_s(ptrs[r][base + col]);
      }
      self_tmp[base + col] = downcast_s<T>(val);
    }
  }

  __musa_barrier_slc();
  __syncthreads_lm();
  if (tid == 0) {
    __threadfence_system_noflush();
  }
  publish_barrier<nranks>(sg, self_sg, rank);

  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int base = row * hidden;
    float square_sum = 0.0f;
    for (int col = tid; col < hidden; col += blockDim.x) {
      const int owner = col & (nranks - 1);
      const T* owner_tmp = get_tmp_buf<T>(sg.signals[owner]);
      const float val =
          upcast_s(owner_tmp[base + col]) + upcast_s(residual_in[base + col]);
      residual_out[base + col] = downcast_s<T>(val);
      square_sum += val * val;
    }

    const float row_square_sum = block_sum_tid0(square_sum, warp_sums);
    if (tid == 0) {
      row_sums[row] = row_square_sum;
    }
  }

  __musa_barrier_slc();
  __syncthreads_lm();
  if (tid == 0) {
    __threadfence_system_noflush();
  }
  __syncthreads_lm();
  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <typename T, int nranks, int vlen = 8>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
custom_all_reduce_residual_2shot_sums_kernel(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual_in,
    T* __restrict__ residual_out,
    float* __restrict__ row_sums,
    int rank,
    int rows,
    int hidden,
    int packed_size) {
  constexpr int nranks_sft = (nranks >> 1) - (nranks >> 3);
  constexpr int coalesce_num = 8;
  constexpr int coalesce_sft = 3;
  constexpr int group_stride_sft = nranks_sft + coalesce_sft;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int target_rank = (tid >> coalesce_sft) & (nranks - 1);
  const int group_id = tid >> group_stride_sft;
  const int coalesce_tid = tid & (coalesce_num - 1);
  const int stride = gridDim.x * blockDim.x;
  const int packed_hidden = hidden / vlen;

  using Vec = array_t<T, vlen>;
  __shared__ float warp_sums[(kMaxThreadsPerBlock + 31) / 32];
  int idx_base = blockIdx.x * blockDim.x;
  int idx_in_blk = coalesce_tid + (rank << coalesce_sft) + (group_id << group_stride_sft);

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);

  Vec* target_ptr = reinterpret_cast<Vec*>(const_cast<void*>(data.ptrs[target_rank]));
  Vec* buffer_ptr = get_tmp_buf<Vec>(sg.signals[rank]);
  do {
    const int idx = idx_in_blk + idx_base;
    float acc[vlen] = {0};
    if (idx < packed_size) {
#if SGL_CUSTOM_AR_VECTOR_LOAD
      Vec raw = target_ptr[idx];
      const T* src = reinterpret_cast<const T*>(&raw);
#else
      const T* src = reinterpret_cast<const T*>(&target_ptr[idx]);
#endif
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        acc[i] = upcast_s(src[i]);
      }
    }
    shfl_reduce<T, nranks, vlen>(acc);
    if constexpr (nranks == 8) {
      __shared__ float smem[kMaxThreadsPerBlock << 1];
      if (lane < coalesce_num) {
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          smem[warp * vlen * coalesce_num + coalesce_tid * vlen + i] = acc[i];
        }
      }
      __syncthreads_lm();
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        acc[i] += smem[(warp ^ 1) * vlen * coalesce_num + coalesce_tid * vlen + i];
      }
    }
    if (rank == target_rank && idx < packed_size) {
      Vec res;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        reinterpret_cast<T*>(&res)[i] = downcast_s<T>(acc[i]);
      }
      buffer_ptr[idx] = res;
    }
    idx_base += stride;
  } while (idx_base < packed_size);

  publish_barrier<nranks>(sg, self_sg, rank);

  buffer_ptr = get_tmp_buf<Vec>(sg.signals[target_rank]);
  const Vec* residual = reinterpret_cast<const Vec*>(residual_in);
  Vec* dst = reinterpret_cast<Vec*>(residual_out);
  if constexpr (nranks == 4 || nranks == 8) {
    if ((hidden == 1536 || hidden == 3072 || hidden == 4096) &&
        blockDim.x == 512) {
      const Vec* owner_tmp = get_tmp_buf<Vec>(sg.signals[target_rank]);
      for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const int idx = row * packed_hidden + tid;
        float square_sum = 0.0f;
        if (tid < packed_hidden) {
          const Vec reduced = owner_tmp[idx];
          const Vec residual_packet = residual[idx];
          Vec out;
#pragma unroll
          for (int i = 0; i < vlen; ++i) {
            const float val =
                upcast_s(reinterpret_cast<const T*>(&reduced)[i]) +
                upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]);
            reinterpret_cast<T*>(&out)[i] = downcast_s<T>(val);
            square_sum += val * val;
          }
          dst[idx] = out;
        }
        const float row_square_sum =
            hidden == 4096
                ? block_sum_16warps_tid0(square_sum, warp_sums)
                : (hidden == 3072
                       ? block_sum_12warps_tid0(square_sum, warp_sums)
                       : block_sum_nwarps_tid0<6>(square_sum, warp_sums));
        if (tid == 0) {
          row_sums[row] = row_square_sum;
        }
      }

      multi_rank_barrier<nranks, false>(sg, self_sg, rank);
      return;
    }
  }
  idx_in_blk = coalesce_tid + (target_rank << coalesce_sft) + (group_id << group_stride_sft);
  idx_base = blockIdx.x * blockDim.x;
  do {
    const int idx = idx_in_blk + idx_base;
    float values[vlen] = {0};
    float square_sum = 0.0f;
    if (idx < packed_size) {
#if SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 1 || \
    SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 2
      const Vec8<T> reduced =
          Vec8<T>::load_byp_slc(reinterpret_cast<const T*>(buffer_ptr), idx * vlen);
#else
      const Vec reduced = buffer_ptr[idx];
#endif
#if SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 1 || \
    SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 3
      const Vec8<T> residual_packet = vllm_fused_load_residual<T, true>(residual_in, idx * vlen);
#else
      const Vec residual_packet = residual[idx];
#endif
      Vec out;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
#if SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 1
        const float val =
            upcast_s(reinterpret_cast<const T*>(&reduced)[i]) +
            upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]);
#elif SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 2
        const float val =
            upcast_s(reinterpret_cast<const T*>(&reduced)[i]) +
            upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]);
#elif SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 3
        const float val =
            upcast_s(reinterpret_cast<const T*>(&reduced)[i]) +
            upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]);
#else
        const float val =
            upcast_s(reinterpret_cast<const T*>(&reduced)[i]) +
            upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]);
#endif
        reinterpret_cast<T*>(&out)[i] = downcast_s<T>(val);
        square_sum += val * val;
      }
      dst[idx] = out;
    }
    const float row_square_sum = block_sum_tid0(square_sum, warp_sums);
    if (tid == 0) {
      const int row = idx_base / packed_hidden;
      if (row < rows) {
        row_sums[row] = row_square_sum;
      }
    }
    idx_base += stride;
  } while (idx_base < packed_size);

  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <
    typename T,
    int nranks,
    bool kSharedInvRms,
    bool kCacheWeightStatic,
    int vlen = 8>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
custom_all_reduce_residual_2shot_rmsnorm_row_kernel(
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual_in,
    T* __restrict__ residual_out,
    T* __restrict__ norm_out,
    T* __restrict__ raw_out,
    const T* __restrict__ weight,
    int rank,
    int rows,
    int hidden,
    float eps,
    int packed_size) {
  constexpr int nranks_sft = (nranks >> 1) - (nranks >> 3);
  constexpr int coalesce_num = 8;
  constexpr int coalesce_sft = 3;
  constexpr int group_stride_sft = nranks_sft + coalesce_sft;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int target_rank = (tid >> coalesce_sft) & (nranks - 1);
  const int group_id = tid >> group_stride_sft;
  const int coalesce_tid = tid & (coalesce_num - 1);
  const int stride = gridDim.x * blockDim.x;
  const int packed_hidden = hidden / vlen;
  const float inv_hidden = 1.0f / static_cast<float>(hidden);

  using Vec = array_t<T, vlen>;
  __shared__ float inv_rms;
  extern __shared__ __align__(16) unsigned char row_smem[];
  const size_t weight_bytes =
      kCacheWeightStatic ? static_cast<size_t>(hidden) * sizeof(T) : 0;
  T* cached_weight = reinterpret_cast<T*>(row_smem);
  float* warp_sums =
      reinterpret_cast<float*>(row_smem + align_up(weight_bytes, alignof(float)));
  const Vec* weight_vec = reinterpret_cast<const Vec*>(weight);

  int idx_base = blockIdx.x * blockDim.x;
  int idx_in_blk = coalesce_tid + (rank << coalesce_sft) + (group_id << group_stride_sft);

  if constexpr (kCacheWeightStatic) {
    for (int packed_col = tid; packed_col < packed_hidden; packed_col += blockDim.x) {
      const int col0 = packed_col * vlen;
      *(Vec*)(cached_weight + col0) = weight_vec[packed_col];
    }
    __syncthreads_lm();
  }
  if constexpr (nranks == 4) {
    // Keep every TP4 row launch in the same per-peer generation chain. A
    // shape change must still wait for the preceding launch's scratch reads.
    multi_rank_start_barrier_no_end<nranks>(sg, self_sg, rank);
  } else {
    multi_rank_barrier<nranks, true>(sg, self_sg, rank);
  }
  if (SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_PUSH && rows == 1 && blockIdx.x == 0 &&
      blockDim.x >= packed_hidden) {
    // Push each rank's row into the same slot of every peer's scratch buffer.
    // The next invocation's entry barrier protects these slots from overwrite.
    const int packed_col = tid;
    const Vec *local_input =
        reinterpret_cast<const Vec *>(data_ptr->ptrs[rank]);
    if (packed_col < packed_hidden) {
      const Vec packet = local_input[packed_col];
#pragma unroll
      for (int peer = 0; peer < nranks; ++peer) {
        Vec *peer_scratch = get_tmp_buf<Vec>(sg.signals[peer]);
        peer_scratch[rank * packed_hidden + packed_col] = packet;
      }
    }
    publish_barrier<nranks>(sg, self_sg, rank);

    float values[vlen] = {0.0f};
    float square_sum = 0.0f;
    Vec out;
    if (packed_col < packed_hidden) {
      const Vec *local_scratch = get_tmp_buf<Vec>(self_sg);
#pragma unroll
      for (int peer = 0; peer < nranks; ++peer) {
        const Vec packet = local_scratch[peer * packed_hidden + packed_col];
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          values[i] += upcast_s(reinterpret_cast<const T *>(&packet)[i]);
        }
      }
      const Vec residual_packet =
          reinterpret_cast<const Vec *>(residual_in)[packed_col];
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        values[i] = vllm_fused_norm_value<T>(values[i], upcast_s(reinterpret_cast<const T *>(&residual_packet)[i]), raw_out, packed_col * vlen + i);
        reinterpret_cast<T *>(&out)[i] = downcast_s<T>(values[i]);
        square_sum += values[i] * values[i];
      }
    }
    const float row_square_sum = block_sum(square_sum, warp_sums);
    const float scale = fast_rsqrt(row_square_sum * inv_hidden + eps);
    if (packed_col < packed_hidden) {
      const Vec w = weight_vec[packed_col];
      Vec norm_packet;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        const float gamma = upcast_s(reinterpret_cast<const T *>(&w)[i]);
        reinterpret_cast<T *>(&norm_packet)[i] =
            vllm_fused_apply_weight<T>(values[i] * scale, gamma);
      }
      reinterpret_cast<Vec *>(residual_out)[packed_col] = out;
      reinterpret_cast<Vec *>(norm_out)[packed_col] = norm_packet;
    }
    return;
  }
  if (vlen != packed_t<T>::P::size && (hidden % vlen) == 0 &&
      blockDim.x >= packed_hidden && packed_hidden <= kMaxThreadsPerBlock) {
    const Vec* residual = reinterpret_cast<const Vec*>(residual_in);
    Vec* residual_dst = reinterpret_cast<Vec*>(residual_out);
    Vec* norm_dst = reinterpret_cast<Vec*>(norm_out);
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
      const int packed_col = tid;
      const int packed_idx = row * packed_hidden + packed_col;
      float values[vlen] = {0};
      float square_sum = 0.0f;
      Vec out;
      if (packed_col < packed_hidden) {
#pragma unroll
        for (int r = 0; r < nranks; ++r) {
          const Vec packet =
              reinterpret_cast<const Vec*>(data_ptr->ptrs[r])[packed_idx];
#pragma unroll
          for (int i = 0; i < vlen; ++i) {
            values[i] += upcast_s(reinterpret_cast<const T*>(&packet)[i]);
          }
        }
        const Vec residual_packet = residual[packed_idx];
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          values[i] = vllm_fused_norm_value<T>(values[i], upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]), raw_out, packed_idx * vlen + i);
          reinterpret_cast<T*>(&out)[i] = downcast_s<T>(values[i]);
          square_sum += values[i] * values[i];
        }
      }
      const float row_square_sum = block_sum(square_sum, warp_sums);
      const float scale = fast_rsqrt(row_square_sum * inv_hidden + eps);
      if (packed_col < packed_hidden) {
        const Vec w = kCacheWeightStatic
            ? *(reinterpret_cast<const Vec*>(cached_weight) + packed_col)
            : weight_vec[packed_col];
        Vec norm_packet;
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          const float gamma = upcast_s(reinterpret_cast<const T*>(&w)[i]);
          reinterpret_cast<T*>(&norm_packet)[i] =
              vllm_fused_apply_weight<T>(values[i] * scale, gamma);
        }
        residual_dst[packed_idx] = out;
        norm_dst[packed_idx] = norm_packet;
      }
    }

    row_multi_rank_end_barrier<nranks>(sg, self_sg, rank);
    return;
  }
  if constexpr (nranks == 2) {
    if (hidden == 6144 && (hidden % vlen) == 0 && blockDim.x == 512) {
      const Vec* local = reinterpret_cast<const Vec*>(data_ptr->ptrs[rank]);
      const Vec* peer = reinterpret_cast<const Vec*>(data_ptr->ptrs[rank ^ 1]);
      const Vec* residual = reinterpret_cast<const Vec*>(residual_in);
      Vec* residual_dst = reinterpret_cast<Vec*>(residual_out);
      Vec* norm_dst = reinterpret_cast<Vec*>(norm_out);
      for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const int row_base = row * packed_hidden;
        const int packed_col0 = tid;
        const int packed_col1 = tid + blockDim.x;
        const int packed_idx0 = row_base + packed_col0;
        const int packed_idx1 = row_base + packed_col1;
        float values0[vlen] = {0};
        float values1[vlen] = {0};
        float square_sum = 0.0f;
        Vec out0;
        Vec out1;

        const Vec local0 = local[packed_idx0];
        const Vec peer0 = peer[packed_idx0];
        const Vec residual0 = residual[packed_idx0];
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          float val = upcast_s(reinterpret_cast<const T*>(&local0)[i]);
          val += upcast_s(reinterpret_cast<const T*>(&peer0)[i]);
          val = vllm_fused_norm_value<T>(val, upcast_s(reinterpret_cast<const T*>(&residual0)[i]), raw_out, packed_idx0 * vlen + i);
          values0[i] = val;
          reinterpret_cast<T*>(&out0)[i] = downcast_s<T>(val);
          square_sum += val * val;
        }

        if (packed_col1 < packed_hidden) {
          const Vec local1 = local[packed_idx1];
          const Vec peer1 = peer[packed_idx1];
          const Vec residual1 = residual[packed_idx1];
#pragma unroll
          for (int i = 0; i < vlen; ++i) {
            float val = upcast_s(reinterpret_cast<const T*>(&local1)[i]);
            val += upcast_s(reinterpret_cast<const T*>(&peer1)[i]);
            val = vllm_fused_norm_value<T>(val, upcast_s(reinterpret_cast<const T*>(&residual1)[i]), raw_out, packed_idx1 * vlen + i);
            values1[i] = val;
            reinterpret_cast<T*>(&out1)[i] = downcast_s<T>(val);
            square_sum += val * val;
          }
        }

        const float row_square_sum = block_sum(square_sum, warp_sums);
        const float scale = fast_rsqrt(row_square_sum * inv_hidden + eps);

        const Vec w0 = kCacheWeightStatic
            ? *(reinterpret_cast<const Vec*>(cached_weight) + packed_col0)
            : weight_vec[packed_col0];
        Vec norm0;
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          const float gamma = upcast_s(reinterpret_cast<const T*>(&w0)[i]);
          reinterpret_cast<T*>(&norm0)[i] =
              vllm_fused_apply_weight<T>(values0[i] * scale, gamma);
        }
        residual_dst[packed_idx0] = out0;
        norm_dst[packed_idx0] = norm0;

        if (packed_col1 < packed_hidden) {
          const Vec w1 = kCacheWeightStatic
              ? *(reinterpret_cast<const Vec*>(cached_weight) + packed_col1)
              : weight_vec[packed_col1];
          Vec norm1;
#pragma unroll
          for (int i = 0; i < vlen; ++i) {
            const float gamma = upcast_s(reinterpret_cast<const T*>(&w1)[i]);
            reinterpret_cast<T*>(&norm1)[i] =
                vllm_fused_apply_weight<T>(values1[i] * scale, gamma);
          }
          residual_dst[packed_idx1] = out1;
          norm_dst[packed_idx1] = norm1;
        }
      }

      row_multi_rank_end_barrier<nranks>(sg, self_sg, rank);
      return;
    }
    if ((hidden == 1536 || hidden == 2048 || hidden == 3072 ||
         (hidden == 6144 && rows >= 8192)) &&
        (hidden % vlen) == 0 && blockDim.x == packed_hidden) {
      const Vec* local = reinterpret_cast<const Vec*>(data_ptr->ptrs[rank]);
      const Vec* peer = reinterpret_cast<const Vec*>(data_ptr->ptrs[rank ^ 1]);
      const Vec* residual = reinterpret_cast<const Vec*>(residual_in);
      Vec* residual_dst = reinterpret_cast<Vec*>(residual_out);
      Vec* norm_dst = reinterpret_cast<Vec*>(norm_out);
      for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const int packed_idx = row * packed_hidden + tid;
        const Vec local_packet = local[packed_idx];
        const Vec peer_packet = peer[packed_idx];
        const Vec residual_packet = residual[packed_idx];
        float values[vlen] = {0};
        float square_sum = 0.0f;
        Vec out;
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          float val = upcast_s(reinterpret_cast<const T*>(&local_packet)[i]);
          val += upcast_s(reinterpret_cast<const T*>(&peer_packet)[i]);
          val = vllm_fused_norm_value<T>(val, upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]), raw_out, packed_idx * vlen + i);
          values[i] = val;
          reinterpret_cast<T*>(&out)[i] = downcast_s<T>(val);
          square_sum += val * val;
        }
        residual_dst[packed_idx] = out;

        const float row_square_sum =
            row_group_sum_all_wide(square_sum, warp_sums, packed_hidden);
        const float scale = fast_rsqrt(row_square_sum * inv_hidden + eps);

        const Vec w = kCacheWeightStatic
            ? *(reinterpret_cast<const Vec*>(cached_weight) + tid)
            : weight_vec[tid];
        Vec norm_packet;
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          const float gamma = upcast_s(reinterpret_cast<const T*>(&w)[i]);
          reinterpret_cast<T*>(&norm_packet)[i] =
              vllm_fused_apply_weight<T>(values[i] * scale, gamma);
        }
        norm_dst[packed_idx] = norm_packet;
      }

      row_multi_rank_end_barrier<nranks>(sg, self_sg, rank);
      return;
    }
  }
  Vec* target_ptr =
      reinterpret_cast<Vec*>(const_cast<void*>(data_ptr->ptrs[target_rank]));
  Vec* buffer_ptr = get_tmp_buf<Vec>(sg.signals[rank]);
  do {
    const int idx = idx_in_blk + idx_base;
    float acc[vlen] = {0};
    if (idx < packed_size) {
#if SGL_CUSTOM_AR_VECTOR_LOAD
      Vec raw = target_ptr[idx];
      const T* src = reinterpret_cast<const T*>(&raw);
#else
      const T* src = reinterpret_cast<const T*>(&target_ptr[idx]);
#endif
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        acc[i] = upcast_s(src[i]);
      }
    }
    shfl_reduce<T, nranks, vlen>(acc);
    if constexpr (nranks == 8) {
      __shared__ float smem[kMaxThreadsPerBlock << 1];
      if (lane < coalesce_num) {
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          smem[warp * vlen * coalesce_num + coalesce_tid * vlen + i] = acc[i];
        }
      }
      __syncthreads_lm();
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        acc[i] += smem[(warp ^ 1) * vlen * coalesce_num + coalesce_tid * vlen + i];
      }
    }
    if (rank == target_rank && idx < packed_size) {
      Vec res;
#if SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
      const Vec residual_packet =
          reinterpret_cast<const Vec*>(residual_in)[idx];
#endif
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
#if SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
        const float val = vllm_fused_norm_value<T>(
            acc[i], upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]),
            raw_out, idx * vlen + i);
        reinterpret_cast<T*>(&res)[i] = downcast_s<T>(val);
#else
        reinterpret_cast<T*>(&res)[i] = downcast_s<T>(acc[i]);
#endif
      }
      buffer_ptr[idx] = res;
    }
    idx_base += stride;
  } while (idx_base < packed_size);

  if constexpr (nranks == 4) {
    if ((hidden == 2048 || SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_PEER_PUBLISH) &&
        rows <= 32) {
      publish_barrier_peer_slots<nranks>(sg, self_sg, rank);
    } else {
      publish_barrier<nranks>(sg, self_sg, rank);
    }
  } else {
    publish_barrier<nranks>(sg, self_sg, rank);
  }

  const Vec* residual = reinterpret_cast<const Vec*>(residual_in);
  Vec* residual_dst = reinterpret_cast<Vec*>(residual_out);
  Vec* norm_dst = reinterpret_cast<Vec*>(norm_out);

  if constexpr (nranks == 4 || nranks == 8) {
    // For non-power-of-two hidden sizes whose CTA covers one packed row,
    // reduce each row independently.  The strided generic path performs a
    // block-wide sum over all values assigned to the CTA, which mixes rows
    // once the launch has fewer CTAs than rows.
    if (hidden > 2048 && hidden != 4096 &&
        (hidden != 8192 || rows <= kOneShotMaxToken) &&
        (packed_hidden & 31) == 0 && blockDim.x == packed_hidden) {
      const int packed_col = tid;
      // packed_hidden is a multiple of 32 for this path, so the owner rank
      // depends only on the packed column, not on the row.  Hoist the signal
      // lookup out of the row loop to avoid repeating it for long sequences.
      const int owner_rank = (packed_col >> coalesce_sft) & (nranks - 1);
      const Vec* owner_tmp = get_tmp_buf<Vec>(sg.signals[owner_rank]);
      const Vec w = kCacheWeightStatic
          ? *(reinterpret_cast<const Vec*>(cached_weight) + packed_col)
          : weight_vec[packed_col];
      for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const int packed_idx = row * packed_hidden + packed_col;
#if SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 1 || \
    SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 2
        const Vec8<T> reduced = Vec8<T>::load_byp_slc(
            reinterpret_cast<const T*>(owner_tmp), packed_idx * vlen);
#else
        const Vec reduced = owner_tmp[packed_idx];
#endif
#if SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 1 || \
    SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 3
        const Vec8<T> residual_packet =
            vllm_fused_load_residual<T, true>(residual_in, packed_idx * vlen);
#else
        const Vec residual_packet = residual[packed_idx];
#endif
        if constexpr (nranks == 8) {
          if (rows == 1) {
            row_multi_rank_end_barrier<nranks>(sg, self_sg, rank);
          }
        }
        float values[vlen] = {0.0f};
        float square_sum = 0.0f;
        Vec out;
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          const float val = vllm_fused_norm_value<T>(
              upcast_s(reinterpret_cast<const T*>(&reduced)[i]),
              upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]),
              raw_out, packed_idx * vlen + i);
          values[i] = val;
          reinterpret_cast<T*>(&out)[i] = downcast_s<T>(val);
          square_sum += val * val;
        }
        if constexpr (nranks == 8) {
          if (rows == 1) {
            __syncthreads_lm();
          }
        }
        float scale;
        if constexpr (kSharedInvRms) {
          scale = row_group_rms_scale_shared(
              square_sum, warp_sums, &inv_rms, inv_hidden, eps,
              packed_hidden);
        } else {
          float row_square_sum;
          if (hidden == 2560) {
            row_square_sum = block_sum_nwarps<10>(square_sum, warp_sums);
          } else if (hidden == 3072) {
            row_square_sum = block_sum_12warps(square_sum, warp_sums);
          } else if (hidden == 3584) {
            row_square_sum = block_sum_nwarps<14>(square_sum, warp_sums);
          } else if (hidden == 5120) {
            row_square_sum = block_sum_nwarps<20>(square_sum, warp_sums);
          } else if (hidden == 6144) {
            row_square_sum = block_sum_nwarps<24>(square_sum, warp_sums);
          } else if (hidden == 7168) {
            row_square_sum = block_sum_nwarps<28>(square_sum, warp_sums);
          } else if (hidden == 8192) {
            row_square_sum = block_sum_nwarps<32>(square_sum, warp_sums);
          } else {
            row_square_sum =
                row_group_sum_all_wide(square_sum, warp_sums, packed_hidden);
          }
          if constexpr (kRowWarpInvRms) {
            scale = lane == 0
                ? fast_rsqrt(row_square_sum * inv_hidden + eps)
                : 0.0f;
            scale = __shfl_sync(0xffffffff, scale, 0, 32);
          } else {
            scale = fast_rsqrt(row_square_sum * inv_hidden + eps);
          }
        }
        Vec norm_packet;
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          const float gamma = upcast_s(reinterpret_cast<const T*>(&w)[i]);
          reinterpret_cast<T*>(&norm_packet)[i] =
              vllm_fused_apply_weight<T>(values[i] * scale, gamma);
        }
        residual_dst[packed_idx] = out;
        norm_dst[packed_idx] = norm_packet;
      }
      if constexpr (nranks == 8) {
        if (rows != 1) {
          row_multi_rank_end_barrier<nranks>(sg, self_sg, rank);
        }
      } else {
        row_multi_rank_end_barrier<nranks>(sg, self_sg, rank);
      }
      return;
    }

    if (vlen != packed_t<T>::P::size && (hidden % vlen) == 0 &&
        blockDim.x >= packed_hidden && packed_hidden <= kMaxThreadsPerBlock) {
      const int packed_col = tid;
      for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        float values[vlen] = {0};
        float square_sum = 0.0f;
        Vec out;
        if (tid < packed_hidden) {
          const int packed_idx = row * packed_hidden + packed_col;
          const int idx_target_rank =
              (packed_idx >> coalesce_sft) & (nranks - 1);
          const Vec* owner_tmp = get_tmp_buf<Vec>(sg.signals[idx_target_rank]);
          const Vec reduced = owner_tmp[packed_idx];
#if !SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
          const Vec residual_packet = residual[packed_idx];
#endif
#pragma unroll
          for (int i = 0; i < vlen; ++i) {
            float val = upcast_s(reinterpret_cast<const T*>(&reduced)[i]);
#if !SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
            val = vllm_fused_norm_value<T>(val, upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]), raw_out, packed_idx * vlen + i);
#endif
            values[i] = val;
            reinterpret_cast<T*>(&out)[i] = downcast_s<T>(val);
            square_sum += val * val;
          }
        }
        const float row_square_sum = block_sum(square_sum, warp_sums);
        const float scale = fast_rsqrt(row_square_sum * inv_hidden + eps);
        if (tid < packed_hidden) {
          const int packed_idx = row * packed_hidden + packed_col;
          const Vec w = kCacheWeightStatic
              ? *(reinterpret_cast<const Vec*>(cached_weight) + packed_col)
              : weight_vec[packed_col];
          Vec norm_packet;
#pragma unroll
          for (int i = 0; i < vlen; ++i) {
            const float gamma = upcast_s(reinterpret_cast<const T*>(&w)[i]);
            reinterpret_cast<T*>(&norm_packet)[i] =
                vllm_fused_apply_weight<T>(values[i] * scale, gamma);
          }
          residual_dst[packed_idx] = out;
          norm_dst[packed_idx] = norm_packet;
        }
      }

      row_multi_rank_end_barrier<nranks>(sg, self_sg, rank);
      return;
    }
  }

  if (SGL_CUSTOM_AR_FUSED_RMSNORM_SMALL_ROWS &&
      (hidden == 512 || hidden == 1024 || hidden == 1536 || hidden == 2048) &&
      blockDim.x == packed_hidden) {
    const int packed_col = tid;
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
      float values[vlen] = {0};
      float square_sum = 0.0f;
      const int packed_idx = row * packed_hidden + packed_col;
      const int owner_rank = (packed_idx >> coalesce_sft) & (nranks - 1);
      const Vec* owner_tmp = get_tmp_buf<Vec>(sg.signals[owner_rank]);
      const Vec reduced = owner_tmp[packed_idx];
#if !SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
      const Vec residual_packet = residual[packed_idx];
#endif
      if constexpr (nranks == 8) {
        if (rows == 1) {
          row_multi_rank_end_barrier<nranks>(sg, self_sg, rank);
        }
      }
      Vec out;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        float val = upcast_s(reinterpret_cast<const T*>(&reduced)[i]);
#if !SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
        val = vllm_fused_norm_value<T>(val, upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]), raw_out, packed_idx * vlen + i);
#endif
        values[i] = val;
        reinterpret_cast<T*>(&out)[i] = downcast_s<T>(val);
        square_sum += val * val;
      }
      if constexpr (nranks == 8) {
        if (rows == 1) {
          __syncthreads_lm();
        }
      }
      const float scale =
          row_group_rms_scale_shared(square_sum, warp_sums, &inv_rms, inv_hidden,
                                     eps, packed_hidden);
      const Vec w = weight_vec[packed_col];
      Vec norm_packet;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        const float gamma = upcast_s(reinterpret_cast<const T*>(&w)[i]);
        reinterpret_cast<T*>(&norm_packet)[i] =
            vllm_fused_apply_weight<T>(values[i] * scale, gamma);
      }
      residual_dst[packed_idx] = out;
      norm_dst[packed_idx] = norm_packet;
    }

    if constexpr (nranks == 8) {
      if (rows != 1) {
        row_multi_rank_end_barrier<nranks>(sg, self_sg, rank);
      }
    } else {
      row_multi_rank_end_barrier<nranks>(sg, self_sg, rank);
    }
    return;
  }

  if constexpr (nranks == 4) {
    if (hidden == 2048 && rows <= 32 && blockDim.x >= packed_hidden &&
        (blockDim.x % packed_hidden) == 0) {
      const int rows_per_block = static_cast<int>(blockDim.x) / packed_hidden;
      const int row_slot = tid / packed_hidden;
      const int packed_col = tid - row_slot * packed_hidden;
      const int row = blockIdx.x * rows_per_block + row_slot;
      const int packed_idx = row * packed_hidden + packed_col;
      const int owner_rank = (packed_idx >> coalesce_sft) & (nranks - 1);
      const Vec* owner_tmp = get_tmp_buf<Vec>(sg.signals[owner_rank]);
      const Vec reduced = owner_tmp[packed_idx];
      const Vec residual_packet = residual[packed_idx];
      float values[vlen] = {0.0f};
      float square_sum = 0.0f;
      Vec out;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        const float val = vllm_fused_norm_value<T>(
                          upcast_s(reinterpret_cast<const T*>(&reduced)[i]),
                          upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]),
                          raw_out, packed_idx * vlen + i);
        values[i] = val;
        reinterpret_cast<T*>(&out)[i] = downcast_s<T>(val);
        square_sum += val * val;
      }
      residual_dst[packed_idx] = out;
      const float scale =
          rows == 32 && blockDim.x == packed_hidden
              ? row_group_rms_scale_shared(
                    square_sum, warp_sums, &inv_rms, inv_hidden, eps,
                    packed_hidden)
              : row_group_rms_scale_all(
                    square_sum, warp_sums, &inv_rms, inv_hidden, eps,
                    packed_hidden);
      const Vec w = weight_vec[packed_col];
      Vec norm_packet;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        const float gamma = upcast_s(reinterpret_cast<const T*>(&w)[i]);
        reinterpret_cast<T*>(&norm_packet)[i] =
            vllm_fused_apply_weight<T>(values[i] * scale, gamma);
      }
      norm_dst[packed_idx] = norm_packet;
      return;
    }
  }

  if ((hidden == 512 || hidden == 1024 || hidden == 2048) &&
      blockDim.x >= packed_hidden && (blockDim.x % packed_hidden) == 0) {
    const int rows_per_block = static_cast<int>(blockDim.x) / packed_hidden;
    const int row_slot = tid / packed_hidden;
    const int packed_col = tid - row_slot * packed_hidden;
    for (int row_base = blockIdx.x * rows_per_block; row_base < rows;
         row_base += gridDim.x * rows_per_block) {
      const int row = row_base + row_slot;
      float values[vlen] = {0};
      float square_sum = 0.0f;
      Vec out;
      if (row < rows) {
        const int packed_idx = row * packed_hidden + packed_col;
        const int owner_rank = (packed_idx >> coalesce_sft) & (nranks - 1);
        const Vec* owner_tmp = get_tmp_buf<Vec>(sg.signals[owner_rank]);
        const Vec reduced = owner_tmp[packed_idx];
#if !SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
        const Vec residual_packet = residual[packed_idx];
#endif
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          float val = upcast_s(reinterpret_cast<const T*>(&reduced)[i]);
#if !SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
          val = vllm_fused_norm_value<T>(val, upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]), raw_out, packed_idx * vlen + i);
#endif
          values[i] = val;
          reinterpret_cast<T*>(&out)[i] = downcast_s<T>(val);
          square_sum += val * val;
        }
      }
      const float row_square_sum =
          row_group_sum_all(square_sum, warp_sums, packed_hidden);
      const float scale = fast_rsqrt(row_square_sum * inv_hidden + eps);
      if (row < rows) {
        const int packed_idx = row * packed_hidden + packed_col;
        const Vec w = weight_vec[packed_col];
        Vec norm_packet;
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          const float gamma = upcast_s(reinterpret_cast<const T*>(&w)[i]);
          reinterpret_cast<T*>(&norm_packet)[i] =
              vllm_fused_apply_weight<T>(values[i] * scale, gamma);
        }
        residual_dst[packed_idx] = out;
        norm_dst[packed_idx] = norm_packet;
      }
    }

    row_multi_rank_end_barrier<nranks>(sg, self_sg, rank);
    return;
  }

  if (SGL_CUSTOM_AR_FUSED_RMSNORM_SMALL_ROWS &&
      hidden == 4096 && blockDim.x == packed_hidden) {
      const Vec* owner_tmp = get_tmp_buf<Vec>(sg.signals[target_rank]);
      for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const int packed_idx = row * packed_hidden + tid;
        float values[vlen] = {0};
        float square_sum = 0.0f;
        const Vec reduced = owner_tmp[packed_idx];
#if !SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
        const Vec residual_packet = residual[packed_idx];
#endif
        if constexpr (nranks == 8) {
          if (rows == 1) {
            row_multi_rank_end_barrier<nranks>(sg, self_sg, rank);
          }
        }
        Vec out;
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          float val = upcast_s(reinterpret_cast<const T*>(&reduced)[i]);
#if !SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
          val = vllm_fused_norm_value<T>(val, upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]), raw_out, packed_idx * vlen + i);
#endif
          values[i] = val;
          reinterpret_cast<T*>(&out)[i] = downcast_s<T>(val);
          square_sum += val * val;
        }
        if constexpr (nranks == 8) {
          if (rows == 1) {
            __syncthreads_lm();
          }
        }
        const float row_square_sum =
            hidden == 3072 ? block_sum_12warps(square_sum, warp_sums)
                           : block_sum(square_sum, warp_sums);
        const float scale = fast_rsqrt(row_square_sum * inv_hidden + eps);

        Vec norm_packet;
        const Vec w = kCacheWeightStatic
            ? *(reinterpret_cast<const Vec*>(cached_weight) + tid)
            : weight_vec[tid];
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          const float gamma = upcast_s(reinterpret_cast<const T*>(&w)[i]);
          reinterpret_cast<T*>(&norm_packet)[i] =
              vllm_fused_apply_weight<T>(values[i] * scale, gamma);
        }
        residual_dst[packed_idx] = out;
        norm_dst[packed_idx] = norm_packet;
      }

      if constexpr (nranks == 8) {
        if (rows != 1) {
          row_multi_rank_end_barrier<nranks>(sg, self_sg, rank);
        }
      } else {
        row_multi_rank_end_barrier<nranks>(sg, self_sg, rank);
      }
      return;
  }

  if constexpr (nranks == 4 || nranks == 8) {
    if ((hidden == 1536 || hidden == 3072 || hidden == 4096) &&
        blockDim.x == 512) {
      const Vec* owner_tmp = get_tmp_buf<Vec>(sg.signals[target_rank]);
      for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const int packed_idx = row * packed_hidden + tid;
        float values[vlen] = {0};
        float square_sum = 0.0f;
        if (tid < packed_hidden) {
          const Vec reduced = owner_tmp[packed_idx];
#if !SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
          const Vec residual_packet = residual[packed_idx];
#endif
          Vec out;
#pragma unroll
          for (int i = 0; i < vlen; ++i) {
            float val = upcast_s(reinterpret_cast<const T*>(&reduced)[i]);
#if !SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
            val = vllm_fused_norm_value<T>(val, upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]), raw_out, packed_idx * vlen + i);
#endif
            values[i] = val;
            reinterpret_cast<T*>(&out)[i] = downcast_s<T>(val);
            square_sum += val * val;
          }
          residual_dst[packed_idx] = out;
        }
        if constexpr (nranks == 8) {
          if (rows == 1) {
            row_multi_rank_end_barrier<nranks>(sg, self_sg, rank);
            __syncthreads_lm();
          }
        }
        const float row_square_sum =
            hidden == 4096
                ? block_sum(square_sum, warp_sums)
                : (hidden == 3072
                       ? block_sum_12warps(square_sum, warp_sums)
                       : block_sum_nwarps<6>(square_sum, warp_sums));
        const float scale = fast_rsqrt(row_square_sum * inv_hidden + eps);

        if (tid < packed_hidden) {
          Vec norm_packet;
          const Vec w = kCacheWeightStatic
              ? *(reinterpret_cast<const Vec*>(cached_weight) + tid)
              : weight_vec[tid];
#pragma unroll
          for (int i = 0; i < vlen; ++i) {
            const float gamma = upcast_s(reinterpret_cast<const T*>(&w)[i]);
            reinterpret_cast<T*>(&norm_packet)[i] =
                vllm_fused_apply_weight<T>(values[i] * scale, gamma);
          }
          norm_dst[packed_idx] = norm_packet;
        }
      }

      if constexpr (nranks == 8) {
        if (rows != 1) {
          row_multi_rank_end_barrier<nranks>(sg, self_sg, rank);
        }
      } else {
        row_multi_rank_end_barrier<nranks>(sg, self_sg, rank);
      }
      return;
    }
  }
  buffer_ptr = get_tmp_buf<Vec>(sg.signals[target_rank]);
  idx_in_blk = coalesce_tid + (target_rank << coalesce_sft) + (group_id << group_stride_sft);
  idx_base = blockIdx.x * blockDim.x;
  do {
    const int idx = idx_in_blk + idx_base;
    float values[vlen] = {0};
    float square_sum = 0.0f;
    if (idx < packed_size) {
#if SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 1 || \
    SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 2
      const Vec8<T> reduced =
          Vec8<T>::load_byp_slc(reinterpret_cast<const T*>(buffer_ptr), idx * vlen);
#else
      const Vec reduced = buffer_ptr[idx];
#endif
#if !SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
#if SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 1 || \
    SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 3
      const Vec8<T> residual_packet = vllm_fused_load_residual<T, true>(residual_in, idx * vlen);
#else
      const Vec residual_packet = residual[idx];
#endif
#endif
      Vec out;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
#if SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
#if SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 1 || \
    SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 2
        float val = upcast_s(reinterpret_cast<const T*>(&reduced)[i]);
#else
        float val = upcast_s(reinterpret_cast<const T*>(&reduced)[i]);
#endif
#else
#if SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 1
        float val = upcast_s(reinterpret_cast<const T*>(&reduced)[i]);
        val = vllm_fused_norm_value<T>(val, upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]), raw_out, idx * vlen + i);
#elif SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 2
        float val = upcast_s(reinterpret_cast<const T*>(&reduced)[i]);
        val = vllm_fused_norm_value<T>(val, upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]), raw_out, idx * vlen + i);
#elif SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 3
        float val = upcast_s(reinterpret_cast<const T*>(&reduced)[i]);
        val = vllm_fused_norm_value<T>(val, upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]), raw_out, idx * vlen + i);
#else
        float val = upcast_s(reinterpret_cast<const T*>(&reduced)[i]);
        val = vllm_fused_norm_value<T>(val, upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]), raw_out, idx * vlen + i);
#endif
#endif
        values[i] = val;
        reinterpret_cast<T*>(&out)[i] = downcast_s<T>(val);
        square_sum += val * val;
      }
      residual_dst[idx] = out;
    }
    const float row_square_sum = block_sum(square_sum, warp_sums);
    if constexpr (kSharedInvRms) {
      if (tid == 0) {
        inv_rms = fast_rsqrt(row_square_sum * inv_hidden + eps);
      }
      __syncthreads_lm();
    }
    if (idx < packed_size) {
      const int packed_col = idx % packed_hidden;
      const Vec w = kCacheWeightStatic
          ? *(reinterpret_cast<const Vec*>(cached_weight) + packed_col)
          : weight_vec[packed_col];
      Vec norm_packet;
      float scale;
      if constexpr (kSharedInvRms) {
        scale = inv_rms;
      } else if constexpr (kRowWarpInvRms) {
        scale = lane == 0 ? fast_rsqrt(row_square_sum * inv_hidden + eps) : 0.0f;
        scale = __shfl_sync(0xffffffff, scale, 0, 32);
      } else {
        scale = fast_rsqrt(row_square_sum * inv_hidden + eps);
      }
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        const float gamma = upcast_s(reinterpret_cast<const T*>(&w)[i]);
        reinterpret_cast<T*>(&norm_packet)[i] =
            vllm_fused_apply_weight<T>(values[i] * scale, gamma);
      }
      norm_dst[idx] = norm_packet;
    }
    idx_base += stride;
  } while (idx_base < packed_size);

  row_multi_rank_end_barrier<nranks>(sg, self_sg, rank);
}

template <typename T, int nranks>
void launch_fused_ar_rmsnorm(
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* residual_in,
    T* residual_out,
    T* norm_out,
    T* raw_out,
    const T* weight,
    int rank,
    int rows,
    int hidden,
    float eps,
    musaStream_t stream) {
  constexpr int pack = packed_t<T>::P::size;
  const int threads = std::min(kDefaultThreads, kMaxThreadsPerBlock);
  const int block_limit =
      hidden == 4096
          ? (nranks == 2 && rows >= 1024
                 ? (rows >= 8192 ? 42 : (rows >= 4096 ? 44 : 42))
                 : kH4096BlockLimit)
          : (hidden == 8192 ? kH8192BlockLimit : kDefaultBlockLimit);
  const int blocks = std::min(kLaunchMaxBlocks, std::min(block_limit, rows));
  if (blocks <= 0) {
    return;
  }
  const size_t smem_bytes =
      align_up(
          align_up(
              hidden <= kCacheHiddenLimit
                  ? static_cast<size_t>(hidden) * sizeof(float)
                  : (hidden != 4096 && hidden <= kTypedCacheHiddenLimit
                         ? static_cast<size_t>(hidden) * sizeof(T)
                         : 0),
              alignof(T)) +
              (rmsnorm_cache_weight_hidden(hidden, rows)
                   ? static_cast<size_t>(hidden) * sizeof(T)
                   : 0),
          alignof(float)) +
      static_cast<size_t>((threads + 31) / 32) * sizeof(float);
  const int64_t packed_size64 =
      (static_cast<int64_t>(rows) * static_cast<int64_t>(hidden)) / pack;
#if SGL_CUSTOM_AR_FUSED_RMSNORM_TOKEN_2STAGE
  if (hidden % pack == 0 && rows >= 128) {
    if (packed_size64 <= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      custom_all_reduce_residual_rmsnorm_2stage_kernel<T, nranks>
          <<<blocks, threads, smem_bytes, stream>>>(
              data_ptr, sg, self_sg, residual_in, residual_out, norm_out, raw_out, weight,
              rank, rows, hidden, eps, static_cast<int>(packed_size64));
      return;
    }
  }
#endif
  custom_all_reduce_residual_rmsnorm_kernel<T, nranks>
      <<<blocks, threads, smem_bytes, stream>>>(
          data_ptr, sg, self_sg, residual_in, residual_out, norm_out, raw_out, weight,
          rank, rows, hidden, eps);
}

template <typename T, int nranks>
int select_residual_2shot_block_limit(int packed_size) {
  int limit = kDefaultBlockLimit;
#if SGL_CUSTOM_AR_DYNAMIC_BLOCKS
  if constexpr (nranks == 4) {
    if (packed_size >= 4 * 1024 * 1024) {
      limit = 40;
    }
  } else if constexpr (nranks == 8) {
    limit = 60;
  }
#endif
  return std::min(limit, kLaunchMaxBlocks);
}

template <typename T, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
custom_all_reduce_residual_scalar_kernel(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual_in,
    T* __restrict__ residual_out,
    int rank,
    int size) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);

  for (int idx = tid; idx < size; idx += stride) {
    float ar_sum = 0.0f;
#pragma unroll
    for (int r = 0; r < nranks; ++r) {
      const auto* peer = reinterpret_cast<const T*>(data.ptrs[r]);
      ar_sum += to_float(peer[idx]);
    }
    residual_out[idx] = from_float<T>(ar_sum + to_float(residual_in[idx]));
  }

  __musa_barrier_slc();
  __syncthreads_lm();
  if (threadIdx.x == 0) {
    __threadfence_system_noflush();
  }
  __syncthreads_lm();
  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <typename T, int nranks>
void launch_residual_ar(
    RankData data,
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* residual_in,
    T* residual_out,
    int rank,
    int size,
    musaStream_t stream) {
  constexpr int pack = packed_t<T>::P::size;
  const int threads = std::min(kDefaultThreads, kMaxThreadsPerBlock);
  if (size % pack != 0) {
    const int blocks = std::min(kLaunchMaxBlocks, (size + threads - 1) / threads);
    if (blocks <= 0) {
      return;
    }
    custom_all_reduce_residual_scalar_kernel<T, nranks>
        <<<blocks, threads, 0, stream>>>(
            data, sg, self_sg, residual_in, residual_out, rank, size);
    return;
  }
  const int packed_size = size / pack;
  if constexpr ((nranks == 4 || nranks == 8) && !std::is_same<T, float>::value) {
    const int block_limit = select_residual_2shot_block_limit<T, nranks>(packed_size);
    const int blocks = std::min(block_limit, (packed_size + threads - 1) / threads);
    if (blocks <= 0) {
      return;
    }
    custom_all_reduce_residual_2shot_kernel<T, nranks>
        <<<blocks, threads, 0, stream>>>(
            data, data_ptr, sg, self_sg, residual_in, residual_out, rank,
            packed_size);
  } else {
    const int blocks = std::min(kLaunchMaxBlocks, (packed_size + threads - 1) / threads);
    if (blocks <= 0) {
      return;
    }
    custom_all_reduce_residual_kernel<T, nranks>
        <<<blocks, threads, 0, stream>>>(
            data, sg, self_sg, residual_in, residual_out, rank, size);
  }
}

template <typename T, int nranks>
bool launch_residual_ar_sums(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    const T* residual_in,
    T* residual_out,
    float* row_sums,
    int rank,
    int rows,
    int hidden,
    musaStream_t stream) {
  constexpr int pack = packed_t<T>::P::size;
  const int packed_hidden = hidden / pack;
  const int threads =
      nranks == 2 ? std::min(packed_hidden, kMaxThreadsPerBlock)
                  : (packed_hidden <= kMaxThreadsPerBlock ? packed_hidden : 0);
  if constexpr (!((nranks == 2 || nranks == 4 || nranks == 8) && !std::is_same<T, float>::value)) {
    return false;
  }
  if ((hidden % pack) != 0) {
    const int scalar_threads = std::min(hidden, kMaxThreadsPerBlock);
    const int blocks = std::min(kLaunchMaxBlocks, std::min(kDefaultBlockLimit, rows));
    if (scalar_threads <= 0 || blocks <= 0) {
      return true;
    }
    custom_all_reduce_residual_sums_scalar_kernel<T, nranks>
        <<<blocks, scalar_threads, 0, stream>>>(
            data, sg, self_sg, residual_in, residual_out, row_sums, rank, rows,
            hidden);
    return true;
  }
  if constexpr (nranks == 4 || nranks == 8) {
    if (packed_hidden > kMaxThreadsPerBlock) {
      return false;
    }
  }
  if (threads <= 0 || threads > kMaxThreadsPerBlock) {
    return false;
  }
  const int64_t packed_size64 =
      (static_cast<int64_t>(rows) * static_cast<int64_t>(hidden)) / pack;
  if (packed_size64 > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
    return false;
  }
  const int packed_size = static_cast<int>(packed_size64);
  if constexpr (nranks == 2) {
    const int blocks = std::min(kLaunchMaxBlocks, std::min(kDefaultBlockLimit, rows));
    if (blocks <= 0) {
      return true;
    }
    custom_all_reduce_residual_sums_direct_kernel<T, 2>
        <<<blocks, threads, 0, stream>>>(
            data, sg, self_sg, residual_in, residual_out, row_sums, rank, rows,
            hidden);
    return true;
  }
  TVM_FFI_ICHECK_EQ(threads, packed_hidden);
  const int block_limit = select_residual_2shot_block_limit<T, nranks>(packed_size);
  const int blocks = std::min(block_limit, (packed_size + threads - 1) / threads);
  if (blocks <= 0) {
    return true;
  }
  custom_all_reduce_residual_2shot_sums_kernel<T, nranks>
      <<<blocks, threads, 0, stream>>>(
          data, sg, self_sg, residual_in, residual_out, row_sums, rank, rows,
          hidden, packed_size);
  return true;
}

template <typename T, int nranks, int vlen>
bool launch_residual_ar_rmsnorm_row_vlen(
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* residual_in,
    T* residual_out,
    T* norm_out,
    T* raw_out,
    const T* weight,
    int rank,
    int rows,
    int hidden,
    float eps,
    musaStream_t stream) {
  if constexpr (!((nranks == 2 || nranks == 4 || nranks == 8) && !std::is_same<T, float>::value)) {
    return false;
  }
  constexpr int pack = vlen;
  if constexpr (vlen != packed_t<T>::P::size) {
    return false;
  }
  const int packed_hidden = hidden / pack;
  const bool use_direct_512 =
      hidden == 4096 || (nranks == 2 && hidden == 6144) ||
       ((nranks == 4 || nranks == 8) &&
       (hidden == 1536 || hidden == 3072) &&
       !(hidden == 1536 && rows <= 128));
  const int threads =
      (SGL_CUSTOM_AR_FUSED_RMSNORM_SMALL_ROWS &&
         (hidden == 512 || hidden == 1024 || hidden == 2048))
          ? packed_hidden
      : (nranks == 4 && hidden == 2048 && rows <= 32 && (rows & 1))
          ? packed_hidden
      : use_direct_512
          ? 512
      : (hidden == 512 || hidden == 1024 || hidden == 2048)
          ? 512
      : (hidden <= 2048 && packed_hidden <= kMaxThreadsPerBlock)
          ? packed_hidden
          : (packed_hidden <= kMaxThreadsPerBlock ? packed_hidden : 0);
  if (threads <= 0 || (hidden % pack) != 0) {
    return false;
  }
  if (hidden <= 2048 && !use_direct_512) {
    TVM_FFI_ICHECK_EQ(threads % packed_hidden, 0);
  } else if (use_direct_512) {
    TVM_FFI_ICHECK_EQ(threads, 512);
  } else {
    TVM_FFI_ICHECK_EQ(threads, packed_hidden);
  }
  const int64_t packed_size64 =
      (static_cast<int64_t>(rows) * static_cast<int64_t>(hidden)) / pack;
  if (packed_size64 > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
    return false;
  }
  const int packed_size = static_cast<int>(packed_size64);
  // The fused row kernel has substantially more work per CTA than the pure
  // residual all-reduce kernel for the TP8, non-power-of-two large-row path.
  // Keep the historical dynamic limit for TP2/TP4 and short rows.
  bool use_dispatch_limit =
      nranks == 8 && rows >= 1024 && hidden >= 1024 &&
      (hidden & (hidden - 1)) != 0;
  const int block_limit =
      use_dispatch_limit ? std::min(kDefaultBlockLimit, kLaunchMaxBlocks)
                         : select_residual_2shot_block_limit<T, nranks>(packed_size);
  // A one-CTA-per-row launch is only valid when the configured launch cap can
  // cover every row.  Otherwise the one-block kernel would silently omit the
  // tail rows (and would ignore SGL_CUSTOM_AR_LAUNCH_MAX_BLOCKS).
  const bool one_block_per_row =
      threads == packed_hidden && rows <= kOneShotMaxToken &&
      rows <= block_limit && rows <= kLaunchMaxBlocks;
  const int blocks = one_block_per_row
      ? rows
      : std::min(block_limit, (packed_size + threads - 1) / threads);
  if (blocks <= 0) {
    return true;
  }
  const bool cache_weight =
      SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WEIGHT_CACHE &&
      rows >= SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WEIGHT_CACHE_MIN_ROWS &&
      hidden <= kWeightCacheHiddenLimit;
  const size_t smem_bytes =
      align_up(cache_weight ? static_cast<size_t>(hidden) * sizeof(T) : 0, alignof(float)) +
      static_cast<size_t>((threads + 31) / 32) * sizeof(float);
  if (cache_weight) {
    custom_all_reduce_residual_2shot_rmsnorm_row_kernel<T, nranks, true, true, vlen>
        <<<blocks, threads, smem_bytes, stream>>>(
            data_ptr, sg, self_sg, residual_in, residual_out, norm_out, raw_out, weight,
            rank, rows, hidden, eps, packed_size);
  } else if constexpr (kRowSharedInvRms) {
    custom_all_reduce_residual_2shot_rmsnorm_row_kernel<T, nranks, true, false, vlen>
        <<<blocks, threads, smem_bytes, stream>>>(
            data_ptr, sg, self_sg, residual_in, residual_out, norm_out, raw_out, weight,
            rank, rows, hidden, eps, packed_size);
  } else {
    custom_all_reduce_residual_2shot_rmsnorm_row_kernel<T, nranks, false, false, vlen>
        <<<blocks, threads, smem_bytes, stream>>>(
            data_ptr, sg, self_sg, residual_in, residual_out, norm_out, raw_out, weight,
            rank, rows, hidden, eps, packed_size);
  }
  return true;
}

template <typename T, int nranks>
bool launch_residual_ar_rmsnorm_row(
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* residual_in,
    T* residual_out,
    T* norm_out,
    T* raw_out,
    const T* weight,
    int rank,
    int rows,
    int hidden,
    float eps,
    musaStream_t stream) {
  if ((hidden % 8) == 0) {
    return launch_residual_ar_rmsnorm_row_vlen<T, nranks, 8>(
        data_ptr, sg, self_sg, residual_in, residual_out, norm_out, raw_out, weight, rank,
        rows, hidden, eps, stream);
  }
  return false;
}

template <typename T>
void dispatch_world_size(
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* residual_in,
    T* residual_out,
    T* norm_out,
    T* raw_out,
    const T* weight,
    int rank,
    int world_size,
    int rows,
    int hidden,
    float eps,
    musaStream_t stream) {
  switch (world_size) {
    case 2:
      launch_fused_ar_rmsnorm<T, 2>(
          data_ptr, sg, self_sg, residual_in, residual_out, norm_out, raw_out,
          weight, rank, rows, hidden, eps, stream);
      break;
    case 4:
      launch_fused_ar_rmsnorm<T, 4>(
          data_ptr, sg, self_sg, residual_in, residual_out, norm_out, raw_out,
          weight, rank, rows, hidden, eps, stream);
      break;
    case 6:
      launch_fused_ar_rmsnorm<T, 6>(
          data_ptr, sg, self_sg, residual_in, residual_out, norm_out, raw_out,
          weight, rank, rows, hidden, eps, stream);
      break;
    case 8:
      launch_fused_ar_rmsnorm<T, 8>(
          data_ptr, sg, self_sg, residual_in, residual_out, norm_out, raw_out,
          weight, rank, rows, hidden, eps, stream);
      break;
    default:
      TVM_FFI_THROW(ValueError) << "world_size must be one of 2/4/6/8";
  }
}

template <typename T>
void dispatch_residual_world_size(
    RankData data,
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* residual_in,
    T* residual_out,
    int rank,
    int world_size,
    int size,
    musaStream_t stream) {
  switch (world_size) {
    case 2:
      launch_residual_ar<T, 2>(
          data, data_ptr, sg, self_sg, residual_in, residual_out, rank, size,
          stream);
      break;
    case 4:
      launch_residual_ar<T, 4>(
          data, data_ptr, sg, self_sg, residual_in, residual_out, rank, size,
          stream);
      break;
    case 6:
      launch_residual_ar<T, 6>(
          data, data_ptr, sg, self_sg, residual_in, residual_out, rank, size,
          stream);
      break;
    case 8:
      launch_residual_ar<T, 8>(
          data, data_ptr, sg, self_sg, residual_in, residual_out, rank, size,
          stream);
      break;
    default:
      TVM_FFI_THROW(ValueError) << "world_size must be one of 2/4/6/8";
  }
}

template <typename T>
bool dispatch_residual_sums_world_size(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    const T* residual_in,
    T* residual_out,
    float* row_sums,
    int rank,
    int world_size,
    int rows,
    int hidden,
    musaStream_t stream) {
  switch (world_size) {
    case 2:
      return launch_residual_ar_sums<T, 2>(
          data, sg, self_sg, residual_in, residual_out, row_sums, rank, rows,
          hidden, stream);
    case 4:
      return launch_residual_ar_sums<T, 4>(
          data, sg, self_sg, residual_in, residual_out, row_sums, rank, rows,
          hidden, stream);
    case 8:
      return launch_residual_ar_sums<T, 8>(
          data, sg, self_sg, residual_in, residual_out, row_sums, rank, rows,
          hidden, stream);
    default:
      return false;
  }
}

template <typename T>
bool dispatch_residual_rmsnorm_row_world_size(
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* residual_in,
    T* residual_out,
    T* norm_out,
    T* raw_out,
    const T* weight,
    int rank,
    int world_size,
    int rows,
    int hidden,
    float eps,
    musaStream_t stream) {
  switch (world_size) {
    case 2:
      return launch_residual_ar_rmsnorm_row<T, 2>(
          data_ptr, sg, self_sg, residual_in, residual_out, norm_out, raw_out, weight, rank,
          rows, hidden, eps, stream);
    case 4:
      return launch_residual_ar_rmsnorm_row<T, 4>(
          data_ptr, sg, self_sg, residual_in, residual_out, norm_out, raw_out, weight, rank,
          rows, hidden, eps, stream);
    case 8:
      return launch_residual_ar_rmsnorm_row<T, 8>(
          data_ptr, sg, self_sg, residual_in, residual_out, norm_out, raw_out, weight, rank,
          rows, hidden, eps, stream);
    default:
      return false;
  }
}

}  // namespace

void vllm_musa_custom_ar_v2_fused_allreduce_rmsnorm_unregistered(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView inp,
    ffi::TensorView norm_out,
    ffi::TensorView residual_in,
    ffi::TensorView residual_out,
    ffi::TensorView raw_out,
    ffi::TensorView weight,
    int64_t self_signal_ptr,
    int64_t self_buffer_ptr,
    int64_t max_size_bytes,
    int64_t rank,
    int64_t world_size,
    float eps) {
  CHECK_MUSA_CONTIGUOUS(inp);
  CHECK_MUSA_CONTIGUOUS(norm_out);
  CHECK_MUSA_CONTIGUOUS(residual_in);
  CHECK_MUSA_CONTIGUOUS(residual_out);
  CHECK_MUSA_CONTIGUOUS(weight);
#if VLLM_MUSA_FUSED_WRITE_RAW
  CHECK_MUSA_CONTIGUOUS(raw_out);
  CHECK_CONTIGUOUS_2D(raw_out);
  TVM_FFI_ICHECK_EQ(raw_out.device().device_type, norm_out.device().device_type);
  TVM_FFI_ICHECK_EQ(raw_out.size(0), norm_out.size(0));
  TVM_FFI_ICHECK_EQ(raw_out.size(1), norm_out.size(1));
  TVM_FFI_ICHECK(dtype_equal(raw_out.dtype(), norm_out.dtype()));
  TVM_FFI_ICHECK_EQ(raw_out.device().device_id, norm_out.device().device_id);
#endif
  CHECK_CONTIGUOUS_2D(inp);
  CHECK_CONTIGUOUS_2D(norm_out);
  CHECK_CONTIGUOUS_2D(residual_in);
  CHECK_CONTIGUOUS_2D(residual_out);
  TVM_FFI_ICHECK_EQ(weight.ndim(), 1);
  TVM_FFI_ICHECK_EQ(inp.size(0), norm_out.size(0));
  TVM_FFI_ICHECK_EQ(inp.size(1), norm_out.size(1));
  TVM_FFI_ICHECK_EQ(inp.size(0), residual_in.size(0));
  TVM_FFI_ICHECK_EQ(inp.size(1), residual_in.size(1));
  TVM_FFI_ICHECK_EQ(inp.size(0), residual_out.size(0));
  TVM_FFI_ICHECK_EQ(inp.size(1), residual_out.size(1));
  TVM_FFI_ICHECK_EQ(inp.size(1), weight.size(0));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), norm_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), residual_in.dtype()));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), residual_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), weight.dtype()));
  TVM_FFI_ICHECK_EQ(rank_data.device().device_type, inp.device().device_type);
  TVM_FFI_ICHECK_EQ(rank_data.device().device_id, inp.device().device_id);
  TVM_FFI_ICHECK(rank_data.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(rank_data.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(rank_data.size(0), kMaxRanks);
  TVM_FFI_ICHECK_EQ(signal_ptrs_cpu.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(signal_ptrs_cpu.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(signal_ptrs_cpu.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(signal_ptrs_cpu.size(0), world_size);
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size);

  RankSignals sg{};
  const auto* sig_ptrs = static_cast<const int64_t*>(signal_ptrs_cpu.data_ptr());
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(sig_ptrs[i]);
  }

  const auto* device_data_ptr =
      reinterpret_cast<const RankData*>(rank_data.data_ptr());

  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(inp.device());
  const int64_t numel64 = tensor_numel(inp);
  TVM_FFI_ICHECK_LE(numel64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  const int64_t nbytes = numel64 * ((static_cast<int64_t>(inp.dtype().bits) * inp.dtype().lanes + 7) / 8);
  TVM_FFI_ICHECK_LE(nbytes, max_size_bytes);

  const int rows = static_cast<int>(inp.size(0));
  const int hidden = static_cast<int>(inp.size(1));
  const musaError_t copy_err = musaMemcpyAsync(
      reinterpret_cast<void*>(self_buffer_ptr), inp.data_ptr(),
      static_cast<size_t>(nbytes), musaMemcpyDeviceToDevice, stream);
  TVM_FFI_ICHECK_EQ(copy_err, musaSuccess)
      << "MUSA custom AR fused RMSNorm copy failed: "
      << musaGetErrorString(copy_err);

  if (dtype_equal(inp.dtype(), dl_float16)) {
    dispatch_world_size(
        device_data_ptr, sg, self_sg,
        static_cast<const half*>(residual_in.data_ptr()),
        static_cast<half*>(residual_out.data_ptr()),
        static_cast<half*>(norm_out.data_ptr()),
        static_cast<half*>(raw_out.data_ptr()),
        static_cast<const half*>(weight.data_ptr()),
        static_cast<int>(rank), static_cast<int>(world_size),
        rows, hidden, eps, stream);
  } else if (dtype_equal(inp.dtype(), dl_bfloat16)) {
    dispatch_world_size(
        device_data_ptr, sg, self_sg,
        static_cast<const __mt_bfloat16*>(residual_in.data_ptr()),
        static_cast<__mt_bfloat16*>(residual_out.data_ptr()),
        static_cast<__mt_bfloat16*>(norm_out.data_ptr()),
        static_cast<__mt_bfloat16*>(raw_out.data_ptr()),
        static_cast<const __mt_bfloat16*>(weight.data_ptr()),
        static_cast<int>(rank), static_cast<int>(world_size),
        rows, hidden, eps, stream);
  } else {
    TVM_FFI_THROW(ValueError) << "custom AR fused RMSNorm only supports fp16/bf16";
  }

  // Do not query the sticky error state here: MUSA forbids that API while a
  // graph is being captured. Synchronization after replay surfaces failures.
}

void vllm_musa_custom_ar_v2_fused_allreduce_rmsnorm_registered(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView residual_in,
    ffi::TensorView residual_out,
    ffi::TensorView norm_out,
    ffi::TensorView raw_out,
    ffi::TensorView weight,
    int64_t self_signal_ptr,
    int64_t rank,
    int64_t world_size,
    float eps) {
  CHECK_MUSA_CONTIGUOUS(residual_in);
  CHECK_MUSA_CONTIGUOUS(residual_out);
  CHECK_MUSA_CONTIGUOUS(norm_out);
  CHECK_MUSA_CONTIGUOUS(weight);
#if VLLM_MUSA_FUSED_WRITE_RAW
  CHECK_MUSA_CONTIGUOUS(raw_out);
  CHECK_CONTIGUOUS_2D(raw_out);
  TVM_FFI_ICHECK_EQ(raw_out.device().device_type, norm_out.device().device_type);
  TVM_FFI_ICHECK_EQ(raw_out.size(0), norm_out.size(0));
  TVM_FFI_ICHECK_EQ(raw_out.size(1), norm_out.size(1));
  TVM_FFI_ICHECK(dtype_equal(raw_out.dtype(), norm_out.dtype()));
  TVM_FFI_ICHECK_EQ(raw_out.device().device_id, norm_out.device().device_id);
#endif
  CHECK_CONTIGUOUS_2D(residual_in);
  CHECK_CONTIGUOUS_2D(residual_out);
  CHECK_CONTIGUOUS_2D(norm_out);
  TVM_FFI_ICHECK_EQ(weight.ndim(), 1);
  TVM_FFI_ICHECK_EQ(residual_in.size(0), residual_out.size(0));
  TVM_FFI_ICHECK_EQ(residual_in.size(1), residual_out.size(1));
  TVM_FFI_ICHECK_EQ(residual_in.size(0), norm_out.size(0));
  TVM_FFI_ICHECK_EQ(residual_in.size(1), norm_out.size(1));
  TVM_FFI_ICHECK_EQ(residual_in.size(1), weight.size(0));
  TVM_FFI_ICHECK(dtype_equal(residual_in.dtype(), residual_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(residual_in.dtype(), norm_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(residual_in.dtype(), weight.dtype()));
  TVM_FFI_ICHECK_EQ(
      rank_data.device().device_type, residual_in.device().device_type);
  TVM_FFI_ICHECK_EQ(
      rank_data.device().device_id, residual_in.device().device_id);
  TVM_FFI_ICHECK(rank_data.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(rank_data.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(rank_data.size(0), kMaxRanks);
  TVM_FFI_ICHECK_EQ(signal_ptrs_cpu.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(signal_ptrs_cpu.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(signal_ptrs_cpu.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(signal_ptrs_cpu.size(0), world_size);
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size);

  RankSignals sg{};
  const auto* sig_ptrs = static_cast<const int64_t*>(signal_ptrs_cpu.data_ptr());
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(sig_ptrs[i]);
  }

  const auto* device_data_ptr =
      reinterpret_cast<const RankData*>(rank_data.data_ptr());
  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(residual_in.device());
  const int rows = static_cast<int>(residual_in.size(0));
  const int hidden = static_cast<int>(residual_in.size(1));
  const int64_t numel64 = tensor_numel(residual_in);
  TVM_FFI_ICHECK_LE(
      numel64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));

  if (dtype_equal(residual_in.dtype(), dl_float16)) {
    dispatch_world_size(
        device_data_ptr, sg, self_sg,
        static_cast<const half*>(residual_in.data_ptr()),
        static_cast<half*>(residual_out.data_ptr()),
        static_cast<half*>(norm_out.data_ptr()),
        static_cast<half*>(raw_out.data_ptr()),
        static_cast<const half*>(weight.data_ptr()), static_cast<int>(rank),
        static_cast<int>(world_size), rows, hidden, eps, stream);
  } else if (dtype_equal(residual_in.dtype(), dl_bfloat16)) {
    dispatch_world_size(
        device_data_ptr, sg, self_sg,
        static_cast<const __mt_bfloat16*>(residual_in.data_ptr()),
        static_cast<__mt_bfloat16*>(residual_out.data_ptr()),
        static_cast<__mt_bfloat16*>(norm_out.data_ptr()),
        static_cast<__mt_bfloat16*>(raw_out.data_ptr()),
        static_cast<const __mt_bfloat16*>(weight.data_ptr()),
        static_cast<int>(rank), static_cast<int>(world_size), rows, hidden, eps,
        stream);
  } else {
    TVM_FFI_THROW(ValueError)
        << "custom AR fused RMSNorm only supports fp16/bf16";
  }

  // Error-state queries are deferred until after graph replay; querying it
  // during capture is rejected by the MUSA runtime.
}

void vllm_musa_custom_ar_v2_fused_allreduce_rmsnorm_row_unregistered(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView inp,
    ffi::TensorView residual_in,
    ffi::TensorView residual_out,
    ffi::TensorView norm_out,
    ffi::TensorView raw_out,
    ffi::TensorView weight,
    int64_t self_signal_ptr,
    int64_t self_buffer_ptr,
    int64_t max_size_bytes,
    int rank,
    int world_size,
    int hidden,
    double eps) {
  CHECK_MUSA_CONTIGUOUS(inp);
  CHECK_MUSA_CONTIGUOUS(residual_in);
  CHECK_MUSA_CONTIGUOUS(residual_out);
  CHECK_MUSA_CONTIGUOUS(norm_out);
  CHECK_MUSA_CONTIGUOUS(weight);
#if VLLM_MUSA_FUSED_WRITE_RAW
  CHECK_MUSA_CONTIGUOUS(raw_out);
  CHECK_CONTIGUOUS_2D(raw_out);
  TVM_FFI_ICHECK_EQ(raw_out.device().device_type, norm_out.device().device_type);
  TVM_FFI_ICHECK_EQ(raw_out.size(0), norm_out.size(0));
  TVM_FFI_ICHECK_EQ(raw_out.size(1), norm_out.size(1));
  TVM_FFI_ICHECK(dtype_equal(raw_out.dtype(), norm_out.dtype()));
  TVM_FFI_ICHECK_EQ(raw_out.device().device_id, norm_out.device().device_id);
#endif
  CHECK_CONTIGUOUS_2D(inp);
  CHECK_CONTIGUOUS_2D(residual_in);
  CHECK_CONTIGUOUS_2D(residual_out);
  CHECK_CONTIGUOUS_2D(norm_out);
  TVM_FFI_ICHECK_EQ(weight.ndim(), 1);
  TVM_FFI_ICHECK_EQ(inp.size(0), residual_in.size(0));
  TVM_FFI_ICHECK_EQ(inp.size(1), residual_in.size(1));
  TVM_FFI_ICHECK_EQ(inp.size(0), residual_out.size(0));
  TVM_FFI_ICHECK_EQ(inp.size(1), residual_out.size(1));
  TVM_FFI_ICHECK_EQ(inp.size(0), norm_out.size(0));
  TVM_FFI_ICHECK_EQ(inp.size(1), norm_out.size(1));
  TVM_FFI_ICHECK_EQ(inp.size(1), hidden);
  TVM_FFI_ICHECK_EQ(weight.size(0), hidden);
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), residual_in.dtype()));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), residual_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), norm_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), weight.dtype()));
  TVM_FFI_ICHECK_EQ(residual_in.device().device_id, inp.device().device_id);
  TVM_FFI_ICHECK_EQ(residual_out.device().device_id, inp.device().device_id);
  TVM_FFI_ICHECK_EQ(norm_out.device().device_id, inp.device().device_id);
  TVM_FFI_ICHECK_EQ(weight.device().device_id, inp.device().device_id);
  TVM_FFI_ICHECK_EQ(rank_data.device().device_type, inp.device().device_type);
  TVM_FFI_ICHECK_EQ(rank_data.device().device_id, inp.device().device_id);
  TVM_FFI_ICHECK(rank_data.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(rank_data.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(rank_data.size(0), kMaxRanks);
  TVM_FFI_ICHECK_EQ(signal_ptrs_cpu.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(signal_ptrs_cpu.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(signal_ptrs_cpu.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(signal_ptrs_cpu.size(0), world_size);
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size);

  const auto* data_ptr = static_cast<const RankData*>(rank_data.data_ptr());

  RankSignals sg{};
  const auto* sig_ptrs = static_cast<const int64_t*>(signal_ptrs_cpu.data_ptr());
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(sig_ptrs[i]);
  }

  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(inp.device());
  const int64_t numel64 = tensor_numel(inp);
  TVM_FFI_ICHECK_LE(numel64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  const int64_t nbytes =
      numel64 * ((static_cast<int64_t>(inp.dtype().bits) * inp.dtype().lanes + 7) / 8);
  TVM_FFI_ICHECK_LE(nbytes, max_size_bytes);

  const musaError_t copy_err = musaMemcpyAsync(
      reinterpret_cast<void*>(self_buffer_ptr),
      inp.data_ptr(),
      static_cast<size_t>(nbytes),
      musaMemcpyDeviceToDevice,
      stream);
  TVM_FFI_ICHECK_EQ(copy_err, musaSuccess)
      << "MUSA custom AR fused rmsnorm row copy failed: "
      << musaGetErrorString(copy_err);

  const int rows = static_cast<int>(inp.size(0));
  bool launched = false;
  if (dtype_equal(inp.dtype(), dl_float16)) {
    launched = dispatch_residual_rmsnorm_row_world_size(
        data_ptr, sg, self_sg,
        static_cast<const half*>(residual_in.data_ptr()),
        static_cast<half*>(residual_out.data_ptr()),
        static_cast<half*>(norm_out.data_ptr()),
        static_cast<half*>(raw_out.data_ptr()),
        static_cast<const half*>(weight.data_ptr()), rank, world_size, rows,
        hidden, static_cast<float>(eps), stream);
  } else if (dtype_equal(inp.dtype(), dl_bfloat16)) {
    launched = dispatch_residual_rmsnorm_row_world_size(
        data_ptr, sg, self_sg,
        static_cast<const __mt_bfloat16*>(residual_in.data_ptr()),
        static_cast<__mt_bfloat16*>(residual_out.data_ptr()),
        static_cast<__mt_bfloat16*>(norm_out.data_ptr()),
        static_cast<__mt_bfloat16*>(raw_out.data_ptr()),
        static_cast<const __mt_bfloat16*>(weight.data_ptr()), rank, world_size,
        rows, hidden, static_cast<float>(eps), stream);
  } else {
    TVM_FFI_THROW(ValueError)
        << "custom AR fused rmsnorm row only supports fp16/bf16";
  }
  TVM_FFI_ICHECK(launched)
      << "MUSA custom AR fused rmsnorm row unsupported shape";
  // Defer sticky-error queries until after graph replay (see generic path).
}

void vllm_musa_custom_ar_v2_fused_allreduce_rmsnorm_row_registered(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView residual_in,
    ffi::TensorView residual_out,
    ffi::TensorView norm_out,
    ffi::TensorView raw_out,
    ffi::TensorView weight,
    int64_t self_signal_ptr,
    int rank,
    int world_size,
    int hidden,
    double eps) {
  CHECK_MUSA_CONTIGUOUS(residual_in);
  CHECK_MUSA_CONTIGUOUS(residual_out);
  CHECK_MUSA_CONTIGUOUS(norm_out);
  CHECK_MUSA_CONTIGUOUS(weight);
#if VLLM_MUSA_FUSED_WRITE_RAW
  CHECK_MUSA_CONTIGUOUS(raw_out);
  CHECK_CONTIGUOUS_2D(raw_out);
  TVM_FFI_ICHECK_EQ(raw_out.device().device_type, norm_out.device().device_type);
  TVM_FFI_ICHECK_EQ(raw_out.size(0), norm_out.size(0));
  TVM_FFI_ICHECK_EQ(raw_out.size(1), norm_out.size(1));
  TVM_FFI_ICHECK(dtype_equal(raw_out.dtype(), norm_out.dtype()));
  TVM_FFI_ICHECK_EQ(raw_out.device().device_id, norm_out.device().device_id);
#endif
  CHECK_CONTIGUOUS_2D(residual_in);
  CHECK_CONTIGUOUS_2D(residual_out);
  CHECK_CONTIGUOUS_2D(norm_out);
  TVM_FFI_ICHECK_EQ(weight.ndim(), 1);
  TVM_FFI_ICHECK_EQ(residual_in.size(0), residual_out.size(0));
  TVM_FFI_ICHECK_EQ(residual_in.size(1), residual_out.size(1));
  TVM_FFI_ICHECK_EQ(residual_in.size(0), norm_out.size(0));
  TVM_FFI_ICHECK_EQ(residual_in.size(1), norm_out.size(1));
  TVM_FFI_ICHECK_EQ(residual_in.size(1), hidden);
  TVM_FFI_ICHECK_EQ(weight.size(0), hidden);
  TVM_FFI_ICHECK(dtype_equal(residual_in.dtype(), residual_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(residual_in.dtype(), norm_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(residual_in.dtype(), weight.dtype()));
  TVM_FFI_ICHECK_EQ(
      rank_data.device().device_type, residual_in.device().device_type);
  TVM_FFI_ICHECK_EQ(
      rank_data.device().device_id, residual_in.device().device_id);
  TVM_FFI_ICHECK(rank_data.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(rank_data.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(rank_data.size(0), kMaxRanks);
  TVM_FFI_ICHECK_EQ(signal_ptrs_cpu.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(signal_ptrs_cpu.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(signal_ptrs_cpu.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(signal_ptrs_cpu.size(0), world_size);
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size);

  const auto* device_data_ptr =
      reinterpret_cast<const RankData*>(rank_data.data_ptr());
  const auto* sig_ptrs = static_cast<const int64_t*>(signal_ptrs_cpu.data_ptr());
  RankSignals sg{};
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(sig_ptrs[i]);
  }
  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(residual_in.device());
  const int rows = static_cast<int>(residual_in.size(0));
  bool launched = false;
  if (dtype_equal(residual_in.dtype(), dl_float16)) {
    launched = dispatch_residual_rmsnorm_row_world_size(
        device_data_ptr, sg, self_sg,
        static_cast<const half*>(residual_in.data_ptr()),
        static_cast<half*>(residual_out.data_ptr()),
        static_cast<half*>(norm_out.data_ptr()),
        static_cast<half*>(raw_out.data_ptr()),
        static_cast<const half*>(weight.data_ptr()), rank, world_size, rows,
        hidden, static_cast<float>(eps), stream);
  } else if (dtype_equal(residual_in.dtype(), dl_bfloat16)) {
    launched = dispatch_residual_rmsnorm_row_world_size(
        device_data_ptr, sg, self_sg,
        static_cast<const __mt_bfloat16*>(residual_in.data_ptr()),
        static_cast<__mt_bfloat16*>(residual_out.data_ptr()),
        static_cast<__mt_bfloat16*>(norm_out.data_ptr()),
        static_cast<__mt_bfloat16*>(raw_out.data_ptr()),
        static_cast<const __mt_bfloat16*>(weight.data_ptr()), rank, world_size,
        rows, hidden, static_cast<float>(eps), stream);
  } else {
    TVM_FFI_THROW(ValueError)
        << "custom AR fused rmsnorm row only supports fp16/bf16";
  }
  TVM_FFI_ICHECK(launched)
      << "MUSA custom AR fused rmsnorm row unsupported shape";
  // Defer sticky-error queries until after graph replay (see generic path).
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    vllm_musa_custom_ar_v2_fused_allreduce_rmsnorm_unregistered,
    vllm_musa_custom_ar_v2_fused_allreduce_rmsnorm_unregistered);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    vllm_musa_custom_ar_v2_fused_allreduce_rmsnorm_registered,
    vllm_musa_custom_ar_v2_fused_allreduce_rmsnorm_registered);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    vllm_musa_custom_ar_v2_fused_allreduce_rmsnorm_row_unregistered,
    vllm_musa_custom_ar_v2_fused_allreduce_rmsnorm_row_unregistered);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    vllm_musa_custom_ar_v2_fused_allreduce_rmsnorm_row_registered,
    vllm_musa_custom_ar_v2_fused_allreduce_rmsnorm_row_registered);

#endif  // SGL_CUSTOM_AR_FUSED_RMSNORM
