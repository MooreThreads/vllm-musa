#pragma once

#include <cstdint>

#include <musa_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>

namespace ffi = tvm::ffi;

constexpr DLDataType dl_float16{kDLFloat, 16, 1};
constexpr DLDataType dl_float32{kDLFloat, 32, 1};
constexpr DLDataType dl_bfloat16{kDLBfloat, 16, 1};
constexpr DLDataType dl_int8{kDLInt, 8, 1};
constexpr DLDataType dl_int32{kDLInt, 32, 1};
constexpr DLDataType dl_int64{kDLInt, 64, 1};
constexpr DLDataType dl_float8_e4m3fn{kDLFloat8_e4m3fn, 8, 1};

inline bool dtype_equal(DLDataType lhs, DLDataType rhs) {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

inline int64_t tensor_numel(ffi::TensorView tensor) {
  int64_t numel = 1;
  for (int i = 0; i < tensor.ndim(); ++i) {
    numel *= tensor.size(i);
  }
  return numel;
}

inline musaStream_t get_stream(DLDevice device) {
  return static_cast<musaStream_t>(
      TVMFFIEnvGetStream(device.device_type, device.device_id));
}

#ifndef TORCH_CHECK
#define TORCH_CHECK(cond, ...) TVM_FFI_ICHECK(cond)
#endif

#ifndef CHECK_EQ
#define CHECK_EQ(a, b) TVM_FFI_ICHECK_EQ((a), (b))
#endif

#ifndef CHECK_MUSA
#define CHECK_MUSA(x) TVM_FFI_ICHECK_EQ((x).device().device_type, kDLExtDev)
#endif

#ifndef CHECK_INPUT
#define CHECK_INPUT(x)                                                         \
  TVM_FFI_ICHECK_EQ((x).device().device_type, kDLExtDev);                      \
  TVM_FFI_ICHECK((x).IsContiguous())
#endif

#ifndef CHECK_MUSA_CONTIGUOUS
#define CHECK_MUSA_CONTIGUOUS(x)                                               \
  TVM_FFI_ICHECK_EQ((x).device().device_type, kDLExtDev);                      \
  TVM_FFI_ICHECK((x).IsContiguous())
#endif

#ifndef CHECK_CONTIGUOUS_2D
#define CHECK_CONTIGUOUS_2D(x)                                                 \
  do {                                                                         \
    TVM_FFI_ICHECK_EQ((x).ndim(), 2);                                          \
    TVM_FFI_ICHECK_EQ((x).stride(1), 1);                                       \
    TVM_FFI_ICHECK_EQ((x).stride(0), (x).size(1));                             \
  } while (0)
#endif
