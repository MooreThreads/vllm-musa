"""Shared helpers for MUSA TileLang kernels."""

import os

# TileLang keeps its own on-disk cache, and it is the one build cache this package
# does not name itself.  Point it at an arch-and-compiler-qualified directory --
# the same tag the ninja-built kernels use -- so a cached TileLang module can never
# be reused across arches, which on MUSA means silently wrong results rather than a
# failure.  Set before ``import tilelang`` because the library reads the variable at
# import time; an explicit TILELANG_CACHE_DIR still wins.
if not os.environ.get("TILELANG_CACHE_DIR"):
    try:
        from vllm_musa.jit_kernel.csrc.jit import build_tag, cache_root

        _tilelang_tag = build_tag()
    except Exception:  # noqa: BLE001 - no device and no arch list: keep the default
        # TileLang keys its own cache on the compile target, so its default is still
        # arch-safe; this default only makes the split explicit and inspectable.
        _tilelang_tag = None
    if _tilelang_tag:
        os.environ["TILELANG_CACHE_DIR"] = str(cache_root() / f"tilelang__{_tilelang_tag}")

import tilelang  # noqa: E402 - must follow the cache-dir default above
import torch  # noqa: E402

tilelang.set_log_level("WARNING")

# if hasattr(torch, "musa") and torch.musa.is_available():
#     tilelang.disable_cache()

MUSA_COMMON_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}
if hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
    MUSA_COMMON_PASS_CONFIGS[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = True
elif hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    MUSA_COMMON_PASS_CONFIGS[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = False
for _key, _value in (
    ("TL_DISABLE_THREAD_STORAGE_SYNC", True),
    ("TL_ENABLE_MUSA_BURST", True),
    ("TL_ENABLE_REDUCE_BURST", True),
    ("TL_DISABLE_SAFE_MEMORY_ACCESS", True),
    ("TL_DISABLE_INDEX_TYPE_PROMOTION", True),
):
    if hasattr(tilelang.PassConfigKey, _key):
        MUSA_COMMON_PASS_CONFIGS[getattr(tilelang.PassConfigKey, _key)] = _value

MUSA_COMPILE_FLAGS = [
    "-Od3",
    "-fno-signed-zeros",
    "-fmusa-flush-denormals-to-zero",
    "-mllvm",
    "-misched=mtgpu-max-ilp",
    "-mllvm",
    "-mtgpu-if-convert=1",
    "-mllvm",
    "-mtgpu-tiny-offset-hint=1",
    "-mllvm",
    "-mtgpu-enable-postra-sched=0",
    "-mllvm",
    "-misched-recompute-slotindex=1",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
]


def tilelang_dtype(dtype: torch.dtype) -> str:
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float32:
        return "float32"
    raise TypeError(f"Unsupported dtype for TileLang MUSA kernel: {dtype}")


def storage_window(tensor: torch.Tensor) -> torch.Tensor:
    storage_size = 1
    for size, stride in zip(tensor.shape, tensor.stride()):
        if stride < 0:
            raise ValueError("tensor must not have negative strides")
        if size == 0:
            return tensor.reshape(-1)
        storage_size += (size - 1) * stride
    return tensor.as_strided((storage_size,), (1,))


def layout_strides(
    tensor: torch.Tensor,
    positions_ndim: int,
    head_size: int,
) -> tuple[int, int, int, int]:
    if tensor.dim() not in (positions_ndim + 1, positions_ndim + 2):
        raise ValueError(
            "tensor must have shape [..., hidden_size] "
            "or [..., num_heads, head_size]"
        )

    batch_stride = tensor.stride(0) if positions_ndim == 2 else 0
    token_stride = tensor.stride(positions_ndim - 1)
    if tensor.dim() == positions_ndim + 2:
        head_stride = tensor.stride(-2)
        dim_stride = tensor.stride(-1)
    else:
        dim_stride = tensor.stride(-1)
        head_stride = head_size * dim_stride
    return batch_stride, token_stride, head_stride, dim_stride
