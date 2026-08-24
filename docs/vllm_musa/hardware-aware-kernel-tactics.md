# Hardware-aware kernel tactics on MUSA

This document defines the vLLM-MUSA policy for kernels whose launch geometry
depends on the number of available multiprocessors (MPs). It applies to AOT,
MUBIN, JIT, TileLang, Triton, MATE, and vendor-library providers.

## Why architecture alone is not enough

Two devices can implement the same `mp_31` ISA while exposing different
`multi_processor_count` values. A fixed grid or tile can therefore cross a
wave boundary on one device but not another. For example, a 120-tile launch
takes two full waves on MP60, but three waves with a small tail on MP52/MP56.
The resulting discontinuity can be much larger than the nominal difference in
peak throughput.

Use the runtime MP count, not a product-name substring or an assumed S5000
default. Do not derive a marketing core-bin label from MP count unless that
mapping has been verified for the exact fleet.

## Combined design pattern

vLLM-MUSA adopts the common ground between three kernel ecosystems:

- **MATE:** resolve the physical MP count at runtime, use it for persistent
  grids and config selection, and share generated configurations between JIT
  and AOT packaging.
- **NVIDIA CUDA/CUTLASS:** pass queried SM count into persistent tile
  schedulers and use grid/SM wave or remainder terms when choosing split-K and
  tile geometry.
- **AMD ROCm:** pass CU count into skinny GEMM, split-K, and attention grids;
  use offline tuned solutions or bounded online tuning where the vendor
  library supports it.

The vLLM-MUSA serving hot path must remain deterministic:

1. Query `MusaKernelHardware(device_capability, multiprocessor_count)` once.
2. Look up an exact, evidence-backed tactic key containing the MP count.
3. If no exact entry exists, apply a documented integer heuristic only when
   that heuristic has an independently validated safe envelope.
4. Otherwise use the established fallback implementation.

Runtime serving must not benchmark, synchronize device data to the host, or
silently borrow the nearest MP-count entry.

## Tactic and cache identity

At minimum, an offline result or runtime cache entry must include:

```text
kernel/provider identity
source and dependency revisions
MUSA architecture and multiprocessor_count
driver, runtime, compiler, MATE, and TileLang/Triton versions
dtype and layout
shape/workload bucket
tile, stages, warps/squads, split count, and persistent-block count
eager/compile/CUDAGraph mode
```

Changing any compiled launch constant must change the JIT cache identity.
Runtime-only scheduling values need not force another binary when the kernel
ABI already accepts them, but the timing-cache/tactic key must still include
the MP count.

## AOT versus JIT

Prefer **AOT plus runtime dispatch** when the candidate set is small, compile
cost is high, or the same binary accepts a runtime grid/split value. Package a
bounded fat set of validated tactics and select one by exact key.

Prefer **JIT** when the MP-sensitive choice changes compile-time constants,
the source template is small, and prewarming can hide compilation. JIT output
must use a persistent cache and fall back to a validated AOT provider on cache,
compile, or device-query failure.

Do not move a kernel to JIT solely because different device bins exist. First
show that the winning compile-time tactic changes across real devices and that
the end-to-end gain pays for added startup and cache complexity.

## Qualification matrix

For every proposed tactic-map entry:

1. Run on the exact physical MP-count bin; `set_num_mps()` is useful for
   scheduler experiments but is not evidence for another physical device.
2. Use identical source/image, driver family, shape, dtype, cooling class, and
   benchmark method where the fleet permits it.
3. Interleave candidates, flush L2 for weight-streaming kernels, retain all raw
   samples, and report median, p95, and IQR.
4. Check outputs against the production fallback, including NaN/Inf and
   poison-output checks.
5. Explain the result with tile count, resident blocks per MP, wave count, and
   tail-wave utilization. Treat this model as a ranking aid, not proof.
6. Validate the winning tactic through the compiled/captured production path
   and a model-level regression before enabling it by default.

The current fused-MoE dispatch policy is the reference implementation: its
shape key includes `multiprocessor_count`, and unknown counts fail closed. The
MP60-only direct FA3 metadata path is another correct example: other counts
fall back until their scheduler equivalence is proven.

## Source references

- CUDA device properties:
  <https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html>
- NVIDIA CUTLASS persistent schedulers:
  <https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel>
- HIP device properties:
  <https://rocm.docs.amd.com/projects/HIP/en/latest/reference/api-reference.html>
- AMD hipBLASLt tuning:
  <https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/how-to/use-hipblaslt.html>
- MATE runtime MP resolution:
  <https://github.com/MooreThreads/mate/blob/main/mate/mate_runtime.py>
