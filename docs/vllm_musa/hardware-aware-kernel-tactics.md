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
host-visible workload class and any data-dependent scheduling class
tile, stages, warps/squads, split count, and persistent-block count
eager/compile/CUDAGraph mode
```

Changing any compiled launch constant must change the JIT cache identity.
Runtime-only scheduling values need not force another binary when the kernel
ABI already accepts them, but the timing-cache/tactic key must still include
the MP count.

Keep the **compiled-binary cache** and the **measured-winner map** separate.
The binary cache answers whether an implementation already exists; the winner
map answers whether that implementation was qualified for the current device
and workload. A practical winner key is:

```text
(schema_version, kernel_abi, source_revision,
 architecture, active_mp, op_stage,
 exact_or_bucketed_shape, dtype, layout, quantization)
```

Record driver and compiler versions with the evidence. Do not make the driver
an exact selector dimension until a same-device A/B demonstrates that it
changes the winner. Invalidate or requalify entries when the kernel ABI,
library release, source revision, or result schema changes.

Use a fail-closed lookup ladder:

1. exact architecture, MP count, and shape;
2. a validated shape bucket on the same MP count;
3. an architecture-common tactic proven robust on all required MP bins; then
4. the current production default.

Never substitute the nearest MP count. Store the winning margin, sample count,
dispersion, correctness status, and covered route distributions with each map
entry so a narrow or noisy result cannot masquerade as a stable policy.

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

JIT does not by itself solve data-dependent scheduling. For example, an MoE
GEMV tile can depend on both the physical MP count and whether routed tokens
are balanced across experts or concentrated on a hot expert. If that route
class exists only in device memory, do not add a host-side MP-only override or
synchronize it to the CPU on the serving hot path. Prefer, in order:

1. a route-invariant tactic that stays within the accepted regression envelope;
2. metadata already computed on-device by the production routing path;
3. a device-side or persistent scheduler that consumes that metadata; or
4. the established fallback when the route class is unknown.

A timing-cache entry may include a route class only when production can obtain
the same class without introducing a new synchronization or scan.

For the current native MoE GEMV path, `topk_ids` and `topk_weights` are already
GPU-resident and replay-safe, but a per-expert histogram is not. The existing
DeepGEMM count/offset buffers belong to a different large-prefill backend, and
the optional routed-expert capture buffer is intended for later host return.
Neither should be borrowed implicitly by decode GEMV.

If route-aware selection is pursued, the smallest graph-safe experiment is a
GPU classifier that writes a fixed-size route-class buffer followed by a
bounded set of graph-static AOT launches guarded by that device value. Measure
classifier and suppressed-launch overhead as part of the candidate. A stronger
long-term design is one fixed resident grid, normally derived from active MP
count, that consumes device-side counts/offsets in a persistent work loop.
Neither design requires copying expert IDs to the host, and neither assumes a
device-side dynamic launch facility.

## Qualification matrix

For every proposed tactic-map entry:

1. Run on the exact physical MP-count bin; `set_num_mps()` is useful for
   scheduler experiments but is not evidence for another physical device.
2. Compare baseline and candidate on the same physical device with identical
   source/image, shape, dtype, and benchmark method. Record the driver and
   cooling class as provenance; they need not match another MP-count type when
   the decision is based on within-device normalized uplift rather than
   cross-device absolute latency.
3. Interleave candidates, flush L2 for weight-streaming kernels, retain all raw
   samples, and report median, p95, and IQR.
4. Check outputs against the production fallback, including NaN/Inf and
   poison-output checks.
5. Explain the result with tile count, resident blocks per MP, wave count, and
   tail-wave utilization. Treat this model as a ranking aid, not proof.
6. Validate the winning tactic through the compiled/captured production path
   and a model-level regression before enabling it by default.
7. For kernels with data-dependent grids or reuse, sweep representative input
   distributions (for example balanced and hot-expert routing). Reject a
   static MP-only entry when its winner changes with a distribution that the
   host dispatcher cannot observe safely.

Across MP-count types, compare the identity of the winning tactic and its
same-device speedup over the local baseline. Do not rank hardware types by raw
latency when their driver, cooling, clocks, or host stack differ.

The current fused-MoE dispatch policy is the reference implementation: its
shape key includes `multiprocessor_count`, and unknown counts fail closed. The
MP60-only direct FA3 metadata path is another correct example: other counts
fall back until their scheduler equivalence is proven.

## Empirical decision records

The initial S5000 MP56/MP60 campaign produced two useful negative/positive
boundaries:

- Existing fused-add-RMSNorm AOT blocks showed large (roughly 15-35%) common
  shape-aware gains on real Qwen and DeepSeek-V4-Flash shapes, but every stable
  winner was identical on MP56 and MP60. An exact-MP RMSNorm map was therefore
  rejected; the common shape map requires a separate model-level E2E gate.
- Existing MoE GEMV AOT blocks did show MP56/MP60 winner changes, but those
  changes also depended on balanced versus hot-expert routing. A static
  MP-only MoE map was rejected because the host dispatcher does not safely
  observe that route distribution. Dense DSV4 GEMV was route-independent, but
  its stable winners were again identical on MP56 and MP60.

These results illustrate why MP count belongs in the evidence and cache key,
but is not necessarily sufficient as the complete runtime dispatch key.

## Cross-vendor implementation guidance

The policy above intentionally combines mechanisms instead of copying one
vendor stack wholesale:

- AMD AITER keys tuned dense GEMM results by both `gfx` and `cu_num`, then by
  shape, dtype, layout, and quantization fields. Its MoE selector extends the
  key with token, expert, and top-k dimensions and retains bounded fallbacks.
  This is the closest existing schema to an S5000 `(mp_31, active_mp, shape)`
  map.
- hipBLASLt supports offline problem-to-solution tuning but scopes solution
  indices to a library release and device architecture. Reuse that invalidation
  discipline; do not treat a numeric tactic ID as permanently portable.
- Composable Kernel retains a finite catalog of AOT instances, filters invalid
  arguments, and profiles the survivors. Its grouped-GEMM tile loop consumes
  group metadata on-device with a persistent grid, demonstrating that a small
  AOT catalog and device-side dynamic scheduling are complementary.
- Triton persistent matmul and CUTLASS grouped scheduling query the real SM
  count and bound a resident grid by it. CUTLASS explicitly provides a
  device-only grouped scheduler for metadata produced by an earlier GPU
  kernel, avoiding a host synchronization.
- MATE supplies the MUSA-side JIT/code-generation and cache layer. Use it when
  a winning tactic changes compile-time constants or the AOT catalog would
  grow without bound; keep a small, stable AOT catalog otherwise.

The resulting best practice is therefore not a blanket AOT-to-JIT migration.
It is a versioned offline winner map keyed by active MP count, a bounded
AOT/JIT implementation catalog, a robust fallback ladder, and device-side
persistent scheduling for truly data-dependent work.

## Source references

- CUDA device properties:
  <https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html>
- NVIDIA CUTLASS persistent schedulers:
  <https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel>
- HIP device properties:
  <https://rocm.docs.amd.com/projects/HIP/en/latest/reference/api-reference.html>
- AMD hipBLASLt tuning:
  <https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/how-to/use-hipblaslt.html>
- AMD AITER dense and MoE tuned selectors:
  <https://github.com/ROCm/aiter/blob/7cb8f3389aeaf76c820ea692f295273f02e6071e/aiter/tuned_gemm.py>
  and
  <https://github.com/ROCm/aiter/blob/7cb8f3389aeaf76c820ea692f295273f02e6071e/aiter/fused_moe.py>
- AMD Composable Kernel grouped tile-loop scheduler:
  <https://github.com/ROCm/rocm-libraries/blob/fb3b576286941d15f63b5037c63844eaf7b53f29/projects/composablekernel/include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_xdl_cshuffle_tile_loop.hpp>
- Triton persistent matmul:
  <https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html>
- CUTLASS device-only grouped scheduler:
  <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html>
- MATE runtime MP resolution:
  <https://github.com/MooreThreads/mate/blob/main/mate/mate_runtime.py>
