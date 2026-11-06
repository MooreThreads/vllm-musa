# DeepSeek-V4.1 native prefill context parallelism

## Code ownership

The implementation is upstream vLLM PCP backported to the immutable v0.32.0 pin
in `third_party/PINS`. The backports live in
`vllm_musa/patches/series/upstream/`, after the existing MUSA series. The upstream
commit mapping and conflict resolutions are listed in that directory's parent
`README.md`.

MUSA uses the normal `GPUModelRunner`, `PCPManager`, and
`DeepseekV41ModelState`. The former MUSA-specific PCP manager, model-state wrapper,
attention interception, and per-MoE-layer routing gathers are not used.

## Data flow

1. The native PCP manager splits each prefill into balanced DualChunkSwap rows
   and builds rank-local request, position, slot and padding metadata. Block
   tables attach at KV initialization; the manager also participates in profiling.
2. Attention and compressor projections operate on rank-local tokens. Ratio-2
   compression exchanges predecessor boundary rows and publishes the latest ring
   state rather than gathering full `kv_score` tensors and recomputing the whole
   chunk on every rank.
3. SWA KV and compressed latents are gathered in rank-major order. Cache slot
   mappings and positions follow exactly that payload order. Main compressed KV
   and indexer K are published from the same gathered latent.
4. MUSA writes SWA KV through a KV-only entry point. The subsequent fused local
   query preparation receives an empty slot mapping, so it does not write KV a
   second time. KV-only execution allocates no dummy query tensor.
5. Input IDs are gathered at the model forward boundary, not at every MoE layer.
   Native PCP padding metadata covers gathered routing rows; replay prepares its
   own aligned inputs. Expert computation retains upstream TP-by-PCP semantics.
6. Engram lookback uses local chunk starts, skips empty chunks, and promotes
   request indices to int64 before multiplying by the token stride.
7. Encoder replay rebuilds only missing window state on a prefix hit. Decoder
   replay publishes source KV first, then keeps the required trailing-window and
   scoring rows. The replay batch has its own PCP metadata and padding.

Collectives stay on the default stream, in identical rank order. The existing
shared-expert-fold PCP guard and independent mHC reduction fix remain in place.

## MUSA-specific code

- `vllm_musa/_custom_ops.py`: packed SWA KV-only dispatch.
- `vllm_musa/kernels/deepseek_v4_rope_kv_insert.py`: KV-only specialization of the
  existing Triton packing kernel; no query programs or query allocation.
- `csrc/musa/attention/deepseek_v4_cache_store.mu`: binds the existing native
  KV/RoPE packing kernel as the fallback when the Triton route is unavailable.
- The final upstream-directory integration patch selects those operations and
  makes MCCL reduce-scatter move the requested axis to the front and back.
- `worker.py` retains request-token staging preallocation, but does not select a
  private PCP runner.

The native extension must be rebuilt for the new KV-only binding. Reapplying
Python patches alone is not sufficient for its fallback path.

## Runtime scope

The target is DeepSeek-V4.1 TP4/PCP2 on eight MUSA GPUs, using MRV2, DCP=PP=DP=1,
packed FP8 KV, and `cudagraph_mode=NONE`. APC requires SWA bounded replay;
PCP bounded replay also requires APC, and is disabled upstream without it. The
upstream model retains its own feature checks; this backport does not enable
PCP with DSpark, KV transfer, microbatching, or MegaMoE. DeepSeek-V4 PCP is not
advertised by these DeepSeek-V4.1 backports.

Keep validated cookbook launch arguments, model length and memory utilization.
This implementation does not add a fixed KV-cache memory budget, change indexer
tile size, or replace custom all-reduce configuration. For prefill comparisons,
use one output token and record all configuration differences explicitly.

## Validation

Offline gates:

Use a pristine pinned checkout for `verify` and replay. Apply the complete series
once to a fresh checkout for build validation; do not force a full replay onto an
already patched tree when updating an entry.

```bash
python3.10 tools/musa_sync.py check-series --repo <clean-pinned-vllm> --replay
python3.10 tools/musa_sync.py check-series --repo <maintenance-vllm> --round-trip
python3.10 tools/musa_sync.py verify --repo <clean-pinned-vllm>
VLLM_PCP_TEST_REPO=<patched-vllm> python3.10 -m pytest -q tests/test_deepseek_pcp.py
```

Device gates, after a clean patch application and native rebuild:

- KV-only cache bytes must match the existing fused MUSA writer for Triton and
  native paths, including negative slots, partial mappings, and zero rows.
- Run upstream PCP compressor, replay, Engram and routing tests on MUSA. Exercise
  ratio-2 groups split across ranks, empty chunks, mixed prefill/decode, prefix
  hits, and nonzero request-state indices.
- The final test-routing patch expands replay, routing-padding and ratio-2
  compressor coverage to MUSA. CUDA-specific cache-format and stream tests remain
  CUDA-only; they are not MUSA acceptance evidence. The MUSA cache-byte tests above
  cover the KV-only writer independently.
- Compare PCP1 and PCP2 semantic outputs with and without bounded replay before
  benchmarking. Validate both TP8 and TP4/PCP2 on the same patched source.
- For cold-prefill TTFT, reset prefix cache outside timing or use distinct prompt
  prefixes. Otherwise APC hits can be mistaken for PCP speedup.
- Report 4K–1M input / 1 output, memory peaks, batch-token limit, cache capacity,
  capture state and repeated TTFT samples. Offline tests do not prove speedup.

Source migration does not alter a running process. Keep existing services intact
until an explicitly scheduled runtime validation.
