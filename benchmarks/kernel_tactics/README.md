# MP-aware kernel tactic campaign

This directory contains the op-first campaign used to discover MUSA kernel
configuration candidates before spending time on model-level A/B runs.

The workflow is intentionally fail-closed:

1. Start with `mp_tactic_campaign.json`, which contains public Qwen, DSV4, and
   RMSNorm shape/route recipes. Device and fleet qualification context is
   supplied at run time; it is not embedded in the recipe.
2. Acquire a development-pool SOL lease and expose exactly one MUSA device.
   Set `MUSA_VISIBLE_DEVICES=0`, `CUDA_VISIBLE_DEVICES=0`, and the image's
   `MTHREADS_VISIBLE_DEVICES=0`; the first two must be identical.
3. Run `run_mp_tactic_campaign.py --mode quick|full --expected-mp N`.
   The runner records the expanded command, source identity, runtime MP, device
   fence, and per-cell result hashes in the selected output directory.
4. Reduce one or more bundles with `summarize_mp_tactic_campaign.py`.  A row
   is promotion-eligible only when correctness, cold-cache p95, seed/route
   agreement, any locally requested host replication, and device-isolation
   checks all pass.
5. Fill the prediction inputs (`op_time_share`, `production_hit_rate`) from a
   current profiler capture.  The reducer computes the Amdahl estimate, but it
   never treats an op win as an E2E claim.

## Timing contract

The GEMV and fused-add-RMSNorm benches use paired alternating order, one
launch per sample, an 8,000 MiB int32 L2 eviction buffer zeroed immediately
before each timed launch, MUSA events, p50/p90/p99, and finite/non-poison
output checks.  GEMV rows retain the exact seeded `topk_ids` digest and expert
load histogram.  DSV4's one-token split-tile arm is recorded as the effective
selector arm, so a requested-but-ignored block cannot become a false winner.

`dsv4-mhc-jit-fuse` additionally separates TileLang compile time from warm
kernel timing and always includes the production resolver configuration as a
paired baseline, even in a reduced quick sweep.

The campaign files do not change production dispatch.  A production map must
be added only after the generated evidence and a final representative E2E
gate are reviewed. Always write run output below the workspace `generated/`
directory (or another explicitly private path); do not commit result JSON,
device UUIDs, hostnames, package receipts, or timing samples.
