# Model-bound RuntimePlans and operator tuning domains

`vllm_musa.engine_plan` is a repository-local builder and adapter for
`vllm_musa.runtime_plan`. It is not a separately distributed plugin yet: use it
from the checked-out vLLM-MUSA source (or the vLLM-MUSA wheel that contains the
package):

Start with the [AutoTuner quickstart](autotuner_quickstart.md) for the shortest
domain discovery, evidence, build, validation, and deployment path. This page
documents the detailed contracts and current implementation boundaries.

```bash
python -m vllm_musa.engine_plan --help
python -m vllm_musa.engine_plan catalog \
  --include-tuning \
  --output /tmp/musa-plan-catalog.json
python -m vllm_musa.engine_plan profiles
python -m vllm_musa.engine_plan autotune domains
```

The `profiles` output describes repository-owned per-model defaults. These
versioned JSON documents choose registered RuntimePlan defaults without
model-specific Python provider branches. AutoTuner never rewrites them in a
serving process: measured decisions are a higher-precedence overlay sealed into
`selected-plan.json`. Catalog fallback, profile default, and measured overlay
remain separate decision sources.

The runtime path is intentionally simple:

```text
serving baseline
    -> profiler hotspot table
    -> AutoTuner collect or qualified producer + import
    -> obtain sealed timing evidence
    -> typed runtime_decisions
    -> clean RuntimePlan replay and serving A/B
```

Two identities deliberately remain separate:

- A RuntimePlan **profile**, such as `qwen3.moe`, binds sealed decisions to the
  built-in provider for the live model. It remains exact in the runtime receipt.
- A tuning **domain**, such as `fused_moe.fp8_block`, names a reusable operator
  capability contract: context extraction, candidates, fallback, correctness
  oracle, and RuntimePlan lowering. It must not branch on architecture or model
  ID.

The target still records the exact architecture, model ID, dtype,
quantization, topology, workload, hardware bin, and software identity. Reusing
a domain for a new model therefore means remeasuring that model's reachable
keys and producing a new model-bound plan; it never means copying a Qwen winner.
Pre-domain timing artifacts remain valid. Their stable operation ID produces a
candidate-domain set; target and case capability predicates may identify one
unique domain. Zero or multiple matches remain `unknown` rather than being
retagged or rejected. Their profile or sealed bytes are never rewritten.

## How to find a model's best measured serving path

You do not need timings for every operator or every physical GPU. Compiler,
runtime, or source changes do invalidate a measured plan and require a bounded
re-tune; a new image digest alone does not when those exact components match.

1. Freeze one model revision, precision/quantization, TP/PP layout, workload
   distribution, concurrency/request rate, image digest, and source commit.
2. Run a real compiled/captured vLLM-MUSA baseline. Save throughput, TTFT/TPOT
   or latency percentiles, memory, graph mode, and a semantic client response.
3. Profile that exact workload and rank the stage/layer/kernel hotspots.
4. Map only the top bottlenecks to registered atomic tactics. The AutoTuner
   selects each exact operation context or token bucket independently; it does
   not treat a hand-written threshold or whole-model policy as a tactic.
5. Repeat candidate measurements across the required route/seed matrix. A
   bucket keeps the baseline when any required slice is incorrect, unstable,
   regresses at p95, or misses the minimum speedup.
6. Seal the winner into a typed plan, replay it through compiled/captured
   serving, and run the focused model regression slice.

The standalone, repository-owned workflow is the
[AutoTuner quickstart](autotuner_quickstart.md). Agent workspaces may compose
that contract with their local GPU-lease, remote-container, profiler,
microbenchmark, and serving-performance skills; those workspace skills are not
part of this repository checkout.

## Builder and artifact lifecycle

The builder consumes a live tactic catalog and timing evidence. The RMSNorm
commands below are the scalar-decision example. The exact CLI
options are checked by `python -m vllm_musa.engine_plan --help`; save that help
output with each experiment because the command surface may evolve while the
feature is experimental.

```bash
python -m vllm_musa.engine_plan target \
  --profile qwen3.text_generation \
  --architecture Qwen3ForCausalLM \
  --model-id /path/to/Qwen3-32B-FP8 \
  --hidden-size 5120 \
  --dtype bfloat16 \
  --quantization fp8 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --source-revision vllm=<pinned-vllm-sha> \
  --source-revision vllm-musa=<candidate-vllm-musa-sha> \
  --image-digest sha256:<immutable-image-digest> \
  --phase serving \
  --batch-min 1 \
  --batch-max 4 \
  --tokens-min 1 \
  --tokens-max 4096 \
  --max-model-len 2624 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 4 \
  --compile-mode VLLM_COMPILE \
  --graph-mode FULL_DECODE_ONLY \
  --output generated/<ticket>/target.json

python -m vllm_musa.engine_plan autotune collect \
  --domain fused_add_rms_norm.bf16_h5120 \
  --target generated/<ticket>/target.json \
  --output generated/<ticket>/autotune.raw.json \
  --summary-output generated/<ticket>/autotune.summary.json

# A second collect may resume the raw cache; exact hits do no GPU measurement.
python -m vllm_musa.engine_plan autotune collect \
  --domain fused_add_rms_norm.bf16_h5120 \
  --target generated/<ticket>/target.json \
  --resume generated/<ticket>/autotune.raw.json \
  --output generated/<ticket>/autotune.reused.raw.json \
  --summary-output generated/<ticket>/autotune.reused.summary.json

python -m vllm_musa.engine_plan cache seal \
  generated/<ticket>/autotune.reused.raw.json \
  --output generated/<ticket>/autotune.timing.json

python -m vllm_musa.engine_plan build \
  --timings generated/<ticket>/autotune.timing.json \
  --plan-id <stable-plan-id> \
  --output generated/<ticket>/selected_plan.json

python -m vllm_musa.engine_plan inspect \
  generated/<ticket>/selected_plan.json
```

Fused-MoE uses atomic `upstream`, `gemv`, and `grouped_gemm` tactics. Run
`benchmarks/fused_moe/benchmark_dispatch_crossover.py` from the exact clean
source checkout for every required `balanced`, `unique_random`, and `hot`
route with at least three common seeds. Run both eager and `--graph-capture`
matrices when the serving target captures decode graphs. The benchmark records
raw event timings for audit, but selection uses one median per alternating
backend round so repeated samples inside a round are not treated as independent.
Run route/seed jobs sequentially on each visible GPU. The producer takes an
exclusive per-checkout/per-device process lock and rejects a concurrent sweep.

Qualified v8 crossover evidence uses at least a 512 MiB cold-cache flush,
which evicts more than eight times the PH1/S5000 60 MiB whole-chip LLC,
together with the built-in, non-weakenable correctness thresholds. It snapshots MUSA
`*.mudmp` files before measurement and rejects any new, changed, missing, or
unscannable dump evidence. Low-flush experiments remain diagnostics and must not
be imported into a sealed plan.

Normalize those files into typed timing-v3 evidence, then build as usual:

```bash
python -m vllm_musa.engine_plan autotune import \
  --domain fused_moe.fp8_block \
  --target generated/<ticket>/target.json \
  --image-digest sha256:<immutable-image-digest> \
  --evidence generated/<ticket>/moe-eager-balanced-seed11.json \
  --evidence generated/<ticket>/moe-eager-balanced-seed23.json \
  --evidence generated/<ticket>/moe-eager-balanced-seed37.json \
  --evidence generated/<ticket>/moe-eager-random-seed11.json \
  --evidence generated/<ticket>/moe-eager-random-seed23.json \
  --evidence generated/<ticket>/moe-eager-random-seed37.json \
  --evidence generated/<ticket>/moe-eager-hot-seed11.json \
  --evidence generated/<ticket>/moe-eager-hot-seed23.json \
  --evidence generated/<ticket>/moe-eager-hot-seed37.json \
  --evidence generated/<ticket>/moe-capture-balanced-seed11.json \
  --evidence generated/<ticket>/moe-capture-balanced-seed23.json \
  --evidence generated/<ticket>/moe-capture-balanced-seed37.json \
  --evidence generated/<ticket>/moe-capture-random-seed11.json \
  --evidence generated/<ticket>/moe-capture-random-seed23.json \
  --evidence generated/<ticket>/moe-capture-random-seed37.json \
  --evidence generated/<ticket>/moe-capture-hot-seed11.json \
  --evidence generated/<ticket>/moe-capture-hot-seed23.json \
  --evidence generated/<ticket>/moe-capture-hot-seed37.json \
  --output generated/<ticket>/moe-eager-capture.timing.json

python -m vllm_musa.engine_plan build \
  --timings generated/<ticket>/moe-eager-capture.timing.json \
  --plan-id <stable-plan-id> \
  --output generated/<ticket>/selected_plan.json
```

For the current contextual fused-MoE domain, `autotune import` emits canonical,
fingerprinted timing-v3 evidence directly. It does not benchmark and its output
does not need a subsequent `cache seal`. Import the same ordered evidence twice
and require byte-identical output before building a plan. Evidence mode and
timing schema are independent; a future scalar import domain may emit
timing-v2.

`autotune import-fused-moe` remains a compatibility alias for this canonical
domain-oriented command.

Eager and capture evidence for the same static serving target must be imported
together. They become two exact operator-shape entries inside one timing cache
and one plan variant. Building from separate eager and capture timing caches
would create duplicate static runtime keys and is rejected.

Eager evidence must cover the target's serving token maximum. Capture evidence
instead must cover every configured `cudagraph_capture_sizes` entry; it need
not benchmark a larger serving-only maximum that vLLM will never capture.

The builder also makes the selected fused-MoE policy representable by vLLM's
compile ranges. Tokens one and two share the first compiled range; if their
measured winners differ, both points use the registered upstream fallback and
the selection receipt records
`fallback_required:compile_range_token_1_2_backend_transition`. The builder
never promotes either tactic across that boundary without evidence.

The importer requires a clean source SHA matching the target, the exact image
digest, S5000 capability/performance bin, package versions, dispatcher identity,
correctness, cold-cache policy, and graph replay proof for capture evidence.
It also requires the exclusive-process-lock and device-dump-free receipts.
Duplicate shape/route/seed files are rejected rather than counted twice.

Every `autotune collect` re-probes the live hardware, software packages, and
vLLM/vLLM-MUSA source revisions before it enumerates candidates or reuses the
resume cache. Applicability-key drift fails closed without performing a GPU
measurement. Regenerate the target and collect fresh evidence after a driver,
MUSA SDK/compiler package, source revision, or hardware performance-bin change.
Evidence-only changes such as device UUID and visible device count do not block
collection. A resume cache must also match the current target's complete
model/workload/runtime applicability envelope. Its UUID and device count may
differ; an image-digest change is allowed through this preflight but changes
the toolchain key and therefore invalidates the affected cached measurements.

`target --source-revision` may declare the intended source identity when target
capture runs in an image without a build manifest. That is target construction
only; the declared values are never copied from `target.json` as live identity.
Before collection, mount a manifest from the exact build being measured. The
collector requires it at the path set by
`MUSA_ENGINE_BUILD_MANIFEST`, or `/etc/vllm-musa/build-manifest.json` by
default. It must contain exactly:

```json
{
  "schema_version": "musa.engine_build.v1",
  "source_revisions": {
    "vllm": "<exact-sha>",
    "vllm-musa": "<exact-sha>"
  }
}
```

Missing, unreadable, or malformed manifests fail closed before candidate
enumeration. The current manifest contract verifies exactly these two source
trees; extra source-revision keys in a target are rejected as unverifiable
rather than being copied from the target or silently ignored. All software
version keys present in the target are live-probed and compared. `driver`,
`musa`, and `torch` have dedicated probes; any additional key is interpreted as
a Python distribution name and read through package metadata. Do not use an
arbitrary executable name as an optional key. Every supplied version key must
resolve to a concrete value; `unknown` is rejected even for optional keys.

`--profile` is the built-in RuntimePlan provider profile, not a descriptive
experiment name. It must match the profile registered for the model runtime;
Qwen3 dense text generation uses `qwen3.text_generation`, while
`Qwen3MoeForCausalLM` uses `qwen3.moe`. Do not pass
`fused_moe.fp8_block` as `--profile`: it is a tuning domain, not a model
provider profile, and the runtime would correctly reject that receipt.

`--compile-mode` and `--graph-mode` use the exact vLLM enum names emitted by
the live config. The supported compilation names are `NONE`,
`STOCK_TORCH_COMPILE`, `DYNAMO_TRACE_ONCE`, and `VLLM_COMPILE`; the supported
graph names are `NONE`, `PIECEWISE`, `FULL`, `FULL_DECODE_ONLY`, and
`FULL_AND_PIECEWISE`. Lowercase aliases and unknown future values fail closed
at target creation instead of producing a plan that cannot pass final runtime
validation.

The artifact's `runtime_decisions` table is the only optimization projection
the host consumes. For repository profile families, each newly built variant
also seals the declarative profile ID and canonical fingerprint. Missing or
stale bindings for profile-owned decisions, fixed decisions, unknown keys,
wrong types, unsupported non-fallback values, context drift, damaged
fingerprints, and semantic failures fail closed. Legacy artifacts without this
binding remain readable only when all projected decisions are external-only.

## AutoTuner boundary

The first-party AutoTuner borrows TensorRT-LLM's runner, context/bucket, and
cache-hit/miss lifecycle without importing TensorRT-LLM or CUDA code. An
operation adapter declares legal candidates and how to produce evidence; the
core owns deterministic keys, invalidation, correctness/status evidence, and
typed timing output. A domain reports its evidence mode through `autotune
domains`: `collect` runs its registered measurement adapter, while `import`
validates and normalizes evidence from its qualified producer.

Both evidence modes are explicit offline work. `collect` and an import
domain's producer may compile candidates and use a GPU; `autotune import`
itself must not launch a benchmark. None can run in a serving request path.
Production starts in a fresh process from a sealed RuntimePlan; a missing or
invalid decision uses the registered fallback and never starts a benchmark.

The scalar vertical slice tunes the registered BF16 H4096 or H5120 fused-add
RMSNorm crossover. Each hidden size is an explicit domain so target validation,
timing evidence, and plan applicability remain exact rather than silently
generalizing a measurement across shapes.
It profiles the native reference plus the MUSA C-extension and JIT runners once
per required rows bucket, then projects those measurements onto legal threshold
candidates. The selected integer decision
`musa.fused_add_rms_norm.min_rows` replaces the built-in `64` before compile
range construction. The built-in value remains the fail-closed fallback.
Because vLLM workers use spawn, platform initialization propagates the resolved
value, plan fingerprint, and selected variant through child-process environment
transport before workers start. It is not a user-facing tuning knob. Every
child reloads the sealed artifact and must match the parent-pinned fingerprint
and variant before it can compile or capture a graph.

The contextual vertical slice tunes fused-MoE. `MusaFusedMoeShape` plus a
token bucket is the runtime context; route mode and seed are evidence axes, not
runtime keys. Its canonical tuning domain is `fused_moe.fp8_block`. The current
domain is qualified for S5000 capability 3.1 and exact active-MP bins, BF16
activations, FP8 E4M3 block-128 expert weights, FP32 scale layouts, SiLU, and
non-EP execution. Timing-v3/case-v2 selects an atomic winner for every contiguous
bucket, coalesces adjacent equal winners, and emits
`musa.fused_moe.dispatch_policy.v1`. The hot path performs an immutable exact
shape lookup and range lookup. Priority is diagnostic override, exact active
plan, existing calibrated table, then established upstream fallback. The
policy is materialized before compile/capture and transported privately to
spawned workers.

The controlled legacy fused-MoE path emits a process-scoped dispatch receipt
on first use of each selected token context on every rank. It includes the
rank, actual token count, graph mode, backend, source, selected range, plan ID,
and exact plan fingerprint. A production replay must observe
`source=runtime_plan` with the pinned fingerprint; policy materialization alone
does not prove that the selected kernel executed.

Some vLLM quantization methods construct an internal modular kernel before
compile/capture (for example, the FP8 Qwen path). Those paths emit a separate
`MUSA fused-MoE plan binding receipt` during kernel setup. The binding receipt
records the final weight shape, implementation, and the sealed plan's
`planned_backend` ranges, but it intentionally has no `actual_tokens` or
selected runtime backend and is never accepted as a dispatch receipt. Until a
modular implementation consumes the plan at its request-time dispatch seam,
that model/path is fallback-only evidence and cannot support a performance or
SOTA claim.

Context boundaries also split vLLM's compile ranges so a compiled graph cannot
bake one backend across a token boundary. RuntimePlan does not own the graph
memory budget: it never adds a capture shape or raises
`max_cudagraph_capture_size`. vLLM intentionally warms compile ranges that have
no configured graph shape; the MUSA patch runs those warmups with graph runtime
disabled so they still compile without triggering lazy capture. Formal capture
continues to use only the final configured graph sizes. RuntimePlan validates
that vLLM preserved every transition endpoint and, after attention/speculative
shape resolution, that padding to a captured graph cannot cross a tactic
transition. A token-1 transition fails closed because this pinned vLLM cannot
represent it.

For contextual fused-MoE capture evidence, the target records the exact
`cudagraph_capture_sizes`. AutoTuner promotes only measurements at those
reachable padded graph keys: each measured capture shape owns the actual-token
interval that vLLM pads to it. Missing graph-ladder identity or missing evidence
for a reachable key fails closed. The final ladder is also a runtime
applicability key, so a plan measured for `[1, 2, 4]` cannot activate under a
different graph-memory budget.

Contextual fused-MoE capture plans currently fail closed when speculative
decoding is enabled. The target/draft graph domains use different decode query
lengths, and the artifact schema does not yet bind both domains. Eager policies
remain usable; capture-policy support should be enabled only after both graph
key sets are represented and tuned independently.

Cache identity includes the operation/shape bucket, candidate-set digest,
implementation fingerprints, relevant software/source dependencies, immutable
image digest, hardware performance bin, and compile/graph mode. Adding a
candidate, changing the compiler image, or changing the measurement policy
(warmup, repetitions, cold-cache size, or graph mode) therefore invalidates
the affected operation instead of letting an older winner hide the new search
space.

AutoTuner solves the inner operator/configuration search. The agent still owns
the workload and SLA, coarse runtime choices such as scheduler or KV layout,
search budgets, and the final compiled/captured end-to-end acceptance gate.

“Optimal” means the best eligible registered tactic for the declared context,
candidate set, metric, hardware/software identity, and measured envelope. It
does not mean a global optimum for an arbitrary model. A production acceptance
claim additionally requires a sealed non-fallback winner, a runtime receipt
showing that the planned backend actually executed, and a clean-start paired
serving A/B whose confidence interval clears the configured gain threshold.
Cross-framework SOTA is a separate comparison and is not implied by a valid
AutoTuner result.

The current first-party adapter set is deliberately bounded:

- `fused_add_rms_norm.bf16_h4096` and
  `fused_add_rms_norm.bf16_h5120` use `collect`;
- `fused_moe.fp8_block` uses a qualified producer plus `import`.

`autotune domains` is the source of truth. A registered domain ID does not
imply that an arbitrary operator or model has a generic collector.

## Enabling another model

Classify the new model by operator capability, not by its class name:

1. **Remeasure only:** the reached operation, semantic shape key, candidates,
   fallback, correctness oracle, and plan application phase all match an
   existing domain. Capture real-forward keys and follow that domain's declared
   `collect` or qualified producer + `import` path, conditionally seal, then run
   clean replay, semantic, and compiled/captured A/B lifecycle.
2. **Extend the domain:** the operator semantics are unchanged, but the model
   reaches a new supported layout, graph context, or candidate. Extend the
   capability predicate/context extractor and its tests without adding a model
   name guard.
3. **Add a domain:** the operation schema, correctness semantics, fallback, or
   RuntimePlan application boundary differs. Register a new adapter rather
   than weakening an existing domain.

Stop when the target has no same-operation fallback or correctness oracle, a
real forward does not reach the tuning point, or a decision cannot be applied
before compilation/capture. Model semantic success does not replace the
operator numeric oracle, and cross-tactic bit-exact text is not required when
the declared numeric and model-quality gates pass.

## Applicability versus measurement provenance

Timing evidence records the exact environment in which it was measured:
device UUID, device count, driver/runtime/SDK, package versions, source
revisions, image digest, and timestamp. Physical UUID and image digest are
measurement provenance; they do not force a new plan by themselves.

Runtime applicability is keyed by the model/workload envelope, device family
and capability/performance bin, graph/compile context, exact compiler/runtime
package set, exact vLLM and vLLM-MUSA source revisions, and tactic-specific
implementation fingerprints. A software or source mismatch fails closed to the
vLLM-MUSA baseline. This deliberately requires a bounded re-tune after compiler
or implementation changes while still allowing reuse across equivalent cards
and differently packaged images with the same runtime bits.

`autotune.domain` is a sealed evidence identity used to route and revalidate
the operator contract. It does not replace any model applicability key or the
RuntimePlan profile. A declared domain/operation mismatch fails during both
build and sealed-plan validation. Legacy caches use the operation only to find
candidate domains, then require a unique target/case-compatible match; this
allows FP8 and BF16 domains to share one runtime operation without ambiguity.

Production deployment must mount the plan as a root-owned, read-only artifact
and should require a fingerprint supplied by trusted deployment configuration:

```bash
export MUSA_ENGINE_PLAN=/etc/vllm-musa/selected-plan.json
export MUSA_ENGINE_PLAN_REQUIRE_PINNED_FINGERPRINT=true
export MUSA_ENGINE_PLAN_FINGERPRINT="${PROMOTED_MUSA_ENGINE_PLAN_FINGERPRINT:?required}"
exec vllm serve ...
```

`MUSA_ENGINE_PLAN_REQUIRE_PINNED_FINGERPRINT` accepts the case-insensitive
boolean values `1`, `true`, `yes`, `on`, `0`, `false`, `no`, and `off`.
Whitespace is ignored. An empty or unrecognized value aborts activation. When
the flag is true, `MUSA_ENGINE_PLAN_FINGERPRINT` must be non-empty before the
parent reads the artifact; the runtime will not derive a pin from the same
artifact it is being asked to trust. The flag and externally supplied pin are
inherited by spawned workers, which reload the artifact and enforce the same
identity.

When the strict flag is unset or false, the parent retains the development
behavior of pinning the first successfully loaded artifact for its workers.
Replacing the path after that selection causes spawned workers to reject the
new content. A fingerprint proves content identity, not authorship: production
promotion must source it from a trusted manifest, signature, or attestation
instead of calculating it from the mutable runtime mount.

## What belongs in a plan

The catalog may expose boolean, integer, enum, and structured decisions. Typical
choices include a sparse page size, scheduler variant, graph staging policy,
hybrid cache layout, or a structured IR provider order. The plan chooses among
legal implementations; implementation code continues to enforce dtype, shape,
device, layout, and capture-safety guards.

Graph-affecting choices must be materialized before compile/graph capture.
Request-time routing can choose only among already materialized variants and
cannot change graph topology or allocation layout inside a captured graph.

## Current scope

The builder/adapter lives under `vllm_musa/engine_plan` and is imported as a
normal first-party Python package in the vLLM-MUSA runtime environment. It does
not require a standalone wheel, console-script entry point, Docker
installation, source-cache checkout, or release-image verifier. Those
distribution layers can be added later if an external producer/consumer ABI
stabilizes.

Schema-v5 scalar decisions remain scoped to TP1/PP1. Schema-v6 permits
multi-TP fused-MoE contextual decisions because the per-rank shape, exact TP/PP
target, hardware bin, graph mode, source identity, and implementation
fingerprints are all sealed and revalidated. Mixing a scalar TP1-only decision
into that multi-TP variant fails closed. Each multi-GPU model still requires
its own compiled/captured serving matrix before production promotion.
