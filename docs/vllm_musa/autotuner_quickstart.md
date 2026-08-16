# AutoTuner quickstart

`vllm_musa.engine_plan` tunes registered operator tactics for one exact
model/workload target. AutoTuner produces timing evidence; the separate
`build` step turns that evidence into a model-bound RuntimePlan. It does not
build a TensorRT-style executable engine.

Only `selected-plan.json` is consumed by serving. Targets, raw measurements,
summaries, and standalone timing caches remain build-time evidence.

```text
target + tactic catalog
          |
  collect or producer + import
          |
  sealed timing evidence
          |
        build
          |
  selected-plan.json
          |
 clean serving replay + receipt
```

See [Model-bound RuntimePlans and operator tuning domains](engine_plan.md) for
the complete target fields, fused-MoE qualification matrix, invalidation
rules, and production boundaries.

## 1. Save the live contract

Run from the checked-out vLLM-MUSA source or an environment containing the
matching vLLM-MUSA package:

```bash
export EVIDENCE_ROOT=/path/to/one/autotune-run
mkdir -p "$EVIDENCE_ROOT/help" "$EVIDENCE_ROOT/collect" \
  "$EVIDENCE_ROOT/import" "$EVIDENCE_ROOT/sealed"

python -m vllm_musa.engine_plan --help \
  > "$EVIDENCE_ROOT/help/engine-plan.txt"
python -m vllm_musa.engine_plan autotune domains \
  > "$EVIDENCE_ROOT/tuning-domains.json"
python -m vllm_musa.engine_plan profiles \
  > "$EVIDENCE_ROOT/runtime-profiles.json"
python -m vllm_musa.engine_plan catalog --include-tuning \
  --output "$EVIDENCE_ROOT/catalog.json"
```

Read `evidence_mode` from `tuning-domains.json`; never assume every domain
uses `collect`.

`runtime-profiles.json` is the packaged default-policy catalog. It is not
timing evidence: a measured override must still pass the domain workflow and
be sealed into `selected-plan.json`.

| Domain | Evidence mode | Current scope |
|---|---|---|
| `fused_add_rms_norm.bf16_h4096` | `collect` | BF16 H4096 fused-add RMSNorm crossover |
| `fused_add_rms_norm.bf16_h5120` | `collect` | BF16 H5120 fused-add RMSNorm crossover |
| `fused_moe.fp8_block` | `import` | S5000 BF16 activation, FP8 block-128 fused-MoE dispatch |

These are the only first-party adapters currently registered. Adding a model
does not automatically add a tuning domain.

For Qwen3-235B or another H4096 target, create the target with
`--hidden-size 4096` and use `--domain fused_add_rms_norm.bf16_h4096` in both
collect and auditor commands. Domain and hidden size must match exactly; an
H5120 cache is not portable to H4096.

## 2. Capture the exact target

The target binds the model profile and revision, dtype/quantization, TP/PP,
workload, compile/graph mode, hardware performance bin, software versions,
source revisions, and image provenance. The example below is abbreviated only
by its placeholder values; every flag is part of the target identity.

```bash
python -m vllm_musa.engine_plan target \
  --profile "${MODEL_PROFILE:?required}" \
  --architecture "${MODEL_ARCHITECTURE:?required}" \
  --model-id "${MODEL_ID:?required}" \
  --hidden-size "${HIDDEN_SIZE:?required}" \
  --dtype "${DTYPE:?required}" \
  --quantization "${QUANTIZATION:?required}" \
  --tensor-parallel-size "${TP_SIZE:?required}" \
  --pipeline-parallel-size "${PP_SIZE:?required}" \
  --source-revision "vllm=${VLLM_SHA:?required}" \
  --source-revision "vllm-musa=${VLLM_MUSA_SHA:?required}" \
  --image-digest "${IMAGE_DIGEST:?required}" \
  --phase serving \
  --batch-min "${BATCH_MIN:?required}" \
  --batch-max "${BATCH_MAX:?required}" \
  --tokens-min "${TOKENS_MIN:?required}" \
  --tokens-max "${TOKENS_MAX:?required}" \
  --max-model-len "${MAX_MODEL_LEN:?required}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:?required}" \
  --max-num-seqs "${MAX_NUM_SEQS:?required}" \
  --compile-mode "${COMPILE_MODE:?required}" \
  --graph-mode "${GRAPH_MODE:?required}" \
  --output "$EVIDENCE_ROOT/target.json"
```

For contextual capture evidence, also pass the final ordered
`--cudagraph-capture-sizes SIZE ...` ladder. It becomes a runtime
applicability key and cannot be inferred after tuning.

For `collect`, the live environment must also expose the exact build manifest
through `MUSA_ENGINE_BUILD_MANIFEST` or
`/etc/vllm-musa/build-manifest.json`. The collector re-probes the live stack;
it does not trust source or package values copied from `target.json`.

## 3A. Run a `collect` domain

`collect` benchmarks cache misses. Always pass the domain explicitly even
when the CLI currently has a compatibility default:

```bash
python -m vllm_musa.engine_plan autotune collect \
  --domain fused_add_rms_norm.bf16_h5120 \
  --target "$EVIDENCE_ROOT/target.json" \
  --output "$EVIDENCE_ROOT/collect/timing-cache.mutable.json" \
  --summary-output "$EVIDENCE_ROOT/collect/summary.json"

python -m vllm_musa.engine_plan autotune collect \
  --domain fused_add_rms_norm.bf16_h5120 \
  --target "$EVIDENCE_ROOT/target.json" \
  --resume "$EVIDENCE_ROOT/collect/timing-cache.mutable.json" \
  --output "$EVIDENCE_ROOT/collect/timing-cache.reused.json" \
  --summary-output "$EVIDENCE_ROOT/collect/reused-summary.json"

python -m vllm_musa.engine_plan cache seal \
  "$EVIDENCE_ROOT/collect/timing-cache.reused.json" \
  --output "$EVIDENCE_ROOT/sealed/timing-cache.json"
```

The exact-resume gate requires `cache_hits>0`, `cache_misses=0`,
`invalidated=0`, `measured=0`, `scheduled_trials=0`, `failed=0`, and
`physical_measurements=0`. A mutable collect output has no sealed fingerprint
and must never be mounted by serving.

## 3B. Run an `import` domain

An import domain has its own qualified benchmark producer. For
`fused_moe.fp8_block`, first run
`benchmarks/fused_moe/benchmark_dispatch_crossover.py` for every required
route, seed, and eager/capture mode described in the full guide. Then normalize
the same ordered evidence twice:

```bash
evidence_args=(
  --evidence "${EVIDENCE_1:?required}"
  --evidence "${EVIDENCE_2:?required}"
)

python -m vllm_musa.engine_plan autotune import \
  --domain fused_moe.fp8_block \
  --target "$EVIDENCE_ROOT/target.json" \
  --image-digest "${IMAGE_DIGEST:?required}" \
  "${evidence_args[@]}" \
  --output "$EVIDENCE_ROOT/sealed/timing-cache.json"

python -m vllm_musa.engine_plan autotune import \
  --domain fused_moe.fp8_block \
  --target "$EVIDENCE_ROOT/target.json" \
  --image-digest "${IMAGE_DIGEST:?required}" \
  "${evidence_args[@]}" \
  --output "$EVIDENCE_ROOT/import/timing-cache.repeat.json"

cmp "$EVIDENCE_ROOT/sealed/timing-cache.json" \
  "$EVIDENCE_ROOT/import/timing-cache.repeat.json"
```

Extend `evidence_args` to every required file; the two entries above only show
the repeatable `--evidence` syntax, not a complete fused-MoE qualification
matrix.

For the current `fused_moe.fp8_block` domain, `autotune import` validates and
seals contextual timing-v3 evidence directly. Do not run `cache seal` on its
output. It must not launch a benchmark, and repeated normalization must be
byte-identical. `autotune import-fused-moe` is only a compatibility alias for
the domain-oriented command. Evidence mode and timing schema are independent
contracts; a future scalar import domain may produce timing-v2.

## 4. Build and inspect the RuntimePlan

Both evidence modes now converge on one sealed timing cache:

```bash
: "${PLAN_ID:?set a stable plan ID}"

python -m vllm_musa.engine_plan build \
  --timings "$EVIDENCE_ROOT/sealed/timing-cache.json" \
  --plan-id "$PLAN_ID" \
  --output "$EVIDENCE_ROOT/sealed/selected-plan.json"

python -m vllm_musa.engine_plan validate \
  "$EVIDENCE_ROOT/sealed/timing-cache.json" \
  > "$EVIDENCE_ROOT/sealed/timing-validation.json"
python -m vllm_musa.engine_plan validate \
  "$EVIDENCE_ROOT/sealed/selected-plan.json" \
  > "$EVIDENCE_ROOT/sealed/plan-validation.json"
python -m vllm_musa.engine_plan validate \
  --context "$EVIDENCE_ROOT/target.json" \
  "$EVIDENCE_ROOT/sealed/selected-plan.json" \
  > "$EVIDENCE_ROOT/sealed/context-validation.json"
python -m vllm_musa.engine_plan validate \
  --context "$EVIDENCE_ROOT/target.json" \
  --early \
  "$EVIDENCE_ROOT/sealed/selected-plan.json" \
  > "$EVIDENCE_ROOT/sealed/early-validation.json"
python -m vllm_musa.engine_plan inspect \
  "$EVIDENCE_ROOT/sealed/selected-plan.json" \
  > "$EVIDENCE_ROOT/sealed/inspect.txt"
python -m vllm_musa.engine_plan explain --json \
  --context "$EVIDENCE_ROOT/target.json" \
  "$EVIDENCE_ROOT/sealed/selected-plan.json" \
  > "$EVIDENCE_ROOT/sealed/explain.json"
```

The plan must retain a registered fallback even when no non-fallback candidate
wins. A valid plan is not by itself a performance claim.

## 5. Clean replay

Terminate the collection/producer process. Start serving in a fresh process
with only the promoted plan and a fingerprint supplied by trusted deployment
configuration:

```bash
export MUSA_ENGINE_PLAN=/etc/vllm-musa/selected-plan.json
export MUSA_ENGINE_PLAN_REQUIRE_PINNED_FINGERPRINT=true
export MUSA_ENGINE_PLAN_FINGERPRINT="${PROMOTED_MUSA_ENGINE_PLAN_FINGERPRINT:?required}"
exec vllm serve "${MODEL_ID:?required}" "${SERVE_ARGS[@]}"
```

The runtime never reads raw evidence, summaries, or a standalone timing cache.
Prove the chosen backend with the runtime receipt, then run clean-start
compiled/captured serving A/B plus semantic and model-quality gates.

## Artifact contract

| Artifact | Producer | Mutable? | Consumer or purpose |
|---|---|---:|---|
| `target.json` | `engine_plan target` | No | Exact measurement and applicability envelope |
| `catalog.json` | `catalog --include-tuning` | No | Registered candidates, fallback, and implementation identities |
| producer evidence JSON | Domain-owned benchmark | No | Raw import-domain timing, correctness, and qualification audit |
| `timing-cache.mutable.json` | `autotune collect` | Yes | Resume input only; never a serving input |
| `summary.json` | `autotune collect` | No | Hits, misses, invalidation, measurements, failures, and cost |
| timing-v2 | current scalar collect adapter then `cache seal` | No | Fingerprinted scalar timing evidence |
| timing-v3 | current contextual import adapter | No | Fingerprinted contextual timing evidence |
| schema-v5 plan | `build` from scalar timing | No | Model/profile-config-bound scalar `runtime_decisions` |
| schema-v6 plan | `build` from contextual timing | No | Model/profile-config-bound contextual tactic policy |
| validation receipts | `validate` from the exact checkout | No | Timing/plan integrity plus final and early target applicability |
| `inspect.txt` / `explain.json` | `inspect` / `explain` | No | Human- and agent-readable selection audit |
| runtime receipt and serving A/B | Clean serving processes | No | Execution proof and production acceptance evidence |

The deployable artifact is `sealed/selected-plan.json`. It embeds the selected
variant, winner and fallback, selection policy, typed `runtime_decisions`,
declarative profile ID/fingerprint where applicable, timing evidence identity,
exact target envelope, and final fingerprint.

## Stop conditions

Stop rather than hand-authoring a winner when any of these is true:

- the selected domain does not support the target or a reached case;
- a candidate lacks a stable identity, support predicate, correctness oracle,
  or same-operation fallback;
- source, toolchain, hardware bin, compile/graph mode, or workload identity
  drifts during measurement;
- exact resume still performs a physical measurement;
- repeated import is not byte-identical;
- validation, clean replay, backend receipt, semantic checks, or the final
  compiled/captured serving A/B fails.
