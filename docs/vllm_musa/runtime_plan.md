# Runtime plans

`vllm_musa.runtime_plan` is the single optimization-policy entry point for
vLLM-MUSA. It turns stable model and execution facts into an immutable
`RuntimePlan`; consumers do not repeat model-name, topology, or shape-policy
checks.

Model defaults are data, not provider-specific Python branches. Versioned
documents under `vllm_musa/runtime_plan/profiles/` classify a normalized model
and execution context, declare supported decisions, and choose conservative
defaults. The generic loader validates a closed condition DSL and produces the
same immutable `RuntimePlan` used by every consumer.

The split is deliberate:

- a plan chooses a provider, algorithm, layout, or policy value for a known
  deployment context;
- the selected implementation still owns per-call correctness checks such as
  dtype, device, tensor layout, and legal shape;
- unsupported or unknown plan decisions fail closed and never manufacture a
  kernel capability.

## Consumer API

Resolve from a complete `VllmConfig` as early as the decision's lifecycle phase
requires:

```python
from vllm_musa.runtime_plan import RuntimeDecision, resolve_runtime_plan

plan = resolve_runtime_plan(vllm_config)
page_size = plan.value(
    RuntimeDecision.DEEPSEEK_V4_FLASHMLA_SPARSE_PAGE_SIZE,
    64,
)
use_fast_path = plan.enabled(RuntimeDecision.QWEN_FA3_SCHEDULER)
separate_pool = plan.selected(
    RuntimeDecision.HYBRID_KV_CACHE_POOL_LAYOUT,
    "separate",
)
```

Runtime owners that are constructed once should bind the same snapshot with
`bind_runtime_plan(owner, vllm_config)`. Request-time code reads that bound
plan through `runtime_plan_enabled()` rather than reconstructing a model
profile.

Do not call `enabled()` for integer, enum, or structured decisions. The catalog
rejects wrong value types and defines a conservative fallback for every key.
`plan.decision_source(key)` reports `profile_default`, `engine_plan`, or
`catalog_fallback`; it is intended for receipts and debugging, not dispatch.

## Decision catalog and phases

`runtime_plan/catalog.py` is the host-owned schema. Every `RuntimeDecision`
must have exactly one `RuntimeDecisionSpec` declaring:

- value kind: boolean, integer, enum, or structured;
- the latest materialization phase: config defaults, cache layout, model init,
  compile, graph capture, or request;
- the conservative fallback and legal choices;
- whether the key is supplied by the repository-local engine-plan projection.

Graph-affecting decisions must be materialized before compilation or graph
capture. Request-time routing may choose only among variants already
materialized by the plan; it must not change graph topology or allocation
layout.

## Declarative per-model profiles

Discover the installed profile documents and their decision metadata with:

```bash
python -m vllm_musa.engine_plan profiles
```

The profile files contain only JSON data. They may reference an allowlisted
`model.*` or `execution.*` field and use `all`, `any`, `not`, named condition
references, typed comparisons, collection membership, or an exact tuple
matrix. Unknown fields/operators, extra keys, reference/dependency cycles,
non-finite numbers, excessive depth, external-only decisions, and invalid
typed values fail closed. The loader never evaluates Python expressions or
calls model objects.

Each decision rule separates:

- `supported_when`: the model and stack capability boundary;
- `value_when` and `value`: the repository default inside that boundary;
- `tunability`: `fixed`, `profile`, or `autotune` lifecycle metadata;
- `requires`: other enabled defaults that must be resolved first.

The effective precedence is:

```text
host catalog fallback
    < versioned per-model profile default
    < validated measured EnginePlan override for a registered lowering
```

An EnginePlan override cannot expand support, change a fixed decision or the
lifecycle phase, or bypass the implementation's tensor/request/resource guard.
Runtime still consumes one sealed plan snapshot; it does not watch profile
files or tune in a request path. The current builder derives measured overlays
automatically for three external decisions. A profile-tunable default may also
be updated in reviewed profile JSON, or supplied explicitly through
`build --runtime-decisions` after an equivalent evidence gate; that explicit
projection is not an AutoTuner-selected winner until its own tuning domain and
lowering exist.

For an existing decision and normalized model family, adding a model or
changing a default is a profile-file change plus evidence and parity tests, not
a new Python provider. New decision kinds, consumers, normalized context
fields, or kernel capabilities still require code. Add a new decision by:

1. adding its stable key to `RuntimeDecision`;
2. registering its type, fallback, and phase in the catalog;
3. adding the consumer and correctness/capability guard;
4. declaring its support/default rule in the relevant profile;
5. consuming only `RuntimePlan.value()`, `selected()`, or `enabled()`;
6. covering positive, fallback, incomplete-context, and one-field-mismatch
   cases.

Keep runtime tensor legality in the implementation's support predicate. A plan
is policy, not permission to skip correctness guards.

## Repository-local engine-plan projection

The repository-local `vllm_musa.engine_plan` package builds schema-v5 plans for
scalar selections and schema-v6 plans for contextual selections from typed
timing evidence. A selected variant exposes one `runtime_decisions` table. IR
provider priority uses the structured key `vllm.ir_op_priority`; model
decisions use the same projection and receipt path. See the
[AutoTuner quickstart](autotuner_quickstart.md) for the evidence-to-plan
lifecycle and artifact contract.

At startup the plugin applies allowlisted config defaults, then projects typed
decisions from the exact selected variant. New variants bind the packaged
profile ID and canonical fingerprint. The host checks that binding, plan
identity, profile, key registration, value type, tunability, support, and
fallback semantics before creating the effective `RuntimePlan`. Damaged plans,
stale profile configuration, context drift, unknown keys, fixed overrides, and
unsupported non-fallback choices abort startup.

The projection is optional. With no selected artifact, built-in model providers
produce the same deterministic defaults from the packaged profile documents.
The profile document ID and canonical fingerprint participate in the
RuntimePlan fingerprint. The EnginePlan is loaded lazily only when
`MUSA_ENGINE_PLAN` is set; no standalone plugin wheel or entry-point
installation is required.
