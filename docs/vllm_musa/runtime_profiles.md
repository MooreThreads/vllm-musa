# Declarative RuntimePlan profiles

vLLM-MUSA keeps generic decision types, lifecycle phases, fallbacks,
capability guards, and consumers in Python. Repository-owned JSON documents
under `vllm_musa/runtime_plan/profiles/` provide model-family classification,
supported decisions, and defaults.

This separation makes an existing code path configurable without adding a new
model-specific Python resolver. It does not turn correctness or architecture
facts into unrestricted knobs.

## Inspect the installed policy

```bash
python -m vllm_musa.engine_plan profiles \
  > /tmp/runtime-profiles.json
python -m vllm_musa.engine_plan profiles \
  --profile vllm_musa/runtime_plan/profiles/qwen.json
```

The result includes all 33 registered decisions with their type, phase,
fallback, profile families, and tunability, followed by each packaged profile's
canonical fingerprint and declared decisions.

Tunability has three values:

- `fixed`: a correctness, architecture, or resource-layout invariant that an
  EnginePlan cannot change;
- `profile`: a repository default that can change through a reviewed,
  evidence-backed profile-file update, or an explicit sealed EnginePlan
  projection inside the same support boundary;
- `autotune`: an EnginePlan decision with a registered measurement workflow.

The host catalog owns this classification. A profile cannot reclassify a
decision.

## Profile format

The schema is `musa.runtime_profile.v1`:

```json
{
  "schema_version": "musa.runtime_profile.v1",
  "id": "example-family",
  "priority": 100,
  "reason": "declarative example-family runtime profile",
  "provider_when": {"ref": "family_identity"},
  "conditions": {
    "family_identity": {
      "path": "model.model_type",
      "op": "eq",
      "value": "example"
    },
    "supported_shape": {
      "all": [
        {"path": "model.dtype", "op": "eq", "value": "bfloat16"},
        {"path": "execution.tensor_parallel_size", "op": "eq", "value": 1}
      ]
    }
  },
  "classifications": [
    {
      "when": {"ref": "family_identity"},
      "family": "qwen3",
      "role": "text_generation"
    }
  ],
  "profiles": [
    {"when": true, "profile": "qwen3.example"}
  ],
  "decisions": [
    {
      "decision": "qwen.v2_sampling",
      "supported_when": {"ref": "supported_shape"},
      "value_when": {"ref": "supported_shape"},
      "value": true,
      "tunability": "profile"
    }
  ]
}
```

Multiple profile documents may be installed. The generic provider evaluates
them by descending priority. No match retains baseline; one highest-priority
match resolves the plan; equal-priority matches fail as ambiguous.
Within one document, classification and runtime-profile arrays are ordered
first-match rules, matching the historical provider precedence. Put the most
specific identity first and cover intentional overlap with a regression test.

## Closed condition language

Conditions may read allowlisted normalized `model.*` and `execution.*` fields.
They may use:

- `all`, `any`, and `not`;
- a named `ref`;
- one `path` with `eq`, `ne`, `in`, `not_in`, `gt`, `ge`, `lt`, or `le`;
- `contains_any` or `contains_all` for normalized collections;
- multiple `paths` with an exact `tuple_in` matrix.

The loader rejects unknown fields/operators, extra keys, missing fields,
duplicate decisions, reference/dependency cycles, invalid values, external-only
defaults, non-finite numbers, excessive depth, and excessive document size. It
does not support Python expressions, imports, regexes, callables, or arithmetic.

Only normalized signatures are visible. Raw model objects, tensors, request
payloads, and resource allocators remain behind trusted code. Existing
tensor-shape and request/resource guards are always applied in addition to a
plan decision.

## Support, default, and dependency

Each rule has separate meanings:

- `supported_when` establishes that the implementation can legally run;
- `value_when` chooses whether the profile supplies its declared `value`;
- omission of a value uses the host catalog fallback;
- `requires` names other declared decisions whose effective default must be
  true before this rule can be supported.

The loader topologically orders dependencies and rejects cycles. The decision
catalog remains authoritative for value type, phase, fallback, legal choices,
profile family, and tunability.

One document cannot declare the same decision twice, so rules never silently
overwrite one another. Mutually exclusive alternatives should normally be one
typed enum decision. `requires` covers boolean prerequisites; a new
cross-decision conflict or resource invariant remains a host-catalog/code
change until it has an explicit validated representation.

## Effective precedence

```text
catalog fallback
    < packaged per-model profile default
    < validated EnginePlan measured override for a registered lowering
```

An EnginePlan override can change only a non-fixed decision already supported
by the live profile, except for explicitly registered external-only decisions.
It cannot change the decision type, phase, fallback, profile binding,
capability predicate, or consumer. At present the builder derives measured
overlays automatically only for the three registered external decisions. One
of the 27 profile-tunable decisions may instead change through reviewed profile
JSON, or be supplied explicitly through `build --runtime-decisions` after an
equivalent evidence gate. The latter is a sealed manual projection, not an
AutoTuner-selected winner, until the decision gains a tuning domain and
lowering.

`RuntimePlan.decision_source()` reports `catalog_fallback`, `profile_default`,
`provider_default`, or `engine_plan`. The profile ID and fingerprint are part
of the RuntimePlan fingerprint.

## Changing a default

For an existing decision and normalized context:

1. record the exact source, model, workload, image, hardware, and old profile
   fingerprint;
2. measure the existing default and proposed value under the same workload;
3. change only the profile JSON rule;
4. run profile-schema/security and Qwen/DeepSeek golden-parity tests;
5. run focused semantic and compiled/captured serving A/B gates;
6. rebuild and validate any promoted EnginePlan against the new source;
7. deploy the immutable package and plan, then restart serving.

Do not mount a mutable profile file into a serving process, hot-reload a
decision, or edit a production artifact in place. Source identity and the
profile fingerprint must change with the reviewed default.

Adding a new model that reuses registered decisions and normalized fields can
be a profile-only change. Adding a new decision, consumer, normalized field,
kernel, correctness oracle, or dynamic request/resource guard still requires
code and tests.
