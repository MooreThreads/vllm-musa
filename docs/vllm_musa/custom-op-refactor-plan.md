# vLLM-MUSA operator integration refactor

## Goal Description

Make vllm-musa v0.32 a maintainable vLLM OOT plugin with explicit ownership and
provider boundaries. MATE remains the reusable MUSA provider for attention and
common libraries. vllm-musa owns vLLM contract integration, non-attention
providers, IR lowerings, capability selection, and model/layer assembly.

## Acceptance Criteria

- AC-1: Every integrated operator/backend has one machine-readable record naming its vLLM contract, provider, registration module, source/build entry, capability, fallback, owner, and tests. The record also carries the adapter or extension/JIT module and minimum package-version constraints when applicable.
  - Positive: inventory validation reports no missing required fields or duplicate contract/provider records.
  - Negative: a record with a missing provider, source, or test is rejected.
- AC-2: Preload and registration are generated from the explicit manifest while preserving AOT compile-cache preload ordering.
  - Positive: installed plugin imports the generated modules and reports registration status.
  - Negative: an unavailable optional provider reports a structured skip or fallback instead of an import traceback during plugin discovery.
- AC-3: Attention integration calls MATE through vllm-musa adapters and does not duplicate MATE kernel implementations.
  - Positive: backend tests cover capability selection, metadata conversion, and fallback.
  - Negative: direct new model-layer imports of MATE or unclassified `_custom_ops` fail the source contract check.
- AC-4: IR, vLLM AOT/JIT, native MUSA, MATE adapter, and model-private paths are separately discoverable and testable.
  - Positive: each category has an inventory slice and focused source/import tests.
  - Negative: category ambiguity is reported for unclassified registrations.
- AC-5: The refactor preserves the v0.32 patch-series and plugin entry-point contracts.
  - Positive: patch-series source checks and Python unit tests pass.
  - Negative: generated manifest drift or patch metadata drift fails CI.
- AC-6: Readiness evidence distinguishes source checks, build checks, MUSA correctness, model smoke, and performance.
  - Positive: PR checklist records exact commands and explicit not-run gates.
  - Negative: source-only checks cannot mark hardware or serving gates passed.

## Path Boundaries

### Upper Bound

Manifest-driven registration, generated preload/diagnostics, attention adapters,
category-specific tests and CI checks, followed by migration of existing entries.

### Lower Bound

A validated inventory and explicit manifest schema with one migrated representative
from each category, without changing default provider selection.

### Allowed Choices

Use existing Python registration, Torch library, vLLM IR, patch-series, and MATE
interfaces. Keep behavior-compatible shims during migration.

Do not duplicate MATE kernels, rewrite the patch series, or create a second
all-purpose MUSA kernel repository in this PR.

## Dependencies and Sequence

1. Inventory current registrations, providers, adapters, call sites, source/build entries, and tests.
2. Add schema and validator with representative records.
3. Generate preload and diagnostics, preserving current entry-point behavior. The dependency-free `tools/operator_manifest.py` command is the source/CI diagnostic entry point.
4. Add attention adapter boundaries and source contracts.
5. Migrate categories incrementally and add CI gates.
6. Run source, package, MUSA correctness, model, and performance gates separately.

## Deferred

Independent repository extraction is deferred until a non-attention provider has
stable ABI/schema, independent CI, and multiple consumers without vLLM private
dependencies.
