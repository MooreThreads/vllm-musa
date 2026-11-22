# vLLM-MUSA v0.32 image runtime split

## Goal Description
Refactor the v0.32 MUSA Docker build so production `vllm-openai` consumes a bundled wheel from a dedicated runtime stage, while a separate source-bearing test target preserves development and CI workflows. Keep MUSA native extension, patched vLLM, Rust frontend, and entrypoint contracts intact.

## Acceptance Criteria

- AC-1: Build and runtime stages are explicit and production runtime has no editable source checkout.
  - Positive Tests: Docker contract checks find `vllm_wheel`, `vllm_runtime`, `vllm-openai`, and `test` stages and no runtime source COPY.
  - Negative Tests: Runtime stage must not contain `/vllm-workspace/third_party/vllm` source paths.
- AC-2: The package build bundles patched vLLM and runtime package data.
  - Positive Tests: Bundle helper and packaging verification tests pass.
  - Negative Tests: Wheel build must fail if the vendored vLLM source or MUSA JIT data is absent.
- AC-3: Existing image contracts remain covered.
  - Positive Tests: Focused Docker, manifest, and packaging tests pass.
  - Negative Tests: A source target remains available when the production target excludes source.
- AC-4: Remote no-GPU validation is recorded.
  - Positive Tests: Remote Dockerfile parse/build reaches the runtime target when required private indexes are supplied.
  - Negative Tests: No GPU lease or serving claim is made from image-only validation.

## Path Boundaries

### Upper Bound
A complete wheel-installed production image with a source-bearing test target and exact upstream workspace contents.

### Lower Bound
A dedicated wheel/runtime target and tests proving source/runtime separation.

### Allowed Choices
- Use setuptools build hooks and Docker multi-stage targets.
- Reuse the existing v0.32 pinned source and private dependency indexes.
- Keep editable install in the source-bearing target.
- Do not change model serving behavior or dependency pins unrelated to packaging.

## Dependencies and Sequence

1. Adapt the existing bundled-wheel implementation to the clean v0.32 branch.
2. Split Docker build, wheel, runtime, production, and test targets.
3. Run focused local tests and remote no-GPU target validation.
4. Review the branch and publish a draft PR.

## Implementation Notes

Do not include plan terminology in production code.
