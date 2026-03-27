# vLLM MUSA Platform Patches

This directory contains patches that modify vLLM source files at runtime to ensure compatibility with the MUSA platform.

## Why Patches Are Needed

The MUSA platform uses a modified version of Triton that has some syntax incompatibilities with the standard Triton used by vLLM. These patches fix these incompatibilities at runtime when the MUSA platform plugin is loaded.

## Patch File Naming Convention

Patch files follow this naming convention:

```
{module_path_with_double_underscores}.patch.py
```

For example:
- `vllm__attention__ops__triton_unified_attention.patch.py` patches `vllm.attention.ops.triton_unified_attention`

The double underscores (`__`) in the filename are converted to dots (`.`) to form the target module path.

## Patch File Format

Each patch file should define a `PATCHES` list containing tuples of `(old_string, new_string)`:

```python
PATCHES = [
    ("old code to replace", "new replacement code"),
    # ... more patches
]
```

## Current Patches

### vllm__v1__sample__ops__topk_topp_triton.patch.py

**Target:** `vllm.v1.sample.ops.topk_topp_triton`

**Issue:** MUSA Triton doesn't support Python's annotated assignment syntax (PEP 526)

**Error:**
```
triton.compiler.errors.CompilationError:
    left: tl.int32 = 0
    ^
AttributeError("'AnnAssign' object has no attribute 'targets'")
```

**Fix:** Replace `left: tl.int32 = 0` with `left = 0`

**Affected Function:** `find_seq_idx()` - Binary search helper for the unified attention kernel

### vllm__v1__worker__gpu_worker.patch.py

**Target:** `vllm.v1.worker.gpu_worker`

**vLLM Versions:** 0.10.x and 0.13.x (all V1 engine versions)

**Issue:** The V1 GPU worker only checks for `device.type == "cuda"`, which doesn't match MUSA devices

**Fix:** Extend device type check to also accept "musa" device type

**Note:** No patch is needed for `torch.device("cuda:X")` because torchada automatically aliases it to `torch.device("musa:X")` when imported.


### vllm__distributed__device_communicators__all2all.patch.py

**Target:** `vllm.distributed.device.communicators.all2all`

**Issue:** MUSA's version of the communicator doesn't support the `explicitly_destroy` argument

**Fix:** Remove `explicitly_destroy=True,` parameter from all2all calls

### vllm__distributed__device_communicators__custom_all_reduce.patch.py

**Target:** `vllm.distributed.device.communicators.custom_all_reduce`

**Changes:**
- Increase `CustomAllreduce.max_size` from 8MB to 128MB to better support MUSA
- Add platform check for CUDA-specific MUSA_VISIBLE_DEVICES behavior
- Enable MUSA's custom_allreduce backend

### vllm__model_executor__layers__fused_moe__deep_gemm_moe.patch.py

**Target:** `vllm.model_executor.layers.fused_moe.deep_gemm_moe.DeepGemmExperts`

**Issue:** The `m_grouped_fp8_gemm_nt_contiguous` function requires `a2q_scale` tensor to be contiguous on MUSA

**Fix:** Add `a2q_scale.contiguous()` before calling `m_grouped_fp8_gemm_nt_contiguous`

### vllm__model_executor__layers__quantization__fp8.patch.py

**Target:** `vllm.model_executor.layers.quantization.fp8`

**Changes:**
- Add MUSA to the list of platforms that don't support Marlin backend
- Adjust minimum compute capability from 75 (NVIDIA Hopper) to 31 (MUSA)

### vllm__model_executor__layers__quantization__utils__fp8_utils.patch.py

**Target:** `vllm.model_executor.layers.quantization.utils.fp8_utils`

**Issue:** Per-token quantization requires contiguous tensors on MUSA

**Fix:** Add MUSA platform check alongside CUDA for contiguous tensor requirement

### vllm__profiler__wrapper.patch.py

**Target:** `vllm.profiler.wrapper`

**Changes:**
- Update `TorchProfilerActivity` type literal to include "MUSA"

### vllm__utils__deep_gemm.patch.py

**Target:** `vllm.utils.deep_gemm`

**Changes:**
- Enable DeepGemm support on MUSA devices
- Adjust device capability check from 90 (NVIDIA Hopper) to 31 (MUSA)

### vllm__v1__attention__backends__fa_utils.patch.py

**Target:** `vllm.v1.attention.backends.fa_utils`

**Issue:** Different Flash Attention version requirements for MUSA vs other platforms

**Fix:** Force Flash Attention v2 for MUSA devices

### vllm__v1__attention__backends__mla__flashmla.patch.py

**Target:** `vllm.v1.attention.backends.mla.flashmla`

**Changes:**
- Adjust reorder batch threshold from 128 to 1 for MUSA performance

### vllm__v1__attention__ops__flashmla.patch.py

**Target:** `vllm.v1.attention.ops.flashmla`

**Issue:** Device capability detection incompatible with MUSA architecture

**Fix:** Replace capability family for MUSA

### vllm__v1__sample__ops__topk_topp_sampler.patch.py

**Target:** `vllm.v1.sample.ops.topk_topp_sampler`

**Issue:** Triton-based sampling kernel not compatible with MUSA Triton compiler

**Fix:** Disable Triton path for MUSA and use PyTorch fallback implementation

### vllm__v1__sample__ops__topk_topp_triton.patch.py

**Target:** `vllm.v1.sample.ops.topk_topp_triton`

**Changes:**
- Remove `tl.cast()` calls for intermediate `tl.int32` conversions (MUSA Triton handles this better)
- Refactor conditional logic for `final_pivot` to improve MUSA Triton compatibility

## Version-Specific Patches

Some patches are version-specific and will be automatically skipped if the target module doesn't exist:

| Patch | vLLM 0.17.0x |
|-------|-------------|
| `vllm__v1__sample__ops__topk_topp_triton` | ✅ Applied |
| `vllm__v1__worker__gpu_worker` | ✅ Applied |
| `vllm__distributed__device_communicators__all2all` | ✅ Applied |
| `vllm__distributed__device_communicators__custom_all_reduce` | ✅ Applied |
| `vllm__model_executor__layers__quantization__fp8` | ✅ Applied |
| `vllm__model_executor__layers__quantization__utils__fp8_utils` | ✅ Applied |
| `vllm__profiler__wrapper` | ✅ Applied |
| `vllm__utils__deep_gemm` | ✅ Applied |
| `vllm__v1__attention__backends__fa_utils` | ✅ Applied |
| `vllm__v1__attention__backends__mla__flashmla` | ✅ Applied |
| `vllm__v1__attention__ops__flashmla` | ✅ Applied |
| `vllm__v1__sample__ops__topk_topp_sampler` | ✅ Applied |
| `vllm__v1__sample__ops__topk_topp_triton` | ✅ Applied |


When a patch is skipped due to a missing module, a debug message is logged (not a warning), as this is expected behavior for version-specific patches.

## How Patches Are Applied

1. When the MUSA platform plugin is loaded, it calls `apply_patches()` from this module
2. The function scans for all `*.patch.py` files in this directory
3. For each patch file:
   - Extracts the target module name from the filename
   - Loads the `PATCHES` list from the patch file
   - Reads the target module's source file
   - Applies string replacements
   - Writes the patched source back to disk
   - Clears the module from `sys.modules` to force a fresh import

## Adding New Patches

1. Create a new file named `{module__path}.patch.py`
2. Add documentation explaining the issue and solution
3. Define the `PATCHES` list with your replacements
4. Test that the patch is applied correctly

Example:
```python
# vllm__some__module.patch.py
"""
Patch for vllm.some.module

Issue: Description of the problem
Solution: Description of the fix
"""

PATCHES = [
    ("problematic code", "fixed code"),
]
```

## Notes

- Patches modify files on disk, so they persist across Python sessions
- Patches are only applied once per module (tracked by `_patches_applied` flag)
- If a patch has already been applied (old string not found), it's skipped silently
- Failed patches log a warning but don't prevent the platform from loading
