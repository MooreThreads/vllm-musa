# DeepSelect on MUSA

vLLM builds the native `vllm._deepselect_C` extension on MP31 through its
existing CMake target. `import vllm._deepselect_C` registers `torch.ops.deep_select` directly
from C++. No Python dispatcher bridge or separately installed DeepSelect
package is required. The MUSA implementation does not fall back to `torch.topk`.

## Source selection

DeepSelect source selection belongs to vLLM's
`cmake/external_projects/deepselect.cmake`. Its upstream `FetchContent`
repository (`https://github.com/vllm-project/DeepSelect.git`) and revision
remain unchanged.
vLLM-MUSA does not maintain a separate DeepSelect pin, download, or extension.

The official repository is a placeholder until the adapted MUSA source is
published. Set `DEEPSELECT_SRC_DIR` to the updated MUSA checkout for current
builds; deployment-specific repository/revision changes belong only in vLLM's
existing `FetchContent` declaration. The explicit source directory takes
precedence over dependency downloads and is not reset or updated by the build.
Use a task-local writable source copy for remote builds, not a shared NFS
checkout as a build tree. Record the revision and dirty state, and rerun the
native contract tests after changes.

The source must include `csrc/musa_api.cpp` and `cmake/vllm_musa.cmake`.
The official repository does not yet provide the required MUSA integration,
so an MP31 build without a MUSA source override fails with an explicit error.

## Build and install

In the prepared MUSA runtime environment, build vLLM-MUSA using the existing
DeepSelect source checkout:

```bash
cd /path/to/vllm-musa-deepselect
DEEPSELECT_SRC_DIR=/path/to/DeepSelect \
  python3 -m pip install -v --no-build-isolation --no-deps -e .
python3 -c 'import torch_musa; import vllm._deepselect_C; import torch; print(torch.ops.deep_select.get_alignment_requirement())'
```

The source checkout must also contain `csrc/musa_kernels/topk_bf16.mu`; a Python
wheel alone is not sufficient for the build. vLLM-MUSA applies its source patch
series, then installs vLLM with `VLLM_TARGET_DEVICE=musa` and passes through the
detected or explicitly configured `MTGPU_TARGET`. vLLM's existing
`cmake_build_ext` compiles and installs `_deepselect_C` using MCC, the PyTorch
C++ ABI, and torch_musa's include/library paths. Other MUSA extensions keep their
existing build path. The patched vLLM extension list includes DeepSelect only
for MP31 (the default when building vLLM directly without `MTGPU_TARGET`).
MP22 remains a MUSA build but does not build or download DeepSelect.

MUSA binding and build details are owned by DeepSelect's `csrc/musa_api.cpp`
and `cmake/vllm_musa.cmake`. vLLM retains extension compilation, installation,
import detection, and the existing `torch.ops.deep_select` call sites.

For an already patched vLLM checkout, the same extension can be built directly:

```bash
VLLM_TARGET_DEVICE=musa DEEPSELECT_SRC_DIR=/path/to/DeepSelect \
  python3 -m pip install -v --no-build-isolation --no-deps -e /path/to/patched/vllm
```

Rebuild vLLM or reinstall vLLM-MUSA after kernel or binding changes, then restart
workers to load the new binary. Reinstalling
the standalone DeepSelect package does not update `vllm._deepselect_C`.
`SKIP_THIRD_PARTY=1` is available when all required source trees are already
prepared. Use the workspace source-cache and ccache procedure for remote builds.

## Runtime selection

Select the backend through the vLLM kernel configuration:

```text
{"sparse_indexer_topk_backend":"deep_select"}
```

Backend validation checks the platform first (MUSA, or CUDA SM100/SM103),
then successful import of `vllm._deepselect_C`, then the input constraints.
Extension loading and import-failure detection follow upstream; there is no
late registration through the general plugin hook.

The extension registers native MUSA top-k, Meta support, and the no-tensor
alignment operator on `CompositeExplicitAutograd`. Its schemas match upstream,
including `output_value`/`output_index` argument names and the `(int, int)`
alignment result. There is no CPU top-k implementation or reference fallback.
Kernel contract tests import the extension directly, require MUSA hardware,
and compare its output against ordinary CPU PyTorch operations.

The backend supports the DSV4 sparse-indexer top-k contract for BF16 and FP32
logits, with 32-byte input and output row-stride alignment. The standard vLLM
path requests indices only and passes `abort_when_nan_found=False`; DeepSelect
still checks for NaNs and writes its marker for invalid rows.

DeepSelect selects over each row's prefix `[0, end)`; its `begin` argument is
not supported. MUSA windowed indexer paths use DeepSelect when all row starts
are zero. Nonzero-start windows retain their existing native or reference
top-k dispatch, including window-local indices and `-1` padding. Windowed
capture paths retain that dispatch when a zero start cannot be established
without reading a device scalar on the host. Prefix-based decode paths can
still use DeepSelect during capture.

For provenance, record the DeepSelect source commit, extension path and hash,
vLLM-MUSA commit, vLLM commit, and the exact serving command with benchmark
artifacts.
