# SPDX-License-Identifier: Apache-2.0
"""Source contracts for the DeepSeek-V4 SwiGLU FP8 fusion."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_native_kernel_keeps_cuda_spelling_for_torchada() -> None:
    if not (ROOT / ".git").exists():
        pytest.skip("pre-build source contract requires a Git worktree")
    kernel = (
        ROOT
        / "csrc"
        / "musa"
        / "quantization"
        / "silu_and_mul_per_token_group_fp8_quant.cu"
    ).read_text()

    # setup.py imports torchada before CUDAExtension, so keep CUDA-compatible
    # spellings here and let torchada map them at build time.  This matches the
    # other .cu quantization sources and avoids direct torch_musa header ties.
    assert "#include <ATen/cuda/CUDAContext.h>" in kernel
    assert "#include <cuda_fp8.h>" in kernel
    assert "cudaStream_t stream = at::cuda::getCurrentCUDAStream();" in kernel
    assert "LAUNCH_SILU_QUANT_KERNEL(scalar_t, __nv_fp8_e4m3);" in kernel
    for direct_musa_spelling in (
        '"torch_musa/csrc/aten/musa/MUSAContext.h"',
        "<musa_fp8.h>",
        "musaStream_t stream = at::musa::getCurrentMUSAStream();",
        "LAUNCH_SILU_QUANT_KERNEL(scalar_t, __mt_fp8_e4m3);",
    ):
        assert direct_musa_spelling not in kernel


def test_native_kernel_has_a_separate_clamp_aware_entrypoint() -> None:
    kernel = (
        ROOT
        / "csrc"
        / "musa"
        / "quantization"
        / "silu_and_mul_per_token_group_fp8_quant.cu"
    ).read_text()
    bindings = (ROOT / "csrc" / "musa" / "torch_bindings.cpp").read_text()
    header = (ROOT / "csrc" / "musa" / "musa_ops.h").read_text()

    assert (
        "template <typename T, typename DST_DTYPE, int GROUP_SIZE, bool APPLY_CLAMP>"
        in kernel
    )
    assert "if constexpr (APPLY_CLAMP)" in kernel
    assert "const T sigmoid_rounded" in kernel
    assert "const T silu_rounded" in kernel
    assert (
        "rounded = static_cast<T>(static_cast<float>(silu_rounded) * up);" in kernel
    )
    assert "silu_and_mul_clamp_per_token_group_fp8_quant" in kernel
    assert "silu_and_mul_clamp_per_token_group_fp8_quant" in bindings
    assert "silu_and_mul_clamp_per_token_group_fp8_quant" in header


def test_deepgemm_custom_op_uses_clamp_aware_quant_before_gemm() -> None:
    source = (
        ROOT
        / "vllm_musa"
        / "model_executor"
        / "kernels"
        / "linear"
        / "scaled_mm"
        / "deep_gemm.py"
    ).read_text()

    start = source.index("def _musa_silu_clamp_deepgemm_fp8_op(")
    end = source.index("def _musa_fused_add_rms_deepgemm_fp8_op(", start)
    block = source[start:end]
    assert "silu_and_mul_clamp_per_token_group_fp8_quant(" in block
    assert "fp8_gemm_nt(" in block
    assert "swiglu_limit > 0.0" in block
    assert '"musa_silu_clamp_deepgemm_fp8_op"' in source
    assert "fake_impl=_musa_silu_clamp_deepgemm_fp8_op_fake" in source


def test_row_parallel_dispatch_is_narrow_and_preserves_tp_semantics() -> None:
    source = (
        ROOT / "vllm_musa" / "model_executor" / "layers" / "linear.py"
    ).read_text()

    start = source.index("    def forward_swiglu_clamp(")
    end = source.index("    def forward(self, input_, out=None):", start)
    block = source[start:end]

    # The optional model hook must fall back for every unsupported linear path.
    assert "if not fast:\n            return None" in block
    for guard in (
        "self.input_is_parallel",
        "_deepgemm_block_fp8(self.quant_method)",
        "self.bias is None",
        "gate_up.dtype == torch.bfloat16",
        "gate_up.is_contiguous()",
        '== (128, 128)',
        "swiglu_limit == 10.0",
    ):
        assert guard in block

    # Serialized DeepSeek block-FP8 checkpoints expose inverse scales.  Keep a
    # compatibility fallback for ordinary weight_scale, but never prefer it.
    inv_pos = block.index('getattr(self, "weight_scale_inv", None)')
    scale_pos = block.index('getattr(self, "weight_scale", None)')
    op_pos = block.index("musa_silu_clamp_deepgemm_fp8_op(")
    assert inv_pos < scale_pos < op_pos

    # RowParallelLinear owns the TP reduction.  The fused hook must preserve
    # the original reduce_results contract rather than returning a local shard.
    assert "if self.reduce_results and self.tp_size > 1:" in block
    assert "tensor_model_parallel_all_reduce(output_parallel)" in block


def test_deepseek_model_patch_uses_optional_hook_before_activation() -> None:
    patch = (
        ROOT
        / "vllm_musa"
        / "patches"
        / "series"
        / "0094-MUSA-DeepSeek-V4-fuse-clamp-SwiGLU-FP8-down-proj.patch"
    ).read_text()

    assert "isinstance(self.act_fn, SiluAndMulWithClamp)" in patch
    hook_pos = patch.index('getattr(self.down_proj, "forward_swiglu_clamp", None)')
    activation_pos = patch.index("x = self.act_fn(gate_up)")
    assert hook_pos < activation_pos
    assert "if fused_output is not None:" in patch
    assert "return fused_output" in patch
