#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Group discovered kernel entrypoints into a public source-only catalog.

Qualification overlays (device bins, timings, receipts, and promotion
decisions) are local generated evidence and must not be embedded here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INVENTORY_PATH = SCRIPT_DIR / "inventory_vllm_musa_kernels.py"
SPEC = importlib.util.spec_from_file_location(
    "inventory_vllm_musa_kernels", INVENTORY_PATH
)
assert SPEC is not None and SPEC.loader is not None
INVENTORY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = INVENTORY
SPEC.loader.exec_module(INVENTORY)

PHYSICAL_INVENTORY_PATH = SCRIPT_DIR / "inventory_vllm_musa_physical_kernels.py"
PHYSICAL_SPEC = importlib.util.spec_from_file_location(
    "inventory_vllm_musa_physical_kernels", PHYSICAL_INVENTORY_PATH
)
assert PHYSICAL_SPEC is not None and PHYSICAL_SPEC.loader is not None
PHYSICAL_INVENTORY = importlib.util.module_from_spec(PHYSICAL_SPEC)
sys.modules[PHYSICAL_SPEC.name] = PHYSICAL_INVENTORY
PHYSICAL_SPEC.loader.exec_module(PHYSICAL_INVENTORY)

SCHEMA = "vllm-musa-public-kernel-catalog.v1"

_FAMILY_METADATA: dict[str, dict[str, object]] = {
    "aot-native-deepseek-v4-c4-indexer-compress-cache": {
        "disposition": "no-tunable-seam",
        "production_consumers": [
            "vllm_musa/patches/series/0093-MUSA-dispatch-DeepSeek-V4-C4-indexer-compression-to-.patch",
            "vllm_musa/_custom_ops.py",
        ],
    },
    "aot-native-deepseek-v4-combine-topk-swa-indices": {
        "disposition": "non-production",
        "notes": "Registered local wrapper has no caller in the v0.28 patch series.",
    },
    "aot-native-deepseek-v4-compute-global-topk-indices-and-lens": {
        "disposition": "non-production",
        "notes": "Registered local wrapper has no caller in the v0.28 patch series.",
    },
    "aot-native-deepseek-v4-dequantize-and-gather-k-cache": {
        "disposition": "non-production",
        "notes": "Registered local wrapper has no caller in the v0.28 patch series.",
    },
    "aot-native-deepseek-v4-fused-inv-rope-fp8-quant": {
        "disposition": "no-tunable-seam",
        "notes": "The fused inverse-rope/FP8 path has fixed launch geometry and no benchmark-safe per-call tile argument.",
        "production_consumers": [
            "vllm_musa/patches/series/0014-MUSA-vllm.models.deepseek_v4.attention.patch",
            "vllm_musa/_custom_ops.py",
        ],
    },
    "aot-native-deepseek-v4-fused-q-kv-rmsnorm": {
        "disposition": "non-production",
        "notes": "Current DSV4 path uses qnorm/rope/KV insert instead.",
    },
    "aot-native-deepseek-v4-indexer-rerank-prefill": {
        "disposition": "no-tunable-seam",
        "notes": "Prefill rerank launch is fixed by the semantic indexer shape.",
        "production_consumers": [
            "vllm_musa/patches/series/0013-MUSA-vllm.model_executor.layers.sparse_attn_indexer.patch",
            "vllm_musa/_custom_ops.py",
        ],
    },
    "aot-native-deepseek-v4-indexer-topk-decode": {
        "disposition": "no-tunable-seam",
        "notes": "Decode top-k launch is fixed by the indexer contract.",
        "production_consumers": [
            "vllm_musa/patches/series/0013-MUSA-vllm.model_executor.layers.sparse_attn_indexer.patch",
            "vllm_musa/_custom_ops.py",
        ],
    },
    "aot-native-deepseek-v4-indexer-topk-prefill": {
        "disposition": "no-tunable-seam",
        "notes": "Prefill top-k launch is fixed by the indexer contract.",
        "production_consumers": [
            "vllm_musa/patches/series/0013-MUSA-vllm.model_executor.layers.sparse_attn_indexer.patch",
            "vllm_musa/_custom_ops.py",
        ],
    },
    "aot-native-deepseek-v4-mhc-pre": {
        "disposition": "no-tunable-seam",
        "production_consumers": [
            "vllm_musa/deepseek_v4_mhc.py",
            "vllm_musa/_custom_ops.py",
        ],
    },
    "aot-native-deepseek-v4-qnorm-rope-kv-insert": {
        "disposition": "no-tunable-seam",
        "notes": "QNorm/rope/KV insertion is a fixed one-token path; changing its launch shape requires a new correctness and graph-capture design.",
        "production_consumers": [
            "vllm_musa/patches/series/0014-MUSA-vllm.models.deepseek_v4.attention.patch",
            "vllm_musa/_custom_ops.py",
        ],
    },
    "aot-native-deepseek-v4-sparse-flashmla-decode": {
        "disposition": "external-provider-excluded",
        "notes": "Production sparse FlashMLA is provided by MATE and excluded.",
    },
    "aot-native-deepseek-v4-store-sparse-kv": {
        "disposition": "non-production",
        "notes": "Current DSV4 path uses fused qnorm/rope/KV insert instead.",
    },
    "aot-native-deepseek-v4-topk-softplus-sqrt": {
        "disposition": "non-production",
        "notes": "No v0.28 production caller was found.",
    },
    "aot-native-fused-add-rms-norm-per-token-group-fp8-quant": {
        "benchmark": "benchmarks/kernel_tactics/benchmark_fp8_quant_groups.py",
        "disposition": "no-tunable-seam",
        "notes": "The production kernel is fixed to H=4096, group size 128, and kThreads=512. The catalog's former block_threads/vector_width labels were hypotheses, not a real ABI or benchmark seam.",
        "production_consumers": [
            "vllm_musa/model_executor/kernels/linear/scaled_mm/deep_gemm.py"
        ],
        "shape_sources": [
            "benchmarks/kernel_tactics/qwen_dsv4_rmsnorm_shapes.json",
            "benchmarks/kernel_tactics/qwen_dsv4_fp8_quant_shapes.json",
        ],
    },
    "aot-native-glm52-indexer-topk-decode": {
        "disposition": "no-tunable-seam",
        "notes": "GLM52 decode top-k launch is fixed by the semantic contract.",
        "production_consumers": [
            "vllm_musa/patches/series/0085-MUSA-vllm.v1.attention.backends.mla.indexer.patch",
            "vllm_musa/_custom_ops.py",
        ],
    },
    "aot-native-glm52-indexer-topk-prefill": {
        "disposition": "no-tunable-seam",
        "notes": "GLM52 prefill top-k launch is fixed by the semantic contract.",
        "production_consumers": [
            "vllm_musa/patches/series/0085-MUSA-vllm.v1.attention.backends.mla.indexer.patch",
            "vllm_musa/_custom_ops.py",
        ],
    },
    "aot-native-min-p-sampling-from-probs": {
        "disposition": "external-provider-excluded"
    },
    "aot-native-musa-chunked-min-p-sampling-from-probs": {
        "disposition": "no-tunable-seam",
        "production_consumers": [
            "vllm_musa/_custom_ops.py",
            "vllm_musa/v1/sample/topk_topp_sampler.py",
        ],
    },
    "aot-native-musa-fused-add-rms-norm": {
        "benchmark": "benchmarks/kernel_tactics/benchmark_fused_add_rmsnorm_paired_ab.py",
        "production_consumers": [
            "vllm_musa/_custom_ops.py",
            "vllm_musa/kernels/musa_ops.py",
            "vllm_musa/model_executor/layers/layernorm.py",
        ],
        "tunable_parameters": ["block_x"],
    },
    "aot-native-musa-fused-gemv": {
        "benchmark": "benchmarks/kernel_tactics/benchmark_dense_fp8_gemv_blocks.py",
        "production_consumers": [
            "vllm_musa/model_executor/kernels/linear/scaled_mm/deep_gemm.py",
            "vllm_musa/deepseek_v4_jit/fp8_einsum.py",
        ],
        "shape_sources": ["benchmarks/kernel_tactics/mp_tactic_campaign.json"],
        "tunable_parameters": ["block_n", "block_k"],
    },
    "aot-native-musa-fused-gemv-moe": {
        "benchmark": "benchmarks/kernel_tactics/benchmark_fused_gemv_moe_blocks.py",
        "production_consumers": [
            "vllm_musa/model_executor/layers/fused_moe/fused_moe.py",
            "vllm_musa/model_executor/layers/fused_moe/dispatch_policy.py",
        ],
        "shape_sources": ["benchmarks/kernel_tactics/mp_tactic_campaign.json"],
        "tunable_parameters": ["w1_block_n", "w1_block_k", "w2_block_n", "w2_block_k"],
    },
    "aot-native-musa-reshape-and-cache-flash-nhd": {
        "benchmark": "benchmarks/kernel_tactics/benchmark_reshape_cache_nhd_blocks.py",
        "production_consumers": ["vllm_musa/v1/attention/backends/fa_utils.py"],
        "shape_sources": [
            "benchmarks/kernel_tactics/benchmark_reshape_cache_nhd_blocks.py::NHD_SHAPES"
        ],
        "tunable_parameters": ["block_x"],
    },
    "aot-native-musa-rubymine-top-k-renorm-probs": {
        "disposition": "no-tunable-seam",
        "production_consumers": [
            "vllm_musa/_custom_ops.py",
            "vllm_musa/v1/sample/topk_topp_sampler.py",
        ],
    },
    "aot-native-musa-top-k-top-p-sampling-from-probs": {
        "disposition": "no-tunable-seam",
        "production_consumers": [
            "vllm_musa/_custom_ops.py",
            "vllm_musa/v1/sample/topk_topp_sampler.py",
        ],
    },
    "aot-native-per-token-group-quant-8bit-vec": {
        "benchmark": "benchmarks/kernel_tactics/benchmark_fp8_quant_groups.py",
        "production_consumers": [
            "vllm_musa/model_executor/layers/quantization/utils/fp8_utils.py"
        ],
        "shape_sources": [
            "benchmarks/kernel_tactics/qwen_dsv4_fp8_quant_shapes.json",
            "benchmarks/kernel_tactics/mp_tactic_campaign.json",
        ],
        "tunable_parameters": ["groups_per_block"],
    },
    "aot-native-silu-and-mul-clamp-per-token-group-fp8-quant": {
        "benchmark": "benchmarks/kernel_tactics/benchmark_fp8_quant_groups.py",
        "production_consumers": [
            "vllm_musa/model_executor/kernels/linear/scaled_mm/deep_gemm.py"
        ],
        "shape_sources": [
            "benchmarks/kernel_tactics/qwen_dsv4_fp8_quant_shapes.json",
            "benchmarks/kernel_tactics/mp_tactic_campaign.json",
        ],
        "tunable_parameters": ["groups_per_block"],
    },
    "aot-native-silu-and-mul-per-token-group-fp8-quant": {
        "benchmark": "benchmarks/kernel_tactics/benchmark_fp8_quant_groups.py",
        "production_consumers": [
            "vllm_musa/model_executor/kernels/linear/scaled_mm/deep_gemm.py",
            "vllm_musa/model_executor/layers/quantization/utils/fp8_utils.py",
        ],
        "shape_sources": [
            "benchmarks/kernel_tactics/qwen_dsv4_fp8_quant_shapes.json",
            "benchmarks/kernel_tactics/mp_tactic_campaign.json",
        ],
        "tunable_parameters": ["groups_per_block"],
    },
    "aot-native-sparse-indexer-fill-all": {
        "disposition": "no-tunable-seam",
        "production_consumers": [
            "vllm_musa/patches/series/0013-MUSA-vllm.model_executor.layers.sparse_attn_indexer.patch",
            "vllm_musa/_custom_ops.py",
        ],
    },
    "aot-native-sparse-indexer-topk": {
        "disposition": "no-tunable-seam",
        "notes": "Sparse-indexer top-k geometry is fixed by the index format.",
        "production_consumers": [
            "vllm_musa/patches/series/0013-MUSA-vllm.model_executor.layers.sparse_attn_indexer.patch",
            "vllm_musa/_custom_ops.py",
        ],
    },
    "aot-native-sparse-indexer-topk-decode": {
        "disposition": "no-tunable-seam",
        "notes": "Sparse-indexer decode geometry is fixed by the index format.",
        "production_consumers": [
            "vllm_musa/patches/series/0013-MUSA-vllm.model_executor.layers.sparse_attn_indexer.patch",
            "vllm_musa/_custom_ops.py",
        ],
    },
    "aot-native-top-k-renorm-probs": {"disposition": "external-provider-excluded"},
    "aot-native-top-p-renorm-probs": {"disposition": "external-provider-excluded"},
    "aot-native-top-p-sampling-from-probs": {
        "disposition": "external-provider-excluded"
    },
    "jit-native-custom-allreduce": {
        "disposition": "no-tunable-seam",
        "production_consumers": [
            "vllm_musa/distributed/device_communicators/custom_all_reduce.py"
        ],
    },
    "jit-native-fused-add-rmsnorm": {
        "benchmark": "benchmarks/kernel_tactics/benchmark_jit_rmsnorm_threads.py",
        "production_consumers": ["vllm_musa/model_executor/layers/layernorm.py"],
        "shape_sources": ["benchmarks/kernel_tactics/qwen_dsv4_rmsnorm_shapes.json"],
        "tunable_parameters": ["block_threads"],
    },
    "jit-native-fused-qk-rmsnorm-mrope": {
        "disposition": "no-tunable-seam",
        "production_consumers": ["vllm_musa/jit_kernel/csrc/fused_qk_rmsnorm_mrope.py"],
    },
    "jit-native-moe-act-and-mul": {
        "disposition": "no-tunable-seam",
        "production_consumers": [
            "vllm_musa/jit_kernel/csrc/moe/moe_act_and_mul.py",
            "vllm_musa/model_executor/layers/fused_moe/fused_moe.py",
        ],
    },
    "jit-native-moe-sum": {
        "disposition": "no-tunable-seam",
        "notes": "The MoE sum launch geometry is fixed by the current FFI wrapper.",
        "production_consumers": [
            "vllm_musa/jit_kernel/csrc/moe/moe_sum.py",
            "vllm_musa/model_executor/layers/fused_moe/fused_moe.py",
        ],
    },
    "jit-native-per-token-group-quant": {
        "disposition": "no-tunable-seam",
        "production_consumers": [
            "vllm_musa/jit_kernel/csrc/quantization/quant_per_token_group.py"
        ],
    },
    "jit-native-rmsnorm": {
        "benchmark": "benchmarks/kernel_tactics/benchmark_jit_rmsnorm_threads.py",
        "production_consumers": ["vllm_musa/model_executor/layers/layernorm.py"],
        "shape_sources": ["benchmarks/kernel_tactics/qwen_dsv4_rmsnorm_shapes.json"],
        "tunable_parameters": ["block_threads"],
    },
    "jit-native-rotary-embedding": {
        "disposition": "no-tunable-seam",
        "notes": "Rotary launch geometry is fixed by the current production ABI.",
        "production_consumers": ["vllm_musa/jit_kernel/csrc/rotary_embedding.py"],
    },
    "jit-native-topk-sigmoid": {
        "benchmark": "benchmarks/kernel_tactics/benchmark_jit_topk_warps.py",
        "production_consumers": [
            "vllm_musa/model_executor/layers/fused_moe/router/grouped_topk_router.py"
        ],
        "shape_sources": [
            "benchmarks/kernel_tactics/qwen_dsv4_fp8_quant_shapes.json",
            "benchmarks/kernel_tactics/mp_tactic_campaign.json",
        ],
        "tunable_parameters": ["warps_per_cta"],
    },
    "jit-native-topk-softmax": {
        "benchmark": "benchmarks/kernel_tactics/benchmark_jit_topk_warps.py",
        "production_consumers": [
            "vllm_musa/model_executor/layers/fused_moe/router/grouped_topk_router.py"
        ],
        "shape_sources": [
            "benchmarks/kernel_tactics/qwen_dsv4_fp8_quant_shapes.json",
            "benchmarks/kernel_tactics/mp_tactic_campaign.json",
        ],
        "tunable_parameters": ["warps_per_cta"],
    },
    "jit-tilelang-causal-conv1d-decode-width4-batched-kernel": {
        "disposition": "no-tunable-seam",
        "notes": "Width-4 batched decode uses a fixed TileLang schedule.",
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/causal_conv1d.py",
            "vllm_musa/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        ],
    },
    "jit-tilelang-causal-conv1d-fwd-kernel": {
        "disposition": "no-tunable-seam",
        "notes": "Forward causal-conv uses a fixed TileLang schedule.",
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/causal_conv1d.py",
            "vllm_musa/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        ],
    },
    "jit-tilelang-causal-conv1d-fwd-width4-vec-kernel": {
        "disposition": "no-tunable-seam",
        "notes": "Width-4 vector forward uses a fixed TileLang schedule.",
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/causal_conv1d.py",
            "vllm_musa/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        ],
    },
    "jit-tilelang-causal-conv1d-prefill-width4": {
        "disposition": "no-tunable-seam",
        "notes": "Width-4 prefill uses a fixed TileLang schedule.",
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/causal_conv1d.py",
            "vllm_musa/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        ],
    },
    "jit-tilelang-deepgemm-contig-assign-compact": {
        "disposition": "no-tunable-seam",
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/deep_gemm_contig_preprocess.py",
            "vllm_musa/model_executor/kernels/linear/scaled_mm/deep_gemm.py",
        ],
        "shape_sources": ["benchmarks/kernel_tactics/mp_tactic_campaign.json"],
    },
    "jit-tilelang-deepgemm-contig-clear-fill": {
        "disposition": "no-tunable-seam",
        "notes": "Clear/fill preprocessing uses a fixed shape-derived config.",
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/deep_gemm_contig_preprocess.py",
            "vllm_musa/model_executor/kernels/linear/scaled_mm/deep_gemm.py",
        ],
        "shape_sources": ["benchmarks/kernel_tactics/mp_tactic_campaign.json"],
    },
    "jit-tilelang-deepgemm-contig-count-prefix": {
        "disposition": "no-tunable-seam",
        "notes": "Count/prefix preprocessing uses a fixed shape-derived config.",
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/deep_gemm_contig_preprocess.py",
            "vllm_musa/model_executor/kernels/linear/scaled_mm/deep_gemm.py",
        ],
        "shape_sources": ["benchmarks/kernel_tactics/mp_tactic_campaign.json"],
    },
    "jit-tilelang-deepgemm-contig-scan-tree": {
        "disposition": "no-tunable-seam",
        "notes": "Tree scan preprocessing uses a fixed shape-derived config.",
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/deep_gemm_contig_preprocess.py",
            "vllm_musa/model_executor/kernels/linear/scaled_mm/deep_gemm.py",
        ],
        "shape_sources": ["benchmarks/kernel_tactics/mp_tactic_campaign.json"],
    },
    "jit-tilelang-dsv4-mhc-weighted-rmsnorm": {
        "production_consumers": [
            "vllm_musa/tuning.py",
            "vllm_musa/deepseek_v4_jit/tilelang_kernels.py",
        ],
        "tunable_parameters": ["threads"],
    },
    "jit-tilelang-fused-zba-kernel": {
        "disposition": "no-tunable-seam",
        "notes": "The GDN projection fusion has one fixed TileLang schedule.",
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/gdn_fused_proj.py",
            "vllm_musa/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        ],
    },
    "jit-tilelang-grouped-topk": {
        "disposition": "no-tunable-seam",
        "notes": "Only a semantic serial/parallel crossover exists; geometry is fixed at 32 threads.",
    },
    "jit-tilelang-kv-rope-pack-kernel": {
        "disposition": "non-production",
        "notes": "Only an orphan TileLang helper calls this factory; production uses AOT.",
    },
    "jit-tilelang-mhc-fused-post-prenorm-kernel": {
        "disposition": "no-tunable-seam",
        "production_consumers": ["vllm_musa/deepseek_v4_mhc.py"],
    },
    "jit-tilelang-mhc-post-kernel": {
        "production_consumers": ["vllm_musa/deepseek_v4_mhc.py"]
    },
    "jit-tilelang-mhc-pre-split-sinkhorn-kernel": {
        "disposition": "non-production",
        "notes": "The current auto selector never selects this fallback provider.",
    },
    "jit-tilelang-mhc-prenorm-splitk-x-tme-cast-kernel": {
        "disposition": "non-production",
        "notes": "The current auto selector always selects DeepGEMM for this stage.",
    },
    "jit-tilelang-musa-sparse-attention-fwd-kernel-v1": {
        "disposition": "no-tunable-seam",
        "production_consumers": ["vllm_musa/v1/attention/ops/sparse_mla_tilelang.py"],
    },
    "jit-tilelang-qnorm-rope-kernel": {
        "disposition": "non-production",
        "notes": "Only an orphan TileLang helper calls this factory; production uses AOT.",
    },
    "jit-tilelang-rmsnorm-gated": {
        "disposition": "no-tunable-seam",
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/layernorm_gated.py",
            "vllm_musa/model_executor/layers/layernorm.py",
        ],
    },
    "jit-triton-build-qwen-single-request-fa3-metadata-kernel": {
        "disposition": "no-tunable-seam"
    },
    "jit-triton-eagle-prepare-next-token-padded-kernel": {
        "disposition": "no-tunable-seam",
        "notes": "Block size is the semantic next-power-of-two token capacity.",
    },
    "jit-triton-extend-topk-with-shared-kernel": {
        "disposition": "no-tunable-seam",
        "notes": "Block size is fixed by the top-k semantic capacity.",
    },
    "jit-triton-fused-gdn-gating-kernel": {
        "disposition": "no-tunable-seam",
        "production_consumers": [
            "vllm_musa/jit_kernel/fused_gdn_gating.py",
            "vllm_musa/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        ],
    },
    "jit-triton-gated-qk-norm-rope-token-kernel": {
        "disposition": "no-tunable-seam",
        "notes": "The token kernel uses a fixed Triton schedule per shape.",
        "production_consumers": ["vllm_musa/kernels/gated_qkv.py"],
    },
    "jit-triton-post-reorder-triton-kernel": {
        "disposition": "no-tunable-seam",
        "notes": "Post-reorder uses a fixed Triton schedule per shape.",
        "production_consumers": [
            "vllm_musa/jit_kernel/post_reorder.py",
            "vllm_musa/model_executor/layers/fused_moe/fused_moe.py",
        ],
    },
    "jit-triton-qwen2-rope-kv-cache-kernel": {"disposition": "no-tunable-seam"},
}

# Keep the checked-in catalog a source inventory.  Qualification overlays are
# intentionally not valid metadata here; they belong in local generated
# evidence and ticket attachments.
_PUBLIC_METADATA_KEYS = frozenset(
    {
        "benchmark",
        "notes",
        "production_consumers",
        "shape_sources",
        "tunable_parameters",
        "disposition",
    }
)

# These owned production kernels have compile-time or shape-derived launch
# geometry, but no benchmark-safe per-call MP override in the current ABI.  A
# future campaign may add seams for them; until then they are explicitly
# classified instead of sitting in ``pending`` as if a runnable sweep existed.
_FIXED_GEOMETRY_FAMILIES: dict[str, dict[str, object]] = {
    "jit-tilelang-mhc-fused-post-prenorm-kernel": {
        "production_consumers": [
            "vllm_musa/deepseek_v4_mhc.py",
        ],
        "notes": (
            "The factory accepts compile-time threads/tile/split parameters, "
            "but the production caller hardcodes them and exposes no per-call "
            "active-MP selector."
        ),
    },
    "jit-native-custom-allreduce": {
        "production_consumers": [
            "vllm_musa/distributed/device_communicators/custom_all_reduce.py",
        ],
        "notes": (
            "Peer count and payload size select the collective path, but the "
            "vLLM-MUSA wrapper exposes no active-MP block/thread override."
        ),
    },
    "jit-native-fused-qk-rmsnorm-mrope": {
        "production_consumers": [
            "vllm_musa/jit_kernel/csrc/fused_qk_rmsnorm_mrope.py",
        ],
        "notes": "The production FFI wrapper compiles one fixed launch contract.",
    },
    "jit-native-moe-act-and-mul": {
        "production_consumers": [
            "vllm_musa/jit_kernel/csrc/moe/moe_act_and_mul.py",
            "vllm_musa/model_executor/layers/fused_moe/fused_moe.py",
        ],
        "notes": "The activation fusion has no per-call MP launch selector.",
    },
    "jit-native-moe-sum": {
        "production_consumers": [
            "vllm_musa/jit_kernel/csrc/moe/moe_sum.py",
            "vllm_musa/model_executor/layers/fused_moe/fused_moe.py",
        ],
        "notes": "The MoE sum launch geometry is fixed by the current FFI wrapper.",
    },
    "jit-native-per-token-group-quant": {
        "production_consumers": [
            "vllm_musa/jit_kernel/csrc/quantization/quant_per_token_group.py",
        ],
        "notes": "The JIT quant wrapper exposes no groups-per-block MP seam.",
    },
    "jit-native-rotary-embedding": {
        "production_consumers": [
            "vllm_musa/jit_kernel/csrc/rotary_embedding.py",
        ],
        "notes": "Rotary launch geometry is fixed by the current production ABI.",
    },
    "jit-tilelang-causal-conv1d-channels-first-kernel": {
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/causal_conv1d.py",
            "vllm_musa/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        ],
        "notes": "Channels-first causal-conv uses a fixed TileLang schedule.",
    },
    "jit-tilelang-causal-conv1d-decode-kernel": {
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/causal_conv1d.py",
            "vllm_musa/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        ],
        "notes": "Decode causal-conv uses a fixed TileLang schedule.",
    },
    "jit-tilelang-causal-conv1d-fwd-kernel": {
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/causal_conv1d.py",
            "vllm_musa/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        ],
        "notes": "Forward causal-conv uses a fixed TileLang schedule.",
    },
    "jit-tilelang-causal-conv1d-update-kernel": {
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/causal_conv1d.py",
            "vllm_musa/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        ],
        "notes": "State-update causal-conv uses a fixed TileLang schedule.",
    },
    "jit-tilelang-causal-conv1d-decode-width4-batched-kernel": {
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/causal_conv1d.py",
            "vllm_musa/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        ],
        "notes": "Width-4 batched decode uses a fixed TileLang schedule.",
    },
    "jit-tilelang-causal-conv1d-fwd-width4-vec-kernel": {
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/causal_conv1d.py",
            "vllm_musa/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        ],
        "notes": "Width-4 vector forward uses a fixed TileLang schedule.",
    },
    "jit-tilelang-causal-conv1d-prefill-width4": {
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/causal_conv1d.py",
            "vllm_musa/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        ],
        "notes": "Width-4 prefill uses a fixed TileLang schedule.",
    },
    "jit-tilelang-deep-gemm-contig-preprocess-full-tile-kernel": {
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/deep_gemm_contig_preprocess.py",
            "vllm_musa/model_executor/kernels/linear/scaled_mm/deep_gemm.py",
        ],
        "notes": "Full-tile preprocessing uses a fixed shape-derived config.",
    },
    "jit-tilelang-deep-gemm-contig-preprocess-kernel": {
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/deep_gemm_contig_preprocess.py",
            "vllm_musa/model_executor/kernels/linear/scaled_mm/deep_gemm.py",
        ],
        "notes": "General preprocessing uses a fixed shape-derived config.",
    },
    "jit-tilelang-deep-gemm-contig-preprocess-noidx-kernel": {
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/deep_gemm_contig_preprocess.py",
            "vllm_musa/model_executor/kernels/linear/scaled_mm/deep_gemm.py",
        ],
        "notes": "No-index preprocessing uses a fixed shape-derived config.",
    },
    "jit-tilelang-deep-gemm-contig-preprocess-padded-kernel": {
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/deep_gemm_contig_preprocess.py",
            "vllm_musa/model_executor/kernels/linear/scaled_mm/deep_gemm.py",
        ],
        "notes": "Padded preprocessing uses a fixed shape-derived config.",
    },
    "jit-tilelang-deepgemm-contig-assign-compact": {
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/deep_gemm_contig_preprocess.py",
            "vllm_musa/model_executor/kernels/linear/scaled_mm/deep_gemm.py",
        ],
        "notes": "Compact assignment uses a fixed shape-derived config.",
    },
    "jit-tilelang-deepgemm-contig-clear-fill": {
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/deep_gemm_contig_preprocess.py",
            "vllm_musa/model_executor/kernels/linear/scaled_mm/deep_gemm.py",
        ],
        "notes": "Clear/fill preprocessing uses a fixed shape-derived config.",
    },
    "jit-tilelang-deepgemm-contig-count-prefix": {
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/deep_gemm_contig_preprocess.py",
            "vllm_musa/model_executor/kernels/linear/scaled_mm/deep_gemm.py",
        ],
        "notes": "Count/prefix preprocessing uses a fixed shape-derived config.",
    },
    "jit-tilelang-deepgemm-contig-scan-tree": {
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/deep_gemm_contig_preprocess.py",
            "vllm_musa/model_executor/kernels/linear/scaled_mm/deep_gemm.py",
        ],
        "notes": "Tree scan preprocessing uses a fixed shape-derived config.",
    },
    "jit-tilelang-fused-zba-kernel": {
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/gdn_fused_proj.py",
            "vllm_musa/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        ],
        "notes": "The GDN projection fusion has one fixed TileLang schedule.",
    },
    "jit-tilelang-musa-sparse-attention-fwd-kernel-v1": {
        "production_consumers": [
            "vllm_musa/v1/attention/ops/sparse_mla_tilelang.py",
        ],
        "notes": "Sparse MLA forward exposes no per-call MP schedule override.",
    },
    "jit-tilelang-rmsnorm-gated": {
        "production_consumers": [
            "vllm_musa/jit_kernel/tilelang/layernorm_gated.py",
            "vllm_musa/model_executor/layers/layernorm.py",
        ],
        "notes": "Gated RMSNorm uses a fixed TileLang schedule.",
    },
    "jit-triton-fused-gdn-gating-kernel": {
        "production_consumers": [
            "vllm_musa/jit_kernel/fused_gdn_gating.py",
            "vllm_musa/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        ],
        "notes": "The Triton GDN gate has no active-MP launch selector.",
    },
    "jit-triton-gated-qk-norm-rope-token-kernel": {
        "production_consumers": [
            "vllm_musa/kernels/gated_qkv.py",
        ],
        "notes": "The token kernel uses a fixed Triton schedule per shape.",
    },
    "jit-triton-post-reorder-triton-kernel": {
        "production_consumers": [
            "vllm_musa/jit_kernel/post_reorder.py",
            "vllm_musa/model_executor/layers/fused_moe/fused_moe.py",
        ],
        "notes": "Post-reorder uses a fixed Triton schedule per shape.",
    },
}


def _keys(backend: str, *symbols: str) -> dict[str, str]:
    return {f"{backend}:{symbol}": "" for symbol in symbols}


_SHARED_FAMILY_GROUPS: dict[str, tuple[str, ...]] = {
    "jit-native-custom-allreduce": (
        "jit-native-ffi:vllm_musa_custom_ar_launch_all_gather",
        "jit-native-ffi:vllm_musa_custom_ar_launch_graph_registered",
        "jit-native-ffi:vllm_musa_custom_ar_launch_registered",
        "jit-native-ffi:vllm_musa_custom_ar_launch_unregistered",
        "jit-native-ffi:vllm_musa_custom_ar_meta_size",
        "jit-native-ffi:vllm_musa_fused_ar_residual_rmsnorm_launch_unregistered",
        "jit-native-ffi:vllm_musa_fused_ar_residual_rmsnorm_no_raw_launch_unregistered",
        "jit-native-ffi:vllm_musa_fused_ar_rmsnorm_launch_unregistered",
    ),
    "jit-native-moe-act-and-mul": (
        "jit-native:musa_fast_silu_and_mul",
        "jit-native-ffi:sgl_musa_moe_act_and_mul",
    ),
    "jit-native-moe-sum": (
        "jit-native:musa_fast_moe_sum",
        "jit-native-ffi:sgl_musa_moe_sum_reduce",
    ),
    "jit-native-fused-add-rmsnorm": (
        "jit-native:musa_csrc_fused_add_rmsnorm",
        "jit-native-ffi:sgl_musa_fused_add_rmsnorm",
    ),
    "jit-native-rmsnorm": (
        "jit-native:musa_csrc_rmsnorm",
        "jit-native-ffi:sgl_musa_rmsnorm",
    ),
    "jit-native-fused-qk-rmsnorm-mrope": (
        "jit-native:musa_csrc_fused_qk_rmsnorm_mrope",
        "jit-native:musa_csrc_fused_qk_rmsnorm_mrope_cache_out",
        "jit-native-ffi:sgl_musa_fused_qk_rmsnorm_mrope",
        "jit-native-ffi:sgl_musa_fused_qk_rmsnorm_mrope_cache",
        "jit-native-ffi:sgl_musa_fused_qk_rmsnorm_mrope_cache_out",
    ),
    "jit-native-per-token-group-quant": (
        "jit-native:musa_csrc_per_token_group_quant_8bit",
        "jit-native-ffi:sgl_per_token_group_quant_8bit_v2",
    ),
    "jit-native-rotary-embedding": (
        "jit-native:musa_rotary_embedding",
        "jit-native-ffi:sgl_rotary_embedding",
    ),
    "jit-native-topk-sigmoid": (
        "jit-native:musa_topk_sigmoid",
        "jit-native-ffi:sgl_musa_topk_sigmoid",
    ),
    "jit-native-topk-softmax": (
        "jit-native:musa_topk_softmax",
        "jit-native-ffi:sgl_musa_topk_softmax",
    ),
    "jit-tilelang-dsv4-mhc-pre-big-fuse": (
        "jit-tilelang:mhc_pre_big_fuse_decode_split_kernel",
        "jit-tilelang:mhc_pre_big_fuse_kernel",
    ),
    "jit-tilelang-dsv4-mhc-weighted-rmsnorm": (
        "jit-tilelang:mhc_weighted_rmsnorm_kernel",
        "jit-tilelang:mhc_weighted_rmsnorm_mudnn_like_kernel",
    ),
    "jit-tilelang-causal-conv1d-prefill-width4": (
        "jit-tilelang:_causal_conv1d_prefill_width4_body_kernel",
        "jit-tilelang:_causal_conv1d_prefill_width4_kernel",
    ),
    "jit-tilelang-deepgemm-contig-assign-compact": (
        "jit-tilelang:_bf16_assign_compact_kernel",
        "jit-tilelang:_fp8_assign_compact_kernel",
    ),
    "jit-tilelang-deepgemm-contig-clear-fill": (
        "jit-tilelang:_clear_i32_kernel",
        "jit-tilelang:_fill_i32_kernel",
    ),
    "jit-tilelang-deepgemm-contig-count-prefix": (
        "jit-tilelang:_count_prefix_topk_single_block_kernel",
        "jit-tilelang:_count_topk_block_hist_kernel",
        "jit-tilelang:_count_topk_single_block_kernel",
        "jit-tilelang:_prefix_counts_kernel",
    ),
    "jit-tilelang-deepgemm-contig-scan-tree": (
        "jit-tilelang:_prefix_counts_scan_kernel",
        "jit-tilelang:_prefix_counts_tree_kernel",
    ),
    "jit-tilelang-grouped-topk": (
        "jit-tilelang:_grouped_topk_parallel_kernel",
        "jit-tilelang:_grouped_topk_serial_kernel",
    ),
    "jit-tilelang-rmsnorm-gated": (
        "jit-tilelang:_rms_norm_gated_kernel",
        "jit-tilelang:_rms_norm_gated_kernel_cta",
    ),
}

_MEMBER_TO_SHARED_FAMILY = {
    member: family
    for family, members in _SHARED_FAMILY_GROUPS.items()
    for member in members
}

_AOT_PHYSICAL_SOURCE_FAMILIES: dict[str, tuple[str, ...]] = {
    "csrc/musa/attention/deepseek_v4_c4_indexer_compressor.mu": (
        "aot-native-deepseek-v4-c4-indexer-compress-cache",
    ),
    "csrc/musa/attention/deepseek_v4_cache_store.mu": (
        "aot-native-deepseek-v4-qnorm-rope-kv-insert",
        "aot-native-deepseek-v4-store-sparse-kv",
    ),
    "csrc/musa/attention/deepseek_v4_cache_utils.mu": (
        "aot-native-deepseek-v4-combine-topk-swa-indices",
        "aot-native-deepseek-v4-compute-global-topk-indices-and-lens",
        "aot-native-deepseek-v4-dequantize-and-gather-k-cache",
    ),
    "csrc/musa/attention/deepseek_v4_fused_qkv_rmsnorm.mu": (
        "aot-native-deepseek-v4-fused-q-kv-rmsnorm",
    ),
    "csrc/musa/attention/deepseek_v4_indexer_topk.mu": (
        "aot-native-deepseek-v4-indexer-rerank-prefill",
        "aot-native-deepseek-v4-indexer-topk-decode",
        "aot-native-deepseek-v4-indexer-topk-prefill",
    ),
    "csrc/musa/attention/deepseek_v4_inv_rope_fp8_quant.mu": (
        "aot-native-deepseek-v4-fused-inv-rope-fp8-quant",
    ),
    "csrc/musa/attention/deepseek_v4_sparse_flashmla.mu": (
        "aot-native-deepseek-v4-sparse-flashmla-decode",
    ),
    "csrc/musa/attention/glm52_indexer_topk.mu": (
        "aot-native-glm52-indexer-topk-decode",
        "aot-native-glm52-indexer-topk-prefill",
        "aot-native-sparse-indexer-fill-all",
        "aot-native-sparse-indexer-topk",
        "aot-native-sparse-indexer-topk-decode",
    ),
    "csrc/musa/cache_kernels.mu": ("aot-native-musa-reshape-and-cache-flash-nhd",),
    "csrc/musa/fused_add_rmsnorm.mu": ("aot-native-musa-fused-add-rms-norm",),
    "csrc/musa/gemv.mu": (
        "aot-native-musa-fused-gemv",
        "aot-native-musa-fused-gemv-moe",
    ),
    "csrc/musa/mhc/deepseek_v4_mhc_pre.mu": ("aot-native-deepseek-v4-mhc-pre",),
    "csrc/musa/min_p_sampler.mu": (
        "aot-native-musa-chunked-min-p-sampling-from-probs",
    ),
    "csrc/musa/moe/deepseek_v4_topk_softplus_sqrt.mu": (
        "aot-native-deepseek-v4-topk-softplus-sqrt",
    ),
    "csrc/musa/quantization/fused_add_rms_norm_per_token_group_fp8_quant.cu": (
        "aot-native-fused-add-rms-norm-per-token-group-fp8-quant",
    ),
    "csrc/musa/quantization/per_token_group_quant_8bit_vec.cu": (
        "aot-native-per-token-group-quant-8bit-vec",
    ),
    "csrc/musa/quantization/silu_and_mul_per_token_group_fp8_quant.cu": (
        "aot-native-silu-and-mul-clamp-per-token-group-fp8-quant",
        "aot-native-silu-and-mul-per-token-group-fp8-quant",
    ),
    "csrc/musa/sampler.mu": ("aot-native-musa-top-k-top-p-sampling-from-probs",),
    "csrc/musa/top_k_renorm.mu": ("aot-native-musa-rubymine-top-k-renorm-probs",),
}

_JIT_PHYSICAL_SOURCE_FAMILIES: dict[str, tuple[str, ...]] = {
    "vllm_musa/jit_kernel/csrc/distributed/custom_all_reduce.mu": (
        "jit-native-custom-allreduce",
    ),
    "vllm_musa/jit_kernel/csrc/moe/act_and_mul.mu": ("jit-native-moe-act-and-mul",),
    "vllm_musa/jit_kernel/csrc/moe/moe_sum_reduce.mu": ("jit-native-moe-sum",),
    "vllm_musa/jit_kernel/csrc/norm/qk_mrope.mu": (
        "jit-native-fused-qk-rmsnorm-mrope",
    ),
    "vllm_musa/jit_kernel/csrc/quant/per_token_group_quant_8bit_v2.mu": (
        "jit-native-per-token-group-quant",
    ),
    "vllm_musa/jit_kernel/csrc/rope/rotary_embedding.mu": (
        "jit-native-rotary-embedding",
    ),
}


def _slug(value: str) -> str:
    value = re.sub(r"^_+", "", value)
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value)
    return value.strip("-").lower()


def family_id_for_symbol(backend: str, symbol: str) -> str:
    short_id = f"{backend}:{symbol}"
    shared = _MEMBER_TO_SHARED_FAMILY.get(short_id)
    if shared is not None:
        return shared
    return f"{backend}-{_slug(symbol)}"


def family_id(entry: INVENTORY.KernelEntry) -> str:
    return family_id_for_symbol(entry.backend, entry.symbol)


def physical_coverage(root: Path) -> dict[str, tuple[str, ...]]:
    """Map every physical kernel definition to reviewed callable families."""
    callable_families = set(grouped_families(root))
    coverage: dict[str, tuple[str, ...]] = {}
    for entry in PHYSICAL_INVENTORY.discover(root):
        families: tuple[str, ...]
        if entry.backend == "aot-native-physical":
            families = _AOT_PHYSICAL_SOURCE_FAMILIES.get(entry.source, ())
        elif entry.backend == "jit-native-physical":
            if entry.source == "vllm_musa/jit_kernel/csrc/norm/rmsnorm.mu":
                families = (
                    "jit-native-fused-add-rmsnorm"
                    if entry.symbol.startswith("fused_add_")
                    else "jit-native-rmsnorm",
                )
            elif entry.source == "vllm_musa/jit_kernel/csrc/topk/topk_gating.mu":
                if entry.symbol == "topk_block_kernel":
                    families = (
                        "jit-native-topk-sigmoid",
                        "jit-native-topk-softmax",
                    )
                elif "sigmoid" in entry.symbol:
                    families = ("jit-native-topk-sigmoid",)
                else:
                    families = ("jit-native-topk-softmax",)
            else:
                families = _JIT_PHYSICAL_SOURCE_FAMILIES.get(entry.source, ())
        elif entry.backend == "jit-tilelang-physical":
            assert entry.owner is not None
            if entry.owner == "_prefix_counts_no_pad_kernel":
                families = ("jit-tilelang-deepgemm-contig-count-prefix",)
            else:
                families = (family_id_for_symbol("jit-tilelang", entry.owner),)
        elif entry.backend == "jit-triton-physical":
            assert entry.owner is not None
            families = (family_id_for_symbol("jit-triton", entry.owner),)
        else:
            families = ()
        if not families:
            raise RuntimeError(f"physical kernel has no family coverage: {entry}")
        unknown = set(families) - callable_families
        if unknown:
            raise RuntimeError(
                f"physical kernel maps to unknown families {sorted(unknown)}: {entry}"
            )
        coverage[entry.id] = families
    return coverage


def grouped_families(root: Path) -> dict[str, list[INVENTORY.KernelEntry]]:
    grouped: dict[str, list[INVENTORY.KernelEntry]] = defaultdict(list)
    for entry in INVENTORY.discover(root):
        grouped[family_id(entry)].append(entry)
    return {family: sorted(entries) for family, entries in sorted(grouped.items())}


def skeleton(root: Path) -> dict[str, object]:
    families = grouped_families(root)
    family_documents = []
    for family, entries in families.items():
        document = {
            "id": family,
            "members": [entry.id for entry in entries],
            "backend_classes": sorted({entry.backend for entry in entries}),
            "disposition": "pending",
            "production_consumers": [],
            "shape_sources": [],
            "tunable_parameters": [],
            "benchmark": None,
            "notes": "",
        }
        metadata = _FAMILY_METADATA.get(family, {})
        unexpected = set(metadata) - _PUBLIC_METADATA_KEYS
        if unexpected:
            raise ValueError(
                f"non-public catalog metadata for {family}: " f"{sorted(unexpected)}"
            )
        document.update(metadata)
        fixed_geometry = _FIXED_GEOMETRY_FAMILIES.get(family)
        if fixed_geometry is not None:
            document.update(
                {
                    "disposition": "no-tunable-seam",
                    "tunable_parameters": [],
                    **fixed_geometry,
                }
            )
        family_documents.append(document)
    physical_payload = PHYSICAL_INVENTORY.payload(root)
    return {
        "schema": SCHEMA,
        "scope": {
            "repository": "vllm-musa",
            "owned_implementations_only": True,
            "external_provider_wrappers": "excluded",
            "runtime_hardware_key": "multi_processor_count",
            "callable_entrypoint_count": len(INVENTORY.discover(root)),
            "physical_implementation_counts": physical_payload["counts"],
            "physical_implementation_count": len(physical_payload["entries"]),
            "physical_kernel_note": (
                "Callable entries and physical definitions are separate "
                "denominators; physical_coverage() maps every definition."
            ),
        },
        "allowed_dispositions": [
            "pending",
            "no-tunable-seam",
            "external-provider-excluded",
            "non-production",
        ],
        "families": family_documents,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--skeleton", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document = skeleton(args.root)
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
