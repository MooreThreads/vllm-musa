# SPDX-License-Identifier: Apache-2.0

"""Shared qualification contract for fused-MoE crossover evidence."""

FUSED_MOE_CROSSOVER_SCHEMA = "musa-fused-moe-crossover.v8"
# PH1/S5000 exposes a 60 MiB whole-chip LLC through the driver cache-size
# field. 512 MiB is over 8x that capacity, so it guarantees eviction without
# spending most of a crossover campaign filling an unrelated 8 GiB tensor.
FUSED_MOE_MIN_QUALIFIED_L2_FLUSH_MB = 512

# Artifact-declared thresholds may be stricter, but never weaker, than these
# production limits. Tuples keep the contract immutable and deterministic.
FUSED_MOE_MAX_ERROR_THRESHOLDS = (
    ("gemv_max_relative_l2", 0.06),
    ("gemv_max_row_relative_l2", 0.08),
    ("gemv_max_normalized_abs_diff", 0.10),
    ("gemv_oracle_max_relative_l2", 0.01),
    ("gemv_oracle_max_row_relative_l2", 0.02),
    ("gemv_oracle_max_normalized_abs_diff", 0.05),
    ("grouped_max_relative_l2", 0.01),
    ("grouped_max_row_relative_l2", 0.02),
    ("grouped_max_normalized_abs_diff", 0.05),
)
FUSED_MOE_MIN_COSINE_THRESHOLDS = (
    ("gemv_min_cosine", 0.998),
    ("gemv_min_row_cosine", 0.995),
    ("gemv_oracle_min_cosine", 0.9999),
    ("gemv_oracle_min_row_cosine", 0.9998),
    ("grouped_min_cosine", 0.9999),
    ("grouped_min_row_cosine", 0.999),
)

__all__ = [
    "FUSED_MOE_CROSSOVER_SCHEMA",
    "FUSED_MOE_MAX_ERROR_THRESHOLDS",
    "FUSED_MOE_MIN_COSINE_THRESHOLDS",
    "FUSED_MOE_MIN_QUALIFIED_L2_FLUSH_MB",
]
