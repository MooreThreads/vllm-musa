# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bridge torchada's autotuned MoE configs into vLLM's search path.

MUSA-0053: vLLM's `get_moe_configs`
(`vllm/model_executor/layers/fused_moe/fused_moe.py`) only searches
`vllm/model_executor/layers/fused_moe/configs/<name>.json`, building the
filename with no space in the `block_shape` segment
(`...block_shape=[128,128].json`, because vLLM does
`f",block_shape={block_shape}".replace(" ", "")`).

torchada ships its autotuned MUSA configs under
`torchada/triton/autotune/fused_moe/configs/triton_*/`, and Python's
`str(list)` repr produces a SPACE after the comma
(`...block_shape=[128, 128].json`).

Net effect on a stock MUSA install: vLLM never finds the tuned configs
and logs `Using default MoE config. Performance might be sub-optimal!`
(observed in the MUSA-0049 MiniMax-M2.7 baseline for
`E=32,N=1536,dtype=fp8_w8a8,block_shape=[128,128]`).

This module, imported once at platform init, copies every
`*device_name=MTT_S5000*` torchada config into vLLM's configs dir with
the spaces stripped from the filename, so vLLM's lookup succeeds. It is
idempotent (skips files that already exist) and best-effort (a missing
torchada dir or an unwritable vLLM dir logs a warning and is otherwise
a no-op — vLLM then falls back to its default config, i.e. exactly the
pre-bridge behaviour).
"""

import glob
import os
import shutil

from vllm.logger import init_logger

logger = init_logger(__name__)

_bridged = False


def _torchada_moe_config_dir() -> str | None:
    try:
        import torchada
    except ImportError:
        return None
    base = os.path.join(
        os.path.dirname(torchada.__file__),
        "triton",
        "autotune",
        "fused_moe",
        "configs",
    )
    return base if os.path.isdir(base) else None


def _vllm_moe_config_dir() -> str | None:
    try:
        import vllm.model_executor.layers.fused_moe.fused_moe as vfm
    except ImportError:
        return None
    return os.path.join(os.path.dirname(vfm.__file__), "configs")


def bridge_musa_moe_configs() -> int:
    """Copy torchada MTT_S5000 MoE configs into vLLM's config dir.

    Returns the number of files newly bridged. Idempotent.
    """
    global _bridged
    if _bridged:
        return 0
    _bridged = True

    ta_dir = _torchada_moe_config_dir()
    if ta_dir is None:
        logger.debug("torchada MoE config dir not found; skipping MoE bridge")
        return 0
    vllm_dir = _vllm_moe_config_dir()
    if vllm_dir is None:
        logger.debug("vLLM MoE config dir not found; skipping MoE bridge")
        return 0

    try:
        os.makedirs(vllm_dir, exist_ok=True)
    except OSError as exc:
        logger.warning("Cannot create vLLM MoE config dir %s: %s", vllm_dir, exc)
        return 0

    bridged = 0
    pattern = os.path.join(ta_dir, "**", "*device_name=MTT_S5000*.json")
    for src in glob.glob(pattern, recursive=True):
        # vLLM builds the lookup filename with spaces stripped.
        dst_name = os.path.basename(src).replace(" ", "")
        dst = os.path.join(vllm_dir, dst_name)
        if os.path.exists(dst):
            continue
        try:
            shutil.copy2(src, dst)
            bridged += 1
        except OSError as exc:
            logger.warning("Failed to bridge MoE config %s -> %s: %s", src, dst, exc)

    if bridged:
        logger.info(
            "Bridged %d torchada MTT_S5000 MoE config(s) into %s",
            bridged,
            vllm_dir,
        )
    else:
        logger.debug("No new MTT_S5000 MoE configs to bridge")
    return bridged


# Run on import (vllm_musa.model_executor wires this module).
bridge_musa_moe_configs()
