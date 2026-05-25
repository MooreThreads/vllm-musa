# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA-0088 iter-2: Python wrapper for the SGLang-ported quick_all_reduce
kernel.

Scaffolding only — calls into `torch.ops._C_quick_ar.*` which are
registered by `vllm-musa/csrc/quick_all_reduce.cu`. The actual op
registration happens at module-load time of the compiled `vllm_musa._C`
extension (built via `vllm-musa/setup.py`).

If the underlying kernel hasn't compiled successfully yet, this module
gracefully no-ops the dispatch: `should_quick_ar()` returns False so
the standard `custom_all_reduce` path is used.

API mirrors `vllm/.../custom_all_reduce.py:CustomAllreduce`:
    - __init__(group, world_size, rank, max_size)
    - should_quick_ar(input)
    - all_reduce(input, output) — in-place via output tensor
    - __del__: free the C++-side context
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from vllm.logger import init_logger

# vllm.logger.init_logger returns a Logger with info_once / warning_once
# helpers; the previous plain logging.getLogger here would have raised
# AttributeError on first miss (PR #40 review comment), turning the
# graceful "ops not available" degrade path into an ImportError.
logger = init_logger(__name__)


# Probe for the compiled ops. If missing, the wrapper degrades gracefully.
try:
    _ = torch.ops._C_quick_ar.init
    _QUICK_AR_AVAILABLE = True
except (AttributeError, RuntimeError) as exc:
    logger.info_once(
        "MUSA-0088: quick_all_reduce ops not available (%s); "
        "QuickAllReduce will no-op and the standard custom_all_reduce path "
        "stays active. This is expected until the kernel compile lands.",
        exc,
    )
    _QUICK_AR_AVAILABLE = False


class QuickAllReduce:
    """One-shot two-shot P2P all-reduce wrapper for MUSA TP>2.

    Mirrors SGLang's quick_all_reduce on AMD ROCm. Designed for small
    payloads (≤ 2 GiB) on fully-connected P2P topologies (S5000 NVLink
    equivalent). Falls back to CustomAllreduce / mccl alltoall when
    payload exceeds kMaxProblemSize.
    """

    _SUPPORTED_WORLD_SIZES = (2, 4, 8)
    _MAX_PROBLEM_BYTES = (2**31) + 1  # mirror C++ qr_max_size() default

    def __init__(
        self,
        group: dist.ProcessGroup,
        world_size: int,
        rank: int,
        max_size: Optional[int] = None,
    ) -> None:
        self.group = group
        self.world_size = world_size
        self.rank = rank
        self.max_size = max_size or self._MAX_PROBLEM_BYTES
        self.fptr: int = 0
        self.disabled = not _QUICK_AR_AVAILABLE

        if world_size not in self._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "MUSA-0088: QuickAllReduce disabled — world_size=%d "
                "not in supported set %s.",
                world_size,
                self._SUPPORTED_WORLD_SIZES,
            )
            self.disabled = True
            return

        if self.disabled:
            return

        try:
            self.fptr = torch.ops._C_quick_ar.init(rank, world_size, self.max_size)
        except Exception as exc:
            logger.warning(
                "MUSA-0088: QuickAllReduce init failed (%s); disabling.", exc
            )
            self.disabled = True
            self.fptr = 0
            return

        # IPC handle exchange via the process group.
        try:
            my_handle: torch.Tensor = torch.ops._C_quick_ar.get_handle(self.fptr)
            all_handles = [torch.empty_like(my_handle) for _ in range(world_size)]
            dist.all_gather(all_handles, my_handle, group=group)
            # Pass the full handle list (length == world_size). The C++
            # DeviceComms::open_ipc_handles asserts size() == world_size
            # and indexes by rank; it already skips the self entry
            # internally (i == rank branch). Filtering out self here
            # would shift the indices and mis-map ranks — see PR #40
            # review comment.
            torch.ops._C_quick_ar.open_handles(self.fptr, all_handles)
        except Exception as exc:
            logger.warning(
                "MUSA-0088: QuickAllReduce IPC handle exchange failed "
                "(%s); disabling.",
                exc,
            )
            self.disabled = True
            self._safe_destroy()

    def should_quick_ar(self, input_: torch.Tensor) -> bool:
        """Gate: True when this all-reduce should use quick_ar over the
        standard custom_all_reduce / mccl path."""
        if self.disabled:
            return False
        if input_.numel() * input_.element_size() > self.max_size:
            return False
        if input_.dtype not in (torch.bfloat16, torch.float16):
            return False
        return True

    # API-compat aliases matching vllm-upstream cuda_communicator.all_reduce naming.
    def should_quick_allreduce(self, input_: torch.Tensor) -> bool:
        return self.should_quick_ar(input_)

    def quick_all_reduce(
        self, input_: torch.Tensor, quant_level: int = 0, cast_bf2half: bool = False
    ) -> Optional[torch.Tensor]:
        """vllm-upstream API alias: returns the all-reduced output tensor
        directly (in-place via a fresh output buffer)."""
        return self.all_reduce(
            input_, output=None, quant_level=quant_level, cast_bf2half=cast_bf2half
        )

    def all_reduce(
        self,
        input_: torch.Tensor,
        output: Optional[torch.Tensor] = None,
        quant_level: int = 0,
        cast_bf2half: bool = False,
    ) -> Optional[torch.Tensor]:
        """Run quick all-reduce on input_, writing to output (or in-place).

        Returns the output tensor on success, or None if the call was
        skipped (disabled / unsupported shape)."""
        if not self.should_quick_ar(input_):
            return None
        if output is None:
            output = torch.empty_like(input_)
        torch.ops._C_quick_ar.all_reduce(
            self.fptr, input_, output, quant_level, cast_bf2half
        )
        return output

    def _safe_destroy(self) -> None:
        if self.fptr:
            try:
                torch.ops._C_quick_ar.destroy(self.fptr)
            except Exception:
                pass
            self.fptr = 0

    def __del__(self) -> None:
        self._safe_destroy()
