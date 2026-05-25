# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
import time
from math import isfinite
from typing import Any

import vllm.v1.executor.multiproc_executor as vllm_multiproc_executor

logger = logging.getLogger(__name__)


def _musa_worker_termination_timeout_s() -> float:
    value = os.environ.get("VLLM_MUSA_WORKER_TERMINATION_TIMEOUT_S", "4")

    try:
        timeout_s = float(value)
    except ValueError:
        logger.warning(
            "Invalid VLLM_MUSA_WORKER_TERMINATION_TIMEOUT_S=%r; using default 4 seconds",
            value,
        )
        return 4
    if timeout_s <= 0 or not isfinite(timeout_s):
        logger.warning(
            "Invalid VLLM_MUSA_WORKER_TERMINATION_TIMEOUT_S=%r; value must be positive and finite; using default 4 seconds",
            value,
        )
        return 4
    return timeout_s


def _ensure_worker_termination(worker_procs: list[Any]) -> None:
    """Ensure that all worker processes are terminated. Assumes workers have
    received termination requests. Waits for processing, then sends
    termination and kill signals if needed."""

    def wait_for_termination(procs, timeout):
        if not time:
            # If we are in late stage shutdown, the interpreter may replace
            # `time` with `None`.
            return all(not proc.is_alive() for proc in procs)
        start_time = time.time()
        while time.time() - start_time < timeout:
            if all(not proc.is_alive() for proc in procs):
                return True
            time.sleep(0.1)
        return False

    active_procs = lambda: [proc for proc in worker_procs if proc.is_alive()]
    # Give processes time to clean themselves up properly first
    logger.debug("Worker Termination: allow workers to gracefully shutdown")
    # ==================== MUSA ADAPTATION ====================
    timeout_s = _musa_worker_termination_timeout_s()
    if wait_for_termination(active_procs(), timeout_s):
        return

    # Send SIGTERM if still running
    logger.debug("Worker Termination: workers still running sending SIGTERM")
    for p in active_procs():
        p.terminate()
    if not wait_for_termination(active_procs(), timeout_s):
        # Send SIGKILL if still running
        logger.debug("Worker Termination: resorting to SIGKILL to take down workers")
        for p in active_procs():
            p.kill()
    # ========================== END ==========================


vllm_multiproc_executor.MultiprocExecutor._ensure_worker_termination = staticmethod(
    _ensure_worker_termination
)
