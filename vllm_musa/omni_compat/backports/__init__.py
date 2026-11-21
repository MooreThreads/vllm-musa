# SPDX-License-Identifier: Apache-2.0
"""Implementations for drift the pinned vLLM cannot serve by relocation alone.

One module per drifted vLLM module, named after it. Each is referenced by an
``ADAPTER`` or ``BACKPORT`` entry in :mod:`vllm_musa.omni_compat.table` and must
document, in its own docstring, which pinned APIs it is built from and when it
should be deleted. Nothing here is imported until the shim serves the module it
belongs to, so a backport for a module the pin already provides costs nothing.
"""
