# SPDX-License-Identifier: Apache-2.0
"""Call-time adapters for pinned vLLM functions Omni calls with newer arguments.

One module per drifted vLLM module, named after it, each exposing
``apply(module) -> list[str]`` that patches the module in place and returns the
names it patched. Returning an empty list makes the build fail, because an
adapter that quietly does nothing is worse than none: the run continues and
breaks later with no mention of the shim.

An adapter must argue, in its docstring, why dropping or translating the newer
argument preserves behaviour on the pin. When it cannot, the right answer is a
``CAPABILITY`` entry in :mod:`vllm_musa.omni_compat.table` declaring the path
unsupported, not a wrapper that guesses.
"""
