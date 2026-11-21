# SPDX-License-Identifier: Apache-2.0
"""Import-only DraftLogitsCache marker for the v0.34 MiniCPM-o Talker.

The pinned vLLM has a different speculative-decoding cache contract and cannot
consume v0.34's cache object. Omni's MiniCPM-o Talker uses this name only in a
future-annotations type and passes any real cache to the pinned rejection
sampler. Keeping construction explicit prevents this marker from being mistaken
for a working speculative-decoding implementation.
"""

from __future__ import annotations


class DraftLogitsCache:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "DraftLogitsCache requires vLLM >= 0.34; speculative decoding is "
            "unsupported on the pinned vLLM-MUSA 0.32 runtime"
        )
