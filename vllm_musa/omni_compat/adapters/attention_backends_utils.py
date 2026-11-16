# SPDX-License-Identifier: Apache-2.0
"""``compute_mm_prefix_ranges`` on a pin that keeps it in the GPU worker.

vLLM 0.33 moved the PrefixLM range builder from
``vllm.v1.worker.gpu.attn_utils`` to ``vllm.v1.attention.backends.utils`` and
changed its calling convention: the pin takes ``(req_ids, mm_features_by_id,
sliding_window)``, 0.33 takes ``(mm_features_per_req, sliding_window,
clamp_sliding_window, span_pad)``.

This one has to be adapted rather than declared unsupported, because
``vllm_omni/patch.py`` calls ``apply_img2img_mm_prefix_patch()`` at **module
import time** and that function reads the attribute off the 0.33 module. Without
it, ``import vllm_omni`` raises AttributeError on the pin and no model runs at
all -- not just Bagel's img2img, which is the only feature that needs the patch.

The translation is exact at the default arguments. Compared statement by
statement, the pinned body and the 0.33 body do the same thing when
``span_pad == 0`` (both take ``mm_position.extract_embeds_range()``) and
``clamp_sliding_window is False`` (both skip a range longer than
``sliding_window``), and both key the result by request index. The pin indexes
its features by request id, so the bridge gives it synthetic ids whose order is
the caller's request order.

The two newer arguments are refused rather than ignored. ``span_pad`` makes 0.33
round the span start, and ``clamp_sliding_window`` makes it keep long ranges;
the pin can express neither, so passing them would return quietly different
ranges -- a wrong attention mask, not a missing feature. A caller that needs
them needs a newer pin.

Delete this adapter when ``third_party/PINS`` reaches a vLLM that ships
``compute_mm_prefix_ranges`` in ``vllm.v1.attention.backends.utils``.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalFeatureSpec

ADAPTED = "__vllm_musa_mm_prefix_bridge__"


def _bridge():
    def compute_mm_prefix_ranges(
        mm_features_per_req: "Sequence[Sequence[MultiModalFeatureSpec]]",
        sliding_window: int | None = None,
        clamp_sliding_window: bool = False,
        span_pad: int = 0,
    ) -> dict[int, list[tuple[int, int]]]:
        """0.33's signature over the pinned implementation. See the module docstring."""
        # Imported here, not at patch time: the patch runs while
        # vllm.v1.attention.backends.utils is still executing, and pulling in
        # another vLLM module at that moment risks a circular import.
        from vllm.v1.worker.gpu.attn_utils import compute_mm_prefix_ranges as pinned

        if clamp_sliding_window or span_pad:
            raise NotImplementedError(
                "vllm_musa.omni_compat: compute_mm_prefix_ranges("
                f"clamp_sliding_window={clamp_sliding_window!r}, span_pad={span_pad!r}) "
                "needs vLLM >= 0.33. The pinned builder always skips ranges longer than "
                "sliding_window and cannot round a span start, so bridging these would "
                "return different attention ranges rather than fail. Bump "
                "third_party/PINS to use this path."
            )
        # The pin keys features by request id and the result by request index,
        # so synthetic ids in the caller's order reproduce its output exactly.
        req_ids = [str(index) for index in range(len(mm_features_per_req))]
        mm_features = {
            str(index): list(features)
            for index, features in enumerate(mm_features_per_req)
        }
        return pinned(req_ids, mm_features, sliding_window)

    setattr(compute_mm_prefix_ranges, ADAPTED, True)
    # vllm_omni.utils.mm_prefix wraps this with functools.wraps and checks its
    # own marker attribute, both of which need a normal function object.
    compute_mm_prefix_ranges.__name__ = "compute_mm_prefix_ranges"
    compute_mm_prefix_ranges.__qualname__ = "compute_mm_prefix_ranges"
    return compute_mm_prefix_ranges


def is_applied(module: ModuleType) -> list[str]:
    """Which names are adapted **right now**, read-only. See the sibling adapter."""
    live = getattr(module, "compute_mm_prefix_ranges", None)
    return ["compute_mm_prefix_ranges"] if getattr(live, ADAPTED, False) else []


def apply(module: ModuleType) -> list[str]:
    """Add the 0.33 range builder to the pinned attention-backend utilities."""
    existing = getattr(module, "compute_mm_prefix_ranges", None)
    if existing is not None and not getattr(existing, ADAPTED, False):
        return []  # the pin provides its own; nothing to adapt
    if existing is not None:
        return []  # already adapted
    module.compute_mm_prefix_ranges = _bridge()
    return ["compute_mm_prefix_ranges"]
