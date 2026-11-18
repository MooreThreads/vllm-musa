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

What this adapter does **not** carry over: on the pin nothing reads the builder
from this module. The v1 model runner fills ``mm_req_doc_ranges`` with its own
inline loop, which filters only audio and so admits img2img already, and the v2
runner's model state binds the builder that stayed in
``vllm.v1.worker.gpu.attn_utils``. So Omni's ``img2img``->``image`` aliasing
reaches no consumer on the pin, and a stage configured with ``model_runner: v2``
gets empty img2img ranges rather than an error. That gap is declared as a
``CAPABILITY`` row on the other module; this bridge exists to keep
``import vllm_omni`` working, which is the part that has to be true for any
model to run at all.

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


def _wraps_the_bridge(obj: object) -> bool:
    """Whether ``obj`` is, or wraps, the bridge this adapter installed.

    ``vllm_omni.utils.mm_prefix`` decorates the bridge with ``functools.wraps``,
    so after ``import vllm_omni`` the live attribute is Omni's wrapper rather
    than ours. It still reaches the bridge, so it is adapted -- but the test
    has to say that by walking ``__wrapped__``, not by relying on
    ``functools.wraps`` having copied a ``True`` flag out of our ``__dict__``.
    The marker is the function itself, so a copy names ours and not itself.

    Local rather than shared: the ``.pth`` bootstrap loads each adapter by file
    path, so an adapter must not import a sibling module.
    """
    seen: set[int] = set()
    while obj is not None and id(obj) not in seen:
        seen.add(id(obj))
        if getattr(obj, ADAPTED, None) is obj:
            return True
        obj = getattr(obj, "__wrapped__", None)
    return False


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

    setattr(compute_mm_prefix_ranges, ADAPTED, compute_mm_prefix_ranges)
    # vllm_omni.utils.mm_prefix wraps this with functools.wraps and checks its
    # own marker attribute, both of which need a normal function object.
    compute_mm_prefix_ranges.__name__ = "compute_mm_prefix_ranges"
    compute_mm_prefix_ranges.__qualname__ = "compute_mm_prefix_ranges"
    return compute_mm_prefix_ranges


def is_applied(module: ModuleType) -> list[str]:
    """Which names are adapted **right now**, read-only. See the sibling adapter."""
    live = getattr(module, "compute_mm_prefix_ranges", None)
    return ["compute_mm_prefix_ranges"] if _wraps_the_bridge(live) else []


def is_satisfied(module: ModuleType) -> bool:
    """Whether the module can serve the call Omni's patch makes.

    The patch reads the attribute; anything callable under that name satisfies
    it, whether the pin grew its own or this adapter installed the bridge. The
    loader needs this to tell a stale entry from an adapter that matched
    nothing, which is the only one worth failing an import over.
    """
    return getattr(module, "compute_mm_prefix_ranges", None) is not None


def apply(module: ModuleType) -> list[str]:
    """Add the 0.33 range builder to the pinned attention-backend utilities."""
    existing = getattr(module, "compute_mm_prefix_ranges", None)
    if existing is not None and not _wraps_the_bridge(existing):
        return []  # the pin provides its own; nothing to adapt
    if existing is not None:
        return []  # already adapted
    module.compute_mm_prefix_ranges = _bridge()
    return ["compute_mm_prefix_ranges"]
