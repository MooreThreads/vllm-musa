# SPDX-License-Identifier: Apache-2.0
"""Every conv-state offset must stay a 64-bit address.

The conv state pool is addressed in elements, so a large pool overflows a
signed 32-bit offset long before the tensor itself is unreasonable. Three
kernels were widened to ``int64`` while the width4 vector kernel still used
``int32``; gate the whole module so the next kernel cannot regress silently.
"""

import re
from pathlib import Path

MODULE = (
    Path(__file__).resolve().parent.parent
    / "vllm_musa"
    / "jit_kernel"
    / "tilelang"
    / "causal_conv1d.py"
)

STATE_BASE = re.compile(r"state_base = T\.alloc_(?:var|local)\([^\n]*\)")
OFFSET = re.compile(r"T\.Cast\(\"int64\", cache_idx\) \* state_stride_seq")


def test_every_conv_state_offset_is_64_bit():
    declarations = STATE_BASE.findall(MODULE.read_text(encoding="utf-8"))

    assert declarations, "no state_base declaration found: the contract moved"
    for declaration in declarations:
        assert '"int64"' in declaration, declaration


WIDENED = 'T.Cast("int64", cache_idx) * state_stride_seq'
UNWIDENED = re.compile(r'(?<!T\.Cast\("int64", )cache_idx \* state_stride_seq')


def test_the_offset_product_is_widened_before_it_is_multiplied():
    """An int64 destination does not promote an int32 multiply."""
    source = MODULE.read_text(encoding="utf-8")

    widened = source.count(WIDENED)
    assert widened == 4, f"expected four widened offsets, found {widened}"
    assert not UNWIDENED.search(source), "an offset still multiplies in int32"
