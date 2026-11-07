# SPDX-License-Identifier: Apache-2.0
"""TileLang's on-disk kernel cache keys on str(target), which must carry the device arch."""

import pytest
import torchada  # noqa: F401
import torch

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "musa") and torch.musa.is_available()),
    reason="needs a MUSA device",
)


def test_tilelang_musa_target_carries_the_device_arch():
    pytest.importorskip("tilelang")
    from tvm.target import Target

    major, minor = torch.musa.get_device_capability()
    target = Target("musa")
    assert str(target.attrs["arch"]) == f"mp_{major}{minor}"
    assert f'"arch":"mp_{major}{minor}"' in str(target)
