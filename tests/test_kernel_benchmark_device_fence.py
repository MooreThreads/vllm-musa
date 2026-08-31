import pytest

from benchmarks.kernel_tactics import _benchmark_utils as utils


def test_device_fence_accepts_exact_visibility_and_uuid(monkeypatch):
    for name in (
        "MTHREADS_VISIBLE_DEVICES",
        "MUSA_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.setenv(name, "2")
    monkeypatch.setattr(
        utils,
        "command_output",
        lambda command: (
            "0, uuid-other, MTT TEST DEVICE\n"
            "2, uuid-target, MTT TEST DEVICE\n"
            "3, uuid-another, MTT TEST DEVICE"
        ),
    )

    receipt = utils.verify_lease_device_fence(2, "uuid-target")

    assert receipt["passed"] is True
    assert receipt["actual_physical_device"] == 2


@pytest.mark.parametrize(
    ("env_name", "env_value"),
    [
        ("MTHREADS_VISIBLE_DEVICES", "1"),
        ("MUSA_VISIBLE_DEVICES", "1"),
        ("CUDA_VISIBLE_DEVICES", "1"),
    ],
)
def test_device_fence_rejects_visibility_drift(monkeypatch, env_name, env_value):
    for name in (
        "MTHREADS_VISIBLE_DEVICES",
        "MUSA_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.setenv(name, "2")
    monkeypatch.setenv(env_name, env_value)

    with pytest.raises(RuntimeError, match="visibility"):
        utils.verify_lease_device_fence(2, "uuid-target")


def test_device_fence_rejects_uuid_drift(monkeypatch):
    for name in (
        "MTHREADS_VISIBLE_DEVICES",
        "MUSA_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.setenv(name, "2")
    monkeypatch.setattr(
        utils,
        "command_output",
        lambda command: "2, uuid-wrong, MTT TEST DEVICE",
    )

    with pytest.raises(RuntimeError, match="expected"):
        utils.verify_lease_device_fence(2, "uuid-target")


def test_device_fence_accepts_legacy_full_query(monkeypatch):
    for name in (
        "MTHREADS_VISIBLE_DEVICES",
        "MUSA_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.setenv(name, "3")
    outputs = iter(
        [
            None,
            """GPU2 00000000:4b:00.0
    Product Name : MTT TEST DEVICE
    GPU UUID : uuid-other
GPU3 00000000:5c:00.0
    Product Name : MTT TEST DEVICE
    GPU UUID : uuid-target
GPU4 00000000:9a:00.0
    Product Name : MTT TEST DEVICE
    GPU UUID : uuid-another
""",
        ]
    )
    monkeypatch.setattr(utils, "command_output", lambda command: next(outputs))

    receipt = utils.verify_lease_device_fence(3, "uuid-target")

    assert receipt["passed"] is True
    assert receipt["mthreads_gmi_query_mode"] == "legacy-full-query"
    assert receipt["mthreads_gmi_query"] == ("3, uuid-target, MTT TEST DEVICE")
