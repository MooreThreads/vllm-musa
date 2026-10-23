# SPDX-License-Identifier: Apache-2.0
"""Source contracts for the MUSA FlashInfer SSD route."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SERIES = ROOT / "vllm_musa" / "patches" / "series"

ROUTE = "0170-MUSA-route-Mamba2-SSD-to-FlashInfer.patch"
SELECT = "0171-MUSA-select-Mamba2-SSD-backend.patch"

GATE = '+    if x.device.type == "musa" and backend == "flashinfer":'


def _read(name: str) -> str:
    return (SERIES / name).read_text()


def test_route_member_adds_the_backend_parameter_and_a_musa_gate():
    source = _read(ROUTE)

    # ``backend`` is the trailing parameter of the varlen signature; the next
    # line of the member is the context line that closes the parameter list.
    assert "+    backend=None,\n ):" in source
    assert GATE in source
    # The branch returns before the Triton pipeline is entered.
    assert source.index(GATE) < source.index(
        "     varlen_states = _mamba_chunk_scan_combined_fwd("
    )


def test_route_member_forwards_the_order_sensitive_arguments_by_name():
    source = _read(ROUTE)

    # FlashInfer orders initial_states, dt_softplus, dt_limit while MATE
    # orders dt_softplus, dt_limit, initial_states, so a positional forward
    # lands dt_limit's tuple where the kernel reads initial_states.
    assert "initial_states=initial_states," in source
    assert "dt_softplus=dt_softplus," in source
    assert "dt_limit=dt_limit," in source


def test_route_member_is_not_switchable_by_an_environment_variable():
    source = _read(ROUTE)

    assert "VLLM_MUSA_FLASHINFER_SSD" not in source
    assert "os.environ" not in source
    assert "getenv" not in source


def test_selection_member_passes_the_backend_at_every_ssd_call_site():
    source = _read(SELECT)

    # The pinned mamba_mixer2.py calls mamba_chunk_scan_combined_varlen at
    # two SSD sites, the autotune warmup and the prefill scan; the kernel
    # tests call it without the argument and keep the Triton default.
    assert source.count("backend=self.mamba_config.backend.value") == 2
