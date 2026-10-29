# SPDX-License-Identifier: Apache-2.0
"""Source contracts for the MUSA FlashInfer SSD route.

The route is a build-time series member, so this test reads the patch text the
build installs rather than importing a vLLM tree. It pins the parts a reviewer
cannot see by running the suite on a host where `--mamba-backend flashinfer`
happens to work:

* the FlashInfer call forwards the ordering-hazard arguments **by name**;
* routing stays off for the checkpointed (intermediate-state) prefill;
* every SSD call site in the Mamba2 mixer selects the configured backend.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SERIES = ROOT / "vllm_musa" / "patches" / "series"

ROUTE = "0209-MUSA-route-Mamba2-SSD-to-FlashInfer.patch"
PROFILE = "0210-MUSA-initialize-the-Mamba2-profiling-output.patch"

GATE = (
    "+    if (\n"
    '+        x.device.type == "musa"\n'
    '+        and backend == "flashinfer"\n'
    "+        and not return_intermediate_states\n"
    "+    ):"
)


def _read(name: str) -> str:
    return (SERIES / name).read_text()


def test_route_member_adds_the_backend_parameter_and_a_musa_gate():
    source = _read(ROUTE)

    # ``backend`` is the trailing parameter of the varlen signature; the next
    # line of the member is the context line that closes the parameter list.
    assert "+    backend=None,\n ):" in source
    assert GATE in source
    # The branch returns before the reference pipeline is entered.
    assert source.index(GATE) < source.index(
        "     varlen_states = _mamba_chunk_scan_combined_fwd("
    )


def test_route_member_forwards_the_order_sensitive_arguments_by_name():
    source = _read(ROUTE)

    # FlashInfer orders initial_states, dt_softplus, dt_limit while MATE
    # orders dt_softplus, dt_limit, initial_states, so a positional forward
    # lands dt_limit's tuple where the kernel reads initial_states. `out` and
    # `state_dtype` keep the caller's buffer and cache dtype.
    assert "initial_states=initial_states," in source
    assert "dt_softplus=dt_softplus," in source
    assert "dt_limit=dt_limit," in source
    assert "out=out," in source
    assert "state_dtype=state_dtype," in source


def test_route_member_is_not_switchable_by_an_environment_variable():
    source = _read(ROUTE)

    assert "VLLM_MUSA_FLASHINFER_SSD" not in source
    assert "os.environ" not in source
    assert "getenv" not in source


def test_route_member_keeps_the_checkpointed_prefill_on_the_reference_path():
    source = _read(ROUTE)

    # `return_intermediate_states` is set from `checkpoint_chunk_idx is not
    # None`; those states feed `store_prefill_checkpoint` in vLLM's layout.
    assert "and not return_intermediate_states" in GATE
    assert "store_prefill_checkpoint" in source


def test_selection_member_passes_the_backend_at_every_ssd_call_site():
    source = _read(ROUTE)

    # The pinned mamba_mixer2.py calls mamba_chunk_scan_combined_varlen at
    # two SSD sites, the autotune warmup and the prefill scan; the kernel
    # tests call it without the argument and keep the reference default.
    assert source.count("backend=self.mamba_config.backend.value") == 2


def test_profiling_member_initializes_the_supplied_output_on_musa():
    source = _read(PROFILE)

    assert "+            if current_platform.is_musa():" in source
    assert "+                output.zero_()" in source
    # The initialization belongs to the attn_metadata is None profile branch.
    assert source.index("if attn_metadata is None:") < source.index("output.zero_()")
