# SPDX-License-Identifier: Apache-2.0
"""The tactic tables, their validation, and the resolve ladder (no device needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_musa import tactics
from vllm_musa.tactics import MoeTactic, TuningProblem

MOE_DEFAULT = {
    "block_m": 64,
    "block_n": 64,
    "block_k": 32,
    "split_k": 1,
    "num_stages": 3,
}
MOE_MP56 = {"block_m": 32, "block_n": 128, "block_k": 64, "split_k": 2, "num_stages": 2}


class Spec:
    """A stand-in for ``MusaDeviceSpec``: same attributes, no torch, no GPU."""

    def __init__(self, arch_name: str = "mp_31", mp_count: int = 56) -> None:
        self.arch_name = arch_name
        self.mp_count = mp_count
        self.warp_size = 32
        self.shared_memory_per_block = 96 * 1024
        self.shared_memory_per_multiprocessor = 192 * 1024
        self.regs_per_multiprocessor = 64 * 1024
        self.max_threads_per_block = 1024
        self.max_threads_per_multiprocessor = 2048


def write_table(
    directory: Path,
    entries: list[dict],
    *,
    kernel: str = "moe",
    schema: int = tactics.SCHEMA_VERSION,
) -> Path:
    path = directory / f"{kernel}.json"
    path.write_text(
        json.dumps({"_schema": schema, "kernel": kernel, "tactics": entries})
    )
    return path


def entry(values: dict, *, arch=None, mp_count=None, m_range=None, note="") -> dict:
    return {
        "arch": arch,
        "mp_count": mp_count,
        "m_range": m_range,
        "values": values,
        "note": note,
    }


def problem(m: int = 8, *, spec: Spec | None = None) -> TuningProblem:
    return TuningProblem(
        layer=tactics.LayerShape(
            shape_n=4096, shape_k=4096, dtype="bf16", weight_dtype="fp8"
        ),
        m=m,
        device=tactics.DeviceProfile.from_spec(spec or Spec()),
    )


def moe_problem(m: int, *, num_experts: int, top_k: int) -> TuningProblem:
    return TuningProblem(
        layer=tactics.LayerShape(
            shape_n=4096,
            shape_k=4096,
            dtype="bf16",
            weight_dtype="fp8",
            num_experts=num_experts,
            top_k=top_k,
        ),
        m=m,
        device=tactics.DeviceProfile.from_spec(Spec()),
    )


@pytest.fixture()
def table(tmp_path: Path) -> Path:
    write_table(
        tmp_path,
        [
            entry(MOE_DEFAULT),
            entry(MOE_MP56, arch="mp_31", mp_count=56, note="measured on 10.20.36.7"),
        ],
    )
    return tmp_path


def test_exact_bin_wins(table: Path) -> None:
    assert tactics.lookup("moe", problem(), table) == MoeTactic.from_values(MOE_MP56)


def test_an_untuned_bin_falls_back_to_the_table_row(tmp_path: Path) -> None:
    write_table(
        tmp_path,
        [
            entry(MOE_DEFAULT),
            entry(MOE_MP56, arch="mp_31", mp_count=60, note="measured on 10.20.36.7"),
        ],
    )
    got = tactics.lookup("moe", problem(spec=Spec(mp_count=56)), tmp_path)
    assert got == MoeTactic.from_values(MOE_DEFAULT)


def test_an_arch_wide_row_covers_every_bin_of_that_arch(tmp_path: Path) -> None:
    write_table(
        tmp_path,
        [
            entry(MOE_DEFAULT),
            entry(MOE_MP56, arch="mp_31", note="measured on 56; assumed for 48-60"),
        ],
    )
    for mp_count in (48, 52, 56, 60):
        got = tactics.lookup("moe", problem(spec=Spec(mp_count=mp_count)), tmp_path)
        assert got == MoeTactic.from_values(MOE_MP56)


def test_a_missing_bin_fails_closed_without_a_fallback_row(tmp_path: Path) -> None:
    write_table(
        tmp_path,
        [entry(MOE_MP56, arch="mp_31", mp_count=56, note="measured on 10.20.36.7")],
    )
    with pytest.raises(tactics.TacticNotFound) as excinfo:
        tactics.lookup("moe", problem(spec=Spec(mp_count=60)), tmp_path)
    message = str(excinfo.value)
    assert "mp_31" in message and "mp_count=60" in message
    assert "default=" in message


def test_a_missing_table_is_a_lookup_failure(tmp_path: Path) -> None:
    with pytest.raises(tactics.TacticNotFound):
        tactics.load("moe", tmp_path)


def test_a_schema_mismatch_is_refused(tmp_path: Path) -> None:
    write_table(tmp_path, [entry(MOE_DEFAULT)], schema=1)
    with pytest.raises(ValueError, match="schema"):
        tactics.load("moe", tmp_path)


def test_the_schema_1_default_arch_sentinel_is_refused(tmp_path: Path) -> None:
    write_table(tmp_path, [entry(MOE_DEFAULT, arch="default")])
    with pytest.raises(ValueError, match="not an arch name"):
        tactics.load("moe", tmp_path)


def test_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    write_table(
        tmp_path,
        [
            entry(MOE_DEFAULT, arch="mp_31", mp_count=56, note="a"),
            entry(MOE_MP56, arch="mp_31", mp_count=56, note="b"),
        ],
    )
    with pytest.raises(ValueError, match="duplicate"):
        tactics.load("moe", tmp_path)


def test_overlapping_bands_at_one_specificity_are_rejected(tmp_path: Path) -> None:
    write_table(
        tmp_path,
        [
            entry(MOE_DEFAULT, arch="mp_31", mp_count=56, m_range=[1, 16], note="a"),
            entry(MOE_MP56, arch="mp_31", mp_count=56, m_range=[16, 64], note="b"),
        ],
    )
    with pytest.raises(ValueError, match="overlap"):
        tactics.load("moe", tmp_path)


def test_a_band_row_beside_a_bandless_row_at_one_specificity_is_rejected(
    tmp_path: Path,
) -> None:
    write_table(
        tmp_path,
        [
            entry(MOE_DEFAULT, arch="mp_31", mp_count=56, note="a"),
            entry(MOE_MP56, arch="mp_31", mp_count=56, m_range=[1, 16], note="b"),
        ],
    )
    with pytest.raises(ValueError, match="redundant"):
        tactics.load("moe", tmp_path)


def test_a_measured_row_must_carry_a_note(tmp_path: Path) -> None:
    write_table(tmp_path, [entry(MOE_MP56, arch="mp_31", mp_count=56)])
    with pytest.raises(ValueError, match="note"):
        tactics.load("moe", tmp_path)


def test_the_generic_row_may_go_without_a_note(tmp_path: Path) -> None:
    write_table(tmp_path, [entry(MOE_DEFAULT)])
    assert tactics.lookup("moe", problem(), tmp_path) == MoeTactic.from_values(
        MOE_DEFAULT
    )


def test_an_unknown_value_key_is_rejected(tmp_path: Path) -> None:
    typo = dict(MOE_DEFAULT)
    typo["block_N"] = 64
    write_table(tmp_path, [entry(typo)])
    with pytest.raises(ValueError, match="unknown value key"):
        tactics.load("moe", tmp_path)


def test_a_missing_value_key_is_rejected(tmp_path: Path) -> None:
    write_table(tmp_path, [entry({"block_m": 64})])
    with pytest.raises(ValueError, match="missing value key"):
        tactics.load("moe", tmp_path)


def test_a_non_integer_value_is_rejected(tmp_path: Path) -> None:
    write_table(tmp_path, [entry(dict(MOE_DEFAULT, block_m="64"))])
    with pytest.raises(ValueError, match="integer"):
        tactics.load("moe", tmp_path)


def test_a_non_positive_value_is_rejected(tmp_path: Path) -> None:
    write_table(tmp_path, [entry(dict(MOE_DEFAULT, split_k=0))])
    with pytest.raises(ValueError, match="positive"):
        tactics.load("moe", tmp_path)


def test_a_malformed_band_is_rejected(tmp_path: Path) -> None:
    for bad in ([16, 1], [0, 4], [1], "1-16", [1, 2, 3]):
        write_table(
            tmp_path, [entry(MOE_DEFAULT, arch="mp_31", mp_count=56, m_range=bad)]
        )
        with pytest.raises(ValueError, match="m_range"):
            tactics.load("moe", tmp_path)


def test_the_band_is_inclusive_on_both_ends(tmp_path: Path) -> None:
    write_table(
        tmp_path,
        [
            entry(MOE_DEFAULT, arch="mp_31", mp_count=56, m_range=[4, 8], note="a"),
            entry(MOE_MP56, arch="mp_31", mp_count=56, m_range=[9, 32], note="b"),
        ],
    )
    for m in (4, 8):
        assert tactics.lookup("moe", problem(m=m), tmp_path) == MoeTactic.from_values(
            MOE_DEFAULT
        )
    for m in (9, 32):
        assert tactics.lookup("moe", problem(m=m), tmp_path) == MoeTactic.from_values(
            MOE_MP56
        )


def test_a_moe_problem_is_keyed_on_rows_per_expert(tmp_path: Path) -> None:
    """64 tokens over 4 experts with top-2 routing is 32 rows per expert, not 64."""
    write_table(
        tmp_path,
        [
            entry(MOE_DEFAULT, arch="mp_31", mp_count=56, m_range=[1, 31], note="a"),
            entry(MOE_MP56, arch="mp_31", mp_count=56, m_range=[32, 128], note="b"),
        ],
    )
    moe = moe_problem(64, num_experts=4, top_k=2)
    assert tactics.lookup("moe", moe, tmp_path) == MoeTactic.from_values(MOE_MP56)


def test_the_shipped_tables_parse_and_resolve() -> None:
    for kernel in tactics.REGISTRY:
        assert tactics.load(kernel), kernel
    # The shipped tables carry one generic row each, and it is the only row that
    # is allowed to be generic: that is what makes an untuned bin resolvable
    # today without pretending it was measured.
    for kernel in tactics.REGISTRY:
        generic = [row for row in tactics.load(kernel) if row.arch is None]
        assert len(generic) == 1, kernel
        assert generic[0].mp_count is None and generic[0].m_range is None
        assert "not tuned" in generic[0].note


def test_the_package_ships_the_tables() -> None:
    shipped = Path(tactics.__file__).resolve().parent
    for kernel in tactics.REGISTRY:
        assert (shipped / f"{kernel}.json").is_file()


def test_an_unknown_kernel_names_the_known_ones() -> None:
    with pytest.raises(KeyError, match="known kernels"):
        tactics.tactic_class("not_a_kernel")
