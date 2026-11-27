# SPDX-License-Identifier: Apache-2.0
"""Source contract for the MUSA rejection of ``--use-fp64-gumbel``.

The flag cannot be honoured on MUSA: float64 works there, but MUDNN has no
float64 reduce-with-indices kernel and the V2 Gumbel sampler calls
``local_max.argmax(dim=-1)`` on a float64 buffer, which aborts the warmup
profile run. The guard must fire at config time, before any engine starts.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLATFORM = ROOT / "vllm_musa/platform.py"


def _check_and_update_config_body() -> str:
    source = PLATFORM.read_text(encoding="utf-8")
    marker = "def check_and_update_config(cls, vllm_config"
    assert marker in source, "MUSAPlatformBase.check_and_update_config is gone"
    body = source.split(marker, 1)[1]
    # Stop at the next method definition so the search stays inside this hook.
    return body.split("\n    def ", 1)[0]


def test_guard_runs_before_the_rest_of_config_setup() -> None:
    body = _check_and_update_config_body()
    guard_at = body.find("use_fp64_gumbel")
    first_work_at = body.find("force_triton_attn_for_diffusion")
    assert guard_at != -1, "the --use-fp64-gumbel guard disappeared"
    assert first_work_at != -1, "the diffusion backend pin must still be present"
    assert guard_at < first_work_at, (
        "the guard must run before the rest of config setup so the failure is "
        "reported as a config error, not as a mid-init MUDNN error"
    )


def test_guard_names_the_flag_and_the_mudnn_limit() -> None:
    body = _check_and_update_config_body()
    assert "--use-fp64-gumbel is not supported on MUSA" in body
    assert "ReduceIndicesOnlyCall MUDNN failed in: RunIndices" in body, (
        "the message must quote the concrete failure so users can search it"
    )
    assert "argmax" in body, "the message must name the failing op"


def test_guard_raises_instead_of_silently_downgrading() -> None:
    body = _check_and_update_config_body()
    assert "raise ValueError(" in body, (
        "the flag is an exactness request; silently falling back to float32 "
        "would change the numerics the user asked for"
    )
    assert "use_fp64_gumbel = False" not in body
    assert "use_fp64_gumbel=False" not in body
