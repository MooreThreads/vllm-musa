# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_contract_checks() -> None:
    import importlib.util
    from types import SimpleNamespace
    from unittest import mock

    package_root = REPO_ROOT / "vllm_musa"
    spec = importlib.util.spec_from_file_location(
        "vllm_musa",
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    assert spec is not None and spec.loader is not None
    candidate = importlib.util.module_from_spec(spec)
    sys.modules["vllm_musa"] = candidate
    spec.loader.exec_module(candidate)

    import torch

    from vllm_musa.model_executor.layers.fused_moe.router import (
        fused_topk_router as plain_router,
    )
    from vllm_musa.model_executor.layers.fused_moe.router import (
        grouped_topk_router as grouped_router,
    )

    class FakeTopKModule:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def topk_softmax(
            self,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            gating_output: torch.Tensor,
            renormalize: bool,
            *,
            correction_bias: torch.Tensor | None,
            shared_expert_gate_output: torch.Tensor | None,
            num_fused_shared_experts: int,
        ) -> None:
            self.calls.append(
                {
                    "shape": tuple(topk_weights.shape),
                    "renormalize": renormalize,
                    "correction_bias": correction_bias,
                    "shared": shared_expert_gate_output,
                    "num_shared": num_fused_shared_experts,
                }
            )

    fake_topk = FakeTopKModule()
    hidden_states = torch.empty(2, 3072)
    router_logits = torch.empty(2, 256)
    shared_logits = torch.empty(2, 1)
    with (
        mock.patch.object(
            grouped_router, "_can_use_musa_jit_topk", return_value=True
        ),
        mock.patch.object(
            grouped_router, "_maybe_import_musa_jit_topk", return_value=fake_topk
        ),
    ):
        result = grouped_router._musa_jit_fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=8,
            renormalize=True,
            indices_type=None,
            scoring_func="softmax",
            shared_expert_gate_output=shared_logits,
            num_fused_shared_experts=1,
        )

    assert result is not None
    topk_weights, topk_ids = result
    assert topk_weights.shape == (2, 9)
    assert topk_ids.shape == (2, 9)
    assert fake_topk.calls == [
        {
            "shape": (2, 9),
            "renormalize": True,
            "correction_bias": None,
            "shared": shared_logits,
            "num_shared": 1,
        }
    ]

    def unexpected_import() -> None:
        raise AssertionError("unsupported shared shape must fall back")

    with (
        mock.patch.object(
            grouped_router, "_can_use_musa_jit_topk", return_value=True
        ),
        mock.patch.object(
            grouped_router,
            "_maybe_import_musa_jit_topk",
            side_effect=unexpected_import,
        ),
    ):
        for experts, renormalize in ((255, False), (257, True)):
            result = grouped_router._musa_jit_fused_topk(
                hidden_states=hidden_states,
                gating_output=torch.empty(2, experts),
                topk=8,
                renormalize=renormalize,
                indices_type=None,
                scoring_func="softmax",
                shared_expert_gate_output=shared_logits,
                num_fused_shared_experts=1,
            )
            assert result is None

    gate_inputs: list[torch.Tensor] = []
    jit_calls: list[dict[str, object]] = []

    def shared_gate(states: torch.Tensor):
        gate_inputs.append(states)
        return shared_logits, None

    expected = (torch.empty(2, 9), torch.empty(2, 9, dtype=torch.int32))

    def fake_jit(**kwargs):
        jit_calls.append(kwargs)
        return expected

    router = SimpleNamespace(
        top_k=8,
        renormalize=False,
        scoring_func="softmax",
        _musa_shared_gate=shared_gate,
        _musa_shared_expert_id=256,
    )
    with mock.patch.object(plain_router, "_musa_jit_fused_topk", fake_jit):
        result = plain_router._compute_routing(
            router,
            hidden_states,
            router_logits,
            None,
        )

    assert result is expected
    assert gate_inputs == [hidden_states]
    assert len(jit_calls) == 1
    assert jit_calls[0]["shared_expert_gate_output"] is shared_logits
    assert jit_calls[0]["num_fused_shared_experts"] == 1

    routed_weights = torch.empty(2, 8)
    routed_ids = torch.empty(2, 8, dtype=torch.int32)
    extended = (torch.empty(2, 9), torch.empty(2, 9, dtype=torch.int32))
    extension_calls: list[tuple[object, ...]] = []

    def fake_extend(*args):
        extension_calls.append(args)
        return extended

    router = SimpleNamespace(
        top_k=8,
        renormalize=False,
        scoring_func="softmax",
        _musa_shared_gate=lambda states: (shared_logits, None),
        _musa_shared_expert_id=256,
    )
    with (
        mock.patch.object(plain_router, "_musa_jit_fused_topk", return_value=None),
        mock.patch.object(
            plain_router,
            "fused_topk",
            return_value=(routed_weights, routed_ids, None),
        ),
        mock.patch.object(plain_router, "extend_topk_with_shared", fake_extend),
    ):
        result = plain_router._compute_routing(
            router,
            hidden_states,
            router_logits,
            None,
        )

    assert result[0] is extended[0]
    assert result[1] is extended[1]
    assert extension_calls == [(routed_weights, routed_ids, shared_logits, 256)]


def test_router_contract_in_isolated_process() -> None:
    env = os.environ.copy()
    env["VLLM_PLUGINS"] = ""
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    if sys.argv[1:] != ["--child"]:
        raise SystemExit("expected --child")
    _run_contract_checks()
