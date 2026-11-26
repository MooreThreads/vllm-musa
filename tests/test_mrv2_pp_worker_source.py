# SPDX-License-Identifier: Apache-2.0
"""Exercise MRv2 runner selection and worker PP hooks without GPUs."""

import ast
import os
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "third_party/vllm/vllm/v1/worker/gpu_worker.py"


class TestMRv2PPWorker(unittest.TestCase):
    def test_runner_dispatches_only_for_musa_pp(self):
        runner = ast.parse((WORKER.parent / "gpu/model_runner.py").read_text())
        (selection,) = [
            node
            for node in ast.walk(runner)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "self.use_pp"
            and "self.pp_handler =" in ast.unparse(node)
        ]
        for kind, depth, deferred in (
            ("musa", 1, False),
            ("musa", 2, True),
            ("musa", 2, False),
            ("cuda", 2, False),
        ):
            with self.subTest(kind=kind, depth=depth, deferred=deferred):
                upstream, musa = MagicMock(), MagicMock()
                runner_instance = SimpleNamespace(
                    parallel_config=object(),
                    use_pp=depth > 1,
                    pp_handler=None,
                    max_num_reqs=4,
                    num_speculative_steps=0,
                    device="device",
                )
                selector = MagicMock(return_value=musa if deferred else upstream)
                plugin = kind == "musa" and depth > 1
                module = SimpleNamespace(select_pp_handler=selector) if plugin else None
                with patch.dict(
                    sys.modules,
                    {"vllm_musa.v1.worker.pp_handler": module},
                ):
                    exec(
                        ast.unparse(selection),
                        dict(
                            PPHandler=upstream,
                            current_platform=SimpleNamespace(
                                is_musa=lambda: kind == "musa"
                            ),
                            self=runner_instance,
                        ),
                    )
                handler = runner_instance.pp_handler
                if plugin:
                    selector.assert_called_once_with(runner_instance.parallel_config, 0)
                else:
                    selector.assert_not_called()
                self.assertEqual(hasattr(runner_instance, "musa_pp_handler"), deferred)
                if deferred:
                    self.assertIs(runner_instance.musa_pp_handler, handler)
                if depth == 1:
                    self.assertIsNone(handler)
                    upstream.assert_not_called()
                    musa.assert_not_called()
                    continue
                selected, unused = (musa, upstream) if deferred else (upstream, musa)
                selected.assert_called_once_with(
                    max_num_reqs=4, num_speculative_steps=0, device="device"
                )
                unused.assert_not_called()
                self.assertIs(handler, selected.return_value)

    def test_handler_policy(self):
        path = ROOT / "vllm_musa/v1/worker/pp_handler.py"
        selector = next(
            node
            for node in ast.parse(path.read_text()).body
            if getattr(node, "name", "") == "select_pp_handler"
        )
        upstream, musa = type("PPHandler", (), {}), type("MusaPPHandler", (), {})
        scope = dict(
            os=os,
            logger=MagicMock(),
            PPHandler=upstream,
            MusaPPHandler=musa,
        )
        exec("from __future__ import annotations\n" + ast.unparse(selector), scope)
        for mode, depth, spec, expected, warns in (
            (None, 2, 0, upstream, False),
            (None, 3, 1, upstream, False),
            ("upstream", 3, 0, upstream, False),
            ("upstream", 2, 1, upstream, False),
            ("musa", 2, 0, musa, False),
            ("musa", 3, 0, musa, True),
            ("musa", 2, 1, musa, True),
            ("musa", 3, 1, musa, True),
            ("auto", 2, 0, ValueError, False),
            ("typo", 2, 0, ValueError, False),
        ):
            with (
                self.subTest(mode=mode, depth=depth, spec=spec),
                patch.dict(os.environ, clear=True),
            ):
                if mode is not None:
                    os.environ["VLLM_MUSA_MRV2_PP_HANDLER"] = mode
                config = SimpleNamespace(
                    pipeline_parallel_size=depth,
                    distributed_executor_backend="mp",
                )
                warning = scope["logger"].warning_once
                warning.reset_mock()
                if expected is ValueError:
                    with self.assertRaises(ValueError):
                        scope["select_pp_handler"](config, spec)
                else:
                    self.assertIs(scope["select_pp_handler"](config, spec), expected)
                if warns:
                    warning.assert_called_once()
                    message, *values = warning.call_args.args
                    self.assertEqual(values, [depth, spec])
                    self.assertIn("experimental", message % tuple(values))
                else:
                    warning.assert_not_called()

    @staticmethod
    def worker_method(name):
        worker_cls = next(
            node
            for node in ast.parse(WORKER.read_text()).body
            if getattr(node, "name", "") == "Worker"
        )
        method = next(
            node for node in worker_cls.body if getattr(node, "name", "") == name
        )
        method.decorator_list = []
        return "from __future__ import annotations\n" + ast.unparse(method)

    def test_worker_uses_reference_returned_by_runner(self):
        scope = dict(
            init_workspace_manager=MagicMock(), _num_workspace_lanes=MagicMock()
        )
        exec(self.worker_method("_init_workspace_and_model_runner"), scope)
        handler = object()
        musa_runner = SimpleNamespace(pp_handler=handler, musa_pp_handler=handler)
        for runner, use_v2, expected in (
            (musa_runner, True, handler),
            (SimpleNamespace(pp_handler=handler), True, None),
            (SimpleNamespace(), False, None),
        ):
            worker = SimpleNamespace(
                device="mock",
                rank=1,
                use_v2_model_runner=use_v2,
                vllm_config=SimpleNamespace(
                    parallel_config=SimpleNamespace(enable_dbo=False)
                ),
                _make_model_runner=lambda: runner,
            )
            scope["_init_workspace_and_model_runner"](worker)
            self.assertIs(worker._musa_pp_handler, expected)

    def test_first_stage_flushes_between_forward_and_activation_send(self):
        timeline = []
        output = SimpleNamespace(tensors={"hidden": object()})
        group = SimpleNamespace(is_first_rank=True, is_last_rank=False)
        send_work = object()

        def send(tensors, *, all_gather_group):
            self.assertIs(tensors, output.tensors)
            timeline.append("send")
            return [object(), send_work]

        group.isend_tensor_dict = send
        scope = dict(
            get_pp_group=lambda: group,
            get_tp_group=lambda: None,
            ModelRunnerOutput=str,
            AsyncModelRunnerOutput=int,
            NoneType=type(None),
            IntermediateTensors=SimpleNamespace,
        )
        exec(self.worker_method("execute_model"), scope)
        for use_v2, ordered in ((False, False), (True, False), (True, True)):
            for produces_activation in (False, True):
                with self.subTest(v2=use_v2, sends=produces_activation):
                    timeline.clear()

                    def forward(scheduler, intermediate):
                        timeline.append("forward")
                        return output if produces_activation else None

                    runner = SimpleNamespace(
                        is_pooling_model=False, execute_model=forward
                    )
                    worker = SimpleNamespace(
                        _musa_pp_handler=(
                            SimpleNamespace(
                                submit_pending_feedback=lambda: timeline.append("flush")
                            )
                            if ordered
                            else None
                        ),
                        model_runner=runner,
                        use_v2_model_runner=use_v2,
                        _pp_send_work=[
                            SimpleNamespace(
                                wait=lambda: timeline.append("previous send")
                            )
                        ],
                        annotate_profile=lambda _: nullcontext(),
                        vllm_config=SimpleNamespace(
                            parallel_config=SimpleNamespace(
                                distributed_executor_backend="mp"
                            )
                        ),
                    )
                    scope["execute_model"](
                        worker, SimpleNamespace(total_num_scheduled_tokens=1)
                    )
                    expected = ["previous send", "forward"]
                    if produces_activation:
                        expected.extend(["flush", "send"] if ordered else ["send"])
                        self.assertEqual(worker._pp_send_work, [send_work])
                    self.assertEqual(timeline, expected)

    def test_last_stage_joins_feedback_only_before_real_activation_recv(self):
        timeline = []

        def receive(*, all_gather_group):
            timeline.append("irecv")
            return {}, [], []

        group = SimpleNamespace(
            is_first_rank=False, is_last_rank=True, irecv_tensor_dict=receive
        )
        scope = dict(
            get_pp_group=lambda: group,
            get_tp_group=lambda: None,
            ModelRunnerOutput=str,
            AsyncModelRunnerOutput=int,
            NoneType=type(None),
            AsyncIntermediateTensors=lambda tensors, **kwargs: tensors,
        )
        exec(self.worker_method("execute_model"), scope)
        runner = SimpleNamespace(
            is_pooling_model=False,
            execute_model=lambda _, intermediate: timeline.append("forward"),
        )
        worker = SimpleNamespace(
            model_runner=runner,
            _musa_pp_handler=SimpleNamespace(
                before_activation_recv=lambda: timeline.append("join")
            ),
            use_v2_model_runner=True,
            _pp_send_work=[],
            annotate_profile=lambda _: nullcontext(),
        )
        for scheduled, expected in [
            (0, ["forward"]),
            (1, ["join", "irecv", "forward"]),
        ]:
            timeline.clear()
            scope["execute_model"](
                worker, SimpleNamespace(total_num_scheduled_tokens=scheduled)
            )
            self.assertEqual(timeline, expected)


if __name__ == "__main__":
    unittest.main()
