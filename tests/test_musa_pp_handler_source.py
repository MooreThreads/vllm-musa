# SPDX-License-Identifier: Apache-2.0
"""Exercise MusaPPHandler feedback ordering and drain behavior without GPUs."""

import ast
import unittest
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock


class Array:
    """Minimal one-dimensional NumPy double for the exercised bookkeeping."""

    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, index):
        if isinstance(index, Array):
            return Array(self.values[i] for i in index.values)
        return self.values[index]

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            self.values[index] = [value] * len(self.values[index])
        else:
            self.values[index] = value

    def binary(self, other, op):
        values = (
            other.values if isinstance(other, Array) else [other] * len(self.values)
        )
        return Array(op(a, b) for a, b in zip(self.values, values))

    def __add__(self, other):
        return self.binary(other, lambda a, b: a + b)

    def __ge__(self, other):
        return self.binary(other, lambda a, b: a >= b)

    def __ne__(self, other):
        return self.binary(other, lambda a, b: a != b)

    def __or__(self, other):
        return self.binary(other, lambda a, b: a or b)

    def __invert__(self):
        return Array(not value for value in self.values)

    def any(self):
        return any(self.values)

    def all(self):
        return all(self.values)


np = NS(array=Array, zeros=lambda size, dtype: Array([0] * size), int32="int32")

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "third_party/vllm/vllm/v1/worker/gpu/pp_utils.py").read_text()
tree = ast.parse(source)
tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
namespace = dict(globals(), dataclass=dataclass, field=field, deque=deque)
exec(
    compile(
        "from __future__ import annotations\n" + ast.unparse(tree), "candidate", "exec"
    ),
    namespace,
)
pp_handler_cls = namespace["PPHandler"]
musa_source = ROOT / "vllm_musa/v1/worker/pp_handler.py"
musa_tree = ast.parse(musa_source.read_text())
musa_tree.body = [node for node in musa_tree.body if isinstance(node, ast.ClassDef)]
exec("from __future__ import annotations\n" + ast.unparse(musa_tree), namespace)


class TestMusaPPHandler(unittest.TestCase):

    def setUp(self):
        global torch, get_pp_group, current_platform
        self.log = []
        torch = MagicMock()
        current_platform = NS(device_type="musa", is_xpu=lambda: False)
        self.pp = NS(
            world_size=2,
            is_last_rank=False,
            last_rank=1,
            make_sibling_device_group=MagicMock(return_value=object()),
        )
        get_pp_group = lambda: self.pp
        namespace.update(
            torch=torch, get_pp_group=get_pp_group, current_platform=current_platform
        )
        self.handler_cls = namespace["MusaPPHandler"]
        self.h = self.handler_cls(4, 0, "mock")
        self.assertIs(self.h.main_stream, self.h.broadcast_stream)
        torch.empty.side_effect = lambda *args, **kw: NS(
            unbind=lambda dim: (1, 2), record_stream=lambda s: None, shape=args
        )
        torch.distributed.broadcast.side_effect = lambda t, **kw: self.log.append(
            t.shape
        )
        self.h.broadcast_stream.wait_stream.side_effect = lambda s: self.log.append(
            "M->B"
        )
        torch.cuda.Event.side_effect = lambda: NS(
            record=lambda s: self.log.append("record")
        )
        self.h.main_stream.wait_event.side_effect = lambda e: self.log.append("wait")
        self.b = NS(
            num_reqs=1,
            idx_mapping_np=np.array([0]),
            idx_mapping="map",
            num_computed_tokens_np=np.array([1]),
            prefill_len_np=np.array([1]),
            num_scheduled_tokens=1,
            max_seq_len_np=None,
        )

    def test_receive_is_submitted_before_the_slot_is_consumed(self):
        self.assertTrue(self.h.receive(self.b))
        self.assertEqual(self.log, [])
        self.assertIsNone(self.h.get_prev_sampled_outputs())
        self.assertIsNotNone(self.h.get_prev_sampled_outputs())
        self.assertEqual(self.log, [(1, 1), (2, 1), "record", "wait"])
        self.pp.make_sibling_device_group.assert_called_once_with(
            group_desc="pp_broadcast"
        )
        for issue in torch.distributed.broadcast.call_args_list:
            self.assertEqual(issue.kwargs["src"], self.pp.last_rank)
            self.assertIs(
                issue.kwargs["group"], self.pp.make_sibling_device_group.return_value
            )

    def test_failed_pair_is_not_reissued_or_followed_by_another_slot(self):
        for failure in ("token", "count", "event"):
            with self.subTest(failure=failure):
                handler = self.handler_cls(4, 0, "mock")
                handler.receive(self.b)
                handler.get_prev_sampled_outputs()
                handler.receive(self.b)
                self.assertEqual(sum(slot is not None for slot in handler.queue), 2)
                failure_error = RuntimeError(failure)
                issue = torch.distributed.broadcast
                issue.reset_mock()
                issue.side_effect = {
                    "token": [failure_error],
                    "count": [None, failure_error],
                    "event": [None, None],
                }[failure]
                if failure == "event":
                    handler.queue[0].event.record = MagicMock(side_effect=failure_error)
                with self.assertRaisesRegex(RuntimeError, failure):
                    handler.submit_pending_feedback()
                handler.submit_pending_feedback()
                self.assertEqual(issue.call_count, 1 if failure == "token" else 2)

    def test_multiple_slots_are_flushed_once_in_order(self):
        self.h.receive(self.b)
        self.h.get_prev_sampled_outputs()
        self.b.num_reqs = 2
        self.b.idx_mapping_np = np.array([0, 1])
        self.b.num_computed_tokens_np = self.b.prefill_len_np = np.array([1, 1])
        self.h.receive(self.b)
        self.log.clear()
        self.h.submit_pending_feedback([NS(wait=lambda: self.log.append("activation"))])
        self.h.submit_pending_feedback()
        self.h.get_prev_sampled_outputs()
        self.h.get_prev_sampled_outputs()
        self.assertEqual(
            self.log,
            [
                "activation",
                (1, 1),
                (2, 1),
                "record",
                (2, 1),
                (2, 2),
                "record",
                "wait",
                "wait",
            ],
        )

    def test_upstream_handler_still_posts_feedback_at_receive(self):
        handler = pp_handler_cls(4, 0, "mock")
        handler.broadcast_stream.wait_stream.side_effect = (
            lambda source: self.log.append("M->B")
        )
        handler.receive(self.b)
        self.assertEqual(self.log, ["M->B", (1, 1), (2, 1)])

    def test_no_payload(self):
        self.b.prefill_len_np[:] = 10
        self.assertFalse(self.h.receive(self.b))
        self.assertEqual(self.log, [])

    def test_sender_returns_without_waiting_and_next_recv_joins_latest_feedback(self):
        self.pp.is_last_rank = True
        handler = self.handler_cls(4, 0, "mock")
        self.assertIsNot(handler.main_stream, handler.broadcast_stream)
        actual_caller = MagicMock()
        torch.cuda.current_stream.return_value = actual_caller
        self.b.idx_mapping = MagicMock()
        tokens = MagicMock(dtype=torch.int64)
        tokens.shape = (1, 1)
        torch.distributed.broadcast.side_effect = (
            lambda *args, **kwargs: self.log.append("broadcast")
        )
        events = iter(("older", "newer"))

        def record():
            self.log.append("record")
            return next(events)

        handler.broadcast_stream.record_event.side_effect = record
        handler.before_activation_recv()
        handler.broadcast(tokens, MagicMock(), MagicMock(), self.b)
        handler.broadcast(tokens, MagicMock(), MagicMock(), self.b)
        self.assertEqual(self.log, ["broadcast", "broadcast", "record"] * 2)
        actual_caller.wait_event.assert_not_called()
        handler.before_activation_recv()
        handler.before_activation_recv()
        actual_caller.wait_event.assert_called_once_with("newer")

    def test_freed_slot_is_not_consumed(self):
        self.h.receive(self.b)
        self.h.on_req_idx_freed(0)
        self.assertIsNone(self.h.get_prev_sampled_outputs())
        self.assertIsNone(self.h.get_prev_sampled_outputs())
        self.assertNotIn("wait", self.log)
        self.assertEqual(self.log[-3:], [(1, 1), (2, 1), "record"])

    def lifecycle_worker(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "third_party/vllm/vllm/v1/worker/gpu_worker.py"
        )
        runner = next(
            node
            for node in ast.parse(source.read_text()).body
            if getattr(node, "name", None) == "Worker"
        )
        runner.body = [
            node
            for node in runner.body
            if getattr(node, "name", "")
            in ("_drain_pp_feedback", "shutdown", "synchronize_device")
        ]
        timeline = self.log

        class WorkerBase:
            def synchronize_device(self):
                timeline.append("sync")

        scope = dict(
            WorkerBase=WorkerBase,
            gc=MagicMock(),
            ensure_kv_transfer_shutdown=None,
            ensure_ec_transfer_shutdown=None,
            current_platform=NS(is_cuda_alike=lambda: False),
        )
        exec(ast.unparse(runner), scope)
        worker = scope["Worker"]()
        worker.model_runner = NS(shutdown=lambda: self.log.append("shutdown"))
        worker._musa_pp_handler = self.h
        worker.elastic_ep_executor = MagicMock()
        worker.profiler = None
        worker._pp_send_work = [NS(wait=lambda: self.log.append("activation"))]
        return worker

    def test_shutdown_flushes_pending_feedback(self):
        worker = self.lifecycle_worker()
        self.h.receive(self.b)
        worker.shutdown()
        self.assertEqual(self.log, ["activation", (1, 1), (2, 1), "record", "shutdown"])

    def test_pause_after_abort_drains_owed_feedback_without_consuming_slot(self):
        worker = self.lifecycle_worker()
        self.h.receive(self.b)
        self.h.on_req_idx_freed(0)
        self.assertIsNone(self.h.get_prev_sampled_outputs())
        worker.synchronize_device()
        worker.synchronize_device()
        self.assertIsNone(self.h.get_prev_sampled_outputs())
        self.assertEqual(
            self.log, ["activation", (1, 1), (2, 1), "record", "sync", "sync"]
        )

    def test_pause_without_a_slot_does_not_wait_for_payload_work(self):
        worker = self.lifecycle_worker()
        self.b.prefill_len_np[:] = 10
        self.assertFalse(self.h.receive(self.b))
        worker.synchronize_device()
        self.assertEqual(self.log, ["sync"])

    def test_keep_pause_leaves_posted_slot_for_resume(self):
        worker = self.lifecycle_worker()
        self.h.receive(self.b)
        slots = list(self.h.queue)
        worker.synchronize_device()
        worker.synchronize_device()
        self.assertEqual(list(self.h.queue), slots)
        self.assertIsNone(self.h.get_prev_sampled_outputs())
        self.assertIsNotNone(self.h.get_prev_sampled_outputs())
        worker.shutdown()
        self.assertEqual(
            self.log,
            [
                "activation",
                (1, 1),
                (2, 1),
                "record",
                "sync",
                "sync",
                "wait",
                "shutdown",
            ],
        )


if __name__ == "__main__":
    unittest.main()
