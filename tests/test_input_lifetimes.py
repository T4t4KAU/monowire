"""Queueing and idle receivers must not retain expanded request activations."""

import asyncio
import gc
import queue
import threading
import unittest
import weakref
from types import SimpleNamespace

import torch

from monowire.channels import ReceivedMessage
from monowire.continuous import ContinuousConfig, ContinuousScheduler, PreparedStep
from monowire.models.h3.adapter import H3RequestAdapter
from monowire.topology import PipelineTopology
from monowire.transport import PipelineExecutor, _receive_packets


class PausedChannel:
    """Hand off one tensor, then block until the test allows STOP."""

    def __init__(self):
        self.waiting = threading.Event()
        self.resume = threading.Event()
        self.reference = None

    def recv(self):
        if self.reference is None:
            hidden = torch.zeros(16)
            self.reference = weakref.ref(hidden)
            return ReceivedMessage({"items": [{"hidden": hidden}]})
        self.waiting.set()
        if not self.resume.wait(5):
            raise TimeoutError("Test did not release receiver")
        return ReceivedMessage({"stop": True})


class InputLifetimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_hidden_materialization_is_bounded_and_resumes_each_step(self):
        materialized = []

        class Adapter(H3RequestAdapter):
            def __init__(self):
                self.pipeline = object()

            async def admit(self, key, value):
                return SimpleNamespace(request_id=key, step_index=0)

            def _prepare_step(self, state):
                hidden = torch.zeros(16)
                materialized.append((state.request_id, state.step_index, weakref.ref(hidden)))
                return PreparedStep(
                    state.request_id, state.step_index, False, "test", {"hidden": hidden}
                )

            def prepare_batch(self, prepared):
                # Exercise the production lazy-materialization boundary without H3 weights.
                from unittest.mock import patch

                with patch(
                    "monowire.models.h3.adapter.prepare_transformer_batch", side_effect=lambda x: x
                ):
                    return super().prepare_batch(prepared)

            def commit(self, state, prepared, output):
                assert state.step_index == prepared.step
                state.step_index += 1

            def finished(self, state):
                return state.step_index == 2

            async def finish(self, state):
                return state.step_index

            def release(self, state):
                pass

        class Executor:
            can_submit = False

            def start(self, *, on_capacity):
                self.on_capacity = on_capacity
                return asyncio.get_running_loop().create_future()

            def submit(self, prepared):
                futures = [asyncio.get_running_loop().create_future() for _ in prepared]
                for future in futures:
                    future.set_result(None)
                return futures

        executor = Executor()
        scheduler = ContinuousScheduler(Adapter(), executor, ContinuousConfig(12, 2))
        outputs = [scheduler.submit(str(i), None) for i in range(12)]
        try:
            for _ in range(40):
                await asyncio.sleep(0)
            self.assertEqual(len(scheduler.active), 12)
            self.assertEqual(len(materialized), 4)  # Two buffered batches, not all active requests.
            self.assertEqual(sum(len(bucket) for bucket in scheduler.ingress.values()), 8)
            self.assertTrue(all(ref() is not None for _, _, ref in materialized))
            scheduler.cancel("11")  # Cancellation before materialization allocates no hidden.
            executor.can_submit = True
            executor.on_capacity()
            results = await asyncio.wait_for(asyncio.gather(*outputs, return_exceptions=True), 3)
            self.assertEqual(results[:-1], [2] * 11)
            self.assertIsInstance(results[-1], asyncio.CancelledError)
            self.assertEqual(
                {(key, step) for key, step, _ in materialized},
                {(str(i), step) for i in range(11) for step in range(2)},
            )
        finally:
            executor.can_submit = True
            executor.on_capacity()
            await scheduler.close()
        gc.collect()
        self.assertTrue(all(ref() is None for _, _, ref in materialized))

    async def test_suffix_receiver_drops_consumed_packet_before_next_receive(self):
        channel = PausedChannel()
        incoming = queue.Queue()
        thread = threading.Thread(
            target=_receive_packets, args=(channel, incoming, 0, threading.Event(), 2)
        )
        thread.start()
        try:
            packet = await asyncio.to_thread(incoming.get, True, 2)
            packet.release()
            del packet
            self.assertTrue(await asyncio.to_thread(channel.waiting.wait, 2))
            gc.collect()
            self.assertIsNone(channel.reference())
        finally:
            channel.resume.set()
            await asyncio.to_thread(thread.join, 3)
        self.assertFalse(thread.is_alive())

    async def test_ingress_receiver_drops_delivered_packet_before_next_receive(self):
        channel = PausedChannel()
        stage = SimpleNamespace(group=SimpleNamespace(rank_in_group=0))
        executor = PipelineExecutor(stage, PipelineTopology((1, 1), 1), {(1, 0): channel})
        delivered = asyncio.Event()
        executor._deliver = lambda message: delivered.set()
        executor.closed = True  # Permit the final STOP without starting send workers.
        receiver = asyncio.create_task(executor._receive())
        try:
            await asyncio.wait_for(delivered.wait(), 2)
            self.assertTrue(await asyncio.to_thread(channel.waiting.wait, 2))
            gc.collect()
            self.assertIsNone(channel.reference())
        finally:
            channel.resume.set()
            await asyncio.wait_for(receiver, 3)
