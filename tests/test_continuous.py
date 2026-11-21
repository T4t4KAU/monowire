"""Completion-driven advancement, bounded admission and cancellation."""

import asyncio
import unittest

from monowire.continuous import ContinuousConfig, ContinuousScheduler, PreparedStep


class Adapter:
    def __init__(self):
        self.released = []

    def admit(self, key, value):
        return dict(key=key, step=0, total=value, value=0)

    def prepare(self, state):
        return PreparedStep(
            state["key"], state["step"], state["key"] != "slow", "same", state["value"]
        )

    def commit(self, state, prepared, output):
        assert prepared.step == state["step"]
        state["step"] += 1
        state["value"] = output

    def finished(self, state):
        return state["step"] == state["total"]

    def finish(self, state):
        return state["value"]

    def release(self, state):
        self.released.append(state["key"])


class ContinuousTests(unittest.IsolatedAsyncioTestCase):
    async def test_fast_request_advances_and_replacement_enters_before_slow_finishes(self):
        gate = asyncio.Event()
        events = []
        adapter = Adapter()

        class Executor:
            async def execute(self, batch):
                if any(p.request_id == "slow" for p in batch):
                    await gate.wait()
                await asyncio.sleep(0)
                return [p.payload + 1 for p in batch]

        def event(value):
            events.append(value)
            if value["event"] == "admitted" and value["request_id"] == "replacement":
                gate.set()

        scheduler = ContinuousScheduler(adapter, Executor(), ContinuousConfig(2, 2, 1), event)
        slow = scheduler.submit("slow", 1)
        fast = scheduler.submit("fast", 3)
        replacement = scheduler.submit("replacement", 2)
        await asyncio.wait_for(scheduler.close(), 5)
        self.assertEqual([slow.result(), fast.result(), replacement.result()], [1, 3, 2])
        slow_end = next(
            i
            for i, e in enumerate(events)
            if e["event"] == "step_completed" and e["request_id"] == "slow"
        )
        fast_end = next(
            i
            for i, e in enumerate(events)
            if e["event"] == "released" and e["request_id"] == "fast"
        )
        admitted = next(
            i
            for i, e in enumerate(events)
            if e["event"] == "admitted" and e["request_id"] == "replacement"
        )
        self.assertLess(fast_end, admitted)
        self.assertLess(admitted, slow_end)
        self.assertCountEqual(adapter.released, ["slow", "fast", "replacement"])

    async def test_cancel_waiting_and_inflight_and_continue_after_batch_error(self):
        gate = asyncio.Event()
        entered = asyncio.Event()
        adapter = Adapter()

        class Executor:
            async def execute(self, batch):
                entered.set()
                await gate.wait()
                if batch[0].request_id == "bad":
                    raise ValueError("injected batch failure")
                return [p.payload + 1 for p in batch]

        scheduler = ContinuousScheduler(adapter, Executor(), ContinuousConfig(1, 1, 0))
        first = scheduler.submit("first", 3)
        waiting = scheduler.submit("waiting", 1)
        waiting.cancel()
        await entered.wait()
        first.cancel()
        bad = scheduler.submit("bad", 1)
        good = scheduler.submit("good", 2)
        gate.set()
        await asyncio.wait_for(scheduler.close(), 5)
        self.assertTrue(first.cancelled())
        self.assertTrue(waiting.cancelled())
        self.assertIsInstance(bad.exception(), ValueError)
        self.assertEqual(good.result(), 2)
        self.assertCountEqual(adapter.released, ["first", "bad", "good"])

    async def test_duplicate_identity_and_late_arrival(self):
        class Executor:
            async def execute(self, batch):
                return [p.payload + 1 for p in batch]

        scheduler = ContinuousScheduler(Adapter(), Executor(), ContinuousConfig(2, 2, 1))
        first = scheduler.submit("a", 2)
        with self.assertRaises(ValueError):
            scheduler.submit("a", 1)
        self.assertEqual(await first, 2)
        second = scheduler.submit("b", 1)
        self.assertEqual(await second, 1)
        await scheduler.close()
        with self.assertRaises(RuntimeError):
            scheduler.submit("c", 1)

    async def test_observer_error_does_not_abort_inference(self):
        class Executor:
            async def execute(self, batch):
                return [p.payload + 1 for p in batch]

        def observer(event):
            raise ValueError("broken observer")

        adapter = Adapter()
        scheduler = ContinuousScheduler(adapter, Executor(), ContinuousConfig(1, 1, 0), observer)
        with self.assertLogs("monowire", level="ERROR"):
            future = scheduler.submit("one", 2)
            await scheduler.close()
        self.assertEqual(future.result(), 2)
        self.assertEqual(adapter.released, ["one"])

    async def test_fatal_shutdown_drains_before_releasing_request(self):
        gate, entered = asyncio.Event(), asyncio.Event()

        class Executor:
            async def execute(self, batch):
                entered.set()
                await gate.wait()
                return [p.payload + 1 for p in batch]

        adapter = Adapter()
        scheduler = ContinuousScheduler(adapter, Executor(), ContinuousConfig(1, 1, 0))
        future = scheduler.submit("one", 2)
        await entered.wait()
        scheduler.task.cancel()
        await asyncio.sleep(0)
        self.assertEqual(adapter.released, [])
        with self.assertRaises(RuntimeError):
            scheduler.submit("two", 1)
        gate.set()
        with self.assertRaises(asyncio.CancelledError):
            await scheduler.close()
        self.assertIsInstance(future.exception(), RuntimeError)
        self.assertEqual(adapter.released, ["one"])


if __name__ == "__main__":
    unittest.main()
