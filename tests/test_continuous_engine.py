"""Public async service lifecycle, source failures and early consumer exit."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from test_continuous import Adapter

from monowire.continuous import ContinuousConfig
from monowire.continuous_engine import ContinuousEngine


class ServiceAdapter(Adapter):
    def admit(self, key, value):
        return super().admit(key, value[1])


class Executor:
    def __init__(self):
        self.closed = False

    async def execute(self, batch):
        await asyncio.sleep(0)
        return [p.payload + 1 for p in batch]

    async def close(self):
        self.closed = True


class EngineTests(IsolatedAsyncioTestCase):
    def engine(self):
        runtime = SimpleNamespace(
            rank=0, context=nullcontext, create_adapter=lambda: None, create_executor=lambda: None
        )
        engine = ContinuousEngine(runtime, ContinuousConfig(2, 2, 0))
        return engine

    async def test_live_source_and_cleanup(self):
        adapter, executor = ServiceAdapter(), Executor()
        engine = self.engine()

        async def source():
            for i in range(5):
                yield str(i), "prompt", i + 1
                await asyncio.sleep(0)

        with (
            patch.object(engine.runtime, "create_adapter", return_value=adapter),
            patch.object(engine.runtime, "create_executor", return_value=executor),
        ):
            outputs = [item async for item in engine.stream(source())]
        self.assertEqual(
            {o.request_id: o.output for o in outputs}, {str(i): i + 1 for i in range(5)}
        )
        self.assertTrue(executor.closed)
        self.assertFalse(engine.running)

    async def test_source_error_does_not_hang_or_hide_error(self):
        engine = self.engine()
        executor = Executor()

        async def source():
            yield "one", "prompt", 1
            raise ValueError("source failed")

        with (
            patch.object(engine.runtime, "create_adapter", return_value=ServiceAdapter()),
            patch.object(engine.runtime, "create_executor", return_value=executor),
        ):
            with self.assertRaisesRegex(ValueError, "source failed"):
                async for _ in engine.stream(source()):
                    pass
        self.assertTrue(executor.closed)

    async def test_consumer_close_drains_inflight(self):
        engine = self.engine()
        executor = Executor()
        adapter = ServiceAdapter()

        async def source():
            for i in range(10):
                yield str(i), "prompt", 1 if i == 0 else 20

        with (
            patch.object(engine.runtime, "create_adapter", return_value=adapter),
            patch.object(engine.runtime, "create_executor", return_value=executor),
        ):
            stream = engine.stream(source())
            await anext(stream)
            await asyncio.wait_for(stream.aclose(), 5)
        self.assertTrue(executor.closed)
        self.assertFalse(engine.running)

    async def test_request_error_does_not_abort_other_requests(self):
        class RejectingAdapter(ServiceAdapter):
            def admit(self, key, value):
                if key == "bad":
                    raise ValueError("invalid request")
                return super().admit(key, value)

        engine = self.engine()
        executor = Executor()

        async def source():
            yield "bad", "prompt", 1
            yield "good", "prompt", 2

        with (
            patch.object(engine.runtime, "create_adapter", return_value=RejectingAdapter()),
            patch.object(engine.runtime, "create_executor", return_value=executor),
        ):
            outputs = {o.request_id: o async for o in engine.stream(source())}
        self.assertEqual(outputs["bad"].error, "invalid request")
        self.assertEqual(outputs["good"].output, 2)

    async def test_cancel_api(self):
        engine = self.engine()
        engine.on_event = lambda e: engine.cancel("cancelled") if e["event"] == "admitted" else None

        async def source():
            yield "cancelled", "prompt", 10

        with (
            patch.object(engine.runtime, "create_adapter", return_value=ServiceAdapter()),
            patch.object(engine.runtime, "create_executor", return_value=Executor()),
        ):
            outputs = [o async for o in engine.stream(source())]
        self.assertEqual(outputs[0].error, "Request cancelled")

    async def test_slow_consumer_bounds_completed_results(self):
        engine = self.engine()
        adapter, executor = ServiceAdapter(), Executor()
        produced = []

        async def source():
            for i in range(100):
                produced.append(i)
                yield str(i), "prompt", 1

        with (
            patch.object(engine.runtime, "create_adapter", return_value=adapter),
            patch.object(engine.runtime, "create_executor", return_value=executor),
        ):
            stream = engine.stream(source())
            await anext(stream)
            try:
                for _ in range(100):
                    await asyncio.sleep(0)
                # One consumed result, bounded submitted work, and one source lookahead.
                self.assertLessEqual(len(produced), 2 * engine.config.max_active_requests + 2)
            finally:
                await stream.aclose()
        self.assertTrue(executor.closed)

    async def test_scheduler_failure_closes_executor_without_hanging(self):
        engine = self.engine()
        executor = Executor()

        async def source():
            yield "one", "prompt", 1
            await asyncio.Event().wait()

        async def collect():
            return [item async for item in engine.stream(source())]

        with (
            patch.object(engine.runtime, "create_adapter", return_value=ServiceAdapter()),
            patch.object(engine.runtime, "create_executor", return_value=executor),
            patch(
                "monowire.continuous.ContinuousScheduler._dispatch",
                side_effect=RuntimeError("scheduler failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "scheduler failed"):
                await asyncio.wait_for(collect(), 1)
        self.assertTrue(executor.closed)
