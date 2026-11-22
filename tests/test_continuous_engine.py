"""Public async service lifecycle, source failures and early consumer exit."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from test_continuous import Adapter, ServiceExecutor, batch_futures

from monowire.continuous import ContinuousConfig
from monowire.continuous_engine import ContinuousEngine


class ServiceAdapter(Adapter):
    async def admit(self, key, value):
        return await super().admit(key, value[1])


class Executor(ServiceExecutor):
    def __init__(self):
        self.closed = False

    def submit(self, batch):
        return batch_futures(self.run(batch), len(batch))

    async def run(self, batch):
        await asyncio.sleep(0)
        return [p.payload + 1 for p in batch]

    async def close(self):
        self.closed = True


class EngineTests(IsolatedAsyncioTestCase):
    def engine(self):
        runtime = SimpleNamespace(
            rank=0,
            ingress_replicas=1,
            context=nullcontext,
            create_adapter=lambda: None,
            create_executor=lambda: None,
        )
        engine = ContinuousEngine(runtime, ContinuousConfig(2, 2))
        return engine

    async def test_cancellation_churn_does_not_retain_pending_requests(self):
        engine = self.engine()
        engine.config = ContinuousConfig(1, 1)
        entered, gate = asyncio.Event(), asyncio.Event()
        consumed = asyncio.Event()

        class HeldExecutor(Executor):
            async def run(self, batch):
                entered.set()
                await gate.wait()
                return [p.payload + 1 for p in batch]

        async def source():
            try:
                yield "held", "prompt", 1
                await entered.wait()
                for _ in range(100):
                    consumed.clear()
                    yield "queued", "prompt", 1
                    self.assertTrue(engine.cancel("queued"))
                    await consumed.wait()
                    self.assertFalse(engine._scheduler.pending)
                    self.assertEqual(engine._scheduler.ids, {"held"})
            finally:
                gate.set()

        async def collect():
            outputs = []
            async for item in engine.stream(source()):
                outputs.append(item)
                if item.request_id == "queued":
                    self.assertEqual(item.error, "Request cancelled")
                    consumed.set()
            return outputs

        with (
            patch.object(engine.runtime, "create_adapter", return_value=ServiceAdapter()),
            patch.object(engine.runtime, "create_executor", return_value=HeldExecutor()),
        ):
            outputs = await asyncio.wait_for(collect(), 3)
        self.assertEqual(len(outputs), 101)
        self.assertEqual(outputs[-1].request_id, "held")

    async def test_idle_transport_failure_stops_source_before_and_between_arrivals(self):
        import torch
        from test_transport import MemoryChannel, RoutedStage

        from monowire.interfaces import PipelineFailure
        from monowire.topology import PipelineTopology
        from monowire.transport import PipelineExecutor

        class MissAdapter(ServiceAdapter):
            def prepare(self, state):
                item = super().prepare(state)
                item.hit = False
                item.payload = torch.zeros(1, 1)
                return item

        for first_request in (False, True):
            with self.subTest(first_request=first_request):
                engine = self.engine()
                outgoing, returning = MemoryChannel(), MemoryChannel()
                executor = PipelineExecutor(
                    RoutedStage(0, "cpu"),
                    PipelineTopology((1, 1)),
                    {(0, 1): outgoing, (1, 0): returning},
                )
                source_idle, first_done, source_closed = (
                    asyncio.Event(),
                    asyncio.Event(),
                    asyncio.Event(),
                )

                async def source():
                    try:
                        if first_request:
                            yield "one", "prompt", 1
                        source_idle.set()
                        await asyncio.Event().wait()
                    finally:
                        source_closed.set()

                async def peer():
                    await source_idle.wait()
                    if first_request:
                        packet = (await asyncio.to_thread(outgoing.recv)).value
                        item = packet["items"][0]
                        returning.send({"items": [item]})
                        await first_done.wait()
                    self.assertFalse(executor.waiting)
                    returning.send({"failure": "idle link failed"})
                    self.assertTrue((await asyncio.to_thread(outgoing.recv)).value["stop"])
                    returning.send({"stop": True})

                async def collect():
                    async for _ in engine.stream(source()):
                        first_done.set()

                with (
                    patch.object(engine.runtime, "create_adapter", return_value=MissAdapter()),
                    patch.object(engine.runtime, "create_executor", return_value=executor),
                ):
                    peer_task = asyncio.create_task(peer())
                    try:
                        with self.assertRaisesRegex(PipelineFailure, "idle link failed"):
                            await asyncio.wait_for(collect(), 2)
                    finally:
                        await asyncio.wait_for(peer_task, 2)
                self.assertTrue(source_closed.is_set())
                self.assertTrue(executor.closed)
                self.assertTrue(executor.receiver.done())
                self.assertFalse(engine.running)

    async def test_second_ingress_owns_an_independent_stream(self):
        engine = self.engine()
        engine.rank = 1
        engine.ingress_replicas = 2
        adapter, executor = ServiceAdapter(), Executor()

        async def source():
            yield "replica-1-request", "prompt", 3

        with (
            patch.object(engine.runtime, "create_adapter", return_value=adapter),
            patch.object(engine.runtime, "create_executor", return_value=executor),
        ):
            outputs = [item async for item in engine.stream(source())]
        self.assertEqual([(o.request_id, o.output) for o in outputs], [("replica-1-request", 3)])
        self.assertTrue(executor.closed)

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
            async def admit(self, key, value):
                if key == "bad":
                    raise ValueError("invalid request")
                return await super().admit(key, value)

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

    async def test_backpressured_source_closes_before_worker_cleanup(self):
        engine = self.engine()
        engine.config = ContinuousConfig(2, 1)
        source_closed = asyncio.Event()
        order = []
        produced = []

        class ClosingExecutor(Executor):
            async def run(self, batch):
                if any(item.request_id != "0" for item in batch):
                    await source_closed.wait()
                return await super().run(batch)

            async def close(self):
                order.append("executor")
                await super().close()

        async def source():
            try:
                for index in range(100):
                    produced.append(index)
                    yield str(index), "prompt", 1
            finally:
                await asyncio.sleep(0)
                order.append("source")
                source_closed.set()

        # An iterable may return a distinct iterator. Close that iterator,
        # without requiring the original iterable to implement aclose.
        class Requests:
            def __aiter__(self):
                return requests

        requests = source()
        with (
            patch.object(engine.runtime, "create_adapter", return_value=ServiceAdapter()),
            patch.object(engine.runtime, "create_executor", return_value=ClosingExecutor()),
        ):
            stream = engine.stream(Requests())
            await anext(stream)
            for _ in range(100):
                if len(produced) == 6:
                    break
                await asyncio.sleep(0)
            self.assertEqual(len(produced), 6)
            await asyncio.wait_for(stream.aclose(), 2)
        self.assertIsNone(requests.ag_frame)
        self.assertEqual(order, ["source", "executor"])
        self.assertFalse(engine.running)

    async def test_source_cleanup_failure_still_closes_workers(self):
        engine = self.engine()
        executor = Executor()
        closed = []

        class ClosingAdapter(ServiceAdapter):
            async def close(self):
                closed.append("adapter")

        class Requests:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

            async def aclose(self):
                raise ValueError("source cleanup failed")

        with (
            patch.object(engine.runtime, "create_adapter", return_value=ClosingAdapter()),
            patch.object(engine.runtime, "create_executor", return_value=executor),
        ):
            with self.assertRaisesRegex(ValueError, "source cleanup failed"):
                async for _ in engine.stream(Requests()):
                    pass
        self.assertTrue(executor.closed)
        self.assertEqual(closed, ["adapter"])

    async def test_early_close_reports_source_cleanup_failure_and_closes_workers(self):
        for waiting_for_input in (True, False):
            with self.subTest(waiting_for_input=waiting_for_input):
                engine = self.engine()
                engine.config = ContinuousConfig(1, 1)
                source_blocked = asyncio.Event()
                order = []

                class ClosingExecutor(Executor):
                    async def close(self):
                        order.append("executor")
                        await super().close()

                class ClosingAdapter(ServiceAdapter):
                    async def close(self):
                        order.append("adapter")

                async def source():
                    try:
                        if waiting_for_input:
                            yield "one", "prompt", 1
                            source_blocked.set()
                            await asyncio.Event().wait()
                        else:
                            for index in range(10):
                                if index == 3:
                                    source_blocked.set()
                                yield str(index), "prompt", 1
                    finally:
                        order.append("source")
                        raise ValueError("source cleanup failed")

                with (
                    patch.object(engine.runtime, "create_adapter", return_value=ClosingAdapter()),
                    patch.object(engine.runtime, "create_executor", return_value=ClosingExecutor()),
                ):
                    stream = engine.stream(source())
                    await anext(stream)
                    await asyncio.wait_for(source_blocked.wait(), 1)
                    with self.assertRaisesRegex(ValueError, "source cleanup failed"):
                        await asyncio.wait_for(stream.aclose(), 2)
                self.assertEqual(order, ["source", "executor", "adapter"])
                self.assertFalse(engine.running)

    async def test_repeated_close_cancellation_preserves_source_cleanup_failure(self):
        engine = self.engine()
        finalizing, leave = asyncio.Event(), asyncio.Event()
        order = []

        class ClosingExecutor(Executor):
            async def close(self):
                order.append("executor")
                await super().close()

        class ClosingAdapter(ServiceAdapter):
            async def close(self):
                order.append("adapter")

        async def source():
            try:
                yield "one", "prompt", 1
                await asyncio.Event().wait()
            finally:
                finalizing.set()
                await leave.wait()
                order.append("source")
                raise ValueError("source cleanup failed")

        with (
            patch.object(engine.runtime, "create_adapter", return_value=ClosingAdapter()),
            patch.object(engine.runtime, "create_executor", return_value=ClosingExecutor()),
        ):
            stream = engine.stream(source())
            await anext(stream)
            closing = asyncio.create_task(stream.aclose())
            try:
                await asyncio.wait_for(finalizing.wait(), 1)
                for _ in range(2):
                    closing.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(closing.done())
                    self.assertFalse(order)
                leave.set()
                with self.assertRaises(asyncio.CancelledError) as caught:
                    await closing
                self.assertIsInstance(caught.exception.__cause__, ValueError)
                self.assertEqual(str(caught.exception.__cause__), "source cleanup failed")
            finally:
                leave.set()
                await asyncio.gather(closing, return_exceptions=True)
        self.assertEqual(order, ["source", "executor", "adapter"])
        self.assertFalse(engine.running)

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

    async def test_transport_failure_stops_an_open_ended_source(self):
        from monowire.interfaces import PipelineFailure

        class FailedExecutor(Executor):
            def submit(self, batch):
                futures = []
                for _ in batch:
                    future = asyncio.get_running_loop().create_future()
                    future.set_exception(PipelineFailure("broken link"))
                    futures.append(future)
                return futures

        executor, engine = FailedExecutor(), self.engine()

        async def source():
            yield "one", "prompt", 1
            await asyncio.Event().wait()

        with (
            patch.object(engine.runtime, "create_adapter", return_value=ServiceAdapter()),
            patch.object(engine.runtime, "create_executor", return_value=executor),
        ):

            async def collect():
                return [item async for item in engine.stream(source())]

            with self.assertRaisesRegex(PipelineFailure, "broken link"):
                await asyncio.wait_for(collect(), 2)
        self.assertTrue(executor.closed)

    async def test_failure_notifies_source_before_waiting_for_admission(self):
        from monowire.interfaces import PipelineFailure

        for cancel_scheduler in (False, True):
            with self.subTest(cancel_scheduler=cancel_scheduler):
                engine, executor = self.engine(), Executor()
                entered, source_closed, release_io = (
                    asyncio.Event(),
                    asyncio.Event(),
                    asyncio.Event(),
                )

                class HeldAdapter(ServiceAdapter):
                    async def admit(self, key, value):
                        entered.set()
                        await release_io.wait()
                        return await super().admit(key, value)

                adapter = HeldAdapter()

                async def source():
                    try:
                        yield "held", "prompt", 1
                        await asyncio.Event().wait()
                    finally:
                        source_closed.set()
                        release_io.set()

                async def collect():
                    return [item async for item in engine.stream(source())]

                with (
                    patch.object(engine.runtime, "create_adapter", return_value=adapter),
                    patch.object(engine.runtime, "create_executor", return_value=executor),
                ):
                    task = asyncio.create_task(collect())
                    try:
                        await asyncio.wait_for(entered.wait(), 1)
                        if cancel_scheduler:
                            engine._scheduler.task.cancel()
                        else:
                            executor.failure_signal.set_result(PipelineFailure("fatal link"))
                        # Do not cancel the collector on failure: that would
                        # close the source externally and hide the deadlock.
                        await asyncio.wait_for(source_closed.wait(), 1)
                        expected = asyncio.CancelledError if cancel_scheduler else PipelineFailure
                        with self.assertRaises(expected):
                            await asyncio.wait_for(asyncio.shield(task), 1)
                    finally:
                        release_io.set()
                        await asyncio.gather(task, return_exceptions=True)
                self.assertEqual(adapter.released, ["held"])
                self.assertTrue(executor.closed)
                self.assertFalse(engine.running)

    async def test_repeated_consumer_cancellation_finishes_source_and_worker_cleanup(self):
        engine = self.engine()
        entered, finalizing, leave = asyncio.Event(), asyncio.Event(), asyncio.Event()
        order = []

        class HeldExecutor(Executor):
            async def run(self, batch):
                entered.set()
                await leave.wait()
                return await super().run(batch)

            async def close(self):
                order.append("executor")
                await super().close()

        class ClosingAdapter(ServiceAdapter):
            async def close(self):
                order.append("adapter")

        async def source():
            try:
                yield "held", "prompt", 1
                await asyncio.Event().wait()
            finally:
                finalizing.set()
                await leave.wait()
                order.append("source")

        async def collect():
            return [item async for item in engine.stream(source())]

        with (
            patch.object(engine.runtime, "create_adapter", return_value=ClosingAdapter()),
            patch.object(engine.runtime, "create_executor", return_value=HeldExecutor()),
        ):
            task = asyncio.create_task(collect())
            try:
                await asyncio.wait_for(entered.wait(), 1)
                task.cancel()
                await asyncio.wait_for(finalizing.wait(), 1)
                for _ in range(2):
                    task.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(task.done())
                    self.assertFalse(order)
                leave.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                leave.set()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(order, ["source", "executor", "adapter"])
        self.assertFalse(engine.running)
