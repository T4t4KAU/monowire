"""Completion-driven advancement, bounded admission and cancellation."""

import asyncio
import unittest

from monowire.continuous import ContinuousConfig, ContinuousScheduler, PreparedStep


def batch_futures(operation, count):
    """A test double for an executor whose physical batch completes together."""
    loop = asyncio.get_running_loop()
    futures = [loop.create_future() for _ in range(count)]

    async def run():
        try:
            outputs = await operation
            for future, output in zip(futures, outputs, strict=True):
                future.set_result(output)
        except Exception as error:
            for future in futures:
                future.set_exception(error)

    task = asyncio.create_task(run())
    # Futures retain the task until completion even without another reference.
    for future in futures:
        future.add_done_callback(lambda _, task=task: None)
    return futures


class ServiceExecutor:
    can_submit = True

    def start(self, *, on_capacity=None):
        self.on_capacity = on_capacity
        if not hasattr(self, "failure_signal"):
            self.failure_signal = asyncio.get_running_loop().create_future()
        return self.failure_signal


class Adapter:
    def __init__(self):
        self.released = []

    async def admit(self, key, value):
        return dict(key=key, step=0, total=value, value=0)

    def prepare(self, state):
        return PreparedStep(
            state["key"], state["step"], state["key"] != "slow", "same", state["value"]
        )

    def commit(self, state, prepared, output):
        assert prepared.step == state["step"]
        state["step"] += 1
        state["value"] = output

    def prepare_batch(self, prepared):
        return prepared

    def finished(self, state):
        return state["step"] == state["total"]

    async def finish(self, state):
        return state["value"]

    async def close(self):
        pass

    def release(self, state):
        self.released.append(state["key"])


class ContinuousTests(unittest.IsolatedAsyncioTestCase):
    async def test_blocked_sender_allows_bounded_prefix_lookahead(self):
        batches = []
        first, second = asyncio.Event(), asyncio.Event()

        class ObservedAdapter(Adapter):
            def prepare_batch(self, prepared):
                batches.append(prepared)
                (first if len(batches) == 1 else second).set()
                return prepared

        class GatedExecutor(ServiceExecutor):
            def __init__(self):
                self.futures = []

            def submit(self, prepared):
                self.can_submit = False
                future = asyncio.get_running_loop().create_future()
                self.futures.append(future)
                return [future]

        executor = GatedExecutor()
        scheduler = ContinuousScheduler(ObservedAdapter(), executor, ContinuousConfig(2, 1))
        scheduler.start()
        outputs = [scheduler.submit(str(i), 1) for i in range(2)]
        await asyncio.wait_for(first.wait(), 1)
        for _ in range(10):
            await asyncio.sleep(0)
        self.assertEqual(len(batches), 2)
        executor.can_submit = True
        executor.on_capacity()
        await asyncio.wait_for(second.wait(), 1)
        for _ in range(10):
            await asyncio.sleep(0)
        self.assertFalse(any(future.done() for future in executor.futures))
        for future in executor.futures:
            future.set_result(1)
        self.assertEqual(await asyncio.wait_for(asyncio.gather(*outputs), 1), [1, 1])
        await scheduler.close()

    async def test_compatible_groups_rotate_while_first_group_stays_busy(self):
        dispatched = []

        class GroupedAdapter(Adapter):
            def prepare(self, state):
                item = super().prepare(state)
                item.compatibility = state["key"][0]
                return item

        class ImmediateExecutor(ServiceExecutor):
            def submit(self, batch):
                dispatched.extend(p.request_id for p in batch)
                futures = [asyncio.get_running_loop().create_future() for _ in batch]
                for future, item in zip(futures, batch, strict=True):
                    future.set_result(item.payload + 1)
                return futures

        scheduler = ContinuousScheduler(
            GroupedAdapter(), ImmediateExecutor(), ContinuousConfig(4, 1)
        )
        results = [scheduler.submit(f"a{i}", 10) for i in range(3)]
        results.append(scheduler.submit("b", 1))
        await asyncio.wait_for(scheduler.close(), 2)
        self.assertEqual([f.result() for f in results], [10, 10, 10, 1])
        self.assertEqual(len(dispatched), 31)
        # Admissions are also bounded per turn; once admitted, the other
        # compatibility group must run without waiting for all a steps.
        self.assertLessEqual(dispatched.index("b"), 4)

    async def test_cancelled_pending_future_is_removed_while_active_slot_is_full(self):
        entered, gate = asyncio.Event(), asyncio.Event()

        class HeldExecutor(ServiceExecutor):
            def submit(self, batch):
                async def run():
                    entered.set()
                    await gate.wait()
                    return [p.payload + 1 for p in batch]

                return batch_futures(run(), len(batch))

        scheduler = ContinuousScheduler(Adapter(), HeldExecutor(), ContinuousConfig(1, 1))
        held = scheduler.submit("held", 1)
        await entered.wait()
        try:
            for _ in range(100):
                pending = scheduler.submit("cancelled", 1)
                pending.cancel()
                await asyncio.sleep(0)
                self.assertFalse(scheduler.pending)
                self.assertNotIn("cancelled", scheduler.ids)
                self.assertFalse(held.done())
        finally:
            gate.set()
            await asyncio.wait_for(scheduler.close(), 2)

    async def test_prefix_batches_obey_capacity_and_classify_after_joint_execution(self):
        class BatchedAdapter(Adapter):
            def __init__(self):
                super().__init__()
                self.prefixes = []

            def prepare(self, state):
                prepared = super().prepare(state)
                prepared.hit = False
                return prepared

            def prepare_batch(self, prepared):
                self.prefixes.append([(p.request_id, p.step) for p in prepared])
                for item in prepared:
                    item.hit = int(item.request_id) % 2 == 0
                return prepared

        class Executor(ServiceExecutor):
            def submit(self, batch):
                return batch_futures(self.run(batch), len(batch))

            async def run(self, batch):
                self.assert_batch(batch)
                return [p.payload + 1 for p in batch]

            def assert_batch(self, batch):
                assert len(batch) <= 8
                assert len({p.hit for p in batch}) == 1

        adapter = BatchedAdapter()
        scheduler = ContinuousScheduler(adapter, Executor(), ContinuousConfig(16, 8))
        futures = [scheduler.submit(str(i), 3) for i in range(16)]
        await asyncio.wait_for(scheduler.close(), 5)
        self.assertEqual([f.result() for f in futures], [3] * 16)
        self.assertEqual([len(b) for b in adapter.prefixes[:2]], [8, 8])
        self.assertTrue(all(len(b) <= 8 for b in adapter.prefixes))
        self.assertCountEqual(
            [identity for batch in adapter.prefixes for identity in batch],
            [(str(i), step) for i in range(16) for step in range(3)],
        )

    async def test_prefix_batch_failure_releases_members_and_keeps_other_batches_running(self):
        class FailingAdapter(Adapter):
            def prepare_batch(self, prepared):
                if prepared[0].request_id == "bad":
                    raise ValueError("prefix failed")
                return prepared

        class Executor(ServiceExecutor):
            def submit(self, batch):
                return batch_futures(self.run(batch), len(batch))

            async def run(self, batch):
                return [p.payload + 1 for p in batch]

        adapter = FailingAdapter()
        scheduler = ContinuousScheduler(adapter, Executor(), ContinuousConfig(3, 2))
        bad = scheduler.submit("bad", 1)
        partner = scheduler.submit("partner", 1)
        good = scheduler.submit("good", 2)
        await asyncio.wait_for(scheduler.close(), 5)
        self.assertIsInstance(bad.exception(), ValueError)
        self.assertIsInstance(partner.exception(), ValueError)
        self.assertEqual(good.result(), 2)
        self.assertCountEqual(adapter.released, ["bad", "partner", "good"])

    async def test_fast_request_advances_and_replacement_enters_before_slow_finishes(self):
        gate = asyncio.Event()
        events = []
        adapter = Adapter()

        class Executor(ServiceExecutor):
            def submit(self, batch):
                return batch_futures(self.run(batch), len(batch))

            async def run(self, batch):
                if any(p.request_id == "slow" for p in batch):
                    await gate.wait()
                await asyncio.sleep(0)
                return [p.payload + 1 for p in batch]

        def event(value):
            events.append(value)
            if value["event"] == "admitted" and value["request_id"] == "replacement":
                gate.set()

        scheduler = ContinuousScheduler(adapter, Executor(), ContinuousConfig(2, 2), event)
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

        class Executor(ServiceExecutor):
            def submit(self, batch):
                return batch_futures(self.run(batch), len(batch))

            async def run(self, batch):
                entered.set()
                await gate.wait()
                if batch[0].request_id == "bad":
                    raise ValueError("injected batch failure")
                return [p.payload + 1 for p in batch]

        scheduler = ContinuousScheduler(adapter, Executor(), ContinuousConfig(1, 1))
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
        class Executor(ServiceExecutor):
            def submit(self, batch):
                return batch_futures(self.run(batch), len(batch))

            async def run(self, batch):
                return [p.payload + 1 for p in batch]

        scheduler = ContinuousScheduler(Adapter(), Executor(), ContinuousConfig(2, 2))
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
        class Executor(ServiceExecutor):
            def submit(self, batch):
                return batch_futures(self.run(batch), len(batch))

            async def run(self, batch):
                return [p.payload + 1 for p in batch]

        def observer(event):
            raise ValueError("broken observer")

        adapter = Adapter()
        scheduler = ContinuousScheduler(adapter, Executor(), ContinuousConfig(1, 1), observer)
        with self.assertLogs("monowire", level="ERROR"):
            future = scheduler.submit("one", 2)
            await scheduler.close()
        self.assertEqual(future.result(), 2)
        self.assertEqual(adapter.released, ["one"])

    async def test_fatal_shutdown_drains_before_releasing_request(self):
        gate, entered = asyncio.Event(), asyncio.Event()

        class Executor(ServiceExecutor):
            def submit(self, batch):
                return batch_futures(self.run(batch), len(batch))

            async def run(self, batch):
                entered.set()
                await gate.wait()
                return [p.payload + 1 for p in batch]

        adapter = Adapter()
        scheduler = ContinuousScheduler(adapter, Executor(), ContinuousConfig(1, 1))
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


class IndependentCompletionTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_batch_can_complete_and_replace_before_its_slowest_member(self):
        slow_gate = asyncio.Event()
        batches = []

        class Executor(ServiceExecutor):
            def submit(self, batch):
                batches.append([p.request_id for p in batch])

                async def run(item):
                    if item.request_id == "held":
                        await slow_gate.wait()
                    return item.payload + 1

                return [asyncio.create_task(run(item)) for item in batch]

        def event(value):
            if value["event"] == "admitted" and value["request_id"] == "new":
                slow_gate.set()

        scheduler = ContinuousScheduler(Adapter(), Executor(), ContinuousConfig(2, 2), event)
        held = scheduler.submit("held", 1)
        fast = scheduler.submit("fast", 3)
        new = scheduler.submit("new", 1)
        await asyncio.wait_for(scheduler.close(), 2)
        self.assertEqual(batches[0], ["held", "fast"])
        self.assertEqual([held.result(), fast.result(), new.result()], [1, 3, 1])

    async def test_slow_admission_and_decode_do_not_block_other_steps(self):
        decode_entered, slow_admit = asyncio.Event(), asyncio.Event()
        fast_finished = asyncio.Event()

        class AsyncAdapter(Adapter):
            async def admit(self, key, value):
                if key == "encoding":
                    await slow_admit.wait()
                return await super().admit(key, value)

            async def finish(self, state):
                if state["key"] == "decoding":
                    decode_entered.set()
                    await fast_finished.wait()
                if state["key"] == "fast":
                    await decode_entered.wait()
                    slow_admit.set()
                    fast_finished.set()
                return await super().finish(state)

        class Executor(ServiceExecutor):
            def submit(self, batch):
                return batch_futures(
                    asyncio.sleep(0, result=[p.payload + 1 for p in batch]), len(batch)
                )

        scheduler = ContinuousScheduler(AsyncAdapter(), Executor(), ContinuousConfig(3, 3))
        results = [
            scheduler.submit(key, steps)
            for key, steps in [("decoding", 1), ("encoding", 1), ("fast", 4)]
        ]
        await asyncio.wait_for(scheduler.close(), 2)
        self.assertEqual([f.result() for f in results], [1, 1, 4])

    async def test_cancel_last_ready_member_still_dispatches_the_partial_batch(self):
        class Executor(ServiceExecutor):
            def submit(self, batch):
                return batch_futures(
                    asyncio.sleep(0, result=[p.payload + 1 for p in batch]), len(batch)
                )

        def event(value):
            if value["event"] == "ready" and value["request_id"] == "cancel":
                scheduler.cancel("cancel")

        scheduler = ContinuousScheduler(Adapter(), Executor(), ContinuousConfig(2, 2), event)
        keep, cancelled = scheduler.submit("keep", 1), scheduler.submit("cancel", 1)
        await asyncio.wait_for(scheduler.close(), 2)
        self.assertEqual(keep.result(), 1)
        self.assertTrue(cancelled.cancelled())

    async def test_failed_scheduler_drains_admission_and_releases_its_late_state(self):
        started, finish = asyncio.Event(), asyncio.Event()

        class SlowAdapter(Adapter):
            async def admit(self, key, value):
                started.set()
                await finish.wait()
                return await super().admit(key, value)

        adapter = SlowAdapter()
        scheduler = ContinuousScheduler(adapter, ServiceExecutor(), ContinuousConfig(1, 1))
        result = scheduler.submit("admitting", 1)
        await started.wait()
        scheduler.task.cancel()
        await asyncio.sleep(0)
        self.assertEqual(adapter.released, [])
        finish.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(scheduler.close(), 2)
        self.assertIsInstance(result.exception(), RuntimeError)
        self.assertEqual(adapter.released, ["admitting"])

    async def test_blocked_misses_have_bounded_lookahead_and_resume(self):
        prefixes = []

        class MissAdapter(Adapter):
            def prepare(self, state):
                prepared = super().prepare(state)
                prepared.hit = False
                return prepared

            def prepare_batch(self, prepared):
                prefixes.append([p.request_id for p in prepared])
                return prepared

        class Executor(ServiceExecutor):
            def __init__(self):
                self.calls = []

            def submit(self, prepared):
                self.can_submit = False
                futures = [asyncio.get_running_loop().create_future() for _ in prepared]
                self.calls.append(futures)
                return futures

        executor = Executor()
        scheduler = ContinuousScheduler(MissAdapter(), executor, ContinuousConfig(10, 2))
        outputs = [scheduler.submit(str(i), 1) for i in range(10)]
        for _ in range(30):
            await asyncio.sleep(0)
        self.assertEqual(len(prefixes), 3)  # One sent, two retained locally.
        self.assertEqual(sum(map(len, scheduler.ready.values())), 4)
        self.assertEqual(len(executor.calls), 1)
        while not all(f.done() for f in outputs):
            for futures in executor.calls:
                for future in futures:
                    if not future.done():
                        future.set_result(1)
            executor.can_submit = True
            executor.on_capacity()
            for _ in range(10):
                await asyncio.sleep(0)
        self.assertEqual(await asyncio.gather(*outputs), [1] * 10)
        await scheduler.close()

    async def test_completion_work_yields_after_one_batch(self):
        from types import SimpleNamespace

        adapter = Adapter()
        scheduler = ContinuousScheduler(adapter, ServiceExecutor(), ContinuousConfig(8, 2))
        states = []
        for i in range(6):
            state = dict(key=str(i), step=0, total=3, value=0)
            request = SimpleNamespace(
                key=str(i),
                state=state,
                inflight=True,
                future=asyncio.get_running_loop().create_future(),
            )
            completion = asyncio.get_running_loop().create_future()
            completion.set_result(1)
            scheduler.tasks[completion] = (request, adapter.prepare(state))
            states.append(state)
        scheduler._complete()
        self.assertEqual([s["step"] for s in states], [1, 1, 0, 0, 0, 0])
        self.assertEqual(sum(map(len, scheduler.ingress.values())), 2)
        scheduler._complete()
        scheduler._complete()
        self.assertFalse(scheduler.tasks)
