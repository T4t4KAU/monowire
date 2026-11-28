"""Shared admission races, delayed CUDA lifetime, and real Gloo wire protocol."""

import asyncio
import tempfile
import threading
import unittest
import weakref
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from monowire.channels import _encode, _spec_allocation_bytes, close_channels, create_channels
from monowire.continuous import ContinuousConfig, ContinuousScheduler, PreparedStep
from monowire.device_memory import DeviceMemoryLimits, DeviceMemoryPool
from monowire.tensor_payload import SharedTensor, TensorDictionary
from monowire.topology import PipelineTopology
from monowire.transport import PipelineExecutor, serve_stage


class DeviceMemoryTests(unittest.TestCase):
    def pool(self, available=lambda: 102):
        return DeviceMemoryPool(
            "cpu",
            DeviceMemoryLimits(20, headroom_bytes=10, progress_bytes=10, dictionary_bytes=1),
            available=available,
        )

    def test_concurrent_admissions_cannot_spend_same_free_space(self):
        pool = self.pool()
        barrier = threading.Barrier(16)
        leases = []

        def reserve():
            barrier.wait()
            leases.append(pool.try_reserve(10))

        threads = [threading.Thread(target=reserve) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sum(lease is not None for lease in leases), 6)
        self.assertEqual(pool.snapshot()["promised_bytes"], 60)
        for lease in leases:
            if lease is not None:
                lease.close()
        self.assertEqual(pool.snapshot()["promised_bytes"], 0)

    def test_hot_admission_does_not_query_driver_or_scan_storage(self):
        calls = []
        pool = self.pool(lambda: calls.append(1) or 102)
        for _ in range(100):
            lease = pool.try_reserve(50)
            lease.update(torch.ones(10))
            lease.close()
        self.assertEqual(len(calls), 1)

    def test_workspace_and_feedback_progress_when_normal_admission_is_full(self):
        pool = self.pool()
        normal = pool.try_reserve(60)
        self.assertIsNone(pool.try_reserve(1))
        feedback = pool.try_reserve(5, purpose="feedback")
        self.assertIsNotNone(feedback)
        for _ in range(100):
            with pool.workspace():
                self.assertEqual(pool.used["normal"], 60)
                self.assertEqual(pool.used["feedback"], 5)
        self.assertEqual(pool.admissions, 2)
        feedback.close()
        normal.close()

    def test_wire_cumulative_limit_cannot_borrow_feedback_capacity(self):
        pool = self.pool()
        first = pool.reserve(3, purpose="wire")
        second = pool.reserve(2, purpose="wire")
        self.assertIsNone(pool.try_reserve(1, purpose="wire"))
        feedback = pool.try_reserve(5, purpose="feedback")
        self.assertIsNotNone(feedback)
        self.assertEqual(pool.snapshot()["used_by_purpose"], dict(normal=0, feedback=5, wire=5))
        first.close()
        replacement = pool.try_reserve(3, purpose="wire")
        self.assertIsNotNone(replacement)
        for lease in (second, replacement, feedback):
            lease.close()
        self.assertEqual(pool.snapshot()["promised_bytes"], 0)

    def test_drain_allows_other_threads_to_make_communication_progress(self):
        pool = self.pool()
        waiting, complete = threading.Event(), threading.Event()

        class Event:
            def query(self):
                return complete.is_set()

            def synchronize(self):
                waiting.set()
                if not complete.wait(1):
                    raise AssertionError("Ledger lock blocks communication progress")

        def finish():
            if waiting.wait(1):
                pool.reserve(1, purpose="wire").close()
                complete.set()

        pool.retain(Event(), None)
        thread = threading.Thread(target=finish)
        thread.start()
        try:
            pool.drain()
        finally:
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertFalse(pool.retired)

    def test_completed_event_releases_promises_and_resources_not_queue_pop(self):
        pool = self.pool()
        lease = pool.try_reserve(60)

        class Event:
            done = False

            def query(self):
                return self.done

            def synchronize(self):
                self.done = True

        event = Event()
        tensor = torch.ones(3)
        reference = weakref.ref(tensor)
        pool.retain(event, tensor, lease)
        del tensor
        self.assertIsNone(pool.try_reserve(1))
        self.assertIsNotNone(reference())
        event.done = True
        self.assertIsNotNone(pool.try_reserve(1))
        self.assertIsNone(reference())

    def test_wire_count_excludes_existing_shared_storage_and_cpu_tensors(self):
        shared = SharedTensor.snapshot(torch.ones(3))
        dictionary = TensorDictionary()
        spec = _encode([shared, shared], [], dictionary)
        self.assertEqual(_spec_allocation_bytes(spec), 0)  # CPU payload
        spec = ("list", [("tensor", 0, (2, 3), torch.float16, True), ("shared_ref", shared.key)])
        self.assertEqual(_spec_allocation_bytes(spec), 12)


class ActiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_active_limit_remains_independent_of_shared_capacity(self):
        pool = DeviceMemoryPool(
            "cpu",
            DeviceMemoryLimits(10, headroom_bytes=10, progress_bytes=10, dictionary_bytes=1),
            available=lambda: 1000,
        )
        admitted = []

        class Adapter:
            memory_pool = pool

            def estimate_request_bytes(self, value):
                return 10

            def resident_payload(self, state):
                return state

            def validate_request_reservation(self, state, reserved):
                pass

            async def admit(self, key, value):
                admitted.append(key)
                return key

            def prepare(self, state):
                return PreparedStep(state, 0, False, "same", state)

            def prepare_batch(self, prepared):
                return prepared

            def commit(self, *args):
                pass

            def finished(self, state):
                return True

            async def finish(self, state):
                return state

            def release(self, state):
                pass

        class Executor:
            can_submit = True
            outputs = []

            def start(self, **kwargs):
                return asyncio.get_running_loop().create_future()

            def can_submit_prepared(self, prepared):
                return True

            def submit(self, prepared):
                future = asyncio.get_running_loop().create_future()
                self.outputs.append(future)
                return [future]

        executor = Executor()
        scheduler = ContinuousScheduler(
            Adapter(), executor, ContinuousConfig(2, 1, 1, memory_limits=pool.limits)
        )
        futures = [scheduler.submit(str(i), None) for i in range(6)]
        for _ in range(30):
            await asyncio.sleep(0)
        self.assertEqual(len(admitted), 2)
        self.assertEqual(len(scheduler.active), 2)
        futures[1].cancel()

        async def finish():
            while not all(f.done() for f in futures):
                self.assertLessEqual(len(scheduler.active), 2)
                for future in executor.outputs:
                    if not future.done():
                        future.set_result(None)
                await asyncio.sleep(0)

        await asyncio.wait_for(finish(), 3)
        await asyncio.gather(*futures, return_exceptions=True)
        await scheduler.close()
        self.assertEqual(pool.snapshot()["live_leases"], 0)


def dynamic_pipeline_worker(rank, store, world=3, cuda=False, strategy="ring"):
    device = torch.device(f"cuda:{rank}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo",
        init_method=Path(store).as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=60),
    )
    topology = PipelineTopology((1,) * world, buffer_size=1)
    limits = DeviceMemoryLimits(
        1024**2,
        headroom_bytes=1024**2,
        progress_bytes=1024**2,
        dictionary_bytes=1024,
        buffer_strategy=strategy,
        buffer_bytes=4096,
        buffer_alignment=16,
    )
    pool = DeviceMemoryPool(device, limits, available=None if cuda else lambda: 8 * 1024**2)
    channels = create_channels(topology, device, memory_pool=pool)

    class Stage:
        group = SimpleNamespace(rank_in_group=rank, device=device)

        def validate(self, prepared):
            pass

        def export_requests(self, prepared):
            return [dict(hidden=p.payload + 1, compatibility="same") for p in prepared]

        def combine_requests(self, items):
            return dict(hidden=torch.cat([item["hidden"] for item in items]))

        def compute(self, packet):
            packet["hidden"] += rank + 1

    async def run():
        executor = PipelineExecutor(Stage(), topology, channels)
        executor.start()
        assert executor.sender.queue.maxsize == 0
        outputs = []
        for i in range(32):
            prepared = PreparedStep(
                str(i), 0, False, "same", torch.full((i % 8 + 1, 2), float(i), device=device)
            )
            outputs.append((i, executor.submit([prepared])[0]))
        for i, future in outputs:
            result = await asyncio.wait_for(future, 20)
            torch.testing.assert_close(
                result, torch.full_like(result, i + world * (world + 1) // 2)
            )
        await executor.close()

    try:
        with torch.inference_mode():
            for _ in range(2):
                if rank == 0:
                    asyncio.run(run())
                else:
                    serve_stage(Stage(), topology, channels, batch_size=1)
                dist.barrier()
        pool.drain()
        assert all(b["live_bytes"] == 0 for b in pool.snapshot()["buffers"])
    finally:
        close_channels(channels)
        assert pool.snapshot()["live_leases"] == 0
        dist.destroy_process_group()


class DynamicProtocolTests(unittest.TestCase):
    def test_variable_lengths_multiple_rounds_and_shutdown(self):
        for strategy in ("ring", "pool"):
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as directory:
                mp.spawn(
                    dynamic_pipeline_worker,
                    args=(str(Path(directory) / "store"), 3, False, strategy),
                    nprocs=3,
                    join=True,
                )
