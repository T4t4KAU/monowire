"""Byte capacity, delayed release, variable lengths and real distributed drain."""

import asyncio
import queue
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from monowire.channels import (
    InFlightQueue,
    _decode,
    _encode,
    _spec_bytes,
    close_channels,
    create_channels,
)
from monowire.continuous import ContinuousConfig, ContinuousScheduler, PreparedStep
from monowire.memory import ByteQueue, MemoryLimits, storage_bytes
from monowire.tensor_payload import SharedTensor, TensorDictionary, TensorParts
from monowire.topology import PipelineTopology
from monowire.transport import PipelineExecutor, serve_stage


class MemoryTests(unittest.TestCase):
    def test_views_and_shared_payloads_count_physical_storage_once(self):
        tensor = torch.zeros(20)
        shared = SharedTensor.snapshot(tensor)
        self.assertEqual(storage_bytes([tensor[:2], tensor[2:], tensor]), 80)
        self.assertEqual(storage_bytes(TensorParts((shared, shared))), 80)

    def test_popped_bytes_stay_reserved_until_consumed_out_of_order(self):
        q = ByteQueue(12, weight=len)
        q.put(b"abcd")
        q.put(b"12345678")
        self.assertEqual(q.get(), b"abcd")
        self.assertEqual(q.get(), b"12345678")
        with self.assertRaises(queue.Full):
            q.put_nowait(b"x")
        q.task_done(8)
        self.assertEqual(q.used_bytes, 4)
        q.put_nowait(b"abcdefgh")
        q.task_done(4)
        self.assertEqual(q.get(), b"abcdefgh")
        q.task_done()
        self.assertEqual(q.used_bytes, 0)
        self.assertEqual(q.peak_bytes, 12)

    def test_capacity_depends_on_bytes_and_control_bypasses_data_pressure(self):
        q = ByteQueue(16)
        for _ in range(4):
            q.put_nowait(torch.zeros(1))
        with self.assertRaises(queue.Full):
            q.put_nowait(torch.zeros(1))
        q.put_nowait(None)
        with self.assertRaisesRegex(ValueError, "budget"):
            q.put_nowait(torch.zeros(5))
        for _ in range(5):
            q.get()
            q.task_done()
        q.put_nowait(torch.zeros(4))
        self.assertFalse(q.can_put(torch.zeros(1)))

    def test_wire_estimate_handles_new_shared_then_reference_without_mutation(self):
        shared = SharedTensor.snapshot(torch.zeros(4))
        sender, receiver = TensorDictionary(), TensorDictionary()
        spec = _encode([shared, shared, shared], [], sender)
        self.assertEqual(_spec_bytes(spec, receiver), 16)
        self.assertFalse(receiver.entries)
        decoded = _decode(spec, [], "cpu", receiver)
        self.assertEqual(storage_bytes(decoded), 16)

    def test_cached_shared_references_count_once_without_mutation(self):
        shared = SharedTensor.snapshot(torch.zeros(4))
        sender, receiver = TensorDictionary(), TensorDictionary()
        sender.add(shared)
        receiver.add(shared)
        before = list(receiver.entries)
        spec = _encode([shared, shared, shared], [], sender)
        self.assertEqual(_spec_bytes(spec, receiver), 16)
        self.assertEqual(list(receiver.entries), before)
        self.assertEqual(storage_bytes(_decode(spec, [], "cpu", receiver)), 16)
        with self.assertRaisesRegex(ValueError, "Missing shared"):
            _spec_bytes(("list", [("shared_ref", shared.key), ("shared_ref", "missing")]), receiver)

    def test_repeated_uncached_declarations_still_charge_new_allocations(self):
        shared = SharedTensor.snapshot(torch.zeros(4))
        spec = _encode([shared, shared, shared], [], TensorDictionary(max_bytes=0))
        self.assertEqual(_spec_bytes(spec, TensorDictionary(max_bytes=0)), 48)
        self.assertEqual(storage_bytes(_decode(spec, [], "cpu", TensorDictionary(max_bytes=0))), 48)

    def test_inflight_releases_only_completed_gpu_storage(self):
        events = []

        class Event:
            done = False

            def query(self):
                return self.done

            def synchronize(self):
                events.append("wait")
                self.done = True

        q = InFlightQueue(1, byte_capacity=16)
        for _ in range(4):
            q.reserve(4)
            q.track(Event(), torch.zeros(1))
        self.assertEqual(len(q.pending), 4)  # Legacy count limit1 is ignored.
        self.assertEqual(q.used_bytes, 16)
        q.reserve(8)
        self.assertEqual(len(events), 2)
        self.assertEqual(q.used_bytes, 8)
        q.drain()
        self.assertEqual(q.used_bytes, 0)

    def test_unlimited_active_requires_budget_and_byte_mode_rejects_bs4(self):
        with self.assertRaises(ValueError):
            ContinuousConfig(max_active_requests=None)
        limits = MemoryLimits(100, 900, 100)
        with self.assertRaises(ValueError):
            ContinuousConfig(memory_limits=limits)
        config = ContinuousConfig(None, 1, 1, memory_limits=limits)
        self.assertIsNone(config.max_active_requests)


class ByteAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_mixed_size_admission_reclaims_on_finish_error_and_cancellation(self):
        admitted = []

        class Adapter:
            def estimate_request_bytes(self, value):
                return value

            def estimate_step_bytes(self, state):
                return 4

            def validate_request_reservation(self, state, reserved):
                assert state["size"] <= reserved

            async def admit(self, key, value):
                if key == "fail":
                    raise ValueError("codec failed")
                admitted.append(key)
                return dict(key=key, size=value)

            def prepare(self, state):
                return PreparedStep(state["key"], 0, False, "same", torch.ones(1))

            def prepare_batch(self, prepared):
                return prepared

            def commit(self, state, prepared, output):
                pass

            def finished(self, state):
                return True

            async def finish(self, state):
                return state["key"]

            def release(self, state):
                pass

        class Executor:
            can_submit = True

            def __init__(self):
                self.outputs = []

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
            Adapter(),
            executor,
            ContinuousConfig(None, 1, 1, memory_limits=MemoryLimits(10, 900, 100)),
        )
        scheduler.start()
        futures = [
            scheduler.submit(k, v)
            for k, v in [("a", 4), ("b", 6), ("cancel", 7), ("fail", 2), ("c", 7), ("large", 11)]
        ]
        for _ in range(20):
            await asyncio.sleep(0)
        self.assertEqual(set(admitted), {"a", "b"})
        self.assertEqual(scheduler.resident_bytes, 10)
        futures[2].cancel()

        async def complete():
            while not all(f.done() for f in futures):
                for f in executor.outputs:
                    if not f.done():
                        f.set_result(torch.ones(1))
                await asyncio.sleep(0)

        await asyncio.wait_for(complete(), 2)
        results = await asyncio.gather(*futures, return_exceptions=True)
        await scheduler.close()
        self.assertEqual(results[0:2], ["a", "b"])
        self.assertIsInstance(results[2], asyncio.CancelledError)
        self.assertIsInstance(results[3], ValueError)
        self.assertEqual(results[4], "c")
        self.assertIsInstance(results[5], ValueError)
        self.assertEqual(scheduler.resident_bytes, 0)
        self.assertLessEqual(scheduler.peak_resident_bytes, 10)


def byte_pipeline_worker(rank, store, world=3, cuda=False):
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
    topology = PipelineTopology((1,) * world, buffer_size=1, buffer_bytes=256)
    channels = create_channels(topology, device)

    class Stage:
        group = SimpleNamespace(rank_in_group=rank, device=device)

        def validate(self, prepared):
            pass

        def export_requests(self, prepared):
            return [dict(hidden=p.payload + 1, compatibility="same") for p in prepared]

        def combine_requests(self, items):
            return dict(hidden=torch.cat([i["hidden"] for i in items]))

        def compute(self, packet):
            packet["hidden"] += rank + 1

    async def run():
        executor = PipelineExecutor(Stage(), topology, channels)
        changed = asyncio.Event()
        executor.start(on_capacity=changed.set)
        outputs = []
        for i in range(24):
            item = PreparedStep(
                str(i), 0, False, "same", torch.full((i % 8 + 1, 2), float(i), device=device)
            )
            while not executor.can_submit_prepared([item]):
                changed.clear()
                if not executor.can_submit_prepared([item]):
                    await asyncio.wait_for(changed.wait(), 10)
            outputs.append((i, executor.submit([item])[0]))
        for i, future in outputs:
            result = await asyncio.wait_for(future, 10)
            torch.testing.assert_close(
                result, torch.full_like(result, i + world * (world + 1) // 2)
            )
        await executor.close()
        assert executor.sender.queue.peak_bytes <= 256
        assert executor.sender.queue.used_bytes == 0

    try:
        with torch.inference_mode():
            for _ in range(2):
                if rank == 0:
                    asyncio.run(run())
                else:
                    serve_stage(Stage(), topology, channels, batch_size=1)
                dist.barrier()
    finally:
        close_channels(channels)
        dist.destroy_process_group()


def byte_startup_rejection_worker(rank, store):
    from monowire.models.h3.runtime import H3Runtime

    dist.init_process_group(
        "gloo",
        init_method=Path(store).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    group = SimpleNamespace(rank_in_group=rank, cpu_group=dist.group.WORLD)
    transformer = SimpleNamespace(
        pp_group=group,
        adaln_cache=None,
        topology=PipelineTopology((1, 1)),
        request_cache=SimpleNamespace(config=SimpleNamespace(policy="none")),
        to=lambda device: None,
    )
    pipeline = SimpleNamespace(
        transformer=transformer, device="cpu", load_text_encoder=(rank == 0), load_vae_encoder=False
    )
    worker = SimpleNamespace(model_runner=SimpleNamespace(pipeline=pipeline))
    config = ContinuousConfig(None, 1, 1, memory_limits=MemoryLimits(1, 9, 1))
    try:
        try:
            H3Runtime(worker, config, channels={})
        except ValueError as error:
            assert "phased/pre-encoded" in str(error)
        else:
            raise AssertionError("Every rank must reject the ingress encoder configuration")
        dist.barrier()
    finally:
        dist.destroy_process_group()


class DistributedByteTests(unittest.TestCase):
    def test_rank_local_startup_error_rejects_all_ranks_without_hanging(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                byte_startup_rejection_worker,
                args=(str(Path(directory) / "store"),),
                nprocs=2,
                join=True,
            )

    def test_variable_lengths_complete_and_restart_without_credit_leaks(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                byte_pipeline_worker, args=(str(Path(directory) / "store"),), nprocs=3, join=True
            )


if __name__ == "__main__":
    unittest.main()
