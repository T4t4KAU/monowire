"""Startup and cancellation of the encoder's rolling CPU inbox."""

import asyncio
import tempfile
import threading
import time
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from monowire.models.h3.rolling import EncoderInbox


def delayed_start_worker(rank, store):
    dist.init_process_group(
        "gloo",
        init_method=Path(store).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    group = dist.new_group(backend="gloo", timeout=timedelta(seconds=0.5))
    inbox = EncoderInbox([1], 8, 1, group) if rank == 0 else None
    try:
        # A healthy checkpoint load longer than the output transport timeout.
        if rank == 1:
            time.sleep(1)
        dist.barrier()
        if rank == 0:
            assert inbox.thread is None
            inbox.start()

            async def collect():
                return [item async for item in inbox.tensors()]

            items = asyncio.run(collect())
            assert len(items) == 1 and items[0][0] == 0
            torch.testing.assert_close(items[0][1], torch.ones((1, 8), dtype=torch.bfloat16))
            inbox.close()
            assert not inbox.thread.is_alive()
        else:
            dist.send(torch.ones((1, 8), dtype=torch.bfloat16), dst=0, group=group)
        dist.barrier()
    finally:
        if inbox is not None:
            inbox.close()
        dist.destroy_process_group(group)
        dist.destroy_process_group()


class EncoderInboxTests(unittest.TestCase):
    def test_checkpoint_load_does_not_consume_receive_timeout(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(delayed_start_worker, args=(str(Path(folder, "store")),), nprocs=2)

    def test_cancelled_wait_does_not_hold_executor_or_consume_late_output(self):
        inbox = EncoderInbox([1], 8, 0, None)
        tensor = torch.ones((1, 8))

        async def cancel():
            source = inbox.tensors()
            task = asyncio.create_task(anext(source))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await source.aclose()

        # Explicitly reject default-executor work: cancelling its Future cannot
        # stop queue.get, and asyncio.run would wait for that thread on exit.
        with patch("asyncio.to_thread", side_effect=AssertionError("blocking executor wait")):
            asyncio.run(cancel())
        inbox.publish(0, tensor)

        async def collect():
            return [item async for item in inbox.tensors()]

        items = asyncio.run(collect())
        self.assertEqual(len(items), 1)
        self.assertIs(items[0][1], tensor)
        inbox.close()

    def test_receiver_thread_wakes_consumer_in_order(self):
        inbox = EncoderInbox([1, 2], 8, 1, object())
        receive_started = threading.Event()
        allow_receive = threading.Event()

        def receive(tensor, **kwargs):
            receive_started.set()
            if not allow_receive.wait(2):
                raise RuntimeError("test consumer did not start")
            tensor.fill_(len(tensor))

        async def collect():
            source = inbox.tensors()
            pending = asyncio.create_task(anext(source))
            await asyncio.sleep(0)
            allow_receive.set()
            first = await pending
            return [first] + [item async for item in source]

        with patch("monowire.models.h3.rolling.dist.recv", side_effect=receive):
            inbox.start()
            self.assertTrue(receive_started.wait(2))
            items = asyncio.run(collect())
            inbox.close()
        self.assertEqual([index for index, _ in items], [0, 1])
        for index, tensor in items:
            self.assertTrue(torch.all(tensor == index + 1))
        self.assertFalse(inbox.thread.is_alive())

    def test_close_wakes_waiter_and_discards_late_output(self):
        inbox = EncoderInbox([1], 8, 0, None)

        async def collect():
            source = inbox.tensors()
            pending = asyncio.create_task(anext(source))
            await asyncio.sleep(0)
            inbox.close()
            with self.assertRaisesRegex(RuntimeError, "inbox is closed"):
                await pending
            await source.aclose()

        asyncio.run(collect())
        inbox.publish(0, torch.ones((1, 8)))
        self.assertFalse(inbox._items)
        inbox.close()
        with self.assertRaisesRegex(RuntimeError, "must be open"):
            inbox.start()

    def test_receiver_failure_reaches_consumer_and_thread_terminates(self):
        inbox = EncoderInbox([1], 8, 1, object())
        failure = ValueError("broken peer")

        async def collect():
            return [item async for item in inbox.tensors()]

        with patch("monowire.models.h3.rolling.dist.recv", side_effect=failure):
            inbox.start()
            with self.assertRaisesRegex(RuntimeError, "receiver failed") as raised:
                asyncio.run(collect())
            inbox.close()
        self.assertIs(raised.exception.__cause__, failure)
        self.assertFalse(inbox.thread.is_alive())

    def test_missing_output_times_out_without_executor_thread(self):
        inbox = EncoderInbox([1], 8, 0, None)

        async def collect():
            return [item async for item in inbox.tensors()]

        with patch("monowire.models.h3.rolling.SHUTDOWN_TIMEOUT", 0.01):
            with self.assertRaisesRegex(TimeoutError, "output did not arrive"):
                asyncio.run(collect())
        self.assertIsNone(inbox._waiter)
        inbox.close()
