"""Codec ownership, cancellation and scheduler responsiveness without GPUs."""

import asyncio
import threading
import unittest
from unittest.mock import Mock, patch

import torch

from monowire.models.h3.io import CodecOffloader, CodecWorker


class CodecOffloaderTests(unittest.TestCase):
    def test_failure_restores_ownership_and_preserves_parameters_and_buffers(self):
        module = torch.nn.Linear(2, 2)
        module.register_buffer("counter", torch.zeros(1))
        other = torch.nn.Linear(2, 2)
        parameter = module.weight
        offloader = CodecOffloader([module, other], "cpu", pin_memory=False)
        with (
            patch("torch.cuda.synchronize", side_effect=AssertionError("device barrier")),
            patch("torch.accelerator.empty_cache", side_effect=AssertionError("cache flush")),
        ):
            with self.assertRaisesRegex(ValueError, "codec failed"):
                with offloader.activate(module):
                    with offloader.activate(module):
                        self.assertIs(offloader.active, module)
                        module.counter.add_(1)
                    self.assertIs(offloader.active, module)
                    with self.assertRaisesRegex(RuntimeError, "serially"):
                        with offloader.activate(other):
                            pass
                    raise ValueError("codec failed")
            self.assertIsNone(offloader.active)
            with offloader.activate(other):
                other(torch.ones(1, 2))
        self.assertIs(module.weight, parameter)
        self.assertEqual(module.counter.item(), 1)
        with self.assertRaisesRegex(ValueError, "not owned"):
            with offloader.activate(torch.nn.Linear(2, 2)):
                pass

    def test_partial_load_failure_waits_for_only_the_owning_stream_before_cleanup(self):
        module = torch.nn.Linear(2, 2)
        offloader = CodecOffloader([module], "cpu", pin_memory=False)
        offloader.device = torch.device("cuda", 0)
        order = []

        weight_storage = module.weight.data_ptr()
        offload = offloader._offload

        def load(module):
            order.append("load")
            module.weight.data = module.weight.detach().clone()
            raise RuntimeError("copy failed")

        def restore(module):
            order.append("restore")
            offload(module)

        stream = Mock()
        stream.synchronize.side_effect = lambda: order.append("stream_wait")
        with (
            patch.object(offloader, "_load", side_effect=load),
            patch.object(offloader, "_offload", side_effect=restore),
            patch("torch.cuda.current_stream", return_value=stream),
            patch("torch.cuda.synchronize", side_effect=AssertionError("device barrier")),
            patch("torch.accelerator.synchronize", side_effect=AssertionError("device barrier")),
        ):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                with offloader.activate(module):
                    self.fail("Forward must not run after a failed load")
        self.assertEqual(order, ["load", "stream_wait", "restore"])
        self.assertEqual(module.weight.data_ptr(), weight_storage)
        self.assertIsNone(offloader.active)

    def test_unload_restores_weight_storage_and_preserves_replaced_buffers(self):
        module = torch.nn.Linear(2, 2)
        module.register_buffer("counter", torch.zeros(1))
        offloader = CodecOffloader([module], "cpu", pin_memory=False)
        weight_storage = module.weight.data_ptr()
        tensor_cpu = torch.Tensor.cpu
        resident_weights = set()

        def checked_cpu(tensor, *args, **kwargs):
            self.assertNotIn(tensor.data_ptr(), resident_weights, "Weights copied back to CPU")
            return tensor_cpu(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "cpu", checked_cpu):
            for count in range(1, 4):
                with offloader.activate(module):
                    # Simulate separate device storage on a CPU test worker.
                    module.weight.data = module.weight.detach().clone()
                    resident_weights.add(module.weight.data_ptr())
                    module.counter = module.counter + 1
                self.assertEqual(module.weight.data_ptr(), weight_storage)
                self.assertEqual(module.counter.item(), count)
                resident_weights.clear()


class CodecTests(unittest.IsolatedAsyncioTestCase):
    async def test_blocking_codec_leaves_event_loop_free_and_serializes_components(self):
        worker = CodecWorker("cpu")
        entered, leave = threading.Event(), threading.Event()
        order, threads = [], []

        def first():
            threads.append(threading.get_ident())
            order.append("first")
            entered.set()
            if not leave.wait(2):
                raise TimeoutError("test codec not released")
            return 1

        def second():
            threads.append(threading.get_ident())
            order.append("second")
            return 2

        first_task = asyncio.create_task(worker.run(first))
        self.assertTrue(await asyncio.to_thread(entered.wait, 1))
        second_task = asyncio.create_task(worker.run(second))
        await asyncio.sleep(0)
        self.assertEqual(order, ["first"])
        self.assertNotEqual(threads[0], threading.get_ident())
        leave.set()
        self.assertEqual(await asyncio.gather(first_task, second_task), [1, 2])
        self.assertEqual(threads[0], threads[1])
        await worker.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await worker.run(second)

    async def test_cancel_does_not_release_the_caller_before_codec_finishes(self):
        worker = CodecWorker("cpu")
        entered, leave, completed = threading.Event(), threading.Event(), threading.Event()

        def codec():
            entered.set()
            leave.wait(2)
            completed.set()

        task = asyncio.create_task(worker.run(codec))
        self.assertTrue(await asyncio.to_thread(entered.wait, 1))
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.assertFalse(completed.is_set())
        leave.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(completed.is_set())
        await worker.close()

    async def test_repeated_cancellation_drains_successful_and_failing_codec_work(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                worker = CodecWorker("cpu")
                entered, leave, completed = threading.Event(), threading.Event(), threading.Event()
                released = []

                def codec():
                    entered.set()
                    if not leave.wait(3):
                        raise TimeoutError("Test did not release codec")
                    completed.set()
                    if fail:
                        raise ValueError("codec failed")

                async def caller():
                    try:
                        await worker.run(codec)
                    finally:
                        released.append(completed.is_set())

                task = asyncio.create_task(caller())
                try:
                    self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                    for _ in range(3):
                        task.cancel()
                        await asyncio.sleep(0)
                        self.assertFalse(task.done())
                        self.assertFalse(released)
                    leave.set()
                    with self.assertRaises(asyncio.CancelledError) as caught:
                        await task
                    if fail:
                        self.assertIsInstance(caught.exception.__cause__, ValueError)
                    self.assertEqual(released, [True])
                finally:
                    leave.set()
                    await asyncio.gather(task, return_exceptions=True)
                    await worker.close()


@unittest.skipUnless(torch.cuda.is_available(), "Requires an allocated CUDA device")
class CodecCUDATests(unittest.IsolatedAsyncioTestCase):
    async def test_offload_round_trip_uses_codec_stream_and_preserves_output(self):
        module = torch.nn.Linear(2, 2).eval()
        module.register_buffer("counter", torch.zeros(1))
        value = torch.ones(1, 2)
        expected = module(value).detach()
        offloader = CodecOffloader([module], "cuda:0", pin_memory=True)
        weight_storage = module.weight.data_ptr()
        worker = CodecWorker("cuda:0")

        def operation(fail=False):
            with offloader.activate(module):
                self.assertEqual(torch.cuda.current_stream(), worker.stream)
                self.assertEqual(module.weight.device.type, "cuda")
                module.counter.add_(1)
                output = module(value.cuda())
                if fail:
                    raise ValueError("codec failed")
                return output

        try:
            with (
                patch("torch.cuda.synchronize", side_effect=AssertionError("device barrier")),
                patch(
                    "torch.accelerator.synchronize", side_effect=AssertionError("device barrier")
                ),
                patch("torch.cuda.empty_cache", side_effect=AssertionError("cache flush")),
            ):
                for _ in range(2):
                    output = await worker.run(operation)
                    torch.testing.assert_close(output.cpu(), expected)
                    self.assertEqual(module.weight.device.type, "cpu")
                    self.assertEqual(module.weight.data_ptr(), weight_storage)
                    self.assertTrue(module.weight.is_pinned())
                    self.assertFalse(module.weight.is_inference())
                    self.assertFalse(module.counter.is_inference())
                with self.assertRaisesRegex(ValueError, "codec failed"):
                    await worker.run(operation, True)
            self.assertEqual(module.counter.device.type, "cpu")
            self.assertEqual(module.counter.item(), 3)
            self.assertIsNone(offloader.active)
        finally:
            await worker.close()

    async def test_handoff_orders_compute_before_codec_and_codec_before_consumer(self):
        from monowire.channels import record_stream

        worker = CodecWorker(torch.device("cuda", 0))
        value = torch.arange(1024, device="cuda", dtype=torch.float32)
        value = value * 2
        try:
            output = await worker.run(lambda: value + 3)
            record_stream(output)
            torch.testing.assert_close(output * 2, value * 2 + 6)
        finally:
            await worker.close()
