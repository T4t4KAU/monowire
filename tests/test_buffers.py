"""Adversarial reuse, mixed-length fragmentation, and event-ordered reclamation."""

import threading
import time
import unittest
from unittest.mock import patch

import torch

from monowire.buffers import ReceiveBuffers
from monowire.channels import _decode, _encode, _receive_layout
from monowire.device_memory import DeviceMemoryLimits, DeviceMemoryPool
from monowire.tensor_payload import SharedTensor, TensorDictionary


class Event:
    def __init__(self):
        self.done = False

    def query(self):
        return self.done

    def synchronize(self):
        self.done = True


class BufferTests(unittest.TestCase):
    def buffer(self, strategy, capacity=256, feedback=False, alignment=16):
        limits = DeviceMemoryLimits(
            32,
            headroom_bytes=32,
            progress_bytes=capacity * 2,
            dictionary_bytes=32,
            buffer_strategy=strategy,
            buffer_bytes=capacity,
            buffer_alignment=alignment,
        )
        memory = DeviceMemoryPool(
            "cpu", limits, available=lambda: max(capacity * 8, capacity * 5 + 128)
        )
        return ReceiveBuffers(memory, feedback=feedback)

    def acquire(self, buffer, size, timeout=0):
        return buffer.acquire([(0, (size,), torch.uint8)], timeout=timeout)

    def test_fifo_ring_waits_for_oldest_but_pool_can_reuse_later_completion(self):
        for strategy in ("ring", "pool"):
            with self.subTest(strategy=strategy):
                buffer = self.buffer(strategy)
                a, av = self.acquire(buffer, 128)
                b, bv = self.acquire(buffer, 128)
                av[0].fill_(11)
                bv[0].fill_(22)
                b.close()
                if strategy == "ring":
                    with self.assertRaises(TimeoutError):
                        self.acquire(buffer, 128)
                    self.assertGreater(buffer.head_blocked_waits, 0)
                else:
                    c, cv = self.acquire(buffer, 128)
                    cv[0].fill_(33)
                    self.assertTrue(torch.all(av[0] == 11))
                    c.close()
                a.close()
                buffer.close()

    def test_ring_contiguous_wrap_padding_and_no_live_overwrite(self):
        buffer = self.buffer("ring")
        a, _ = self.acquire(buffer, 96)
        b, bv = self.acquire(buffer, 96)
        bv[0].fill_(7)
        a.close()
        c, cv = self.acquire(buffer, 80)
        self.assertEqual(c.start % buffer.capacity, 0)
        self.assertEqual(buffer.wrap_padding_bytes, 64)
        cv[0].fill_(9)
        self.assertTrue(torch.all(bv[0] == 7))
        c.close()
        b.close()
        buffer.close()

    def test_completed_cpu_dequeue_does_not_release_inflight_gpu_storage(self):
        for strategy in ("ring", "pool"):
            buffer = self.buffer(strategy)
            a, _ = self.acquire(buffer, 256)
            event = Event()
            buffer.memory.retain(event, None, a)
            with self.assertRaises(TimeoutError):
                self.acquire(buffer, 16)
            event.done = True
            b, _ = self.acquire(buffer, 16)
            b.close()
            buffer.close()

    def test_release_wakes_blocked_receiver(self):
        for strategy in ("ring", "pool"):
            buffer = self.buffer(strategy)
            a, _ = self.acquire(buffer, 256)
            result = []
            thread = threading.Thread(target=lambda: result.append(self.acquire(buffer, 64, 2)))
            thread.start()
            time.sleep(0.03)
            self.assertFalse(result)
            a.close()
            thread.join(2)
            self.assertFalse(thread.is_alive())
            result[0][0].close()
            buffer.close()

    def test_mixed_lengths_dtypes_zero_shape_reuse_and_shared_dictionary(self):
        for strategy in ("ring", "pool"):
            buffer = self.buffer(strategy, 2048)
            dictionary = TensorDictionary()
            shared = SharedTensor.snapshot(torch.arange(5))
            for i in range(40):
                original = {
                    "hidden": torch.arange(i % 13 + 1, dtype=torch.float32),
                    "mask": torch.ones(3, dtype=torch.bool),
                    "empty": torch.empty(0),
                    "shared": shared,
                }
                tensors = []
                spec = _encode(original, tensors, TensorDictionary())
                use, views = buffer.acquire(_receive_layout(spec, "cpu"))
                received = []
                value = _decode(spec, received, "cpu", dictionary, views)
                for index, tensor in received:
                    tensor.copy_(tensors[index])
                torch.testing.assert_close(value["hidden"], original["hidden"])
                self.assertEqual(value["empty"].numel(), 0)
                use.close()
            # Shared declarations never alias recycled ring/pool storage.
            torch.testing.assert_close(dictionary.get(shared.key).value, shared.value)
            self.assertGreater(buffer.reuses, 0)
            buffer.close()

    def test_feedback_capacity_survives_exhausted_request_admission(self):
        for strategy in ("ring", "pool"):
            buffer = self.buffer(strategy, feedback=True)
            normal = buffer.memory.try_reserve(buffer.memory.normal_capacity)
            self.assertIsNotNone(normal)
            self.assertIsNone(buffer.memory.try_reserve(1))
            use, _ = self.acquire(buffer, 256)
            with buffer.memory.workspace():
                pass
            use.close()
            normal.close()
            buffer.close()
            self.assertEqual(buffer.memory.snapshot()["live_leases"], 0)

    def test_first_feedback_arena_can_be_created_with_wire_budget_full(self):
        for strategy in ("ring", "pool"):
            ordinary = self.buffer(strategy)
            memory = ordinary.memory
            wire = memory.reserve(memory.capacities["wire"], purpose="wire")
            self.assertIsNone(memory.try_reserve(1, purpose="wire"))
            feedback = ReceiveBuffers(memory, feedback=True)
            use, _ = self.acquire(feedback, feedback.capacity)
            use.close()
            feedback.close()
            wire.close()
            ordinary.close()
            self.assertEqual(memory.snapshot()["promised_bytes"], 0)

    def test_ring_small_alignment_supports_mixed_dtypes_and_wrap(self):
        buffer = self.buffer("ring", capacity=37, alignment=1)
        a, av = self.acquire(buffer, 1)
        av[0].fill_(7)
        b, bv = buffer.acquire([(0, (2,), torch.float32)])
        self.assertEqual(b.start % 4, 0)
        bv[0].fill_(8)
        c, cv = buffer.acquire([(0, (1,), torch.complex128)])
        self.assertEqual(c.start % 16, 0)
        cv[0].fill_(9)
        a.close()
        b.close()
        d, dv = buffer.acquire([(0, (2,), torch.float32)])
        self.assertEqual(d.start % buffer.capacity, 0)
        dv[0].fill_(10)
        self.assertEqual(cv[0].item(), 9)
        c.close()
        d.close()
        self.assertGreater(buffer.alignment_padding_bytes, 0)
        self.assertGreater(buffer.wrap_padding_bytes, 0)
        buffer.close()

    def test_view_construction_failure_releases_acquired_slot(self):
        for strategy in ("ring", "pool"):
            buffer = self.buffer(strategy)
            with patch.object(torch.Tensor, "view", side_effect=RuntimeError("view failed")):
                with self.assertRaisesRegex(RuntimeError, "view failed"):
                    self.acquire(buffer, 256)
            self.assertEqual(buffer.live_bytes, 0)
            use, _ = self.acquire(buffer, 256)
            use.close()
            buffer.close()
