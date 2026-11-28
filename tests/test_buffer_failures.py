"""Exceptional ownership exits must retire storage after its final GPU event."""

import queue
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import torch

from monowire.buffers import SLOT_KEY, ReceiveBuffers
from monowire.channels import InFlightQueue
from monowire.device_memory import DeviceMemoryLimits, DeviceMemoryPool
from monowire.transport import _ReceivedPacket, _StageService


class Abort(BaseException):
    pass


class Completion:
    def __init__(self):
        self.done = False

    def query(self):
        return self.done

    def synchronize(self):
        self.done = True


def setup(strategy):
    memory = DeviceMemoryPool(
        "cpu",
        DeviceMemoryLimits(
            32,
            headroom_bytes=32,
            progress_bytes=256,
            dictionary_bytes=32,
            buffer_bytes=256,
            buffer_strategy=strategy,
            buffer_alignment=16,
        ),
        available=lambda: 2048,
    )
    buffer = ReceiveBuffers(memory)
    use, tensors = buffer.acquire([(0, (16,), torch.uint8)])
    service = object.__new__(_StageService)
    service.memory_pool = memory
    service.rank = service.first_rank = service.last_rank = 1
    service.stage = SimpleNamespace(group=SimpleNamespace(device="cpu"))
    service.on_event = None
    service.computations = InFlightQueue(2, memory_pool=memory)
    service.senders = {}
    service.stopped = set()
    service.pending = deque()
    service.incoming = queue.Queue()
    service.batch_size = 1
    return service, buffer, use, tensors[0]


class BufferFailureTests(unittest.TestCase):
    def assert_retired_after_event(self, buffer, event):
        # A host exception must not overwrite storage still used by queued GPU work.
        self.assertEqual(buffer.live_bytes, 16)
        event.done = True
        buffer.memory.drain()
        self.assertEqual(buffer.live_bytes, 0)
        buffer.close()

    def test_compute_abort_and_invalid_result_retire_owned_slot(self):
        for strategy in ("ring", "pool"):
            for abort in (True, False):
                with self.subTest(strategy=strategy, abort=abort):
                    service, buffer, use, _ = setup(strategy)

                    def compute(payload):
                        if abort:
                            raise Abort("compute interrupted")
                        # Malformed output: split on missing hidden must still clean up.

                    service.stage.compute = compute
                    envelope = {SLOT_KEY: [use], "routes": [(0, 0, 1)], "payload": {}}
                    event = Completion()
                    with (
                        patch("monowire.transport.ready_event", return_value=event),
                        self.assertRaises(Abort if abort else KeyError),
                    ):
                        service._process_batch(envelope)
                    self.assert_retired_after_event(buffer, event)

    def test_pack_abort_retires_consumed_packet(self):
        for strategy in ("ring", "pool"):
            with self.subTest(strategy=strategy):
                service, buffer, use, hidden = setup(strategy)

                def combine(items):
                    raise Abort("packing interrupted")

                service.stage.combine_requests = combine
                service.pending.append(
                    _ReceivedPacket(
                        0,
                        {
                            SLOT_KEY: [use],
                            "items": [dict(hidden=hidden, compatibility="x", owner=0, ticket=0)],
                        },
                        None,
                    )
                )
                event = Completion()
                with (
                    patch("monowire.transport.ready_event", return_value=event),
                    self.assertRaises(Abort),
                ):
                    service._pack_pending_batch()
                self.assert_retired_after_event(buffer, event)

    def test_invalid_received_packet_retires_receive_storage(self):
        for strategy in ("ring", "pool"):
            with self.subTest(strategy=strategy):
                service, buffer, use, _ = setup(strategy)
                event = Completion()
                service.incoming.put(
                    _ReceivedPacket(0, {SLOT_KEY: [use], "items": []}, None, event)
                )
                with self.assertRaisesRegex(ValueError, "Empty pipeline"):
                    service._receive()
                self.assert_retired_after_event(buffer, event)
