"""Shutdown must free queued receive capacity while receivers are still alive."""

import queue
import threading
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import torch

from monowire.buffers import SLOT_KEY, ReceiveBuffers, retire_message
from monowire.channels import ReceivedMessage
from monowire.device_memory import DeviceMemoryLimits, DeviceMemoryPool
from monowire.transport import _receive_packets, _ReceivedPacket, _StageService


class BufferShutdownTests(unittest.TestCase):
    def test_prefetch_credit_does_not_release_resident_storage(self):
        for strategy in ("ring", "pool"):
            for depth in (1, 3):
                with self.subTest(strategy=strategy, depth=depth):
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
                    incoming, draining = queue.Queue(), threading.Event()
                    extra_received = threading.Event()

                    class Channel:
                        memory_pool = memory
                        calls = 0

                        def recv(self):
                            self.calls += 1
                            if self.calls > depth + 1:
                                return ReceivedMessage({"stop": True})
                            use, _ = buffer.acquire([(0, (16,), torch.uint8)])
                            if self.calls == depth + 1:
                                extra_received.set()
                            return ReceivedMessage({SLOT_KEY: [use]})

                    receiver = threading.Thread(
                        target=_receive_packets,
                        args=(Channel(), incoming, 0, draining, 99),
                        kwargs={"receive_prefetch": depth},
                        daemon=True,
                    )
                    packets = []
                    receiver.start()
                    try:
                        packets.extend(incoming.get(timeout=1) for _ in range(depth))
                        # Merely dequeuing does not release the receive credit.
                        self.assertFalse(extra_received.wait(0.1))
                        self.assertEqual(buffer.live_bytes, depth * 16)
                        packets[0].release()  # Consumed; last GPU use is still pending.
                        self.assertTrue(extra_received.wait(1))
                        packets.append(incoming.get(timeout=1))
                        self.assertEqual(buffer.live_bytes, (depth + 1) * 16)
                    finally:
                        # Drain must bypass exhausted prefetch credits to read STOP.
                        draining.set()
                        receiver.join(2)
                        while not incoming.empty():
                            packets.append(incoming.get_nowait())
                        for packet in packets:
                            retire_message(packet.message, packet.ready)
                            packet.release()
                        memory.drain()
                    self.assertFalse(receiver.is_alive())
                    self.assertEqual(buffer.live_bytes, 0)
                    buffer.close()

    def test_full_incoming_and_racing_arrivals_drain_before_join(self):
        for strategy in ("ring", "pool"):
            for late_arrival in (False, True):
                with self.subTest(strategy=strategy, late_arrival=late_arrival):
                    self.check_shutdown(strategy, late_arrival)

    def check_shutdown(self, strategy, late_arrival):
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
        use, _ = buffer.acquire([(0, (256,), torch.uint8)])
        service = object.__new__(_StageService)
        service.draining = threading.Event()
        service.pending = deque()
        service.incoming = queue.Queue()
        service.computations = SimpleNamespace(drain=memory.drain)
        service.senders = {}
        packet = _ReceivedPacket(0, {SLOT_KEY: [use]}, None)
        entered = threading.Event()
        failures = []

        class Channel:
            memory_pool = memory
            calls = 0

            def recv(self):
                self.calls += 1
                if self.calls == 2:
                    return ReceivedMessage({"stop": True})
                entered.set()
                slot, _ = buffer.acquire([(0, (16,), torch.uint8)], timeout=2)
                return ReceivedMessage({SLOT_KEY: [slot]})

        def receive():
            try:
                _receive_packets(Channel(), service.incoming, 0, service.draining, 2)
            except BaseException as error:
                failures.append(error)

        receiver = threading.Thread(target=receive)
        service.receivers = [receiver]
        producer = None
        if late_arrival:
            first_discard = threading.Event()
            discard = service._discard_incoming

            def observed_discard():
                discard()
                first_discard.set()

            service._discard_incoming = observed_discard

            def publish():
                if first_discard.wait(1):
                    service.incoming.put(packet)

            producer = threading.Thread(target=publish)
            producer.start()
        else:
            service.incoming.put(packet)
        receiver.start()
        self.assertTrue(entered.wait(1))
        with patch("monowire.transport.SHUTDOWN_TIMEOUT", 1):
            errors = service._close()
        receiver.join(2)
        if producer is not None:
            producer.join(2)
        self.assertEqual(errors, [])
        self.assertEqual(failures, [])
        self.assertFalse(receiver.is_alive())
        self.assertTrue(service.incoming.empty())
        self.assertEqual(buffer.live_bytes, 0)
        buffer.close()
