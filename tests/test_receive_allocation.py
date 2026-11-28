"""Receive storage must be allocated on its first writer's CUDA stream."""

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch

from monowire.buffers import ReceiveBuffers, retire_message
from monowire.channels import TensorChannel, _encode
from monowire.device_memory import DeviceMemoryLimits, DeviceMemoryPool


def make_channel(strategy, device):
    memory = DeviceMemoryPool(
        device,
        DeviceMemoryLimits(
            1024**2,
            headroom_bytes=1024**2,
            progress_bytes=1024**2,
            dictionary_bytes=1024,
            buffer_strategy=strategy,
            buffer_bytes=4 * 1024**2,
        ),
        available=lambda: 64 * 1024**2,
    )
    return TensorChannel(0, 1, None, None, device, memory_pool=memory)


class ReceiveAllocationTests(unittest.TestCase):
    def test_failed_admission_acknowledgement_releases_slot(self):
        for strategy in ("ring", "pool"):
            with self.subTest(strategy=strategy):
                channel = make_channel(strategy, "cpu")
                spec = _encode({"hidden": torch.ones(32, dtype=torch.uint8)}, [])

                def metadata(values, **kwargs):
                    values[0] = spec

                with (
                    patch("monowire.channels.dist.recv_object_list", side_effect=metadata),
                    patch("monowire.channels.dist.send_object_list", side_effect=OSError("ack")),
                    patch.object(channel, "_transfer") as transfer,
                    self.assertRaisesRegex(OSError, "ack"),
                ):
                    channel.recv()
                transfer.assert_not_called()
                self.assertEqual(channel.receive_buffers.live_bytes, 0)
                channel.drain()
                channel.receive_buffers.close()
                self.assertEqual(channel.memory_pool.snapshot()["live_leases"], 0)

    def test_admission_allocates_inside_receive_context(self):
        for strategy in ("ring", "pool"):
            with self.subTest(strategy=strategy):
                channel = make_channel(strategy, "cpu")
                in_context = False

                @contextmanager
                def context():
                    nonlocal in_context
                    self.assertFalse(in_context)
                    in_context = True
                    try:
                        yield
                    finally:
                        in_context = False

                acquire = ReceiveBuffers.acquire

                def allocate(buffer, *args, **kwargs):
                    self.assertTrue(in_context)
                    return acquire(buffer, *args, **kwargs)

                spec = _encode({"hidden": torch.empty(32, dtype=torch.uint8)}, [])

                def metadata(values, **kwargs):
                    values[0] = spec

                with (
                    patch.object(channel, "_context", context),
                    patch("monowire.channels.dist.recv_object_list", side_effect=metadata),
                    patch("monowire.channels.dist.send_object_list"),
                    patch.object(channel, "_transfer"),
                    patch.object(ReceiveBuffers, "acquire", allocate),
                ):
                    received = channel.recv()
                retire_message(received.value)
                channel.drain()
                channel.receive_buffers.close()
                self.assertEqual(channel.memory_pool.snapshot()["live_leases"], 0)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_cuda_allocation_uses_first_writer_stream(self):
        for strategy in ("ring", "pool"):
            with self.subTest(strategy=strategy):
                channel = make_channel(strategy, "cuda:0")
                # No NCCL is needed to test allocator/first-writer ordering.
                spec = ("dict", [("hidden", ("tensor", 0, (4 * 1024**2,), torch.uint8, True))])
                streams = []
                allocate = ReceiveBuffers.acquire

                def acquire(buffer, *args, **kwargs):
                    streams.append(torch.cuda.current_stream(channel.device).cuda_stream)
                    return allocate(buffer, *args, **kwargs)

                def metadata(values, **kwargs):
                    values[0] = spec

                def transfer(tensors, *_):
                    for tensor in tensors:
                        tensor.fill_(7)

                with (
                    patch("monowire.channels.dist.recv_object_list", side_effect=metadata),
                    patch("monowire.channels.dist.send_object_list"),
                    patch.object(channel, "_transfer", transfer),
                    patch.object(ReceiveBuffers, "acquire", acquire),
                ):
                    received = channel.recv()
                try:
                    self.assertEqual(streams, [channel.stream.cuda_stream])
                    message = received.on_current_stream()
                    self.assertTrue(bool((message["hidden"] == 7).all()))
                finally:
                    torch.cuda.synchronize(channel.device)
                    retire_message(received.value)
                    channel.drain()
                    channel.receive_buffers.close()
