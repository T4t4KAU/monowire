"""Bounded GPU submission, stream dependencies and asynchronous buffer ownership."""

import gc
import tempfile
import threading
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from monowire.channels import (
    BufferedSender,
    InFlightQueue,
    ReceivedMessage,
    TensorChannel,
    _encode,
    close_channels,
    ready_event,
)
from monowire.tensor_payload import SharedTensor, TensorParts
from monowire.topology import PipelineTopology
from monowire.transport import _ReceivedPacket, _StageService


class Event:
    def __init__(self, name, calls, *, complete=False):
        self.name, self.calls, self.complete = name, calls, complete

    def query(self):
        return self.complete

    def wait(self):
        self.calls.append(("dependency", self.name))

    def synchronize(self):
        self.calls.append(("host_wait", self.name))
        self.complete = True


def p2p_operation(operation, tensor, peer, group):
    """Test operation descriptor without a live distributed process group."""
    return SimpleNamespace(op=operation, tensor=tensor, peer=peer, group=group)


class InFlightTests(unittest.TestCase):
    def test_sender_failure_cannot_leave_late_publication_without_consumer(self):
        for dynamic in (False, True):
            with self.subTest(dynamic=dynamic):
                fail, failed, checked, resume = (threading.Event() for _ in range(4))

                class Channel:
                    memory_pool = object() if dynamic else None

                    def send(self, *args):
                        if not fail.wait(2):
                            raise TimeoutError("test failed to release sender")
                        raise OSError("send failed")

                sender = BufferedSender(Channel(), 2, on_error=lambda _: failed.set())
                sender.submit({"first": True})
                check = sender.check

                def paused_check():
                    check()
                    checked.set()
                    if not resume.wait(2):
                        raise TimeoutError("test failed to release publisher")

                errors = []

                def publish():
                    try:
                        sender.submit_nowait({"late": True})
                    except RuntimeError as error:
                        errors.append(error)

                with patch.object(sender, "check", paused_check):
                    producer = threading.Thread(target=publish)
                    producer.start()
                    try:
                        self.assertTrue(checked.wait(1))
                        fail.set()
                        # Old code finishes its final drain here, then accepts
                        # the paused publisher. Atomic publication prevents it.
                        failed.wait(0.1)
                    finally:
                        resume.set()
                    producer.join(2)
                    sender.thread.join(2)
                self.assertFalse(producer.is_alive())
                self.assertFalse(sender.thread.is_alive())
                self.assertIsInstance(sender.error, OSError)
                self.assertTrue(sender.queue.empty())

    def test_count_channel_does_not_enter_buffer_admission_protocol(self):
        channel = TensorChannel(0, 1, "gloo", None, "cpu", capacity=2)
        spec = _encode({"hidden": torch.ones(2)}, [])

        def metadata(values, **kwargs):
            values[0] = spec

        with (
            patch("monowire.channels.dist.send_object_list") as send_metadata,
            patch("monowire.channels.dist.recv_object_list", side_effect=metadata) as recv_metadata,
            patch.object(channel, "_transfer"),
            patch("monowire.channels.ReceiveBuffers") as buffers,
        ):
            channel.send({"hidden": torch.ones(2)})
            self.assertEqual(send_metadata.call_count, 1)
            recv_metadata.assert_not_called()  # No downstream approval round trip.
            received = channel.recv()
            self.assertEqual(recv_metadata.call_count, 1)
            self.assertEqual(send_metadata.call_count, 1)  # No approval response.
            buffers.assert_not_called()
        self.assertEqual(received.value["hidden"].shape, (2,))
        self.assertIsNone(channel.memory_pool)
        self.assertIsNone(channel.receive_buffers)

    def test_async_transport_is_default_and_can_be_disabled(self):
        from monowire.continuous import ContinuousConfig

        self.assertTrue(ContinuousConfig().async_transport)
        self.assertFalse(ContinuousConfig(async_transport=False).async_transport)
        with self.assertRaises(ValueError):
            ContinuousConfig(async_transport="false")

    def test_disabled_async_channel_waits_for_transfer_completion(self):
        calls = []
        channel = TensorChannel(0, 1, "gloo", None, "cpu", async_transport=False)
        spec = _encode({"hidden": torch.ones(2)}, [])

        def metadata(values, **kwargs):
            values[0] = spec

        def transfer(tensor, peer, *, group):
            calls.append(("transfer", tensor.numel()))
            tensor.fill_(9)
            return SimpleNamespace(wait=lambda: calls.append(("work_wait", None)))

        with (
            patch("monowire.channels.dist.send_object_list"),
            patch("monowire.channels.dist.recv_object_list", side_effect=metadata),
            patch("monowire.channels.dist.isend", side_effect=transfer),
            patch("monowire.channels.dist.irecv", side_effect=transfer),
            patch("monowire.channels.dist.batch_isend_irecv") as grouped,
            patch(
                "monowire.channels.ready_event",
                side_effect=[Event("send", calls), Event("recv", calls)],
            ),
        ):
            channel.send({"hidden": torch.ones(2)})
            received = channel.recv()
        grouped.assert_not_called()
        self.assertIsNone(received.ready)
        self.assertEqual(
            calls,
            [
                ("transfer", 2),
                ("work_wait", None),
                ("host_wait", "send"),
                ("transfer", 2),
                ("work_wait", None),
                ("host_wait", "recv"),
            ],
        )
        self.assertFalse(channel.inflight.pending)
        torch.testing.assert_close(received.value["hidden"], torch.full((2,), 9.0))

    def test_disabled_async_stage_waits_after_publishing(self):
        calls = []
        stage = SimpleNamespace(
            group=SimpleNamespace(rank_in_group=1, device="cpu"),
            compute=lambda packet: calls.append(("compute", None)),
        )
        with patch("monowire.transport.BufferedSender") as sender:
            sender.return_value.submit.side_effect = lambda *args: calls.append(("send", None))
            service = _StageService(
                stage,
                PipelineTopology((1, 1, 1)),
                {(0, 1): None, (1, 2): None},
                8,
                None,
                async_transport=False,
            )
            with patch("monowire.transport.ready_event", return_value=Event("done", calls)):
                service._process_batch({"routes": [], "payload": {}})
        self.assertEqual(calls, [("compute", None), ("send", None), ("host_wait", "done")])

    def test_work_is_retained_and_only_capacity_exhaustion_waits(self):
        class Storage:
            pass

        calls = []
        pending = InFlightQueue(2)
        storage = Storage()
        reference = weakref.ref(storage)
        pending.reserve()
        pending.track(Event("first", calls), storage)
        del storage
        gc.collect()
        self.assertIsNotNone(reference())
        pending.reserve()
        pending.track(Event("second", calls), None)
        self.assertEqual(calls, [])
        pending.reserve()
        self.assertEqual(calls, [("host_wait", "first")])
        self.assertIsNone(reference())
        pending.drain()
        self.assertEqual(calls[-1], ("host_wait", "second"))
        self.assertFalse(pending.pending)

    def test_finished_work_and_cpu_messages_need_no_host_wait(self):
        calls = []
        pending = InFlightQueue(1)
        pending.track(Event("done", calls, complete=True), None)
        pending.reserve()
        pending.track(None, None)
        pending.drain()
        self.assertEqual(calls, [])
        self.assertFalse(pending.pending)

    def test_failed_completion_retains_resources_and_propagates(self):
        event = Event("failed", [])
        pending = InFlightQueue(1)
        pending.track(event, object())
        with patch.object(event, "synchronize", side_effect=RuntimeError("device failed")):
            with self.assertRaisesRegex(RuntimeError, "device failed"):
                pending.reserve()
            self.assertEqual(len(pending.pending), 1)
            with self.assertRaisesRegex(RuntimeError, "device failed"):
                pending.drain()
        pending.drain()

    def test_receiver_installs_dependency_without_a_host_wait(self):
        calls = []
        value = {"hidden": torch.ones(2)}
        received = ReceivedMessage(value, Event("recv", calls))
        with patch("monowire.channels.record_stream") as record:
            self.assertIs(received.on_current_stream(), value)
            record.assert_called_once_with(value)
        self.assertEqual(calls, [("dependency", "recv")])

    def test_stage_launches_two_batches_before_waiting_for_capacity(self):
        calls = []

        class Stage:
            group = SimpleNamespace(rank_in_group=1, device="cpu")

            def compute(self, packet):
                calls.append(("compute", packet["id"]))

        class Sender:
            def __init__(self, channel, capacity):
                pass

            def submit(self, message, ready=None):
                calls.append(("send", ready.name))

        topology = PipelineTopology((1, 1, 1), buffer_size=2)
        with patch("monowire.transport.BufferedSender", Sender):
            service = _StageService(Stage(), topology, {(0, 1): None, (1, 2): None}, 8, None)
        with patch(
            "monowire.transport.ready_event",
            side_effect=[Event(str(i), calls) for i in range(3)],
        ):
            for i in range(3):
                service.computations.reserve()
                service._process_batch({"routes": [], "payload": {"id": i}})
        self.assertEqual(
            calls,
            [
                ("compute", 0),
                ("send", "0"),
                ("compute", 1),
                ("send", "1"),
                ("host_wait", "0"),
                ("compute", 2),
                ("send", "2"),
            ],
        )
        service.computations.drain()
        self.assertEqual(calls[-2:], [("host_wait", "1"), ("host_wait", "2")])

    def test_regrouping_waits_only_for_selected_packets(self):
        calls = []
        stage = SimpleNamespace(
            group=SimpleNamespace(rank_in_group=2, device="cpu"),
            combine_requests=lambda items: {"hidden": torch.cat([x["hidden"] for x in items])},
        )
        with patch("monowire.transport.BufferedSender"):
            service = _StageService(
                stage,
                PipelineTopology((1, 1), 2),
                {(0, 2): None, (1, 2): None, (2, 0): None, (2, 1): None},
                1,
                None,
            )
        for owner, size in ((0, 2), (1, 1)):
            service.pending.append(
                _ReceivedPacket(
                    owner,
                    {
                        "items": [
                            {
                                "hidden": torch.ones(1, 2),
                                "owner": owner,
                                "ticket": i,
                                "compatibility": owner,
                            }
                            for i in range(size)
                        ]
                    },
                    None,
                    Event(str(owner), calls),
                )
            )
        for expected_calls in (
            [("dependency", "0")],
            [("dependency", "0")],
            [("dependency", "0"), ("dependency", "1")],
        ):
            service._pack_pending_batch()
            self.assertEqual(calls, expected_calls)

    def test_channel_posts_all_tensors_without_per_message_host_wait(self):
        calls = []
        channel = TensorChannel(0, 1, "gloo", None, "cpu", capacity=2, async_transport=True)

        def post(operations):
            calls.append(("post", len(operations)))
            return [
                SimpleNamespace(wait=lambda: calls.append(("work_wait", None))) for _ in operations
            ]

        with (
            patch("monowire.channels.dist.send_object_list"),
            patch("monowire.channels.dist.P2POp", side_effect=p2p_operation),
            patch("monowire.channels.dist.batch_isend_irecv", side_effect=post),
            patch(
                "monowire.channels.ready_event",
                side_effect=[Event(str(i), calls) for i in range(3)],
            ),
        ):
            channel.send({"values": [torch.ones(1), torch.ones(2)]})
            self.assertEqual(calls, [("post", 2), ("work_wait", None), ("work_wait", None)])
            channel.send({"value": torch.ones(1)})
            self.assertFalse(any(kind == "host_wait" for kind, _ in calls))
            channel.send({"value": torch.ones(1)})
            self.assertEqual([x for x in calls if x[0] == "host_wait"], [("host_wait", "0")])
        channel.drain()

    def test_channel_returns_receive_event_before_gpu_completion(self):
        calls = []
        channel = TensorChannel(0, 1, "gloo", None, "cpu", async_transport=True)
        spec = _encode({"hidden": torch.ones(2)}, [])

        def metadata(values, **kwargs):
            values[0] = spec

        def post(operations):
            for operation in operations:
                operation.tensor.fill_(9)
            return [SimpleNamespace(wait=lambda: None)]

        with (
            patch("monowire.channels.dist.recv_object_list", side_effect=metadata),
            patch("monowire.channels.dist.P2POp", side_effect=p2p_operation),
            patch("monowire.channels.dist.batch_isend_irecv", side_effect=post),
            patch("monowire.channels.ready_event", return_value=Event("recv", calls)),
        ):
            received = channel.recv()
        self.assertEqual(calls, [])
        self.assertFalse(received.ready.query())
        message = received.on_current_stream()
        self.assertEqual(calls, [("dependency", "recv")])
        torch.testing.assert_close(message["hidden"], torch.full((2,), 9.0))
        channel.drain()

    def test_transfer_failure_still_joins_other_posted_work(self):
        calls = []
        channel = TensorChannel(0, 1, "gloo", None, "cpu", async_transport=True)

        def fail():
            calls.append("failed")
            raise RuntimeError("transfer failed")

        works = [SimpleNamespace(wait=fail), SimpleNamespace(wait=lambda: calls.append("joined"))]
        with (
            patch("monowire.channels.dist.send_object_list"),
            patch("monowire.channels.dist.P2POp", side_effect=p2p_operation),
            patch("monowire.channels.dist.batch_isend_irecv", return_value=works),
        ):
            with self.assertRaises(ExceptionGroup) as caught:
                channel.send({"values": [torch.ones(1), torch.ones(1)]})
        self.assertEqual(calls, ["failed", "joined"])
        self.assertEqual(str(caught.exception.exceptions[0]), "transfer failed")

    def test_channel_cleanup_continues_after_drain_failure(self):
        def fail():
            raise RuntimeError("drain failed")

        channels = {
            0: SimpleNamespace(
                drain=fail,
                gpu_group="gpu",
                cpu_group="cpu0",
                dictionary=SimpleNamespace(clear=lambda: None),
            ),
            1: SimpleNamespace(
                drain=lambda: None,
                gpu_group=None,
                cpu_group="cpu1",
                dictionary=SimpleNamespace(clear=lambda: None),
            ),
        }
        with patch("monowire.channels.dist.destroy_process_group") as destroy:
            with self.assertRaises(ExceptionGroup):
                close_channels(channels)
        self.assertEqual([call.args[0] for call in destroy.call_args_list], ["gpu", "cpu0", "cpu1"])

    def test_sender_drains_before_stop_and_reports_drain_failure(self):
        class Channel:
            def __init__(self, fail=False):
                self.calls, self.fail = [], fail

            def send(self, message, ready=None):
                self.calls.append(message)

            def drain(self):
                self.calls.append("drain")
                if self.fail:
                    raise RuntimeError("completion failed")

        channel = Channel()
        sender = BufferedSender(channel, 2)
        sender.submit({"data": 1})
        sender.close()
        self.assertEqual(channel.calls, [{"data": 1}, "drain", {"stop": True}, "drain"])
        failed = Channel(fail=True)
        sender = BufferedSender(failed, 2)
        with self.assertRaisesRegex(RuntimeError, "sender failed") as caught:
            sender.close()
        self.assertEqual(str(caught.exception.__cause__), "completion failed")
        self.assertFalse(sender.thread.is_alive())


def transfer_worker(rank, store, cuda, async_transport):
    device = torch.device(f"cuda:{rank}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("gloo", init_method=Path(store).as_uri(), rank=rank, world_size=2)
    cpu = dist.new_group([0, 1], backend="gloo")
    gpu = dist.new_group([0, 1], backend="nccl") if cuda else None
    channel = TensorChannel(0, 1, cpu, gpu, device, capacity=2, async_transport=async_transport)
    try:
        with torch.inference_mode():
            shared = SharedTensor.snapshot(torch.arange(16, device=device).reshape(8, 2))
            for index in range(12):
                if rank == 0:
                    # Noncontiguous views require staging on the communication
                    # stream, while CPU data travels independently via Gloo.
                    value = torch.arange(24, device=device).reshape(4, 6).t() + index
                    channel.send(
                        {
                            "index": index,
                            "values": [value, value[:, ::2]],
                            "cpu": torch.tensor(index),
                            "static": TensorParts((shared, shared)),
                        },
                        ready_event(device),
                    )
                    del value
                else:
                    received = channel.recv()
                    if cuda:
                        # Consumption on a third stream must obey receive
                        # readiness and retain storage after the wrapper is gone.
                        stream = torch.cuda.Stream(device=device)
                        with torch.cuda.stream(stream):
                            message = received.on_current_stream()
                            outputs = [value.clone() for value in message["values"]]
                            static = message["static"].materialize().clone()
                        stream.synchronize()
                    else:
                        message = received.on_current_stream()
                        outputs = message["values"]
                        static = message["static"].materialize()
                    expected = torch.arange(24, device=device).reshape(4, 6).t() + index
                    assert message["index"] == index
                    assert message["cpu"].item() == index
                    torch.testing.assert_close(outputs[0], expected)
                    torch.testing.assert_close(outputs[1], expected[:, ::2])
                    torch.testing.assert_close(
                        static, torch.arange(16, device=device).reshape(8, 2).repeat(2, 1)
                    )
                    del message, received, outputs
            channel.drain()
            dist.barrier()
    finally:
        channel.drain()
        if gpu is not None:
            dist.destroy_process_group(gpu)
        dist.destroy_process_group(cpu)
        dist.destroy_process_group()


class ChannelTransferTests(unittest.TestCase):
    def test_gloo_grouped_tensor_transfers(self):
        for enabled in (False, True):
            with self.subTest(async_transport=enabled), tempfile.TemporaryDirectory() as folder:
                mp.spawn(
                    transfer_worker, args=(str(Path(folder, "store")), False, enabled), nprocs=2
                )

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires two CUDA devices")
    def test_nccl_cross_stream_noncontiguous_transfers(self):
        for enabled in (False, True):
            with self.subTest(async_transport=enabled), tempfile.TemporaryDirectory() as folder:
                mp.spawn(
                    transfer_worker, args=(str(Path(folder, "store")), True, enabled), nprocs=2
                )
