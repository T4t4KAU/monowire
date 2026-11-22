"""Real distributed routing: overlapping tickets, ragged rows and early owner exit."""

import asyncio
import queue
import tempfile
import threading
import time
import unittest
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from monowire.channels import close_channels, create_channels
from monowire.continuous import PreparedStep
from monowire.topology import PipelineTopology
from monowire.transport import PipelineExecutor, _ReceivedPacket, serve_stage


class RoutedStage:
    def __init__(self, rank, device):
        self.group = SimpleNamespace(rank_in_group=rank, device=device)

    def validate(self, prepared):
        pass

    def reuse(self, prepared):
        return [p.payload + 100 for p in prepared]

    def export_requests(self, prepared):
        return [{"hidden": p.payload + 1, "compatibility": "toy"} for p in prepared]

    def combine_requests(self, items):
        return {"hidden": torch.cat([item["hidden"] for item in items])}

    def compute(self, packet):
        if bool((packet["hidden"] < 0).any()):
            raise ValueError("toy stage failure")
        packet["hidden"] += self.group.rank_in_group + 1
        return packet


def routed_worker(rank, store, cuda):
    device = torch.device(f"cuda:{rank}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo",
        init_method=Path(store).as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    topology = PipelineTopology((1, 1, 1), 2, buffer_size=1)
    channels = create_channels(topology, device)
    stage = RoutedStage(rank, device)
    events = []

    async def run():
        executor = PipelineExecutor(stage, topology, channels)

        def request(i, hit=False):
            return PreparedStep(
                str(i),
                i % 3,
                hit,
                "toy",
                torch.full((i % 3 + 1, 2), float(rank * 100 + i), device=device),
            )

        batches = [[request(i), request(i + 1)] for i in range(0, 8 if rank == 0 else 16, 2)]
        results = await asyncio.wait_for(
            asyncio.gather(*(asyncio.gather(*executor.submit(batch)) for batch in batches)), 30
        )
        for batch, outputs in zip(batches, results, strict=True):
            for item, output in zip(batch, outputs, strict=True):
                torch.testing.assert_close(output, item.payload + 8)
        hit = request(5, True)
        torch.testing.assert_close((await executor.submit([hit])[0]), hit.payload + 100)
        if rank == 1:
            bad = request(0)
            bad.payload.fill_(-100)
            try:
                await executor.submit([bad])[0]
            except RuntimeError as error:
                assert "toy stage failure" in str(error)
            else:
                raise AssertionError("Stage error did not return to its request owner")
            good = request(20)
            torch.testing.assert_close((await executor.submit([good])[0]), good.payload + 8)
        await executor.close()
        await executor.close()

    try:
        with torch.inference_mode():
            for _ in range(2):
                events.clear()
                if rank < 2:
                    asyncio.run(run())
                else:
                    serve_stage(stage, topology, channels, batch_size=2, on_event=events.append)
                    assert events and all(1 <= e["batch"] <= 2 for e in events)
                    assert sum(e["batch"] for e in events) == 26
                dist.barrier()
    finally:
        close_channels(channels)
        dist.destroy_process_group()


def idle_worker(rank, store):
    """Normal idle survives; a peer that stops heartbeats still times out."""
    dist.init_process_group(
        "gloo",
        init_method=Path(store).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    topology = PipelineTopology((1, 1))
    with (
        patch("monowire.channels.CHANNEL_TIMEOUT", timedelta(seconds=2)),
        patch("monowire.channels.HEARTBEAT_INTERVAL", 0.1),
    ):
        channels = create_channels(topology, "cpu")
        stage = RoutedStage(rank, "cpu")

        async def run():
            executor = PipelineExecutor(stage, topology, channels)
            executor.start()
            await asyncio.sleep(0)  # Start receiving before blocking the owner.
            for index in range(2):
                # Heartbeats must not depend on the ingress event loop or on
                # request tickets, both before and between requests.
                time.sleep(3)
                request = PreparedStep(str(index), 0, False, "toy", torch.ones(1, 2))
                output = await asyncio.wait_for(executor.submit([request])[0], 2)
                torch.testing.assert_close(output, torch.full((1, 2), 4.0))
            await executor.close()

        try:
            if rank == 0:
                asyncio.run(run())
            else:
                serve_stage(stage, topology, channels, batch_size=2)
            dist.barrier()
            # No sender is running now: a genuinely unresponsive edge must
            # still fail within the configured timeout, not wait forever.
            if rank == 0:
                time.sleep(3)
            else:
                started = time.monotonic()
                try:
                    channels[0, 1].recv()
                except RuntimeError as error:
                    assert "Timed out" in str(error), error
                    assert time.monotonic() - started < 3
                else:
                    raise AssertionError("An unresponsive edge did not time out")
            dist.barrier()
        finally:
            close_channels(channels)
            dist.destroy_process_group()


class RoutedTransportTests(unittest.TestCase):
    def test_gloo_idle_heartbeats_and_unresponsive_peer_timeout(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(idle_worker, args=(str(Path(folder, "store")),), nprocs=2)

    def test_partial_arrivals_merge_across_owners_and_return_to_the_right_owner(self):
        topology = PipelineTopology((1, 1), 2)
        incoming, events, returned = queue.Queue(), [], defaultdict(list)
        for owner in range(2):
            incoming.put(
                _ReceivedPacket(
                    owner,
                    {
                        "items": [
                            {
                                "ticket": ticket,
                                "hidden": torch.full((ticket + 1, 2), owner * 10.0 + ticket),
                                "compatibility": "toy",
                            }
                            for ticket in range(3)
                        ]
                    },
                    None,
                )
            )
        for owner in range(2):
            incoming.put(_ReceivedPacket(owner, {"stop": True}, None))

        class Sender:
            def __init__(self, owner, capacity):
                self.owner = owner

            def submit(self, message, ready=None):
                returned[self.owner].extend(message.get("items", []))

            def check(self):
                pass

            def close(self):
                pass

        links = {(0, 2): None, (1, 2): None, (2, 0): 0, (2, 1): 1}
        with (
            patch("monowire.transport.queue.Queue", return_value=incoming),
            patch("monowire.transport._receive_packets"),
            patch("monowire.transport.BufferedSender", Sender),
        ):
            serve_stage(
                RoutedStage(2, "cpu"), topology, links, batch_size=2, on_event=events.append
            )
        self.assertEqual([event["batch"] for event in events], [2, 2, 2])
        self.assertEqual({owner for owner, _, _ in events[1]["routes"]}, {0, 1})
        for owner, items in returned.items():
            self.assertEqual([item["ticket"] for item in items], [0, 1, 2])
            for item in items:
                ticket = item["ticket"]
                torch.testing.assert_close(
                    item["hidden"], torch.full((ticket + 1, 2), owner * 10.0 + ticket + 3)
                )

    def test_layer_coverage_and_replication(self):
        topology = PipelineTopology((6, 7, 7, 7, 7, 8, 8), 2)
        self.assertEqual(topology.world_size, 8)
        self.assertEqual(topology.layer_range(0), (0, 6))
        self.assertEqual(topology.layer_range(1), (0, 6))
        suffix = [topology.layer_range(rank) for rank in range(2, 8)]
        self.assertEqual(
            [i for start, end in suffix for i in range(start, end)], list(range(6, 50))
        )

    def test_gloo_two_owners_shared_suffix(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(routed_worker, args=(str(Path(folder, "store")), False), nprocs=4)

    @unittest.skipUnless(torch.cuda.device_count() >= 4, "Requires four CUDA devices")
    def test_nccl_two_owners_shared_suffix(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(routed_worker, args=(str(Path(folder, "store")), True), nprocs=4)


class MemoryChannel:
    """FIFO test link with the same blocking send/recv contract as Gloo."""

    def __init__(self):
        self.messages = queue.Queue()

    def send(self, message, ready=None):
        self.messages.put(message)

    def recv(self):
        return self.messages.get(timeout=3)

    def heartbeat(self):
        pass


class ExecutorLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_service_failure_drains_stop_while_packet_credit_is_held(self):
        class StageAbort(BaseException):
            pass

        class AbortingStage(RoutedStage):
            def compute(self, packet):
                raise StageAbort("stage aborted")

        topology = PipelineTopology((1, 1), buffer_size=1)
        incoming, returning = MemoryChannel(), MemoryChannel()
        for start in (0, 16):
            incoming.send(
                {
                    "items": [
                        {"ticket": i, "hidden": torch.ones(1, 2), "compatibility": "toy"}
                        for i in range(start, start + 16)
                    ]
                }
            )
        incoming.send({"stop": True})
        with patch("monowire.transport.SHUTDOWN_TIMEOUT", 0.5):
            with self.assertRaisesRegex(StageAbort, "stage aborted"):
                await asyncio.wait_for(
                    asyncio.to_thread(
                        serve_stage,
                        AbortingStage(1, "cpu"),
                        topology,
                        {(0, 1): incoming, (1, 0): returning},
                        batch_size=8,
                    ),
                    2,
                )
        self.assertTrue(incoming.messages.empty())
        self.assertIn("stage aborted", returning.recv()["failure"])
        self.assertTrue(returning.recv()["stop"])

    async def test_partial_packet_holds_receive_credit_until_all_items_are_consumed(self):
        entered, resume = threading.Event(), threading.Event()
        received = [0, 0]
        events = []

        class InputChannel(MemoryChannel):
            def __init__(self, owner):
                super().__init__()
                self.owner = owner

            def recv(self):
                message = super().recv()
                if "items" in message:
                    received[self.owner] += 1
                return message

        class HeldStage(RoutedStage):
            def compute(self, packet):
                if not entered.is_set():
                    entered.set()
                    if not resume.wait(3):
                        raise TimeoutError("Test did not release stage")
                return super().compute(packet)

        topology = PipelineTopology((1, 1), 2, buffer_size=1)
        links = {(owner, 2): InputChannel(owner) for owner in range(2)}
        links.update({(2, owner): MemoryChannel() for owner in range(2)})
        for owner in range(2):
            for start in range(0, 64, 16):
                links[owner, 2].send(
                    {
                        "items": [
                            {"ticket": i, "hidden": torch.ones(1, 2), "compatibility": "toy"}
                            for i in range(start, start + 16)
                        ]
                    }
                )
            links[owner, 2].send({"stop": True})
        service = asyncio.create_task(
            asyncio.to_thread(
                serve_stage,
                HeldStage(2, "cpu"),
                topology,
                links,
                batch_size=8,
                on_event=events.append,
            )
        )
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            await asyncio.sleep(0.1)
            # The first forward uses only half a packet. Neither input edge
            # may allocate another packet just because its first left the inbox.
            self.assertEqual(received, [1, 1])
        finally:
            resume.set()
            await asyncio.wait_for(service, 3)
        self.assertEqual(sum(event["batch"] for event in events), 128)
        self.assertTrue(all(event["batch"] <= 8 for event in events))
        for owner in range(2):
            tickets = []
            while not (message := links[2, owner].recv()).get("stop"):
                for item in message["items"]:
                    tickets.append(item["ticket"])
                    torch.testing.assert_close(item["hidden"], torch.full((1, 2), 4.0))
            self.assertEqual(tickets, list(range(64)))

    async def test_idle_sender_failure_notifies_owner_without_a_request(self):
        from monowire.interfaces import PipelineFailure

        class BrokenChannel(MemoryChannel):
            def heartbeat(self):
                raise OSError("idle send failed")

        outgoing, returning = BrokenChannel(), MemoryChannel()
        executor = PipelineExecutor(
            RoutedStage(0, "cpu"),
            PipelineTopology((1, 1)),
            {(0, 1): outgoing, (1, 0): returning},
        )
        with patch("monowire.channels.HEARTBEAT_INTERVAL", 0.01):
            failure = executor.start()
            error = await asyncio.wait_for(asyncio.shield(failure), 1)
            self.assertIsInstance(error, PipelineFailure)
            self.assertIn("idle send failed", str(error))
            returning.send({"stop": True})
            with self.assertRaisesRegex(RuntimeError, "sender failed"):
                await asyncio.wait_for(executor.close(), 1)
        self.assertFalse(executor.sender.thread.is_alive())

    async def test_malformed_idle_control_message_fails_service_and_drains_stop(self):
        from monowire.interfaces import PipelineFailure

        outgoing, returning = MemoryChannel(), MemoryChannel()
        executor = PipelineExecutor(
            RoutedStage(0, "cpu"),
            PipelineTopology((1, 1)),
            {(0, 1): outgoing, (1, 0): returning},
        )
        failure = executor.start()
        returning.send(None)
        error = await asyncio.wait_for(asyncio.shield(failure), 1)
        self.assertIsInstance(error, PipelineFailure)
        self.assertIsInstance(error.__cause__, AttributeError)
        closing = asyncio.create_task(executor.close())
        self.assertTrue((await asyncio.to_thread(outgoing.recv))["stop"])
        returning.send({"stop": True})
        with self.assertRaises(PipelineFailure):
            await asyncio.wait_for(closing, 1)
        self.assertTrue(executor.receiver.done())

    async def test_unexpected_idle_stop_is_a_service_failure(self):
        from monowire.interfaces import PipelineFailure

        outgoing, returning = MemoryChannel(), MemoryChannel()
        executor = PipelineExecutor(
            RoutedStage(0, "cpu"),
            PipelineTopology((1, 1)),
            {(0, 1): outgoing, (1, 0): returning},
        )
        failure = executor.start()
        returning.send({"stop": True})
        self.assertIsInstance(await asyncio.wait_for(asyncio.shield(failure), 1), PipelineFailure)
        with self.assertRaisesRegex(PipelineFailure, "Suffix stopped"):
            await executor.close()
        self.assertTrue(outgoing.recv()["stop"])

    async def test_response_split_preserves_independent_success_and_failure(self):
        topology = PipelineTopology((1, 1))
        outgoing, returning = MemoryChannel(), MemoryChannel()
        executor = PipelineExecutor(
            RoutedStage(0, "cpu"), topology, {(0, 1): outgoing, (1, 0): returning}
        )
        requests = [PreparedStep(str(i), 0, False, "toy", torch.ones(1, 2)) for i in range(2)]
        first, second = executor.submit(requests)
        packet = await asyncio.to_thread(outgoing.recv)
        a, b = packet["items"]
        returning.send({"items": [{"ticket": b["ticket"], "hidden": torch.tensor(7)}]})
        self.assertEqual((await asyncio.wait_for(asyncio.shield(second), 1)).item(), 7)
        self.assertFalse(first.done())
        returning.send({"items": [{"ticket": a["ticket"], "error": "only first failed"}]})
        with self.assertRaisesRegex(RuntimeError, "only first failed"):
            await first
        closing = asyncio.create_task(executor.close())
        self.assertTrue((await asyncio.to_thread(outgoing.recv))["stop"])
        returning.send({"stop": True})
        await asyncio.wait_for(closing, 1)

    async def test_single_stage_uses_same_executor_without_channels(self):
        executor = PipelineExecutor(RoutedStage(0, "cpu"), PipelineTopology((50,)), {})
        prepared = PreparedStep("one", 0, False, "toy", torch.tensor(2))
        self.assertEqual((await executor.submit([prepared])[0]).item(), 3)
        await executor.close()
        await executor.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            executor.submit([prepared])

    async def test_service_failure_reaches_owner_and_drains_stop_protocol(self):
        outgoing, returning = MemoryChannel(), MemoryChannel()
        topology = PipelineTopology((1, 1))
        links = {(0, 1): outgoing, (1, 0): returning}
        stage = RoutedStage(1, "cpu")
        # A malformed internal envelope is a service failure, not a model error.
        service = asyncio.create_task(
            asyncio.to_thread(serve_stage, stage, topology, links, batch_size=8)
        )
        executor = PipelineExecutor(RoutedStage(0, "cpu"), topology, links)
        executor.start()
        outgoing.send({"unexpected": True})
        for _ in range(100):
            if executor.failure is not None:
                break
            await asyncio.sleep(0.01)
        self.assertIsNotNone(executor.failure)
        with self.assertRaisesRegex(RuntimeError, "communication failed"):
            await asyncio.wait_for(executor.close(), 2)
        with self.assertRaises(KeyError):
            await asyncio.wait_for(service, 2)

    async def test_packing_error_returns_to_requests_and_service_continues(self):
        class FailingStage(RoutedStage):
            def combine_requests(self, items):
                if any(bool((item["hidden"] < 0).any()) for item in items):
                    raise ValueError("cannot pack")
                return super().combine_requests(items)

        topology = PipelineTopology((1, 1))
        links = {(0, 1): MemoryChannel(), (1, 0): MemoryChannel()}
        service = asyncio.create_task(
            asyncio.to_thread(serve_stage, FailingStage(1, "cpu"), topology, links, batch_size=8)
        )
        executor = PipelineExecutor(RoutedStage(0, "cpu"), topology, links)
        bad = PreparedStep("bad", 0, False, "toy", torch.full((1, 2), -10.0))
        good = PreparedStep("good", 0, False, "toy", torch.full((1, 2), 10.0))
        with self.assertRaisesRegex(RuntimeError, "cannot pack"):
            await asyncio.wait_for(asyncio.shield(executor.submit([bad])[0]), 2)
        torch.testing.assert_close(await executor.submit([good])[0], good.payload + 3)
        await asyncio.wait_for(executor.close(), 2)
        await asyncio.wait_for(service, 2)

    async def test_sender_failure_is_joined_and_reported(self):
        from monowire.channels import BufferedSender

        class BrokenChannel:
            def send(self, *args):
                raise OSError("link failed")

        sender = BufferedSender(BrokenChannel(), 1)
        sender.submit({"items": []})
        await asyncio.to_thread(sender.thread.join, 1)
        with self.assertRaisesRegex(RuntimeError, "sender failed"):
            sender.close()
        self.assertFalse(sender.thread.is_alive())

    async def test_malformed_response_fails_the_ticket_and_still_drains_stop(self):
        from monowire.interfaces import PipelineFailure

        topology = PipelineTopology((1, 1))
        outgoing, returning = MemoryChannel(), MemoryChannel()
        executor = PipelineExecutor(
            RoutedStage(0, "cpu"), topology, {(0, 1): outgoing, (1, 0): returning}
        )
        future = executor.submit([PreparedStep("bad", 0, False, "toy", torch.ones(1, 2))])[0]
        item = (await asyncio.to_thread(outgoing.recv))["items"][0]
        returning.send({"items": [{"ticket": item["ticket"]}]})  # Missing output payload.
        with self.assertRaises(PipelineFailure):
            await asyncio.wait_for(asyncio.shield(future), 1)
        closing = asyncio.create_task(executor.close())
        self.assertTrue((await asyncio.to_thread(outgoing.recv))["stop"])
        returning.send({"stop": True})
        with self.assertRaisesRegex(RuntimeError, "communication failed"):
            await asyncio.wait_for(closing, 1)
