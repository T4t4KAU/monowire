"""Exercise generic actors with a model that has no H3 fields or dependencies."""

import asyncio
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from monowire.continuous import PreparedStep
from monowire.transport import PipelineExecutor, serve_stage


class Group:
    device = "cpu"

    def __init__(self, rank, size):
        self.rank_in_group, self.world_size = rank, size

    def send_tensor_dict(self, packet, dst):
        dist.send_object_list([packet], dst=dst)

    def recv_tensor_dict(self, src):
        values = [None]
        dist.recv_object_list(values, src=src)
        return values[0]


class ToyStage:
    def __init__(self, group):
        self.group = group

    def validate(self, prepared):
        pass

    def reuse(self, prepared):
        return [p.payload + 100 for p in prepared]

    def pack(self, prepared):
        packet = {"hidden": torch.stack([p.payload for p in prepared])}
        return self.compute(packet), None

    def compute(self, packet):
        if self.group.rank_in_group == 1 and bool((packet["hidden"] < 0).any()):
            raise ValueError("toy stage failure")
        packet["hidden"] += self.group.rank_in_group + 1
        return packet

    def unpack(self, output, context):
        return list(output.unbind())


def worker(rank, size, path):
    dist.init_process_group("gloo", init_method=Path(path).as_uri(), rank=rank, world_size=size)
    stage = ToyStage(Group(rank, size))

    async def run():
        executor = PipelineExecutor(stage)

        def request(value, hit=False):
            return PreparedStep(str(value), 0, hit, "toy", torch.tensor(float(value)))

        tasks = [asyncio.create_task(executor.execute([request(i)])) for i in range(4)]
        hit = await executor.execute([request(5, True)])
        assert hit[0].item() == 105
        for i, task in enumerate(tasks):
            result = await asyncio.wait_for(task, 20)
            assert result[0].item() == i + sum(range(1, size + 1))
        try:
            await executor.execute([request(-100)])
        except RuntimeError as exc:
            assert "toy stage failure" in str(exc)
        else:
            raise AssertionError("Actor failure did not reach ingress")
        result = await executor.execute([request(20)])
        assert result[0].item() == 26
        await executor.close()
        await executor.close()

    try:
        if rank == 0:
            asyncio.run(run())
        else:
            serve_stage(stage)
    finally:
        dist.destroy_process_group()


class TransportTests(unittest.TestCase):
    def test_sent_activations_are_released_before_response(self):
        async def run():
            sent = asyncio.Event()
            response = asyncio.Event()
            references = []

            class Stage(ToyStage):
                def pack(self, prepared):
                    packet, context = super().pack(prepared)
                    references.append(weakref.ref(packet["hidden"]))
                    return packet, context

            class Executor(PipelineExecutor):
                async def _publish(self, packet):
                    sent.set()

                async def _receive(self):
                    await response.wait()
                    self.waiting.pop(0).set_result(torch.tensor([3.0]))

            executor = Executor(Stage(SimpleNamespace(rank_in_group=0, world_size=2)))
            step = PreparedStep("one", 0, False, "toy", torch.tensor(2.0))
            task = asyncio.create_task(executor.execute([step]))
            await sent.wait()
            self.assertIsNone(references[0]())
            response.set()
            result = await asyncio.wait_for(task, 1)
            self.assertEqual(result[0].item(), 3)
            await executor.receiver

        asyncio.run(run())

    def test_model_independent_actors_error_recovery_and_shutdown(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(worker, args=(3, str(Path(folder, "store"))), nprocs=3)

    def test_single_rank(self):
        async def run():
            stage = ToyStage(SimpleNamespace(rank_in_group=0, world_size=1))
            executor = PipelineExecutor(stage)
            step = PreparedStep("one", 0, False, "toy", torch.tensor(2.0))
            result = await executor.execute([step])
            self.assertEqual(result[0].item(), 3)
            await executor.close()
            await executor.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                await executor.execute([step])

        asyncio.run(run())

    def test_receiver_failure_rejects_later_requests(self):
        class BrokenGroup:
            rank_in_group = 0
            world_size = 2
            device = "cpu"

            def send_tensor_dict(self, packet, dst):
                pass

            def recv_tensor_dict(self, src):
                raise OSError("connection lost")

        async def run():
            executor = PipelineExecutor(ToyStage(BrokenGroup()))
            step = PreparedStep("one", 0, False, "toy", torch.tensor(2.0))
            for _ in range(2):
                with self.assertRaisesRegex((OSError, RuntimeError), "connection lost"):
                    await asyncio.wait_for(executor.execute([step]), 1)
            with self.assertRaisesRegex(RuntimeError, "connection lost"):
                await asyncio.wait_for(executor.close(), 1)

        asyncio.run(run())
