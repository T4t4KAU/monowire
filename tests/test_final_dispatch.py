"""Real scheduler, recurrent transport and decoder handoff on three Gloo ranks."""

import asyncio
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_continuous import Adapter
from test_transport import RoutedStage

from monowire.channels import close_channels, create_channels
from monowire.continuous import ContinuousConfig, PreparedStep
from monowire.continuous_engine import ContinuousEngine
from monowire.models.h3.phased import PhasedH3Engine
from monowire.topology import PipelineTopology
from monowire.transport import PipelineExecutor, serve_stage


class RecurrentAdapter(Adapter):
    async def admit(self, key, value):
        return await super().admit(key, value[1])

    def prepare(self, state):
        return PreparedStep(
            state["key"], state["step"], False, "toy", torch.full((1, 1), float(state["value"]))
        )

    def commit(self, state, prepared, output):
        super().commit(state, prepared, output.item())

    def is_final_step(self, state, prepared):
        return prepared.step == state["total"] - 1


def recurrent_worker(rank, store, decoded):
    dist.init_process_group(
        "gloo",
        init_method=Path(store).as_uri(),
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=30),
    )
    topology = PipelineTopology((1, 1, 1), buffer_size=2)
    channels = create_channels(topology, "cpu")
    count = 0

    class Stage(RoutedStage):
        def compute(self, packet):
            nonlocal count
            count += 1
            if rank == 2 and count == 12:
                assert decoded.wait(10), "Decoder did not start before final DiT compute finished"
            return super().compute(packet)

    stage = Stage(rank, "cpu")
    runtime = SimpleNamespace(
        rank=rank,
        ingress_replicas=1,
        context=torch.inference_mode,
        create_adapter=RecurrentAdapter,
        create_executor=lambda: PipelineExecutor(stage, topology, channels),
        serve=lambda: serve_stage(stage, topology, channels, batch_size=1),
    )
    engine = PhasedH3Engine(
        "FL2VA",
        pp_size=3,
        config=None,
        cache=None,
        topology=None,
        encoder_stage_layers=None,
        trust_remote_code=False,
        overlap_dit_decoder=True,
        phase_parallel={"decoder": {"mode": "dp", "ranks": "all"}},
    )
    engine.rank, engine.cpu_group = rank, dist.group.WORLD
    engine.worker = SimpleNamespace(device="cpu")
    engine._load_decoder = lambda: object()

    def decode(values, sampling, *, decoder):
        if rank == 1:
            # Only this replica can retire while the tail's final compute is blocked.
            decoded.set()
        return values

    engine._decode = decode

    def operation(publish):
        service = ContinuousEngine(runtime, ContinuousConfig(2, 1), retire_on_final_dispatch=True)

        async def source():
            for key, steps in enumerate((1, 3, 8)):
                yield str(key), None, steps

        async def collect():
            async for output in service.stream(source()):
                publish(output)

        try:
            asyncio.run(collect())
        finally:
            close_channels(channels)

    try:
        with (
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.empty_cache"),
            patch("torch.cuda.memory_allocated", return_value=0),
        ):
            outputs = engine._rolling_decode(operation, None)
        if rank == 0:
            assert [(v.request_id, v.output, v.error) for v in outputs] == [
                ("0", 6.0, None),
                ("1", 18.0, None),
                ("2", 48.0, None),
            ]
        else:
            assert count == 12
        assert decoded.is_set()
    finally:
        dist.destroy_process_group()


class FinalDispatchTests(unittest.TestCase):
    def test_recurrent_pending_requests_drain_before_real_early_retirement(self):
        context = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(
                recurrent_worker,
                args=(str(Path(folder, "store")), context.Event()),
                nprocs=3,
                join=True,
            )
