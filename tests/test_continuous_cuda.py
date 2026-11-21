"""Real NCCL actor pipeline, overlapping submissions, bypass and clean shutdown."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

from monowire.continuous import PreparedStep
from monowire.models.h3.adapter import H3StageAdapter
from monowire.transport import PipelineExecutor, serve_stage


class Block(torch.nn.Module):
    def __init__(self, amount):
        super().__init__()
        self.amount = amount

    def forward(self, hidden, **kwargs):
        return hidden + self.amount


def worker(rank, size, store):
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_pp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=Path(store).as_uri(),
        backend="nccl",
    )
    initialize_model_parallel(pipeline_parallel_size=size, backend="nccl")
    transformer = torch.nn.Module()
    transformer.pp_group = get_pp_group()
    transformer.start_layer, transformer.end_layer = rank, rank + 1
    transformer.adaln_cache = None
    transformer.blocks = torch.nn.ModuleList([Block(i + 1) for i in range(size)])

    def prepared(key, value, hit=False):
        hidden = torch.full((3, 2), float(value), device=f"cuda:{rank}")
        args = dict(
            num_requests=1,
            packed_total=3,
            max_seqlen=3,
            cu_seqlens=torch.tensor([0, 3], dtype=torch.int32, device=hidden.device),
            t_emb=hidden.new_zeros(1, 4),
            combined_indices=torch.zeros(3, dtype=torch.long, device=hidden.device),
            rope_table=hidden.new_zeros(3, 2),
        )
        return PreparedStep(
            key,
            0,
            hit,
            "same",
            dict(
                hidden=hidden,
                arguments=args,
                transformer=transformer,
                first_done=False,
                entry=SimpleNamespace(residual=torch.ones_like(hidden) * 10),
            ),
        )

    async def run():
        executor = PipelineExecutor(H3StageAdapter(transformer))
        tasks = [
            asyncio.create_task(executor.execute([prepared(f"{i}a", i), prepared(f"{i}b", i + 1)]))
            for i in range(4)
        ]
        hit = await executor.execute([prepared("hit", 100, True)])
        torch.testing.assert_close(hit[0], torch.full_like(hit[0], 110))
        for i, task in enumerate(tasks):
            outputs = await asyncio.wait_for(task, 60)
            for j, out in enumerate(outputs):
                torch.testing.assert_close(
                    out, torch.full_like(out, i + j + sum(range(1, size + 1)))
                )
        await asyncio.wait_for(executor.close(), 60)

    try:
        with torch.inference_mode():
            if rank == 0:
                asyncio.run(run())
            else:
                serve_stage(H3StageAdapter(transformer))
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@unittest.skipUnless(
    os.environ.get("MONOWIRE_TEST_CUDA_FOUR") == "1", "Requires four allocated GPUs"
)
class ContinuousCUDATests(unittest.TestCase):
    def test_four_rank_actor_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(worker, args=(4, str(Path(directory, "store"))), nprocs=4)
