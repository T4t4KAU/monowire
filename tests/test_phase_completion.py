"""Completion control failures stay bounded and cannot turn into success."""

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch.distributed as dist
import torch.multiprocessing as mp

from monowire.models.h3 import phase_completion


def broken_control_worker(rank, store):
    dist.init_process_group(
        "gloo",
        init_method=Path(store).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    original_send = phase_completion.PhaseCompletion._send

    def send(value, peer, group):
        if rank == 1:
            raise RuntimeError("injected completion link failure")
        original_send(value, peer, group)

    try:
        with (
            patch.object(phase_completion, "CHANNEL_TIMEOUT", timedelta(seconds=1)),
            patch.object(phase_completion.PhaseCompletion, "_send", staticmethod(send)),
        ):
            completion = phase_completion.PhaseCompletion()
            try:
                errors = completion.finish()
                assert any("completion transport failed" in error for error in errors), errors
                assert all(not thread.is_alive() for thread in completion.threads)
            finally:
                completion.close()
        dist.barrier()
    finally:
        dist.destroy_process_group()


class PhaseCompletionTests(unittest.TestCase):
    def test_broken_control_link_reports_failure_on_both_ranks(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(broken_control_worker, args=(str(Path(folder, "store")),), nprocs=2, join=True)

    def test_single_worker_completes_and_preserves_local_error(self):
        with tempfile.TemporaryDirectory() as folder:
            dist.init_process_group(
                "gloo", init_method=Path(folder, "store").as_uri(), rank=0, world_size=1
            )
            try:
                completion = phase_completion.PhaseCompletion()
                try:
                    self.assertEqual(completion.finish("local failure"), ["local failure"])
                    self.assertFalse(completion.threads)
                finally:
                    completion.close()
            finally:
                dist.destroy_process_group()
