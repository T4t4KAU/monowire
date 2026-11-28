"""Real Gloo ready-driven decoding and local-drain-before-load lifecycle."""

import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch.distributed as dist
import torch.multiprocessing as mp

from monowire.continuous_engine import ContinuousOutput
from monowire.models.h3.phased import PhasedH3Engine


def handoff_worker(rank, store, decoded, allow_tail, mode):
    dist.init_process_group(
        "gloo",
        init_method=Path(store).as_uri(),
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=25),
    )
    try:
        engine = PhasedH3Engine(
            "FL2VA",
            pp_size=3,
            config=None,
            cache=None,
            topology=None,
            encoder_stage_layers=None,
            trust_remote_code=False,
            overlap_dit_decoder=True,
            phase_parallel={"decoder": {"mode": "dp", "ranks": [1, 2]}},
        )
        engine.rank = rank
        engine.cpu_group = dist.group.WORLD
        engine.worker = SimpleNamespace(device="cpu")
        released = False
        timeline = []
        engine._mark = lambda name, **values: timeline.append(name)

        def operation(publish):
            nonlocal released
            if rank == 0:
                for index in (2, 0, 1):
                    publish(ContinuousOutput(str(index), index))
                if mode == "rolling":
                    assert decoded.wait(10), "Decode waited for ingress/global drain"
            if rank == 2 and mode == "rolling":
                assert allow_tail.wait(10), "Fast replica did not decode before tail drained"
            released = True

        def load():
            assert released, "Decoder loaded before local DiT owners were released"
            if mode == "load_failure":
                raise ValueError("decoder load failed")
            return object()

        def decode(values, sampling, *, decoder):
            if mode == "decode_failure":
                raise ValueError("decoder compute failed")
            value = values[0]
            value.output = (value.output * 10, rank)
            if mode == "rolling":
                assert rank == 1, "Late replica received a job before it was ready"
                decoded.set()
                # All queued requests can be consumed by the first ready replica.
                if value.request_id == "1":
                    allow_tail.set()
            return values

        engine._load_decoder = load
        engine._decode = decode
        with (
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.empty_cache"),
            patch("torch.cuda.memory_allocated", return_value=0),
        ):
            if mode in {"load_failure", "decode_failure"}:
                try:
                    engine._rolling_decode(operation, None)
                except RuntimeError as exc:
                    assert "Rolling stage handoff failed" in str(exc)
                    assert "decoder" in str(exc)
                else:
                    raise AssertionError("Decoder failure was not propagated to every rank")
            else:
                outputs = engine._rolling_decode(operation, None)
                if rank == 0:
                    assert [value.request_id for value in outputs] == ["0", "1", "2"]
                    assert [value.output[0] for value in outputs] == [0, 10, 20]
                else:
                    assert outputs == []
                if rank in (1, 2):
                    assert timeline.index("dit_local_released") < timeline.index(
                        "decoder_load_start"
                    )
    finally:
        dist.destroy_process_group()


def long_decode_worker(rank, store, ready):
    from monowire.models.h3 import decoder_handoff as module

    dist.init_process_group(
        "gloo",
        init_method=Path(store).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    # Each healthy phase lasts longer than a complete transport timeout.
    with patch.object(module, "CHANNEL_TIMEOUT", timedelta(seconds=2)):
        handoff = module.DecoderHandoff((1,))
        try:
            if rank == 0:
                assert ready.wait(10)
                time.sleep(3)  # Ready replica waits for the first completed DiT request.
                handoff.publish(ContinuousOutput("0", 7))
                handoff.finish()
                outputs = handoff.collect()
                assert not handoff.errors, handoff.errors
                assert len(outputs) == 1 and outputs[0].output == 8
            else:

                def load():
                    time.sleep(3)
                    ready.set()
                    return object()

                def decode(model, value):
                    time.sleep(3)
                    value.output += 1
                    return value

                handoff.work(load, decode)
                assert not handoff.errors, handoff.errors
                assert not handoff.heartbeat.is_alive()
            dist.barrier()
        finally:
            handoff.close()
            dist.destroy_process_group()


def completion_worker(rank, store, mode):
    from monowire.models.h3 import decoder_handoff, phase_completion

    dist.init_process_group(
        "gloo",
        init_method=Path(store).as_uri(),
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=2),
    )
    placement = {
        "single": {"mode": "single", "ranks": [0]},
        "subset": {"mode": "single", "ranks": [1]},
        "uneven": {"mode": "dp", "ranks": "all"},
        "late_error": {"mode": "dp", "ranks": "all"},
    }[mode]
    engine = PhasedH3Engine(
        "FL2VA",
        pp_size=3,
        config=None,
        cache=None,
        topology=None,
        encoder_stage_layers=None,
        trust_remote_code=False,
        overlap_dit_decoder=True,
        phase_parallel={"decoder": placement},
    )
    engine.rank, engine.cpu_group = rank, dist.group.WORLD
    engine.worker = SimpleNamespace(device="cpu")

    def operation(publish):
        if rank == 0:
            publish(ContinuousOutput("0", 7))

    def load():
        if mode in {"uneven", "late_error"} and rank == 2:
            # Other replicas can consume the only job and finish before this
            # healthy worker loads; late failure must still reach every rank.
            time.sleep(4)
            if mode == "late_error":
                raise ValueError("late replica load failure")
        return object()

    def decode(values, sampling, *, decoder):
        if mode in {"single", "subset"}:
            time.sleep(4)
        values[0].output += 1
        return values

    engine._load_decoder, engine._decode = load, decode
    try:
        with (
            patch.object(phase_completion, "CHANNEL_TIMEOUT", timedelta(seconds=2)),
            patch.object(decoder_handoff, "CHANNEL_TIMEOUT", timedelta(seconds=2)),
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.empty_cache"),
            patch("torch.cuda.memory_allocated", return_value=0),
        ):
            if mode == "late_error":
                try:
                    engine._rolling_decode(operation, None)
                except RuntimeError as exc:
                    assert "late replica load failure" in str(exc), str(exc)
                else:
                    raise AssertionError("Late replica failure was lost")
            else:
                outputs = engine._rolling_decode(operation, None)
                if rank == 0:
                    assert len(outputs) == 1 and outputs[0].output == 8
                else:
                    assert outputs == []
        dist.barrier()
    finally:
        dist.destroy_process_group()


class DecoderHandoffTests(unittest.TestCase):
    def run_completion_case(self, mode):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(
                completion_worker, args=(str(Path(folder, "store")), mode), nprocs=3, join=True
            )

    def test_non_decoder_ranks_wait_for_long_rank_zero_decode(self):
        self.run_completion_case("single")

    def test_non_decoder_ranks_wait_for_long_nonzero_decode(self):
        self.run_completion_case("subset")

    def test_finished_replicas_wait_for_late_empty_replica(self):
        self.run_completion_case("uneven")

    def test_late_replica_failure_reaches_early_finishers(self):
        self.run_completion_case("late_error")

    def test_heartbeats_cover_loading_idle_and_long_decode(self):
        context = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(
                long_decode_worker,
                args=(str(Path(folder, "store")), context.Event()),
                nprocs=2,
                join=True,
            )

    def run_case(self, mode):
        context = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(
                handoff_worker,
                args=(str(Path(folder, "store")), context.Event(), context.Event(), mode),
                nprocs=3,
                join=True,
            )

    def test_early_decoder_runs_before_tail_drain_and_results_restore_input_order(self):
        self.run_case("rolling")

    def test_load_failure_propagates_without_waiting_for_unavailable_replica(self):
        self.run_case("load_failure")

    def test_cli_combined_flag_enables_both_handoffs(self):
        from monowire.cli import main

        with (
            patch(
                "sys.argv",
                [
                    "monowire",
                    "--model",
                    "unused",
                    "--output",
                    "unused",
                    "--prompt",
                    "hi",
                    "--phased",
                    "--pp",
                    "3",
                    "--overlap-stages",
                    "--phase-parallel-config",
                    '{"decoder":{"mode":"dp","ranks":"all"}}',
                ],
            ),
            patch(
                "monowire.models.h3.phased.create_phased_engine",
                side_effect=RuntimeError("captured"),
            ) as create,
        ):
            with self.assertRaisesRegex(RuntimeError, "captured"):
                main()
        self.assertTrue(create.call_args.kwargs["overlap_encoder_dit"])
        self.assertTrue(create.call_args.kwargs["overlap_dit_decoder"])
        self.assertEqual(create.call_args.kwargs["phase_parallel"].decoder.ranks, (0, 1, 2))

    def test_decode_failure_propagates_and_releases_workers(self):
        self.run_case("decode_failure")
