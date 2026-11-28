"""Independent phase configuration and real CPU subgroup handoffs."""

import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from monowire.models.h3.encoder_transport import run_encoder_pipeline
from monowire.models.h3.phase_dispatch import map_phase
from monowire.models.h3.phase_parallel import PhaseParallelConfig, StageParallelConfig
from monowire.models.h3.phased import PhasedH3Engine
from monowire.topology import PipelineTopology


def distributed_phases(rank, store):
    dist.init_process_group(
        "gloo",
        init_method=Path(store).as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    group = dist.new_group(ranks=[1, 3], backend="gloo")
    try:
        # DP excludes ingress; uneven and empty shards must preserve order.
        for values in (list(range(7)), [4], []):

            def operate(items):
                assert rank in (1, 3)
                return [(value * 10, rank) for value in items]

            result = map_phase(values if rank == 0 else None, operate, (1, 3), dist.group.WORLD)
            if rank == 0:
                assert result == [(v * 10, (1, 3)[i % 2]) for i, v in enumerate(values)]
            else:
                assert result is None

        def fail(items):
            if rank == 3:
                raise ValueError("replica failure")
            return items

        try:
            map_phase([1, 2] if rank == 0 else None, fail, (1, 3), dist.group.WORLD)
        except RuntimeError as exc:
            assert "rank 3: ValueError: replica failure" in str(exc)
        else:
            raise AssertionError("Replica failure was not propagated")

        engine = PhasedH3Engine(
            "FL2VA",
            pp_size=4,
            config=None,
            cache=None,
            topology=None,
            encoder_stage_layers=None,
            trust_remote_code=False,
            phase_parallel={"encoder": {"mode": "pp", "ranks": [1, 3]}},
        )
        engine.rank, engine.cpu_group = rank, dist.group.WORLD
        lengths = [2, 5, 1]

        class Stage:
            device_target = "cpu"
            text_model = SimpleNamespace(last_positions=None)

            def first(self, ids, kwargs):
                assert rank == 1
                return ids[:, None].repeat(1, 8).to(torch.bfloat16), torch.zeros(
                    (3, 1, len(ids)), dtype=torch.long
                )

            def continuation(self, hidden, positions):
                assert rank == 3
                return hidden + 10

        engine._encode_text_local = lambda rows, lens: run_encoder_pipeline(
            Stage(), rows, lens, group, group, hidden_dim=8
        )
        result = engine._encode_text(
            [(torch.arange(n), {}) for n in lengths] if rank == 0 else None, lengths
        )
        if rank == 0:
            for value, n in zip(result, lengths):
                torch.testing.assert_close(
                    value, (torch.arange(n) + 10)[:, None].repeat(1, 8).to(torch.bfloat16)
                )
        else:
            assert result is None

        # Encoder request DP uses singleton communicators, never the DiT group.
        singleton_groups = []
        for member in (1, 3):
            singleton = dist.new_group(ranks=[member], backend="gloo")
            if rank == member:
                singleton_groups.append(singleton)
                local_group = singleton
        engine.phase_parallel = PhaseParallelConfig.resolve(
            4,
            {
                "encoder": {"mode": "dp", "ranks": [1, 3]},
            },
        )

        class ReplicaStage(Stage):
            def first(self, ids, kwargs):
                return ids[:, None].repeat(1, 8).to(torch.bfloat16), torch.zeros(
                    (3, 1, len(ids)), dtype=torch.long
                )

        engine._encode_text_local = lambda rows, lens: run_encoder_pipeline(
            ReplicaStage(), rows, lens, local_group, local_group, hidden_dim=8
        )
        result = engine._encode_text(
            [(torch.arange(n), {}) for n in lengths] if rank == 0 else None, lengths
        )
        if rank == 0:
            for value, n in zip(result, lengths):
                torch.testing.assert_close(
                    value, torch.arange(n)[:, None].repeat(1, 8).to(torch.bfloat16)
                )
        else:
            assert result is None
        for singleton in singleton_groups:
            dist.destroy_process_group(singleton)

        # All-card decoding: every worker owns complete requests, including rank 0.
        engine.phase_parallel = PhaseParallelConfig.resolve(
            4,
            {
                "decoder": {"mode": "dp", "ranks": "all"},
            },
        )
        engine._decode = lambda rows, sampling: [(row, rank) for row in rows]
        for count in (9, 2, 0):
            result = engine._decode_phase(list(range(count)) if rank == 0 else None, None)
            assert result == ([(i, i % 4) for i in range(count)] if rank == 0 else None)

        # Exercise actual codec dispatch entry points on nonzero ranks.
        engine.phase_parallel = PhaseParallelConfig.resolve(
            4,
            {
                "decoder": {"mode": "single", "ranks": [3]},
                "media_encoder": {"mode": "dp", "ranks": [1, 3]},
            },
        )
        engine._decode = lambda rows, sampling: [(row, rank) for row in rows]
        result = engine._decode_phase([8, 9] if rank == 0 else None, None)
        assert result == ([(8, 3), (9, 3)] if rank == 0 else None)
        engine._encode_media = lambda rows: [row + 1 for row in rows]
        engine._conditioning_payload = lambda h, t, m: (h, t, m)
        result = engine._media_phase(
            [10, 20] if rank == 0 else None,
            [1, 2] if rank == 0 else None,
            [3, 4] if rank == 0 else None,
        )
        assert result == ([(1, 3, 11), (2, 4, 21)] if rank == 0 else None)
    finally:
        if rank in (1, 3):
            dist.destroy_process_group(group)
        dist.destroy_process_group()


def slow_phases(rank, store):
    from monowire.models.h3 import phase_completion

    dist.init_process_group(
        "gloo",
        init_method=Path(store).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=2),
    )
    try:
        with patch.object(phase_completion, "CHANNEL_TIMEOUT", timedelta(seconds=2)):

            def operate(items):
                time.sleep(4)
                return items

            # The empty replica must not enter result gathering during long work.
            result = map_phase([7] if rank == 0 else None, operate, (0, 1), dist.group.WORLD)
            assert result == ([7] if rank == 0 else None)

            engine = PhasedH3Engine(
                "FL2VA",
                pp_size=2,
                config=None,
                cache=None,
                topology=None,
                encoder_stage_layers=None,
                trust_remote_code=False,
                phase_parallel={"encoder": {"mode": "pp", "ranks": [1]}},
            )
            engine.rank, engine.cpu_group = rank, dist.group.WORLD
            for fail in (False, True):

                def encode(rows, lengths):
                    time.sleep(4)
                    if fail:
                        raise ValueError("late encoder failure")
                    return rows

                engine._encode_text_local = encode
                try:
                    result = engine._encode_text([7] if rank == 0 else None, [1])
                except RuntimeError as exc:
                    assert fail and "late encoder failure" in str(exc), str(exc)
                else:
                    assert not fail
                    assert result == ([7] if rank == 0 else None)
    finally:
        dist.destroy_process_group()


class PhaseParallelTests(unittest.TestCase):
    def test_long_replica_work_does_not_timeout_idle_ranks(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(slow_phases, args=(str(Path(folder, "store")),), nprocs=2, join=True)

    def test_legacy_defaults_and_explicit_partitions(self):
        value = PhaseParallelConfig.resolve(8)
        self.assertEqual(value.encoder.ranks, tuple(range(8)))
        self.assertEqual(sum(value.encoder.stage_layers), 50)
        self.assertEqual(value.decoder, StageParallelConfig())
        value = PhaseParallelConfig.resolve(
            4,
            {
                "encoder": {"mode": "pp", "ranks": [1, 3], "stage_layers": [25, 25]},
                "denoiser": {"mode": "pp", "ranks": [0, 1, 2, 3], "stage_layers": [13, 13, 12, 12]},
                "decoder": {"mode": "dp", "ranks": [0, 2]},
            },
        )
        self.assertEqual(value.encoder.stage_layers, (25, 25))
        self.assertEqual(value.denoiser.stage_layers, (13, 13, 12, 12))
        with self.assertRaisesRegex(ValueError, "identical"):
            value.validate_overlap()
        for mode in ("single", "dp"):
            value = PhaseParallelConfig.resolve(4, {"encoder": {"mode": mode, "ranks": [1]}})
            with self.assertRaisesRegex(ValueError, "requires Qwen"):
                value.validate_overlap()
        PhaseParallelConfig.resolve(4).validate_overlap()

    def test_all_card_decoder_resolves_launch_size(self):
        for world in (1, 4, 8):
            value = PhaseParallelConfig.resolve(
                world,
                {
                    "decoder": {"mode": "dp", "ranks": "all"},
                },
            )
            self.assertEqual(value.decoder.mode, "dp")
            self.assertEqual(value.decoder.ranks, tuple(range(world)))

    def test_invalid_configs_fail_before_runtime_imports(self):
        cases = [
            {"encoder": {"mode": "tp"}},
            {"encoder": {"mode": "single", "ranks": [0, 1]}},
            {"encoder": {"mode": "dp", "ranks": [4]}},
            {"encoder": {"mode": "pp", "ranks": [2, 1]}},
            {"encoder": {"mode": "pp", "ranks": [1, 1]}},
            {"encoder": {"mode": "pp", "ranks": [True]}},
            {"encoder": {"mode": "dp", "stage_layers": [50]}},
            {"decoder": {"mode": "pp"}},
            {"media_encoder": {"stage_layers": [50]}},
            {"denoiser": {"mode": "dp", "ranks": [0, 1, 2, 3]}},
            {"denoiser": {"mode": "pp", "ranks": [0, 1]}},
            {"denoiser": {"mode": "pp", "ranks": [0, 1, 2, 3], "stage_layers": [1, 1, 1, 1]}},
            {"typo": {}},
            {"encoder": {"typo": 1}},
            [],
        ]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                PhaseParallelConfig.resolve(4, value)
        with self.assertRaisesRegex(ValueError, "Conflicting encoder"):
            PhaseParallelConfig.resolve(
                2,
                {
                    "encoder": {"mode": "pp", "ranks": [0, 1], "stage_layers": [25, 25]},
                },
                encoder_stage_layers=[24, 26],
            )
        with self.assertRaisesRegex(ValueError, "Conflicting DiT"):
            PhaseParallelConfig.resolve(
                2,
                {
                    "denoiser": {"mode": "pp", "ranks": [0, 1], "stage_layers": [25, 25]},
                },
                topology=PipelineTopology((24, 26)),
            )

    def test_cli_rejects_overlap_for_non_pp_encoder(self):
        import contextlib
        import io

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
                    "4",
                    "--overlap-encoder-dit",
                    "--phase-parallel-config",
                    '{"encoder":{"mode":"dp","ranks":[0,1]}}',
                ],
            ),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as caught,
        ):
            main()
        self.assertEqual(caught.exception.code, 2)

    def test_cli_passes_independent_config_to_factory(self):
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
                    "4",
                    "--phase-parallel-config",
                    '{"encoder":{"mode":"pp","ranks":[1,3]},"decoder":{"mode":"dp","ranks":[0,2]}}',
                ],
            ),
            patch(
                "monowire.models.h3.phased.create_phased_engine",
                side_effect=RuntimeError("captured"),
            ) as create,
        ):
            with self.assertRaisesRegex(RuntimeError, "captured"):
                main()
        options = create.call_args.kwargs
        self.assertEqual(options["pp_size"], 4)
        self.assertEqual(options["phase_parallel"].encoder.ranks, (1, 3))
        self.assertEqual(options["phase_parallel"].decoder.ranks, (0, 2))
        self.assertEqual(options["phase_parallel"].encoder.stage_layers, (25, 25))

    def test_distributed_ordering_failures_and_nonzero_pp_subgroup(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(distributed_phases, args=(str(Path(folder, "store")),), nprocs=4, join=True)
