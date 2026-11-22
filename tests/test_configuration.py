"""Batch limits reach every deployment independently of layer placement."""

import unittest
from unittest.mock import patch

from monowire.cli import main
from monowire.continuous import ContinuousConfig
from monowire.models.h3.runtime import H3Runtime
from monowire.topology import PipelineTopology


class ConfigurationTests(unittest.TestCase):
    def test_cli_keeps_distinct_ingress_and_suffix_limits_for_both_layouts(self):
        arguments = [
            "monowire",
            "--model",
            "unused",
            "--prompt",
            "test",
            "--output",
            "unused.pt",
            "--batch-size",
            "32",
            "--microbatch-size",
            "8",
            "--downstream-batch-size",
            "16",
        ]
        for layout in ([], ["--stage-layers", "6", "6", "6", "6", "6", "6", "7", "7"]):
            with (
                self.subTest(layout=layout),
                patch("sys.argv", arguments + layout),
                patch(
                    "monowire.models.h3.runtime.create_continuous_engine",
                    side_effect=RuntimeError("configuration captured"),
                ) as create,
                self.assertRaisesRegex(RuntimeError, "configuration captured"),
            ):
                main()
            self.assertEqual(create.call_args.kwargs["config"], ContinuousConfig(32, 8, 16))

    def test_runtime_uses_configured_suffix_limit(self):
        runtime = object.__new__(H3Runtime)
        runtime.config = ContinuousConfig(32, 8, 16)
        runtime.stage, runtime.topology, runtime.channels = object(), object(), {}
        runtime.on_stage_event = None
        with patch("monowire.models.h3.runtime.serve_stage") as serve:
            runtime.serve()
        self.assertEqual(serve.call_args.kwargs["batch_size"], 16)

    def test_suffix_limit_can_exceed_one_ingress_capacity(self):
        self.assertEqual(ContinuousConfig(8, 8, 16).downstream_batch_size, 16)
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ContinuousConfig(downstream_batch_size=value)

    def test_old_topology_batch_argument_cannot_become_buffer_capacity(self):
        with self.assertRaises(TypeError):
            PipelineTopology((6, 44), 1, 16)
        with self.assertRaises(TypeError):
            PipelineTopology((6, 44), downstream_batch_size=16)
