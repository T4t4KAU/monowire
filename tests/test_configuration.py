"""Batch limits reach every deployment independently of layer placement."""

import unittest
from unittest.mock import patch

from monowire.cli import main
from monowire.continuous import ContinuousConfig
from monowire.device_memory import DeviceMemoryLimits
from monowire.models.h3.runtime import H3Runtime
from monowire.topology import PipelineTopology


class ConfigurationTests(unittest.TestCase):
    def test_receive_prefetch_is_independent_of_storage_count(self):
        topology = PipelineTopology((6, 44), buffer_size=9)
        self.assertEqual(topology.receive_prefetch, 2)
        self.assertEqual(PipelineTopology((6, 44), receive_prefetch=4).receive_prefetch, 4)
        for value in (0, -1, 1.5, True, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "prefetch"):
                PipelineTopology((6, 44), receive_prefetch=value)

    def test_count_remains_default_and_explicit_option(self):
        for config in (ContinuousConfig(), ContinuousConfig(queue_policy="count")):
            self.assertEqual(config.resolved_queue_policy, "count")
            self.assertIsNone(config.memory_limits)
            self.assertTrue(config.async_transport)
            self.assertEqual(config.max_active_requests, 32)

    def test_explicit_buffer_policies_and_legacy_selection(self):
        for policy in ("ring", "pool"):
            config = ContinuousConfig(32, 1, 1, queue_policy=policy)
            self.assertEqual(config.resolved_queue_policy, policy)
            self.assertEqual(config.memory_limits.buffer_strategy, policy)
            self.assertEqual(config.max_active_requests, 32)
            limits = DeviceMemoryLimits(123, buffer_strategy=policy)
            for kwargs in ({}, {"queue_policy": policy}):
                custom = ContinuousConfig(32, 1, 1, memory_limits=limits, **kwargs)
                self.assertIs(custom.memory_limits, limits)
                self.assertEqual(custom.resolved_queue_policy, policy)

    def test_policy_conflicts_are_not_silently_ignored(self):
        for policy in ("count", "pool", "bytes"):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "conflicts"):
                ContinuousConfig(
                    32, 1, 1, queue_policy=policy, memory_limits=DeviceMemoryLimits(123)
                )
        with self.assertRaisesRegex(ValueError, "queue_policy"):
            ContinuousConfig(queue_policy="unknown")
        with self.assertRaisesRegex(ValueError, "Unlimited active"):
            ContinuousConfig(None, 1, 1, queue_policy="count")

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
