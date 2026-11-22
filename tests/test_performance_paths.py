"""Regression coverage for bounded scheduling and reusable wire data."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from monowire.channels import _decode, _encode
from monowire.tensor_payload import SharedTensor, TensorDictionary, TensorParts
from monowire.topology import PipelineTopology
from monowire.transport import _ReceivedPacket, _StageService


class WireDictionaryTests(unittest.TestCase):
    def test_snapshot_reuse_and_eviction_preserve_values(self):
        sender, receiver = TensorDictionary(32), TensorDictionary(32)
        original = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        a = SharedTensor.snapshot(original)
        original.add_(100)
        b = SharedTensor.snapshot(torch.ones(2, 2))
        c = SharedTensor.snapshot(torch.full((2, 2), 9.0))
        transferred = []
        for parts in ((a, a), (a, b), (b, a), (c,), (b,), (a,)):
            source = TensorParts(parts)
            tensors, destinations = [], []
            spec = _encode(source, tensors, sender)
            received = _decode(spec, destinations, "cpu", receiver)
            for tensor, (_, destination) in zip(tensors, destinations, strict=True):
                destination.copy_(tensor)
            torch.testing.assert_close(received.materialize(), source.materialize())
            transferred.append(len(tensors))
            self.assertLessEqual(sender.bytes, 32)
            self.assertEqual(list(sender.entries), list(receiver.entries))
        self.assertEqual(transferred, [1, 1, 0, 1, 1, 1])
        torch.testing.assert_close(a.value, torch.arange(4, dtype=torch.float32).reshape(2, 2))

    def test_oversized_values_are_retransmitted_and_unknown_references_fail(self):
        dictionary = TensorDictionary(1)
        shared = SharedTensor.snapshot(torch.ones(2))
        for _ in range(2):
            tensors = []
            self.assertEqual(_encode(shared, tensors, dictionary)[0], "shared")
            self.assertEqual(len(tensors), 1)
        self.assertFalse(dictionary.entries)
        with self.assertRaisesRegex(ValueError, "Missing shared"):
            _decode(("shared_ref", shared.key), [], "cpu", dictionary)


class ReadyBatchTests(unittest.TestCase):
    def test_ready_source_bypasses_transfer_then_oldest_regains_priority(self):
        class Event:
            complete = False

            def query(self):
                return self.complete

            def wait(self):
                if not self.complete:
                    raise AssertionError("Unready source blocked compute")

        pending_event = Event()
        stage = SimpleNamespace(
            group=SimpleNamespace(rank_in_group=2, device="cpu"),
            combine_requests=lambda items: {"ids": [i["ticket"] for i in items]},
        )
        with patch("monowire.transport.BufferedSender"):
            service = _StageService(
                stage,
                PipelineTopology((1, 1), 2),
                {(0, 2): None, (1, 2): None, (2, 0): None, (2, 1): None},
                8,
                None,
            )
        for owner, event in ((0, pending_event), (1, None)):
            service.pending.append(
                _ReceivedPacket(
                    owner,
                    {
                        "items": [
                            {
                                "owner": owner,
                                "ticket": owner,
                                "hidden": torch.ones(1),
                                "compatibility": "same",
                            }
                        ]
                    },
                    None,
                    event,
                )
            )
        self.assertEqual(service._pack_pending_batch()["payload"]["ids"], [1])
        pending_event.complete = True
        self.assertEqual(service._pack_pending_batch()["payload"]["ids"], [0])
