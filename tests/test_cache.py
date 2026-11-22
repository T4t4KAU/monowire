"""Request cache identity, retry and capacity boundaries."""

import unittest
from unittest.mock import patch

import torch

from monowire.cache import CacheConfig, RequestCache


def decide(cache, entry, probe, step, total_steps):
    return cache.should_reuse_batch([entry], [probe], [step], [total_steps])[0]


class CacheTests(unittest.TestCase):
    def test_batch_decisions_have_one_host_read_and_independent_guards(self):
        cache = RequestCache(CacheConfig("first_stage", warmup_steps=1), capacity=8)
        entries = [cache.entry(str(i), 0) for i in range(8)]
        for entry in entries:
            entry.probe = torch.ones(2)
            cache.commit(entry, 0, False, torch.ones(2))
        entries[5].skips = 2
        probes = [torch.full((2,), v) for v in (1.0, 1.3, 1.0, 1.0, float("nan"), 1.0, 1.1, 1.0)]
        steps = [1, 1, 0, 7, 1, 2, 3, 1]
        host_reads = []
        original = torch.Tensor.tolist

        def read(tensor):
            host_reads.append(tensor.numel())
            return original(tensor)

        with (
            patch.object(torch.Tensor, "tolist", read),
            patch.object(torch.Tensor, "item", side_effect=AssertionError("scalar GPU read")),
        ):
            decisions = cache.should_reuse_batch(entries, probes, steps, [8] * 8)
        self.assertEqual(decisions, [True, False, False, False, False, False, True, True])
        self.assertEqual(host_reads, [5])

    def test_batch_guard_preserves_python_float_threshold_precision(self):
        cache = RequestCache(CacheConfig("first_stage", threshold=1.00000001), capacity=1)
        entry = cache.entry("a", 0)
        entry.probe = torch.ones(2)
        cache.commit(entry, 0, False, torch.ones(2))
        # Distance is exactly 1.0; rounding the threshold to float32 would reject it.
        self.assertTrue(decide(cache, entry, torch.full((2,), 2.0), 2, 8))

    def test_forced_full_batch_needs_no_host_read(self):
        cache = RequestCache(CacheConfig("first_stage"), capacity=1)
        with patch.object(torch.Tensor, "tolist", side_effect=AssertionError("GPU read")):
            self.assertEqual(
                cache.should_reuse_batch([cache.entry("a", 0)], [torch.ones(2)], [0], [8]),
                [False],
            )

    def test_selected_probes_use_maximum_of_independently_normalized_changes(self):
        cache = RequestCache(
            CacheConfig("first_stage", warmup_steps=1, probe_blocks=(1, 3, 6)), capacity=1
        )
        entry = cache.entry("a", 0)
        probe = torch.tensor([[100.0], [1.0], [100.0]])
        self.assertFalse(decide(cache, entry, probe, 0, 8))
        cache.commit(entry, 0, False, torch.ones(1))
        # Aggregate or average distances would pass, but block 3 changed by 30%.
        self.assertFalse(
            decide(cache, cache.entry("a", 1), torch.tensor([[100.0], [1.3], [100.0]]), 1, 8)
        )

    def test_selected_periodic_probes_keep_adjacent_reference_and_two_step_limit(self):
        cache = RequestCache(
            CacheConfig("first_stage_periodic", warmup_steps=1, probe_blocks=(1, 3, 6)), capacity=1
        )
        decisions = []
        for step in range(7):
            entry = cache.entry("a", step)
            probe = torch.tensor([[1.0], [3.0], [6.0]]) * 1.1**step
            reuse = decide(cache, entry, probe, step, 7)
            decisions.append(reuse)
            cache.commit(entry, step, reuse, None if reuse else torch.ones(1))
        self.assertEqual(decisions, [False, True, True, False, True, True, False])

    def test_probe_selection_validation(self):
        for blocks in ((0,), (1, 1), (3, 1), (True,), (1.5,)):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                CacheConfig("first_stage", probe_blocks=blocks)
        with self.assertRaisesRegex(ValueError, "stage cache policy"):
            CacheConfig("first_block", probe_blocks=(1,))
        cache = RequestCache(CacheConfig("first_stage", probe_blocks=(1, 3, 6)), capacity=1)
        with self.assertRaisesRegex(ValueError, "one residual tensor"):
            decide(cache, cache.entry("a", 0), torch.ones(2, 4), 0, 8)

    def decisions(self, policy, probes, *, threshold=0.17, warmup=1):
        cache = RequestCache(CacheConfig(policy, threshold, warmup, 2), capacity=2)
        decisions = []
        for step, value in enumerate(probes):
            entry = cache.entry("a", step)
            reuse = decide(cache, entry, torch.tensor([value]), step, len(probes))
            decisions.append(reuse)
            cache.commit(entry, step, reuse, None if reuse else torch.tensor([float(step)]))
        return decisions

    def test_periodic_stage_tracks_adjacent_probes_and_forces_refresh(self):
        probes = [1.0, 1.1, 1.21, 1.331, 1.4641, 1.61051, 1.771561, 1.9487171]
        self.assertEqual(
            self.decisions("first_stage_periodic", probes),
            [False, True, True, False, True, True, False, False],
        )
        # The last-full reference sees 21% drift after two steps and rejects it.
        self.assertEqual(self.decisions("first_stage", probes)[:3], [False, True, False])

    def test_periodic_guard_interrupts_a_cycle_and_preserves_warmup_and_tail(self):
        self.assertEqual(
            self.decisions("first_stage_periodic", [1.0, 1.0, 1.1, 2.0, 2.1, 2.2, 2.3], warmup=2),
            [False, False, True, False, True, True, False],
        )
        self.assertEqual(
            self.decisions("first_stage_periodic", [1.0] * 5, threshold=0), [False] * 5
        )

    def test_periodic_cycles_are_request_private_and_keep_full_residual(self):
        cache = RequestCache(CacheConfig("first_stage_periodic", warmup_steps=1), capacity=2)
        for key in ("a", "b"):
            entry = cache.entry(key, 0)
            self.assertFalse(decide(cache, entry, torch.ones(2), 0, 8))
            cache.commit(entry, 0, False, torch.full((2,), 7.0))
        for step in (1, 2):
            entry = cache.entry("a", step)
            self.assertTrue(decide(cache, entry, torch.ones(2), step, 8))
            cache.commit(entry, step, True)
            torch.testing.assert_close(entry.residual, torch.full((2,), 7.0))
        self.assertFalse(decide(cache, cache.entry("a", 3), torch.ones(2), 3, 8))
        self.assertTrue(decide(cache, cache.entry("b", 1), torch.ones(2), 1, 8))
        self.assertIsNone(cache.entry("a", 0).residual)

    def test_nonfinite_periodic_probe_cannot_trigger_reuse(self):
        for value in (float("nan"), float("inf")):
            self.assertEqual(
                self.decisions("first_stage_periodic", [1.0, value, 1.0, 1.0]),
                [False] * 4,
            )

    def test_reset_retry_eviction_and_threshold(self):
        cache = RequestCache(CacheConfig("first_block", threshold=0, warmup_steps=1), 1)
        entry = cache.entry("a", 0)
        probe = torch.ones(2)
        self.assertFalse(decide(cache, entry, probe, 0, 5))
        cache.commit(entry, 0, False, torch.ones(2))
        entry = cache.entry("a", 1)
        self.assertFalse(decide(cache, entry, probe, 1, 5))
        cache.commit(entry, 1, False, torch.ones(2))
        self.assertIsNone(cache.entry("a", 1).residual)
        cache.entry("b", 0)
        self.assertEqual(list(cache.entries), ["b"])
        self.assertIsNone(cache.entry("a", 3).residual)
