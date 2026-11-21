"""Request cache identity, retry and capacity boundaries."""

import unittest

import torch

from monowire.cache import CacheConfig, RequestCache


class CacheTests(unittest.TestCase):
    def test_reset_retry_eviction_and_threshold(self):
        cache = RequestCache(CacheConfig("first_block", threshold=0, warmup_steps=1), 1)
        entry = cache.entry("a", 0)
        probe = torch.ones(2)
        self.assertFalse(cache.should_reuse(entry, probe, 0, 5))
        cache.commit(entry, 0, False, torch.ones(2))
        entry = cache.entry("a", 1)
        self.assertFalse(cache.should_reuse(entry, probe, 1, 5))
        cache.commit(entry, 1, False, torch.ones(2))
        self.assertIsNone(cache.entry("a", 1).residual)
        cache.entry("b", 0)
        self.assertEqual(list(cache.entries), ["b"])
        self.assertIsNone(cache.entry("a", 3).residual)
