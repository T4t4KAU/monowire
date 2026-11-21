"""Bounded, request-private residual caches for pipeline-parallel H3."""

import math
from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CacheConfig:
    policy: str = "none"
    threshold: float = 0.17
    warmup_steps: int = 2
    max_skip_steps: int = 2

    def __post_init__(self):
        if self.policy not in {"none", "first_block"}:
            raise ValueError("cache policy must be none or first_block")
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("cache threshold must be finite and nonnegative")
        if self.warmup_steps < 1 or self.max_skip_steps < 1:
            raise ValueError("cache warmup_steps and max_skip_steps must be positive")


@dataclass
class CacheEntry:
    step: int = -1
    probe: torch.Tensor | None = None
    residual: torch.Tensor | None = None
    skips: int = 0
    hits: int = 0
    computes: int = 0


class RequestCache:
    def __init__(self, config: CacheConfig, capacity: int):
        if capacity < 1:
            raise ValueError("cache capacity must be positive")
        self.config, self.capacity = config, capacity
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = self.misses = 0

    def entry(self, key: str, step: int) -> CacheEntry:
        entry = self.entries.pop(key, CacheEntry())
        # A retry, restart, skipped step or reused request ID cannot use old state.
        if step != entry.step + 1:
            entry = CacheEntry()
        self.entries[key] = entry
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return entry

    def should_reuse(self, entry, probe, step, total_steps):
        config = self.config
        reusable = (
            config.policy != "none"
            and config.threshold > 0
            and step >= config.warmup_steps
            and step < total_steps - 1
            and entry.residual is not None
            and entry.probe is not None
            and entry.probe.shape == probe.shape
            and entry.skips < config.max_skip_steps
        )
        if reusable:
            previous = entry.probe.float()
            difference = (probe.float() - previous).abs().mean()
            denominator = previous.abs().mean().clamp_min(1e-6)
            distance = float((difference / denominator).item())
            reusable = math.isfinite(distance) and distance < config.threshold
        # FBC compares against the last full evaluation.
        if not reusable:
            entry.probe = probe.detach().clone()
        return reusable

    def commit(self, entry, step, reuse, residual=None):
        entry.step = step
        entry.skips = entry.skips + 1 if reuse else 0
        if residual is not None:
            entry.residual = residual.detach().clone()
        entry.hits += int(reuse)
        entry.computes += int(not reuse)
        self.hits += int(reuse)
        self.misses += int(not reuse)
