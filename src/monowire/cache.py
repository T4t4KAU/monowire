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
    probe_blocks: tuple[int, ...] = ()

    def __post_init__(self):
        if self.policy not in {"none", "first_block", "first_stage", "first_stage_periodic"}:
            raise ValueError(
                "cache policy must be none, first_block, first_stage or first_stage_periodic"
            )
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("cache threshold must be finite and nonnegative")
        if self.warmup_steps < 1 or self.max_skip_steps < 1:
            raise ValueError("cache warmup_steps and max_skip_steps must be positive")
        blocks = tuple(self.probe_blocks)
        if any(type(block) is not int or block < 1 for block in blocks):
            raise ValueError("probe_blocks must contain positive, one-based block indices")
        if blocks != tuple(sorted(set(blocks))):
            raise ValueError("probe_blocks must be unique and sorted")
        if blocks and self.policy not in {"first_stage", "first_stage_periodic"}:
            raise ValueError("probe_blocks requires a stage cache policy")
        object.__setattr__(self, "probe_blocks", blocks)


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

    def should_reuse_batch(self, entries, probes, steps, total_steps):
        """Evaluate request-private guards, reading the whole decision vector once."""
        config = self.config
        members = list(zip(entries, probes, steps, total_steps, strict=True))
        reusable = [False] * len(members)
        candidates, distances = [], []
        for index, (entry, probe, step, total) in enumerate(members):
            if config.probe_blocks and (
                probe.ndim < 2 or probe.shape[0] != len(config.probe_blocks)
            ):
                raise ValueError("Expected one residual tensor per configured probe block")
            eligible = (
                config.policy != "none"
                and config.threshold > 0
                and step >= config.warmup_steps
                and step < total - 1
                and entry.residual is not None
                and entry.probe is not None
                and entry.probe.shape == probe.shape
                and entry.skips < config.max_skip_steps
            )
            if not eligible:
                continue
            previous = entry.probe.float()
            difference = (probe.float() - previous).abs()
            magnitude = previous.abs()
            if config.probe_blocks:
                # Normalize each block independently before taking the maximum.
                difference = difference.flatten(1).mean(1)
                magnitude = magnitude.flatten(1).mean(1)
            else:
                difference = difference.mean()
                magnitude = magnitude.mean()
            candidates.append(index)
            distances.append((difference / magnitude.clamp_min(1e-6)).max())
        if distances:
            # Match the scalar guard's Python-float threshold comparison even
            # when a threshold lies between adjacent float32 values.
            distance = torch.stack(distances).to(torch.float64)
            decisions = (torch.isfinite(distance) & (distance < config.threshold)).tolist()
            for index, decision in zip(candidates, decisions, strict=True):
                reusable[index] = decision
        # Periodic stage reuse compares adjacent, freshly computed prefixes.
        # The skip count bounds the reuse cycle; a failed distance guard ends it
        # early. Other policies retain their last-full-evaluation reference.
        for (entry, probe, _, _), reuse in zip(members, reusable, strict=True):
            if config.policy == "first_stage_periodic" or not reuse:
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
