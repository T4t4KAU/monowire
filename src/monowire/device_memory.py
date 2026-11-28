"""CPU capacity ledger; asynchronous CUDA events control storage lifetimes.

Capacity is sampled once after weights are loaded. Request admission is a CPU
counter operation. Sequential compute operations share one workspace reserve.
"""

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceMemoryLimits:
    """BS1, single-ingress admission for pre-encoded DiT.

    buffer_strategy: FIFO variable-size 'ring' or independently recycled 'pool'.
    buffer_bytes optionally caps ordinary receive capacity, otherwise derived
    from available space. Each receive byte also reserves two bytes for retained
    packing/outputs. Half of progress_bytes protects feedback; half wire copies.
    """

    workspace_bytes: int
    headroom_bytes: int = 1024**3
    progress_bytes: int = 1024**3
    dictionary_bytes: int = 256 * 1024**2
    request_bytes_per_token: int = 16384
    request_fixed_bytes: int = 1024**2
    buffer_strategy: str = "ring"
    buffer_bytes: int | None = None
    buffer_alignment: int = 256

    def __post_init__(self):
        for name in (
            "workspace_bytes",
            "headroom_bytes",
            "progress_bytes",
            "dictionary_bytes",
            "request_bytes_per_token",
            "request_fixed_bytes",
            "buffer_alignment",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.buffer_strategy not in {"ring", "pool"}:
            raise ValueError("buffer_strategy must be 'ring' or 'pool'")
        if self.buffer_bytes is not None and (
            type(self.buffer_bytes) is not int or self.buffer_bytes < 1
        ):
            raise ValueError("buffer_bytes must be positive or None")
        if self.buffer_alignment & (self.buffer_alignment - 1):
            raise ValueError("buffer_alignment must be a power of two")


class MemoryLease:
    def __init__(self, pool, size, purpose):
        self.pool, self.size, self.purpose = pool, size, purpose
        self.closed = False

    def update(self, value):
        # One reservation covers allocated and future request storage throughout
        # its lifetime. No tensor scan, or second charge for allocated bytes.
        if self.closed:
            raise RuntimeError("Updating a closed memory lease")

    def close(self):
        with self.pool.changed:
            if not self.closed:
                self.closed = True
                self.pool.leases.remove(self)
                self.pool.used[self.purpose] -= self.size
                self.pool.changed.notify_all()


class DeviceMemoryPool:
    def __init__(self, device, limits, *, available=None):
        self.device = torch.device(device)
        self.limits = limits
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.leases = set()
        self.retired = []
        self._available = available or self._cuda_available
        self.initial_available = self._available()
        self.fixed_margin = (
            limits.workspace_bytes + limits.headroom_bytes + 2 * limits.dictionary_bytes
        )
        self.normal_capacity = self.initial_available - self.fixed_margin - limits.progress_bytes
        if self.normal_capacity <= 0:
            raise ValueError("Insufficient space for workspace, dictionaries and progress")
        # Separate cumulative budgets, established before any channel receives
        # its first packet. Wire copies cannot borrow the feedback reserve.
        self.capacities = dict(
            normal=self.normal_capacity,
            feedback=limits.progress_bytes // 2,
            wire=limits.progress_bytes - limits.progress_bytes // 2,
        )
        self.used = dict.fromkeys(self.capacities, 0)
        self.peak_promised_bytes = 0
        self.backpressure_count = 0
        self.wait_seconds = 0.0
        self.admissions = 0
        self._compute_stream = None
        self.buffers = []

    def _cuda_available(self):
        free, _ = torch.cuda.mem_get_info(self.device)
        stats = torch.cuda.memory_stats(self.device)
        reusable = max(
            0,
            stats.get("reserved_bytes.all.current", 0)
            - stats.get("active_bytes.all.current", 0)
            - stats.get("inactive_split_bytes.all.current", 0),
        )
        return free + reusable

    def _reap(self):
        pending, retired = [], self.retired
        self.retired = []
        for event, resources, lease in retired:
            if event.query():
                if lease is not None:
                    lease.close()
            else:
                pending.append((event, resources, lease))
        self.retired.extend(pending)

    def try_reserve(self, size, *, purpose="normal"):
        if type(size) is not int or size < 0:
            raise ValueError("Reservation must be a nonnegative byte count")
        if purpose not in self.used:
            raise ValueError("Reservation purpose must be normal, feedback, or wire")
        with self.changed:
            self._reap()
            capacity = self.capacities[purpose]
            if size > capacity:
                raise ValueError(
                    f"Allocation of {size} bytes exceeds {purpose} capacity {capacity}"
                )
            if self.used[purpose] + size > capacity:
                self.backpressure_count += 1
                return None
            lease = MemoryLease(self, size, purpose)
            self.leases.add(lease)
            self.used[purpose] += size
            self.admissions += 1
            self.peak_promised_bytes = max(self.peak_promised_bytes, sum(self.used.values()))
            return lease

    def reserve(self, size, *, purpose="normal", timeout=180):
        start = time.monotonic()
        with self.changed:
            while True:
                lease = self.try_reserve(size, purpose=purpose)
                if lease is not None:
                    self.wait_seconds += time.monotonic() - start
                    return lease
                remaining = timeout - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError(f"Memory admission stalled for {purpose}: {size} bytes")
                # CPU releases notify; CUDA completion needs bounded event polling.
                self.changed.wait(min(remaining, 0.01))

    def retain(self, event, resources, lease=None):
        with self.changed:
            if event is None:
                if lease is not None:
                    lease.close()
            else:
                self.retired.append((event, resources, lease))
            self._reap()

    def release_after_compute(self, lease):
        event = None
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
        self.retain(event, None, lease)

    @contextmanager
    def workspace(self):
        if self.device.type == "cuda":
            stream = torch.cuda.current_stream(self.device).cuda_stream
            if self._compute_stream is not None and self._compute_stream != stream:
                raise RuntimeError("Shared workspace requires one compute stream")
            self._compute_stream = stream
        yield

    def drain(self):
        while True:
            with self.changed:
                self._reap()
                pending = list(self.retired)
            if not pending:
                return
            # Communication completion may need another thread to acquire
            # capacity or retire storage. Never hold its ledger lock while waiting.
            for event, _, _ in pending:
                event.synchronize()

    def snapshot(self):
        with self.changed:
            self._reap()
            return dict(
                available_bytes=self._available(),
                normal_capacity=self.normal_capacity,
                capacity_by_purpose=dict(self.capacities),
                used_by_purpose=dict(self.used),
                promised_bytes=sum(self.used.values()),
                peak_promised_bytes=self.peak_promised_bytes,
                backpressure_count=self.backpressure_count,
                wait_seconds=self.wait_seconds,
                admissions=self.admissions,
                live_leases=len(self.leases),
                retained_operations=len(self.retired),
                buffers=[buffer.snapshot() for buffer in self.buffers],
            )
