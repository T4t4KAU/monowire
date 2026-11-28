"""Byte-based residency limits. Progress domains cannot borrow each other's reserve.

Weights and allocator/library headroom are outside these dynamic reservations.
Shared storage is deduplicated within one payload; overlapping owners in different
domains are deliberately charged twice, rather than releasing live CUDA storage.
"""

import queue
import time
from collections import deque
from dataclasses import dataclass, fields, is_dataclass

import torch


def storage_bytes(value):
    """Count unique physical tensor storage without traversing model objects."""
    seen, storages = set(), set()

    def visit(item):
        if id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            storages.add((str(item.device), storage.data_ptr(), storage.nbytes()))
        elif isinstance(item, torch.nn.Module):
            return
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        elif is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name))

    visit(value)
    return sum(size for _, _, size in storages)


@dataclass(frozen=True)
class MemoryLimits:
    """Opt-in BS1 byte policy; values are per GPU, in bytes.

    pipeline_bytes is split into nine equal progress domains: two channel
    in-flight windows, receive inbox, compute window, send queue, prefix lookahead,
    and three maximum-payload handoff reserves. This intentionally conservative split
    bounds handoffs without acquiring a shared pool while holding a full inbox.
    request_bytes covers persistent ingress state. workspace_bytes is a required
    measured peak reserve, not an automatic prediction of arbitrary kernel scratch.
    """

    request_bytes: int
    pipeline_bytes: int
    workspace_bytes: int
    headroom_bytes: int = 1024**3
    dictionary_bytes: int = 256 * 1024**2
    request_bytes_per_token: int = 16384
    request_fixed_bytes: int = 1024**2

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field.name} must be a positive byte count")
        if self.pipeline_bytes < 9:
            raise ValueError("Pipeline budget must cover nine progress domains")

    @property
    def buffer_bytes(self):
        return self.pipeline_bytes // 9

    @property
    def dynamic_bytes(self):
        return (
            self.request_bytes
            + self.pipeline_bytes
            + self.workspace_bytes
            + self.headroom_bytes
            + 2 * self.dictionary_bytes
        )


class ByteQueue(queue.Queue):
    """Queue bounded by payload bytes; a popped item keeps credit until task_done.

    Only one consumer is supported. Control messages carry zero tensor bytes and
    can always enter, so STOP/failure does not require a data residency credit.
    """

    def __init__(self, capacity, weight=storage_bytes):
        super().__init__(maxsize=0)
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Byte queue capacity must be positive")
        self.capacity = capacity
        self.weight = weight
        self.used_bytes = 0
        self.peak_bytes = 0
        self._weights = deque()
        self._queued_weights = deque()

    def can_put(self, item):
        size = self.weight(item)
        if size > self.capacity:
            raise ValueError(f"Payload needs {size} bytes; queue budget is {self.capacity}")
        with self.mutex:
            return self.used_bytes + size <= self.capacity

    def full(self):
        with self.mutex:
            return self.used_bytes >= self.capacity

    def put(self, item, block=True, timeout=None):
        size = self.weight(item)
        if size > self.capacity:
            raise ValueError(f"Payload needs {size} bytes; queue budget is {self.capacity}")
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.not_full:
            while self.used_bytes + size > self.capacity:
                if not block:
                    raise queue.Full
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise queue.Full
                self.not_full.wait(remaining)
            self.used_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.used_bytes)
            self._queued_weights.append(size)
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()

    def _get(self):
        self._weights.append(self._queued_weights.popleft())
        return super()._get()

    def task_done(self, weight=None):
        with self.all_tasks_done:
            if not self._weights:
                raise ValueError("Byte queue has no consumed item to release")
            if weight is None:
                weight = self._weights.popleft()
            else:
                self._weights.remove(weight)
            self.used_bytes -= weight
            self.unfinished_tasks -= 1
            self.not_full.notify_all()
            if not self.unfinished_tasks:
                self.all_tasks_done.notify_all()
