"""Reusable receive storage with explicit end-of-use, not queue-pop lifetime.

Both implementations expose the same byte budget. Ring storage is contiguous
and FIFO-reclaimed. Pool storage uses power-of-two size classes and can reclaim
completed messages independently. Neither implementation splits a tensor.
"""

import math
import time
from collections import deque

import torch

SLOT_KEY = "__monowire_receive_storage__"


def align(size, alignment):
    return (size + alignment - 1) // alignment * alignment


class BufferUse:
    def __init__(self, owner, raw, start, end, size, slot=None):
        self.owner, self.raw = owner, raw
        self.start, self.end, self.size, self.slot = start, end, size, slot
        self.closed = False

    def close(self):
        with self.owner.memory.changed:
            if self.closed:
                return
            self.closed = True
            self.owner.live_bytes -= self.size
            if self.slot is not None:
                self.slot[1] = False
            else:
                while self.owner.pending and self.owner.pending[0].closed:
                    self.owner.tail = self.owner.pending.popleft().end
            self.owner.memory.changed.notify_all()


class ReceiveBuffers:
    """Single receiver, CPU-owned allocation; any thread may complete a use."""

    def __init__(self, memory, *, feedback=False):
        self.memory, self.feedback = memory, feedback
        limits = memory.limits
        self.strategy, self.alignment = limits.buffer_strategy, limits.buffer_alignment
        if feedback:
            capacity = memory.capacities["feedback"]
        else:
            capacity = (memory.normal_capacity - memory.used["normal"]) // 3
            if limits.buffer_bytes is not None:
                capacity = min(capacity, limits.buffer_bytes)
        self.capacity = capacity // self.alignment * self.alignment
        if not self.capacity:
            raise ValueError("Receive capacity is smaller than its alignment")
        self.lease = memory.reserve(
            self.capacity * (1 if feedback else 3), purpose="feedback" if feedback else "normal"
        )
        self.arena = None
        self.head = self.tail = 0
        self.pending = deque()
        self.slots = []  # [storage, busy], retained only within this byte budget
        self.live_bytes = self.peak_live_bytes = 0
        self.allocations = self.reuses = self.waits = 0
        self.wrap_padding_bytes = self.head_blocked_waits = 0
        self.alignment_padding_bytes = 0
        self.rounding_bytes = 0
        self.wait_seconds = 0.0
        self.closed = False
        memory.buffers.append(self)

    def _try(self, size, alignment):
        if self.strategy == "ring":
            if not self.pending:
                self.head = self.tail = 0
            offset = self.head % self.capacity
            aligned_offset = align(offset, alignment)
            wrap = aligned_offset + size > self.capacity
            padding = self.capacity - offset if wrap else aligned_offset - offset
            if self.head + padding + size - self.tail > self.capacity:
                if any(use.closed for use in self.pending):
                    self.head_blocked_waits += 1
                return None
            if self.arena is None:
                self.arena = torch.empty(
                    self.capacity, dtype=torch.uint8, device=self.memory.device
                )
                self.allocations += 1
            else:
                self.reuses += 1
            start = self.head + padding
            self.head = start + size
            if wrap:
                self.wrap_padding_bytes += padding
            else:
                self.alignment_padding_bytes += padding
            use = BufferUse(
                self, self.arena.narrow(0, start % self.capacity, size), start, self.head, size
            )
            self.pending.append(use)
            return use
        rounded = 1 << (size - 1).bit_length()
        if rounded > self.capacity:
            # A payload fitting the entire pool must not fail solely due to rounding.
            rounded = self.capacity
        candidates = [slot for slot in self.slots if not slot[1] and slot[0].numel() == rounded]
        if candidates:
            slot = candidates[0]
            self.reuses += 1
        else:
            # Idle incompatible size classes can be discarded. Live classes
            # cannot be evicted; no duplicate physical allocation beyond capacity.
            allocated = sum(slot[0].numel() for slot in self.slots)
            for candidate in list(self.slots):
                if allocated + rounded <= self.capacity:
                    break
                if not candidate[1]:
                    allocated -= candidate[0].numel()
                    self.slots = [slot for slot in self.slots if slot is not candidate]
            candidate = None
            if allocated + rounded > self.capacity:
                return None
            slot = [torch.empty(rounded, dtype=torch.uint8, device=self.memory.device), False]
            self.slots.append(slot)
            self.allocations += 1
        slot[1] = True
        return BufferUse(self, slot[0], 0, rounded, rounded, slot)

    def acquire(self, layout, *, retained_bytes=0, timeout=180):
        """Return (use, index->tensor) for non-shared tensors in one wire packet."""
        offsets, size = [], 0
        alignment = self.alignment
        for index, shape, dtype in layout:
            element = torch.empty((), dtype=dtype).element_size()
            alignment = max(alignment, element)
            size = align(size, max(self.alignment, element))
            count = math.prod(shape)
            offsets.append((index, size, count * element, shape, dtype))
            size += count * element
        size = align(size + retained_bytes, self.alignment)
        if not size:
            return None, {}
        if size > self.capacity:
            raise ValueError(f"Wire payload {size} exceeds receive capacity {self.capacity}")
        started = time.monotonic()
        with self.memory.changed:
            while True:
                if self.closed:
                    raise RuntimeError("Receive buffer is closed")
                self.memory._reap()
                use = self._try(size, alignment)
                if use is not None:
                    if self.strategy == "pool":
                        self.rounding_bytes += use.size - size
                    self.live_bytes += use.size
                    self.peak_live_bytes = max(self.peak_live_bytes, self.live_bytes)
                    self.wait_seconds += time.monotonic() - started
                    try:
                        tensors = {
                            index: use.raw.narrow(0, offset, count).view(dtype).reshape(shape)
                            for index, offset, count, shape, dtype in offsets
                        }
                    except BaseException:
                        use.close()
                        raise
                    return use, tensors
                self.waits += 1
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError(f"{self.strategy} receive buffer did not retire")
                self.memory.changed.wait(min(remaining, 0.01))

    def close(self):
        with self.memory.changed:
            if self.closed:
                return
            self.memory._reap()
            if self.live_bytes:
                raise RuntimeError("Receive buffer still has live consumers")
            self.arena = None
            self.slots.clear()
            self.pending.clear()
            self.lease.close()
            self.closed = True

    def snapshot(self):
        return dict(
            strategy=self.strategy,
            feedback=self.feedback,
            capacity=self.capacity,
            live_bytes=self.live_bytes,
            peak_live_bytes=self.peak_live_bytes,
            allocations=self.allocations,
            reuses=self.reuses,
            waits=self.waits,
            wait_seconds=self.wait_seconds,
            wrap_padding_bytes=self.wrap_padding_bytes,
            alignment_padding_bytes=self.alignment_padding_bytes,
            head_blocked_waits=self.head_blocked_waits,
            rounding_bytes=self.rounding_bytes,
        )


def retire_message(message, event=None):
    """Transfer the receive ownership to a last-use event exactly once."""
    if isinstance(message, dict):
        for use in message.pop(SLOT_KEY, ()):
            use.owner.memory.retain(event, None, use)
