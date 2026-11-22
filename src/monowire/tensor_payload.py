"""Read-only tensor payloads with bounded, per-edge reuse on the wire."""

from collections import OrderedDict
from dataclasses import dataclass, field
from uuid import uuid4

import torch


@dataclass(frozen=True)
class SharedTensor:
    """An immutable snapshot. Keys are identities, never hashes of mutable storage.

    Producers create snapshots once; receivers preserve the key and never mutate
    the value. A fresh value must use a fresh snapshot/key.
    """

    key: str
    value: torch.Tensor

    @classmethod
    def snapshot(cls, value):
        return cls(uuid4().hex, value.detach().clone())


@dataclass
class TensorParts:
    """Materialize a packed tensor locally while retaining reusable wire parts."""

    parts: tuple
    _value: torch.Tensor | None = field(default=None, init=False, repr=False)

    def materialize(self):
        if self._value is None:
            values = [part.value if isinstance(part, SharedTensor) else part for part in self.parts]
            self._value = values[0] if len(values) == 1 else torch.cat(values)
        return self._value


class TensorDictionary:
    """Matching LRU dictionaries on the two ends of a FIFO channel.

    The wire explicitly declares a miss, so the receiver never infers whether
    a tensor is present. Both ends promote/evict in descriptor traversal order.
    """

    def __init__(self, max_bytes=256 * 1024**2):
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0

    def get(self, key):
        value = self.entries.get(key)
        if value is not None:
            self.entries.move_to_end(key)
        return value

    def add(self, shared):
        size = shared.value.numel() * shared.value.element_size()
        if size > self.max_bytes:
            return
        while self.entries and self.bytes + size > self.max_bytes:
            _, removed = self.entries.popitem(last=False)
            self.bytes -= removed.value.numel() * removed.value.element_size()
        self.entries[shared.key] = shared
        self.bytes += size

    def clear(self):
        self.entries.clear()
        self.bytes = 0
