"""Finite-job CPU output inbox, independent of the encoder's GPU pipeline."""

import asyncio
import threading
from collections import deque

import torch
import torch.distributed as dist

from ...channels import SHUTDOWN_TIMEOUT


class EncoderInbox:
    """Receive on a dedicated Gloo group while ingress computes or loads weights.

    CPU storage is bounded by the finite job's output sizes, not GPU credits.
    A small bounded queue would deadlock if it filled before ingress drains Qwen.
    """

    def __init__(self, lengths, hidden_dim, source, group, on_output=None):
        self._items = deque()
        self._lock = threading.Lock()
        self._waiter = None
        self._closed = False
        self._started = False
        self.lengths = lengths
        self.hidden_dim, self.source, self.group = hidden_dim, source, group
        self.on_output = on_output
        self.thread = None

    def start(self):
        """Start only after all encoder stages have loaded their checkpoints."""
        if self._closed or self._started:
            raise RuntimeError("Qwen output inbox must be open and not started")
        self._started = True
        if self.source != 0 and self.lengths:
            self.thread = threading.Thread(target=self._receive, daemon=True)
            self.thread.start()

    @staticmethod
    def _wake(waiter):
        if not waiter.done():
            waiter.set_result(None)

    def _notify(self):
        # Called under the lock. Cancellation may already have closed the loop;
        # the item stays buffered and a later consumer can still retrieve it.
        if self._waiter is not None:
            try:
                self._waiter.get_loop().call_soon_threadsafe(self._wake, self._waiter)
            except RuntimeError:
                pass

    def _put(self, item):
        with self._lock:
            if not self._closed:
                self._items.append(item)
                self._notify()

    def publish(self, index, tensor):
        if self.on_output is not None:
            self.on_output(index)
        self._put((index, tensor))

    def _receive(self):
        try:
            for index, length in enumerate(self.lengths):
                with self._lock:
                    if self._closed:
                        return
                tensor = torch.empty((length, self.hidden_dim), dtype=torch.bfloat16)
                dist.recv(tensor, src=self.source, group=self.group)
                self.publish(index, tensor)
        except Exception as error:
            self._put(error)

    async def _next(self):
        while True:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Qwen output inbox is closed")
                if self._items:
                    return self._items.popleft()
                if self._waiter is not None:
                    raise RuntimeError("Qwen output inbox supports one consumer")
                waiter = asyncio.get_running_loop().create_future()
                self._waiter = waiter
            try:
                await asyncio.wait_for(waiter, timeout=SHUTDOWN_TIMEOUT)
            except TimeoutError as error:
                raise TimeoutError("Qwen output did not arrive") from error
            finally:
                with self._lock:
                    if self._waiter is waiter:
                        self._waiter = None

    async def tensors(self):
        for expected in range(len(self.lengths)):
            item = await self._next()
            if isinstance(item, Exception):
                raise RuntimeError("Qwen output receiver failed") from item
            index, tensor = item
            if index != expected:
                raise RuntimeError(f"Qwen output order mismatch: {index} != {expected}")
            yield index, tensor

    def close(self):
        with self._lock:
            self._closed = True
            self._items.clear()
            self._notify()
        # Gloo's blocking recv has no supported Python cancellation API. Its
        # process-group timeout bounds this join before group teardown. No
        # executor thread remains waiting for an inbox item meanwhile.
        if self.thread is not None:
            self.thread.join(timeout=SHUTDOWN_TIMEOUT)
            if self.thread.is_alive():
                raise TimeoutError("Qwen output receiver did not terminate")
