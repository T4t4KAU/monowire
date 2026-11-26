"""Finite-job CPU output inbox, independent of the encoder's GPU pipeline."""

import asyncio
import queue
import threading

import torch
import torch.distributed as dist

from ...channels import SHUTDOWN_TIMEOUT


class EncoderInbox:
    """Receive on a dedicated Gloo group while ingress computes or loads weights.

    CPU storage is bounded by the finite job's output sizes, not GPU credits.
    A small bounded queue would deadlock if it filled before ingress drains Qwen.
    """

    def __init__(self, lengths, hidden_dim, source, group, on_output=None):
        self.queue = queue.Queue()
        self.lengths = lengths
        self.hidden_dim, self.source, self.group = hidden_dim, source, group
        self.on_output = on_output
        self.thread = None
        if source != 0:
            self.thread = threading.Thread(target=self._receive, daemon=True)
            self.thread.start()

    def publish(self, index, tensor):
        if self.on_output is not None:
            self.on_output(index)
        self.queue.put((index, tensor))

    def _receive(self):
        try:
            for index, length in enumerate(self.lengths):
                tensor = torch.empty((length, self.hidden_dim), dtype=torch.bfloat16)
                dist.recv(tensor, src=self.source, group=self.group)
                self.publish(index, tensor)
        except Exception as error:
            self.queue.put(error)

    async def tensors(self):
        for expected in range(len(self.lengths)):
            try:
                item = await asyncio.to_thread(self.queue.get, True, SHUTDOWN_TIMEOUT)
            except queue.Empty as error:
                raise TimeoutError("Qwen output did not arrive") from error
            if isinstance(item, Exception):
                raise RuntimeError("Qwen output receiver failed") from item
            index, tensor = item
            if index != expected:
                raise RuntimeError(f"Qwen output order mismatch: {index} != {expected}")
            yield index, tensor

    def close(self):
        if self.thread is not None:
            self.thread.join(timeout=SHUTDOWN_TIMEOUT)
            if self.thread.is_alive():
                raise TimeoutError("Qwen output receiver did not terminate")
