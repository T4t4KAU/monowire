"""Model-independent tensor pipeline transport and actor lifecycle."""

import asyncio

import torch

from .interfaces import StageAdapter


class PipelineExecutor:
    """Ingress executes stage 0; independent stage actors process queued misses.

    A dedicated receiver drains final activations while ingress prepares/dispatches
    other requests. Hits finish at ingress,
    where native output heads and the whole-suffix cache are already resident.
    """

    def __init__(self, stage: StageAdapter):
        self.stage = stage
        self.group = stage.group
        if self.group.rank_in_group != 0:
            raise ValueError("Only ingress can construct PipelineExecutor")
        self.send_lock = asyncio.Lock()
        self.waiting = {}
        self.next_ticket = 0
        self.receiver = None
        self.failure = None
        self.closed = False

    def _check_open(self):
        if self.failure is not None:
            raise RuntimeError(f"Pipeline communication failed: {self.failure}") from self.failure
        if self.closed:
            raise RuntimeError("Pipeline executor is closed")

    def _fail(self, error):
        self.failure = error
        for future in self.waiting.values():
            if not future.done():
                future.set_exception(error)
        self.waiting.clear()

    async def _publish(self, packet):
        async with self.send_lock:
            if self.failure is None:
                try:
                    await asyncio.to_thread(self._send, packet)
                except Exception as exc:
                    self._fail(exc)

    def _send(self, packet):
        if packet["hidden"].is_cuda:
            torch.cuda.set_device(packet["hidden"].device)
        self.group.send_tensor_dict(packet, dst=1)

    def _recv(self):
        if torch.device(self.group.device).type == "cuda":
            torch.cuda.set_device(self.group.device)
        packet = self.group.recv_tensor_dict(src=self.group.world_size - 1)
        if packet["hidden"].is_cuda:
            torch.cuda.synchronize(packet["hidden"].device)
        return packet

    async def _receive(self):
        try:
            while True:
                packet = await asyncio.to_thread(self._recv)
                future = self.waiting.pop(packet["ticket"])
                if not future.done():
                    if packet.get("error"):
                        future.set_exception(RuntimeError(packet["error"]))
                    else:
                        future.set_result(packet["hidden"])
                if packet["hidden"].numel() == 0:
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail(exc)

    async def execute(self, prepared):
        self._check_open()
        if not prepared:
            raise ValueError("Cannot execute an empty physical batch")
        self.stage.validate(prepared)
        if any(p.hit != prepared[0].hit for p in prepared):
            raise ValueError("A physical batch must have one cache route")
        if prepared[0].hit:
            with torch.profiler.record_function("monowire/continuous/cache_hit"):
                return self.stage.reuse(prepared)
        packet, context = self.stage.pack(prepared)
        packet["ticket"] = self.next_ticket
        self.next_ticket += 1
        if self.group.world_size == 1:
            return self.stage.unpack(packet["hidden"], context)
        if self.receiver is None:
            self.receiver = asyncio.create_task(self._receive())
        future = asyncio.get_running_loop().create_future()
        self.waiting[packet["ticket"]] = future
        # Publish producer work before handing buffers to the communication
        # thread's CUDA stream. Different pipeline GPUs continue independently.
        if packet["hidden"].is_cuda:
            torch.cuda.synchronize(packet["hidden"].device)
        await self._publish(packet)
        # The blocking send has consumed these buffers. Keep only the unpack
        # context while downstream stages work, not another batch of activations.
        del packet
        output = await asyncio.shield(future)
        return self.stage.unpack(output, context)

    async def close(self):
        if self.closed:
            return
        self._check_open()
        if self.waiting:
            raise RuntimeError("Drain all requests before shutting down pipeline actors")
        self.closed = True
        if self.group.world_size > 1:
            # A stop packet traverses the same FIFO path and produces a final
            # response, unblocking the receiver thread before it is joined.
            ticket = self.next_ticket
            future = asyncio.get_running_loop().create_future()
            self.waiting[ticket] = future
            if self.receiver is None:
                self.receiver = asyncio.create_task(self._receive())
            packet = dict(ticket=ticket, hidden=torch.empty(0, device=self.group.device))
            await self._publish(packet)
            await future
            await self.receiver


def serve_stage(stage: StageAdapter):
    """Non-ingress ranks execute only miss batches; no step-wide collective."""
    group = stage.group
    rank = group.rank_in_group
    if rank == 0:
        raise ValueError("Ingress owns the request scheduler")
    while True:
        packet = group.recv_tensor_dict(src=rank - 1)
        stop = packet["hidden"].numel() == 0
        if not stop and not packet.get("error"):
            try:
                stage.compute(packet)
            except Exception as exc:
                packet["error"] = f"Pipeline stage {rank}: {exc}"
        if rank == group.world_size - 1:
            # Only the final activation returns; block metadata is not broadcast.
            group.send_tensor_dict(
                dict(ticket=packet["ticket"], hidden=packet["hidden"], error=packet.get("error")),
                dst=0,
            )
        else:
            group.send_tensor_dict(packet, dst=rank + 1)
        if stop:
            return
