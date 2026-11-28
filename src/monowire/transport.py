"""Replicated request owners feeding a buffered, shared pipeline suffix."""

import asyncio
import logging
import queue
import threading
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from .buffers import SLOT_KEY, retire_message
from .channels import (
    SHUTDOWN_TIMEOUT,
    BufferedSender,
    InFlightQueue,
    ready_event,
    record_stream,
)
from .interfaces import PipelineFailure, StageAdapter
from .memory import ByteQueue, storage_bytes
from .topology import PipelineTopology


class PipelineExecutor:
    """Cache hits stay with their owner; miss results return by owner and ticket."""

    def __init__(self, stage: StageAdapter, topology: PipelineTopology, channels):
        self.stage, self.topology, self.channels = stage, topology, channels
        self.memory_pool = next(
            (
                c.memory_pool
                for c in channels.values()
                if getattr(c, "memory_pool", None) is not None
            ),
            None,
        )
        self.rank = stage.group.rank_in_group
        if self.rank >= topology.ingress_replicas:
            raise ValueError("Only ingress replicas can submit requests")
        self.waiting = {}
        self.next_ticket = 0
        self.receiver = None
        self.sender = None
        self.failure = None
        self.failure_signal = None
        self.closed = False

    def start(self, *, on_capacity=None):
        """Keep the return/control channel live for the entire service lifetime."""
        self._check_open()
        if self.failure_signal is None:
            loop = asyncio.get_running_loop()
            self.failure_signal = loop.create_future()
            if self.topology.edges:
                self.sender = BufferedSender(
                    self.channels[self.rank, self.topology.ingress_replicas],
                    self.topology.buffer_size,
                    on_error=lambda error: loop.call_soon_threadsafe(self._fail, error),
                    on_capacity=(
                        (lambda: loop.call_soon_threadsafe(on_capacity))
                        if on_capacity is not None
                        else None
                    ),
                )
                self.receiver = loop.create_task(self._receive())
        return self.failure_signal

    def _check_open(self):
        if self.failure is not None:
            raise self.failure
        if self.closed:
            raise RuntimeError("Pipeline executor is closed")

    def _fail(self, error):
        if self.failure is None:
            self.failure = PipelineFailure(f"Pipeline communication failed: {error}")
            self.failure.__cause__ = error
            self.failure_signal.set_result(self.failure)
        for future in self.waiting.values():
            if not future.done():
                future.set_exception(self.failure)
        self.waiting.clear()

    def _deliver(self, message):
        if "failure" in message:
            self._fail(RuntimeError(message["failure"]))
            return
        if self.failure is not None:
            return
        for item in message["items"]:
            future = self.waiting[item["ticket"]]
            if item.get("error"):
                future.set_exception(RuntimeError(item["error"]))
            else:
                record_stream(item["hidden"])
                if getattr(future, "memory_lease", None) is not None:
                    future.memory_lease.update(item["hidden"])
                future.set_result(item["hidden"])
            del self.waiting[item["ticket"]]

    async def _receive(self):
        channel = self.channels[self.topology.world_size - 1, self.rank]
        loop = asyncio.get_running_loop()
        credits = (
            threading.Semaphore()
            if self.memory_pool is not None
            else threading.BoundedSemaphore(self.topology.buffer_size)
        )
        feedback = (
            ByteQueue(
                self.topology.buffer_bytes, weight=lambda received: storage_bytes(received.value)
            )
            if self.topology.buffer_bytes
            else None
        )

        def deliver(received):
            try:
                if feedback is not None:
                    assert feedback.get_nowait() is received
                message = received.on_current_stream()
                if message.get("stop"):
                    if not self.closed or self.waiting:
                        self._fail(RuntimeError("Suffix stopped before ingress closed and drained"))
                else:
                    if SLOT_KEY in message:
                        for item in message.get("items", ()):
                            if item.get("hidden") is not None:
                                item["hidden"] = item["hidden"].clone()
                    self._deliver(message)
            except Exception as error:
                # Continue draining after a protocol error so peers can STOP.
                self._fail(error)
            finally:
                if isinstance(received.value, dict) and SLOT_KEY in received.value:
                    retire_message(received.value, ready_event(channel.device))
                if feedback is not None:
                    feedback.task_done()
                elif self.memory_pool is None:
                    credits.release()

        def receive_loop():
            while True:
                # Pending deliveries count against capacity until the owner
                # consumes them. I/O can advance independently of its event loop.
                if feedback is None and self.memory_pool is None:
                    credits.acquire()
                try:
                    received = channel.recv()
                    if received.ready is not None:
                        # Never insert an unfinished return's dependency ahead
                        # of unrelated computation on the owner's CUDA stream.
                        received.ready.synchronize()
                except Exception as error:
                    if feedback is None and self.memory_pool is None:
                        credits.release()
                    loop.call_soon_threadsafe(self._fail, error)
                    return
                stop = isinstance(received.value, dict) and received.value.get("stop")
                if feedback is not None:
                    try:
                        feedback.put(received, timeout=SHUTDOWN_TIMEOUT)
                    except Exception as error:
                        loop.call_soon_threadsafe(self._fail, error)
                        return
                loop.call_soon_threadsafe(deliver, received)
                del received  # The callback owns it, not the next blocking receive.
                if stop:
                    return

        await asyncio.to_thread(receive_loop)

    @property
    def can_submit(self):
        """The sole publisher checks capacity before exporting a miss batch."""
        return self.sender is None or not self.sender.queue.full()

    def can_submit_prepared(self, prepared):
        if self.sender is None or not isinstance(self.sender.queue, ByteQueue):
            return self.can_submit
        # H3 export preserves row shapes. Charge the entire prepared payload
        # conservatively; model modules are excluded by storage_bytes.
        return self.sender.queue.can_put([p.payload for p in prepared])

    def submit(self, prepared):
        """Launch one physical batch and return independent completion futures."""
        self.start()
        if not prepared:
            raise ValueError("Cannot submit an empty physical batch")
        if self.topology.buffer_bytes is not None and len(prepared) != 1:
            raise ValueError("Byte buffering currently requires ingress BS1")
        if not prepared[0].hit and not self.can_submit:
            raise BufferError("Reserve send capacity before computing another miss batch")
        self.stage.validate(prepared)
        if any(p.hit != prepared[0].hit for p in prepared):
            raise ValueError("A physical batch must have one cache route")
        loop = asyncio.get_running_loop()
        if prepared[0].hit:
            with torch.profiler.record_function("monowire/continuous/cache_hit"):
                outputs = self.stage.reuse(prepared)
            if len(outputs) != len(prepared):
                raise RuntimeError("Stage reuse changed the request count")
            futures = [loop.create_future() for _ in outputs]
            for future, output in zip(futures, outputs, strict=True):
                future.set_result(output)
            return futures
        with self.memory_pool.workspace() if self.memory_pool else nullcontext():
            items = self.stage.export_requests(prepared)
        if len(items) != len(prepared):
            raise RuntimeError("Stage export changed the request count")
        futures = [loop.create_future() for _ in items]
        for prepared_step, item, future in zip(prepared, items, futures, strict=True):
            lease = getattr(prepared_step, "memory_lease", None)
            if lease is not None:
                lease.update(item)
                future.memory_lease = lease
        if not self.topology.edges:
            for future, item in zip(futures, items, strict=True):
                future.set_result(item["hidden"])
            return futures
        for item, future in zip(items, futures, strict=True):
            ticket = self.next_ticket
            self.next_ticket += 1
            self.waiting[ticket] = future
            item["ticket"] = ticket
        try:
            self.sender.submit_nowait({"items": items}, ready_event(self.stage.group.device))
        except Exception as error:
            self._fail(error)
        return futures

    async def close(self):
        if self.closed:
            return
        if self.waiting:
            raise RuntimeError("Drain requests before closing ingress")
        if self.failure_signal is None:
            self.start()
        self.closed = True
        if not self.topology.edges:
            return
        # The sender owns every published packet and drains it before STOP.
        try:
            await asyncio.to_thread(self.sender.close)
        finally:
            await self.receiver
        if self.failure is not None:
            raise self.failure


@dataclass
class _ReceivedPacket:
    """An input message and the capacity credit held until its last item is consumed."""

    owner: int
    message: dict
    credit: threading.BoundedSemaphore | None
    ready: torch.cuda.Event | None = None

    def release(self):
        if self.credit is not None:
            self.credit.release()
            self.credit = None


class _ByteCredit:
    def __init__(self, incoming, size):
        self.incoming, self.size = incoming, size

    def release(self):
        self.incoming.task_done(self.size)


def _receive_packets(channel, incoming, owner, draining, capacity, *, receive_prefetch=2):
    byte_mode = isinstance(incoming, ByteQueue)
    dynamic = getattr(channel, "memory_pool", None) is not None
    # Receive lookahead and storage lifetime are separate budgets. Deep posted
    # NCCL receives can wait on future upstream computation and delay first-use
    # CUDA calls. Return this credit on consumption, not on GPU slot retirement:
    # consumed packets may remain resident under the byte ledger until last use.
    credits = (
        None if byte_mode else threading.BoundedSemaphore(receive_prefetch if dynamic else capacity)
    )
    while True:
        credit = None
        # Reserve before receiving tensor storage. Moving a packet from the
        # inbox to pending does not return its credit; consuming it does.
        while not draining.is_set():
            if credits is None:
                break
            if credits.acquire(timeout=0.1):
                credit = credits
                break
        try:
            received = channel.recv()
            message, ready = received.value, received.ready
            del received  # Transfer ownership to the packet below.
            stop = message.get("stop")
        except Exception as error:
            message, stop, ready = {"failure": str(error)}, True, None
        packet = _ReceivedPacket(owner, message, credit, ready)
        if byte_mode:
            packet.credit = _ByteCredit(incoming, storage_bytes(message))
        while not draining.is_set():
            try:
                incoming.put(packet, timeout=0.1)
                packet = None  # The actor now owns the receive credit.
                break
            except queue.Full:
                pass
        if packet is not None:
            retire_message(packet.message, packet.ready)
            if byte_mode:
                packet.credit = None  # Never entered the queue, so no credit to return.
            packet.release()
        del packet, message
        if stop:
            return


def serve_stage(
    stage: StageAdapter,
    topology: PipelineTopology,
    channels,
    *,
    batch_size: int,
    on_event=None,
    async_transport=True,
):
    """Serve a suffix stage until all upstream peers have drained and sent STOP.

    Packing and model errors fail only the affected requests. Transport failures
    stop the service and notify downstream peers before draining the links.
    """
    _StageService(
        stage, topology, channels, batch_size, on_event, async_transport=async_transport
    ).run()


class _StageService:
    """Own one suffix actor's receive credits, batching and I/O worker lifetime."""

    def __init__(self, stage, topology, channels, batch_size, on_event, *, async_transport=True):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Downstream batch size must be a positive integer")
        self.stage = stage
        self.memory_pool = next(
            (
                c.memory_pool
                for c in channels.values()
                if getattr(c, "memory_pool", None) is not None
            ),
            None,
        )
        self.rank = stage.group.rank_in_group
        self.first_rank = topology.ingress_replicas
        self.last_rank = topology.world_size - 1
        if self.rank < self.first_rank:
            raise ValueError("Ingress owns a request scheduler")
        self.batch_size = batch_size
        if topology.buffer_bytes is not None and batch_size != 1:
            raise ValueError("Byte buffering currently requires downstream BS1")
        self.on_event = on_event
        self.async_transport = async_transport
        self.sources = (
            list(range(self.first_rank)) if self.rank == self.first_rank else [self.rank - 1]
        )
        destinations = (
            list(range(self.first_rank)) if self.rank == self.last_rank else [self.rank + 1]
        )
        self.incoming = (
            queue.Queue()
            if self.memory_pool is not None
            else ByteQueue(
                topology.buffer_bytes, weight=lambda packet: storage_bytes(packet.message)
            )
            if topology.buffer_bytes
            else queue.Queue(maxsize=topology.buffer_size * len(self.sources))
        )
        self.draining = threading.Event()
        self.receivers = [
            threading.Thread(
                target=_receive_packets,
                args=(
                    channels[source, self.rank],
                    self.incoming,
                    source,
                    self.draining,
                    topology.buffer_size,
                ),
                kwargs={"receive_prefetch": topology.receive_prefetch},
                daemon=True,
            )
            for source in self.sources
        ]
        self.senders = {
            destination: BufferedSender(channels[self.rank, destination], topology.buffer_size)
            for destination in destinations
        }
        self.stopped = set()
        self.pending = deque()
        self.computations = InFlightQueue(
            topology.buffer_size, byte_capacity=topology.buffer_bytes, memory_pool=self.memory_pool
        )

    def run(self):
        for thread in self.receivers:
            thread.start()
        failure = None
        try:
            while len(self.stopped) < len(self.sources) or self.pending:
                self.computations.reserve()
                envelope = self._next_batch()
                if envelope is not None:
                    self._process_batch(envelope)
                del envelope
        except BaseException as error:
            failure = error
            self._notify_failure(error)
        finally:
            cleanup_errors = self._close()
            if failure is not None:
                raise failure
            if cleanup_errors:
                raise ExceptionGroup("Pipeline teardown failed", cleanup_errors)

    def _receive(self, *, block=True):
        """Handle control messages; retain entrance packets until they are consumed."""
        while True:
            for sender in self.senders.values():
                sender.check()
            try:
                received = self.incoming.get(timeout=0.1) if block else self.incoming.get_nowait()
                break
            except queue.Empty:
                if not block:
                    raise
        try:
            message = received.message
            if "failure" in message:
                retire_message(message, received.ready)
                raise RuntimeError(message["failure"])
            if message.get("stop"):
                retire_message(message, received.ready)
                self.stopped.add(received.owner)
                return None
            if self.rank == self.first_rank:
                if not message["items"]:
                    raise ValueError("Empty pipeline input packet")
                for item in message["items"]:
                    item["owner"] = received.owner
                self.pending.append(received)
                received = None  # Keep the credit until every item is consumed.
            elif received.ready is not None:
                received.ready.wait()
            return message
        except BaseException:
            if received is not None:
                retire_message(received.message, received.ready)
            raise
        finally:
            if received is not None:
                received.release()

    def _next_batch(self):
        if self.rank != self.first_rank:
            envelope = self._receive()
            if envelope is not None:
                record_stream(envelope.get("payload"))
            return envelope

        if not self.pending:
            self._receive()
        # Collect only arrivals already in the inbox; do not wait for a full batch.
        # Inbox and pending share credits, so merging cannot grow an unbounded queue.
        for _ in range(self.incoming.qsize()):
            self._receive(block=False)
        if not self.pending:
            return None
        return self._pack_pending_batch()

    def _pack_pending_batch(self):
        """Merge compatible requests across owners, preserving each packet's credit."""
        # Choose the oldest GPU-ready packet, not merely the oldest descriptor.
        # Once an older transfer completes it regains priority on the next pass.
        available = [p for p in self.pending if p.ready is None or p.ready.query()]
        selected_packet = available[0] if available else self.pending[0]
        compatibility = selected_packet.message["items"][0]["compatibility"]
        items = []
        uses = []
        for _ in range(len(self.pending)):
            received = self.pending.popleft()
            if available and received.ready is not None and not received.ready.query():
                self.pending.append(received)
                continue
            remaining = []
            selected = False
            for item in received.message["items"]:
                if item["compatibility"] == compatibility and len(items) < self.batch_size:
                    items.append(item)
                    selected = True
                else:
                    remaining.append(item)
            if selected and received.ready is not None:
                received.ready.wait()
                received.ready = None  # Later slices use the same compute stream.
            received.message["items"] = remaining
            if remaining:
                if selected and SLOT_KEY in received.message:
                    raise ValueError("Reusable receive storage currently requires one-item packets")
                self.pending.append(received)
            else:
                uses.extend(received.message.pop(SLOT_KEY, ()))
                received.release()
        record_stream(items)
        routes = [(item["owner"], item["ticket"], item["hidden"].shape[0]) for item in items]
        envelope = {"routes": routes}
        if uses:
            envelope[SLOT_KEY] = uses
        try:
            with self.memory_pool.workspace() if self.memory_pool else nullcontext():
                envelope["payload"] = self.stage.combine_requests(items)
        except Exception as error:
            envelope["error"] = f"Pipeline stage {self.rank} packing: {error}"
        except BaseException:
            retire_message(envelope, ready_event(self.stage.group.device))
            raise
        return envelope

    def _process_batch(self, envelope):
        try:
            self._execute_batch(envelope)
        except BaseException:
            # Until submission succeeds this actor owns the slot. Fatal model
            # aborts and result-splitting errors bypass the normal send path,
            # but queued compute must still finish before storage is recycled.
            retire_message(envelope, ready_event(self.stage.group.device))
            raise

    def _execute_batch(self, envelope):
        if self.computations.byte_capacity is not None:
            self.computations.reserve(storage_bytes(envelope))
        packet = envelope.get("payload")
        routes = envelope["routes"]
        if self.on_event is not None:
            try:
                self.on_event(
                    {
                        "event": "suffix_batch",
                        "time": time.monotonic(),
                        "rank": self.rank,
                        "routes": routes,
                        "batch": len(routes),
                    }
                )
            except Exception:
                logging.getLogger("monowire").exception("Stage event observer failed")
        try:
            if not envelope.get("error"):
                try:
                    with self.memory_pool.workspace() if self.memory_pool else nullcontext():
                        self.stage.compute(packet)
                except Exception as error:
                    envelope["error"] = f"Pipeline stage {self.rank}: {error}"
        finally:
            ready = ready_event(self.stage.group.device)
            self.computations.track(ready, envelope)
        if self.rank == self.last_rank:
            self._return_results(envelope, ready)
        else:
            try:
                self.senders[self.rank + 1].submit(envelope, ready)
            except BaseException:
                retire_message(envelope, ready)
                raise
        if not self.async_transport:
            self.computations.drain()

    def _return_results(self, envelope, ready):
        """Split the final hidden states back into per-owner completion messages."""
        routes = envelope["routes"]
        error = envelope.get("error")
        if error:
            outputs = [None] * len(routes)
        else:
            lengths = [length for _, _, length in routes]
            outputs = envelope["payload"]["hidden"].split(lengths)
        grouped = defaultdict(list)
        for (owner, ticket, _), output in zip(routes, outputs, strict=True):
            grouped[owner].append({"ticket": ticket, "hidden": output, "error": error})
        if SLOT_KEY in envelope and len(grouped) != 1:
            raise ValueError("Reusable receive storage requires one ingress owner")
        for owner, items in grouped.items():
            message = {"items": items}
            if SLOT_KEY in envelope:
                message[SLOT_KEY] = envelope.pop(SLOT_KEY)
            try:
                self.senders[owner].submit(message, ready)
            except BaseException:
                retire_message(message, ready)
                raise

    def _notify_failure(self, error):
        for sender in self.senders.values():
            try:
                sender.submit({"failure": f"Pipeline stage {self.rank}: {error}"})
            except Exception:
                logging.getLogger("monowire").exception("Failed to notify downstream peer")

    def _close(self):
        self.draining.set()
        while self.pending:
            packet = self.pending.popleft()
            retire_message(packet.message, packet.ready)
            packet.release()
        # Owners seeing a service failure stop submitting, drain published work
        # and send STOP. Receivers discard queued work until those STOPs arrive.
        deadline = time.monotonic() + SHUTDOWN_TIMEOUT
        errors = []
        # A blocked receiver may need slots owned by incoming packets. Retire
        # them before joining, and keep draining racing puts until receivers exit.
        # Retirement still waits for the packet's CUDA completion event.
        while True:
            self._discard_incoming()
            alive = [thread for thread in self.receivers if thread.is_alive()]
            remaining = deadline - time.monotonic()
            if not alive or remaining <= 0:
                break
            alive[0].join(timeout=min(remaining, 0.01))
        for thread in self.receivers:
            if thread.is_alive():
                errors.append(TimeoutError("Pipeline receiver did not stop"))
        self._discard_incoming()
        try:
            self.computations.drain()
        except Exception as error:
            errors.append(error)
        for sender in self.senders.values():
            try:
                sender.close()
            except Exception as error:
                errors.append(error)
        return errors

    def _discard_incoming(self):
        while True:
            try:
                packet = self.incoming.get_nowait()
                retire_message(packet.message, packet.ready)
                packet.release()
            except queue.Empty:
                break
