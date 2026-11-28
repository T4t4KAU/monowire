"""Bounded point-to-point transport with independent CUDA communication streams."""

import queue
import threading
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, fields, is_dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

from .buffers import SLOT_KEY, ReceiveBuffers, retire_message
from .memory import ByteQueue, storage_bytes
from .tensor_payload import SharedTensor, TensorDictionary, TensorParts

CHANNEL_TIMEOUT = timedelta(seconds=180)
HEARTBEAT_INTERVAL = 30.0
SHUTDOWN_TIMEOUT = CHANNEL_TIMEOUT.total_seconds() + 10


def _encode(value, tensors, dictionary=None):
    if isinstance(value, TensorParts):
        return "parts", [_encode(part, tensors, dictionary) for part in value.parts]
    if isinstance(value, SharedTensor):
        cached = dictionary.get(value.key) if dictionary is not None else None
        if cached is not None:
            return "shared_ref", value.key
        spec = _encode(value.value, tensors, dictionary)
        if dictionary is not None:
            dictionary.add(value)
        return "shared", value.key, spec
    if isinstance(value, torch.Tensor):
        index = len(tensors)
        tensors.append(value)
        return "tensor", index, tuple(value.shape), value.dtype, value.is_cuda
    if isinstance(value, dict):
        return "dict", [
            (key, _encode(item, tensors, dictionary))
            for key, item in value.items()
            if key != SLOT_KEY
        ]
    if isinstance(value, (tuple, list)):
        return "tuple" if isinstance(value, tuple) else "list", [
            _encode(item, tensors, dictionary) for item in value
        ]
    return "value", value


def _decode(spec, tensors, device, dictionary=None, buffers=None):
    tag = spec[0]
    if tag == "parts":
        return TensorParts(
            tuple(_decode(part, tensors, device, dictionary, buffers) for part in spec[1])
        )
    if tag == "shared_ref":
        shared = dictionary.get(spec[1]) if dictionary is not None else None
        if shared is None:
            raise ValueError("Missing shared tensor declaration")
        return shared
    if tag == "shared":
        shared = SharedTensor(spec[1], _decode(spec[2], tensors, device, dictionary))
        if dictionary is not None:
            dictionary.add(shared)
        return shared
    if tag == "tensor":
        _, index, shape, dtype, cuda = spec
        tensor = buffers.get(index) if buffers is not None else None
        if tensor is None:
            tensor = torch.empty(shape, dtype=dtype, device=device if cuda else "cpu")
        tensors.append((index, tensor))
        return tensor
    if tag == "dict":
        return {key: _decode(item, tensors, device, dictionary, buffers) for key, item in spec[1]}
    if tag in ("tuple", "list"):
        values = [_decode(item, tensors, device, dictionary, buffers) for item in spec[1]]
        return tuple(values) if tag == "tuple" else values
    if tag == "value":
        return spec[1]
    raise ValueError(f"Unknown tensor message tag: {tag}")


def _spec_bytes(spec, dictionary, declared=None):
    """Conservative decoded footprint, evaluated before allocating tensors."""
    tag = spec[0]
    declared = {} if declared is None else declared
    if tag == "tensor":
        import math

        return math.prod(spec[2]) * torch.empty((), dtype=spec[3]).element_size()
    if tag == "shared_ref":
        if spec[1] in declared:
            return 0  # This message already owns the referenced storage.
        shared = dictionary.entries.get(spec[1])
        if shared is None:
            raise ValueError("Missing shared tensor declaration")
        size = storage_bytes(shared)
        declared[spec[1]] = size
        return size
    if tag == "shared":
        size = _spec_bytes(spec[2], dictionary, declared)
        declared[spec[1]] = size
        return size
    if tag == "dict":
        return sum(_spec_bytes(value, dictionary, declared) for _, value in spec[1])
    if tag in ("list", "tuple", "parts"):
        return sum(_spec_bytes(value, dictionary, declared) for value in spec[1])
    return 0


def _spec_allocation_bytes(spec):
    """New CUDA allocation only; cached shared references allocate no storage."""
    import math

    tag = spec[0]
    if tag == "tensor":
        return math.prod(spec[2]) * torch.empty((), dtype=spec[3]).element_size() if spec[4] else 0
    if tag == "shared":
        return _spec_allocation_bytes(spec[2])
    if tag == "dict":
        return sum(_spec_allocation_bytes(child) for _, child in spec[1])
    if tag in ("list", "tuple", "parts"):
        return sum(_spec_allocation_bytes(child) for child in spec[1])
    return 0


def _receive_layout(spec, device, *, shared=False):
    """Non-shared device tensors only; dictionaries outlive individual messages."""
    tag = spec[0]
    if tag == "tensor":
        if not shared and (spec[4] or torch.device(device).type == "cpu"):
            return [(spec[1], spec[2], spec[3])]
        return []
    if tag == "shared":
        return []
    if tag == "dict":
        return [item for _, child in spec[1] for item in _receive_layout(child, device)]
    if tag in ("list", "tuple", "parts"):
        return [item for child in spec[1] for item in _receive_layout(child, device)]
    return []


def _shared_allocation_bytes(spec):
    if spec[0] == "shared":
        return _spec_allocation_bytes(spec[2])
    if spec[0] == "dict":
        return sum(_shared_allocation_bytes(child) for _, child in spec[1])
    if spec[0] in ("list", "tuple", "parts"):
        return sum(_shared_allocation_bytes(child) for child in spec[1])
    return 0


def record_stream(value):
    """Keep received storage alive until the compute stream finishes using it."""
    if isinstance(value, torch.Tensor) and value.is_cuda:
        value.record_stream(torch.cuda.current_stream(value.device))
    elif isinstance(value, dict):
        for item in value.values():
            record_stream(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            record_stream(item)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            record_stream(getattr(value, field.name))


def ready_event(device):
    if torch.device(device).type != "cuda":
        return None
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(device))
    return event


class InFlightQueue:
    """Retain CUDA work and its storage; wait only at capacity or during teardown.

    One host thread owns the queue. Reserve a slot before allocating or launching
    work, then track its completion event and any objects that must stay alive.
    """

    def __init__(self, capacity, *, byte_capacity=None, memory_pool=None):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("In-flight capacity must be a positive integer")
        self.capacity = capacity
        self.memory_pool = memory_pool
        self.byte_capacity = byte_capacity
        self._bytes = deque()
        self.used_bytes = 0
        self.peak_bytes = 0
        self.pending = deque()

    def reserve(self, size=0):
        if self.memory_pool is not None:
            with self.memory_pool.lock:
                self.memory_pool._reap()
            return
        self.reap()
        if self.byte_capacity is not None and size > self.byte_capacity:
            raise ValueError(
                f"Payload needs {size} bytes; in-flight budget is {self.byte_capacity}"
            )
        while self.pending and (
            self.used_bytes + size > self.byte_capacity
            if self.byte_capacity is not None
            else len(self.pending) >= self.capacity
        ):
            self.pending[0][0].synchronize()
            self.pending.popleft()
            self.used_bytes -= self._bytes.popleft()

    def reap(self):
        while self.pending and self.pending[0][0].query():
            self.pending.popleft()
            self.used_bytes -= self._bytes.popleft()

    def track(self, event, resources):
        if self.memory_pool is not None:
            self.memory_pool.retain(event, resources)
            return
        if event is not None:
            size = storage_bytes(resources) if self.byte_capacity is not None else 0
            if self.byte_capacity is not None:
                self.reserve(size)
            self.pending.append((event, resources))
            self._bytes.append(size)
            self.used_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.used_bytes)

    def drain(self):
        if self.memory_pool is not None:
            self.memory_pool.drain()
            return
        while self.pending:
            self.pending[0][0].synchronize()
            self.pending.popleft()
            self.used_bytes -= self._bytes.popleft()


@dataclass
class ReceivedMessage:
    """A decoded message whose CUDA tensors may still be arriving."""

    value: dict
    ready: torch.cuda.Event | None = None

    def on_current_stream(self):
        """Order consumption after reception without blocking the host thread."""
        if self.ready is not None:
            self.ready.wait()
        record_stream(self.value)
        return self.value


class TensorChannel:
    """One directed edge with bounded, asynchronous CUDA transfers."""

    def __init__(
        self,
        source,
        destination,
        cpu_group,
        gpu_group,
        device,
        *,
        capacity=2,
        async_transport=True,
        byte_capacity=None,
        memory_pool=None,
        feedback=False,
    ):
        self.source, self.destination = source, destination
        self.cpu_group, self.gpu_group = cpu_group, gpu_group
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=device) if self.device.type == "cuda" else None
        self.memory_pool, self.feedback = memory_pool, feedback
        self.inflight = InFlightQueue(
            capacity, byte_capacity=byte_capacity, memory_pool=memory_pool
        )
        self.async_transport = async_transport
        self.dictionary = TensorDictionary()
        self.receive_buffers = None

    def _context(self):
        if self.stream is None:
            return nullcontext()
        torch.cuda.set_device(self.device)
        return torch.cuda.stream(self.stream)

    def _transfer(self, tensors, operation, peer, works):
        if not self.async_transport:
            for tensor in tensors:
                group = self.gpu_group if tensor.is_cuda else self.cpu_group
                work = operation(tensor, peer, group=group)
                works.append(work)
                work.wait()
            return
        # Post every operation before waiting. CUDA P2P operations are grouped
        # into one launch; CPU tensors retain Gloo's host-completion semantics.
        errors = []
        try:
            for cuda, group in ((False, self.cpu_group), (True, self.gpu_group)):
                operations = [
                    dist.P2POp(operation, tensor, peer, group)
                    for tensor in tensors
                    if tensor.is_cuda == cuda
                ]
                if operations:
                    works.extend(dist.batch_isend_irecv(operations))
        except Exception as error:
            errors.append(error)
        # Join every posted operation, even if another launch or wait failed.
        # Otherwise the completion event could precede outstanding transfers.
        for work in works:
            try:
                work.wait()  # NCCL orders this stream; Gloo waits for CPU data.
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Tensor transfer failed", errors)

    def _finish_transfer(self, done, resources):
        self.inflight.track(done, resources)
        if not self.async_transport:
            self.inflight.drain()

    def send(self, message, ready=None):
        # A non-contiguous tensor may require a contiguous wire copy.
        self.inflight.reserve(2 * storage_bytes(message) if self.inflight.byte_capacity else 0)
        tensors = []
        spec = _encode(message, tensors, self.dictionary)
        dist.send_object_list([spec], dst=self.destination, group=self.cpu_group)
        lease = None
        if self.memory_pool is not None:
            approval = [None]
            dist.recv_object_list(approval, src=self.destination, group=self.cpu_group)
            if approval[0] is not True:
                raise RuntimeError(f"Downstream memory admission rejected: {approval[0]}")
            copies = sum(
                t.numel() * t.element_size() for t in tensors if t.is_cuda and not t.is_contiguous()
            )
            if copies > self.memory_pool.capacities["wire"]:
                raise ValueError("Wire copies exceed the configured progress margin")
            lease = self.memory_pool.reserve(copies, purpose="wire")
        buffers, works = [], []
        with torch.inference_mode(), self._context():
            if ready is not None:
                self.stream.wait_event(ready)
            try:
                buffers = [tensor.contiguous() for tensor in tensors]
                self._transfer(buffers, dist.isend, self.destination, works)
            finally:
                done = ready_event(self.device)
                if lease is not None:
                    self.memory_pool.retain(done, None, lease)
                self._finish_transfer(done, (message, tensors, buffers, works))
                retire_message(message, done)

    def recv(self):
        if self.inflight.byte_capacity is None:
            self.inflight.reserve()
        metadata = [None]
        while metadata[0] is None:
            # A raw None is an idle heartbeat, distinct from an encoded message.
            dist.recv_object_list(metadata, src=self.source, group=self.cpu_group)
        if self.inflight.byte_capacity is not None:
            self.inflight.reserve(_spec_bytes(metadata[0], self.dictionary))
        lease = None
        if self.memory_pool is not None:
            try:
                if self.receive_buffers is None:
                    self.receive_buffers = ReceiveBuffers(self.memory_pool, feedback=self.feedback)
                if _shared_allocation_bytes(metadata[0]) > self.memory_pool.limits.dictionary_bytes:
                    raise ValueError("Shared declarations exceed the dictionary staging reserve")
                layout = _receive_layout(metadata[0], self.device)
                # An evicted shared declaration can still be referenced by a
                # queued packet. Charge its lifetime to that packet, even though
                # its storage is independent of the recyclable arena.
                import math

                local_bytes = sum(
                    math.prod(shape) * torch.empty((), dtype=dtype).element_size()
                    for _, shape, dtype in layout
                )
                retained = max(0, _spec_bytes(metadata[0], self.dictionary) - local_bytes)
                # Allocate on the stream that first writes these buffers. The
                # caching allocator can return storage whose previous uses are
                # still queued on its allocation stream. Allocating on the
                # compute stream and immediately receiving on another stream
                # would race those uses, even for an initially empty tensor.
                with self._context():
                    lease, buffers = self.receive_buffers.acquire(layout, retained_bytes=retained)
            except Exception as error:
                dist.send_object_list([str(error)], dst=self.source, group=self.cpu_group)
                raise
            try:
                dist.send_object_list([True], dst=self.source, group=self.cpu_group)
            except BaseException:
                # Admission owns the slot already, but no tensor receive has
                # been posted. A failed acknowledgement must not orphan it.
                if lease is not None:
                    lease.close()
                raise
        tensors, works = [], []
        success = False
        with torch.inference_mode(), self._context():
            try:
                message = _decode(
                    metadata[0],
                    tensors,
                    self.device,
                    self.dictionary,
                    buffers if self.memory_pool is not None else None,
                )
                self._transfer(
                    [tensor for _, tensor in sorted(tensors)], dist.irecv, self.source, works
                )
                success = True
            finally:
                done = ready_event(self.device)
                self._finish_transfer(done, (locals().get("message"), tensors, works))
                if lease is not None:
                    if success:
                        message[SLOT_KEY] = [lease]
                    else:
                        self.memory_pool.retain(done, None, lease)
        return ReceivedMessage(message, done if self.async_transport else None)

    def drain(self):
        """Finish transfers before dropping storage or destroying process groups."""
        self.inflight.drain()

    def heartbeat(self):
        """Keep an idle edge live without allocating CUDA tensors or events."""
        self.inflight.reap()
        dist.send_object_list([None], dst=self.destination, group=self.cpu_group)


def create_channels(topology, device, *, async_transport=True, memory_pool=None):
    """All world ranks must create the directed edge groups in this order."""
    result = {}
    rank = dist.get_rank()
    for source, destination in topology.edges:
        members = sorted((source, destination))
        cpu = dist.new_group(members, backend="gloo", timeout=CHANNEL_TIMEOUT)
        gpu = (
            dist.new_group(members, backend="nccl", timeout=CHANNEL_TIMEOUT)
            if torch.device(device).type == "cuda"
            else None
        )
        if rank in members:
            result[source, destination] = TensorChannel(
                source,
                destination,
                cpu,
                gpu,
                device,
                capacity=topology.buffer_size,
                async_transport=async_transport,
                byte_capacity=topology.buffer_bytes,
                memory_pool=memory_pool,
                feedback=destination < topology.ingress_replicas,
            )
    return result


def close_channels(channels):
    errors = []
    for channel in channels.values():
        try:
            channel.drain()
            channel.dictionary.clear()
            if channel.receive_buffers is not None:
                channel.receive_buffers.close()
        except Exception as error:
            errors.append(error)
        for group in (channel.gpu_group, channel.cpu_group):
            if group is not None:
                try:
                    dist.destroy_process_group(group)
                except Exception as error:
                    errors.append(error)
    if errors:
        raise ExceptionGroup("Pipeline channel teardown failed", errors)


class BufferedSender:
    """Own an edge's sends, heartbeats and final STOP on one I/O thread."""

    def __init__(self, channel, capacity, *, on_error=None, on_capacity=None):
        self.channel = channel
        byte_capacity = getattr(getattr(channel, "inflight", None), "byte_capacity", None)
        dynamic = getattr(channel, "memory_pool", None) is not None
        self.queue = (
            queue.Queue()
            if dynamic
            else ByteQueue(byte_capacity)
            if byte_capacity
            else queue.Queue(maxsize=capacity)
        )
        self.on_error = on_error
        self.on_capacity = on_capacity
        self.error = None
        self.publication = threading.Condition()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError("Pipeline sender failed") from self.error

    def submit(self, message, ready=None):
        deadline = time.monotonic() + SHUTDOWN_TIMEOUT
        with self.publication:
            while True:
                self.check()
                try:
                    self.queue.put_nowait((message, ready))
                    return
                except queue.Full:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Pipeline send queue did not drain") from None
                    self.publication.wait(min(remaining, 0.1))

    def submit_nowait(self, message, ready=None):
        """Publish from the sole ingress owner after checking queue capacity."""
        with self.publication:
            self.check()
            self.queue.put_nowait((message, ready))

    def _capacity_changed(self):
        with self.publication:
            self.publication.notify_all()
        if self.on_capacity is not None:
            self.on_capacity()

    def _run(self):
        try:
            while True:
                try:
                    message, ready = self.queue.get(timeout=HEARTBEAT_INTERVAL)
                except queue.Empty:
                    self.channel.heartbeat()
                    continue
                self._capacity_changed()
                if message is None:
                    self.channel.drain()
                    self.channel.send({"stop": True})
                    self.channel.drain()
                    if isinstance(self.queue, ByteQueue):
                        self.queue.task_done()
                    return
                self.channel.send(message, ready)
                if isinstance(self.queue, ByteQueue):
                    self.queue.task_done()
                    self._capacity_changed()
                # The worker may wait a long time for the next message.
                del message, ready
        except Exception as error:
            # Publish failure atomically with admission. Otherwise a producer
            # that passed check() can enqueue after the failed worker's final
            # drain, leaving a packet (and its receive slot) without a consumer.
            with self.publication:
                self.error = error
                self.publication.notify_all()
            if isinstance(locals().get("message"), dict):
                retire_message(message, ready)
            while True:
                try:
                    abandoned, completion = self.queue.get_nowait()
                    retire_message(abandoned, completion)
                except queue.Empty:
                    break
            if self.on_error is not None:
                self.on_error(error)

    def close(self):
        try:
            if self.thread.is_alive() and self.error is None:
                self.submit(None)
        finally:
            self.thread.join(timeout=SHUTDOWN_TIMEOUT)
        if self.thread.is_alive():
            raise TimeoutError("Pipeline sender did not drain")
        self.check()
