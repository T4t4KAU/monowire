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
        return "dict", [(key, _encode(item, tensors, dictionary)) for key, item in value.items()]
    if isinstance(value, (tuple, list)):
        return "tuple" if isinstance(value, tuple) else "list", [
            _encode(item, tensors, dictionary) for item in value
        ]
    return "value", value


def _decode(spec, tensors, device, dictionary=None):
    tag = spec[0]
    if tag == "parts":
        return TensorParts(tuple(_decode(part, tensors, device, dictionary) for part in spec[1]))
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
        tensor = torch.empty(shape, dtype=dtype, device=device if cuda else "cpu")
        tensors.append((index, tensor))
        return tensor
    if tag == "dict":
        return {key: _decode(item, tensors, device, dictionary) for key, item in spec[1]}
    if tag in ("tuple", "list"):
        values = [_decode(item, tensors, device, dictionary) for item in spec[1]]
        return tuple(values) if tag == "tuple" else values
    if tag == "value":
        return spec[1]
    raise ValueError(f"Unknown tensor message tag: {tag}")


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

    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("In-flight capacity must be a positive integer")
        self.capacity = capacity
        self.pending = deque()

    def reserve(self):
        self.reap()
        if len(self.pending) >= self.capacity:
            self.pending[0][0].synchronize()
            self.pending.popleft()

    def reap(self):
        while self.pending and self.pending[0][0].query():
            self.pending.popleft()

    def track(self, event, resources):
        if event is not None:
            self.pending.append((event, resources))

    def drain(self):
        while self.pending:
            self.pending[0][0].synchronize()
            self.pending.popleft()


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
    ):
        self.source, self.destination = source, destination
        self.cpu_group, self.gpu_group = cpu_group, gpu_group
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=device) if self.device.type == "cuda" else None
        self.inflight = InFlightQueue(capacity)
        self.async_transport = async_transport
        self.dictionary = TensorDictionary()

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
        self.inflight.reserve()
        tensors = []
        spec = _encode(message, tensors, self.dictionary)
        dist.send_object_list([spec], dst=self.destination, group=self.cpu_group)
        buffers, works = [], []
        with torch.inference_mode(), self._context():
            if ready is not None:
                self.stream.wait_event(ready)
            try:
                buffers = [tensor.contiguous() for tensor in tensors]
                self._transfer(buffers, dist.isend, self.destination, works)
            finally:
                self._finish_transfer(ready_event(self.device), (message, tensors, buffers, works))

    def recv(self):
        self.inflight.reserve()
        metadata = [None]
        while metadata[0] is None:
            # A raw None is an idle heartbeat, distinct from an encoded message.
            dist.recv_object_list(metadata, src=self.source, group=self.cpu_group)
        tensors, works = [], []
        with torch.inference_mode(), self._context():
            message = _decode(metadata[0], tensors, self.device, self.dictionary)
            try:
                self._transfer(
                    [tensor for _, tensor in sorted(tensors)], dist.irecv, self.source, works
                )
            finally:
                done = ready_event(self.device)
                self._finish_transfer(done, (message, tensors, works))
        return ReceivedMessage(message, done if self.async_transport else None)

    def drain(self):
        """Finish transfers before dropping storage or destroying process groups."""
        self.inflight.drain()

    def heartbeat(self):
        """Keep an idle edge live without allocating CUDA tensors or events."""
        self.inflight.reap()
        dist.send_object_list([None], dst=self.destination, group=self.cpu_group)


def create_channels(topology, device, *, async_transport=True):
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
            )
    return result


def close_channels(channels):
    errors = []
    for channel in channels.values():
        try:
            channel.drain()
            channel.dictionary.clear()
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
        self.queue = queue.Queue(maxsize=capacity)
        self.on_error = on_error
        self.on_capacity = on_capacity
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError("Pipeline sender failed") from self.error

    def submit(self, message, ready=None):
        deadline = time.monotonic() + SHUTDOWN_TIMEOUT
        while True:
            self.check()
            try:
                self.queue.put((message, ready), timeout=0.1)
                return
            except queue.Full:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Pipeline send queue did not drain") from None

    def submit_nowait(self, message, ready=None):
        """Publish from the sole ingress owner after checking queue capacity."""
        self.check()
        self.queue.put_nowait((message, ready))

    def _run(self):
        try:
            while True:
                try:
                    message, ready = self.queue.get(timeout=HEARTBEAT_INTERVAL)
                except queue.Empty:
                    self.channel.heartbeat()
                    continue
                if self.on_capacity is not None:
                    self.on_capacity()
                if message is None:
                    self.channel.drain()
                    self.channel.send({"stop": True})
                    self.channel.drain()
                    return
                self.channel.send(message, ready)
                # The worker may wait a long time for the next message.
                del message, ready
        except Exception as error:
            self.error = error
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
