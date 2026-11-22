"""Bounded point-to-point transport with independent CUDA communication streams."""

import queue
import threading
import time
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

CHANNEL_TIMEOUT = timedelta(seconds=180)
HEARTBEAT_INTERVAL = 30.0
SHUTDOWN_TIMEOUT = CHANNEL_TIMEOUT.total_seconds() + 10


def _encode(value, tensors):
    if isinstance(value, torch.Tensor):
        index = len(tensors)
        tensors.append(value)
        return "tensor", index, tuple(value.shape), value.dtype, value.is_cuda
    if isinstance(value, dict):
        return "dict", [(key, _encode(item, tensors)) for key, item in value.items()]
    if isinstance(value, (tuple, list)):
        return "tuple" if isinstance(value, tuple) else "list", [
            _encode(item, tensors) for item in value
        ]
    return "value", value


def _decode(spec, tensors, device):
    tag = spec[0]
    if tag == "tensor":
        _, index, shape, dtype, cuda = spec
        tensor = torch.empty(shape, dtype=dtype, device=device if cuda else "cpu")
        tensors.append((index, tensor))
        return tensor
    if tag == "dict":
        return {key: _decode(item, tensors, device) for key, item in spec[1]}
    if tag in ("tuple", "list"):
        values = [_decode(item, tensors, device) for item in spec[1]]
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


class TensorChannel:
    """One directed edge; a single sender serializes data and idle heartbeats."""

    def __init__(self, source, destination, cpu_group, gpu_group, device):
        self.source, self.destination = source, destination
        self.cpu_group, self.gpu_group = cpu_group, gpu_group
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=device) if self.device.type == "cuda" else None

    def _context(self):
        if self.stream is None:
            return nullcontext()
        torch.cuda.set_device(self.device)
        return torch.cuda.stream(self.stream)

    def send(self, message, ready=None):
        tensors = []
        spec = _encode(message, tensors)
        dist.send_object_list([spec], dst=self.destination, group=self.cpu_group)
        with torch.inference_mode(), self._context():
            if ready is not None:
                self.stream.wait_event(ready)
            buffers = [tensor.contiguous() for tensor in tensors]
            for tensor in buffers:
                group = self.gpu_group if tensor.is_cuda else self.cpu_group
                dist.isend(tensor, dst=self.destination, group=group).wait()
            done = ready_event(self.device)
        # Retain send buffers until this edge has consumed them. No device-wide wait.
        if done is not None:
            done.synchronize()

    def recv(self):
        metadata = [None]
        while metadata[0] is None:
            # A raw None is an idle heartbeat, distinct from an encoded message.
            # Keep waiting without turning an idle service into a Gloo timeout.
            dist.recv_object_list(metadata, src=self.source, group=self.cpu_group)
        with torch.inference_mode(), self._context():
            tensors = []
            message = _decode(metadata[0], tensors, self.device)
            for _, tensor in sorted(tensors):
                group = self.gpu_group if tensor.is_cuda else self.cpu_group
                dist.irecv(tensor, src=self.source, group=group).wait()
            done = ready_event(self.device)
        if done is not None:
            done.synchronize()
        return message

    def heartbeat(self):
        """Keep an idle edge live without allocating CUDA tensors or events."""
        dist.send_object_list([None], dst=self.destination, group=self.cpu_group)


def create_channels(topology, device):
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
            result[source, destination] = TensorChannel(source, destination, cpu, gpu, device)
    return result


def close_channels(channels):
    for channel in channels.values():
        if channel.gpu_group is not None:
            dist.destroy_process_group(channel.gpu_group)
        dist.destroy_process_group(channel.cpu_group)


class BufferedSender:
    """Own an edge's sends, heartbeats and final STOP on one I/O thread."""

    def __init__(self, channel, capacity, *, on_error=None):
        self.channel = channel
        self.queue = queue.Queue(maxsize=capacity)
        self.on_error = on_error
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

    def _run(self):
        try:
            while True:
                try:
                    message, ready = self.queue.get(timeout=HEARTBEAT_INTERVAL)
                except queue.Empty:
                    self.channel.heartbeat()
                    continue
                if message is None:
                    self.channel.send({"stop": True})
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
