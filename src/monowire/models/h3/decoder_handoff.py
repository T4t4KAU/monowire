"""Ready-driven CPU handoff to decoder replicas after local DiT STOP/drain.

Only the main thread touches a GPU. Ingress broker threads use dedicated Gloo
pairs, so a decoder may load while other ranks still own their DiT service.
"""

import queue
import threading

import torch.distributed as dist

from ...channels import CHANNEL_TIMEOUT, SHUTDOWN_TIMEOUT
from .phase_utils import cpu_tree


class DecoderHandoff:
    def __init__(self, ranks, *, on_event=None):
        self.rank = dist.get_rank()
        self.ranks = tuple(ranks)
        self.on_event = on_event or (lambda *a, **kw: None)
        self.groups = {}
        # Every world rank creates the same groups before any stage can drain.
        for rank in self.ranks:
            if rank != 0:
                group = dist.new_group(ranks=[0, rank], backend="gloo", timeout=CHANNEL_TIMEOUT)
                if self.rank in (0, rank):
                    self.groups[rank] = group
        self.jobs = queue.Queue()
        self.results = {}
        self.errors = []
        self.lock = threading.Lock()
        self.count = 0
        self.finished = False
        self.threads = []
        self.replies = queue.Queue()
        self.heartbeat = None
        if self.rank != 0 and self.rank in self.ranks:
            self.heartbeat = threading.Thread(target=self._wait_ready, daemon=True)
            self.heartbeat.start()
        if self.rank == 0:
            for rank, group in self.groups.items():
                thread = threading.Thread(target=self._broker, args=(rank, group), daemon=True)
                thread.start()
                self.threads.append(thread)

    def _wait_ready(self):
        # A single CPU sender owns every reply, including heartbeats during
        # loading, idle waits and long GPU decode calls. No GPU work runs here.
        try:
            while True:
                try:
                    message = self.replies.get(timeout=CHANNEL_TIMEOUT.total_seconds() / 3)
                except queue.Empty:
                    message = {"waiting": True}
                if message is None:
                    return
                self._send(message, 0, self.groups[self.rank])
                if message.get("stopped") or message.get("error"):
                    return
        except Exception as exc:
            self._error(self.rank, str(exc))

    def _announce(self, message):
        self.replies.put(message)

    def _receive_reply(self, rank, group):
        message = self._recv(rank, group)
        while message == {"waiting": True}:
            message = self._recv(rank, group)
        if message.get("error"):
            raise RuntimeError(message["error"])
        return message

    def publish(self, output):
        if self.rank != 0 or self.finished:
            raise RuntimeError("Only an open ingress may publish decoder work")
        index = self.count
        self.count += 1
        self.jobs.put((index, cpu_tree(output)))

    def finish(self):
        if self.rank == 0 and not self.finished:
            self.finished = True
            for _ in self.ranks:
                self.jobs.put(None)

    def _error(self, rank, error):
        with self.lock:
            self.errors.append(f"rank {rank}: {error}")

    @staticmethod
    def _send(value, destination, group):
        dist.send_object_list([value], dst=destination, group=group)

    @staticmethod
    def _recv(source, group):
        values = [None]
        dist.recv_object_list(values, src=source, group=group)
        return values[0]

    def _next(self):
        return self.jobs.get(timeout=SHUTDOWN_TIMEOUT)

    def _broker(self, rank, group):
        try:
            ready = self._receive_reply(rank, group)
            if ready != {"ready": True}:
                raise RuntimeError("Invalid decoder ready message")
            self.on_event("decoder_replica_ready", decoder_rank=rank)
            while True:
                try:
                    job = self.jobs.get(timeout=CHANNEL_TIMEOUT.total_seconds() / 3)
                except queue.Empty:
                    # Keep the command direction alive too. A decoder can be
                    # ready long before the next final-step result is published.
                    self._send({"waiting": True}, rank, group)
                    if self._receive_reply(rank, group) != {"idle": True}:
                        raise RuntimeError("Invalid decoder idle reply")
                    continue
                self._send(job, rank, group)
                if job is None:
                    if self._receive_reply(rank, group) != {"stopped": True}:
                        raise RuntimeError("Invalid decoder STOP reply")
                    return
                index, value = job
                del value, job
                answer = self._receive_reply(rank, group)
                if answer["index"] != index:
                    raise RuntimeError("Decoder result index mismatch")
                with self.lock:
                    self.results[index] = answer["value"]
        except Exception as exc:
            self._error(rank, str(exc))

    def work(self, load, decode, *, service_error=None):
        """Called only after the local service and all GPU owners are released."""
        if self.rank not in self.ranks:
            return
        decoder = None
        group = self.groups.get(self.rank)
        try:
            if service_error is not None:
                raise RuntimeError(service_error)
            self.on_event("decoder_load_start")
            decoder = load()
            self.on_event("decoder_load_end")
            if self.rank != 0:
                self._announce({"ready": True})
            else:
                self.on_event("decoder_replica_ready", decoder_rank=0)
            while True:
                job = self._next() if self.rank == 0 else self._recv(0, group)
                if job is None:
                    if self.rank != 0:
                        self._announce({"stopped": True})
                    break
                if isinstance(job, dict) and job == {"waiting": True}:
                    self._announce({"idle": True})
                    continue
                index, value = job
                self.on_event("decode_start", request_id=value.request_id)
                value = cpu_tree(decode(decoder, value))
                self.on_event("decode_end", request_id=value.request_id)
                if self.rank == 0:
                    self.results[index] = value
                else:
                    self._announce({"index": index, "value": value})
                del value, job
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._error(self.rank, error)
            if self.rank != 0:
                self._announce({"error": error})
        finally:
            del decoder
            if self.heartbeat is not None:
                self.replies.put(None)
                self.heartbeat.join(timeout=SHUTDOWN_TIMEOUT)
                if self.heartbeat.is_alive():
                    self._error(self.rank, "Decoder reply sender did not terminate")

    def collect(self):
        for thread in self.threads:
            # Bound individual queue/transport stalls, not total finite-job
            # decode time: a healthy job may take many channel timeouts.
            thread.join()
        if self.rank != 0:
            return []
        if len(self.results) != self.count:
            self._error(0, f"Expected {self.count} decoded outputs, got {len(self.results)}")
        return [self.results[index] for index in sorted(self.results)]

    def close(self):
        for group in self.groups.values():
            dist.destroy_process_group(group)
        self.groups.clear()
