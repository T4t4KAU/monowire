"""Heartbeat-protected completion/error agreement across every worker rank."""

import threading

import torch.distributed as dist

from ...channels import CHANNEL_TIMEOUT, HEARTBEAT_INTERVAL


class PhaseCompletion:
    """Keep early finishers alive until every rank finishes its local work.

    Dedicated CPU pairs run independently of model and decoder traffic. Errors
    travel in the final notification itself, so no rank enters a world collective
    while another rank is still computing or loading a model.
    """

    def __init__(self):
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.groups = {}
        for rank in range(1, self.world):
            group = dist.new_group(ranks=[0, rank], backend="gloo", timeout=CHANNEL_TIMEOUT)
            if self.rank in (0, rank):
                self.groups[rank] = group
        self.interval = min(HEARTBEAT_INTERVAL, CHANNEL_TIMEOUT.total_seconds() / 3)
        self.local_done = threading.Event()
        self.all_done = threading.Event()
        self.lock = threading.Lock()
        self.done = set()
        self.errors = [""] * self.world
        self.threads = []
        self.transport_error = None
        for rank, group in self.groups.items():
            target = self._coordinate if self.rank == 0 else self._report
            thread = threading.Thread(target=target, args=(rank, group), daemon=True)
            thread.start()
            self.threads.append(thread)

    @staticmethod
    def _send(value, peer, group):
        dist.send_object_list([value], dst=peer, group=group)

    @staticmethod
    def _recv(peer, group):
        message = [None]
        dist.recv_object_list(message, src=peer, group=group)
        return message[0]

    def _record(self, rank, error):
        with self.lock:
            self.errors[rank] = error
            self.done.add(rank)
            if len(self.done) == self.world:
                self.all_done.set()

    def _coordinate(self, rank, group):
        try:
            while True:
                status = self._recv(rank, group)
                if status["done"]:
                    self._record(rank, status["error"])
                    # Early finishers receive heartbeats while other ranks work.
                    if self.all_done.wait(self.interval):
                        self._send({"errors": list(self.errors)}, rank, group)
                        return
                self._send({"waiting": True}, rank, group)
        except Exception as exc:
            self._record(rank, f"rank {rank}: completion transport failed: {exc}")

    def _report(self, rank, group):
        try:
            while True:
                done = self.local_done.is_set()
                self._send({"done": done, "error": self.errors[rank] if done else ""}, 0, group)
                response = self._recv(0, group)
                if "errors" in response:
                    self.errors = response["errors"]
                    return
                if not done:
                    # Wake immediately on local completion, otherwise heartbeat.
                    self.local_done.wait(self.interval)
        except Exception as exc:
            self.transport_error = f"rank {rank}: completion transport failed: {exc}"

    def finish(self, error=""):
        if not self.local_done.is_set():
            if self.rank == 0:
                self._record(0, error)
            else:
                self.errors[self.rank] = error
            self.local_done.set()
        for thread in self.threads:
            thread.join()
        if self.transport_error is not None:
            self.errors[self.rank] = self.transport_error
        return list(self.errors)

    def close(self):
        if not self.local_done.is_set():
            self.finish(f"rank {self.rank}: phase exited before completion agreement")
        for group in self.groups.values():
            dist.destroy_process_group(group)
        self.groups.clear()


def complete_phase(operation, *, name="Phase"):
    """Agree before the next collective, even when some ranks have no work."""
    completion = PhaseCompletion()
    try:
        result, error = None, ""
        try:
            result = operation()
        except Exception as exc:
            error = f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
        errors = completion.finish(error)
        if any(errors):
            raise RuntimeError(name + " failed: " + "; ".join(e for e in errors if e))
        return result
    finally:
        completion.close()
