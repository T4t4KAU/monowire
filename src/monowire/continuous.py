"""Bounded, completion-driven request scheduling, without denoising-step barriers."""

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from .interfaces import BatchExecutor, RequestAdapter


@dataclass(frozen=True)
class ContinuousConfig:
    max_active_requests: int = 32
    microbatch_size: int = 8
    max_batch_wait_ms: float = 2.0

    def __post_init__(self):
        if self.max_active_requests < 1 or self.microbatch_size < 1:
            raise ValueError("Request capacity and batch size must be positive")
        if self.microbatch_size > self.max_active_requests:
            raise ValueError("Batch size exceeds active request capacity")
        if not math.isfinite(self.max_batch_wait_ms) or self.max_batch_wait_ms < 0:
            raise ValueError("Batch wait must be nonnegative")


@dataclass
class PreparedStep:
    request_id: str
    step: int
    hit: bool
    compatibility: Any
    payload: Any


@dataclass
class _Request:
    key: str
    value: Any
    future: asyncio.Future
    state: Any = None
    inflight: bool = False


class ContinuousScheduler:
    """Adapter methods run on one owner thread; executor returns independent futures.

    adapter: admit(key, value), prepare(state), commit(state, prepared, output),
    finished(state), finish(state), release(state).
    executor: async execute(list[PreparedStep]) -> list[output].
    Each request has at most one step in flight. Different requests need not
    have equal step indices. Cancellation drains in-flight work before cleanup.
    """

    def __init__(
        self, adapter: RequestAdapter, executor: BatchExecutor, config=None, on_event=None
    ):
        self.adapter, self.executor = adapter, executor
        self.config = config or ContinuousConfig()
        self.on_event = on_event or (lambda event: None)
        self.pending = deque()
        self.active = {}
        self.ready = {}
        self.tasks = {}
        self.ids = set()
        self.wakeup = asyncio.Event()
        self.closing = False
        self.task = None

    def _event(self, name, **values):
        try:
            self.on_event({"event": name, "time": time.monotonic(), **values})
        except Exception:
            logging.getLogger("monowire").exception("Continuous event observer failed")

    def submit(self, request_id, value):
        if self.closing:
            raise RuntimeError("Scheduler is closing")
        if request_id in self.ids:
            raise ValueError(f"Duplicate request ID: {request_id}")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request = _Request(request_id, value, future)
        self.ids.add(request_id)
        self.pending.append(request)
        future.add_done_callback(lambda _: self.wakeup.set())
        if self.task is None:
            self.task = loop.create_task(self._run())
        self.wakeup.set()
        return future

    def cancel(self, request_id):
        request = self.active.get(request_id)
        if request is None:
            request = next((r for r in self.pending if r.key == request_id), None)
        return request.future.cancel() if request is not None else False

    async def close(self):
        self.closing = True
        self.wakeup.set()
        if self.task is not None:
            await asyncio.shield(self.task)

    def _release(self, request, error=None, result=None):
        try:
            if request.state is not None:
                self.adapter.release(request.state)
        except Exception as exc:
            error = error or exc
        self.active.pop(request.key, None)
        self.ids.discard(request.key)
        if not request.future.done():
            if error is None:
                request.future.set_result(result)
            else:
                request.future.set_exception(error)
        self._event("released", request_id=request.key, error=str(error) if error else None)

    def _prepare(self, request):
        if request.future.cancelled():
            self._release(request)
            return
        prepared = self.adapter.prepare(request.state)
        if prepared.request_id != request.key:
            raise RuntimeError("Adapter returned the wrong request identity")
        bucket = self.ready.setdefault((prepared.hit, prepared.compatibility), deque())
        bucket.append((asyncio.get_running_loop().time(), request, prepared))
        self._event("ready", request_id=request.key, step=prepared.step, hit=prepared.hit)

    def _admit(self):
        while self.pending and len(self.active) < self.config.max_active_requests:
            request = self.pending.popleft()
            if request.future.cancelled():
                self._release(request)
                continue
            try:
                request.state = self.adapter.admit(request.key, request.value)
                self.active[request.key] = request
                self._event("admitted", request_id=request.key)
                self._prepare(request)
            except Exception as exc:
                self._release(request, exc)

    async def _execute(self, members):
        outputs = await self.executor.execute([p for _, p in members])
        if len(outputs) != len(members):
            raise RuntimeError("Executor output count does not match batch")
        return outputs

    def _dispatch(self):
        loop = asyncio.get_running_loop()
        delay = None
        for key, bucket in list(self.ready.items()):
            bucket = deque(item for item in bucket if item[1].key in self.active)
            self.ready[key] = bucket
            for _, request, _ in list(bucket):
                if request.future.cancelled():
                    self._release(request)
            bucket = deque(item for item in bucket if item[1].key in self.active)
            self.ready[key] = bucket
            while bucket:
                age = loop.time() - bucket[0][0]
                remaining = self.config.max_batch_wait_ms / 1000 - age
                if len(bucket) < self.config.microbatch_size and remaining > 0:
                    delay = remaining if delay is None else min(delay, remaining)
                    break
                members = [
                    (request, prepared)
                    for _, request, prepared in (
                        bucket.popleft()
                        for _ in range(min(len(bucket), self.config.microbatch_size))
                    )
                ]
                for request, _ in members:
                    if request.inflight:
                        raise RuntimeError("Two steps of the same request are in flight")
                    request.inflight = True
                task = loop.create_task(self._execute(members))
                self.tasks[task] = members
                task.add_done_callback(lambda _: self.wakeup.set())
                self._event(
                    "dispatched",
                    requests=[(p.request_id, p.step) for _, p in members],
                    hit=key[0],
                )
            if not bucket:
                self.ready.pop(key, None)
        return delay

    def _complete(self):
        for task in [task for task in self.tasks if task.done()]:
            members = self.tasks.pop(task)
            try:
                outputs = task.result()
            except Exception as exc:
                for request, _ in members:
                    self._release(request, exc)
                continue
            for (request, prepared), output in zip(members, outputs, strict=True):
                request.inflight = False
                if request.future.cancelled():
                    self._release(request)
                    continue
                try:
                    self.adapter.commit(request.state, prepared, output)
                    self._event("step_completed", request_id=request.key, step=prepared.step)
                    if self.adapter.finished(request.state):
                        result = self.adapter.finish(request.state)
                        self._release(request, result=result)
                    else:
                        self._prepare(request)
                except Exception as exc:
                    self._release(request, exc)

    async def _run(self):
        try:
            while not self.closing or self.pending or self.active or self.tasks:
                self.wakeup.clear()
                self._complete()
                self._admit()
                delay = self._dispatch()
                if self.closing and not (self.pending or self.active or self.tasks):
                    break
                try:
                    if delay is None:
                        await self.wakeup.wait()
                    else:
                        await asyncio.wait_for(self.wakeup.wait(), delay)
                except TimeoutError:
                    pass
        except BaseException as exc:
            self.closing = True
            # Cancellation cannot stop communication already running in a
            # worker thread. Drain it before releasing request-owned tensors.
            await asyncio.gather(*self.tasks, return_exceptions=True)
            for request in [*self.active.values(), *self.pending]:
                self._release(request, RuntimeError(f"Continuous scheduler failed: {exc}"))
            self.pending.clear()
            self.ready.clear()
            self.tasks.clear()
            raise
