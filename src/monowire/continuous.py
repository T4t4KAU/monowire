"""Bounded, completion-driven request scheduling, without denoising-step barriers."""

import asyncio
import logging
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any

from .async_utils import await_completion
from .interfaces import BatchExecutor, PipelineFailure, RequestAdapter


@dataclass(frozen=True)
class ContinuousConfig:
    max_active_requests: int = 32
    microbatch_size: int = 8
    downstream_batch_size: int = 8

    def __post_init__(self):
        sizes = (self.max_active_requests, self.microbatch_size, self.downstream_batch_size)
        if any(type(size) is not int or size < 1 for size in sizes):
            raise ValueError("Request capacity and batch size must be positive")
        if self.microbatch_size > self.max_active_requests:
            raise ValueError("Batch size exceeds active request capacity")


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
    """DiT methods run on the owner thread; codec methods are asynchronous.

    adapter: admit(key, value), prepare(state), prepare_batch(prepared),
    commit(state, prepared, output),
    finished(state), finish(state), release(state).
    executor: submit(list[PreparedStep]) -> list[Future[output]].
    Each request has at most one step in flight. Different requests need not
    have equal step indices. Cancellation drains in-flight work before cleanup.
    """

    def __init__(
        self, adapter: RequestAdapter, executor: BatchExecutor, config=None, on_event=None
    ):
        self.adapter, self.executor = adapter, executor
        self.config = config or ContinuousConfig()
        self.on_event = on_event or (lambda event: None)
        self.pending = OrderedDict()  # Submitted requests awaiting an active slot.
        self.active = {}  # Includes encoding, denoising and decoding.
        self.ingress = OrderedDict()  # Compatible steps awaiting prefix computation.
        self.ready = {}  # Steps grouped by cache route after prefix computation.
        self.tasks = {}  # Dispatched DiT steps, one per request at most.
        self.io_tasks = {}  # Asynchronous admission and output decoding.
        self.ids = set()
        self.wakeup = asyncio.Event()
        self.closing = False
        self.task = None
        self.failure_signal = None
        self._executor_failure = None

    def start(self):
        """Monitor the executor while waiting for the first or next arrival."""
        if self.closing:
            raise RuntimeError("Scheduler is closing")
        if self.task is None:
            self._executor_failure = self.executor.start()
            self._executor_failure.add_done_callback(self._wake)
            self.failure_signal = asyncio.get_running_loop().create_future()
            self.task = asyncio.create_task(self._run())
        return self.task

    def _wake(self, _):
        self.wakeup.set()

    def _request_done(self, request):
        # A queued request owns no in-flight state. Release it before the
        # engine consumes cancellation and returns its submission credit.
        if request.future.cancelled() and self.pending.get(request.key) is request:
            del self.pending[request.key]
            self._release(request)
        self.wakeup.set()

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
        self.start()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request = _Request(request_id, value, future)
        self.ids.add(request_id)
        self.pending[request_id] = request
        future.add_done_callback(lambda _: self._request_done(request))
        self.wakeup.set()
        return future

    def cancel(self, request_id):
        request = self.active.get(request_id)
        if request is None:
            request = self.pending.get(request_id)
        return request.future.cancel() if request is not None else False

    async def close(self):
        self.closing = True
        self.wakeup.set()
        if self.task is not None:
            await await_completion(self.task)

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
        bucket = self.ingress.setdefault(prepared.compatibility, deque())
        bucket.append((request, prepared))
        self._event("prefix_ready", request_id=request.key, step=prepared.step)

    def _prepare_batch(self):
        """Run one ingress batch, then let dispatched work advance before the next."""
        while self.ingress:
            compatibility, bucket = self.ingress.popitem(last=False)
            members = self._take_batch(bucket)
            if bucket:
                # Round-robin compatible groups, keeping FIFO order within
                # each group. A busy group cannot starve another shape/task.
                self.ingress[compatibility] = bucket
            if not members:
                continue
            try:
                expected = [(request.key, prepared.step) for request, prepared in members]
                outputs = self.adapter.prepare_batch([prepared for _, prepared in members])
                if [(prepared.request_id, prepared.step) for prepared in outputs] != expected:
                    raise RuntimeError("Ingress batch changed request identities or order")
                self._event("prefix_dispatched", requests=expected)
                for (request, _), prepared in zip(members, outputs, strict=True):
                    ready = self.ready.setdefault((prepared.hit, prepared.compatibility), deque())
                    ready.append((request, prepared))
                    self._event(
                        "ready", request_id=request.key, step=prepared.step, hit=prepared.hit
                    )
            except Exception as exc:
                for request, _ in members:
                    self._release(request, exc)
            return

    def _take_batch(self, bucket):
        """Take a FIFO batch, releasing canceled requests without consuming a slot."""
        members = []
        while bucket and len(members) < self.config.microbatch_size:
            request, prepared = bucket.popleft()
            if request.future.cancelled():
                self._release(request)
            else:
                members.append((request, prepared))
        return members

    def _start_io(self, request, phase, operation):
        task = asyncio.create_task(operation)
        self.io_tasks[task] = (request, phase)
        task.add_done_callback(self._wake)

    def _admit(self):
        while self.pending and len(self.active) < self.config.max_active_requests:
            _, request = self.pending.popitem(last=False)
            if request.future.cancelled():
                self._release(request)
                continue
            # Reserve the slot before asynchronous encoding starts. Admission,
            # denoising and decoding all count toward bounded residency.
            self.active[request.key] = request
            self._start_io(request, "admit", self.adapter.admit(request.key, request.value))

    def _complete_io(self):
        for task in [task for task in self.io_tasks if task.done()]:
            request, phase = self.io_tasks.pop(task)
            try:
                result = task.result()
                if phase == "admit":
                    request.state = result
                    self._event("admitted", request_id=request.key)
                    self._prepare(request)
                else:
                    self._release(request, result=result)
            except Exception as exc:
                self._release(request, exc)

    def _dispatch(self):
        for key, bucket in list(self.ready.items()):
            while bucket:
                members = self._take_batch(bucket)
                if not members:
                    continue
                try:
                    outputs = self.executor.submit([prepared for _, prepared in members])
                    if len(outputs) != len(members):
                        raise RuntimeError("Executor future count does not match batch")
                except PipelineFailure:
                    raise
                except Exception as exc:
                    for request, _ in members:
                        self._release(request, exc)
                else:
                    for (request, prepared), future in zip(members, outputs, strict=True):
                        if request.inflight:
                            raise RuntimeError("Two steps of the same request are in flight")
                        request.inflight = True
                        self.tasks[future] = (request, prepared)
                        future.add_done_callback(self._wake)
                    self._event(
                        "dispatched",
                        requests=[(prepared.request_id, prepared.step) for _, prepared in members],
                        hit=key[0],
                    )
            self.ready.pop(key, None)

    def _complete(self):
        for future in [future for future in self.tasks if future.done()]:
            request, prepared = self.tasks.pop(future)
            request.inflight = False
            try:
                output = future.result()
                if request.future.cancelled():
                    self._release(request)
                    continue
                self.adapter.commit(request.state, prepared, output)
                self._event("step_completed", request_id=request.key, step=prepared.step)
                if self.adapter.finished(request.state):
                    self._start_io(request, "finish", self.adapter.finish(request.state))
                else:
                    self._prepare(request)
            except PipelineFailure:
                raise
            except Exception as exc:
                self._release(request, exc)

    def _has_work(self):
        return bool(self.pending or self.active or self.tasks or self.io_tasks)

    async def _run(self):
        try:
            while not self.closing or self._has_work():
                self.wakeup.clear()
                if self._executor_failure.done():
                    raise self._executor_failure.result()
                self._complete_io()
                self._complete()
                self._admit()
                self._prepare_batch()
                self._dispatch()
                if self.closing and not self._has_work():
                    break
                if self.ingress:
                    await asyncio.sleep(0)
                else:
                    await self.wakeup.wait()
        except BaseException as error:
            await self._drain_failed(error)
            raise
        finally:
            self._executor_failure.remove_done_callback(self._wake)

    async def _drain_failed(self, error):
        self.closing = True
        # Notify the owner before draining: its source finalizer may release
        # resources needed by outstanding I/O. Task completion is too late.
        self.failure_signal.set_result(error)
        # Cancellation cannot stop CUDA work or a worker thread. Drain before
        # releasing storage, including admissions completed during shutdown.
        try:
            await await_completion(
                asyncio.gather(*self.tasks, *self.io_tasks, return_exceptions=True)
            )
        finally:
            for task, (request, phase) in self.io_tasks.items():
                if phase == "admit" and not task.cancelled() and task.exception() is None:
                    request.state = task.result()
            for request in [*self.active.values(), *self.pending.values()]:
                self._release(request, RuntimeError(f"Continuous scheduler failed: {error}"))
            self.pending.clear()
            self.ingress.clear()
            self.ready.clear()
            self.tasks.clear()
            self.io_tasks.clear()
