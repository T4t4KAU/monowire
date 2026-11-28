"""Model-independent continuous request service lifecycle."""

import asyncio
import json
import logging
from contextlib import AsyncExitStack, aclosing, suppress
from dataclasses import dataclass

from .async_utils import await_completion
from .continuous import ContinuousScheduler
from .interfaces import ModelRuntime

logger = logging.getLogger("monowire")


@dataclass
class ContinuousOutput:
    request_id: str
    output: object = None
    error: str | None = None


class ContinuousEngine:
    """One instance per torchrun rank; ingress replicas consume request sources.

    stream() accepts an async iterable of (request_id, prompt, sampling_params).
    It yields model output objects as requests finish, not input order.
    Each ingress consumes its own stream and yields its own results. generate()
    shards a common prompt list by index across ingress replicas automatically.
    All ranks must enter stream/generate and close in the same service lifecycle.
    """

    def __init__(self, runtime: ModelRuntime, config):
        self.runtime, self.config = runtime, config
        self.rank = runtime.rank
        self.ingress_replicas = runtime.ingress_replicas
        self.running = False
        self.on_event = None
        self._scheduler = None

    async def stream(self, requests):
        if self.running:
            raise RuntimeError("One continuous service loop may own a worker")
        self.running = True
        try:
            with self.runtime.context():
                if not self.is_ingress:
                    self.runtime.serve()
                    return
                async with aclosing(self._stream_ingress(requests)) as stream:
                    async for output in stream:
                        yield output
        finally:
            self._scheduler = None
            self.running = False

    @property
    def is_ingress(self):
        return self.rank < self.ingress_replicas

    async def _stream_ingress(self, requests):
        cleanup = AsyncExitStack()
        try:
            adapter = self.runtime.create_adapter()
            cleanup.push_async_callback(adapter.close)
            executor = self.runtime.create_executor()
            cleanup.push_async_callback(executor.close)
            scheduler = ContinuousScheduler(
                adapter, executor, self.config, on_event=self._record_event
            )
            intake_cleanup = await cleanup.enter_async_context(AsyncExitStack())
            intake_cleanup.push_async_callback(scheduler.close)
            self._scheduler = scheduler
            source = aiter(requests)
            # Async iterators need not expose aclose. When they do, the service
            # owns it even if backpressure suspends the feeder outside __anext__.
            if (close_source := getattr(source, "aclose", None)) is not None:
                intake_cleanup.push_async_callback(close_source)
            async with aclosing(self._run_requests(scheduler, source, intake_cleanup)) as outputs:
                async for output in outputs:
                    yield output
        finally:
            await await_completion(cleanup.aclose())

    def _record_event(self, event):
        if logger.isEnabledFor(logging.INFO):
            logger.info("MonoWire continuous %s", json.dumps(event))
        if self.on_event is not None:
            self.on_event(event)

    async def _run_requests(self, scheduler, source, intake_cleanup):
        """Feed requests with backpressure and yield results in completion order."""
        completed = asyncio.Queue()
        # Count submitted requests until their results are consumed, including
        # completed outputs. A slow consumer must also backpressure the source.
        host_limit = (
            self.config.host_pending_limit
            if self.config.memory_limits is not None
            else 2 * self.config.max_active_requests
        )
        slots = asyncio.Semaphore(host_limit) if host_limit is not None else None
        futures = set()
        accepting = True

        def enqueue_result(request_id, future):
            completed.put_nowait((request_id, future))

        def stop_feeder(_=None):
            if accepting and not feeder.done() and not feeder.cancelling():
                feeder.cancel()

        async def feed():
            nonlocal accepting
            try:
                async for request_id, prompt, sampling in source:
                    if slots is not None:
                        await slots.acquire()
                    try:
                        future = scheduler.submit(request_id, (prompt, sampling))
                    except BaseException:
                        if slots is not None:
                            slots.release()
                        raise
                    futures.add(future)
                    future.add_done_callback(
                        lambda result, key=request_id: enqueue_result(key, result)
                    )
            finally:
                # Stop the source before draining requests. Its finalizer
                # may release resources needed by outstanding operations.
                # Do not interrupt cleanup with a second cancellation.
                accepting = False
                await await_completion(intake_cleanup.aclose())

        feeder = asyncio.create_task(feed())
        feeder.add_done_callback(lambda _: completed.put_nowait(None))

        async def drain_feeder():
            # stop_feeder() cancels intake deliberately. Other failures must
            # reach the caller; await_completion protects this drain separately.
            with suppress(asyncio.CancelledError):
                await feeder

        try:
            scheduler.start()
            scheduler.failure_signal.add_done_callback(stop_feeder)
            while (item := await completed.get()) is not None:
                request_id, future = item
                futures.discard(future)
                if slots is not None:
                    slots.release()
                if future.cancelled():
                    yield ContinuousOutput(request_id, error="Request cancelled")
                elif (error := future.exception()) is not None:
                    yield ContinuousOutput(request_id, error=str(error))
                else:
                    yield ContinuousOutput(request_id, future.result())
            await feeder
        finally:
            stop_feeder()
            for future in list(futures):
                future.cancel()
            try:
                await await_completion(drain_feeder())
            finally:
                if scheduler.failure_signal is not None:
                    scheduler.failure_signal.remove_done_callback(stop_feeder)
                # Retrieve unconsumed errors when the consumer exits early.
                for future in futures:
                    if future.done() and not future.cancelled():
                        future.exception()

    def cancel(self, request_id):
        """Cancel from the ingress event-loop thread; dispatched work is drained."""
        return self._scheduler.cancel(request_id) if self._scheduler is not None else False

    def generate(self, prompts, sampling_params):
        async def source():
            for index, prompt in enumerate(prompts):
                if index % self.ingress_replicas == self.rank:
                    yield str(index), prompt, sampling_params

        async def collect():
            return [output async for output in self.stream(source())]

        return asyncio.run(collect())

    def close(self):
        if self.running:
            raise RuntimeError("Drain or close the stream before closing workers")
        self.runtime.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
