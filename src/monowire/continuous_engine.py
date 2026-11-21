"""Model-independent continuous request service lifecycle."""

import asyncio
import json
import logging
from contextlib import aclosing
from dataclasses import dataclass

from .continuous import ContinuousScheduler
from .interfaces import ModelRuntime


@dataclass
class ContinuousOutput:
    request_id: str
    output: object = None
    error: str | None = None


class ContinuousEngine:
    """One instance per torchrun rank; only rank 0 consumes the request source.

    stream() accepts an async iterable of (request_id, prompt, sampling_params).
    It yields model output objects as requests finish, not input order.
    All ranks must enter stream/generate and close in the same service lifecycle.
    """

    def __init__(self, runtime: ModelRuntime, config):
        self.runtime, self.config = runtime, config
        self.rank = runtime.rank
        self.running = False
        self.on_event = None
        self._scheduler = None

    async def stream(self, requests):
        if self.running:
            raise RuntimeError("One continuous service loop may own a worker")
        self.running = True
        try:
            with self.runtime.context():
                if self.rank != 0:
                    self.runtime.serve()
                    return
                async with aclosing(self._stream_ingress(requests)) as stream:
                    async for output in stream:
                        yield output
        finally:
            self._scheduler = None
            self.running = False

    async def _stream_ingress(self, requests):
        adapter = self.runtime.create_adapter()
        executor = self.runtime.create_executor()
        log = logging.getLogger("monowire")

        def record(event):
            if log.isEnabledFor(logging.INFO):
                log.info("MonoWire continuous %s", json.dumps(event))
            if self.on_event is not None:
                self.on_event(event)

        scheduler = ContinuousScheduler(
            adapter,
            executor,
            self.config,
            on_event=record,
        )
        self._scheduler = scheduler
        completed = asyncio.Queue()
        # Count submitted requests until their results are consumed, including
        # completed outputs. A slow consumer must also backpressure the source.
        slots = asyncio.Semaphore(2 * self.config.max_active_requests)
        futures = set()

        def done(key, future):
            completed.put_nowait((key, future))

        def scheduler_done(task):
            if task.cancelled() or task.exception() is not None:
                # The source may be blocked waiting for its next arrival.
                feeder.cancel()

        async def feed():
            watching = False
            try:
                async for key, prompt, sampling in requests:
                    await slots.acquire()
                    try:
                        future = scheduler.submit(key, (prompt, sampling))
                    except BaseException:
                        slots.release()
                        raise
                    if not watching:
                        scheduler.task.add_done_callback(scheduler_done)
                        watching = True
                    futures.add(future)
                    future.add_done_callback(lambda f, k=key: done(k, f))
            finally:
                await scheduler.close()

        feeder = asyncio.create_task(feed())
        feeder.add_done_callback(lambda _: completed.put_nowait(None))
        try:
            while (item := await completed.get()) is not None:
                key, future = item
                futures.discard(future)
                slots.release()
                if future.cancelled():
                    yield ContinuousOutput(key, error="Request cancelled")
                elif (error := future.exception()) is not None:
                    yield ContinuousOutput(key, error=str(error))
                else:
                    yield ContinuousOutput(key, future.result())
            await feeder
        finally:
            if not feeder.done():
                feeder.cancel()
            for future in list(futures):
                future.cancel()
            await asyncio.gather(feeder, return_exceptions=True)
            try:
                await scheduler.close()
            finally:
                # Retrieve unconsumed errors when the consumer exits early.
                for future in futures:
                    if future.done() and not future.cancelled():
                        future.exception()
                await executor.close()

    def cancel(self, request_id):
        """Cancel from the ingress event-loop thread; dispatched work is drained."""
        return self._scheduler.cancel(request_id) if self._scheduler is not None else False

    def generate(self, prompts, sampling_params):
        async def source():
            for index, prompt in enumerate(prompts):
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
