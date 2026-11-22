"""Cancellation-safe waits for work whose resources must outlive the caller."""

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


async def await_completion(work: Awaitable[T]) -> T:
    """Finish owned work before propagating cancellation, including repeated cancels."""
    future = asyncio.ensure_future(work)
    cancellation = None
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError as error:
            cancellation = error
        except Exception:
            break  # Retrieve the operation's error below, after it has finished.
    if cancellation is not None and not future.cancelled():
        raise cancellation from future.exception()
    return future.result()
