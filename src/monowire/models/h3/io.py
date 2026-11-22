"""Serialized codec work, independent of the ingress scheduling thread."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from functools import partial

import torch

from ...async_utils import await_completion
from ...channels import ready_event


class CodecOffloader:
    """Scoped codec residency on the caller's stream, without global hooks.

    Codec weights are immutable during inference. Keep their CPU storage for
    the lifetime of the offloader; only mutable buffers need a return copy.
    A serialized CodecWorker owns the scopes and their CUDA stream.
    """

    def __init__(self, modules, device, *, pin_memory):
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Codec offload requires CPU or CUDA")
        self.pin_memory = pin_memory and self.device.type == "cuda"
        self.active = None
        self._weights = {}
        for module in dict.fromkeys(modules):
            weights = [(parameter, self._host_copy(parameter)) for parameter in module.parameters()]
            self._weights[module] = weights
            self._offload(module)

    def _host_copy(self, tensor):
        # Preserve version-counter semantics even inside CodecWorker's
        # inference_mode. Pin only new CPU storage, never weights on unload.
        with torch.inference_mode(tensor.is_inference()):
            data = tensor.detach().cpu()
            if self.pin_memory and not data.is_pinned():
                data = data.pin_memory()
            return data

    def _load(self, module):
        for parameter, host in self._weights[module]:
            with torch.inference_mode(parameter.is_inference()):
                parameter.data = host.to(self.device, non_blocking=True)
        for tensor in module.buffers():
            with torch.inference_mode(tensor.is_inference()):
                tensor.data = tensor.to(self.device, non_blocking=True)

    def _offload(self, module):
        for parameter, host in self._weights[module]:
            parameter.data = host
        # Enumerate current buffers: a forward may mutate or replace them.
        for tensor in module.buffers():
            tensor.data = self._host_copy(tensor)

    @contextmanager
    def activate(self, module):
        if module not in self._weights:
            raise ValueError("Component is not owned by this codec offloader")
        if self.active is module:
            # Nested calls within one component share the outer residency scope.
            yield
            return
        if self.active is not None:
            raise RuntimeError("Codec components must execute serially")
        stream = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        self.active = module
        try:
            self._load(module)
            yield
        finally:
            try:
                if stream is not None:
                    stream.synchronize()
                self._offload(module)
            finally:
                self.active = None


class CodecWorker:
    """One codec owner and stream per ingress; active requests bound the queue.

    DiT is never moved by this worker. The pipeline's codec residency scopes manage
    only encoder/VAE weights. Stream events order latent inputs; the worker waits
    for its own output before handing request state back to the scheduler.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=device) if self.device.type == "cuda" else None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="monowire-codec")
        self.closed = False

    def _run(self, operation, ready):
        if self.stream is not None:
            torch.cuda.set_device(self.device)
        context = torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()
        with torch.inference_mode(), context:
            if ready is not None:
                self.stream.wait_event(ready)
            try:
                return operation()
            finally:
                if self.stream is not None:
                    self.stream.synchronize()

    async def run(self, operation, *args):
        if self.closed:
            raise RuntimeError("Codec worker is closed")
        ready = ready_event(self.device)
        future = asyncio.get_running_loop().run_in_executor(
            self.pool, self._run, partial(operation, *args), ready
        )
        return await await_completion(future)

    async def close(self):
        if not self.closed:
            self.closed = True
            await await_completion(asyncio.to_thread(self.pool.shutdown, wait=True))
