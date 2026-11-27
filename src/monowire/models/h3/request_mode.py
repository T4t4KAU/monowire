"""Native multi-pass sampling through a resident layer pipeline.

Continuation/refinement keep the native pass state machine. This execution lane
is deliberately serial across requests: every transformer call traverses PP,
while codecs and high-resolution keyframe encoding remain separate phases.
"""

from contextlib import contextmanager
from types import MethodType

import torch

from ...channels import ready_event
from ...continuous_engine import ContinuousOutput
from .adapter import compute_stage
from .phase_utils import cpu_tree


class RequestPipeline:
    """Synchronous request-mode ring, sharing the tested tensor channel protocol."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.transformer = runtime.stage.transformer
        self.rank = runtime.rank
        self.world = runtime.topology.world_size
        self.channels = runtime.channels
        self.resident = True

    def _roundtrip(self, message):
        self.channels[0, 1].send(message, ready_event(self.runtime.pipeline.device))
        result = self.channels[self.world - 1, 0].recv().on_current_stream()
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result

    def forward(self, **kwargs):
        transformer = self.transformer
        if not self.resident:
            transformer.to(self.runtime.pipeline.device)
            self.resident = True
        hidden, arguments, context = transformer.prepare_transformer_forward(**kwargs)
        cache = transformer.adaln_cache
        packet = {
            **arguments,
            "hidden": hidden,
            "adaln_key": cache.prepared_key(arguments["t_emb"]) if cache is not None else None,
        }
        compute_stage(transformer, packet)
        if self.world > 1:
            packet = self._roundtrip({"packet": packet})["packet"]
        return transformer.finish_transformer_forward(packet["hidden"], **context)

    def serve(self):
        source = self.rank - 1
        destination = (self.rank + 1) % self.world
        while True:
            message = self.channels[source, self.rank].recv().on_current_stream()
            if not message.get("stop") and not message.get("error"):
                try:
                    compute_stage(self.transformer, message["packet"])
                except Exception as error:
                    # Return the request error around the ring and keep the
                    # service live for subsequent requests and the final STOP.
                    message = {"error": f"PP rank {self.rank}: {type(error).__name__}: {error}"}
            self.channels[self.rank, destination].send(
                message, ready_event(self.runtime.pipeline.device)
            )
            if message.get("stop"):
                return

    def close(self):
        if self.rank == 0 and self.world > 1:
            self._roundtrip({"stop": True})


@contextmanager
def native_request_context(engine, runtime, bridge):
    """Bind preencoded conditions and PP without changing vendored native code."""
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _REFINE_KEYFRAME_CONDITION,
    )

    pipeline = runtime.pipeline
    transformer = pipeline.transformer
    # Restore instance overrides exactly, including the absence of an override.
    originals = []

    def bind(target, name, function):
        originals.append((target, name, name in target.__dict__, target.__dict__.get(name)))
        setattr(target, name, MethodType(function, target))

    def prepare(owner, prompt, sampling):
        conditioning = owner._extract_encoder_conditioning(prompt)
        context = owner._prepare_encoder_conditioning_inputs(conditioning, sampling)
        additional = prompt["additional_information"]
        windows = additional.get("monowire_window_text")
        if windows is not None:
            # Keep the finite window list on the host. Only the active pass
            # copies its text into GPU memory in diffuse() below.
            context["continuation_text_conditioning"] = windows
        refine = additional.get("monowire_refine_condition")
        if refine is not None:
            rows, shapes = refine
            context[_REFINE_KEYFRAME_CONDITION] = (rows, shapes)
        return context

    original_upscale = pipeline._upscaled_latent
    original_diffuse = pipeline.diffuse
    pass_index = 0

    def diffuse(owner, **kwargs):
        nonlocal pass_index
        pass_index += 1
        for name in ("text_embeddings", "text_tags", "visual_condition"):
            if kwargs.get(name) is not None:
                kwargs[name] = kwargs[name].to(owner.device)
        engine._mark(
            "native_pass_start",
            pass_index=pass_index,
            frames=kwargs.get("num_frames"),
            refine=kwargs.get("refine") is not None,
        )
        with torch.profiler.record_function("monowire/native_denoise_pass"):
            result = original_diffuse(**kwargs)
        engine._mark("native_pass_done", pass_index=pass_index)
        return result

    def upscale(owner, video, target):
        if target is None:
            return video
        # Free ingress DiT residency before the upscaler enters the device.
        # Suffix stages are idle and retain their weights. The next native
        # transformer call (refine or next request) reloads the ingress stage.
        transformer.to("cpu")
        bridge.resident = False
        if transformer.adaln_cache is not None:
            transformer.adaln_cache.clear()
        torch.cuda.synchronize(owner.device)
        torch.cuda.empty_cache()
        # Do not reset allocator peaks inside the enclosing DiT phase.
        engine._mark(
            "latent_upscale_start", allocated_bytes=torch.cuda.memory_allocated(owner.device)
        )
        with torch.profiler.record_function("monowire/latent_upscale"):
            result = original_upscale(video, target)
        torch.cuda.synchronize(owner.device)
        engine._mark(
            "latent_upscale_done", allocated_bytes=torch.cuda.memory_allocated(owner.device)
        )
        return result

    old_step = pipeline.od_config.step_execution
    pipeline.od_config.step_execution = False
    try:
        bind(transformer, "forward", lambda owner, **kwargs: bridge.forward(**kwargs))
        bind(pipeline, "_prepare_request_inputs", prepare)
        bind(pipeline, "_upscaled_latent", upscale)
        bind(pipeline, "diffuse", diffuse)
        yield
    finally:
        pipeline.od_config.step_execution = old_step
        for target, name, existed, original in reversed(originals):
            if existed:
                setattr(target, name, original)
            else:
                delattr(target, name)


def run_requests(engine, runtime, encoded, sampling):
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

    bridge = RequestPipeline(runtime)
    if runtime.rank != 0:
        bridge.serve()
        return []
    outputs = []
    try:
        with native_request_context(engine, runtime, bridge):
            for index, prompt in enumerate(encoded):
                key = str(index)
                engine._mark("request_start", request_id=key)
                try:
                    batch = DiffusionRequestBatch(
                        requests=[
                            OmniDiffusionRequest(
                                request_id=key,
                                prompt=prompt,
                                sampling_params=sampling,
                            )
                        ]
                    )
                    result = runtime.pipeline(batch)
                    result.output = cpu_tree(result.output)
                    outputs.append(ContinuousOutput(key, result))
                except Exception as error:
                    outputs.append(ContinuousOutput(key, error=f"{type(error).__name__}: {error}"))
                finally:
                    encoded[index] = None
                    engine._mark("request_done", request_id=key)
        return outputs
    finally:
        bridge.close()
