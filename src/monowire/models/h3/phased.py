"""Finite H3 jobs: Qwen PP, media encoding, DiT PP, then VAE decoding."""

import asyncio
import gc
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

from ...cache import CacheConfig
from ...continuous import ContinuousConfig
from ...continuous_engine import ContinuousEngine
from .phase_utils import cpu_tree, encoder_partition
from .runtime import H3Runtime
from .validation import validate_runtime

logger = logging.getLogger("monowire")


class PhasedH3Engine:
    """All ranks call generate with a common finite job; only rank 0 returns outputs.

    Phase tensors live in CPU RAM. No model weights survive a phase boundary.
    Optional rolling handoff overlaps local DiT loading with Qwen drain.
    """

    def __init__(
        self,
        model,
        *,
        pp_size,
        config,
        cache,
        topology,
        encoder_stage_layers,
        trust_remote_code,
        overlap_encoder_dit=False,
        additional_config=None,
    ):
        self.additional_config = dict(additional_config or {})
        self.overlap_encoder_dit = overlap_encoder_dit
        self.timeline_events = []
        self.model = str(model)
        self.pp_size = pp_size
        self.config, self.cache, self.topology = config, cache, topology
        self.encoder_stage_layers = encoder_partition(pp_size, encoder_stage_layers)
        self.trust_remote_code = trust_remote_code
        self.rank = int(os.environ.get("RANK", "0"))
        self.ingress_replicas = 1
        self.running = False
        self.closed = False
        self.worker = None
        self.on_event = None
        self.phase_events = []

    @property
    def is_ingress(self):
        return self.rank == 0

    def _leader(self, operation):
        """Propagate preprocessing/codec failures before peers enter the next phase."""
        result = None
        error = None
        if self.is_ingress:
            try:
                result = operation()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        status = [error]
        dist.broadcast_object_list(status, src=0, group=self.cpu_group)
        if status[0] is not None:
            raise RuntimeError(status[0])
        return result

    def _mark(self, name, **values):
        event = {"event": name, "rank": self.rank, "time": time.monotonic(), **values}
        self.timeline_events.append(event)
        logger.info("MonoWire timeline %s", json.dumps(event))

    def _finish_dit(self, operation):
        """Agree on service failures after STOP, before entering decoder collectives."""
        result, error = None, None
        try:
            result = operation()
        except Exception as exc:
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        errors = [error]
        if self.pp_size > 1:
            errors = [None] * self.pp_size
            dist.all_gather_object(errors, error, group=self.cpu_group)
        if any(errors):
            raise RuntimeError("DiT service failed: " + "; ".join(item for item in errors if item))
        return result

    def _dit_event(self, event):
        self._mark(
            "dit_" + event["event"],
            **{key: value for key, value in event.items() if key not in {"event", "time", "rank"}},
        )

    def _phase(self, name, operation, *, barrier=True):
        if barrier:
            dist.barrier(group=self.cpu_group)
        torch.cuda.reset_peak_memory_stats(self.worker.device)
        start = time.perf_counter()
        result = operation()
        torch.cuda.synchronize(self.worker.device)
        gc.collect()
        torch.cuda.empty_cache()
        event = {
            "phase": name,
            "rank": self.rank,
            "seconds": time.perf_counter() - start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.worker.device),
            "remaining_allocated_bytes": torch.cuda.memory_allocated(self.worker.device),
        }
        self.phase_events.append(event)
        logger.info("MonoWire phase %s", json.dumps(event))
        if self.on_event is not None:
            self.on_event(event)
        return result

    def _make_worker(self):
        from .plugin import register

        os.environ["MONOWIRE_H3_PP"] = "1"
        register()
        from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
        from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

        od_config = OmniDiffusionConfig.from_kwargs(
            model=self.model,
            model_class_name="MiniMaxH3Pipeline",
            trust_remote_code=self.trust_remote_code,
            parallel_config=DiffusionParallelConfig(pipeline_parallel_size=self.pp_size),
            enforce_eager=True,
            cache_backend="none",
            step_execution=True,
            additional_config=self.additional_config,
            max_num_seqs=self.config.max_active_requests,
            enable_cpu_offload=False,
            model_loaded={"text_encoder": False, "vae_encoder": False, "vae_decoder": False},
            cache_config={
                "monowire": asdict(self.cache),
                "monowire_topology": asdict(self.topology) if self.topology else None,
            },
        )
        self.worker = DiffusionWorker(
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            rank=self.rank,
            od_config=od_config,
            distributed_init_method="env://",
            skip_load_model=True,
        )
        from vllm_omni.diffusion.distributed.parallel_state import get_world_group

        world = get_world_group()
        self.cpu_group, self.device_group = world.cpu_group, world.device_group

    def _prepare(self, prompts, sampling):
        from transformers import Qwen2TokenizerFast, Qwen3VLProcessor
        from vllm_omni.model_executor.models.minimax_h3.encoder_processing import (
            prepare_encoder_inputs,
        )
        from vllm_omni.model_executor.models.minimax_h3.preprocessing import (
            build_minimax_h3_presentation,
        )

        extra = sampling.extra_args or {}
        if sampling.lora_request is not None or extra.get("sampler", "euler") != "euler":
            raise ValueError("Phased H3 requires original weights and Euler sampling")
        if getattr(sampling, "quality", None) == "high":
            raise ValueError("Phased H3 does not support shared Cache-DiT hooks (quality=high)")
        if getattr(sampling, "num_outputs_per_prompt", 1) != 1:
            raise ValueError("Phased H3 requires one output per request")
        if extra.get("latent_upscale") and not self.additional_config.get("latent_upscaler_path"):
            raise ValueError("latent_upscale requires additional_config['latent_upscaler_path']")
        self._window_slices = []
        tokenizer = Qwen2TokenizerFast.from_pretrained(
            self.model, subfolder="tokenizer", local_files_only=True
        )
        processor = Qwen3VLProcessor.from_pretrained(
            self.model, subfolder="processor", local_files_only=True
        )
        prepared, inputs, tags = [], [], []
        for prompt in prompts:
            implicit_ref = Path(self.model).name.lower() == "ref2va" and extra.get("task") is None
            item = prepare_encoder_inputs(prompt, sampling, task="ref2va" if implicit_ref else None)
            expected = (
                {"ref2va"} if Path(self.model).name.lower() == "ref2va" else {"t2va", "fl2va"}
            )
            if item.media.task not in expected:
                raise ValueError(
                    f"Checkpoint {self.model} does not serve {item.media.task}; use its matching partition"
                )
            kwargs = {}
            if item.images:
                vision = processor.image_processor(images=item.images, return_tensors="pt")
                kwargs.update(
                    pixel_values=vision["pixel_values"], image_grid_thw=vision["image_grid_thw"]
                )
            if item.qwen_videos:
                vision = processor.video_processor(
                    videos=[frames for frames, _ in item.qwen_videos],
                    do_sample_frames=False,
                    return_tensors="pt",
                )
                kwargs.update(
                    pixel_values_videos=vision["pixel_values_videos"],
                    video_grid_thw=vision["video_grid_thw"],
                )
            window_prompts = self._continuation_prompts(item, sampling)
            start = len(inputs)
            for text in window_prompts:
                ids, token_tags = build_minimax_h3_presentation(
                    tokenizer,
                    prompt=text,
                    task=item.media.task,
                    condition_labels=item.condition_labels,
                    image_grid_thw=kwargs.get("image_grid_thw"),
                    video_grid_thw=kwargs.get("video_grid_thw"),
                    video_timestamps=item.video_timestamps,
                    merge_size=int(processor.image_processor.merge_size),
                )
                inputs.append((ids, kwargs))
                tags.append(token_tags.cpu())
            self._window_slices.append((start, len(inputs)))
            prepared.append(item)
        return prepared, inputs, tags

    @staticmethod
    def _continuation_prompts(item, sampling):
        from vllm_omni.diffusion.models.minimax_h3.continuation import (
            plan_continuation_windows,
            resolve_continuation,
        )

        extra = sampling.extra_args or {}
        mode = resolve_continuation(extra, task=item.media.task, step_execution=False)
        prompts = extra.get("continuation_prompts")
        if prompts is None:
            return [item.prompt]
        if mode is None:
            raise ValueError("continuation_prompts requires continuation mode")
        count = len(plan_continuation_windows(item.media.num_frames, *mode))
        if (
            not isinstance(prompts, list)
            or len(prompts) != count
            or any(not isinstance(text, str) or not text.strip() for text in prompts)
        ):
            raise ValueError(f"continuation_prompts requires exactly {count} nonempty strings")
        return prompts

    @staticmethod
    def _request_mode(sampling):
        extra = sampling.extra_args or {}
        return (
            any(
                extra.get(key)
                for key in (
                    "latent_refine",
                    "latent_upscale",
                    "continuation_prompts",
                    "long_video",
                )
            )
            or extra.get("long_video_mode") == "continuation"
        )

    def _encode_text(self, inputs, lengths, *, on_output=None, on_ready=None):
        from transformers import Qwen3VLConfig

        from .encoder_pp import QwenPPEncoder, run_encoder_pipeline

        path = Path(self.model) / "text_encoder"
        config = Qwen3VLConfig.from_pretrained(str(path), local_files_only=True)
        stage = QwenPPEncoder(config, self.encoder_stage_layers, self.rank, self.worker.device)

        def ready():
            if on_ready is not None:
                on_ready()
            self._mark("qwen_compute_start")

        try:
            stage.load_checkpoint(path)
            return run_encoder_pipeline(
                stage,
                inputs,
                lengths,
                self.device_group,
                self.cpu_group,
                on_output=on_output,
                on_ready=ready,
            )
        finally:
            # Frozen weights are discarded, never copied back to the host.
            del stage

    def _encode_media(self, prepared, hidden=None, tags=None):
        from vllm_omni.diffusion.models.minimax_h3.vae import (
            MiniMaxH3AudioVAE,
            MiniMaxH3VideoVAE,
        )
        from vllm_omni.model_executor.models.minimax_h3.encoder_processing import encode_media

        video_vae = audio_vae = None
        outputs = []
        try:
            if any(
                item.media.images
                or item.media.videos
                or item.media.video_edit is not None
                or any(guide.frames is not None for guide in item.media.timeline_guides)
                for item in prepared
            ):
                video_vae = MiniMaxH3VideoVAE(
                    str(Path(self.model) / "video_vae"),
                    device=self.worker.device,
                    encode_only=True,
                    trust_remote_code=self.trust_remote_code,
                )
            if any(
                item.media.audios
                or any(value is not None for value in item.media.video_audios)
                or item.media.audio_edit is not None
                or any(guide.audio is not None for guide in item.media.timeline_guides)
                for item in prepared
            ):
                audio_vae = MiniMaxH3AudioVAE(
                    str(Path(self.model) / "audio_vae"),
                    device=self.worker.device,
                    encode_only=True,
                    trust_remote_code=self.trust_remote_code,
                )
            for index, item in enumerate(prepared):
                media = encode_media(
                    item.media, video_vae=video_vae, audio_vae=audio_vae, emit_conditioning=True
                )
                media = cpu_tree(media)
                outputs.append(
                    media
                    if hidden is None
                    else self._conditioning_payload(hidden[index], tags[index], media)
                )
            return outputs
        finally:
            del video_vae, audio_vae

    @staticmethod
    def _conditioning_payload(states, tags, media):
        from vllm_omni.model_executor.models.minimax_h3.conditioning import (
            MiniMaxH3EncoderConditioning,
            MiniMaxH3TextConditioning,
        )

        conditioning = MiniMaxH3EncoderConditioning.from_components(
            MiniMaxH3TextConditioning(states, tags), media
        )
        if conditioning.guide_blocks:
            # Native Omni's stage wire omits timeline guides. Phased execution
            # stays in one ingress process, so retain the typed CPU condition.
            return {
                "additional_information": {"monowire_encoder_conditioning": cpu_tree(conditioning)}
            }
        return {
            "additional_information": {"encoder_output": cpu_tree(conditioning.to_omni_payload())}
        }

    def _denoise(self, encoded, sampling):
        from vllm.config import LoadConfig
        from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

        loader = DiffusersPipelineLoader(LoadConfig(), od_config=self.worker.od_config)
        runtime = engine = None
        try:
            self.worker.model_runner.pipeline = loader.load_model(
                load_device="cpu", device=self.worker.device
            )
            runtime = H3Runtime(self.worker, self.config)
            engine = ContinuousEngine(runtime, self.config)
            return engine.generate(encoded or [], sampling)
        finally:
            if runtime is not None:
                runtime.close(shutdown_worker=False)
            # Keep distributed groups alive for the next phase, drop all DiT owners.
            self.worker.model_runner.pipeline = None
            del engine, runtime, loader

    def _generate_overlap(self, prepared, inputs, tags, lengths, sampling):
        from vllm.config import LoadConfig
        from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

        from ...channels import CHANNEL_TIMEOUT, close_channels, create_channels
        from .rolling import EncoderInbox

        # FL2VA image conditions must exist before Qwen drain starts.
        media = self._phase(
            "vae_encode", lambda: self._leader(lambda: self._encode_media(prepared))
        )
        # These private job lists are also referenced by generate(). Clear their
        # contents at the last consumer instead of retaining images/pixels while
        # the full DiT checkpoint and conditioning tensors occupy host memory.
        if prepared is not None:
            prepared.clear()
        loader = DiffusersPipelineLoader(LoadConfig(), od_config=self.worker.od_config)
        channels = inbox = output_group = runtime = engine = None
        try:
            # Checkpoint I/O is outside the drain window. Only the local GPU copy
            # starts after this rank's final Qwen activation handoff completes.
            self._mark("dit_cpu_load_start")
            self.worker.model_runner.pipeline = loader.load_model(
                load_device="cpu", device=self.worker.device
            )
            if any(p.is_cuda for p in self.worker.model_runner.pipeline.transformer.parameters()):
                raise RuntimeError("DiT preload must leave all weights on CPU")
            self._mark("dit_cpu_ready")
            channels = create_channels(
                self.worker.model_runner.pipeline.transformer.topology,
                self.worker.device,
                async_transport=self.config.async_transport,
            )
            output_group = dist.new_group(backend="gloo", timeout=CHANNEL_TIMEOUT)
            if self.is_ingress:
                inbox = EncoderInbox(
                    lengths,
                    5120,
                    self.pp_size - 1,
                    output_group,
                    on_output=lambda index: self._mark("condition_received", request_id=str(index)),
                )

            def publish(index, tensor):
                self._mark("qwen_output", request_id=str(index))
                if self.pp_size == 1:
                    inbox.publish(index, tensor)
                else:
                    dist.send(tensor, dst=0, group=output_group)

            self._phase(
                "qwen_pp",
                lambda: self._encode_text(
                    inputs,
                    lengths,
                    on_output=publish,
                    on_ready=inbox.start if inbox is not None else None,
                ),
            )
            if inputs is not None:
                inputs.clear()
            self._mark("qwen_released", allocated_bytes=torch.cuda.memory_allocated())
            self._mark("dit_gpu_load_start")
            runtime = H3Runtime(self.worker, self.config, channels=channels)
            torch.cuda.synchronize(self.worker.device)
            self._mark("dit_gpu_ready")
            runtime.on_stage_event = self._dit_event
            engine = ContinuousEngine(runtime, self.config)
            engine.on_event = self._dit_event

            async def requests():
                if not self.is_ingress:
                    return
                async for index, hidden in inbox.tensors():
                    payload = self._conditioning_payload(hidden, tags[index], media[index])
                    tags[index] = None
                    media[index] = None
                    yield str(index), payload, sampling

            async def collect():
                return [output async for output in engine.stream(requests())]

            # No collective between local encoder release and DiT serving.
            outputs = self._phase(
                "dit", lambda: self._finish_dit(lambda: asyncio.run(collect())), barrier=False
            )
        finally:
            try:
                if inbox is not None:
                    inbox.close()
            finally:
                try:
                    if runtime is not None:
                        runtime.close(shutdown_worker=False)
                    elif channels is not None:
                        close_channels(channels)
                finally:
                    try:
                        if output_group is not None:
                            dist.destroy_process_group(output_group)
                    finally:
                        self.worker.model_runner.pipeline = None
                        engine = runtime = loader = None
                        gc.collect()
                        torch.cuda.empty_cache()
        self._mark("dit_weights_released", allocated_bytes=torch.cuda.memory_allocated())
        return (
            self._phase("vae_decode", lambda: self._leader(lambda: self._decode(outputs, sampling)))
            or []
        )

    def _refine_conditions(self, prepared, sampling):
        """Encode high-resolution FL2VA anchors before either large model loads."""
        from PIL import Image
        from vllm_omni.diffusion.models.minimax_h3.latent_upscaler import (
            parse_minimax_h3_latent_upscale_request,
            resolve_minimax_h3_latent_upscale_target,
        )
        from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

        extra = sampling.extra_args or {}
        spec = parse_minimax_h3_latent_upscale_request(extra.get("latent_upscale"))
        outputs = [None] * len(prepared)
        if not extra.get("latent_refine") or spec is None:
            return outputs
        vae = None
        try:
            for index, item in enumerate(prepared):
                if item.media.task != "fl2va":
                    continue
                target = resolve_minimax_h3_latent_upscale_target(
                    latent_height=item.media.height // 16,
                    latent_width=item.media.width // 16,
                    **spec,
                )
                if vae is None:
                    vae = MiniMaxH3VideoVAE(
                        str(Path(self.model) / "video_vae"),
                        device=self.worker.device,
                        encode_only=True,
                        trust_remote_code=self.trust_remote_code,
                    )
                rows = [
                    vae.encode_image(
                        image.resize(
                            (target.width, target.height),
                            Image.Resampling.LANCZOS,
                        )
                    )
                    for image in item.images
                ]
                outputs[index] = (
                    torch.cat(rows).cpu(),
                    [(1, target.latent_height, target.latent_width)] * len(rows),
                )
            return outputs
        finally:
            del vae

    def _generate_request_mode(self, prepared, inputs, tags, lengths, sampling):
        """Barrier-separated native multi-pass execution, with each DiT call using PP."""
        from vllm.config import LoadConfig
        from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

        from .request_mode import run_requests

        self._mark("request_mode", encoder_dit_overlap=False)
        media = self._phase(
            "vae_encode", lambda: self._leader(lambda: self._encode_media(prepared))
        )
        refine = self._phase(
            "vae_encode_refine",
            lambda: self._leader(lambda: self._refine_conditions(prepared, sampling)),
        )
        if prepared is not None:
            prepared.clear()
        hidden = self._phase("qwen_pp", lambda: self._encode_text(inputs, lengths))
        if inputs is not None:
            inputs.clear()

        def pack_conditions():
            encoded = []
            for index, (start, end) in enumerate(self._window_slices):
                payload = self._conditioning_payload(hidden[start], tags[start], media[index])
                additional = payload["additional_information"]
                if (sampling.extra_args or {}).get("continuation_prompts") is not None:
                    additional["monowire_window_text"] = list(
                        zip(hidden[start:end], tags[start:end])
                    )
                if refine[index] is not None:
                    additional["monowire_refine_condition"] = refine[index]
                encoded.append(payload)
            tags.clear()
            return encoded

        encoded = self._leader(pack_conditions)
        hidden = media = refine = None
        loader = DiffusersPipelineLoader(LoadConfig(), od_config=self.worker.od_config)
        runtime = None
        try:
            self.worker.model_runner.pipeline = loader.load_model(
                load_device="cpu",
                device=self.worker.device,
            )
            runtime = H3Runtime(self.worker, self.config)
            outputs = self._phase(
                "dit_request",
                lambda: self._finish_dit(lambda: run_requests(self, runtime, encoded, sampling)),
            )
        finally:
            try:
                if runtime is not None:
                    runtime.close(shutdown_worker=False)
            finally:
                self.worker.model_runner.pipeline = None
                runtime = loader = None
                gc.collect()
                torch.cuda.empty_cache()
        return (
            self._phase("vae_decode", lambda: self._leader(lambda: self._decode(outputs, sampling)))
            or []
        )

    def _decode(self, outputs, sampling):
        if not any(item.error is None for item in outputs):
            return outputs
        from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3_decoder import (
            MiniMaxH3DecoderPipeline,
        )
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

        decoder = MiniMaxH3DecoderPipeline(od_config=self.worker.od_config)
        try:
            for item in outputs:
                if item.error is not None:
                    continue
                value = item.output.output
                latents = value["payload"]["trajectory"]["latents"]
                payload = dict(
                    value["metadata"]["minimax_h3_decode"],
                    video_latents=latents["video"],
                    audio_latents=latents["audio"],
                )
                request = DiffusionRequestBatch(
                    requests=[
                        OmniDiffusionRequest(
                            request_id=item.request_id,
                            prompt={"additional_information": {"minimax_h3_decode": payload}},
                            sampling_params=sampling,
                        )
                    ],
                )
                item.output = decoder(request)
                item.output.output = cpu_tree(item.output.output)
            return outputs
        finally:
            del decoder

    def generate(self, prompts, sampling_params):
        if self.closed or self.running:
            raise RuntimeError("Phased engine must be open and idle")
        self.running = True
        self.phase_events = []
        self.timeline_events = []
        try:
            self._make_worker()
            # H3Runtime's context only needs the worker; no DiT is constructed here.
            with H3Runtime.context(self):
                job = self._leader(lambda: self._prepare(list(prompts), sampling_params))
                prepared, inputs, tags = job if self.is_ingress else (None, None, None)
                header = [[len(ids) for ids, _ in inputs] if self.is_ingress else None]
                dist.broadcast_object_list(header, src=0, group=self.cpu_group)
                if not header[0]:
                    return []
                if self._request_mode(sampling_params):
                    return self._generate_request_mode(
                        prepared, inputs, tags, header[0], sampling_params
                    )
                if self.overlap_encoder_dit:
                    return self._generate_overlap(
                        prepared, inputs, tags, header[0], sampling_params
                    )
                hidden = self._phase("qwen_pp", lambda: self._encode_text(inputs, header[0]))
                encoded = self._phase(
                    "vae_encode",
                    lambda: self._leader(lambda: self._encode_media(prepared, hidden, tags)),
                )
                job = prepared = inputs = tags = hidden = None
                outputs = self._phase(
                    "dit", lambda: self._finish_dit(lambda: self._denoise(encoded, sampling_params))
                )
                encoded = None
                return (
                    self._phase(
                        "vae_decode",
                        lambda: self._leader(lambda: self._decode(outputs, sampling_params)),
                    )
                    or []
                )
        finally:
            self.running = False
            if self.worker is not None:
                self.worker.shutdown()
                self.worker = None
            gc.collect()

    def close(self):
        if self.running:
            raise RuntimeError("Wait for the finite job before closing")
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def create_phased_engine(
    model,
    *,
    pp_size=8,
    config=None,
    cache=None,
    topology=None,
    encoder_stage_layers=None,
    trust_remote_code=False,
    overlap_encoder_dit=False,
    additional_config=None,
):
    """Construct a lazy finite-job engine for a local native H3 partition."""
    validate_runtime(pp_size)
    if int(os.environ.get("WORLD_SIZE", "1")) != pp_size:
        raise ValueError("Launch one torchrun process per pipeline stage")
    if topology is not None and (topology.world_size != pp_size or topology.ingress_replicas != 1):
        raise ValueError("Phased H3 currently requires one ingress and pp_size GPUs")
    root = Path(model)
    if root.name not in {"FL2VA", "Ref2VA"} or not (root / "model_index.json").is_file():
        raise ValueError("Phased H3 requires a local native FL2VA or Ref2VA partition directory")
    additional_config = dict(additional_config or {})
    if additional_config.get("latent_upscaler_resident"):
        raise ValueError("Phased H3 requires a nonresident latent upscaler")
    if additional_config.get("latent_refine") or additional_config.get("latent_upscale"):
        raise ValueError("Set latent_refine/latent_upscale per request in sampling.extra_args")
    return PhasedH3Engine(
        model,
        pp_size=pp_size,
        config=config or ContinuousConfig(),
        cache=cache or CacheConfig(),
        topology=topology,
        encoder_stage_layers=encoder_stage_layers,
        trust_remote_code=trust_remote_code,
        overlap_encoder_dit=overlap_encoder_dit,
        additional_config=additional_config,
    )
