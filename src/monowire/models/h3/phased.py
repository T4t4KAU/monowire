"""Finite H3 jobs: Qwen PP, media encoding, DiT PP, then VAE decoding."""

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
    This baseline uses barriers, not overlapping encoder/DiT weight loading.
    """

    def __init__(
        self, model, *, pp_size, config, cache, topology, encoder_stage_layers, trust_remote_code
    ):
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

    def _phase(self, name, operation):
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
        unsupported = {"continuation_prompts", "continuation", "latent_refine", "latent_upscale"}
        if unsupported.intersection(extra):
            raise ValueError(
                "Phased H3 does not support continuation or latent refinement/upscaling"
            )
        tokenizer = Qwen2TokenizerFast.from_pretrained(
            self.model, subfolder="tokenizer", local_files_only=True
        )
        processor = Qwen3VLProcessor.from_pretrained(
            self.model, subfolder="processor", local_files_only=True
        )
        prepared, inputs, tags = [], [], []
        for prompt in prompts:
            item = prepare_encoder_inputs(prompt, sampling)
            if item.media.task not in {"t2va", "fl2va"}:
                raise ValueError("Phased H3 supports T2VA and FL2VA only")
            if (
                item.media.video_edit is not None
                or item.media.audio_edit is not None
                or item.media.timeline_guides
            ):
                raise ValueError("Phased H3 does not yet support editing or timeline guides")
            kwargs = {}
            if item.images:
                vision = processor.image_processor(images=item.images, return_tensors="pt")
                kwargs.update(
                    pixel_values=vision["pixel_values"], image_grid_thw=vision["image_grid_thw"]
                )
            if item.qwen_videos:
                raise ValueError("Phased FL2VA accepts first/last images, not reference videos")
            ids, token_tags = build_minimax_h3_presentation(
                tokenizer,
                prompt=item.prompt,
                task=item.media.task,
                condition_labels=item.condition_labels,
                image_grid_thw=kwargs.get("image_grid_thw"),
                video_grid_thw=None,
                video_timestamps=item.video_timestamps,
                merge_size=int(processor.image_processor.merge_size),
            )
            prepared.append(item)
            inputs.append((ids, kwargs))
            tags.append(token_tags.cpu())
        return prepared, inputs, tags

    def _encode_text(self, inputs, lengths):
        from transformers import Qwen3VLConfig

        from .encoder_pp import QwenPPEncoder, run_encoder_pipeline

        path = Path(self.model) / "text_encoder"
        config = Qwen3VLConfig.from_pretrained(str(path), local_files_only=True)
        stage = QwenPPEncoder(config, self.encoder_stage_layers, self.rank, self.worker.device)
        try:
            stage.load_checkpoint(path)
            return run_encoder_pipeline(stage, inputs, lengths, self.device_group, self.cpu_group)
        finally:
            # Frozen weights are discarded, never copied back to the host.
            del stage

    def _encode_media(self, prepared, hidden, tags):
        from vllm_omni.diffusion.models.minimax_h3.vae import (
            MiniMaxH3AudioVAE,
            MiniMaxH3VideoVAE,
        )
        from vllm_omni.model_executor.models.minimax_h3.conditioning import (
            MiniMaxH3EncoderConditioning,
            MiniMaxH3TextConditioning,
        )
        from vllm_omni.model_executor.models.minimax_h3.encoder_processing import encode_media

        video_vae = audio_vae = None
        outputs = []
        try:
            if any(item.media.images or item.media.videos for item in prepared):
                video_vae = MiniMaxH3VideoVAE(
                    str(Path(self.model) / "video_vae"),
                    device=self.worker.device,
                    encode_only=True,
                    trust_remote_code=self.trust_remote_code,
                )
            if any(item.media.audios or any(item.media.video_audios) for item in prepared):
                audio_vae = MiniMaxH3AudioVAE(
                    str(Path(self.model) / "audio_vae"),
                    device=self.worker.device,
                    encode_only=True,
                    trust_remote_code=self.trust_remote_code,
                )
            for item, states, token_tags in zip(prepared, hidden, tags, strict=True):
                media = encode_media(
                    item.media, video_vae=video_vae, audio_vae=audio_vae, emit_conditioning=True
                )
                conditioning = MiniMaxH3EncoderConditioning.from_components(
                    MiniMaxH3TextConditioning(states, token_tags), media
                )
                outputs.append(
                    {
                        "additional_information": {
                            "encoder_output": cpu_tree(conditioning.to_omni_payload()),
                        }
                    }
                )
            return outputs
        finally:
            del video_vae, audio_vae

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

    def _decode(self, outputs, sampling):
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
                hidden = self._phase("qwen_pp", lambda: self._encode_text(inputs, header[0]))
                encoded = self._phase(
                    "vae_encode",
                    lambda: self._leader(lambda: self._encode_media(prepared, hidden, tags)),
                )
                job = prepared = inputs = tags = hidden = None
                outputs = self._phase("dit", lambda: self._denoise(encoded, sampling_params))
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
):
    """Construct a lazy finite-job engine for a local native FL2VA checkpoint."""
    validate_runtime(pp_size)
    if int(os.environ.get("WORLD_SIZE", "1")) != pp_size:
        raise ValueError("Launch one torchrun process per pipeline stage")
    if topology is not None and (topology.world_size != pp_size or topology.ingress_replicas != 1):
        raise ValueError("Phased H3 currently requires one ingress and pp_size GPUs")
    root = Path(model)
    if root.name != "FL2VA" or not (root / "model_index.json").is_file():
        raise ValueError("Phased H3 requires a local native FL2VA partition directory")
    return PhasedH3Engine(
        model,
        pp_size=pp_size,
        config=config or ContinuousConfig(),
        cache=cache or CacheConfig(),
        topology=topology,
        encoder_stage_layers=encoder_stage_layers,
        trust_remote_code=trust_remote_code,
    )
