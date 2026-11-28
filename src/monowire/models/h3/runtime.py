"""Compose the H3 adapter with the generic continuous engine."""

import os
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace

import torch
import torch.distributed as dist

from ...cache import CacheConfig
from ...channels import close_channels, create_channels
from ...continuous import ContinuousConfig
from ...continuous_engine import ContinuousEngine
from ...device_memory import DeviceMemoryLimits, DeviceMemoryPool
from ...transport import PipelineExecutor, serve_stage
from .adapter import H3RequestAdapter, H3StageAdapter
from .validation import validate_runtime


class H3Runtime:
    def __init__(self, worker, config, *, channels=None):
        self.worker = worker
        self.config = config
        self.pipeline = worker.model_runner.pipeline
        self.stage = H3StageAdapter(self.pipeline.transformer)
        limits = config.memory_limits
        adaln_cache = self.stage.transformer.adaln_cache
        invalid_adaln = adaln_cache is not None and adaln_cache.sidecar is not None
        if invalid_adaln and limits is None:
            raise ValueError("Continuous H3 requires the request-independent AdaLN runtime cache")
        # Place DiT once at startup; codecs cannot evict or move these weights.
        self.stage.transformer.to(self.pipeline.device)
        self.rank = self.stage.group.rank_in_group
        self.topology = self.stage.transformer.topology
        self.memory_pool = None
        if limits is not None:
            local_errors = []
            if invalid_adaln:
                local_errors.append(
                    "Continuous H3 requires the request-independent AdaLN runtime cache"
                )
            if self.topology.ingress_replicas != 1:
                local_errors.append("Byte residency currently requires one ingress")
            if self.pipeline.load_text_encoder or self.pipeline.load_vae_encoder:
                local_errors.append("Byte residency requires phased/pre-encoded DiT inputs")
            if self.stage.transformer.request_cache.config.policy != "none":
                local_errors.append("Byte residency currently requires cache policy none")
            if torch.device(self.pipeline.device).type == "cuda":
                free, total = torch.cuda.mem_get_info(self.pipeline.device)
                reclaimable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
                required = (
                    limits.workspace_bytes + limits.headroom_bytes + limits.progress_bytes
                    if isinstance(limits, DeviceMemoryLimits)
                    else limits.dynamic_bytes - (limits.request_bytes if self.rank else 0)
                )
                if required > free + reclaimable:
                    local_errors.append(
                        f"Byte policy requires {required} free bytes including workspace/headroom; "
                        f"only {free + reclaimable} available on rank {self.rank}"
                    )
            errors = [None] * self.topology.world_size
            error = "; ".join(local_errors) or None
            dist.all_gather_object(errors, error, group=self.stage.group.cpu_group)
            if any(errors):
                raise ValueError("; ".join(e for e in errors if e))
            if isinstance(limits, DeviceMemoryLimits):
                try:
                    self.memory_pool = DeviceMemoryPool(self.pipeline.device, limits)
                    error = None
                except Exception as exc:
                    error = str(exc)
                dist.all_gather_object(errors, error, group=self.stage.group.cpu_group)
                if any(errors):
                    raise ValueError("; ".join(e for e in errors if e))
                self.pipeline.monowire_memory_pool = self.memory_pool
                self.topology = replace(self.topology, buffer_bytes=None)
            else:
                self.topology = replace(self.topology, buffer_bytes=limits.buffer_bytes)
            self.stage.transformer.topology = self.topology
            self.pipeline.monowire_memory_limits = limits
        self.ingress_replicas = self.topology.ingress_replicas
        self.channels = (
            channels
            if channels is not None
            else create_channels(
                self.topology,
                self.pipeline.device,
                async_transport=config.async_transport,
                memory_pool=self.memory_pool,
            )
        )
        self.on_stage_event = None
        if limits is not None:
            for channel in self.channels.values():
                channel.dictionary.max_bytes = limits.dictionary_bytes

    @contextmanager
    def context(self):
        from vllm.config import set_current_vllm_config
        from vllm_omni.diffusion.forward_context import set_forward_context

        with ExitStack() as stack:
            stack.enter_context(torch.inference_mode())
            stack.enter_context(set_current_vllm_config(self.worker.vllm_config))
            stack.enter_context(
                set_forward_context(
                    vllm_config=self.worker.vllm_config, omni_diffusion_config=self.worker.od_config
                )
            )
            yield

    def create_adapter(self):
        return H3RequestAdapter(self.pipeline)

    def create_executor(self):
        return PipelineExecutor(self.stage, self.topology, self.channels)

    def serve(self):
        serve_stage(
            self.stage,
            self.topology,
            self.channels,
            batch_size=self.config.downstream_batch_size,
            on_event=self.on_stage_event,
            async_transport=self.config.async_transport,
        )

    def close(self, *, shutdown_worker=True):
        try:
            if self.channels is not None:
                close_channels(self.channels)
        finally:
            if self.memory_pool is not None:
                self.memory_pool.drain()
            self.channels = None
            if shutdown_worker:
                self.worker.shutdown()


def create_continuous_engine(
    model,
    *,
    pp_size=8,
    config=None,
    cache=None,
    topology=None,
    cpu_offload=True,
    trust_remote_code=False,
):
    """Construct one continuous inference worker per torchrun pipeline rank."""
    validate_runtime(pp_size)
    config = config or ContinuousConfig()
    cache = cache or CacheConfig()
    if topology is not None and topology.world_size != pp_size:
        raise ValueError("Topology must use exactly pp_size GPUs")
    if int(os.environ.get("WORLD_SIZE", "1")) != pp_size:
        raise ValueError("Launch one torchrun process per pipeline stage")
    from .plugin import register

    # Enable H3 registration before the runtime first imports its plugins.
    os.environ["MONOWIRE_H3_PP"] = "1"
    register()
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

    ingress_replicas = topology.ingress_replicas if topology is not None else 1
    is_ingress = int(os.environ.get("RANK", "0")) < ingress_replicas
    od_config = OmniDiffusionConfig.from_kwargs(
        model=model,
        model_class_name="MiniMaxH3Pipeline",
        trust_remote_code=trust_remote_code,
        parallel_config=DiffusionParallelConfig(pipeline_parallel_size=pp_size),
        enforce_eager=True,
        cache_backend="none",
        step_execution=True,
        max_num_seqs=config.max_active_requests or 1,
        enable_cpu_offload=cpu_offload,
        model_loaded={
            "text_encoder": is_ingress,
            "vae_encoder": is_ingress,
            "vae_decoder": is_ingress,
        },
        cache_config={
            "monowire": asdict(cache),
            "monowire_topology": asdict(topology) if topology is not None else None,
        },
    )
    worker = DiffusionWorker(
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        rank=int(os.environ.get("RANK", "0")),
        od_config=od_config,
        distributed_init_method="env://",
    )
    try:
        return ContinuousEngine(H3Runtime(worker, config), config)
    except BaseException:
        worker.shutdown()
        raise
