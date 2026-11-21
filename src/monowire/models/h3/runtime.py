"""Compose the H3 adapter with the generic continuous engine."""

import os
from contextlib import ExitStack, contextmanager
from dataclasses import asdict

import torch

from ...cache import CacheConfig
from ...continuous import ContinuousConfig
from ...continuous_engine import ContinuousEngine
from ...transport import PipelineExecutor, serve_stage
from .adapter import H3RequestAdapter, H3StageAdapter
from .validation import validate_runtime


class H3Runtime:
    def __init__(self, worker):
        self.worker = worker
        self.pipeline = worker.model_runner.pipeline
        self.stage = H3StageAdapter(self.pipeline._transformer_for_task("t2va"))
        self.rank = self.stage.group.rank_in_group

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
        return PipelineExecutor(self.stage)

    def serve(self):
        serve_stage(self.stage)

    def close(self):
        self.worker.shutdown()


def create_continuous_engine(
    model, *, pp_size=8, config=None, cache=None, cpu_offload=True, trust_remote_code=False
):
    """Construct one continuous inference worker per torchrun pipeline rank."""
    validate_runtime(pp_size)
    config = config or ContinuousConfig()
    cache = cache or CacheConfig("first_block")
    if cache.policy not in ("none", "first_block"):
        raise ValueError("Continuous H3 currently supports none or first_block")
    if int(os.environ.get("WORLD_SIZE", "1")) != pp_size:
        raise ValueError("Launch one torchrun process per pipeline stage")
    from .plugin import register

    # Enable H3 registration before the runtime first imports its plugins.
    os.environ["MONOWIRE_H3_PP"] = "1"
    register()
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

    od_config = OmniDiffusionConfig.from_kwargs(
        model=model,
        model_class_name="MiniMaxH3Pipeline",
        trust_remote_code=trust_remote_code,
        parallel_config=DiffusionParallelConfig(pipeline_parallel_size=pp_size),
        enforce_eager=True,
        cache_backend="none",
        step_execution=True,
        max_num_seqs=config.max_active_requests,
        enable_cpu_offload=cpu_offload,
        cache_config={"monowire": asdict(cache)},
    )
    worker = DiffusionWorker(
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        rank=int(os.environ.get("RANK", "0")),
        od_config=od_config,
        distributed_init_method="env://",
    )
    return ContinuousEngine(H3Runtime(worker), config)
