"""Native vLLM-Omni H3 integration, using its loader, workers and PP group."""

from vllm.distributed.utils import get_pp_indices
from vllm.logger import init_logger
from vllm_omni.diffusion.distributed.parallel_state import get_pp_group
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
    MiniMaxH3DiTBlock,
    MiniMaxH3DiTModel,
)
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    MiniMaxH3Pipeline,
    get_minimax_h3_post_process_func,
)

from ...cache import CacheConfig, RequestCache
from ...partition import build_stage_blocks, local_weights

logger = init_logger("monowire")

# Re-exported for Omni's post-process registry lookup.
__all__ = ["MonoWireH3Pipeline", "get_minimax_h3_post_process_func"]


class MonoWireH3Transformer(MiniMaxH3DiTModel):
    def __init__(self, od_config, quant_config=None, **kwargs):
        if quant_config is not None:
            raise ValueError("Initial H3 PP integration requires unquantized original weights")
        super().__init__(od_config, quant_config=quant_config, **kwargs)
        options = od_config.cache_config.monowire
        config = CacheConfig(**options)
        self.request_cache = RequestCache(config, od_config.max_num_seqs)

    def build_transformer_blocks(self, arch, quant_config, attention_cls):
        self.pp_group = get_pp_group()
        self.start_layer, self.end_layer = get_pp_indices(
            arch.num_layers,
            self.pp_group.rank_in_group,
            self.pp_group.world_size,
        )

        logger.info(
            "MonoWire H3 PP rank %d/%d owns blocks [%d, %d) of %d",
            self.pp_group.rank_in_group,
            self.pp_group.world_size,
            self.start_layer,
            self.end_layer,
            arch.num_layers,
        )

        def factory(index):
            return MiniMaxH3DiTBlock(
                arch,
                quant_config,
                prefix=f"blocks.{index}",
                adaln_cache=self.adaln_cache,
                attention_cls=attention_cls,
            )

        return build_stage_blocks(
            arch.num_layers,
            self.start_layer,
            self.end_layer,
            factory,
        )

    def forward(self, **kwargs):
        raise RuntimeError("Use the continuous H3 request adapter to execute pipeline stages")

    def load_weights(self, weights):
        return super().load_weights(local_weights(weights, self.start_layer, self.end_layer))


class MonoWireH3Pipeline(MiniMaxH3Pipeline):
    _transformer_cls = MonoWireH3Transformer

    def __init__(self, *, od_config, prefix=""):
        parallel = od_config.parallel_config
        if not od_config.enforce_eager:
            raise ValueError("Native H3 PP currently requires enforce_eager=True")
        if od_config.lora_path:
            raise ValueError("Use the original H3 checkpoint without LoRA for this PP integration")
        if od_config.enable_layerwise_offload or od_config.enable_distributed_layerwise_offload:
            raise ValueError("Only component CPU offload is supported by this PP integration")
        # Fail before model allocation rather than silently producing a wrong hybrid plan.
        for field in (
            "tensor_parallel_size",
            "cfg_parallel_size",
            "ulysses_degree",
            "ring_degree",
        ):
            if getattr(parallel, field) != 1:
                raise ValueError(f"MonoWire H3 currently requires {field}=1")
        if parallel.use_hsdp:
            raise ValueError("HSDP is not qualified with MonoWire H3 PP")
        if od_config.cache_backend not in (None, "none"):
            raise ValueError(
                "Use MonoWire CacheConfig; upstream global cache hooks are not request-isolated"
            )
        super().__init__(od_config=od_config, prefix=prefix)

    def post_decode(self, state, **kwargs):
        from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
            _STEP_TRANSFORMER,
        )

        transformer = state.extra[_STEP_TRANSFORMER]
        try:
            return super().post_decode(state, **kwargs)
        finally:
            entry = transformer.request_cache.entries.pop(state.request_id, None)
            if entry is not None:
                logger.info(
                    "MonoWire cache request=%s rank=%d policy=%s hits=%d computes=%d",
                    state.request_id,
                    transformer.pp_group.rank_in_group,
                    transformer.request_cache.config.policy,
                    entry.hits,
                    entry.computes,
                )
