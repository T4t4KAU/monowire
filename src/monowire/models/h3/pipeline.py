"""Native vLLM-Omni H3 integration, using its loader, workers and PP group."""

from contextlib import contextmanager

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
from vllm_omni.diffusion.models.minimax_h3.vae import _VideoVAEPartProxy

from ...cache import CacheConfig, RequestCache
from ...partition import build_stage_blocks, local_weights
from ...topology import PipelineTopology
from .io import CodecOffloader

logger = init_logger("monowire")

# Re-exported for Omni's post-process registry lookup.
__all__ = ["MonoWireH3Pipeline", "get_minimax_h3_post_process_func"]


class MonoWireH3Transformer(MiniMaxH3DiTModel):
    def __init__(self, od_config, quant_config=None, **kwargs):
        if quant_config is not None:
            raise ValueError("Initial H3 PP integration requires unquantized original weights")
        self.pp_group = get_pp_group()
        topology = od_config.cache_config.monowire_topology
        self.topology = PipelineTopology(**topology) if topology is not None else None
        replicas = self.topology.ingress_replicas if self.topology is not None else 1
        self.is_ingress = self.pp_group.rank_in_group < replicas
        super().__init__(
            od_config, quant_config=quant_config, load_io_modules=self.is_ingress, **kwargs
        )
        options = od_config.cache_config.monowire
        config = CacheConfig(**options)
        self.request_cache = RequestCache(config, od_config.max_num_seqs)

    def build_transformer_blocks(self, arch, quant_config, attention_cls):
        if self.topology is None:
            ranges = [
                get_pp_indices(arch.num_layers, rank, self.pp_group.world_size)
                for rank in range(self.pp_group.world_size)
            ]
            self.topology = PipelineTopology(tuple(end - start for start, end in ranges))
        if sum(self.topology.stage_layers) != arch.num_layers:
            raise ValueError("Topology must cover every transformer block exactly once")
        if self.topology.world_size != self.pp_group.world_size:
            raise ValueError("Topology does not match the distributed world")
        self.start_layer, self.end_layer = self.topology.layer_range(self.pp_group.rank_in_group)

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
        return super().load_weights(
            local_weights(weights, self.start_layer, self.end_layer, include_io=self.is_ingress)
        )


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
        self._codec_offloader = None

    def _io_rank_world(self):
        """Each ingress owns its codecs; DiT ranks never join codec collectives."""
        return None, 0, 1

    @staticmethod
    def _is_output_owner_rank():
        return True

    def enable_omni_model_cpu_offload(
        self, *, device, pin_memory, use_hsdp, offload_components=None
    ):
        """Offload codec components only; in-flight DiT weights stay resident."""
        from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery

        if use_hsdp:
            raise ValueError("Codec offload does not support HSDP")
        if offload_components is not None:
            raise ValueError("MonoWire manages codec offload as a single I/O lifecycle")
        if self._codec_offloader is not None:
            raise RuntimeError("Codec offload is already enabled")
        components = ModuleDiscovery.discover(self)
        stages = [*components.encoders, *components.vaes]
        if self.latent_upscaler is not None:
            stages.append(self.latent_upscaler)
        self._codec_offloader = CodecOffloader(stages, device, pin_memory=pin_memory)

    def disable_omni_model_cpu_offload(self):
        if self._codec_offloader is not None and self._codec_offloader.active is not None:
            raise RuntimeError("Drain codec work before disabling offload")
        self._codec_offloader = None

    @contextmanager
    def _component_on_device(self, component):
        if self._codec_offloader is None:
            with super()._component_on_device(component):
                yield
            return
        if isinstance(component, _VideoVAEPartProxy):
            component = component.sequential_offload_target
        with self._codec_offloader.activate(component):
            yield

    def _encode_text_hidden(self, input_ids, vision_kwargs):
        if self._codec_offloader is None:
            return super()._encode_text_hidden(input_ids, vision_kwargs)
        with self._component_on_device(self.text_encoder):
            return self.text_encoder.encode_ids(input_ids, **vision_kwargs)

    def _offload_model_cpu_stage_output(self, tensor):
        return tensor.cpu() if self._codec_offloader is not None else tensor

    def _release_stage_cache(self):
        # Other requests still own live DiT work. Do not flush the device
        # allocator at a per-request codec boundary.
        return
