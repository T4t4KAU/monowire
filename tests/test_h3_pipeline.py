"""H3 integration boundaries; no model weights or CUDA allocation required."""

import importlib.util
import unittest
from contextlib import ExitStack
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import torch

from monowire.cache import CacheConfig


class TinyBlock(torch.nn.Module):
    """Keep real named parameters for checkpoint loading without allocating H3."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.out_proj = torch.nn.Linear(2, 2)


@unittest.skipUnless(importlib.util.find_spec("vllm_omni"), "Requires bundled vLLM-Omni")
class H3PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

        from monowire.models.h3.pipeline import MonoWireH3Pipeline, MonoWireH3Transformer

        cls.native = MiniMaxH3Pipeline
        cls.pipeline_class = MonoWireH3Pipeline
        cls.transformer_class = MonoWireH3Transformer

    def build_tiny_transformer(self, rank=None):
        from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as native

        config = SimpleNamespace(
            tf_model_config={"num_layers": 4},
            parallel_config=SimpleNamespace(ulysses_degree=1, use_hsdp=False),
            cache_config=SimpleNamespace(
                monowire_topology={"stage_layers": (2, 2), "ingress_replicas": 2},
                monowire=asdict(CacheConfig()),
                minimax_h3_adaln_offload=False,
            ),
            max_num_seqs=8,
        )

        def io_module(*args, **kwargs):
            if rank == 2:
                raise AssertionError("Suffix allocated an ingress-only module")
            return torch.nn.Linear(2, 2)

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(native, "get_tensor_model_parallel_world_size", return_value=1)
            )
            stack.enter_context(patch.object(native, "MiniMaxH3DiTBlock", TinyBlock))
            stack.enter_context(patch("monowire.models.h3.pipeline.MiniMaxH3DiTBlock", TinyBlock))
            stack.enter_context(
                patch(
                    "monowire.models.h3.pipeline.get_pp_group",
                    return_value=SimpleNamespace(rank_in_group=rank, world_size=3),
                )
            )
            for name in (
                "ColumnParallelLinear",
                "MiniMaxH3TimeEmbedder",
                "MiniMaxH3TokenRefiner",
                "MiniMaxH3FinalLayer",
            ):
                stack.enter_context(patch.object(native, name, side_effect=io_module))
            cls = native.MiniMaxH3DiTModel if rank is None else self.transformer_class
            return cls(config)

    def test_only_ingress_constructs_io_and_loads_its_checkpoint_parameters(self):
        full = self.build_tiny_transformer()
        checkpoint = [
            (key, torch.full_like(value, index + 1))
            for index, (key, value) in enumerate(full.state_dict().items())
        ]
        io_names = {
            "video_patch_proj",
            "audio_patch_proj",
            "condition_proj",
            "time_embedder",
            "token_refiner",
            "rope",
            "final_layer",
        }
        self.assertTrue(io_names.issubset(dict(full.named_children())))
        for rank in range(3):
            with self.subTest(rank=rank):
                model = self.build_tiny_transformer(rank)
                children = dict(model.named_children())
                if rank < 2:
                    self.assertTrue(io_names.issubset(children))
                    expected = {
                        key: value
                        for key, value in checkpoint
                        if not key.startswith(("blocks.2.", "blocks.3."))
                    }
                else:
                    self.assertTrue(io_names.isdisjoint(children))
                    expected = {
                        key: value
                        for key, value in checkpoint
                        if key.startswith(("blocks.2.", "blocks.3."))
                    }
                self.assertEqual(model.load_weights(checkpoint), set(expected))
                self.assertEqual(set(model.state_dict()), set(expected))
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, expected[key])

    def test_diffusers_suffix_load_does_not_reconstruct_unowned_rope(self):
        model = self.build_tiny_transformer(2)
        model._diffusers_weights = True
        checkpoint = [("proj_in.weight", torch.zeros(2, 2))]
        for block in range(4):
            checkpoint.extend(
                [
                    (
                        f"transformer_blocks.{block}.attn.to_out.0.weight",
                        torch.full((2, 2), float(block)),
                    ),
                    (
                        f"transformer_blocks.{block}.attn.to_out.0.bias",
                        torch.full((2,), float(block)),
                    ),
                ]
            )
        self.assertEqual(model.load_weights(checkpoint), set(model.state_dict()))
        for block in (2, 3):
            torch.testing.assert_close(
                model.blocks[block].attn.out_proj.weight, torch.full((2, 2), float(block))
            )

    def test_codec_scope_is_per_instance_and_does_not_change_native_collectives(self):
        native = object.__new__(self.native)
        ingress = object.__new__(self.pipeline_class)
        group = object()
        module = "vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3"
        with (
            patch(f"{module}.dist.is_initialized", return_value=True),
            patch(f"{module}.get_world_group", return_value=SimpleNamespace(device_group=group)),
            patch(f"{module}.dist.get_rank", return_value=3),
            patch(f"{module}.dist.get_world_size", return_value=8),
            patch(f"{module}.dist.broadcast", side_effect=AssertionError("unexpected collective")),
        ):
            self.assertEqual(native._io_rank_world(), (group, 3, 8))
            self.assertEqual(ingress._io_rank_world(), (None, 0, 1))
            value = torch.ones(2)
            torch.testing.assert_close(
                ingress._broadcast_tensor(value, dtype=torch.float32, device=torch.device("cpu")),
                value,
            )
            with self.assertRaisesRegex(ValueError, "encode failed"):
                ingress._broadcast_rank0_exception(ValueError("encode failed"))
            self.assertEqual(native._io_rank_world(), (group, 3, 8))
            self.assertTrue(ingress._is_output_owner_rank())

    def test_offload_never_registers_or_moves_dit_weights(self):
        pipeline = object.__new__(self.pipeline_class)
        torch.nn.Module.__init__(pipeline)
        pipeline.transformer = torch.nn.Linear(2, 2)
        pipeline.latent_upscaler = None
        pipeline._codec_offloader = None
        codec = torch.nn.Linear(2, 2)
        with (
            patch(
                "vllm_omni.diffusion.offloader.module_collector.ModuleDiscovery.discover",
                return_value=SimpleNamespace(
                    dits=[pipeline.transformer], encoders=[codec], vaes=[]
                ),
            ),
            patch("monowire.models.h3.pipeline.CodecOffloader") as install,
        ):
            pipeline.enable_omni_model_cpu_offload(
                device=torch.device("cpu"), pin_memory=False, use_hsdp=False
            )
        install.assert_called_once_with([codec], torch.device("cpu"), pin_memory=False)
        self.assertIs(pipeline._codec_offloader, install.return_value)

    def test_default_partition_constructs_the_same_topology_contract(self):
        from monowire.topology import PipelineTopology

        transformer = object.__new__(self.transformer_class)
        torch.nn.Module.__init__(transformer)
        transformer.topology = None
        transformer.adaln_cache = None
        transformer.pp_group = SimpleNamespace(rank_in_group=0, world_size=8)
        with (
            patch("monowire.models.h3.pipeline.build_stage_blocks", return_value=None),
        ):
            transformer.build_transformer_blocks(SimpleNamespace(num_layers=50), None, None)
        self.assertIsInstance(transformer.topology, PipelineTopology)
        self.assertEqual(transformer.topology.world_size, 8)
        self.assertEqual(sum(transformer.topology.stage_layers), 50)
        self.assertEqual(
            (transformer.start_layer, transformer.end_layer), transformer.topology.layer_range(0)
        )

    def test_text_and_vae_proxy_use_scoped_residency_without_upstream_hooks(self):
        from vllm_omni.diffusion.models.minimax_h3.vae import _VideoVAEPartProxy

        pipeline = object.__new__(self.pipeline_class)
        torch.nn.Module.__init__(pipeline)
        pipeline._codec_offloader = None
        pipeline.latent_upscaler = None
        pipeline.transformer = torch.nn.Linear(2, 2)

        class Encoder(torch.nn.Linear):
            def encode_ids(encoder, ids, **kwargs):
                self.assertIs(pipeline._codec_offloader.active, encoder)
                return super(Encoder, encoder).forward(ids)

        pipeline.text_encoder = Encoder(2, 2)
        pipeline.video_vae = torch.nn.Linear(2, 2)
        with (
            patch(
                "vllm_omni.diffusion.offloader.module_collector.ModuleDiscovery.discover",
                return_value=SimpleNamespace(
                    dits=[pipeline.transformer],
                    encoders=[pipeline.text_encoder],
                    vaes=[pipeline.video_vae],
                ),
            ),
            patch(
                "vllm_omni.diffusion.offloader.apply_sequential_offload",
                side_effect=AssertionError("global offload hooks installed"),
            ),
        ):
            pipeline.enable_omni_model_cpu_offload(
                device=torch.device("cpu"), pin_memory=False, use_hsdp=False
            )
        with self.assertRaisesRegex(ValueError, "not owned"):
            with pipeline._codec_offloader.activate(pipeline.transformer):
                self.fail("DiT must not be offloaded")
        result = pipeline._encode_text_hidden(torch.ones(1, 2), {})
        self.assertEqual(result.shape, (1, 2))
        self.assertIsNone(pipeline._codec_offloader.active)
        for part in ("encoder", "decoder"):
            with self.assertRaisesRegex(ValueError, "decode failed"):
                with pipeline._component_on_device(_VideoVAEPartProxy(pipeline.video_vae, part)):
                    self.assertIs(pipeline._codec_offloader.active, pipeline.video_vae)
                    with self.assertRaisesRegex(RuntimeError, "Drain codec"):
                        pipeline.disable_omni_model_cpu_offload()
                    raise ValueError("decode failed")
            self.assertIsNone(pipeline._codec_offloader.active)
        pipeline.disable_omni_model_cpu_offload()
        self.assertIsNone(pipeline._codec_offloader)
