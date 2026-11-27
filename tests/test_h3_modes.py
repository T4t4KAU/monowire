"""Mode routing, codec residency and native multi-pass PP contracts."""

import importlib.util
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from monowire.models.h3.phased import PhasedH3Engine
from monowire.models.h3.request_mode import RequestPipeline, native_request_context


class Channel:
    def __init__(self, messages=()):
        self.messages = iter(messages)
        self.sent = []

    def recv(self):
        message = next(self.messages)
        return NS(on_current_stream=lambda: message)

    def send(self, message, ready=None):
        self.sent.append(message)


def failed_service_worker(rank, store):
    dist.init_process_group(
        "gloo",
        init_method=Path(store).as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=15),
    )
    engine = object.__new__(PhasedH3Engine)
    engine.rank, engine.pp_size, engine.cpu_group = rank, 4, dist.group.WORLD
    try:
        for failing_rank in (0, 2):

            def operation():
                if rank == failing_rank:
                    raise ValueError("injected service failure")
                return []

            try:
                engine._finish_dit(operation)
            except RuntimeError as error:
                assert f"rank {failing_rank}: ValueError: injected service failure" in str(error)
            else:
                raise AssertionError("A peer entered the decode phase after service failure")
        assert engine._finish_dit(lambda: rank) == rank
    finally:
        dist.destroy_process_group()


def ring_worker(rank, store):
    from monowire.channels import close_channels, create_channels
    from monowire.topology import PipelineTopology

    dist.init_process_group(
        "gloo",
        init_method=Path(store).as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    topology = PipelineTopology((1, 1, 1, 1))
    channels = create_channels(topology, "cpu")

    class Block(torch.nn.Module):
        def forward(self, hidden, **kwargs):
            if rank == 2 and hidden.shape[0] == 3:
                raise ValueError("injected second-pass failure")
            return hidden + rank + 1

    transformer = NS(
        start_layer=rank,
        end_layer=rank + 1,
        adaln_cache=None,
        pp_group=NS(rank_in_group=rank),
        blocks={rank: Block()},
        prepare_transformer_forward=lambda **kw: (kw["hidden"], {}, {}),
        finish_transformer_forward=lambda hidden, **kw: (hidden, hidden * 2),
    )
    runtime = NS(
        rank=rank,
        topology=topology,
        channels=channels,
        stage=NS(transformer=transformer),
        pipeline=NS(device="cpu"),
    )
    bridge = RequestPipeline(runtime)
    try:
        if rank:
            bridge.serve()
        else:
            for length in (2, 3, 4):
                try:
                    video, audio = bridge.forward(hidden=torch.zeros(length, 2))
                except RuntimeError as error:
                    assert length == 3 and "second-pass failure" in str(error)
                else:
                    assert length != 3
                    torch.testing.assert_close(video, torch.full((length, 2), 10.0))
                    torch.testing.assert_close(audio, torch.full((length, 2), 20.0))
            bridge.close()
    finally:
        close_channels(channels)
        dist.destroy_process_group()


class ModeTests(unittest.TestCase):
    def test_all_ranks_observe_service_failure_before_decoder_collectives(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                failed_service_worker, args=(str(Path(directory, "store")),), nprocs=4, join=True
            )

    def test_real_four_rank_ring_handles_shape_changes_failure_and_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(ring_worker, args=(str(Path(directory, "store")),), nprocs=4, join=True)

    def test_multi_pass_modes_select_request_lane(self):
        for extra in (
            {"latent_refine": 0.5},
            {"latent_upscale": 2},
            {"long_video_mode": "continuation"},
            {"long_video": True},
            {"continuation_prompts": ["a", "b"]},
        ):
            with self.subTest(extra=extra):
                self.assertTrue(PhasedH3Engine._request_mode(NS(extra_args=extra)))
        for extra in ({}, {"latent_refine": False}, {"task": "ref2va"}):
            self.assertFalse(PhasedH3Engine._request_mode(NS(extra_args=extra)))

    def test_prepare_rejects_unsupported_options_before_tokenizer_loading(self):
        engine = object.__new__(PhasedH3Engine)
        engine.model = "FL2VA"
        engine.additional_config = {}

        class Factory:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise AssertionError("Unsupported sampling reached tokenizer loading")

        with patch.dict(
            sys.modules,
            {
                "transformers": NS(Qwen2TokenizerFast=Factory, Qwen3VLProcessor=Factory),
                "vllm_omni.model_executor.models.minimax_h3.encoder_processing": NS(
                    prepare_encoder_inputs=None
                ),
                "vllm_omni.model_executor.models.minimax_h3.preprocessing": NS(
                    build_minimax_h3_presentation=None
                ),
            },
        ):
            for values in (
                {"quality": "high"},
                {"num_outputs_per_prompt": 2},
                {"extra_args": {"latent_upscale": 2}},
            ):
                sampling = NS(
                    **dict(
                        dict(
                            extra_args={}, quality=None, num_outputs_per_prompt=1, lora_request=None
                        ),
                        **values,
                    )
                )
                with self.subTest(values=values), self.assertRaises(ValueError):
                    engine._prepare(["prompt"], sampling)

    def test_reference_video_grids_and_window_text_keep_original_order(self):
        engine = object.__new__(PhasedH3Engine)
        engine.model = "Ref2VA"
        engine.additional_config = {}
        engine._continuation_prompts = lambda item, sampling: ["first window", "second window"]
        item = NS(
            media=NS(task="ref2va"),
            images=[],
            qwen_videos=[("frames", {})],
            video_timestamps=[[0.0, 0.5]],
            condition_labels=["video"],
            prompt="unused",
        )
        captures = []

        def presentation(tokenizer, **kwargs):
            captures.append(kwargs)
            return torch.ones(len(captures), dtype=torch.long), torch.ones(
                len(captures), dtype=torch.long
            )

        processor = NS(
            video_processor=lambda **kwargs: {
                "pixel_values_videos": torch.ones(2),
                "video_grid_thw": torch.tensor([[2, 4, 4]]),
            }
        )
        processor.image_processor = NS(merge_size=2)
        factory = NS(from_pretrained=lambda *a, **kw: processor)
        with patch.dict(
            sys.modules,
            {
                "transformers": NS(Qwen2TokenizerFast=factory, Qwen3VLProcessor=factory),
                "vllm_omni.model_executor.models.minimax_h3.encoder_processing": NS(
                    prepare_encoder_inputs=lambda *a, **kw: item
                ),
                "vllm_omni.model_executor.models.minimax_h3.preprocessing": NS(
                    build_minimax_h3_presentation=presentation
                ),
            },
        ):
            prepared, inputs, tags = engine._prepare(
                ["prompt"], NS(extra_args={}, lora_request=None)
            )
        self.assertEqual(engine._window_slices, [(0, 2)])
        self.assertEqual([x["prompt"] for x in captures], ["first window", "second window"])
        self.assertEqual(captures[0]["video_timestamps"], [[0.0, 0.5]])
        self.assertEqual(captures[0]["video_grid_thw"].tolist(), [[2, 4, 4]])
        self.assertEqual(len(prepared), 1)
        self.assertEqual(len(inputs), len(tags))

    def test_edit_and_guide_sources_load_required_codecs(self):
        engine = object.__new__(PhasedH3Engine)
        engine.model = "FL2VA"
        engine.worker = NS(device="cpu")
        engine.trust_remote_code = False
        for field, value, expected in (
            ("video_edit", torch.ones(1), {"video"}),
            ("audio_edit", (torch.ones(1), 24000), {"audio"}),
            (
                "timeline_guides",
                [NS(frames=torch.ones(1), audio=(torch.ones(1), 24000))],
                {"video", "audio"},
            ),
            ("video_audios", [(torch.ones(4), 24000)], {"audio"}),
        ):
            loaded = set()

            def load(name):
                def factory(*args, **kwargs):
                    loaded.add(name)
                    return name

                return factory

            media = NS(
                images=[],
                videos=[],
                video_edit=None,
                audios=[],
                video_audios=[],
                audio_edit=None,
                timeline_guides=[],
            )
            setattr(media, field, value)
            with patch.dict(
                sys.modules,
                {
                    "vllm_omni.diffusion.models.minimax_h3.vae": NS(
                        MiniMaxH3VideoVAE=load("video"), MiniMaxH3AudioVAE=load("audio")
                    ),
                    "vllm_omni.model_executor.models.minimax_h3.encoder_processing": NS(
                        encode_media=lambda *a, **kw: {"encoded": torch.ones(1)}
                    ),
                },
            ):
                engine._encode_media([NS(media=media)])
            self.assertEqual(loaded, expected)

    def test_suffix_returns_error_and_still_drains_stop(self):
        incoming = Channel([{"packet": {}}, {"stop": True}])
        outgoing = Channel()
        runtime = NS(
            rank=1,
            topology=NS(world_size=2),
            stage=NS(transformer=None),
            pipeline=NS(device="cpu"),
            channels={(0, 1): incoming, (1, 0): outgoing},
        )
        with patch(
            "monowire.models.h3.request_mode.compute_stage", side_effect=ValueError("bad packet")
        ):
            RequestPipeline(runtime).serve()
        self.assertIn("bad packet", outgoing.sent[0]["error"])
        self.assertEqual(outgoing.sent[1], {"stop": True})

    def test_native_bindings_restore_on_failed_pass(self):
        transformer = NS(forward=lambda **kwargs: "original")
        original_forward = transformer.forward
        pipeline = NS(
            transformer=transformer,
            od_config=NS(step_execution=True),
            _upscaled_latent=lambda video, target: video,
            diffuse=lambda **kwargs: None,
            _prepare_request_inputs=lambda *a: "original context",
        )
        original_prepare = pipeline._prepare_request_inputs
        bridge = NS(forward=lambda **kwargs: "pipeline output")
        with patch.dict(
            sys.modules,
            {
                "vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3": NS(
                    _REFINE_KEYFRAME_CONDITION="refine"
                )
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "failed pass"):
                with native_request_context(None, NS(pipeline=pipeline), bridge):
                    self.assertFalse(pipeline.od_config.step_execution)
                    self.assertEqual(transformer.forward(), "pipeline output")
                    raise RuntimeError("failed pass")
        self.assertIs(transformer.forward, original_forward)
        self.assertIs(pipeline._prepare_request_inputs, original_prepare)
        self.assertTrue(pipeline.od_config.step_execution)


@unittest.skipUnless(importlib.util.find_spec("vllm_omni"), "Requires bundled vLLM-Omni")
class NativeModeTests(unittest.TestCase):
    def test_timeline_handoff_preserves_guides_without_unsupported_native_wire(self):
        from dataclasses import replace

        from vllm_omni.model_executor.models.minimax_h3.conditioning import (
            MiniMaxH3EncoderMediaConditioning,
        )

        from monowire.models.h3.pipeline import MonoWireH3Pipeline

        media = MiniMaxH3EncoderMediaConditioning(
            task="t2va",
            height=32,
            width=32,
            num_frames=22,
            latent_t=7,
            audio_t=37,
            guide_blocks=({"kind": "image", "frame_index": 0, "latent_h": 2, "latent_w": 2},),
            guide_visual_condition=torch.arange(96, dtype=torch.float32).reshape(1, 96),
        )
        hidden, tags = torch.zeros(2, 5120, dtype=torch.bfloat16), torch.zeros(2, dtype=torch.long)
        payload = PhasedH3Engine._conditioning_payload(hidden, tags, media)
        restored = MonoWireH3Pipeline._extract_encoder_conditioning(payload)
        self.assertEqual(restored.guide_blocks, media.guide_blocks)
        torch.testing.assert_close(restored.guide_visual_condition, media.guide_visual_condition)
        self.assertEqual(restored.guide_visual_condition.device.type, "cpu")
        plain = PhasedH3Engine._conditioning_payload(
            hidden, tags, replace(media, guide_blocks=(), guide_visual_condition=None)
        )
        self.assertIn("encoder_output", plain["additional_information"])
        self.assertEqual(MonoWireH3Pipeline._extract_encoder_conditioning(plain).guide_blocks, ())

    def test_native_upscale_exact_result_and_ingress_residency_transition(self):
        from vllm_omni.diffusion.models.minimax_h3.latent_upscaler import (
            MiniMaxH3LatentResizer3D,
            MiniMaxH3LatentUpscaler,
            MiniMaxH3LatentUpscalerArch,
            resolve_minimax_h3_latent_upscale_target,
        )

        # Small synthetic weights qualify routing/residency, not released-model quality.
        torch.manual_seed(7)
        resizer = MiniMaxH3LatentResizer3D(
            MiniMaxH3LatentUpscalerArch(channels=32, in_blocks=1, out_blocks=1, temporal_every=0)
        )
        upscaler = MiniMaxH3LatentUpscaler(
            resizer,
            device=torch.device("cpu"),
            dtype=torch.float32,
            latents_mean=[0.0] * 24,
            latents_std=[1.0] * 24,
        )
        target = resolve_minimax_h3_latent_upscale_target(
            latent_height=2, latent_width=2, scale=2, align=32
        )
        video = torch.randn(1, 24, 2, 2, 2)
        expected = upscaler.upscale(video, target)
        transformer = NS(
            adaln_cache=None,
            to=lambda device: None,
            forward=lambda **kw: None,
            start_layer=0,
            end_layer=0,
            prepare_transformer_forward=lambda **kw: (kw["hidden"], {}, {}),
            finish_transformer_forward=lambda hidden, **kw: hidden,
        )
        pipeline = NS(
            transformer=transformer,
            device="cpu",
            od_config=NS(step_execution=True),
            _upscaled_latent=upscaler.upscale,
            diffuse=lambda **kw: None,
        )
        runtime = NS(
            rank=0,
            topology=NS(world_size=1),
            channels={},
            stage=NS(transformer=transformer),
            pipeline=pipeline,
        )
        bridge = RequestPipeline(runtime)
        engine = NS(_mark=lambda *args, **kwargs: None)
        with (
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.empty_cache"),
            patch("torch.cuda.memory_allocated", return_value=0),
        ):
            with native_request_context(engine, runtime, bridge):
                actual = pipeline._upscaled_latent(video, target)
                self.assertFalse(bridge.resident)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(actual.shape, (1, 24, 2, 4, 4))
                self.assertIs(bridge.forward(hidden=actual), actual)
                self.assertTrue(bridge.resident)
        self.assertTrue(all(parameter.device.type == "cpu" for parameter in upscaler.parameters()))

    def test_native_continuation_preserves_global_clock_and_trims_both_modalities(self):
        from vllm_omni.diffusion.models.minimax_h3.continuation import diffuse_continuation

        calls = []

        def diffuse(**kwargs):
            calls.append(kwargs)
            number = float(len(calls))
            return (
                torch.full((1, 24, kwargs["latent_t"], 2, 2), number),
                torch.full((2, 32, kwargs["audio_t"]), number),
            )

        kwargs = dict(num_frames=226, latent_t=67, audio_t=377, latent_h=2, latent_w=2)
        text = [(torch.ones(2, 4), torch.zeros(2)), (torch.ones(5, 4), torch.zeros(5))]
        video, audio = diffuse_continuation(
            diffuse, kwargs, window_frames=124, overlap_frames=22, text_conditioning=text
        )
        self.assertEqual(video.shape, (1, 24, 67, 2, 2))
        self.assertEqual(audio.shape, (2, 32, 377))
        self.assertEqual(calls[1]["temporal_offset"], 170.0)
        self.assertEqual(calls[1]["media_time_origin"], 5)
        self.assertEqual(calls[1]["visual_condition"].shape, (7, 96))
        self.assertEqual(calls[1]["audio_condition"].shape, (74, 32))
        self.assertTrue(torch.all(video[:, :, :37] == 1))
        self.assertTrue(torch.all(video[:, :, 37:] == 2))
        self.assertTrue(torch.all(audio[..., :207] == 1))
        self.assertTrue(torch.all(audio[..., 207:] == 2))

    def test_native_refine_reuses_preencoded_high_resolution_anchors(self):
        from vllm_omni.diffusion.models.minimax_h3.latent_upscaler import MiniMaxH3LatentRefineSpec
        from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
            _REFINE_KEYFRAME_CONDITION,
            MiniMaxH3Pipeline,
        )

        pipeline = object.__new__(MiniMaxH3Pipeline)
        torch.nn.Module.__init__(pipeline)
        calls = []
        pipeline.diffuse = lambda **kwargs: calls.append(kwargs) or kwargs["init_latents"]
        pipeline._denoise_kwargs = lambda context: {
            "pad_seq_len": 128,
            "latent_h": 2,
            "latent_w": 2,
        }
        anchors = torch.ones(4, 96)
        context = dict(
            task="fl2va",
            latent_refine=MiniMaxH3LatentRefineSpec(0.4),
            latent_upscale=NS(latent_height=4, latent_width=4),
            **{_REFINE_KEYFRAME_CONDITION: (anchors, [(1, 4, 4)])},
        )
        video, audio = torch.ones(1, 24, 2, 4, 4), torch.ones(2, 32, 8)
        actual = pipeline._refined_latents(video, audio, context=context, seed=42)
        self.assertIs(actual[0], video)
        self.assertIs(calls[0]["visual_condition"], anchors)
        self.assertIsNone(calls[0]["pad_seq_len"])
        self.assertEqual(calls[0]["latent_h"], 4)
        self.assertEqual(calls[0]["seed"], 42)


if __name__ == "__main__":
    unittest.main()
