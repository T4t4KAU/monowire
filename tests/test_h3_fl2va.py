"""Native FL2VA step lifecycle with tiny tensors instead of model weights."""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from monowire.cache import CacheConfig, RequestCache
from monowire.models.h3.adapter import H3RequestAdapter
from monowire.models.h3.packing import pack_requests


@unittest.skipUnless(importlib.util.find_spec("vllm_omni"), "Requires bundled vLLM-Omni")
class FL2VAAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as native

        from monowire.models.h3.pipeline import MonoWireH3Pipeline

        self.native = native
        self.pipeline = object.__new__(MonoWireH3Pipeline)
        torch.nn.Module.__init__(self.pipeline)
        self.pipeline.device = torch.device("cpu")
        self.pipeline.partition = "fl2va"
        self.pipeline.supported_tasks = {"t2va", "fl2va", "ref2va"}
        self.pipeline._fasth3 = None
        self.pipeline._codec_offloader = None
        self.pipeline.load_text_encoder = True
        self.pipeline.load_vae_encoder = True
        self.pipeline.load_vae_decoder = True
        self.pipeline.od_config = None
        self.pipeline._resolve_latent_refine = lambda extra: None
        self.pipeline._denoise_kwargs = lambda context: context
        self.pipeline._prepare_request_inputs = self.prepare_inputs
        self.pipeline._build_denoise_inputs = lambda **inputs: inputs["inputs"]
        self.pipeline.decode = Mock(return_value=(torch.zeros(1, 3, 1, 32, 32), torch.zeros(2, 16)))
        self.pipeline.transformer = SimpleNamespace(
            start_layer=0,
            end_layer=1,
            adaln_cache=None,
            request_cache=RequestCache(CacheConfig(), 8),
            prepare_transformer_forward=self.prepare_forward,
            finish_transformer_forward=lambda output, **ctx: (
                output[: ctx["video_count"]],
                output[ctx["video_count"] :, :4],
            ),
        )
        self.adapter = H3RequestAdapter(self.pipeline)
        self.states = []

    async def asyncTearDown(self):
        await self.adapter.close()
        for state in self.states:
            self.adapter.release(state)

    def prepare_inputs(self, prompt, sampling):
        # Substitute media encoding and branch construction. Native prepare_encode,
        # the Euler scheduler, row unpacking and post_decode execute unchanged;
        # the large DiT and VAE forwards use tiny test doubles.
        images = prompt.get("multi_modal_data", {}).get("image")
        anchors = None if images is None else torch.tensor(images).reshape(-1, 1).repeat(1, 96)
        count = 1 + (0 if anchors is None else len(anchors))
        mask = torch.arange(count) == 0
        branch = SimpleNamespace(
            update_mask=mask,
            update_mask_dev=mask,
            audio_update_mask=torch.ones(2, dtype=torch.bool),
            audio_update_mask_dev=torch.ones(2, dtype=torch.bool),
            img_pos=torch.arange(count),
            audio_pos=torch.arange(2),
            locked_audio_rows=None,
            forward_kwargs=lambda **kwargs: kwargs,
        )
        return dict(
            task="t2va" if images is None else "fl2va",
            height=32,
            width=32,
            latent_t=1,
            latent_h=2,
            latent_w=2,
            audio_t=1,
            inputs=dict(
                branch=branch,
                video_rows=torch.ones(count, 96),
                audio_rows=torch.ones(2, 4),
                cond_anchor=anchors,
                audio_anchor=None,
                sigmas_video=[1.0, 0.5, 0.0],
                sigmas_audio=[1.0, 0.5, 0.0],
            ),
        )

    def prepare_forward(self, **inputs):
        video, audio = inputs["video_rows"], inputs["audio_rows"]
        hidden = torch.cat((video, torch.nn.functional.pad(audio, (0, 92))))
        length = len(hidden)
        arguments = dict(
            num_requests=1,
            packed_total=length,
            max_seqlen=length,
            cu_seqlens=torch.tensor([0, length], dtype=torch.int32),
            t_emb=torch.tensor([[inputs["t_video"], inputs["imgvid_cond_timestep"]]]),
            combined_indices=torch.zeros(length, dtype=torch.long),
            rope_table=torch.zeros(length, 4),
        )
        return hidden, arguments, {"video_count": len(video)}

    async def admit(self, key, images=None, task=None):
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        prompt = {"prompt": "Animate the scene"}
        if images is not None:
            prompt["multi_modal_data"] = {"image": images}
        extra = {} if task is None else {"task": task}
        state = await self.adapter.admit(
            key, (prompt, OmniDiffusionSamplingParams(extra_args=extra))
        )
        self.states.append(state)
        return state

    async def test_mixed_requests_preserve_image_anchors_and_decode_only_targets(self):
        states = [
            await self.admit("first", [7.0], "fl2va"),
            await self.admit("first-last", [11.0, 13.0]),
            await self.admit("text", task="t2va"),
        ]
        for step in range(2):
            prepared = self.adapter.prepare_batch([self.adapter.prepare(s) for s in states])
            packet, lengths = pack_requests([item.payload for item in prepared])
            self.assertEqual(lengths, [4, 5, 3])
            self.assertEqual(packet["cu_seqlens"].tolist(), [0, 4, 9, 12])
            # Simulate distinct velocity outputs returned by the shared suffix.
            outputs = torch.cat(
                [torch.full_like(p.payload["hidden"], i + 1) for i, p in enumerate(prepared)]
            )
            for index, (state, item, output) in enumerate(
                zip(states, prepared, outputs.split(lengths), strict=True)
            ):
                self.adapter.commit(state, item, output)
                self.assertEqual(state.step_index, step + 1)
                expected = torch.full((96,), 1.0 + (step + 1) * 0.5 * (index + 1))
                torch.testing.assert_close(state.latents[0], expected)
                anchor = state.extra[self.native._STEP_COND_ANCHOR]
                if anchor is not None:
                    torch.testing.assert_close(state.latents[1:], anchor)
                self.assertEqual(self.adapter.finished(state), step == 1)
        for state in states:
            result = await self.adapter.finish(state)
            self.assertEqual(result.output[0].shape, (1, 1, 32, 32, 3))
            self.assertEqual(result.output[0].dtype, torch.uint8)
            args = self.pipeline.decode.call_args.args
            self.assertEqual(args[0].shape, (1, 24, 1, 2, 2))
            self.assertEqual(args[1].shape, (2, 4, 1))
        self.assertEqual(self.pipeline.decode.call_count, 3)

    async def test_ref2va_admitted_only_with_matching_checkpoint(self):
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        for extra, media in (({"task": "ref2va"}, {}), ({}, {"video": object()})):
            self.pipeline.supported_tasks = {"t2va", "fl2va"}
            with patch.object(self.pipeline, "prepare_encode") as encode:
                with self.assertRaisesRegex(ValueError, "supports"):
                    await self.adapter.admit(
                        "bad",
                        (
                            {"prompt": "scene", "multi_modal_data": media},
                            OmniDiffusionSamplingParams(extra_args=extra),
                        ),
                    )
                encode.assert_not_called()
        self.pipeline.supported_tasks = {"ref2va"}
        self.pipeline.partition = "ref2va"
        state = await self.admit("reference", task="ref2va")
        self.assertIs(state.extra[self.native._STEP_TRANSFORMER], self.pipeline.transformer)

    async def test_native_image_preprocessing_preserves_first_last_order(self):
        from PIL import Image
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        images = [Image.new("RGB", (1280, 720), color) for color in ("red", "blue")]
        sampling = OmniDiffusionSamplingParams(
            height=384, width=672, num_frames=124, extra_args={"task": "fl2va"}
        )
        prepared = self.native.prepare_encoder_inputs(
            {"prompt": "Animate the scene", "multi_modal_data": {"image": images}},
            sampling,
            task="fl2va",
        )
        self.assertEqual(prepared.media.task, "fl2va")
        self.assertEqual([tuple(im.shape) for im in prepared.media.images], [(384, 672, 3)] * 2)
        self.assertEqual(prepared.media.images[0][0, 0].tolist(), [255, 0, 0])
        self.assertEqual(prepared.media.images[1][0, 0].tolist(), [0, 0, 255])
        self.assertEqual(prepared.media.keyframe_frame_indices, (0, -1))
        with self.assertRaisesRegex(Exception, "image conditions"):
            self.native.prepare_encoder_inputs({"prompt": "scene"}, sampling, task="fl2va")

    async def test_codec_failure_cleans_partial_request_state(self):
        def fail(state):
            state.extra[self.native._STEP_TRANSFORMER] = self.pipeline.transformer
            state.latents = torch.ones(2)
            self.states.append(state)
            raise ValueError("image encode failed")

        with patch.object(self.pipeline, "prepare_encode", side_effect=fail):
            with self.assertRaisesRegex(ValueError, "image encode failed"):
                await self.admit("failed", [1.0], "fl2va")
        self.assertIsNone(self.states[-1].latents)
        self.assertEqual(self.states[-1].extra, {})

    def encoded_prompt(self, task):
        from vllm_omni.model_executor.models.minimax_h3.conditioning import (
            MiniMaxH3EncoderConditioning,
        )

        conditioning = MiniMaxH3EncoderConditioning(
            hidden_states=torch.zeros(1, 5120, dtype=torch.bfloat16),
            token_tags=torch.zeros(1, dtype=torch.int64),
            task=task,
            height=32,
            width=32,
            num_frames=1,
            latent_t=1,
            audio_t=1,
            visual_condition=torch.full((1, 96), 7.0) if task == "fl2va" else None,
            visual_condition_shapes=((1, 2, 2),) if task == "fl2va" else (),
            keyframe_frame_indices=(0,) if task == "fl2va" else (),
        )
        return {"additional_information": {"encoder_output": conditioning.to_omni_payload()}}

    async def test_preencoded_requests_need_no_original_prompt_or_images(self):
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        self.pipeline.load_text_encoder = self.pipeline.load_vae_encoder = False
        for task in ("t2va", "fl2va", "ref2va"):
            prompt = self.encoded_prompt(task)
            context = self.prepare_inputs(
                {"multi_modal_data": {"image": [7.0]}} if task == "fl2va" else {}, None
            )
            with (
                patch.object(
                    self.pipeline, "_extract_prompt", side_effect=AssertionError("raw input parsed")
                ),
                patch.object(
                    self.pipeline, "_prepare_request_inputs", return_value=context
                ) as prepare,
                patch.object(
                    self.pipeline, "_resolve_task", wraps=self.pipeline._resolve_task
                ) as resolve,
            ):
                sampling = OmniDiffusionSamplingParams(extra_args={})
                state = await self.adapter.admit(task, (prompt, sampling))
                self.states.append(state)
                self.assertIs(state.prompt, prompt)
                prepare.assert_called_once_with(prompt, sampling)
                resolve.assert_called_once_with(task)

    async def test_preencoded_payload_validation_and_task_restrictions_remain(self):
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        self.pipeline.load_text_encoder = self.pipeline.load_vae_encoder = False
        self.pipeline.supported_tasks = {"t2va", "fl2va"}
        cases = (
            ({}, "requires encoder conditioning"),
            ({"additional_information": {"encoder_output": {"invalid": True}}}, "wire requires"),
            (self.encoded_prompt("ref2va"), "supports"),
        )
        for prompt, message in cases:
            with patch.object(self.pipeline, "prepare_encode") as encode:
                with self.assertRaisesRegex((ValueError, self.native.OmniClientError), message):
                    await self.adapter.admit("bad", (prompt, OmniDiffusionSamplingParams()))
                encode.assert_not_called()

    async def test_preencoded_task_mismatch_still_reaches_native_validation(self):
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        self.pipeline.load_text_encoder = self.pipeline.load_vae_encoder = False
        # Restore the native request-input path for its consistency checks.
        del self.pipeline._prepare_request_inputs
        prompt = self.encoded_prompt("fl2va")
        with self.assertRaisesRegex(self.native.OmniClientError, "does not match"):
            await self.adapter.admit(
                "mismatch", (prompt, OmniDiffusionSamplingParams(extra_args={"task": "t2va"}))
            )
