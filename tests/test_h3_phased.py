"""Phase ownership, partition bounds, and native Qwen layer equivalence."""

import asyncio
import importlib.util
import sys
import tempfile
import time
import unittest
import weakref
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from monowire.models.h3.phase_utils import cpu_tree, encoder_partition
from monowire.models.h3.phased import PhasedH3Engine
from monowire.models.h3.rolling import EncoderInbox


def rolling_encoder_worker(rank, store, count, drained, first_received, tail_loaded):
    from monowire.models.h3.encoder_pp import run_encoder_pipeline

    dist.init_process_group(
        "gloo",
        init_method=Path(store).as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    output_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=30))
    lengths = [index + 1 for index in range(count)]
    inbox = (
        EncoderInbox(lengths, 8, 3, output_group, on_output=lambda _: first_received.set())
        if rank == 0
        else None
    )

    class Stage:
        device_target = "cpu"
        text_model = SimpleNamespace(last_positions=None)

        def first(self, ids, kwargs):
            if count > 4 and len(ids) == count:
                assert first_received.wait(10), "Tail buffered outputs until global drain"
            return torch.ones((len(ids), 8), dtype=torch.bfloat16), torch.zeros(
                (3, 1, len(ids)), dtype=torch.long
            )

        def continuation(self, hidden, positions):
            # The tail cannot finish until ingress has returned and released
            # its encoder. This deterministically catches a hidden drain barrier.
            if rank == 3 and len(hidden) == count:
                assert drained.wait(10), "Ingress waited for the global encoder drain"
            return hidden + 1

    def publish(index, tensor):
        dist.send(tensor, dst=0, group=output_group)

    def ready():
        assert tail_loaded.is_set(), "Receiver started before all Qwen stages loaded"
        inbox.start()

    try:
        if rank == 3:
            time.sleep(0.1)
            tail_loaded.set()
        with patch("monowire.models.h3.encoder_pp.native.MINIMAX_H3_QWEN3VL_HIDDEN_DIM", 8):
            run_encoder_pipeline(
                Stage(),
                [(torch.zeros(n), {}) for n in lengths],
                lengths,
                dist.group.WORLD,
                dist.group.WORLD,
                on_output=publish,
                on_ready=ready if inbox is not None else None,
            )
        if rank == 0:
            drained.set()

            async def check():
                seen = []
                async for index, tensor in inbox.tensors():
                    torch.testing.assert_close(
                        tensor, torch.full((lengths[index], 8), 4.0, dtype=torch.bfloat16)
                    )
                    seen.append(index)
                assert seen == list(range(count))

            asyncio.run(check())
            inbox.close()
    finally:
        dist.destroy_process_group(output_group)
        dist.destroy_process_group()


class PhaseTests(unittest.TestCase):
    def test_decode_skips_model_loading_when_no_successful_outputs(self):
        engine = object.__new__(PhasedH3Engine)
        for outputs in ([], [SimpleNamespace(error="admission failed")]):
            with self.subTest(outputs=outputs):
                # No worker or native runtime exists: returning request errors
                # must not depend on allocating a VAE successfully.
                self.assertIs(engine._decode(outputs, None), outputs)

    def test_overlap_releases_spent_inputs_before_loading_next_model(self):
        engine = PhasedH3Engine(
            "FL2VA",
            pp_size=1,
            config=SimpleNamespace(async_transport=True),
            cache=None,
            topology=None,
            encoder_stage_layers=None,
            trust_remote_code=False,
            overlap_encoder_dit=True,
        )
        engine.rank = 0
        engine.worker = SimpleNamespace(model_runner=SimpleNamespace(pipeline=None))
        prepared, inputs, tags = [torch.ones(1)], [(torch.ones(1), {})], [torch.ones(1)]
        prepared_ref, input_ref = weakref.ref(prepared[0]), weakref.ref(inputs[0][0])
        test = self
        transformer = SimpleNamespace(parameters=lambda: [], topology=None)

        class Loader:
            def __init__(self, *args, **kwargs):
                pass

            def load_model(self, **kwargs):
                test.assertIsNone(prepared_ref(), "Raw media retained during DiT CPU preload")
                return SimpleNamespace(transformer=transformer)

        class Runtime:
            def __init__(self, *args, **kwargs):
                test.assertIsNone(input_ref(), "Encoder pixels retained during DiT GPU loading")

            def close(self, **kwargs):
                pass

        class Continuous:
            def __init__(self, *args):
                pass

            async def stream(self, source):
                async for key, payload, sampling in source:
                    test.assertIsNone(tags[0])
                    # The submitted payload must still own its live conditions.
                    test.assertEqual(payload[0].shape, (1, 5120))
                    test.assertEqual(payload[1].item(), 1)
                    yield SimpleNamespace(error=None, output=payload)

        def encode_text(values, lengths, *, on_output, on_ready):
            test.assertIs(values, inputs)
            test.assertIsNotNone(input_ref())
            on_ready()
            on_output(0, torch.ones((1, 5120)))

        engine._phase = lambda name, operation, **kwargs: operation()
        engine._leader = lambda operation: operation()
        engine._encode_media = lambda values: ["encoded media"]
        engine._encode_text = encode_text
        engine._conditioning_payload = lambda hidden, tag, media: (hidden, tag, media)
        engine._decode = lambda outputs, sampling: outputs
        with (
            patch.dict(
                sys.modules,
                {
                    "vllm.config": SimpleNamespace(LoadConfig=lambda: None),
                    "vllm_omni.diffusion.model_loader.diffusers_loader": SimpleNamespace(
                        DiffusersPipelineLoader=Loader
                    ),
                },
            ),
            patch("monowire.models.h3.phased.H3Runtime", Runtime),
            patch("monowire.models.h3.phased.ContinuousEngine", Continuous),
            patch("monowire.channels.create_channels", return_value={}),
            patch("monowire.models.h3.phased.dist.new_group", return_value=object()),
            patch("monowire.models.h3.phased.dist.destroy_process_group"),
            patch("torch.cuda.memory_allocated", return_value=0),
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.empty_cache"),
        ):
            engine.worker.device = "cpu"
            engine.worker.od_config = None
            outputs = engine._generate_overlap(prepared, inputs, tags, [1], None)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].output[2], "encoded media")

    def test_encoder_partition_keeps_deepstack_on_first_rank(self):
        self.assertEqual(encoder_partition(4), (13, 13, 12, 12))
        for layers in ((2, 16, 16, 16), (13, 13, 12, 11), (0, 20, 15, 15), (25, 25)):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                encoder_partition(4, layers)

    def test_cpu_payload_detaches_nested_latents_without_changing_metadata(self):
        tensor = torch.ones(2, requires_grad=True)
        value = {"payload": ([tensor], {"shape": (2,), "task": "fl2va"})}
        result = cpu_tree(value)
        self.assertFalse(result["payload"][0][0].requires_grad)
        self.assertEqual(result["payload"][1], value["payload"][1])
        self.assertIsNot(result, value)

    def test_media_dataclass_is_detached_before_qwen(self):
        @dataclass(frozen=True)
        class Media:
            visual: torch.Tensor
            task: str

        media = Media(torch.ones(2, requires_grad=True), "fl2va")
        result = cpu_tree(media)
        self.assertIsInstance(result, Media)
        self.assertFalse(result.visual.requires_grad)
        self.assertEqual(result.task, "fl2va")

    def test_single_rank_inbox_and_receiver_error(self):
        inbox = EncoderInbox([2], 8, 0, None)
        inbox.publish(0, torch.ones((2, 8)))

        async def collect():
            return [item async for item in inbox.tensors()]

        self.assertEqual(asyncio.run(collect())[0][0], 0)
        inbox._put(ValueError("broken peer"))
        with self.assertRaisesRegex(RuntimeError, "receiver failed"):
            asyncio.run(collect())
        inbox.close()

    def test_job_phases_and_failure_cleanup(self):
        from unittest.mock import Mock

        for fail in (False, True, "peer"):
            with self.subTest(fail=fail):
                engine = PhasedH3Engine(
                    "FL2VA",
                    pp_size=2,
                    config=None,
                    cache=None,
                    topology=None,
                    encoder_stage_layers=None,
                    trust_remote_code=False,
                )
                worker = Mock()
                engine.cpu_group = None
                events = []

                def initialize():
                    engine.worker = worker

                def phase(name, operation):
                    events.append(name)
                    return operation()

                engine._make_worker = initialize
                engine._leader = lambda operation: operation()
                engine._prepare = lambda *args: (["media"], [(torch.ones(3), {})], ["tags"])
                engine._encode_text = lambda *args: ["hidden"]
                engine._encode_media = lambda *args: ["encoded"]
                engine._denoise = Mock(return_value=["latents"])
                if fail is True:
                    engine._denoise.side_effect = RuntimeError("denoise failed")

                def gather_errors(errors, local_error, **kwargs):
                    errors[:] = [local_error, "rank 1: denoise failed" if fail == "peer" else None]

                engine._decode = lambda outputs, sampling: ["decoded"]
                engine._phase = phase
                with (
                    patch(
                        "monowire.models.h3.phased.H3Runtime.context", return_value=nullcontext()
                    ),
                    patch("monowire.models.h3.phased.dist.broadcast_object_list"),
                    patch(
                        "monowire.models.h3.phased.dist.all_gather_object",
                        side_effect=gather_errors,
                    ) as gather,
                ):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "denoise failed"):
                            engine.generate(["prompt"], SimpleNamespace(extra_args={}))
                    else:
                        self.assertEqual(
                            engine.generate(["prompt"], SimpleNamespace(extra_args={})), ["decoded"]
                        )
                self.assertEqual(
                    events, ["qwen_pp", "vae_encode", "dit"] + ([] if fail else ["vae_decode"])
                )
                gather.assert_called_once()
                worker.shutdown.assert_called_once()
                self.assertIsNone(engine.worker)
                self.assertFalse(engine.running)


@unittest.skipUnless(importlib.util.find_spec("vllm_omni"), "Requires bundled vLLM-Omni")
class QwenPPTests(unittest.TestCase):
    def test_rolling_return_before_tail_finishes_for_short_and_full_jobs(self):
        for count in (1, 5):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as folder:
                drained = mp.get_context("spawn").Event()
                first_received = mp.get_context("spawn").Event()
                tail_loaded = mp.get_context("spawn").Event()
                mp.spawn(
                    rolling_encoder_worker,
                    args=(str(Path(folder, "store")), count, drained, first_received, tail_loaded),
                    nprocs=4,
                    join=True,
                )

    def test_fifty_layer_partition_matches_native_with_deepstack_and_mrope(self):
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm_omni.diffusion.models.minimax_h3 import encoder as native

        from monowire.models.h3.encoder_pp import QwenPPTextModel

        config = SimpleNamespace(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=50,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            rms_norm_eps=1e-6,
            rope_theta=10000,
            rope_parameters={"rope_theta": 10000, "mrope_section": [1, 1, 2]},
        )
        group = SimpleNamespace(rank_in_group=0, world_size=1)
        with (
            set_current_vllm_config(VllmConfig()),
            torch.inference_mode(),
            patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0),
            patch(
                "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1
            ),
        ):
            full = native.MiniMaxH3Qwen3VLTextModel(group, config, 50, torch.bfloat16)
            torch.manual_seed(42)
            for name, parameter in full.named_parameters():
                parameter.fill_(1) if "norm" in name else parameter.normal_(std=0.02)
            stages = []
            start = 0
            for count in (13, 13, 12, 12):
                stage = QwenPPTextModel(config, start, start + count)
                stage.load_state_dict(
                    {k: v for k, v in full.state_dict().items() if k in stage.state_dict()},
                    strict=True,
                )
                stages.append(stage)
                start += count
            inputs = torch.randn(1, 7, 16, dtype=torch.bfloat16)
            positions = torch.arange(7).expand(3, 1, 7).clone()
            positions[1, :, 2:5] += 2
            mask = torch.tensor([[False, False, True, True, True, False, False]])
            deepstack = [torch.randn(3, 16, dtype=torch.bfloat16) for _ in range(3)]
            expected = full(
                inputs, positions, visual_pos_masks=mask, deepstack_visual_embeds=deepstack
            )
            actual = stages[0](
                inputs, positions, visual_pos_masks=mask, deepstack_visual_embeds=deepstack
            )
            for stage in stages[1:]:
                self.assertFalse(hasattr(stage, "embed_tokens"))
                actual = stage(actual, positions)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertFalse(any("norm.weight" == key for key in stages[-1].state_dict()))
