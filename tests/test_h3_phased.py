"""Phase ownership, partition bounds, and native Qwen layer equivalence."""

import asyncio
import importlib.util
import tempfile
import unittest
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


def rolling_encoder_worker(rank, store, count, drained, first_received):
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

    try:
        with patch("monowire.models.h3.encoder_pp.native.MINIMAX_H3_QWEN3VL_HIDDEN_DIM", 8):
            run_encoder_pipeline(
                Stage(),
                [(torch.zeros(n), {}) for n in lengths],
                lengths,
                dist.group.WORLD,
                dist.group.WORLD,
                on_output=publish,
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
        inbox.queue.put(ValueError("broken peer"))
        with self.assertRaisesRegex(RuntimeError, "receiver failed"):
            asyncio.run(collect())
        inbox.close()

    def test_job_phases_and_failure_cleanup(self):
        from unittest.mock import Mock

        for fail in (False, True):
            with self.subTest(fail=fail):
                engine = PhasedH3Engine(
                    "FL2VA",
                    pp_size=1,
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
                if fail:
                    engine._denoise.side_effect = RuntimeError("denoise failed")
                engine._decode = lambda outputs, sampling: ["decoded"]
                engine._phase = phase
                with (
                    patch(
                        "monowire.models.h3.phased.H3Runtime.context", return_value=nullcontext()
                    ),
                    patch("monowire.models.h3.phased.dist.broadcast_object_list"),
                ):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "denoise failed"):
                            engine.generate(["prompt"], None)
                    else:
                        self.assertEqual(engine.generate(["prompt"], None), ["decoded"])
                self.assertEqual(
                    events, ["qwen_pp", "vae_encode", "dit"] + ([] if fail else ["vae_decode"])
                )
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
                mp.spawn(
                    rolling_encoder_worker,
                    args=(str(Path(folder, "store")), count, drained, first_received),
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
