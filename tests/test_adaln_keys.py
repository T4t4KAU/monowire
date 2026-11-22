"""Exact AdaLN keys survive batching and transport without repeated GPU reads."""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from monowire.models.h3.adapter import compute_stage
from monowire.models.h3.packing import pack_requests


@unittest.skipUnless(importlib.util.find_spec("vllm_omni"), "Requires bundled vLLM-Omni")
class AdalnKeyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vllm_omni.diffusion.models.minimax_h3.adaln_cache import MiniMaxH3RuntimeAdalnCache

        cls.cache_class = MiniMaxH3RuntimeAdalnCache

    def setUp(self):
        self.cache = self.cache_class()
        self.module = torch.nn.Linear(4, 2, bias=False)
        self.tp = patch("vllm.distributed.get_tensor_model_parallel_world_size", return_value=1)
        self.tp.start()
        self.addCleanup(self.tp.stop)

    def item(self, values):
        embedding = torch.tensor(values, dtype=torch.float32).reshape(-1, 4)
        self.cache.prepare(embedding)
        return {
            "hidden": torch.zeros(1, 2),
            "adaln_key": self.cache.prepared_key(embedding),
            "arguments": {
                "t_emb": embedding,
                "num_requests": 1,
                "packed_total": 1,
                "max_seqlen": 1,
                "cu_seqlens": torch.tensor([0, 1], dtype=torch.int32),
                "combined_indices": torch.zeros(1, dtype=torch.long),
                "rope_table": torch.zeros(1, 2),
            },
        }

    def project(self, packet):
        embedding = packet["t_emb"]
        self.cache.prepare_from_key(embedding, packet["adaln_key"])
        return self.cache.project(
            "projection", self.module, embedding, lambda: self.module(embedding)
        )

    def test_content_order_shape_and_missing_keys_are_isolated(self):
        with torch.no_grad():
            a = self.item([1, 2, 3, 4])
            b = self.item([4, 3, 2, 1])
            same_a = self.item([1, 2, 3, 4])
            with patch(
                "vllm_omni.diffusion.cache.exact_projection_cache.tensor_digest",
                side_effect=AssertionError("Repeated device read"),
            ):
                for items in ([a, b], [same_a, b], [b, a], [a], [b, a, b]):
                    packet = pack_requests(items)[0]
                    torch.testing.assert_close(self.project(packet), self.module(packet["t_emb"]))
                self.assertEqual((self.cache.hits, self.cache.misses), (1, 4))
                missing = dict(a, adaln_key=None)
                packet = pack_requests([missing, b])[0]
                self.assertIsNone(packet["adaln_key"])
                torch.testing.assert_close(self.project(packet), self.module(packet["t_emb"]))
                self.assertIsNone(self.cache.prepared_key(packet["t_emb"]))

    def test_rebinding_keeps_weight_and_precision_invalidation(self):
        with torch.no_grad():
            packet = pack_requests([self.item([1, 2, 3, 4])])[0]
            self.project(packet)
            self.project(packet)
            self.assertEqual(self.cache.hits, 1)
            self.module.weight.add_(1)
            torch.testing.assert_close(self.project(packet), self.module(packet["t_emb"]))
            self.assertEqual(self.cache.hits, 1)
            with patch.object(self.cache, "_numerical_settings", return_value=("different",)):
                self.project(packet)
            self.assertEqual(self.cache.hits, 1)
            self.assertEqual(self.cache.misses, 3)
            self.assertIsNone(self.cache.prepared_key(packet["t_emb"].clone()))
        self.cache.prepare_from_key(packet["t_emb"], packet["adaln_key"])
        self.assertIsNone(self.cache.prepared_key(packet["t_emb"]))  # Grad-enabled bypass.

    def test_sidecar_selection_is_reset_when_binding_a_packed_key(self):
        with torch.no_grad():
            item = self.item([1, 2, 3, 4])
            embedding, key = item["arguments"]["t_emb"], item["adaln_key"]
            self.cache.sidecar = SimpleNamespace(
                manifest={"math": "test"}, _embedding_plans={key: 7}
            )
            with patch(
                "vllm_omni.diffusion.models.minimax_h3.adaln_cache.math_identity",
                return_value="test",
            ):
                self.cache.prepare_from_key(embedding, key)
                self.assertEqual(self.cache._sidecar_plan, 7)
                self.cache.prepare_from_key(torch.cat([embedding, embedding]), (key, key))
                self.assertIsNone(self.cache._sidecar_plan)
                self.cache.prepare_from_key(embedding, None)
                self.assertIsNone(self.cache._sidecar_plan)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_real_cache_does_not_wait_for_previous_stage_compute(self):
        device = torch.device("cuda:0")
        module = self.module.to(device)
        cache = self.cache

        class Block:
            def __call__(self, hidden, *, t_emb, **kwargs):
                return hidden + cache.project("projection", module, t_emb, lambda: module(t_emb))

        model = SimpleNamespace(
            start_layer=0,
            end_layer=1,
            adaln_cache=cache,
            blocks=[Block()],
            pp_group=SimpleNamespace(rank_in_group=0),
        )
        with torch.no_grad():
            packet = pack_requests([self.item([1, 2, 3, 4])])[0]
            packet = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in packet.items()
            }
            # Populate lazy projection/runtime state before the nonblocking check.
            compute_stage(model, dict(packet))
            torch.cuda.synchronize(device)
            with patch(
                "vllm_omni.diffusion.cache.exact_projection_cache.tensor_digest",
                side_effect=AssertionError("Stage rehashed a GPU embedding"),
            ):
                torch.cuda._sleep(500_000_000)
                previous = torch.cuda.Event()
                previous.record()
                result = compute_stage(model, dict(packet))["hidden"]
                completed_early = previous.query()
            torch.cuda.synchronize(device)
            self.assertFalse(completed_early, "Stage preparation waited for prior GPU compute")
            torch.testing.assert_close(result, packet["hidden"] + module(packet["t_emb"]))

    def test_time_embedding_reuse_tracks_contents_weights_and_math(self):
        module = torch.nn.Linear(4, 2).eval()
        plan = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        with torch.no_grad():
            first = self.cache.embed_timesteps(module, plan, "cpu")
            with patch.object(self.cache, "prepare", side_effect=AssertionError("Repeated hash")):
                again = self.cache.embed_timesteps(module, plan.clone(), "cpu")
            self.assertIs(first, again)
            module.weight.add_(1)
            changed = self.cache.embed_timesteps(module, plan, "cpu")
            torch.testing.assert_close(changed, module(plan))
            self.assertIsNot(first, changed)
            plan.add_(2)
            torch.testing.assert_close(
                self.cache.embed_timesteps(module, plan, "cpu"), module(plan)
            )
            with patch.object(self.cache, "_numerical_settings", return_value=("new",)):
                different = self.cache.embed_timesteps(module, plan, "cpu")
            self.assertIsNot(different, changed)
            self.cache.clear()
            self.assertFalse(self.cache._time_embeddings)

    def test_time_embedding_hooks_and_tp_never_skip_execution(self):
        module = torch.nn.Linear(4, 2).eval()
        plan = torch.ones(1, 4)
        calls = []
        handle = module.register_forward_hook(lambda *args: calls.append(1))
        with torch.no_grad():
            for _ in range(2):
                self.cache.embed_timesteps(module, plan, "cpu")
        handle.remove()
        self.assertEqual(len(calls), 2)
        self.assertFalse(self.cache._time_embeddings)
        with (
            torch.no_grad(),
            patch("vllm.distributed.get_tensor_model_parallel_world_size", return_value=2),
        ):
            self.cache.embed_timesteps(module, plan, "cpu")
        self.assertFalse(self.cache._time_embeddings)

    def test_scalar_timestep_plan_matches_dense_assignment(self):
        from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MiniMaxH3DenoiseBranch

        branch = MiniMaxH3DenoiseBranch.__new__(MiniMaxH3DenoiseBranch)
        branch.seq_len = 6
        branch.device = torch.device("cpu")
        branch.img_pos_dev = torch.tensor([1, 2])
        branch.audio_pos_dev = torch.tensor([3, 4])
        branch.update_mask_dev = torch.tensor([True, False])
        branch.audio_update_mask_dev = torch.tensor([True, False])
        branch._present_time_groups = torch.arange(5)
        branch._time_group_indices = torch.tensor([0, 1, 2, 3, 4, 0])
        branch.x_base = torch.zeros(1, 6, 2)
        branch.audio_x_base = torch.zeros(1, 6, 2)
        branch.static_kwargs = {}
        for locked in (None, torch.ones(1)):
            branch.locked_audio_rows = locked
            for video, audio in ((0.8, 0.3), (0.3, 0.3), (1.0, 1.0)):
                values = dict(
                    t_video=video,
                    t_audio=audio,
                    imgvid_cond_timestep=0.999,
                    audio_ref_cond_timestep=1.0,
                )
                dense = torch.empty(6)
                branch.fill_timesteps(dense, **values)
                expected, inverse = torch.unique(dense, sorted=True, return_inverse=True)
                output = branch.forward_kwargs(
                    video_rows=torch.ones(2, 2), audio_rows=torch.ones(2, 2), **values
                )
                torch.testing.assert_close(output["unique_timesteps"], expected, rtol=0, atol=0)
                torch.testing.assert_close(output["inverse_indices"], inverse, rtol=0, atol=0)
                # Editing remains on the general path and preserves exact per-row values.
                edits = dict(
                    video_target_timesteps=torch.tensor([0.123]),
                    audio_target_timesteps=torch.tensor([0.456]),
                )
                branch.fill_timesteps(dense, **values, **edits)
                expected, inverse = torch.unique(dense, sorted=True, return_inverse=True)
                output = branch.forward_kwargs(
                    video_rows=torch.ones(2, 2), audio_rows=torch.ones(2, 2), **values, **edits
                )
                torch.testing.assert_close(output["unique_timesteps"], expected, rtol=0, atol=0)
                torch.testing.assert_close(output["inverse_indices"], inverse, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_warm_time_embedding_does_not_sync_compute_stream(self):
        module = torch.nn.Linear(4, 2).cuda().eval()
        plan = torch.ones(1, 4)
        with torch.no_grad():
            first = self.cache.embed_timesteps(module, plan, "cuda:0")
            torch.cuda.synchronize()
            torch.cuda._sleep(500_000_000)
            before = torch.cuda.Event()
            before.record()
            actual = self.cache.embed_timesteps(module, plan, "cuda:0")
            waited = before.query()
            torch.cuda.synchronize()
            self.assertFalse(waited)
            self.assertIs(first, actual)
