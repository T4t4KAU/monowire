"""Check cache boundaries against a complete multi-block transformer on CPU."""

import asyncio
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from monowire.cache import CacheConfig, RequestCache
from monowire.models.h3.adapter import (
    H3StageAdapter,
    compute_stage,
    prepare_transformer_batch,
    prepare_transformer_step,
)
from monowire.topology import PipelineTopology
from monowire.transport import PipelineExecutor


class Block(torch.nn.Module):
    def __init__(self, amount):
        super().__init__()
        self.amount = amount
        self.calls = 0
        self.scale = 1

    def forward(self, hidden, **kwargs):
        self.calls += 1
        return hidden * self.scale + self.amount


def transformer(policy, end=6, threshold=0.17, probe_blocks=()):
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([Block(i + 1) for i in range(8)])
    model.start_layer, model.end_layer = 0, end
    model.pp_group = SimpleNamespace(rank_in_group=0, world_size=2)
    model.adaln_cache = None
    model.request_cache = RequestCache(
        CacheConfig(policy, threshold=threshold, warmup_steps=1, probe_blocks=probe_blocks),
        capacity=4,
    )
    return model


def raw_prepare(model, step, value=1.0, key="request", timestep=0.0, tokens=2):
    hidden = torch.full((tokens, 3), float(value))

    block_inputs = (
        hidden,
        dict(
            num_requests=1,
            t_emb=torch.full((1, 4), timestep),
            packed_total=tokens,
            max_seqlen=tokens,
            cu_seqlens=torch.tensor([0, tokens], dtype=torch.int32),
            combined_indices=torch.zeros(tokens, dtype=torch.long),
            rope_table=torch.zeros(tokens, 4),
        ),
        {},
    )

    return prepare_transformer_step(model, block_inputs, key, step, 12)


def prepare(model, step, value=1.0, key="request"):
    return prepare_transformer_batch([raw_prepare(model, step, value, key)])[0]


def full_evaluation(model, prepared):
    """Run the ingress pack and then the remaining, remote blocks."""
    stage = H3StageAdapter(model)
    stage.validate([prepared])
    packet, lengths = stage.pack([prepared])
    output = packet["hidden"]
    for block in model.blocks[model.end_layer :]:
        output = block(output)
    output = output.split(lengths)[0]
    if prepared.payload["entry"] is not None:
        model.request_cache.commit(
            prepared.payload["entry"],
            prepared.step,
            False,
            output - prepared.payload["hidden"],
        )
    return output


class H3CacheRoutingTests(unittest.TestCase):
    def test_no_cache_export_releases_original_hidden(self):
        model = transformer("none")
        prepared = raw_prepare(model, 0)
        original = weakref.ref(prepared.payload["hidden"])
        with torch.inference_mode():
            exported = H3StageAdapter(model).export_requests([prepared])
        self.assertIsNone(original())
        self.assertNotIn("hidden", prepared.payload)
        self.assertIn("output_context", prepared.payload)
        torch.testing.assert_close(exported[0]["hidden"], torch.full((2, 3), 22.0))

    def test_cached_miss_retains_residual_anchor(self):
        for policy in ("first_block", "first_stage"):
            with self.subTest(policy=policy), torch.inference_mode():
                model = transformer(policy)
                prepared = prepare(model, 0)
                anchor = weakref.ref(prepared.payload["hidden"])
                exported = H3StageAdapter(model).export_requests([prepared])
                self.assertIs(anchor(), prepared.payload["hidden"])
                suffix_output = exported[0]["hidden"]
                for block in model.blocks[model.end_layer :]:
                    suffix_output = block(suffix_output)
                reconstructed = anchor() + (suffix_output - anchor())
                torch.testing.assert_close(reconstructed, torch.full((2, 3), 37.0))

    def test_stage_exit_export_keeps_existing_rows_without_repacking(self):
        model = transformer("first_stage")
        prepared = prepare_transformer_batch(
            [raw_prepare(model, 0, key=str(i), value=i) for i in range(2)]
        )
        stage = H3StageAdapter(model)
        with patch.object(stage, "pack", side_effect=AssertionError("Unnecessary local repack")):
            exported = stage.export_requests(prepared)
        for item, exported_item in zip(prepared, exported, strict=True):
            self.assertIs(exported_item["hidden"], item.payload["hidden"])

    def test_export_and_cross_owner_merge_preserve_timesteps_and_ragged_rows(self):
        models = [transformer("first_block"), transformer("first_stage")]
        items = []
        expected = []
        for owner, model in enumerate(models):
            prepared = prepare_transformer_batch(
                [
                    raw_prepare(
                        model,
                        i,
                        key=str(i),
                        value=owner * 10 + i,
                        timestep=i + owner * 4,
                        tokens=i + 1,
                    )
                    for i in range(2)
                ]
            )
            items.extend(H3StageAdapter(model).export_requests(prepared))
            expected.extend([owner * 10 + i + 21 for i in range(2)])
            self.assertEqual([b.calls for b in model.blocks], [1] * 6 + [0] * 2)
        packet = H3StageAdapter(models[0]).combine_requests(items)
        for hidden, value in zip(packet["hidden"].split([1, 2, 1, 2]), expected, strict=True):
            torch.testing.assert_close(hidden, torch.full_like(hidden, value))
        self.assertEqual(packet["num_requests"], 4)
        self.assertEqual(packet["t_emb"][:, 0].tolist(), [0, 1, 4, 5])

    def test_step_execution_never_moves_resident_transformer_weights(self):
        model = transformer("first_stage")
        with patch.object(model, "to", side_effect=AssertionError("DiT moved during a step")):
            for step in range(3):
                full_evaluation(model, prepare(model, step))

    def test_eight_requests_execute_each_prefix_block_once_with_independent_timesteps(self):
        class TimeBlock(Block):
            def forward(self, hidden, **kwargs):
                self.calls += 1
                time = kwargs["t_emb"][kwargs["combined_indices"] // 3, :1]
                return hidden + self.amount * (time + 1)

        model = transformer("first_stage", probe_blocks=(1, 3, 6))
        model.request_cache = RequestCache(model.request_cache.config, capacity=8)
        model.blocks = torch.nn.ModuleList([TimeBlock(i + 1) for i in range(8)])
        raw = [
            raw_prepare(model, i, value=float(i), key=str(i), timestep=float(i), tokens=i + 1)
            for i in range(8)
        ]
        self.assertEqual([b.calls for b in model.blocks], [0] * 8)
        prepared = prepare_transformer_batch(raw)
        self.assertEqual([b.calls for b in model.blocks], [1] * 6 + [0] * 2)
        for i, item in enumerate(prepared):
            output = item.payload["hidden"]
            torch.testing.assert_close(output, torch.full_like(output, i + 21 * (i + 1)))
            torch.testing.assert_close(
                item.payload["entry"].probe,
                torch.stack([torch.full_like(output, block * (i + 1)) for block in (1, 3, 6)]),
            )
            self.assertEqual(item.step, i)

    def test_batched_prefix_keeps_cache_decisions_request_private(self):
        model = transformer("first_stage")
        model.blocks[5].scale = 2
        for key in ("hit", "miss"):
            full_evaluation(model, prepare(model, 0, key=key))
        for block in model.blocks:
            block.calls = 0
        batch = prepare_transformer_batch(
            [
                raw_prepare(model, 1, value=1.0, key="hit"),
                raw_prepare(model, 1, value=100, key="miss"),
            ]
        )
        self.assertEqual([p.hit for p in batch], [True, False])
        self.assertEqual([b.calls for b in model.blocks], [1] * 6 + [0] * 2)
        self.assertEqual([p.payload["next_layer"] for p in batch], [6, 6])

    def test_selected_probes_are_individual_block_residuals(self):
        for policy in ("first_stage", "first_stage_periodic"):
            with self.subTest(policy=policy):
                model = transformer(policy, probe_blocks=(1, 3, 6))
                prepared = prepare(model, 0)
                output = full_evaluation(model, prepared)
                torch.testing.assert_close(
                    prepared.payload["entry"].probe,
                    torch.stack([torch.full_like(output, i) for i in (1, 3, 6)]),
                )
                self.assertEqual([b.calls for b in model.blocks], [1] * 8)
                torch.testing.assert_close(output, torch.full_like(output, 37))
                hit = prepare(model, 1, value=2.0)
                self.assertTrue(hit.hit)
                torch.testing.assert_close(
                    H3StageAdapter(model).reuse([hit])[0], torch.full_like(output, 38)
                )
                self.assertEqual([b.calls for b in model.blocks], [2] * 6 + [1] * 2)

    def test_selected_max_detects_changes_hidden_by_cumulative_cancellation(self):
        for blocks, expected in (((), True), ((1, 3, 6), False)):
            with self.subTest(probe_blocks=blocks):
                model = transformer("first_stage", probe_blocks=blocks)
                full_evaluation(model, prepare(model, 0))
                model.blocks[0].amount += 10
                model.blocks[2].amount -= 10
                self.assertEqual(prepare(model, 1).hit, expected)

    def test_probe_outside_ingress_is_rejected_before_computation(self):
        model = transformer("first_stage", end=5, probe_blocks=(1, 3, 6))
        with self.assertRaisesRegex(ValueError, "ingress stage"):
            prepare(model, 0)
        self.assertEqual([b.calls for b in model.blocks], [0] * 8)

    def test_full_evaluation_runs_every_block_once_for_each_policy(self):
        for policy, prefix in (
            ("none", 0),
            ("first_block", 1),
            ("first_stage", 6),
            ("first_stage_periodic", 6),
        ):
            with self.subTest(policy=policy):
                model = transformer(policy)
                prepared = prepare(model, 0)
                self.assertFalse(prepared.hit)
                self.assertEqual([b.calls for b in model.blocks], [1] * prefix + [0] * (8 - prefix))
                output = full_evaluation(model, prepared)
                torch.testing.assert_close(output, torch.full_like(output, 37))
                self.assertEqual([b.calls for b in model.blocks], [1] * 8)
                if prefix:
                    entry = prepared.payload["entry"]
                    prefix_residual = sum(range(1, prefix + 1))
                    torch.testing.assert_close(
                        entry.probe, torch.full_like(output, prefix_residual)
                    )
                    torch.testing.assert_close(
                        entry.residual, torch.full_like(output, 36 - prefix_residual)
                    )

    def test_hit_stays_on_ingress_and_skips_exactly_the_cached_suffix(self):
        from test_transport import MemoryChannel

        for policy, prefix in (
            ("first_block", 1),
            ("first_stage", 6),
            ("first_stage_periodic", 6),
        ):
            with self.subTest(policy=policy):
                model = transformer(policy)
                full_evaluation(model, prepare(model, 0))
                for block in model.blocks:
                    block.calls = 0
                hit = prepare(model, 1, value=2.0)
                self.assertTrue(hit.hit)
                stage = H3StageAdapter(model)
                outgoing, returning = MemoryChannel(), MemoryChannel()
                executor = PipelineExecutor(
                    stage, PipelineTopology((6, 2)), {(0, 1): outgoing, (1, 0): returning}
                )
                with patch.object(stage, "pack", side_effect=AssertionError("hit sent downstream")):

                    async def run():
                        output = await executor.submit([hit])[0]
                        self.assertTrue(outgoing.messages.empty())
                        closing = asyncio.create_task(executor.close())
                        self.assertTrue((await asyncio.to_thread(outgoing.recv)).value["stop"])
                        returning.send({"stop": True})
                        await closing
                        return output

                    output = asyncio.run(run())
                torch.testing.assert_close(output, torch.full_like(output, 38))
                self.assertEqual([b.calls for b in model.blocks], [1] * prefix + [0] * (8 - prefix))
                self.assertTrue(executor.receiver.done())
                self.assertFalse(executor.waiting)

    def test_first_stage_uses_whole_stage_change_for_the_decision(self):
        for policy, expected_hit in (("first_block", True), ("first_stage", False)):
            with self.subTest(policy=policy):
                model = transformer(policy)
                model.blocks[5].scale = 2
                full_evaluation(model, prepare(model, 0))
                # Block 1's residual remains constant; the six-block residual changes.
                self.assertEqual(prepare(model, 1, value=100.0).hit, expected_hit)

    def test_one_block_stage_makes_both_cache_policies_equivalent(self):
        models = [transformer(policy, end=1) for policy in ("first_block", "first_stage")]
        for step in range(6):
            decisions, outputs = [], []
            for model in models:
                prepared = prepare(model, step, value=float(step))
                decisions.append(prepared.hit)
                if prepared.hit:
                    output = H3StageAdapter(model).reuse([prepared])[0]
                    model.request_cache.commit(prepared.payload["entry"], step, True)
                else:
                    output = full_evaluation(model, prepared)
                outputs.append(output)
            self.assertEqual(decisions[0], decisions[1])
            torch.testing.assert_close(outputs[0], outputs[1])

    def test_disabled_reuse_keeps_exact_full_computation(self):
        model = transformer("first_stage", threshold=0)
        for step in range(3):
            prepared = prepare(model, step, value=float(step))
            self.assertFalse(prepared.hit)
            output = full_evaluation(model, prepared)
            torch.testing.assert_close(output, torch.full_like(output, step + 36))
        self.assertEqual([b.calls for b in model.blocks], [3] * 8)

    def test_mixed_prefix_boundaries_and_out_of_stage_execution_are_rejected(self):
        model = transformer("first_stage")
        first, second = prepare(model, 0), prepare(model, 0, key="other")
        second.payload["next_layer"] = 1
        with self.assertRaisesRegex(ValueError, "prefix boundaries"):
            H3StageAdapter(model).validate([first, second])
        with self.assertRaisesRegex(ValueError, "outside"):
            compute_stage(model, {"hidden": torch.ones(2, 3)}, start_layer=7)
