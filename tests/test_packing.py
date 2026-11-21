"""H3 packed attention and timestep isolation."""

import unittest

import torch


class PackingTests(unittest.TestCase):
    def test_single_and_packed_batches_reject_unsupported_attention(self):
        from monowire.models.h3.packing import pack_block_kwargs

        for option in ({"vdn_window": 0},):
            for size in (1, 2):
                with self.subTest(option=option, size=size), self.assertRaises(ValueError):
                    pack_block_kwargs([dict(num_requests=1, **option)] * size)

    def test_packed_padding_and_timestep_offsets(self):
        from monowire.models.h3.packing import pack_block_kwargs

        def args(used, total, timesteps):
            return dict(
                num_requests=1,
                packed_total=total,
                max_seqlen=used,
                cu_seqlens=torch.tensor([0, used, total], dtype=torch.int32),
                t_emb=torch.zeros(timesteps, 4),
                combined_indices=torch.zeros(total, dtype=torch.long),
                rope_table=torch.zeros(total, 4),
            )

        packed = pack_block_kwargs([args(5, 5, 2), args(4, 6, 1)])
        self.assertEqual(packed["cu_seqlens"].tolist(), [0, 5, 9, 11])
        self.assertEqual(packed["combined_indices"].tolist(), [0] * 5 + [6] * 6)
        self.assertEqual(packed["t_emb"].shape[0], 3)
