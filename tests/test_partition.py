"""Stage-local parameter allocation and checkpoint filtering."""

import unittest

import torch
from torch import nn

from monowire.partition import MissingBlock, build_stage_blocks, local_weights


class AffineBlock(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(index + 1)))

    def forward(self, hidden, **kwargs):
        return hidden * self.weight + 1


class PartitionTests(unittest.TestCase):
    def test_only_owned_parameters_are_allocated(self):
        blocks = build_stage_blocks(7, 2, 4, AffineBlock)
        self.assertEqual(sum(isinstance(block, MissingBlock) for block in blocks), 5)
        self.assertEqual(list(blocks.state_dict()), ["2.weight", "3.weight"])

    def test_weight_filter_preserves_shared_parameters(self):
        weights = [
            ("blocks.0.weight", 0),
            ("blocks.1.weight", 1),
            ("transformer_blocks.2.attn.weight", 2),
            ("final_layer.weight", 3),
            ("token_refiner.blocks.0.weight", 4),
        ]
        self.assertEqual(list(local_weights(weights, 1, 3)), weights[1:])

    def test_single_rank(self):
        blocks = build_stage_blocks(2, 0, 2, AffineBlock)
        hidden = torch.tensor(1.0)
        for block in blocks:
            hidden = block(hidden)
        self.assertEqual(hidden.item(), 5.0)


if __name__ == "__main__":
    unittest.main()
