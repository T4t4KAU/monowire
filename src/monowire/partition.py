"""Allocate only owned H3 blocks while preserving global checkpoint names."""

from collections.abc import Callable, Iterable

import torch
from torch import nn


class MissingBlock(nn.Module):
    """Parameter-free placeholder that preserves global checkpoint block indices."""

    def forward(self, hidden: torch.Tensor, **kwargs) -> torch.Tensor:
        return hidden


def build_stage_blocks(
    num_layers: int,
    start: int,
    end: int,
    factory: Callable[[int], nn.Module],
) -> nn.ModuleList:
    if not 0 <= start < end <= num_layers:
        raise ValueError("Invalid stage layer range")
    return nn.ModuleList(
        [factory(index) if start <= index < end else MissingBlock() for index in range(num_layers)]
    )


def local_weights(
    weights: Iterable[tuple[str, torch.Tensor]], start: int, end: int, *, include_io: bool
):
    """Keep local H3 blocks; only request owners load input/output components."""
    for name, value in weights:
        parts = name.split(".")
        if parts[0] in {"blocks", "transformer_blocks"}:
            if len(parts) < 3 or not parts[1].isdigit():
                raise ValueError(f"Malformed H3 block weight: {name}")
            if not start <= int(parts[1]) < end:
                continue
        elif not include_io:
            continue
        yield name, value
