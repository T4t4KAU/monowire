"""CPU handoffs and encoder partition validation, independent of native imports."""

from dataclasses import fields, is_dataclass, replace

import torch


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                field.name: cpu_tree(getattr(value, field.name))
                for field in fields(value)
                if field.init
            },
        )
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def encoder_partition(world_size, layers=None, *, total=50, deepstack=3):
    if not 1 <= world_size <= total:
        raise ValueError("Encoder PP size must be between 1 and the retained layer count")
    if layers is None:
        base, remainder = divmod(total, world_size)
        layers = tuple(base + (rank < remainder) for rank in range(world_size))
    layers = tuple(layers)
    if len(layers) != world_size or any(type(n) is not int or n < 1 for n in layers):
        raise ValueError("Encoder stage layers must contain one positive integer per GPU")
    if sum(layers) != total:
        raise ValueError(f"Encoder stage layers must cover exactly {total} layers")
    if layers[0] < deepstack:
        raise ValueError(f"Encoder first stage must contain the {deepstack} DeepStack layers")
    return layers
