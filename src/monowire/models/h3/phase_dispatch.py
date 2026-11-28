"""CPU phase handoffs for independent request replicas, preserving input order."""

import torch.distributed as dist

from .phase_completion import complete_phase
from .phase_utils import cpu_tree


def map_phase(items, operation, ranks, group):
    """All world ranks participate; only selected ranks load/run the phase.

    operation maps a list to an equally sized list. Errors are agreed before
    gathering results, including errors on replicas without the ingress role.
    """
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    shards = None
    if rank == 0:
        shards = [[] for _ in range(world)]
        for index, item in enumerate(items):
            shards[ranks[index % len(ranks)]].append((index, cpu_tree(item)))
    incoming = [None]
    dist.scatter_object_list(incoming, shards, src=0, group=group)
    indexed = incoming[0]

    def operate():
        if not indexed:
            return []
        values = operation([item for _, item in indexed])
        if len(values) != len(indexed):
            raise ValueError("Phase changed the number of requests")
        return [(index, cpu_tree(value)) for (index, _), value in zip(indexed, values)]

    result = complete_phase(operate)
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(result, gathered, dst=0, group=group)
    if rank != 0:
        return None
    return [
        value
        for _, value in sorted(
            (item for shard in gathered for item in shard), key=lambda item: item[0]
        )
    ]
