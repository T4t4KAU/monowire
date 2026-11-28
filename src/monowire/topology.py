"""Layer placement for replicated ingress and a shared pipeline suffix."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PipelineTopology:
    """Each ingress replica owns the first logical stage and its request state.

    stage_layers lists logical stages, excluding duplicate ingress copies.
    Downstream workers batch ready misses from all ingress replicas.
    """

    stage_layers: tuple[int, ...]
    ingress_replicas: int = 1
    buffer_size: int = field(default=2, kw_only=True)
    buffer_bytes: int | None = field(default=None, kw_only=True)
    receive_prefetch: int = field(default=2, kw_only=True)

    def __post_init__(self):
        layers = tuple(self.stage_layers)
        if not layers or any(type(n) is not int or n < 1 for n in layers):
            raise ValueError("A topology requires nonempty stages")
        sizes = (self.ingress_replicas, self.buffer_size)
        if any(type(n) is not int or n < 1 for n in sizes):
            raise ValueError("Replica count and buffer size must be positive")
        if type(self.receive_prefetch) is not int or self.receive_prefetch < 1:
            raise ValueError("Receive prefetch must be a positive integer")
        if self.buffer_bytes is not None:
            if type(self.buffer_bytes) is not int or self.buffer_bytes < 1:
                raise ValueError("Buffer byte budget must be positive")
            if self.ingress_replicas != 1:
                raise ValueError("Byte buffering currently requires one ingress replica")
        object.__setattr__(self, "stage_layers", layers)

    @property
    def world_size(self):
        return self.ingress_replicas + len(self.stage_layers) - 1

    def layer_range(self, rank):
        if not 0 <= rank < self.world_size:
            raise ValueError("Rank is outside the pipeline topology")
        index = 0 if rank < self.ingress_replicas else rank - self.ingress_replicas + 1
        start = sum(self.stage_layers[:index])
        return start, start + self.stage_layers[index]

    @property
    def edges(self):
        if len(self.stage_layers) == 1:
            return []
        first, last = self.ingress_replicas, self.world_size - 1
        return (
            [(rank, first) for rank in range(first)]
            + [(rank, rank + 1) for rank in range(first, last)]
            + [(last, rank) for rank in range(first)]
        )
