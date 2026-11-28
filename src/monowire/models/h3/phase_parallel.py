"""Independent phase placement and supported H3 parallel strategies.

Ranks are torchrun global ranks (not CUDA device ordinals). The current DiT
backend owns the whole worker world; encoder and codec placement are independent.
"""

from dataclasses import dataclass, replace

from .phase_utils import encoder_partition


@dataclass(frozen=True)
class StageParallelConfig:
    mode: str = "single"
    ranks: tuple[int, ...] = (0,)
    stage_layers: tuple[int, ...] | None = None

    def __post_init__(self):
        object.__setattr__(self, "ranks", tuple(self.ranks))
        if self.stage_layers is not None:
            object.__setattr__(self, "stage_layers", tuple(self.stage_layers))
        if self.mode not in {"single", "pp", "dp"}:
            raise ValueError(f"Unsupported parallel mode: {self.mode!r}; use single, pp or dp")
        if not self.ranks or any(type(r) is not int or r < 0 for r in self.ranks):
            raise ValueError("ranks must contain nonnegative integers")
        if tuple(sorted(set(self.ranks))) != self.ranks:
            raise ValueError("ranks must be unique and increasing")
        if self.mode == "single" and len(self.ranks) != 1:
            raise ValueError("single mode requires exactly one rank")
        if self.stage_layers is not None and self.mode == "dp":
            raise ValueError("DP replicas own the full model; stage_layers is not applicable")


@dataclass(frozen=True)
class PhaseParallelConfig:
    encoder: StageParallelConfig
    denoiser: StageParallelConfig
    decoder: StageParallelConfig = StageParallelConfig()
    media_encoder: StageParallelConfig = StageParallelConfig()

    @classmethod
    def resolve(cls, world_size, value=None, *, encoder_stage_layers=None, topology=None):
        if type(world_size) is not int or world_size < 1:
            raise ValueError("worker count must be positive")
        full = StageParallelConfig("pp" if world_size > 1 else "single", tuple(range(world_size)))
        if value is None:
            value = cls(full, full)
        elif isinstance(value, dict):
            unknown = set(value) - {"encoder", "denoiser", "decoder", "media_encoder"}
            if unknown:
                raise ValueError(f"Unknown phase configuration keys: {sorted(unknown)}")

            def parse_stage(options):
                options = dict(options)
                if options.get("ranks") == "all":
                    options["ranks"] = tuple(range(world_size))
                return StageParallelConfig(**options)

            try:
                value = cls(
                    **{
                        name: parse_stage(value[name]) if name in value else default
                        for name, default in (
                            ("encoder", full),
                            ("denoiser", full),
                            ("decoder", StageParallelConfig()),
                            ("media_encoder", StageParallelConfig()),
                        )
                    }
                )
            except TypeError as exc:
                raise ValueError(f"Invalid phase parallel configuration: {exc}") from exc
        if not isinstance(value, cls):
            raise ValueError("phase_parallel must be a PhaseParallelConfig or an object")
        for name in ("encoder", "denoiser", "decoder", "media_encoder"):
            stage = getattr(value, name)
            if not isinstance(stage, StageParallelConfig):
                raise ValueError(f"{name} must be a StageParallelConfig")
            if max(stage.ranks) >= world_size:
                raise ValueError(f"{name} rank is outside the worker world")
            if name in {"decoder", "media_encoder"}:
                if stage.mode not in {"single", "dp"} or stage.stage_layers is not None:
                    raise ValueError(f"{name} supports single or request DP, without stage_layers")
        if value.denoiser.mode not in {"single", "pp"} or value.denoiser.ranks != full.ranks:
            raise ValueError("DiT currently requires single/PP over all worker ranks")
        enc = value.encoder
        if encoder_stage_layers is not None:
            if enc.stage_layers is not None and enc.stage_layers != tuple(encoder_stage_layers):
                raise ValueError("Conflicting encoder layer partitions")
            enc = replace(enc, stage_layers=tuple(encoder_stage_layers))
        enc = (
            replace(
                enc,
                stage_layers=encoder_partition(
                    len(enc.ranks) if enc.mode == "pp" else 1, enc.stage_layers
                ),
            )
            if enc.mode != "dp"
            else enc
        )
        dit = value.denoiser
        if topology is not None:
            if topology.world_size != world_size or topology.ingress_replicas != 1:
                raise ValueError("Phased H3 requires one DiT ingress and all worker ranks")
            if dit.stage_layers is not None and dit.stage_layers != topology.stage_layers:
                raise ValueError("Conflicting DiT layer partitions")
            dit = replace(dit, stage_layers=topology.stage_layers)
        if dit.stage_layers is not None:
            layers = dit.stage_layers
            if (
                len(layers) != world_size
                or any(type(n) is not int or n < 1 for n in layers)
                or sum(layers) != 50
            ):
                raise ValueError(
                    "DiT stage_layers must cover 50 layers, one positive count per rank"
                )
        return replace(value, encoder=enc, denoiser=dit)

    def validate_overlap(self):
        if self.encoder.mode != "pp" or len(self.encoder.ranks) <= 1:
            raise ValueError("overlap_encoder_dit requires Qwen pipeline parallelism")
        if self.encoder.ranks != self.denoiser.ranks:
            raise ValueError("Rolling encoder/DiT overlap requires identical ordered rank sets")
