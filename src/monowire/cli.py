"""Continuous H3 pipeline inference; launch one worker per GPU with torchrun."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt", action="append", required=True, help="Repeat for multiple requests"
    )
    parser.add_argument("--pp", type=int, default=8)
    parser.add_argument("--ingress-replicas", type=int, default=1)
    parser.add_argument("--stage-layers", type=int, nargs="+")
    parser.add_argument("--downstream-batch-size", type=int, default=8)
    parser.add_argument(
        "--async-transport",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Grouped transfers and deferred CUDA completion waits (enabled by default)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Maximum concurrent requests")
    parser.add_argument(
        "--microbatch-size",
        type=int,
        default=1,
        help="Requests per physical block forward",
    )
    parser.add_argument(
        "--cache",
        choices=("none", "first_block", "first_stage", "first_stage_periodic"),
        default="none",
    )
    parser.add_argument("--cache-threshold", type=float, default=0.17)
    parser.add_argument("--cache-warmup", type=int, default=2)
    parser.add_argument("--cache-max-skip", type=int, default=2)
    parser.add_argument(
        "--cache-probe-blocks",
        type=int,
        nargs="+",
        default=(),
        help="One-based ingress block indices; use the maximum relative residual change",
    )
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument(
        "--aspect-ratio",
        choices=("21:9", "16:9", "4:3", "1:1", "3:4", "9:16"),
        default="1:1",
    )
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if args.pp < 1:
        parser.error("--pp must be positive")
    import torch

    from .cache import CacheConfig
    from .continuous import ContinuousConfig
    from .models.h3.runtime import create_continuous_engine
    from .topology import PipelineTopology

    if args.ingress_replicas != 1 and not args.stage_layers:
        parser.error("--ingress-replicas requires explicit --stage-layers")
    topology = (
        PipelineTopology(tuple(args.stage_layers), args.ingress_replicas)
        if args.stage_layers
        else None
    )

    engine = create_continuous_engine(
        args.model,
        pp_size=args.pp,
        topology=topology,
        config=ContinuousConfig(
            max_active_requests=args.batch_size,
            microbatch_size=args.microbatch_size,
            downstream_batch_size=args.downstream_batch_size,
            async_transport=args.async_transport,
        ),
        cache=CacheConfig(
            args.cache,
            args.cache_threshold,
            args.cache_warmup,
            args.cache_max_skip,
            probe_blocks=tuple(args.cache_probe_blocks),
        ),
        cpu_offload=args.cpu_offload,
        trust_remote_code=args.trust_remote_code,
    )
    try:
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        outputs = list(
            engine.generate(
                [{"prompt": prompt} for prompt in args.prompt],
                OmniDiffusionSamplingParams(
                    height=args.height,
                    width=args.width,
                    num_frames=args.frames,
                    num_inference_steps=args.steps,
                    seed=args.seed,
                    extra_args={"task": "t2va", "aspect_ratio": args.aspect_ratio},
                ),
            )
        )
        if not engine.is_ingress:
            return
        outputs = [
            {
                "request_id": item.request_id,
                "output": item.output.output if item.output else None,
                "error": item.error,
            }
            for item in outputs
        ]
        destination = Path(args.output)
        if engine.ingress_replicas > 1:
            destination = destination.with_stem(f"{destination.stem}.rank{engine.rank}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(outputs, destination)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
