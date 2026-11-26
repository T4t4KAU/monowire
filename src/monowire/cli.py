"""Continuous H3 pipeline inference; launch one worker per GPU with torchrun."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt", action="append", required=True, help="Repeat for multiple requests"
    )
    parser.add_argument(
        "--image", action="append", help="Local first-frame image; repeat once per --prompt"
    )
    parser.add_argument(
        "--task", choices=("t2va", "fl2va"), help="Defaults to fl2va with images, otherwise t2va"
    )
    parser.add_argument("--pp", type=int, default=8)
    parser.add_argument(
        "--phased", action="store_true", help="Finite job: Qwen PP, VAE encode, DiT PP, VAE decode"
    )
    parser.add_argument(
        "--overlap-encoder-dit",
        action="store_true",
        help="With --phased, switch each drained Qwen stage to DiT independently",
    )
    parser.add_argument(
        "--encoder-stage-layers",
        type=int,
        nargs="+",
        help="Qwen PP layers per GPU (sum 50; first stage at least 3)",
    )
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
    if args.overlap_encoder_dit and not args.phased:
        parser.error("--overlap-encoder-dit requires --phased")
    if args.encoder_stage_layers and not args.phased:
        parser.error("--encoder-stage-layers requires --phased")
    if args.phased and args.ingress_replicas != 1:
        parser.error("--phased currently requires --ingress-replicas 1")
    if args.phased:
        from .models.h3.phase_utils import encoder_partition

        try:
            encoder_partition(args.pp, args.encoder_stage_layers)
        except ValueError as error:
            parser.error(str(error))
    task = args.task or ("fl2va" if args.image else "t2va")
    if task == "fl2va" and not args.image:
        parser.error("--task fl2va requires --image")
    if args.image and task != "fl2va":
        parser.error("--image requires --task fl2va")
    if args.image and len(args.image) != len(args.prompt):
        parser.error("Provide exactly one --image per --prompt, in matching order")
    prompts = [{"prompt": prompt} for prompt in args.prompt]
    if args.image:
        from PIL import Image

        for prompt, filename in zip(prompts, args.image, strict=True):
            try:
                with Image.open(filename) as image:
                    prompt["multi_modal_data"] = {"image": image.convert("RGB")}
            except (OSError, ValueError) as error:
                parser.error(f"Cannot load --image {filename!r}: {error}")
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

    factory = create_continuous_engine
    options = {"cpu_offload": args.cpu_offload}
    if args.phased:
        from .models.h3.phased import create_phased_engine

        factory = create_phased_engine
        options = {
            "encoder_stage_layers": args.encoder_stage_layers,
            "overlap_encoder_dit": args.overlap_encoder_dit,
        }
    engine = factory(
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
        trust_remote_code=args.trust_remote_code,
        **options,
    )
    try:
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        outputs = list(
            engine.generate(
                prompts,
                OmniDiffusionSamplingParams(
                    height=args.height,
                    width=args.width,
                    num_frames=args.frames,
                    num_inference_steps=args.steps,
                    seed=args.seed,
                    extra_args={"task": task, "aspect_ratio": args.aspect_ratio},
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
