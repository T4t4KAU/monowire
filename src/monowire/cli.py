"""Continuous H3 pipeline inference; launch one worker per GPU with torchrun."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt", action="append", required=True, help="Repeat for multiple requests"
    )
    parser.add_argument("--pp", type=int, default=8)
    parser.add_argument("--max-batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1, help="Maximum concurrent requests")
    parser.add_argument(
        "--microbatch-size",
        type=int,
        default=1,
        help="Requests per physical block forward",
    )
    parser.add_argument("--cache", choices=("none", "first_block"), default="none")
    parser.add_argument("--cache-threshold", type=float, default=0.17)
    parser.add_argument("--cache-warmup", type=int, default=2)
    parser.add_argument("--cache-max-skip", type=int, default=2)
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
    if sys.version_info < (3, 11):
        parser.error(
            "The pinned vLLM-Omni requires Python >=3.11; use an isolated Omni environment"
        )
    if args.pp < 1:
        parser.error("--pp must be positive")
    import torch

    from .cache import CacheConfig
    from .continuous import ContinuousConfig
    from .models.h3.runtime import create_continuous_engine

    engine = create_continuous_engine(
        args.model,
        pp_size=args.pp,
        config=ContinuousConfig(args.batch_size, args.microbatch_size, args.max_batch_wait_ms),
        cache=CacheConfig(args.cache, args.cache_threshold, args.cache_warmup, args.cache_max_skip),
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
        if engine.rank != 0:
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
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(outputs, destination)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
