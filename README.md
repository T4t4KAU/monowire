# MonoWire

Pipeline-parallel inference for the original MiniMax H3, built on vLLM-Omni Diffusion. Supports continuous batching, replicated ingress stages, and FirstBlockCache / FirstStageCache bypass.

## Installation

Install on the GPU server with Python 3.12. Runtime versions are pinned in [runtime.lock.json](runtime.lock.json); the vLLM-Omni source is included in `third_party/vllm-omni`.

```bash
python3.12 -m venv .venv-omni
source .venv-omni/bin/activate
python -m pip install --upgrade pip
python -m pip install "vllm==0.32.0+cu129" "torch==2.13.0+cu129" \
  --extra-index-url https://wheels.vllm.ai/0.32.0/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129
VLLM_OMNI_TARGET_DEVICE=cuda python -m pip install -e third_party/vllm-omni
python -m pip install -e .
python -m pip check
```

## Usage

Run on eight GPUs with the original H3 native FL2VA checkpoint:

```bash
torchrun --standalone --nproc-per-node=8 -m monowire.cli \
  --pp 8 --model /path/to/FL2VA \
  --prompt "A cat walks along a beach." \
  --prompt "A dog runs through a field." \
  --batch-size 32 --microbatch-size 8 --downstream-batch-size 8 \
  --cache first_stage --cache-max-skip 2 \
  --height 384 --width 672 --aspect-ratio 16:9 \
  --frames 124 --steps 20 --seed 42 \
  --cpu-offload --trust-remote-code \
  --output outputs/results.pt
```

Repeat `--prompt` to add requests. The native VAE requires `--trust-remote-code`. Results are saved with `torch.save` as request IDs, video/audio outputs, and errors.

| Option | Meaning |
| --- | --- |
| `--pp` | Total GPU workers; must match the process count. |
| `--batch-size` | Maximum active requests per ingress replica. |
| `--microbatch-size` | Maximum requests per ingress forward. |
| `--downstream-batch-size` | Maximum requests per suffix forward. |
| `--cache` | `none` (default), `first_block`, `first_stage`, or `first_stage_periodic`. |
| `--cache-threshold` | Reuse threshold; defaults to `0.17`. |
| `--cache-max-skip` | Maximum consecutive reuse steps; defaults to `2`. |

`first_block` and `first_stage` compare prefix residuals with the last full evaluation. `first_stage_periodic` compares adjacent steps. Cache hits finish the step at ingress; misses enter the shared suffix. Batch sizes are upper limits, so batches may be partial. Use `--cache none` to disable reuse and calibrate thresholds against output quality.

For two ingress replicas and six suffix GPUs, add:

```bash
--ingress-replicas 2 --stage-layers 6 7 7 7 7 8 8
```

The first six blocks are replicated on GPUs 0 and 1; the remaining 44 blocks run on GPUs 2–7. Active and ingress batch limits apply to each replica. Outputs use separate `.rank0.pt` and `.rank1.pt` files.

For Python, import `create_continuous_engine`, `ContinuousConfig`, `CacheConfig`, and `PipelineTopology` from `monowire`. Use `engine.generate(prompts, sampling_params)` for a finite request list or `engine.stream(source)` for an async source yielding `(request_id, prompt, sampling_params)`. All ranks must participate; `generate` shards the common list across ingress replicas, while `stream` accepts a separate source on each ingress. Close an unfinished stream with `aclose()` before closing the engine.

The current H3 adapter supports dense text-to-audio/video inference with Euler sampling. Run `monowire --help` for all options.
