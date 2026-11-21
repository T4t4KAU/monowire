# MonoWire

Pipeline-parallel inference for the original MiniMax H3 model, built on vLLM-Omni Diffusion. Supports continuous request scheduling, cache-aware batching, and FirstBlockCache pipeline bypass.

## Installation

Run the following commands on the GPU server from the repository root. Use Python 3.12 and a CUDA environment compatible with the pinned vLLM runtime. Local source editing does not require installing this GPU runtime.

```bash
python3.12 -m venv .venv-omni
source .venv-omni/bin/activate
python -m pip install --upgrade pip
```

The vLLM-Omni source is included in `third_party/vllm-omni`, with the H3 integration already applied. Install this source directly. Its upstream revision and required runtime versions are recorded in `runtime.lock.json`.

The pinned runtime uses the prebuilt **vLLM 0.32.0 / PyTorch 2.13.0 / CUDA 12.9** packages. No vLLM compilation is needed:

```bash
python -m pip install "vllm==0.32.0+cu129" "torch==2.13.0+cu129" \
  --extra-index-url https://wheels.vllm.ai/0.32.0/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129
VLLM_OMNI_TARGET_DEVICE=cuda python -m pip install -e third_party/vllm-omni
python -m pip install -e .
python -m pip check
monowire --help
```

MonoWire targets the bundled vLLM-Omni **v0.32.0** source and its native H3 step interfaces. Use the versions in `runtime.lock.json`.

## Usage

Use the original H3 native **FL2VA** checkpoint. Replace `/path/to/FL2VA` below with its directory. The native VAE requires `--trust-remote-code`.

### Continuous inference with FirstBlockCache

Launch one process per pipeline stage. This example uses eight GPUs:

```bash
torchrun --standalone --nproc-per-node=8 -m monowire.cli \
  --pp 8 \
  --model /path/to/FL2VA \
  --prompt "A cat walks along a beach." \
  --prompt "A dog runs through a field." \
  --batch-size 32 --microbatch-size 8 \
  --cache first_block --max-batch-wait-ms 2 \
  --height 384 --width 672 --aspect-ratio 16:9 \
  --frames 124 --steps 20 --seed 42 \
  --cpu-offload --trust-remote-code \
  --output outputs/results.pt
```

Repeat `--prompt` to submit more requests. Requests advance independently and are grouped by cache status and input compatibility. Completed requests leave the active set so waiting requests can enter.

| Option | Meaning |
| --- | --- |
| `--pp` | Number of pipeline stages; must match the process count. |
| `--batch-size` | Maximum number of active requests. |
| `--microbatch-size` | Maximum requests per physical batch; must not exceed `--batch-size`. |
| `--max-batch-wait-ms` | Waiting limit before dispatching a partial ready batch. |
| `--cache` | Accepts `none` or `first_block`. |
| `--cache-threshold` | Cache reuse threshold; defaults to `0.17`. |
| `--cache-warmup` | Initial steps without reuse; defaults to `2`. |
| `--cache-max-skip` | Maximum consecutive cache hits; defaults to `2`. |

Use `--cache none` for uncached inference. Continuous execution currently supports dense H3 text-to-audio/video inference with Euler sampling and pure pipeline parallelism.

### Outputs and Python integration

The CLI saves results with `torch.save`; it does not write MP4 files. The CLI writes on rank 0 only, saving a list of dictionaries containing `request_id`, the native video/audio `output`, and any `error`.

For Python applications, import `create_continuous_engine`, `ContinuousConfig`, and `CacheConfig` from `monowire`. Use `engine.generate(prompts, sampling_params)` for a finite batch or `engine.stream(async_request_source)` for live arrivals. The source yields `(request_id, prompt, sampling_params)` tuples. All pipeline ranks must enter the same service lifecycle; only rank 0 consumes requests and emits results.
