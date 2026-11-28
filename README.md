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

For image-and-text-to-video (FL2VA), add one local `--image` per `--prompt`, in
matching order. The task defaults to `fl2va` when images are provided:

```bash
torchrun --standalone --nproc-per-node=8 -m monowire.cli \
  --pp 8 --model /path/to/FL2VA \
  --prompt "Animate this scene with natural, smooth motion and a steady camera." \
  --image /path/to/first-frame.png \
  --height 384 --width 672 --aspect-ratio 16:9 \
  --frames 124 --steps 20 --seed 42 --cache none \
  --cpu-offload --trust-remote-code \
  --output outputs/image-video.pt
```

Images are converted to RGB; native H3 preprocessing handles resizing and image
encoding. Each request retains its own image conditions through every Euler step
and uses native video/audio decoding. This is an API example, not a validated
4090 memory or throughput configuration: codec weights and activations must fit
alongside the resident ingress DiT stage even with CPU offload enabled.

| Option | Meaning |
| --- | --- |
| `--pp` | Total GPU workers; must match the process count. |
| `--batch-size` | Maximum active requests per ingress replica. |
| `--microbatch-size` | Maximum requests per ingress forward. |
| `--downstream-batch-size` | Maximum requests per suffix forward. |
| `--async-transport` / `--no-async-transport` | Grouped transfers and deferred CUDA completion waits; enabled by default. Use `--no-async-transport` to disable. |
| `--cache` | `none` (default), `first_block`, `first_stage`, or `first_stage_periodic`. |
| `--cache-threshold` | Reuse threshold; defaults to `0.17`. |
| `--cache-max-skip` | Maximum consecutive reuse steps; defaults to `2`. |

`first_block` and `first_stage` compare prefix residuals with the last full evaluation. `first_stage_periodic` compares adjacent steps. Cache hits finish the step at ingress; misses enter the shared suffix. Batch sizes are upper limits, so batches may be partial. Prefix lookahead is bounded separately from the send queue (`ContinuousConfig.prefix_buffer_batches`, default 2). Use `--cache none` to disable reuse and calibrate thresholds against output quality.

For two ingress replicas and six suffix GPUs, add:

```bash
--ingress-replicas 2 --stage-layers 6 7 7 7 7 8 8
```

The first six blocks are replicated on GPUs 0 and 1; the remaining 44 blocks run on GPUs 2–7. Active and ingress batch limits apply to each replica. Outputs use separate `.rank0.pt` and `.rank1.pt` files.

For Python, import `create_continuous_engine`, `ContinuousConfig`, `CacheConfig`, and `PipelineTopology` from `monowire`. Use `engine.generate(prompts, sampling_params)` for a finite request list or `engine.stream(source)` for an async source yielding `(request_id, prompt, sampling_params)`. All ranks must participate; `generate` shards the common list across ingress replicas, while `stream` accepts a separate source on each ingress. Close an unfinished stream with `aclose()` before closing the engine.

The H3 adapter supports dense T2VA/FL2VA with the original FL2VA checkpoint and
Ref2VA with its original Ref2VA checkpoint, using Euler sampling. In Python, pass
`{"prompt": text, "multi_modal_data": {"image": rgb_pil_image}}` and set
`sampling_params.extra_args["task"] = "fl2va"`, or omit the task to let native H3
infer it from the image. T2VA and FL2VA requests share the same DiT and may be
batched together; their attention boundaries and conditions stay independent.
For first-and-last-frame conditioning in Python, pass
`"image": [first_rgb_image, last_rgb_image]`; native H3 assigns frame indices
`[0, -1]`. The CLI accepts one first-frame image per request.
Use the matching partition for each job; FL2VA and Ref2VA weights are never
co-resident. Run `monowire --help` for all options.

### Experimental byte-based DiT residency

The Python API can replace GPU task-count limits with byte reservations for
BS1, one ingress, cache-free T2VA and first-frame FL2VA. Use phased execution or
pre-encoded conditions; encoder/DiT loading overlap is not supported in this mode.

```python
from monowire import ContinuousConfig, MemoryLimits

MiB, GiB = 1024**2, 1024**3
config = ContinuousConfig(
    max_active_requests=None,
    microbatch_size=1,
    downstream_batch_size=1,
    host_pending_limit=None,  # Finite job already fits CPU RAM; no intake count cap.
    memory_limits=MemoryLimits(
        request_bytes=4 * GiB,
        pipeline_bytes=9 * 640 * MiB,
        workspace_bytes=3 * GiB,
    ),
)
# Pass config to create_phased_engine(..., config=config).
```

These values are trial settings, not a validated 4090 performance recommendation.
Admission reserves request state and one returned hidden tensor using encoded
token length. Live state and in-flight context are checked against that estimate;
`request_bytes_per_token` is a configurable conservative coefficient. It is not a
measurement of kernel scratch memory. Oversize requests receive an error instead
of waiting forever. `host_pending_limit` still bounds submitted/unconsumed CPU
requests when set (default128), independently of GPU admission. Set it to `None`
for a finite job whose CPU residency is already accounted for; otherwise this
host backpressure can indirectly limit the number of GPU-active requests too.

The pipeline budget is divided into nine equal domains: two channel in-flight
windows, receive queue, computation window, send queue, prefix lookahead, and
three handoff reserves. This preserves space for computation and return progress
when input queues fill. Queue counts vary with payload size; `buffer_size` and
`prefix_buffer_batches` no longer govern these GPU buffers in byte mode.
Storage views are deduplicated within a payload; different domains conservatively
charge overlapping ownership separately. This is not yet a freely shared,
globally deduplicated memory pool.

In-flight CUDA buffers remain charged until their completion event. Removing a
message from a receive/send queue does not immediately return its bytes: receive
credit is held through consumption, and send credit through handoff to the
channel's separately bounded in-flight window. Shared dictionaries have a
separate fixed byte cap. After weights load, all ranks check that the configured
dynamic reservations plus workspace and allocator/library headroom fit before
creating pipeline channels. Workspace must be sized from profiling; this policy
does not predict arbitrary kernel allocation peaks or cover VAE/encoder phases.
The default configuration retains the existing count-based behavior.

### Finite jobs with separate model phases

`--phased` runs the entire finite job as Qwen-VL layer PP → VAE encoding →
DiT PP → VAE decoding. Conditions and generated latents are retained in CPU RAM;
model weights are released at phase boundaries. Qwen weights are partitioned by
layer, with vision, embeddings and the first three DeepStack injections on the first encoder rank.
The encoder preserves H3's unnormalized layer-50 output and multimodal positions.
There is no encoder TP in this mode.

```bash
torchrun --standalone --nproc-per-node=8 -m monowire.cli \
  --phased --pp 8 --model /path/to/FL2VA \
  --encoder-stage-layers 7 7 6 6 6 6 6 6 --stage-layers 6 7 7 6 6 6 6 6 \
  --prompt "Animate this scene naturally." --image /path/to/first.png \
  --height 384 --width 672 --aspect-ratio 16:9 --frames 124 --steps 20 \
  --batch-size 16 --microbatch-size 1 --downstream-batch-size 1 --trust-remote-code \
  --output outputs/phased.pt
```

Each phase can select its own parallel strategy with `--phase-parallel-config`
(JSON), or `create_phased_engine(..., phase_parallel=...)` in Python. The Python
argument accepts the same dictionary or `PhaseParallelConfig` / `StageParallelConfig`
from `monowire.models.h3.phase_parallel`.

```bash
# Add to an eight-worker --phased invocation:
--phase-parallel-config '{
  "encoder": {"mode": "pp", "ranks": [0, 1, 2, 3], "stage_layers": [13, 13, 12, 12]},
  "media_encoder": {"mode": "dp", "ranks": [0, 4]},
  "denoiser": {"mode": "pp", "ranks": [0, 1, 2, 3, 4, 5, 6, 7],
               "stage_layers": [6, 7, 7, 6, 6, 6, 6, 6]},
  "decoder": {"mode": "dp", "ranks": "all"}
}'
```

| Phase | Implemented strategies | Placement |
| --- | --- | --- |
| `encoder` (Qwen) | `single`, layer `pp`, request `dp` | Independent subset of worker ranks |
| `media_encoder` (video/audio VAE, including refinement anchors) | `single`, request `dp` | Independent subset |
| `denoiser` (DiT) | `pp`; `single` with one worker | All launched workers in this backend |
| `decoder` (video/audio VAE) | `single`, request `dp` | Independent subset |

For all-card decoding, use
`--phase-parallel-config '{"decoder":{"mode":"dp","ranks":"all"}}'`.
`"all"` expands to every launched worker (for example, ranks 0–7 with eight workers).
Each GPU decodes complete requests independently after DiT weights are released.
In barrier-separated execution, only ranks assigned requests load the decoder.
With rolling handoff, every configured replica loads before requesting work;
some replicas may finish loading after all requests have been claimed.
The existing single-rank default remains available when no decoder setting is supplied.

`ranks` are increasing, unique **global torchrun ranks**, not physical GPU IDs.
PP divides model layers; DP assigns whole requests round-robin to replicas, each
of which must fit the complete phase model. DP does not pack requests or estimate
compute cost from request counts. CPU handoffs restore original request order;
only global rank 0 returns final outputs. Idle replicas do not load codecs or Qwen.
Qwen vision/embeddings/DeepStack live on the first *encoder* rank, which need not
be global rank 0. This CPU object transport is intended for finite phased jobs,
not a zero-copy or streaming cross-stage transport.

Without this option, behavior remains Qwen PP and DiT PP on all workers, with
VAE encoding/decoding on rank 0. `--pp` remains the worker count / DiT PP degree.
Omitted phase entries retain these defaults. Legacy `--encoder-stage-layers` and
`--stage-layers` remain supported; conflicting new/legacy partitions are rejected.
Each PP partition must cover the 50 retained layers. Unimplemented TP/SP, codec
PP, DiT DP and DiT subsets are rejected rather than silently ignored.

Independent placement uses barriers between phases; it does not enable concurrent
execution by itself. Request DP and encoder PP subgroups use heartbeat-protected
completion agreement so idle workers do not enter result collectives while peers
are still loading or computing. `--overlap-encoder-dit` requires Qwen **PP with more than one
rank**, and the current rolling handoff additionally requires exactly the same
ordered rank set as DiT. Encoder and DiT layer partitions may still differ.

To enable both rolling phase boundaries with all-card decoder DP, add:

```bash
--overlap-stages --phase-parallel-config '{"decoder":{"mode":"dp","ranks":"all"}}'
```

Python uses `overlap_encoder_dit=True, overlap_dit_decoder=True`. The individual
CLI flags `--overlap-encoder-dit` and `--overlap-dit-decoder` can be enabled separately.
They are opt-in; existing invocations retain barrier-separated execution.

For the decoder handoff, each rank must first finish its DiT STOP protocol, drain
CUDA work and communication, and release all local DiT owners. It then loads its
VAE without a world barrier. Independent CPU/Gloo broker threads queue completed
request latents; a replica advertises readiness **after loading**, then takes the
next available request. Slow-loading replicas do not reserve work. Replies are
restored to original input order on rank 0. Dedicated completion links include
every worker, including non-decoder ranks and replicas with no requests. Early
finishers exchange heartbeats until all local work ends; the final notification
carries the agreed errors without an early world collective. Dedicated
heartbeats also keep decoder links alive during loading, idle waits and long decoding.
Model loading and decoding stay on each rank's main GPU-owning thread.

Timeline events include `dit_local_released`, `decoder_load_start`,
`decoder_load_end`, `decoder_replica_ready`, `decode_start` and `decode_end`.
With decoder overlap enabled, ingress seals forward dispatch after the request
source closes, pending admissions and prepared queues empty, and all outstanding
steps are known to be final. FIFO STOP then lets suffix ranks drain and retire
in order while ingress still receives and commits final results. An early rank
can load its decoder and process completed requests while later DiT ranks compute.
Available overlap depends on the remaining tail work and decoder loading time;
a temporarily empty queue never authorizes retirement of a recurrent stage.
The `dit_forward_sealed` event records this decision in the combined overlap path.
Native multi-pass refinement/upscaling/continuation still requires the phased
baseline and rejects `overlap_dit_decoder` explicitly.

The phased engine accepts a local native `FL2VA` or `Ref2VA` partition directory
and one DiT ingress. It supports the original dense model with Euler sampling.
FL2VA accepts first-only `[0]`, last-only `[-1]` and first/last `[0, -1]` images.
Ref2VA accepts the native image/video/audio references; video grids and timestamps
are retained through Qwen PP. Native latent-mask editing, timeline guides and
`audio_mode="lock_source"` use the media VAE phase. `--cpu-offload` is unnecessary
in phased mode: encoders, DiT and decoders have separate lifetimes. CPU RAM must hold the finite job's conditions, latents and outputs.
Each phase's weights, activations and communication buffers still must fit on
its GPUs; this example is not a 4090 throughput or memory guarantee.

Advanced requests use a separate native request execution lane:

- `long_video_mode="continuation"` uses native Ref2VA windows, AV overlap trimming,
  global temporal positions, and optional `continuation_prompts` (one per window).
- `latent_refine=0.4` performs a second native denoise pass with its own schedule.
- `latent_upscale={"scale": 2, "align": 32}` uses a learned upscaler. Supply its
  checkpoint through `create_phased_engine(..., additional_config={
  "latent_upscaler_path": "/path/to/upscaler.safetensors"})`. The ingress DiT is
  offloaded before upscaling and reloaded on the next DiT call. High-resolution
  FL2VA keyframe conditions are VAE-encoded before Qwen/DiT loading.

Each transformer call in this lane traverses the partitioned DiT. Requests are
processed serially and use exact forwards without MonoWire residual-cache reuse;
this lane currently retains Qwen/DiT phase barriers even if
`overlap_encoder_dit=True`. Ordinary requests keep continuous batching and the
optional drain overlap. Native incompatibilities still apply, including
continuation with latent-mask editing or refinement. `quality="high"`, LoRA,
non-Euler sampling and multiple outputs per request remain outside this contract
and are rejected before encoding. Use separate requests for multiple seeds.

For complete native media inputs, the CLI accepts `--requests-json requests.json`
instead of `--prompt`/`--image`, plus `--extra-args` and `--additional-config`
JSON objects. Media file paths inside JSON are resolved relative to the process
working directory, just as in the Python API. For example, a Ref2VA job:

```json
[{"prompt": "Keep the character consistent while the camera moves.",
  "multi_modal_data": {"image": ["reference.png"]}}]
```

```bash
torchrun --standalone --nproc-per-node=4 -m monowire.cli \
  --phased --pp 4 --model /path/to/Ref2VA --requests-json requests.json \
  --extra-args '{"task":"ref2va","long_video":true,"long_video_mode":"continuation"}' \
  --frames 430 --steps 20 --height 384 --width 672 --output outputs/long-video.pt
```

The mode adapters do not imply qualification on every GPU/checkpoint. Full-size
Ref2VA, learned upscaling and their combinations need corresponding model assets
and GPU validation; a missing checkpoint is not replaced with another partition.

For Python, use `create_phased_engine(...)` and `engine.generate(prompts, sampling)`
on every rank. Rank 0 returns CPU outputs with the original request IDs; other
ranks return an empty list. `engine.phase_events` reports phase time (including
loading), peak allocated GPU memory and remaining allocation after release.
The API accepts finite jobs, not an unbounded stream. The default retains phase
barriers for comparison.

Add `--overlap-encoder-dit` (Python: `overlap_encoder_dit=True`) to enable rolling
Qwen → DiT handoff. This mode encodes image/audio conditions first, prepares each
rank's DiT weights in CPU RAM and creates communication groups before Qwen runs.
The Qwen tail returns each completed request immediately over a dedicated Gloo
group. Each rank releases its Qwen weights after its final activation handoff,
copies its own DiT weights to GPU, and begins serving without a global drain or
weight-ready barrier. Rank 0 consumes available conditions through the existing
continuous scheduler; downstream stages accept activations once locally ready.
VAE decoding still waits for DiT to finish.

This mode additionally needs CPU RAM for all partitioned DiT weights (about
62 GiB total for the native checkpoint), alongside Qwen loading and job tensors.
The CPU inbox can retain the finite job's entire text conditioning; GPU handoffs
remain bounded. There is no disk spill or host-memory admission budget. Weight
copies on different GPUs may run concurrently; overlap is not a guarantee of
speedup. On multi-socket hosts, benchmark GPU-local NUMA CPU placement before
attributing weight-copy stalls to pipeline scheduling. Bind each worker before
allocating host weights, using only CPUs allowed by the job scheduler. Resolve
the actual GPU UUID/PCI device rather than assuming CUDA-visible indices match
physical GPU indices. Compare overlap and barrier runs with the same affinity.

`engine.timeline_events` records local Qwen release, per-request output
arrival, DiT copy start/completion and scheduler events using the host monotonic
clock (direct cross-rank comparison requires the same host). `dit_prefix_ready`
only marks prepared input entering the queue; `dit_dispatched` follows local
stage submission and can establish that rank 0 launched work before all weights
were ready. These host events do not measure GPU kernel overlap; use CUDA events
or a profiler for that. In overlap mode,
the `dit` phase excludes CPU preload/GPU copying and its end allocation precedes
DiT release; `dit_weights_released` records the allocation after release. Measure the
whole `generate()` call for end-to-end comparisons.


### Configurable queue policies

Count, ring, and pool are parallel configuration options for asynchronous queues.
Count remains the default for compatibility and serves as a performance control.
Select a policy when constructing a new engine:

```python
from monowire import ContinuousConfig

config = ContinuousConfig(
    max_active_requests=32,
    microbatch_size=1,
    downstream_batch_size=1,
    queue_policy="count",  # "count", "ring", or "pool"
)
```

- `count`: original bounded queues controlled by `PipelineTopology.buffer_size`
  (default 2), with normal tensor allocation and no new admission handshake,
  reusable receive arena, or feedback clone. Keep `memory_limits=None`.
- `ring` / `pool`: experimental byte-capacity receive buffers described below.
  Without explicit `memory_limits`, these create `DeviceMemoryLimits` with a
  3 GiB workspace margin and its other defaults. Supply matching
  `DeviceMemoryLimits(..., buffer_strategy="ring" or "pool")` to customize them.

All three keep asynchronous transport and Active as separate settings. Policy
changes require draining and recreating the engine; live buffer ownership cannot
be switched in place. Conflicting policy and memory-limit settings raise an error.
The default `queue_policy="auto"` preserves existing callers: no limits selects
`count`, `DeviceMemoryLimits` selects its buffer strategy, and `MemoryLimits`
selects the older `bytes` policy. `config.resolved_queue_policy` reports the actual
selection. Paired 8-GPU RTX 4090 tests with 8 requests and 2 denoising steps
(384p, 243-frame first-frame I2V, BS1) produced identical output hashes and
completed in 46.12 s for count, 46.29 s for pool, and 46.26 s for ring. This
validates the short-run regression fix, not sustained-throughput acceptance.
Longer paired measurements remain pending; separate initial pipeline propagation,
sustained throughput, and drain time when comparing policies.

For ring/pool, `PipelineTopology(..., receive_prefetch=2)` independently bounds
suffix receive lookahead per upstream peer. A credit covers a posted receive
until the compute owner consumes its packet (all items for a split packet).
This also bounds unconsumed received packets; it is not a total resident-task
limit. Consuming a packet returns the prefetch credit while its storage remains
charged until the final GPU consumer completes. Active and ring/pool byte
capacities are unchanged. Count continues to use `buffer_size`; the older
`bytes` policy retains its existing behavior. The default prefetch depth of 2
addresses the measured cold-start regression from deep queued NCCL receives;
it is configurable, not a claim of optimal depth for every workload.

Receive storage is allocated on the receiving CUDA stream before its first
write. Slot recycling still waits for the final GPU consumer, including when
packing, computation, or result handling aborts. Failed admission acknowledgements
release slots that have not entered tensor transfer. Send failure and queue
publication are serialized so a failed sender cannot accept orphaned work.

### Shared dynamic GPU admission (experimental)

Use `DeviceMemoryLimits` to keep `Active` independent of GPU queue capacity:

```python
from monowire import ContinuousConfig, DeviceMemoryLimits

config = ContinuousConfig(
    max_active_requests=32,
    microbatch_size=1,
    downstream_batch_size=1,
    memory_limits=DeviceMemoryLimits(workspace_bytes=3 * 1024**3),
)
```

Each GPU has one CPU-side ledger initialized after weights and any known external
allocations. Admission updates CPU counters; it does not query the driver or scan
live tensor storage. Request reservations cover their whole lifetime. One compute
stream shares one workspace margin. Active remains an independent upper bound.

Select either `DeviceMemoryLimits(..., buffer_strategy="ring")` (default) or
`buffer_strategy="pool"`. Both use the same byte-budget interface:

- `ring`: one contiguous variable-length receive arena; tensors never straddle
  its end. Reclamation follows FIFO completion, with wrap padding recorded.
- `pool`: power-of-two size-class receive buffers, independently reclaimed;
  idle incompatible classes can be replaced within the same byte capacity.

`buffer_bytes=None` derives ordinary receive capacity from the remaining ledger;
set a positive byte count to cap it for experiments. A receive arena reserves
three times its capacity: storage plus two retained packing/output copies. This
is a conservative H3 BS1 bound, not a general allocator guarantee. There is no
fixed task-count limit on receive occupancy. Both strategies retain ownership
through the final send event, not just dequeue or compute submission. Shared
condition declarations use separate dictionary storage and never alias recyclable
receive slots.

Half of `progress_bytes` is reserved exclusively for feedback staging; the other
half protects non-contiguous outgoing copies. These are independent cumulative
ledger budgets (`feedback` and `wire`), established before the first packet.
Multiple in-flight copies cannot borrow feedback capacity. Feedback is copied on the GPU into
request-owned storage before recycling its receive buffer. This additional copy
is explicit and must be included in performance comparisons. The wire keeps its
existing metadata/approval handshake; this change does not claim to implement
batched credit announcements or eliminate every CPU round trip.

This mode currently supports BS1, one ingress, pre-encoded/phased T2V and first-frame
I2V, with feature caching disabled. Workspace/progress/headroom must be measured
for each model and length range. Capacity is fixed for the runtime lifetime:
external allocations must precede runtime construction or be explicitly charged;
untracked external allocations are not automatically detected by hot admission.
`MemoryLimits` remains available for older fixed-partition experiments.
`runtime.memory_pool.snapshot()` reports buffer reuse, occupancy, wait time,
ring wrap padding and head-of-line waits, and pool size-class rounding. Persistent
pool leases remain until channel shutdown; live consumers must be zero after drain.
Ring slot starts also satisfy each packet's dtype alignment when a smaller
`buffer_alignment` is configured. Failed view construction releases the acquired
slot. During shutdown, queued packets retire while receivers drain, including
racing arrivals; storage still waits for its CUDA last-use event before reuse.
Waiting for completion does not hold the shared capacity ledger lock.
