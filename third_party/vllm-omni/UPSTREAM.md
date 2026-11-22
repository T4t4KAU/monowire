# Bundled vLLM-Omni source

- Upstream: https://github.com/vllm-project/vllm-omni
- Revision: `5917730a4b02aedb44d374173cdc1b10484ccd52`
- MonoWire source version: `0.32.0+monowire.g5917730a`
- Required vLLM: `0.32.0`
- License: Apache-2.0; see [LICENSE](LICENSE). Embedded components retain their own notices.

This directory is ordinary source tracked by the MonoWire repository. It is not
a submodule and does not require a separate checkout or patch application.
The source version identifies this modified v0.32.0 source snapshot. Future changes are made directly to these files.

## MonoWire modifications

- `vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py`:
  exposes block construction, block execution, and a resumable boundary between
  input preparation and output heads for pipeline-parallel inference. Block-only
  workers can omit input/output modules at construction; Diffusers RoPE
  reconstruction follows the modules actually owned by the worker.
- `vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py`:
  scopes encoder/media collectives through an instance method. Native H3 keeps
  its distributed behavior; MonoWire ingress instances use local codec I/O
  without replacing process-global functions.
- `vllm_omni/diffusion/cache/exact_projection_cache.py` and
  `vllm_omni/diffusion/models/minimax_h3/adaln_cache.py`: allow immutable
  embeddings to carry their existing content keys through pipeline transport
  and ordered batching, avoiding repeated GPU-to-CPU hashing. Projection weight,
  numerical-setting, and sidecar validation remain in place.
- `minimax_h3/denoise_loop.py` and `minimax_h3/adaln_cache.py`: construct
  unedited timestep plans from CPU scalars and reuse exact time embeddings under
  input, weight, evaluation-mode and numerical-setting guards. New plans still
  hash the computed embedding; edited plans keep the general tensor path.
- `setup.py`, `pyproject.toml`, `.gitignore`, `vllm_omni/_version.py`, and
  `vllm_omni/version.py`: use a fixed source version instead of deriving it from
  a nested Git repository. Runtime dependency selection remains upstream's.

These modifications retain Apache-2.0 licensing. The containing MonoWire
framework is MIT-licensed; that license does not replace third-party licenses.

Install from this directory using the containing repository's README. The upstream repository at the pinned revision contains the full development
checkout and documentation.

## Runtime snapshot boundaries

The complete `vllm_omni` package, installation requirements, and embedded
license notices are retained. Upstream CI, agent configuration, applications,
documentation, examples, recipes, benchmark suites, development tools, and test
suites are omitted. The reference audio at
`tests/assets/qwen3_tts/clone_2.wav` is retained because an upstream stage
processor resolves it at runtime.

Development-only dependency extras and test-tool configuration are removed
from `pyproject.toml`; runtime dependencies and entry points are retained.
The bundled README is replaced with installation and provenance links.

Machine-specific example output paths are replaced with relative paths in
`vllm_omni/quantization/tools/compare_diffusion_trajectory_similarity.py`, and
the checkpoint example in `vllm_omni/deploy/causvid_wan.yaml` uses a placeholder.
`vllm_omni/model_executor/stage_input_processors/aura_omni.py` no longer probes
hardcoded workspace model directories; explicit configuration, environment
variables, the user cache, and the upstream model ID remain available.
