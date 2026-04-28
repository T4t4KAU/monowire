# Monowire

`monowire` is a high-performance local inference system derived from an upstream GGUF runtime.

It is designed for resource-constrained deployment and keeps the upstream hot path intact: core inference, quantization, MoE support, and major hardware backends are preserved, while training-related code, example clutter, conversion utilities, web/server extras, and other non-essential components are removed to reduce maintenance overhead without giving up performance.

## Supported Platforms and Backends

The project keeps support for the following platforms and acceleration paths:

- Intel
  - CPU / OpenMP
  - BLAS
  - SYCL
  - OpenVINO
- AMD
  - CPU / BLAS
  - HIP / ROCm
  - Vulkan
- Apple Silicon
  - CPU / Accelerate / BLAS
  - Metal
- NVIDIA
  - CUDA
- Ascend NPU
  - CANN
- MUSA
  - MUSA backend

Common CMake switches follow upstream `ggml` conventions:

- `-DGGML_CUDA=ON`
- `-DGGML_HIP=ON`
- `-DGGML_METAL=ON`
- `-DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL`
- `-DGGML_OPENVINO=ON`
- `-DGGML_VULKAN=ON`
- `-DGGML_CANN=ON -DCANN_INSTALL_DIR=/path/to/ascend-toolkit`
- `-DGGML_MUSA=ON`

## Build

The default build produces:

- `monowire-cli`
- `monowire-bench`
- `monowire-quantize`

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

If you want to disable a specific tool:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DMONOWIRE_BUILD_CLI=ON \
  -DMONOWIRE_BUILD_BENCH=ON \
  -DMONOWIRE_BUILD_QUANTIZE=OFF
cmake --build build -j
```

## Testing and Regression

The project includes a `CTest`-based regression suite that covers the core functionality kept in this repository, including:

- GGUF parsing and model metadata handling
- tokenizer and vocabulary compatibility
- grammar, chat template, and JSON schema behavior
- sampling and other inference-side logic
- backend ops and shared multi-backend execution paths
- model-backed integration paths such as model loading, state restore, and thread safety

Recommended regression tiers:

- Fast offline regression: `check-fast`
- Full offline regression: `check`
- Model-backed integration regression: `check-model`
- Everything enabled in the build: `check-all`

Using CMake targets directly:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMONOWIRE_BUILD_TESTS=ON
cmake --build build -j

# Fast offline regression
cmake --build build --target check-fast

# Full offline regression, excluding model-backed tests
cmake --build build --target check
```

To include model-backed integration tests, provide a GGUF model path during configuration:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DMONOWIRE_BUILD_TESTS=ON \
  -DMONOWIRE_TEST_MODELFILE=/path/to/model.gguf
cmake --build build -j
cmake --build build --target check-model
```

You can also use the repository script for a single, consistent entry point:

```bash
# Fast offline regression
./scripts/run-tests.sh

# Full offline regression
./scripts/run-tests.sh full

# Model-backed integration regression
./scripts/run-tests.sh model /path/to/model.gguf

# Run every test available in the current build
./scripts/run-tests.sh all /path/to/model.gguf
```

For day-to-day development, running `check-fast` is a good baseline. If you touch model loading, KV cache logic, sampling, chat templates, backend execution, or threading behavior, also run `check` or `check-model`.

## Platform-Specific Build Examples

NVIDIA CUDA:

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build-cuda -j
```

AMD ROCm:

```bash
cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON
cmake --build build-hip -j
```

AMD / Intel Vulkan:

```bash
cmake -S . -B build-vulkan -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build-vulkan -j
```

Apple Silicon:

```bash
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_BLAS=ON
cmake --build build-metal -j
```

Intel SYCL:

```bash
cmake -S . -B build-sycl -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL
cmake --build build-sycl -j
```

Intel OpenVINO:

```bash
cmake -S . -B build-openvino -DCMAKE_BUILD_TYPE=Release -DGGML_OPENVINO=ON
cmake --build build-openvino -j
```

Ascend NPU:

```bash
cmake -S . -B build-cann \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CANN=ON \
  -DCANN_INSTALL_DIR=/path/to/ascend-toolkit
cmake --build build-cann -j
```

MUSA:

```bash
cmake -S . -B build-musa -DCMAKE_BUILD_TYPE=Release -DGGML_MUSA=ON
cmake --build build-musa -j
```

## Usage Examples

### One-Shot Text Generation

This mode uses `monowire-cli` as a single-pass generator. It does not enter chat mode and does not rely on multi-turn conversation state.

It is a good fit for:

- continuation
- summarization
- single-pass rewriting
- batch jobs with fixed prompts

Use `-no-cnv` explicitly to avoid automatically switching into chat mode through a model-provided chat template.

F16:

```bash
./build-cuda/bin/monowire-cli \
  -m /path/to/Qwen3-8B-f16.gguf \
  -p "Explain CPU and GPU hybrid inference in one paragraph." \
  -n 128 \
  -t 20 \
  -ngl 0 \
  -no-cnv
```

Q8_0:

```bash
./build-cuda/bin/monowire-cli \
  -m /path/to/Qwen3-8B-Q8_0.gguf \
  -p "Give me a short summary of MoE models." \
  -n 128 \
  -t 20 \
  -ngl 0 \
  -no-cnv
```

### Multi-Turn Chat

This mode uses `monowire-cli` as a chat interface. It enables conversation mode and keeps history across turns.

Recommended flags:

- `-cnv`: explicitly enable conversation mode
- `-i`: enter interactive multi-turn chat
- `-sys`: provide a system prompt
- `-mli`: enable multi-line input when needed

Example:

```bash
./build-cuda/bin/monowire-cli \
  -m /path/to/Qwen3-8B-f16.gguf \
  -cnv \
  -i \
  -sys "You are a concise and technically accurate assistant." \
  -t 20 \
  -ngl 0
```

After startup, you can keep entering follow-up messages and `monowire-cli` will preserve the conversation context.

If you want a single question-answer exchange while still going through the chat template, use single-turn chat mode instead of raw generation:

```bash
./build-cuda/bin/monowire-cli \
  -m /path/to/Qwen3-8B-Q8_0.gguf \
  -cnv \
  -st \
  -sys "You are a helpful assistant." \
  -p "Explain what an MoE model is in three sentences." \
  -n 256 \
  -t 20 \
  -ngl 0
```

In short:

- one-shot generation: `-p ... -no-cnv`
- single-turn chat: `-cnv -st -p ...`
- multi-turn chat: `-cnv -i`

### GPU / Hybrid Inference

If a GPU is available, you can enable common hybrid inference paths with flags such as:

- `-ngl <n>`: offload the first `n` layers to the GPU
- `-fa`: enable Flash Attention
- `-sm layer|row|tensor`: choose the split mode
- `-ts a/b/...`: configure multi-GPU tensor split
- `-mg <i>`: choose the main GPU
- `-nkvo 1`: disable KV offload for comparison runs

Example:

```bash
./build-cuda/bin/monowire-cli \
  -m /path/to/model.gguf \
  -n 128 \
  -t 20 \
  -ngl 99 \
  -fa \
  -sm layer
```

### Memory-Aware Streaming

Monowire includes a memory-aware streaming path for constrained devices. It selects important mmap-backed tensor groups under a RAM budget, keeps those ranges resident with `mlock` when supported, and can optionally stream the remaining layer weights with a prefetch window.

Recommended starting point for memory pressure:

```bash
./build-cuda/bin/monowire-cli \
  -m /path/to/Qwen3-8B-Q8_0.gguf \
  -p "Explain CPU and GPU hybrid inference in one paragraph." \
  -n 128 \
  -t 20 \
  -ngl 0 \
  -no-cnv \
  --streaming-budget-mib 3072 \
  --streaming-window 0
```

Useful flags:

- `--streaming-budget-mib N`: preserve up to `N` MiB of selected mmap-backed model weights in RAM.
- `--streaming-window N`: enable runtime layer-window streaming when `N > 0`; use `0` for preserve-only mode.
- `--no-streaming-prefetch`: disable preserved-range prefetching while keeping the selection path available.
- `--no-streaming-evict`: keep streamed ranges resident instead of evicting them after use.

The preserve-only mode is the safest default for CPU-only systems under memory pressure. Runtime streaming can help memory residency experiments, but it adds scheduler callback overhead and should be benchmarked on the target platform before enabling.

### Benchmark

```bash
./build-cuda/bin/monowire-bench \
  -m /path/to/Qwen3-8B-Q8_0.gguf \
  -p 512 \
  -n 64 \
  -r 3 \
  -t 20 \
  -ngl 0 \
  -fa 1
```

Benchmark the streaming path with the same tool:

```bash
./build-cuda/bin/monowire-bench \
  -m /path/to/Qwen3-8B-Q8_0.gguf \
  -p 32 \
  -n 64 \
  -b 32 \
  -ub 32 \
  -r 3 \
  -t 20 \
  -ngl 0 \
  --streaming-budget-mib 3072 \
  --streaming-window 0
```

Observed CPU-only Q8_0 behavior on the development machine:

- No memory pressure: the upstream baseline and Monowire are effectively tied, with Monowire within about `0.5%` of upstream token generation throughput.
- No memory pressure with runtime streaming: `--streaming-window > 0` is slower because scheduler callbacks dominate.
- Under synthetic memory pressure: Monowire with `--streaming-budget-mib 3072 --streaming-window 0` improved token generation from about `0.796 tok/s` upstream to about `0.907 tok/s`, roughly `+14%`.

### Quantization

```bash
./build-cuda/bin/monowire-quantize \
  /path/to/model-f16.gguf \
  /path/to/model-q8_0.gguf \
  Q8_0
```
