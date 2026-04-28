#pragma once

#include "monowire-mmap.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;

// Tensor metadata projected from GGUF so the planner can reason about mmap
// ranges without touching the tensor payload.
struct monowire_streaming_tensor {
    int layer;
    int file_idx;
    size_t offset;
    size_t size;
    std::string name;
};

// Half-open byte range [first, last) inside one mmap-backed model shard.
struct monowire_streaming_range {
    int file_idx;
    size_t first;
    size_t last;
};

// Static preservation plan built during model loading. The selected ranges are
// prefetched or mlocked once and remain resident for the model lifetime.
struct monowire_streaming_plan {
    size_t budget_bytes = 0;
    size_t candidate_bytes = 0;
    size_t reserved_bytes = 0;
    std::vector<std::string> selected_groups;
    std::vector<monowire_streaming_range> ranges;
};

// Select weight groups to keep resident under the requested memory budget.
monowire_streaming_plan monowire_streaming_build_plan(const std::vector<monowire_streaming_tensor> &tensors,
                                                      size_t budget_bytes);

// Runtime plan for weights that were not preserved. Ranges are grouped by layer
// so graph callbacks can prefetch upcoming windows and evict old windows.
struct monowire_streaming_runtime_plan {
    size_t prefetch_window = 0;
    size_t dynamic_bytes = 0;
    int first_layer = -1;
    int last_layer = -1;
    std::vector<std::vector<monowire_streaming_range>> dynamic_ranges_by_layer;
};

// Build the per-layer streaming plan by subtracting preserved groups from all
// eligible tensors.
monowire_streaming_runtime_plan
monowire_streaming_build_runtime_plan(const std::vector<monowire_streaming_tensor> &tensors,
                                      const monowire_streaming_plan &preserve_plan, int n_layer,
                                      size_t prefetch_window);

// Owns residency state while a graph is evaluated. The callbacks are small and
// deterministic because the expensive grouping work has already been planned.
class monowire_streaming_runtime_state {
public:
    monowire_streaming_runtime_state(monowire_streaming_runtime_plan plan, const monowire_mmaps &mappings,
                                     bool evict_enabled);

    void begin_graph();
    bool on_eval(ggml_tensor *t, bool ask);

private:
    void prefetch_layer(int layer);
    void evict_layer(int layer);
    bool is_boundary_tensor(const ggml_tensor *t, int &layer) const;
    bool is_window_boundary(int layer) const;

    monowire_streaming_runtime_plan plan;
    const monowire_mmaps &mappings;
    bool evict_enabled = true;
    std::vector<bool> resident_layers;
};
