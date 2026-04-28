#pragma once

#include "monowire-mmap.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
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
    ggml_tensor *tensor = nullptr;
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
    std::vector<std::vector<monowire_streaming_tensor>> dynamic_tensors_by_layer;
};

// Build the per-layer streaming plan by subtracting preserved groups from all
// eligible tensors.
monowire_streaming_runtime_plan
monowire_streaming_build_runtime_plan(const std::vector<monowire_streaming_tensor> &tensors,
                                      const monowire_streaming_plan &preserve_plan, int n_layer,
                                      size_t prefetch_window);

struct monowire_streaming_runtime_stats {
    uint64_t graph_begins = 0;
    uint64_t boundary_callbacks = 0;
    uint64_t queued_layers = 0;
    uint64_t prefetched_layers = 0;
    uint64_t evicted_layers = 0;
    uint64_t queued_ranges = 0;
    uint64_t prefetched_ranges = 0;
    uint64_t read_prefetched_ranges = 0;
    uint64_t hint_prefetched_ranges = 0;
    uint64_t direct_prefetched_tensors = 0;
    uint64_t direct_bound_layers = 0;
    uint64_t evicted_ranges = 0;
    uint64_t stale_tasks = 0;
    uint64_t prefetched_bytes = 0;
    uint64_t evicted_bytes = 0;
};

// Owns residency state while a graph is evaluated. The callbacks are small and
// deterministic because the expensive grouping work has already been planned.
class monowire_streaming_runtime_state {
public:
    monowire_streaming_runtime_state(monowire_streaming_runtime_plan plan, const monowire_mmaps &mappings,
                                     bool evict_enabled, uint32_t io_threads = 1, bool read_prefetch = false,
                                     bool direct_buffer = false);
    ~monowire_streaming_runtime_state();

    void begin_graph();
    bool on_eval(ggml_tensor *t, bool ask);
    monowire_streaming_runtime_stats stats() const;

private:
    struct prefetch_task {
        int layer = -1;
        uint64_t epoch = 0;
        monowire_streaming_range range = {};
    };

    struct runtime_counters {
        std::atomic<uint64_t> graph_begins{0};
        std::atomic<uint64_t> boundary_callbacks{0};
        std::atomic<uint64_t> queued_layers{0};
        std::atomic<uint64_t> prefetched_layers{0};
        std::atomic<uint64_t> evicted_layers{0};
        std::atomic<uint64_t> queued_ranges{0};
        std::atomic<uint64_t> prefetched_ranges{0};
        std::atomic<uint64_t> read_prefetched_ranges{0};
        std::atomic<uint64_t> hint_prefetched_ranges{0};
        std::atomic<uint64_t> direct_prefetched_tensors{0};
        std::atomic<uint64_t> direct_bound_layers{0};
        std::atomic<uint64_t> evicted_ranges{0};
        std::atomic<uint64_t> stale_tasks{0};
        std::atomic<uint64_t> prefetched_bytes{0};
        std::atomic<uint64_t> evicted_bytes{0};
    };

    struct aligned_buffer_deleter {
        void operator()(uint8_t *ptr) const;
    };

    using aligned_buffer_ptr = std::unique_ptr<uint8_t, aligned_buffer_deleter>;

    struct direct_tensor_buffer {
        ggml_tensor *tensor = nullptr;
        void *original_data = nullptr;
        int file_idx = -1;
        size_t offset = 0;
        size_t size = 0;
        aligned_buffer_ptr data;
    };

    void prefetch_layer(int layer, uint64_t expected_epoch = UINT64_MAX);
    bool prefetch_layer_direct(int layer, uint64_t expected_epoch);
    void bind_layer_direct(int layer);
    void release_layer_direct(int layer);
    bool wait_until_layer_ready(int layer);
    aligned_buffer_ptr allocate_direct_buffer(size_t size) const;
    size_t prefetch_range(const monowire_streaming_range &range);
    void enqueue_prefetch_layer(int layer);
    void prefetch_worker_loop();
    void evict_layer(int layer);
    bool is_boundary_tensor(const ggml_tensor *t, int &layer) const;
    bool is_window_boundary(int layer) const;
    void print_trace_stats() const;

    monowire_streaming_runtime_plan plan;
    const monowire_mmaps &mappings;
    bool evict_enabled = true;
    bool read_prefetch = false;
    bool direct_buffer = false;
    bool trace_enabled = false;
    runtime_counters counters;
    std::vector<bool> resident_layers;
    std::vector<bool> bound_layers;
    std::vector<bool> queued_layers;
    std::vector<uint64_t> layer_epochs;
    std::vector<size_t> pending_ranges;
    std::vector<std::vector<direct_tensor_buffer>> direct_tensors_by_layer;
    std::deque<prefetch_task> prefetch_queue;
    std::mutex prefetch_mutex;
    std::condition_variable prefetch_cv;
    std::vector<std::thread> prefetch_workers;
    bool stop_prefetch_worker = false;
};
