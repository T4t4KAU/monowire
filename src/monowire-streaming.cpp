#include "monowire-streaming.h"

#include "ggml.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <map>
#if defined(_WIN32)
#include <malloc.h>
#endif
#include <regex>
#include <system_error>
#include <tuple>
#include <unordered_set>

namespace {

enum class monowire_streaming_category {
    attention,
    ffn,
    other,
};

struct monowire_streaming_group {
    std::string key;
    monowire_streaming_category category = monowire_streaming_category::other;
    size_t total_size = 0;
    std::vector<monowire_streaming_tensor> tensors;
};

// GGUF tensor names encode the layer number in a few architecture-specific
// prefixes. Normalizing this early lets the planner handle encoder, decoder,
// and decoder-only models with the same grouping logic.
bool parse_layer_prefix(const std::string &name, int &layer, size_t &prefix_len) {
    if (sscanf(name.c_str(), "blk.%d.", &layer) == 1) {
        prefix_len = name.find('.', 4);
        if (prefix_len == std::string::npos) {
            return false;
        }
        ++prefix_len;
        return true;
    }

    if (sscanf(name.c_str(), "dec.blk.%d.", &layer) == 1) {
        prefix_len = name.find('.', 8);
        if (prefix_len == std::string::npos) {
            return false;
        }
        ++prefix_len;
        return true;
    }

    if (sscanf(name.c_str(), "enc.blk.%d.", &layer) == 1) {
        prefix_len = name.find('.', 8);
        if (prefix_len == std::string::npos) {
            return false;
        }
        ++prefix_len;
        return true;
    }

    return false;
}

bool ends_with(const std::string &value, const std::string &suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string normalize_group_name(const std::string &name) {
    int layer = -1;
    size_t prefix_len = 0;
    if (!parse_layer_prefix(name, layer, prefix_len)) {
        return {};
    }

    std::string stem = name.substr(prefix_len);
    if (ends_with(stem, ".weight")) {
        stem.resize(stem.size() - 7);
    } else if (ends_with(stem, ".bias")) {
        stem.resize(stem.size() - 5);
    } else {
        return {};
    }

    // Collapse MoE expert ids so all experts of the same weight family compete
    // as one budget item instead of fragmenting the plan by expert number.
    static const std::regex expert_suffix("\\.\\d+$");
    stem = std::regex_replace(stem, expert_suffix, ".expert");

    return stem;
}

// Only attention/FFN families are selected. Layer norms and small metadata-like
// tensors are cheap enough that locking them does not move the memory needle.
monowire_streaming_category classify_group(const std::string &group) {
    if (group.rfind("ffn_", 0) == 0 || group.find(".ffn_") != std::string::npos || group.rfind("vis_down", 0) == 0
        || group.rfind("vis_up", 0) == 0 || group.rfind("vis_gate", 0) == 0) {
        return monowire_streaming_category::ffn;
    }

    if (group.rfind("attn_", 0) == 0 || group.find("attn") != std::string::npos || group.rfind("ssm_", 0) == 0
        || group.rfind("shortconv.", 0) == 0) {
        return monowire_streaming_category::attention;
    }

    return monowire_streaming_category::other;
}

int attention_priority(const std::string &key) {
    if (key.find("attn_k") != std::string::npos || key.find("attn_v") != std::string::npos) {
        return 0;
    }
    if (key.find("attn_q") != std::string::npos || key.find("attn_qkv") != std::string::npos) {
        return 1;
    }
    if (key.find("attn_output") != std::string::npos) {
        return 2;
    }
    return 3;
}

std::vector<monowire_streaming_range> merge_ranges(const std::vector<monowire_streaming_range> &ranges) {
    std::vector<monowire_streaming_range> merged = ranges;

    std::sort(merged.begin(), merged.end(), [](const auto &lhs, const auto &rhs) {
        return std::tie(lhs.file_idx, lhs.first, lhs.last) < std::tie(rhs.file_idx, rhs.first, rhs.last);
    });

    std::vector<monowire_streaming_range> result;
    for (const auto &range : merged) {
        if (result.empty() || result.back().file_idx != range.file_idx || result.back().last < range.first) {
            result.push_back(range);
            continue;
        }

        result.back().last = std::max(result.back().last, range.last);
    }

    return result;
}

// The graph builder names layer outputs as l_out-N. These tensors form stable
// boundaries where the runtime can advance its streaming window safely.
bool parse_boundary_layer_name(const char *name, int &layer) {
    return name != nullptr && sscanf(name, "l_out-%d", &layer) == 1;
}

bool env_flag_enabled(const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
}

} // namespace

monowire_streaming_plan monowire_streaming_build_plan(const std::vector<monowire_streaming_tensor> &tensors,
                                                      size_t budget_bytes) {
    monowire_streaming_plan plan;
    plan.budget_bytes = budget_bytes;

    if (budget_bytes == 0 || tensors.empty()) {
        return plan;
    }

    // Group by weight family across layers. Preserving an entire family avoids
    // mixing resident and streamed tensors in the same hot operation.
    std::map<std::string, monowire_streaming_group> groups;
    for (const auto &tensor : tensors) {
        const std::string group_name = normalize_group_name(tensor.name);
        if (group_name.empty()) {
            continue;
        }

        auto &group = groups[group_name];
        group.key = group_name;
        group.category = classify_group(group_name);
        if (group.category == monowire_streaming_category::other) {
            continue;
        }

        group.total_size += tensor.size;
        group.tensors.push_back(tensor);
        plan.candidate_bytes += tensor.size;
    }

    std::vector<const monowire_streaming_group *> ffn_groups;
    std::vector<const monowire_streaming_group *> attn_groups;

    for (const auto &kv : groups) {
        const auto &group = kv.second;
        if (group.category == monowire_streaming_category::other || group.tensors.empty()) {
            continue;
        }

        if (group.category == monowire_streaming_category::ffn) {
            ffn_groups.push_back(&group);
        } else {
            attn_groups.push_back(&group);
        }
    }

    // FFN tensors are usually the largest recurrent memory pressure source, so
    // try them first and spend the remaining budget on attention groups.
    std::sort(ffn_groups.begin(), ffn_groups.end(), [](const auto *lhs, const auto *rhs) {
        if (lhs->total_size != rhs->total_size) {
            return lhs->total_size > rhs->total_size;
        }
        return lhs->key < rhs->key;
    });

    // Attention groups use a fixed semantic priority first, then prefer smaller
    // groups to reduce page-in fan-out when the budget is tight.
    std::sort(attn_groups.begin(), attn_groups.end(), [](const auto *lhs, const auto *rhs) {
        const int lp = attention_priority(lhs->key);
        const int rp = attention_priority(rhs->key);
        if (lp != rp) {
            return lp < rp;
        }
        if (lhs->total_size != rhs->total_size) {
            return lhs->total_size < rhs->total_size;
        }
        return lhs->key < rhs->key;
    });

    size_t remaining = budget_bytes;
    auto select_group = [&](const monowire_streaming_group *group) {
        if (group->total_size > remaining) {
            return false;
        }

        remaining -= group->total_size;
        plan.reserved_bytes += group->total_size;
        plan.selected_groups.push_back(group->key);
        for (const auto &tensor : group->tensors) {
            plan.ranges.push_back({tensor.file_idx, tensor.offset, tensor.offset + tensor.size});
        }
        return true;
    };

    bool selected_any_ffn = false;
    for (const auto *group : ffn_groups) { selected_any_ffn = select_group(group) || selected_any_ffn; }

    if (!selected_any_ffn) {
        // When the budget cannot retain one FFN family across all layers,
        // prioritize smaller attention groups first to reduce I/O fan-out.
        for (const auto *group : attn_groups) { select_group(group); }
    } else {
        for (const auto *group : attn_groups) { select_group(group); }
    }

    // Merge adjacent shards before mlock/madvise so the OS sees fewer calls and
    // wider contiguous ranges.
    plan.ranges = merge_ranges(plan.ranges);
    return plan;
}

monowire_streaming_runtime_plan
monowire_streaming_build_runtime_plan(const std::vector<monowire_streaming_tensor> &tensors,
                                      const monowire_streaming_plan &preserve_plan, int n_layer,
                                      size_t prefetch_window) {
    monowire_streaming_runtime_plan runtime_plan;
    runtime_plan.prefetch_window = prefetch_window;

    if (prefetch_window == 0 || n_layer <= 0 || tensors.empty()) {
        return runtime_plan;
    }

    // The runtime only streams weight families that were not preserved by the
    // static plan, preventing duplicate residency work.
    std::unordered_set<std::string> preserved_groups(preserve_plan.selected_groups.begin(),
                                                     preserve_plan.selected_groups.end());

    runtime_plan.dynamic_ranges_by_layer.resize(n_layer);
    runtime_plan.dynamic_tensors_by_layer.resize(n_layer);

    for (const auto &tensor : tensors) {
        if (tensor.layer < 0 || tensor.layer >= n_layer) {
            continue;
        }

        const std::string group_name = normalize_group_name(tensor.name);
        if (group_name.empty()) {
            continue;
        }

        if (classify_group(group_name) == monowire_streaming_category::other) {
            continue;
        }

        if (preserved_groups.find(group_name) != preserved_groups.end()) {
            continue;
        }

        runtime_plan.dynamic_bytes += tensor.size;
        runtime_plan.dynamic_ranges_by_layer[tensor.layer].push_back({
            tensor.file_idx,
            tensor.offset,
            tensor.offset + tensor.size,
        });
        runtime_plan.dynamic_tensors_by_layer[tensor.layer].push_back(tensor);
    }

    for (int layer = 0; layer < n_layer; ++layer) {
        auto &ranges = runtime_plan.dynamic_ranges_by_layer[layer];
        ranges = merge_ranges(ranges);

        if (!ranges.empty()) {
            runtime_plan.first_layer = runtime_plan.first_layer == -1 ? layer : runtime_plan.first_layer;
            runtime_plan.last_layer = layer;
        }
    }

    return runtime_plan;
}

monowire_streaming_runtime_state::monowire_streaming_runtime_state(monowire_streaming_runtime_plan plan,
                                                                   const monowire_mmaps &mappings, bool evict_enabled,
                                                                   uint32_t io_threads, bool read_prefetch,
                                                                   bool direct_buffer)
    : plan(std::move(plan)), mappings(mappings), evict_enabled(evict_enabled), read_prefetch(read_prefetch),
      direct_buffer(direct_buffer), trace_enabled(env_flag_enabled("MONOWIRE_STREAMING_TRACE")),
      resident_layers(this->plan.dynamic_ranges_by_layer.size(), false),
      bound_layers(this->plan.dynamic_ranges_by_layer.size(), false),
      queued_layers(this->plan.dynamic_ranges_by_layer.size(), false),
      layer_epochs(this->plan.dynamic_ranges_by_layer.size(), 0),
      pending_ranges(this->plan.dynamic_ranges_by_layer.size(), 0) {
    if (this->direct_buffer) {
        bool has_direct_tensors = false;
        direct_tensors_by_layer.resize(this->plan.dynamic_tensors_by_layer.size());
        for (size_t layer = 0; layer < this->plan.dynamic_tensors_by_layer.size(); ++layer) {
            auto &dst = direct_tensors_by_layer[layer];
            for (const auto &tensor : this->plan.dynamic_tensors_by_layer[layer]) {
                if (tensor.tensor == nullptr || tensor.tensor->data == nullptr) {
                    continue;
                }
                dst.push_back({
                    /*.tensor       =*/tensor.tensor,
                    /*.original_data=*/tensor.tensor->data,
                    /*.file_idx     =*/tensor.file_idx,
                    /*.offset       =*/tensor.offset,
                    /*.size         =*/tensor.size,
                    /*.data         =*/nullptr,
                });
            }
            has_direct_tensors = has_direct_tensors || !dst.empty();
        }
        this->direct_buffer = has_direct_tensors;
    }

    const bool has_runtime_work
        = std::any_of(this->plan.dynamic_ranges_by_layer.begin(), this->plan.dynamic_ranges_by_layer.end(),
                      [](const auto &ranges) { return !ranges.empty(); });

    if (!has_runtime_work || mappings.empty() || io_threads == 0) {
        return;
    }

    prefetch_workers.reserve(io_threads);
    for (uint32_t ith = 0; ith < io_threads; ++ith) {
        try {
            // FlexInfer overlaps storage preparation with layer computation.
            // Multiple workers let independent tensor/file ranges issue I/O
            // hints concurrently, closer to the paper's multi-IO-thread design.
            prefetch_workers.emplace_back([this]() { prefetch_worker_loop(); });
        } catch (const std::system_error &err) {
            std::fprintf(stderr, "%s: failed to start streaming prefetch worker %u/%u: %s\n", __func__, ith + 1,
                         io_threads, err.what());
            break;
        }
    }
}

monowire_streaming_runtime_state::~monowire_streaming_runtime_state() {
    if (!prefetch_workers.empty()) {
        {
            std::lock_guard<std::mutex> lock(prefetch_mutex);
            stop_prefetch_worker = true;
            prefetch_queue.clear();
        }
        prefetch_cv.notify_all();
        for (auto &worker : prefetch_workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    if (direct_buffer) {
        for (int layer = 0; layer < (int)direct_tensors_by_layer.size(); ++layer) { release_layer_direct(layer); }
    }

    if (trace_enabled) {
        print_trace_stats();
    }
}

void monowire_streaming_runtime_state::aligned_buffer_deleter::operator()(uint8_t *ptr) const {
    if (ptr == nullptr) {
        return;
    }
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

monowire_streaming_runtime_state::aligned_buffer_ptr
monowire_streaming_runtime_state::allocate_direct_buffer(size_t size) const {
    if (size == 0) {
        return nullptr;
    }

    void *ptr = nullptr;
#if defined(_WIN32)
    ptr = _aligned_malloc(size, 32);
    if (ptr == nullptr) {
        return nullptr;
    }
#else
    if (posix_memalign(&ptr, 32, size) != 0) {
        return nullptr;
    }
#endif

    return aligned_buffer_ptr(static_cast<uint8_t *>(ptr));
}

void monowire_streaming_runtime_state::prefetch_layer(int layer, uint64_t expected_epoch) {
    if (layer < 0 || layer >= (int)plan.dynamic_ranges_by_layer.size()) {
        return;
    }

    if (direct_buffer) {
        prefetch_layer_direct(layer, expected_epoch);
        return;
    }

    uint64_t epoch = 0;
    std::vector<monowire_streaming_range> ranges;
    {
        std::lock_guard<std::mutex> lock(prefetch_mutex);
        if (expected_epoch != UINT64_MAX && layer_epochs[layer] != expected_epoch) {
            return;
        }

        if (resident_layers[layer]) {
            return;
        }

        epoch = layer_epochs[layer];
        ranges = plan.dynamic_ranges_by_layer[layer];
    }

    for (const auto &range : ranges) { prefetch_range(range); }

    {
        std::lock_guard<std::mutex> lock(prefetch_mutex);
        if (layer_epochs[layer] == epoch) {
            resident_layers[layer] = true;
            queued_layers[layer] = false;
            pending_ranges[layer] = 0;
            if (!ranges.empty()) {
                counters.prefetched_layers.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
}

bool monowire_streaming_runtime_state::prefetch_layer_direct(int layer, uint64_t expected_epoch) {
    if (layer < 0 || layer >= (int)direct_tensors_by_layer.size()) {
        return false;
    }

    uint64_t epoch = 0;
    std::vector<direct_tensor_buffer> tensors;
    {
        std::lock_guard<std::mutex> lock(prefetch_mutex);
        if (expected_epoch != UINT64_MAX && layer_epochs[layer] != expected_epoch) {
            return false;
        }
        if (resident_layers[layer]) {
            return true;
        }

        epoch = layer_epochs[layer];
        tensors.reserve(direct_tensors_by_layer[layer].size());
        for (const auto &tensor : direct_tensors_by_layer[layer]) {
            tensors.push_back({
                /*.tensor       =*/tensor.tensor,
                /*.original_data=*/tensor.original_data,
                /*.file_idx     =*/tensor.file_idx,
                /*.offset       =*/tensor.offset,
                /*.size         =*/tensor.size,
                /*.data         =*/nullptr,
            });
        }
    }

    bool ok = true;
    std::vector<aligned_buffer_ptr> buffers;
    buffers.reserve(tensors.size());

    for (const auto &tensor : tensors) {
        if (tensor.file_idx < 0 || tensor.file_idx >= (int)mappings.size()) {
            ok = false;
            break;
        }

        auto buffer = allocate_direct_buffer(tensor.size);
        if (!buffer) {
            std::fprintf(stderr, "%s: failed to allocate %.2f MiB direct buffer for layer %d\n", __func__,
                         tensor.size / 1024.0 / 1024.0, layer);
            ok = false;
            break;
        }
        if (!mappings[tensor.file_idx]->read_to(tensor.offset, buffer.get(), tensor.size)) {
            std::fprintf(stderr, "%s: failed to read %.2f MiB direct buffer for layer %d at file %d offset %zu\n",
                         __func__, tensor.size / 1024.0 / 1024.0, layer, tensor.file_idx, tensor.offset);
            ok = false;
            break;
        }
        buffers.push_back(std::move(buffer));
    }

    {
        std::lock_guard<std::mutex> lock(prefetch_mutex);
        if (layer_epochs[layer] != epoch) {
            prefetch_cv.notify_all();
            return false;
        }

        queued_layers[layer] = false;
        pending_ranges[layer] = 0;
        resident_layers[layer] = true;

        if (ok && buffers.size() == direct_tensors_by_layer[layer].size()) {
            for (size_t i = 0; i < buffers.size(); ++i) {
                direct_tensors_by_layer[layer][i].data = std::move(buffers[i]);
            }
            counters.direct_prefetched_tensors.fetch_add(direct_tensors_by_layer[layer].size(),
                                                         std::memory_order_relaxed);
            counters.prefetched_layers.fetch_add(1, std::memory_order_relaxed);
            counters.prefetched_ranges.fetch_add(direct_tensors_by_layer[layer].size(), std::memory_order_relaxed);
            counters.prefetched_bytes.fetch_add(
                [&]() {
                    size_t bytes = 0;
                    for (const auto &tensor : direct_tensors_by_layer[layer]) { bytes += tensor.size; }
                    return bytes;
                }(),
                std::memory_order_relaxed);
        }
    }

    prefetch_cv.notify_all();
    return ok;
}

void monowire_streaming_runtime_state::bind_layer_direct(int layer) {
    if (!direct_buffer || layer < 0 || layer >= (int)direct_tensors_by_layer.size()) {
        return;
    }

    std::lock_guard<std::mutex> lock(prefetch_mutex);
    if (!resident_layers[layer] || bound_layers[layer]) {
        return;
    }

    bool bound_any = false;
    for (auto &tensor : direct_tensors_by_layer[layer]) {
        if (tensor.tensor == nullptr || !tensor.data) {
            continue;
        }
        tensor.tensor->data = tensor.data.get();
        bound_any = true;
    }

    bound_layers[layer] = bound_any;
    if (bound_any) {
        counters.direct_bound_layers.fetch_add(1, std::memory_order_relaxed);
    }
}

void monowire_streaming_runtime_state::release_layer_direct(int layer) {
    if (!direct_buffer || layer < 0 || layer >= (int)direct_tensors_by_layer.size()) {
        return;
    }

    std::lock_guard<std::mutex> lock(prefetch_mutex);
    for (auto &tensor : direct_tensors_by_layer[layer]) {
        if (tensor.tensor != nullptr && tensor.original_data != nullptr) {
            tensor.tensor->data = tensor.original_data;
        }
        tensor.data.reset();
    }
    bound_layers[layer] = false;
}

bool monowire_streaming_runtime_state::wait_until_layer_ready(int layer) {
    if (!direct_buffer || layer < 0 || layer >= (int)resident_layers.size()) {
        return false;
    }

    {
        std::unique_lock<std::mutex> lock(prefetch_mutex);
        if (resident_layers[layer]) {
            return true;
        }
        if (queued_layers[layer]) {
            prefetch_cv.wait(lock, [this, layer]() {
                return stop_prefetch_worker || resident_layers[layer] || !queued_layers[layer];
            });
            return resident_layers[layer];
        }
    }

    return prefetch_layer_direct(layer, UINT64_MAX);
}

size_t monowire_streaming_runtime_state::prefetch_range(const monowire_streaming_range &range) {
    if (range.file_idx < 0 || range.file_idx >= (int)mappings.size()) {
        return 0;
    }

    const size_t bytes = range.last > range.first ? range.last - range.first : 0;
    if (read_prefetch) {
        mappings[range.file_idx]->read_fragment(range.first, range.last);
        counters.read_prefetched_ranges.fetch_add(1, std::memory_order_relaxed);
    } else {
        mappings[range.file_idx]->advise_fragment(range.first, range.last);
        counters.hint_prefetched_ranges.fetch_add(1, std::memory_order_relaxed);
    }
    counters.prefetched_ranges.fetch_add(1, std::memory_order_relaxed);
    counters.prefetched_bytes.fetch_add(bytes, std::memory_order_relaxed);
    return bytes;
}

void monowire_streaming_runtime_state::enqueue_prefetch_layer(int layer) {
    if (prefetch_workers.empty()) {
        prefetch_layer(layer);
        return;
    }

    if (layer < 0 || layer >= (int)plan.dynamic_ranges_by_layer.size()) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(prefetch_mutex);
        if (resident_layers[layer] || queued_layers[layer]) {
            return;
        }

        if (direct_buffer) {
            if (layer >= (int)direct_tensors_by_layer.size() || direct_tensors_by_layer[layer].empty()) {
                resident_layers[layer] = true;
                return;
            }

            queued_layers[layer] = true;
            pending_ranges[layer] = 1;
            counters.queued_layers.fetch_add(1, std::memory_order_relaxed);
            counters.queued_ranges.fetch_add(direct_tensors_by_layer[layer].size(), std::memory_order_relaxed);
            prefetch_queue.push_back({layer, layer_epochs[layer], {}});
            prefetch_cv.notify_all();
            return;
        }

        const auto &ranges = plan.dynamic_ranges_by_layer[layer];
        if (ranges.empty()) {
            resident_layers[layer] = true;
            return;
        }

        queued_layers[layer] = true;
        pending_ranges[layer] = ranges.size();
        counters.queued_layers.fetch_add(1, std::memory_order_relaxed);
        counters.queued_ranges.fetch_add(ranges.size(), std::memory_order_relaxed);
        const uint64_t epoch = layer_epochs[layer];
        for (const auto &range : ranges) { prefetch_queue.push_back({layer, epoch, range}); }
    }
    prefetch_cv.notify_all();
}

void monowire_streaming_runtime_state::prefetch_worker_loop() {
    for (;;) {
        prefetch_task task;
        {
            std::unique_lock<std::mutex> lock(prefetch_mutex);
            prefetch_cv.wait(lock, [this]() { return stop_prefetch_worker || !prefetch_queue.empty(); });
            if (stop_prefetch_worker) {
                return;
            }

            task = prefetch_queue.front();
            prefetch_queue.pop_front();

            if (task.layer < 0 || task.layer >= (int)queued_layers.size() || !queued_layers[task.layer]
                || layer_epochs[task.layer] != task.epoch || pending_ranges[task.layer] == 0) {
                counters.stale_tasks.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
        }

        if (direct_buffer) {
            prefetch_layer_direct(task.layer, task.epoch);
            continue;
        }

        prefetch_range(task.range);

        bool evict_stale_range = false;
        {
            std::lock_guard<std::mutex> lock(prefetch_mutex);
            if (task.layer < 0 || task.layer >= (int)queued_layers.size() || layer_epochs[task.layer] != task.epoch) {
                evict_stale_range = true;
                counters.stale_tasks.fetch_add(1, std::memory_order_relaxed);
            } else if (pending_ranges[task.layer] > 0) {
                --pending_ranges[task.layer];
                if (pending_ranges[task.layer] == 0) {
                    queued_layers[task.layer] = false;
                    resident_layers[task.layer] = true;
                    counters.prefetched_layers.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }

        if (evict_stale_range && evict_enabled) {
            if (task.range.file_idx >= 0 && task.range.file_idx < (int)mappings.size()) {
                mappings[task.range.file_idx]->evict_fragment(task.range.first, task.range.last);
                const size_t bytes = task.range.last > task.range.first ? task.range.last - task.range.first : 0;
                counters.evicted_ranges.fetch_add(1, std::memory_order_relaxed);
                counters.evicted_bytes.fetch_add(bytes, std::memory_order_relaxed);
            }
        }
    }
}

void monowire_streaming_runtime_state::evict_layer(int layer) {
    if (layer < 0 || layer >= (int)plan.dynamic_ranges_by_layer.size()) {
        return;
    }

    std::vector<monowire_streaming_range> ranges;
    {
        std::lock_guard<std::mutex> lock(prefetch_mutex);
        queued_layers[layer] = false;
        pending_ranges[layer] = 0;
        ++layer_epochs[layer];

        if (direct_buffer && layer < (int)direct_tensors_by_layer.size()) {
            for (auto &tensor : direct_tensors_by_layer[layer]) {
                if (tensor.tensor != nullptr && tensor.original_data != nullptr) {
                    tensor.tensor->data = tensor.original_data;
                }
                tensor.data.reset();
            }
            if (layer < (int)bound_layers.size()) {
                bound_layers[layer] = false;
            }
        }

        if (!resident_layers[layer]) {
            return;
        }

        resident_layers[layer] = false;
        ranges = plan.dynamic_ranges_by_layer[layer];
    }

    for (const auto &range : ranges) {
        if (range.file_idx < 0 || range.file_idx >= (int)mappings.size()) {
            continue;
        }
        mappings[range.file_idx]->evict_fragment(range.first, range.last);
        counters.evicted_ranges.fetch_add(1, std::memory_order_relaxed);
        counters.evicted_bytes.fetch_add(range.last > range.first ? range.last - range.first : 0,
                                         std::memory_order_relaxed);
    }
    if (!ranges.empty()) {
        counters.evicted_layers.fetch_add(1, std::memory_order_relaxed);
    }
}

bool monowire_streaming_runtime_state::is_boundary_tensor(const ggml_tensor *t, int &layer) const {
    return t != nullptr && parse_boundary_layer_name(ggml_get_name(t), layer);
}

bool monowire_streaming_runtime_state::is_window_boundary(int layer) const {
    if (plan.prefetch_window == 0 || layer < plan.first_layer || layer > plan.last_layer) {
        return false;
    }

    return true;
}

void monowire_streaming_runtime_state::begin_graph() {
    if (plan.first_layer < 0 || plan.last_layer < 0) {
        return;
    }
    counters.graph_begins.fetch_add(1, std::memory_order_relaxed);

    // Prime the first layer before graph execution starts, then let the
    // background workers prepare the rest of the initial window while layer 0
    // computes. This reduces startup stalls while keeping the immediate layer
    // from faulting on every streamed page.
    const int initial_first = plan.first_layer;
    const int initial_last = std::min(plan.last_layer, plan.first_layer + int(plan.prefetch_window) - 1);

    if (evict_enabled) {
        for (int layer = 0; layer < (int)resident_layers.size(); ++layer) {
            if (layer < initial_first || layer > initial_last) {
                evict_layer(layer);
            }
        }
    }

    for (int layer = initial_first; layer <= initial_last; ++layer) {
        if (layer == initial_first || prefetch_workers.empty()) {
            prefetch_layer(layer);
        } else {
            enqueue_prefetch_layer(layer);
        }
    }

    if (direct_buffer) {
        bind_layer_direct(initial_first);
    }
}

bool monowire_streaming_runtime_state::on_eval(ggml_tensor *t, bool ask) {
    int layer = -1;
    if (!is_boundary_tensor(t, layer) || !is_window_boundary(layer)) {
        return ask ? false : true;
    }

    // Scheduler callbacks are two-phase: ask=true queries whether we care about
    // this tensor, ask=false performs the prefetch/evict side effects.
    if (ask) {
        return true;
    }
    counters.boundary_callbacks.fetch_add(1, std::memory_order_relaxed);

    // Keep a sliding FlexInfer-style window: each completed layer schedules the
    // next window edge, so I/O has several layer times to complete before use.
    const int next_first = layer + 1;
    const int next_last = std::min(plan.last_layer, layer + int(plan.prefetch_window));
    for (int next_layer = next_first; next_layer <= next_last; ++next_layer) { enqueue_prefetch_layer(next_layer); }

    if (evict_enabled) {
        // Layer weights are not reused after l_out-N is produced, so release the
        // consumed layer immediately instead of waiting for a whole window.
        evict_layer(layer);
    }

    if (direct_buffer && next_first <= plan.last_layer && wait_until_layer_ready(next_first)) {
        bind_layer_direct(next_first);
    }

    return true;
}

monowire_streaming_runtime_stats monowire_streaming_runtime_state::stats() const {
    monowire_streaming_runtime_stats result;
    result.graph_begins = counters.graph_begins.load(std::memory_order_relaxed);
    result.boundary_callbacks = counters.boundary_callbacks.load(std::memory_order_relaxed);
    result.queued_layers = counters.queued_layers.load(std::memory_order_relaxed);
    result.prefetched_layers = counters.prefetched_layers.load(std::memory_order_relaxed);
    result.evicted_layers = counters.evicted_layers.load(std::memory_order_relaxed);
    result.queued_ranges = counters.queued_ranges.load(std::memory_order_relaxed);
    result.prefetched_ranges = counters.prefetched_ranges.load(std::memory_order_relaxed);
    result.read_prefetched_ranges = counters.read_prefetched_ranges.load(std::memory_order_relaxed);
    result.hint_prefetched_ranges = counters.hint_prefetched_ranges.load(std::memory_order_relaxed);
    result.direct_prefetched_tensors = counters.direct_prefetched_tensors.load(std::memory_order_relaxed);
    result.direct_bound_layers = counters.direct_bound_layers.load(std::memory_order_relaxed);
    result.evicted_ranges = counters.evicted_ranges.load(std::memory_order_relaxed);
    result.stale_tasks = counters.stale_tasks.load(std::memory_order_relaxed);
    result.prefetched_bytes = counters.prefetched_bytes.load(std::memory_order_relaxed);
    result.evicted_bytes = counters.evicted_bytes.load(std::memory_order_relaxed);
    return result;
}

void monowire_streaming_runtime_state::print_trace_stats() const {
    const auto s = stats();
    std::fprintf(stderr,
                 "monowire streaming trace: graphs=%" PRIu64 ", boundaries=%" PRIu64 ", queued_layers=%" PRIu64
                 ", prefetched_layers=%" PRIu64 ", evicted_layers=%" PRIu64 ", queued_ranges=%" PRIu64
                 ", prefetched_ranges=%" PRIu64 " (read=%" PRIu64 ", hint=%" PRIu64 ", direct_tensors=%" PRIu64
                 ", %.2f MiB)"
                 ", direct_bound_layers=%" PRIu64 ", evicted_ranges=%" PRIu64 " (%.2f MiB), stale_tasks=%" PRIu64 "\n",
                 s.graph_begins, s.boundary_callbacks, s.queued_layers, s.prefetched_layers, s.evicted_layers,
                 s.queued_ranges, s.prefetched_ranges, s.read_prefetched_ranges, s.hint_prefetched_ranges,
                 s.direct_prefetched_tensors, s.prefetched_bytes / 1024.0 / 1024.0, s.direct_bound_layers,
                 s.evicted_ranges, s.evicted_bytes / 1024.0 / 1024.0, s.stale_tasks);
}
