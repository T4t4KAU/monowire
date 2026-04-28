#include "monowire-streaming.h"

#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <regex>
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
                                                                   const monowire_mmaps &mappings, bool evict_enabled)
    : plan(std::move(plan)), mappings(mappings), evict_enabled(evict_enabled),
      resident_layers(this->plan.dynamic_ranges_by_layer.size(), false) {}

void monowire_streaming_runtime_state::prefetch_layer(int layer) {
    if (layer < 0 || layer >= (int)plan.dynamic_ranges_by_layer.size()) {
        return;
    }

    if (resident_layers[layer]) {
        return;
    }

    for (const auto &range : plan.dynamic_ranges_by_layer[layer]) {
        if (range.file_idx < 0 || range.file_idx >= (int)mappings.size()) {
            continue;
        }
        mappings[range.file_idx]->advise_fragment(range.first, range.last);
    }

    resident_layers[layer] = true;
}

void monowire_streaming_runtime_state::evict_layer(int layer) {
    if (layer < 0 || layer >= (int)plan.dynamic_ranges_by_layer.size()) {
        return;
    }

    if (!resident_layers[layer]) {
        return;
    }

    for (const auto &range : plan.dynamic_ranges_by_layer[layer]) {
        if (range.file_idx < 0 || range.file_idx >= (int)mappings.size()) {
            continue;
        }
        mappings[range.file_idx]->evict_fragment(range.first, range.last);
    }

    resident_layers[layer] = false;
}

bool monowire_streaming_runtime_state::is_boundary_tensor(const ggml_tensor *t, int &layer) const {
    return t != nullptr && parse_boundary_layer_name(ggml_get_name(t), layer);
}

bool monowire_streaming_runtime_state::is_window_boundary(int layer) const {
    if (plan.prefetch_window == 0 || layer < plan.first_layer || layer > plan.last_layer) {
        return false;
    }

    const int rel_layer = layer - plan.first_layer + 1;
    return layer == plan.last_layer || rel_layer % int(plan.prefetch_window) == 0;
}

void monowire_streaming_runtime_state::begin_graph() {
    if (plan.first_layer < 0 || plan.last_layer < 0) {
        return;
    }

    // Prime the first window before graph execution starts. If the same runtime
    // is reused, stale resident layers outside the new initial window are
    // evicted.
    const int initial_first = plan.first_layer;
    const int initial_last = std::min(plan.last_layer, plan.first_layer + int(plan.prefetch_window) - 1);

    if (evict_enabled) {
        for (int layer = 0; layer < (int)resident_layers.size(); ++layer) {
            if (layer < initial_first || layer > initial_last) {
                evict_layer(layer);
            }
        }
    }

    for (int layer = initial_first; layer <= initial_last; ++layer) { prefetch_layer(layer); }
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

    const int next_first = layer + 1;
    const int next_last = std::min(plan.last_layer, layer + int(plan.prefetch_window));
    for (int next_layer = next_first; next_layer <= next_last; ++next_layer) { prefetch_layer(next_layer); }

    if (evict_enabled) {
        const int evict_first = std::max(plan.first_layer, layer - int(plan.prefetch_window) + 1);
        for (int old_layer = evict_first; old_layer <= layer; ++old_layer) { evict_layer(old_layer); }
    }

    return true;
}
