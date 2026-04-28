#include "../src/monowire-streaming.h"

#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#define REQUIRE(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "REQUIRE failed: %s at %s:%d\n", #expr, __FILE__, __LINE__); \
        return 1; \
    } \
} while (0)

static bool has_group(const monowire_streaming_plan & plan, const std::string & name) {
    return std::find(plan.selected_groups.begin(), plan.selected_groups.end(), name) != plan.selected_groups.end();
}

static ggml_tensor * make_layer_boundary(ggml_context * ctx, int layer) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    const std::string name = "l_out-" + std::to_string(layer);
    ggml_set_name(t, name.c_str());
    return t;
}

int main() {
    const std::vector<monowire_streaming_tensor> tensors = {
        {0, 0,   0, 150, "blk.0.ffn_down.weight"},
        {1, 0, 150, 150, "blk.1.ffn_down.weight"},
        {0, 0, 300, 130, "blk.0.ffn_up.weight"},
        {1, 0, 430, 130, "blk.1.ffn_up.weight"},
        {0, 0, 560,  40, "blk.0.attn_k.weight"},
        {1, 0, 600,  40, "blk.1.attn_k.weight"},
        {0, 0, 640,  60, "blk.0.attn_q.weight"},
        {1, 0, 700,  60, "blk.1.attn_q.weight"},
    };

    {
        const auto plan = monowire_streaming_build_plan(tensors, 300);
        REQUIRE(plan.reserved_bytes == 300);
        REQUIRE(has_group(plan, "ffn_down"));
        REQUIRE(!has_group(plan, "ffn_up"));
        REQUIRE(!has_group(plan, "attn_k"));
    }

    {
        const auto plan = monowire_streaming_build_plan(tensors, 640);
        REQUIRE(plan.reserved_bytes == 640);
        REQUIRE(has_group(plan, "ffn_down"));
        REQUIRE(has_group(plan, "ffn_up"));
        REQUIRE(has_group(plan, "attn_k"));
        REQUIRE(!has_group(plan, "attn_q"));
    }

    {
        const auto attention_only = monowire_streaming_build_plan(tensors, 200);
        REQUIRE(attention_only.reserved_bytes == 200);
        REQUIRE(!has_group(attention_only, "ffn_down"));
        REQUIRE(!has_group(attention_only, "ffn_up"));
        REQUIRE(has_group(attention_only, "attn_k"));
        REQUIRE(has_group(attention_only, "attn_q"));
    }

    {
        const auto plan = monowire_streaming_build_plan(tensors, 300);
        const auto runtime = monowire_streaming_build_runtime_plan(tensors, plan, 2, 3);

        REQUIRE(runtime.prefetch_window == 3);
        REQUIRE(runtime.dynamic_bytes == 460);
        REQUIRE(runtime.first_layer == 0);
        REQUIRE(runtime.last_layer == 1);
        REQUIRE(runtime.dynamic_ranges_by_layer.size() == 2);
        REQUIRE(runtime.dynamic_ranges_by_layer[0].size() == 3);
        REQUIRE(runtime.dynamic_ranges_by_layer[1].size() == 3);
    }

    {
        const auto plan = monowire_streaming_build_plan(tensors, 640);
        const auto runtime = monowire_streaming_build_runtime_plan(tensors, plan, 2, 2);

        REQUIRE(runtime.dynamic_bytes == 120);
        REQUIRE(runtime.first_layer == 0);
        REQUIRE(runtime.last_layer == 1);
        REQUIRE(runtime.dynamic_ranges_by_layer[0].size() == 1);
        REQUIRE(runtime.dynamic_ranges_by_layer[1].size() == 1);
        REQUIRE(runtime.dynamic_ranges_by_layer[0][0].first == 640);
        REQUIRE(runtime.dynamic_ranges_by_layer[0][0].last  == 700);
    }

    {
        const auto plan = monowire_streaming_build_plan(tensors, 760);
        const auto runtime = monowire_streaming_build_runtime_plan(tensors, plan, 2, 3);

        REQUIRE(runtime.dynamic_bytes == 0);
        REQUIRE(runtime.first_layer == -1);
        REQUIRE(runtime.last_layer == -1);
    }

    {
        monowire_streaming_runtime_plan runtime;
        runtime.prefetch_window = 3;
        runtime.first_layer = 0;
        runtime.last_layer = 5;
        runtime.dynamic_ranges_by_layer.resize(6);

        monowire_mmaps mappings;
        monowire_streaming_runtime_state state(std::move(runtime), mappings, true);
        state.begin_graph();

        ggml_init_params params = {
            /*.mem_size   =*/ 16 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        ggml_context * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * l0 = make_layer_boundary(ctx, 0);
        ggml_tensor * l1 = make_layer_boundary(ctx, 1);
        ggml_tensor * l2 = make_layer_boundary(ctx, 2);
        ggml_tensor * l3 = make_layer_boundary(ctx, 3);
        ggml_tensor * l4 = make_layer_boundary(ctx, 4);
        ggml_tensor * l5 = make_layer_boundary(ctx, 5);

        REQUIRE(!state.on_eval(l0, true));
        REQUIRE(!state.on_eval(l1, true));
        REQUIRE(state.on_eval(l2, true));
        REQUIRE(!state.on_eval(l3, true));
        REQUIRE(!state.on_eval(l4, true));
        REQUIRE(state.on_eval(l5, true));
        REQUIRE(state.on_eval(l2, false));
        REQUIRE(state.on_eval(l5, false));

        ggml_free(ctx);
    }

    return 0;
}
