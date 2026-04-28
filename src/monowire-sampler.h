#pragma once

#include "monowire.h"

#include <vector>

struct monowire_vocab;
struct monowire_grammar;

// sampler chain

struct monowire_sampler_chain {
    monowire_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;

    struct info {
        bool is_backend;

        monowire_sampler *ptr;
    };

    std::vector<info> samplers;

    // pre-allocated buffer for monowire_sampler_sample to avoid repeated
    // allocations
    std::vector<monowire_token_data> cur;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

struct monowire_sampler *
monowire_sampler_init_dry_testing(int32_t context_size, float dry_multiplier, float dry_base,
                                  int32_t dry_allowed_length, int32_t dry_penalty_last_n,
                                  const std::vector<std::vector<monowire_token>> &seq_breakers);
