#pragma once

#ifndef __cplusplus
#error "This header is for C++ only"
#endif

#include <memory>

#include "monowire.h"

struct monowire_model_deleter {
    void operator()(monowire_model * model) { monowire_model_free(model); }
};

struct monowire_context_deleter {
    void operator()(monowire_context * context) { monowire_free(context); }
};

struct monowire_sampler_deleter {
    void operator()(monowire_sampler * sampler) { monowire_sampler_free(sampler); }
};

struct monowire_adapter_lora_deleter {
    void operator()(monowire_adapter_lora * adapter) { monowire_adapter_lora_free(adapter); }
};

typedef std::unique_ptr<monowire_model, monowire_model_deleter> monowire_model_ptr;
typedef std::unique_ptr<monowire_context, monowire_context_deleter> monowire_context_ptr;
typedef std::unique_ptr<monowire_sampler, monowire_sampler_deleter> monowire_sampler_ptr;
typedef std::unique_ptr<monowire_adapter_lora, monowire_adapter_lora_deleter> monowire_adapter_lora_ptr;
