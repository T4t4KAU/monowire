#pragma once

// This is a staging header for new Monowire API.
// Breaking changes and C++ are allowed. Everything here should be considered
// WIP.

#include "monowire.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to
// monowire_graph_reserve.
MONOWIRE_API struct ggml_cgraph *monowire_graph_reserve(struct monowire_context *ctx, uint32_t n_tokens,
                                                        uint32_t n_seqs, uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
MONOWIRE_API ggml_type monowire_ftype_get_default_type(monowire_ftype ftype);

struct quantize_state_impl;

MONOWIRE_API quantize_state_impl *monowire_quant_init(const monowire_model *model,
                                                      const monowire_model_quantize_params *params);

MONOWIRE_API void monowire_quant_free(quantize_state_impl *qs);

// Descriptor for constructing a mock model for quantization testing.
struct monowire_quant_model_desc {
    const char *architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with monowire_model_free().
MONOWIRE_API monowire_model *monowire_quant_model_from_metadata(const monowire_quant_model_desc *desc);

// Returns true if this tensor should be quantized (based on name, dims,
// params).
MONOWIRE_API bool monowire_quant_tensor_allows_quantization(const quantize_state_impl *qs, const ggml_tensor *tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use
// monowire_quant_tensor_allows_quantization to filter). result_types:
// caller-allocated array of n_tensors elements, filled with assigned types.
MONOWIRE_API void monowire_quant_compute_types(quantize_state_impl *qs, monowire_ftype ftype, ggml_tensor **tensors,
                                               ggml_type *result_types, size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct monowire_memory_breakdown_data {
    size_t model = 0;   // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const { return model + context + compute; }
};

struct monowire_device_memory_data {
    int64_t total;
    int64_t free;
    monowire_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using monowire_memory_breakdown = std::map<ggml_backend_buffer_type_t, monowire_memory_breakdown_data>;

MONOWIRE_API int32_t monowire_model_n_expert(const struct monowire_model *model);
MONOWIRE_API int32_t monowire_model_n_devices(const struct monowire_model *model);

MONOWIRE_API ggml_backend_dev_t monowire_model_get_device(const struct monowire_model *model, int i);

MONOWIRE_API monowire_memory_breakdown monowire_get_memory_breakdown(const struct monowire_context *ctx);
